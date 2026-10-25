#!/usr/bin/env python3
"""最小 Agent —— 纯标准库实现，不依赖任何 agent 框架（无需 pip install）。

核心就是一个循环：把消息发给模型 -> 模型要工具就本地执行 -> 结果回填消息历史
-> 再发一次，直到模型不再调用工具。其余都是围绕这个循环的工程细节：
  1. 并行工具调用：一轮里返回多个 tool_calls 时并发执行
  2. 流式输出：SSE 边生成边打印，不用干等
  3. 上下文压缩：旧的工具输出自动折叠，避免撑爆窗口
  4. 路径沙箱：文件工具被限制在工作区内，写不出去
  5. 重试 / 超时 / 错误回灌 / 未知工具兜底
  6. 多轮会话：-i 模式保留上下文
  7. 子 agent：spawn_subagent 把子任务派出去，上下文隔离，可并行
  8. 全局预算：token / 秒数 / 调用次数整棵树共享，超了立刻停

用法：
    set DEEPSEEK_API_KEY=sk-xxx
    python agent.py "把目录下所有 TODO 找出来按文件汇总"
    python agent.py -i                   # 交互模式，记住上下文
    python agent.py --demo               # 不联网，演示 trace + 并行 + 子 agent
    python agent.py --selftest           # 不联网，自检工具与并发
    python agent.py --full-trace "..."   # trace 里打印完整工具返回

换任何 OpenAI 兼容端点：
    set AGENT_BASE_URL=http://localhost:11434/v1
    set AGENT_MODEL=qwen2.5:7b
    set AGENT_WORKDIR=E:\\some\\project   # 沙箱根目录，默认当前目录
"""

import argparse
import fnmatch
import glob as globmod
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# ---------------- 配置 ----------------
BASE_URL = os.environ.get("AGENT_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.environ.get("AGENT_MODEL", "deepseek-chat")
MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "25"))
MAX_TOOL_OUTPUT = 8000        # 单条工具结果回灌上限
MAX_CONTEXT_CHARS = 96_000    # 超过就折叠旧工具输出
WORKSPACE = os.path.abspath(os.environ.get("AGENT_WORKDIR") or os.getcwd())
TIMEOUT = 180
STREAM = True                 # --no-stream 可关
FAKE = False                  # --demo 用假模型
FULL_TRACE = False            # --full-trace 打印完整工具返回
MAX_DEPTH = int(os.environ.get("AGENT_MAX_DEPTH", "2"))   # 允许嵌套几层

_state = threading.local()    # 每个线程自己的「当前深度 / 日志前缀」
_sub_counter = itertools.count(1)


# ---------------- 全局预算 ----------------
class BudgetExhausted(Exception):
    """预算耗尽。整棵 agent 树都要停，不是某一层自己的问题。"""


class Budget:
    """父子 agent 共享同一份预算对象，跨线程访问所以带锁。

    单位是「整棵树」而不是「单个 agent」——这正是它存在的意义：
    子 agent 再多再深，也花不出预算之外的钱和时间。
    """

    def __init__(self, max_tokens=None, max_seconds=None, max_calls=None):
        self.max_tokens = max_tokens
        self.max_calls = max_calls
        self.max_seconds = max_seconds
        self.tokens = 0
        self.calls = 0
        self.start = time.time()
        self.deadline = self.start + max_seconds if max_seconds else None
        self.lock = threading.Lock()

    def remaining_seconds(self):
        if not self.deadline:
            return None
        return max(0.0, self.deadline - time.time())

    def check(self):
        with self.lock:
            if self.deadline and time.time() >= self.deadline:
                raise BudgetExhausted(f"时间预算用完（上限 {self.max_seconds:.0f}s）")
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise BudgetExhausted(f"模型调用次数用完（{self.calls}/{self.max_calls}）")
            if self.max_tokens is not None and self.tokens >= self.max_tokens:
                raise BudgetExhausted(f"token 预算用完（{self.tokens}/{self.max_tokens}）")

    def charge(self, usage, fallback_chars=0):
        with self.lock:
            self.calls += 1
            if usage:
                self.tokens += (int(usage.get("prompt_tokens") or 0)
                                + int(usage.get("completion_tokens") or 0))
            elif fallback_chars:                    # 端点没回 usage 就粗估（中文约 2 字符/token）
                self.tokens += fallback_chars // 2
            return self.tokens

    def report(self):
        used = f"{self.calls} 次调用 / {self.tokens:,} tokens / {time.time() - self.start:.1f}s"
        caps = []
        if self.max_calls is not None:
            caps.append(f"calls≤{self.max_calls}")
        if self.max_tokens is not None:
            caps.append(f"tokens≤{self.max_tokens:,}")
        if self.max_seconds is not None:
            caps.append(f"{self.max_seconds:.0f}s")
        return "预算用量: " + used + ("  （上限 " + ", ".join(caps) + "）" if caps else "")


BUDGET = Budget()             # 默认不限；main() 里按参数重建，子 agent 共用这一个


def _depth():
    return getattr(_state, "depth", 0)


def system_prompt():
    return (
        "你是一个能调用工具的编程助手，工作目录是 " + WORKSPACE + "。"
        "需要事实时必须先调用工具（读文件、列目录、搜索、跑命令），不要凭空猜测。"
        "多个互不依赖的操作可以在同一轮里一起发起，它们会并行执行。"
        "遇到能独立完成的子任务（比如分别调研几块代码、分别改几个文件），"
        "用 spawn_subagent 派出去，多个子任务可以在同一轮并发派发。"
        "工具报错时读错误信息自行修正，不要重复同样的失败调用。"
    )


def subagent_system():
    return (
        "你是子 agent，只负责完成交给你的那一个子任务。你看不到父 agent 的对话，"
        "所以每个事实都要自己用工具查，不要假设。工作目录是 " + WORKSPACE + "。"
        "完成后直接给出结论（结论要自包含，父 agent 只能看到这一段文字），不要客套。"
    )


# Windows 控制台默认 GBK，强制 UTF-8 免乱码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def log(msg=""):
    label = getattr(_state, "label", "")
    print((f"  {label} {msg}" if label else msg), flush=True)


# ---------------- 路径沙箱 ----------------
def _path(p):
    """相对路径按工作区解析；任何越界访问直接拒绝。"""
    full = os.path.abspath(p if os.path.isabs(p) else os.path.join(WORKSPACE, p))
    try:
        inside = os.path.commonpath([full, WORKSPACE]) == WORKSPACE
    except ValueError:                      # 不同盘符
        inside = False
    if not inside:
        raise ValueError(f"路径越界（沙箱只允许 {WORKSPACE}）：{p}")
    return full


# ---------------- 工具注册 ----------------
TOOLS = {}


def tool(desc, params, required=()):
    """把普通函数注册成模型可调用的工具。params: {名字: (类型, 说明)}"""
    def deco(fn):
        TOOLS[fn.__name__] = {
            "fn": fn,
            "schema": {
                "type": "function",
                "function": {
                    "name": fn.__name__,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            k: {"type": t, "description": d}
                            for k, (t, d) in params.items()
                        },
                        "required": list(required),
                    },
                },
            },
        }
        return fn
    return deco


@tool("读取文本文件，带行号返回，可用 offset/limit 分页。",
      {"path": ("string", "文件路径"), "offset": ("integer", "起始行，0 开始"),
       "limit": ("integer", "最多读多少行，默认 2000")},
      required=["path"])
def read_file(path, offset=0, limit=2000):
    full = _path(path)
    size = os.path.getsize(full)
    if size > 5_000_000:
        raise ValueError(f"文件太大（{size} 字节），请用 run_bash 分段查看")
    with open(full, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    chunk = lines[int(offset): int(offset) + int(limit)]
    body = "".join(f"{int(offset) + i + 1}\t{ln}" for i, ln in enumerate(chunk))
    if len(lines) > int(offset) + len(chunk):
        body += f"\n...(共 {len(lines)} 行，用 offset 继续)"
    return body or "(空文件)"


@tool("写入文件（覆盖），目录不存在会自动创建。",
      {"path": ("string", "文件路径"), "content": ("string", "完整内容")},
      required=["path", "content"])
def write_file(path, content):
    full = _path(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    return f"已写入 {len(content)} 字符 -> {os.path.relpath(full, WORKSPACE)}"


@tool("精确替换文件中的一段文本（old 必须在文件里唯一出现，除非 replace_all=true）。",
      {"path": ("string", "文件路径"), "old": ("string", "要替换的原文本"),
       "new": ("string", "新文本"), "replace_all": ("boolean", "替换全部出现")},
      required=["path", "old", "new"])
def edit_file(path, old, new, replace_all=False):
    full = _path(path)
    with open(full, "r", encoding="utf-8") as f:
        text = f.read()
    n = text.count(old)
    if n == 0:
        raise ValueError("未找到要替换的文本，请先 read_file 确认原文")
    if n > 1 and not replace_all:
        raise ValueError(f"old 出现了 {n} 次，请给出更长的唯一片段，或设置 replace_all=true")
    with open(full, "w", encoding="utf-8", newline="\n") as f:
        f.write(text.replace(old, new, -1 if replace_all else 1))
    return f"已替换 {n if replace_all else 1} 处"


@tool("列出目录内容。",
      {"path": ("string", "目录路径，默认 .")})
def list_dir(path="."):
    full = _path(path)
    entries = sorted(os.scandir(full), key=lambda e: (not e.is_dir(), e.name))
    out = []
    for e in entries:
        if e.is_dir():
            out.append(f"{e.name}/")
        else:
            out.append(f"{e.name}  {e.stat().st_size} B")
    return "\n".join(out[:300]) or "(空目录)"


@tool("按 glob 模式递归查找文件，如 **/*.py。",
      {"pattern": ("string", "glob 模式")},
      required=["pattern"])
def glob_files(pattern):
    hits = globmod.glob(pattern, root_dir=WORKSPACE, recursive=True)
    hits = [h.replace(os.sep, "/") for h in hits if os.path.isfile(os.path.join(WORKSPACE, h))]
    hits.sort()
    body = "\n".join(hits[:200])
    return body + (f"\n...(共 {len(hits)} 个)" if len(hits) > 200 else "") or "(无匹配)"


SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".idea", "dist", "build"}


@tool("用正则搜索文件内容，返回 文件:行号:内容。",
      {"pattern": ("string", "Python 正则"), "path": ("string", "搜索起点，默认 ."),
       "include": ("string", "只搜匹配该 glob 的文件，如 *.py")},
      required=["pattern"])
def grep(pattern, path=".", include="*"):
    rx = re.compile(pattern)
    root = _path(path)
    hits, scanned = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if not fnmatch.fnmatch(name, include):
                continue
            fp = os.path.join(dirpath, name)
            if os.path.getsize(fp) > 2_000_000:
                continue
            scanned += 1
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if rx.search(line):
                            hits.append(f"{os.path.relpath(fp, WORKSPACE)}:{i}: {line.rstrip()[:200]}")
                            if len(hits) >= 100:
                                return "\n".join(hits) + "\n...(已截断，共 100 条)"
            except OSError:
                continue
    return "\n".join(hits) or f"(在 {scanned} 个文件里没找到)"


@tool("执行 bash/shell 命令，返回 exit code + stdout + stderr。",
      {"cmd": ("string", "命令"), "timeout": ("integer", "超时秒数，默认 120")},
      required=["cmd"])
def run_bash(cmd, timeout=120):
    left = BUDGET.remaining_seconds()
    if left is not None:                        # 别让一条命令把时间预算整个吞掉
        timeout = min(float(timeout), max(1.0, left))
    p = subprocess.run(cmd, shell=True, cwd=WORKSPACE, capture_output=True,
                       timeout=float(timeout))
    return (f"exit={p.returncode}\nSTDOUT:\n{_decode(p.stdout)}\n"
            f"STDERR:\n{_decode(p.stderr)}")


def _decode(b):
    for enc in ("utf-8", "gbk"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", "replace")


def tool_schemas(include_spawn=True):
    return [t["schema"] for name, t in TOOLS.items()
            if include_spawn or name != "spawn_subagent"]


# ---------------- 工具执行（并行） ----------------
def _clip(s, n=300):
    s = str(s).replace("\r\n", "\n")
    return s if len(s) <= n else s[:n] + f"…(+{len(s) - n})"


def _human(n):
    return f"{n}B" if n < 1024 else f"{n / 1024:.1f}KB"


def _fmt_call(call):
    """把工具调用显示成 name({...})，中文不转义，太长就截断。"""
    raw = call["function"].get("arguments") or "{}"
    try:
        pretty = json.dumps(json.loads(raw), ensure_ascii=False)
    except Exception:
        pretty = str(raw)
    return f"{call['function']['name']}({_clip(pretty, 160)})"


def _exec(call, depth=0):
    """执行单个 tool_call，任何异常都转成字符串结果回灌，绝不让循环崩掉。"""
    _state.depth = depth                      # 线程里也要知道自己处在第几层
    name = call["function"]["name"]
    raw = call["function"].get("arguments") or "{}"
    t0 = time.time()
    args_text = _clip(raw, 200)
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
        args_text = _clip(json.dumps(args, ensure_ascii=False), 200)
    except json.JSONDecodeError as e:
        return {"id": call["id"], "name": name, "ok": False, "secs": time.time() - t0,
                "args": args_text,
                "content": f"ERROR: 参数不是合法 JSON（{e}）。你发送的是：{raw[:200]}"}
    entry = TOOLS.get(name)
    if entry is None:
        return {"id": call["id"], "name": name, "ok": False, "secs": time.time() - t0,
                "args": args_text,
                "content": f"ERROR: 没有名为 {name} 的工具。可用工具：{', '.join(TOOLS)}"}
    # 类型纠正：模型偶尔把数字传成字符串
    props = entry["schema"]["function"].get("parameters", {}).get("properties", {})
    for k, v in list(args.items()):
        t = props.get(k, {}).get("type")
        if t == "integer" and isinstance(v, str) and v.strip().lstrip("-").isdigit():
            args[k] = int(v)
        elif t == "boolean" and isinstance(v, str) and v.lower() in ("true", "false"):
            args[k] = v.lower() == "true"
    try:
        out = entry["fn"](**args)
        ok = True
    except Exception as e:
        out = f"ERROR: {type(e).__name__}: {e}"
        ok = False
    out = str(out)
    if len(out) > MAX_TOOL_OUTPUT:            # 保头保尾，错误通常在尾部
        cut = len(out) - MAX_TOOL_OUTPUT
        out = out[:MAX_TOOL_OUTPUT - 2000] + f"\n...(省略 {cut} 字符)...\n" + out[-2000:]
    return {"id": call["id"], "name": name, "ok": ok, "secs": time.time() - t0,
            "args": args_text, "content": out}


def run_tools(calls, depth=0, step=0):
    """核心：一轮多个工具 -> 并发执行；单个则直接跑，省线程开销。"""
    for c in calls:                           # 先打印"要调什么"，结果随后打印
        log(f"[{step}] -> {_fmt_call(c)}")
    if len(calls) > 1:
        log(f"       {len(calls)} 个工具并发执行")
        with ThreadPoolExecutor(max_workers=min(len(calls), 8)) as pool:
            return list(pool.map(lambda c: _exec(c, depth), calls))
    return [_exec(calls[0], depth)]


def trace_result(r):
    """trace 的后半段：这个工具返回了什么。"""
    flag = "ok " if r["ok"] else "ERR"
    log(f"    <- {r['name']}  {flag}  {r['secs']:.2f}s  {_human(len(r['content']))}")
    lines = (r["content"].replace("\r\n", "\n")).splitlines() or [""]
    shown = lines if FULL_TRACE else lines[:4]
    for line in shown:
        log(f"       | {_clip(line, 200)}")
    if not FULL_TRACE and len(lines) > 4:
        log(f"       | ...(共 {len(lines)} 行，加 --full-trace 看全部)")


# ---------------- 上下文压缩 ----------------
def _size(messages):
    return sum(len(str(m.get("content") or "")) + 40 for m in messages)


def compact(messages):
    """只折叠旧工具输出的内容，不动消息结构——保住 tool_calls 与 tool 的配对，
    否则 API 会直接 400。真产品这里会做带结构的摘要。"""
    if _size(messages) <= MAX_CONTEXT_CHARS:
        return messages
    folded = 0
    for m in messages[1:-6]:
        if m.get("role") == "tool" and len(m.get("content") or "") > 600:
            c = m["content"]
            m["content"] = c[:300] + f"\n...(上下文压缩，省略 {len(c) - 500} 字符)...\n" + c[-200:]
            folded += 1
    if folded:
        log(f"    [上下文] 折叠了 {folded} 条旧工具输出（约 {_size(messages)} 字符）")
    return messages


# ---------------- 模型调用 ----------------
def _post(payload):
    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("请先设置 DEEPSEEK_API_KEY")
    url = BASE_URL.rstrip("/") + "/chat/completions"
    last = None
    for attempt in range(1, 4):
        body = json.dumps(payload).encode("utf-8")   # 每轮重建，降级时会改 payload
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
        })
        left = BUDGET.remaining_seconds()
        tmo = TIMEOUT if left is None else max(1.0, min(TIMEOUT, left))
        try:
            return urllib.request.urlopen(req, timeout=tmo)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            # 有些端点（老版 Ollama / 部分代理）不认 stream_options，自动降级重试
            if e.code == 400 and "stream_options" in detail and "stream_options" in payload:
                payload.pop("stream_options")
                last = detail
                continue
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                last = f"HTTP {e.code}: {detail}"
                time.sleep(2 ** attempt)
                continue
            sys.exit(f"请求失败 HTTP {e.code}\n{detail}")   # 400/401 重试无意义
        except urllib.error.URLError as e:
            last = str(e.reason)
            if attempt < 3:
                time.sleep(2 ** attempt)
                continue
            sys.exit(f"网络错误：{last}")
    sys.exit(f"重试 3 次仍失败：{last}")


def _payload(messages, stream, schemas):
    p = {
        "model": MODEL,
        "messages": messages,
        "tools": schemas,
        "tool_choice": "auto",
        "stream": stream,
    }
    if stream:
        p["stream_options"] = {"include_usage": True}
    return p


def _on_text(chunk):
    sys.stdout.write(chunk)
    sys.stdout.flush()


def call_stream(messages, schemas):
    """SSE 流式，返回 (assistant 消息, usage, 是否打印过文本)。"""
    resp = _post(_payload(messages, True, schemas))
    parts, slots, usage = [], {}, None
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    parts.append(delta["content"])
                    _on_text(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    slot = slots.setdefault(tc.get("index", 0),
                                            {"id": "", "name": "", "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]
    msg = {"role": "assistant", "content": "".join(parts) or None}
    if slots:
        msg["tool_calls"] = [{
            "id": s["id"] or f"call_{i}",
            "type": "function",
            "function": {"name": s["name"], "arguments": s["args"] or "{}"},
        } for i, s in sorted(slots.items())]
    return msg, usage, bool(parts)


def call_plain(messages, schemas):
    resp = _post(_payload(messages, False, schemas))
    with resp:
        data = json.load(resp)
    return data["choices"][0]["message"], data.get("usage"), False


def call_model(messages, stream, schemas):
    if FAKE:
        return fake_model(messages)
    return call_stream(messages, schemas) if stream else call_plain(messages, schemas)


def fake_model(messages):
    """--demo 用的假模型：父 agent 并发派两个子 agent，子 agent 各自跑真工具。"""
    done = sum(1 for m in messages if m.get("role") == "tool")

    if _depth() == 0:
        if done == 0:
            return {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_a", "type": "function", "function": {
                    "name": "spawn_subagent",
                    "arguments": json.dumps({"task": "查一下这台机器的 python 版本"}, ensure_ascii=False)}},
                {"id": "call_b", "type": "function", "function": {
                    "name": "spawn_subagent",
                    "arguments": json.dumps({"task": "列一下当前目录有哪些文件"}, ensure_ascii=False)}},
            ]}, None, False
        text = "演示完成：两个子 agent 并行跑完，父 agent 只拿到了它们的结论。"
        for ch in text:
            _on_text(ch)
            time.sleep(0.01)
        return {"role": "assistant", "content": text}, None, True

    # 子 agent 的分支：先真的跑一条命令，再给出结论
    if done == 0:
        return {"role": "assistant", "content": None, "tool_calls": [
            {"id": "child_1", "type": "function",
             "function": {"name": "run_bash", "arguments": '{"cmd": "python --version"}'}},
        ]}, None, False
    last = [m for m in messages if m.get("role") == "tool"][-1]["content"]
    return {"role": "assistant", "content": "子 agent 结论：" + last.splitlines()[2].strip()}, None, False


# ---------------- 子 agent ----------------
@tool("派一个子 agent 独立完成某个子任务，只返回它的最终结论。"
      "子 agent 看不到当前对话，任务描述必须自包含（背景、目标、验收标准）。"
      "多个互不依赖的子任务可以在同一轮里并发派出，它们会并行执行。",
      {"task": ("string", "完整自包含的子任务描述"),
       "max_steps": ("integer", "子 agent 最多几步，默认 12")},
      required=["task"])
def spawn_subagent(task, max_steps=12):
    child_depth = _depth() + 1
    if child_depth > MAX_DEPTH:
        raise ValueError(f"嵌套层数已达上限（MAX_DEPTH={MAX_DEPTH}），请直接用现有工具完成")
    BUDGET.check()                            # 预算不够就别开新坑了
    n = next(_sub_counter)
    prev_label = getattr(_state, "label", "")
    _state.label = f"[sub{n}]"
    t0 = time.time()
    try:
        answer = run_turn(
            [{"role": "system", "content": subagent_system()}],
            task,
            depth=child_depth,
            stream=False,                     # 子 agent 不抢终端的流式输出
            schemas=tool_schemas(include_spawn=child_depth + 1 <= MAX_DEPTH),
            max_steps=max(1, min(int(max_steps), 15)),
        )
    finally:
        log(f"    结束，用时 {time.time() - t0:.1f}s")
        _state.label = prev_label
    return answer or "(子 agent 没有产出)"


# ---------------- 主循环 ----------------
def run_turn(convo, task, depth=0, stream=True, schemas=None, max_steps=None):
    prev = _depth()
    _state.depth = depth
    if schemas is None:
        schemas = tool_schemas(include_spawn=depth + 1 <= MAX_DEPTH)
    budget = max_steps or MAX_STEPS
    try:
        convo.append({"role": "user", "content": task})
        for step in range(1, budget + 1):
            try:
                BUDGET.check()                  # 整棵树共享，子 agent 花超了父的也停
            except BudgetExhausted as e:
                log(f"    [预算] {e}，提前收尾")
                return f"(提前停止：{e})"

            compact(convo)
            log(f"[{step}] 思考中…")
            msg, usage, streamed = call_model(convo, stream, schemas)
            BUDGET.charge(usage, _size(convo) + len(msg.get("content") or ""))
            if streamed:
                log()
            convo.append(msg)                       # 必须原样记录，tool_calls 要靠它对上 id

            calls = msg.get("tool_calls") or []
            if not calls:
                if not streamed and msg.get("content"):   # 流式模式已经边生成边打印过了
                    log(msg["content"])
                if usage:
                    log(f"    tokens: in {usage.get('prompt_tokens')} / "
                        f"out {usage.get('completion_tokens')}")
                return msg.get("content") or ""

            results = run_tools(calls, depth, step)
            for r in results:
                trace_result(r)
                convo.append({"role": "tool", "tool_call_id": r["id"], "content": r["content"]})
        return "(达到最大步数，已停止)"
    finally:
        _state.depth = prev


# ---------------- 自检（不联网） ----------------
def selftest():
    global MAX_DEPTH
    import shutil
    ok = True

    def check(label, cond, extra=""):
        nonlocal ok
        ok = ok and cond
        log(f"  {'PASS' if cond else 'FAIL'} {label} {extra}")

    log("[工具]")
    check("write_file", "已写入" in write_file("._selftest/probe.txt", "hello\nworld\n"))
    check("read_file", "1\thello" in read_file("._selftest/probe.txt"))
    check("edit_file", "已替换" in edit_file("._selftest/probe.txt", "world", "agent"))
    check("回读", "agent" in read_file("._selftest/probe.txt"))
    check("list_dir", "probe.txt" in list_dir("._selftest"))
    check("glob", "probe.txt" in glob_files("._selftest/*.txt"))
    check("grep", "agent" in grep("agent", "._selftest"))

    log("[沙箱]")
    try:
        read_file("C:/Windows/win.ini")
        check("越界拒绝", False)
    except ValueError:
        check("越界拒绝", True)

    log("[并行]")
    TOOLS["_slow"] = {
        "fn": lambda ms=300: (time.sleep(ms / 1000), "done")[1],
        "schema": {"type": "function", "function": {"name": "_slow", "parameters": {}}},
    }
    calls = [{"id": f"c{i}", "function": {"name": "_slow",
                                          "arguments": json.dumps({"ms": 400})}} for i in range(4)]
    t0 = time.time()
    run_tools(calls)
    elapsed = time.time() - t0
    del TOOLS["_slow"]
    check("4 个 400ms 任务并发", elapsed < 1.2, f"耗时 {elapsed:.2f}s（串行需 1.6s）")

    log("[子 agent]")
    check("工具已注册", "spawn_subagent" in TOOLS)
    check("限深时不暴露给模型",
          all(t["function"]["name"] != "spawn_subagent" for t in tool_schemas(include_spawn=False)))
    old_depth = MAX_DEPTH
    try:
        MAX_DEPTH = 0
        try:
            spawn_subagent("不该被执行")
            check("超限被拒绝", False)
        except ValueError:
            check("超限被拒绝", True)
    finally:
        MAX_DEPTH = old_depth

    log("[预算]")
    b = Budget(max_tokens=10)
    b.charge({"prompt_tokens": 6, "completion_tokens": 5})
    try:
        b.check()
        check("token 超限被拦截", False)
    except BudgetExhausted:
        check("token 超限被拦截", True)
    b2 = Budget(max_seconds=0.01)
    time.sleep(0.02)
    try:
        b2.check()
        check("时间超限被拦截", False)
    except BudgetExhausted:
        check("时间超限被拦截", True)
    try:
        Budget().check()
        check("无限预算放行", True)
    except BudgetExhausted:
        check("无限预算放行", False)

    shutil.rmtree(os.path.join(WORKSPACE, "._selftest"), ignore_errors=True)
    log("selftest " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


# ---------------- 入口 ----------------
def main():
    global MODEL, BASE_URL, MAX_STEPS, STREAM, FAKE, WORKSPACE, MAX_DEPTH, BUDGET, FULL_TRACE

    ap = argparse.ArgumentParser(description="极简但能干活的 agent")
    ap.add_argument("task", nargs="*", help="要完成的任务")
    ap.add_argument("-i", "--interactive", action="store_true", help="交互模式，保留上下文")
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--workdir", help="沙箱根目录，默认当前目录")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--max-depth", type=int, help="允许嵌套几层子 agent，默认 2")
    ap.add_argument("--budget-tokens", type=int,
                    default=int(os.environ.get("AGENT_BUDGET_TOKENS", "200000")),
                    help="整棵树的 token 上限，默认 200000")
    ap.add_argument("--budget-seconds", type=float,
                    default=float(os.environ.get("AGENT_BUDGET_SECONDS", "600")),
                    help="整棵树的时间上限，默认 600 秒")
    ap.add_argument("--budget-calls", type=int,
                    default=int(os.environ.get("AGENT_BUDGET_CALLS", "80")),
                    help="整棵树的模型调用次数上限，默认 80")
    ap.add_argument("--no-budget", action="store_true", help="关闭预算限制")
    ap.add_argument("--no-stream", action="store_true", help="关闭流式输出")
    ap.add_argument("--full-trace", action="store_true", help="trace 里打印完整工具返回")
    ap.add_argument("--demo", action="store_true", help="假模型演示并行调用，不联网")
    ap.add_argument("--selftest", action="store_true", help="工具与并发自检，不联网")
    a = ap.parse_args()

    if a.workdir:
        WORKSPACE = os.path.abspath(a.workdir)
    if a.model:
        MODEL = a.model
    if a.base_url:
        BASE_URL = a.base_url
    if a.max_steps:
        MAX_STEPS = a.max_steps
    if a.max_depth is not None:
        MAX_DEPTH = a.max_depth
    if a.no_stream:
        STREAM = False
    if a.full_trace:
        FULL_TRACE = True
    if a.demo:
        FAKE = True

    if a.selftest:
        return selftest()

    BUDGET = Budget() if a.no_budget else Budget(
        a.budget_tokens, a.budget_seconds, a.budget_calls)

    if not FAKE and not (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        sys.exit("请先设置环境变量 DEEPSEEK_API_KEY（key 只从环境变量读，代码里没有硬编码）")

    convo = [{"role": "system", "content": system_prompt()}]
    log(f"model={MODEL}  workspace={WORKSPACE}  stream={'off' if not STREAM else 'on'}"
        + ("  demo" if FAKE else ""))
    log("预算: " + ("不限" if a.no_budget else
                    f"调用≤{a.budget_calls} / tokens≤{a.budget_tokens:,} / {a.budget_seconds:.0f}s"))

    try:
        if a.interactive:
            log("交互模式，Ctrl+C 退出。")
            while True:
                try:
                    task = input("\n你> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if task:
                    log("")
                    run_turn(convo, task)
        else:
            run_turn(convo, " ".join(a.task) or "你好，介绍一下你能做什么")
    except KeyboardInterrupt:
        log("\n已中断。")
        log(BUDGET.report())
        return 130

    log("")
    log(BUDGET.report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
