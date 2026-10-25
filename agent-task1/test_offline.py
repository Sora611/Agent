#!/usr/bin/env python3
"""离线验证 mini_agent 的真实网络路径：起一个假的 OpenAI 兼容服务，
让 agent 走完整链路（流式解析 -> 并行工具 -> 回灌 -> 子 agent -> 收尾）。
不花钱、不联网。

覆盖点：
  * SSE 分片拼装 content / tool_calls.arguments
  * 一轮多个工具并发
  * spawn_subagent 走嵌套循环，子 agent 用非流式请求
  * 深度上限后，子 agent 的工具清单里不再出现 spawn_subagent
"""

import contextlib
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REQUESTS = []


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _sse(self, obj):
        try:
            self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        n = len(REQUESTS)
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        REQUESTS.append(req)

        # 子 agent 用非流式：返回普通 JSON
        if not req.get("stream"):
            body = json.dumps({
                "id": "c", "object": "chat.completion", "model": "fake",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "子结论：Python 3.14"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            }, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        if n == 0:
            # 父 agent 第一轮：派一个子 agent（参数故意分片）
            self._sse({"choices": [{"index": 0, "delta": {"content": "我派个"}}]})
            self._sse({"choices": [{"index": 0, "delta": {"content": "子 agent 去查。"}}]})
            self._sse({"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "sp1", "type": "function",
                 "function": {"name": "spawn_subagent", "arguments": '{"task": "'}}]}}]})
            self._sse({"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '查一下 python 版本"}'}}]}}]})
            self._sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
            self._sse({"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 4}})
        else:
            # 父 agent 第二轮：收尾
            self._sse({"choices": [{"index": 0, "delta": {"content": "父 agent 收尾完成。"}}]})
            self._sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            self._sse({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}})

        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    os.environ["AGENT_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    os.environ["AGENT_MODEL"] = "fake-model"
    os.environ["DEEPSEEK_API_KEY"] = "sk-test"
    os.environ["AGENT_MAX_DEPTH"] = "1"      # 只允许一层子 agent，便于验证限深

    import agent as m

    convo = [{"role": "system", "content": "test"}]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        answer = m.run_turn(convo, "查一下 python 版本")
    out = buf.getvalue()
    srv.shutdown()

    ok = True

    def check(label, cond, extra=""):
        nonlocal ok
        ok = ok and cond
        print(f"  {'PASS' if cond else 'FAIL'} {label} {extra}")

    def names(req):
        return [t["function"]["name"] for t in req.get("tools") or []]

    print("[离线集成测试]")
    check("三次请求（父-子-父）", len(REQUESTS) == 3, f"实际 {len(REQUESTS)}")
    check("父 agent 用流式", REQUESTS[0].get("stream") is True)
    check("父 agent 能拿到 spawn 工具", "spawn_subagent" in names(REQUESTS[0]))
    check("子 agent 用非流式", REQUESTS[1].get("stream") is False)
    check("子 agent 有自己的 system 提示",
          "子 agent" in (REQUESTS[1]["messages"][0]["content"]))
    check("子 agent 看不到父对话", REQUESTS[1]["messages"][1]["content"] == "查一下 python 版本")
    check("深度到顶后 spawn 不再下放", "spawn_subagent" not in names(REQUESTS[1]))
    check("子 agent 工具数 = 7", len(names(REQUESTS[1])) == 7, str(len(names(REQUESTS[1]))))

    asst = [x for x in convo if x["role"] == "assistant"]
    check("解析出子 agent 调用", [c["id"] for c in asst[0]["tool_calls"]] == ["sp1"])
    check("碎片参数拼回完整 JSON",
          json.loads(asst[0]["tool_calls"][0]["function"]["arguments"])["task"] == "查一下 python 版本")

    tools = [x for x in convo if x["role"] == "tool"]
    check("父 agent 只拿到结论", tools[0]["content"] == "子结论：Python 3.14", repr(tools[0]["content"]))
    check("结论里没有子 agent 的中间过程", "run_bash" not in tools[0]["content"])
    check("收尾请求带上结论",
          REQUESTS[2]["messages"][-1]["content"] == "子结论：Python 3.14")
    check("最终答案", "收尾完成" in answer, repr(answer))
    check("父 agent 的流式文本已打印", "我派个" in out)

    # 预算：父 agent 两次 + 子 agent 一次，全部记在同一份账上
    check("预算统计含子 agent", m.BUDGET.calls == 3, f"calls={m.BUDGET.calls}")
    check("token 跨层累加", m.BUDGET.tokens == 33, f"tokens={m.BUDGET.tokens}")

    # 预算耗尽 -> 一次请求都不该再发
    m.BUDGET = m.Budget(max_calls=0)
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        stopped = m.run_turn([{"role": "system", "content": "t"}], "再来一次")
    check("预算耗尽立刻停", stopped.startswith("(提前停止"), repr(stopped))
    check("停了就不再发请求", len(REQUESTS) == 3, f"请求数 {len(REQUESTS)}")

    print("offline " + ("ok" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
