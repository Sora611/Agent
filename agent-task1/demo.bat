@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo ==========================================
echo   Minimal Agent - offline demo
echo   (no network, no API key needed)
echo ==========================================
echo.
echo [1/2] selftest: tools / sandbox / parallel / budget
echo.
python agent.py --selftest
echo.
echo [2/2] demo: trace / parallel tools / subagent
echo.
python agent.py --demo
echo.
echo To run it for real:
echo    set DEEPSEEK_API_KEY=your-key
echo    python agent.py "count .py files in this folder"
echo.
pause
