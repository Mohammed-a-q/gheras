@echo off
REM Batch script to launch the FastAPI app with virtualenv activated and port cleaned.

REM change to script directory
cd /d "%~dp0"

REM activate the virtual environment
call venv\Scripts\Activate.bat

REM kill any process using port 8000 (Windows style)
for /f "tokens=5" %%a in ('netstat -a -n -o ^| find ":8000" ^| find "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)

REM start uvicorn
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --log-level debug
