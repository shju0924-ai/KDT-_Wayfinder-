@echo off
rem Backend dev server, called by "backend" in .claude/launch.json.
rem Resolves backend/.venv relative to this file, so it works from any clone path or cwd.
rem (ASCII only: cmd.exe parses .cmd files in the OEM code page, not UTF-8.)
setlocal
set "BACKEND=%~dp0..\..\backend"
set "PY=%BACKEND%\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [backend] venv not found: "%PY%"
  echo [backend] See README: in backend/, run "python -m venv .venv" then "pip install -r requirements.txt".
  exit /b 1
)

"%PY%" -m uvicorn app.main:app --reload --reload-dir "%BACKEND%\app" --app-dir "%BACKEND%" --host 127.0.0.1 --port 8000
