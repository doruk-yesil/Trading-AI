@echo off
REM run_daily_refresh.bat
REM ----------------------
REM Wrapper so Windows Task Scheduler can run the daily refresh pipeline
REM without needing to "activate" a venv (which is unreliable in a
REM non-interactive scheduled task). Uses the venv's python.exe directly
REM if it exists, otherwise falls back to whatever "python" resolves to.
REM
REM One-time setup: register this with Task Scheduler, see README.md
REM ("Automated Daily Refresh" section) for the exact schtasks command.

setlocal
cd /d "%~dp0.."

if exist "venv\Scripts\python.exe" (
    set "PYTHON_EXE=venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

if not exist "reports\refresh_logs" mkdir "reports\refresh_logs"

"%PYTHON_EXE%" scripts\daily_refresh.py >> "reports\refresh_logs\bat_output.log" 2>&1

endlocal
