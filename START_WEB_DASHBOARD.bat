@echo off
setlocal
title Driver Monitoring - Web Dashboard
cd /d "%~dp0"
set PYTHONUTF8=1

echo ============================================================
echo  Driver Fatigue and Distraction Monitoring - Web Dashboard
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    where py >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python was not found on PATH.
        echo Install Python 3.12 from https://www.python.org/downloads/
        echo and tick "Add Python to PATH" during installation.
        echo.
        pause
        exit /b 1
    )
    set "PY=py -3"
) else (
    set "PY=python"
)

echo Starting the dashboard...
echo Local URL: http://localhost:8501
echo Keep this window open. Press Ctrl+C to stop.
echo.
%PY% -m streamlit run app.py

echo.
echo ============================================================
echo Dashboard stopped. If you saw an error above, copy it.
echo ============================================================
pause
