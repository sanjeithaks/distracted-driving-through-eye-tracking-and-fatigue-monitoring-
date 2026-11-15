@echo off
setlocal
title Driver Monitoring - Live Webcam
cd /d "%~dp0"
set PYTHONUTF8=1

where python >nul 2>nul
if errorlevel 1 (
    where py >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Python was not found on PATH.
        echo Install Python 3.12 from https://www.python.org/downloads/
        echo and tick "Add Python to PATH" during installation.
        pause
        exit /b 1
    )
    set "PY=py -3"
) else (
    set "PY=python"
)

echo ============================================================
echo  Driver Fatigue and Distraction Monitoring - Live Webcam
echo ============================================================
echo.
echo A camera window will open. Press q or Esc there to stop.
echo.
%PY% main.py
echo.
pause
