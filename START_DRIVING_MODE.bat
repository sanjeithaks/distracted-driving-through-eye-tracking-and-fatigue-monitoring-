@echo off
setlocal
title Driving Mode Launcher
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
echo   IMMERSIVE DRIVING MODE - Driver Monitoring
echo ============================================================
echo.
echo   A / D  or  Left / Right  - steer
echo   W / S  or  Up / Down      - accelerate / brake
echo   R - reset and recalibrate
echo   F - toggle full-screen
echo   Q / Esc - quit
echo.
echo Starting the driving cockpit... look straight ahead for 2 seconds
echo so your neutral head pose calibrates. Press F for full-screen.
echo.
%PY% driving_mode.py --windowed --width 1024 --height 576
echo.
echo Driving mode closed.
pause
