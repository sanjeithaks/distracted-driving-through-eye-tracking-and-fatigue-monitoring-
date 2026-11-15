@echo off
title Driver Monitoring - Install Dependencies
cd /d "%~dp0"
echo ============================================================
echo  Installing project dependencies (one-time setup)
echo ============================================================
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Done. You can now use START_WEB_DASHBOARD.bat or START_LIVE_CAMERA.bat
pause
