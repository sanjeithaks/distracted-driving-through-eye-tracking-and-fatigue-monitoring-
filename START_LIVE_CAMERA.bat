@echo off
title Driver Monitoring - Live Webcam
cd /d "%~dp0"
echo ============================================================
echo  Driver Fatigue and Distraction Monitoring - Live Webcam
echo ============================================================
echo.
echo A camera window will open.
echo Press q or Esc in that window to stop.
echo.
python main.py
pause
