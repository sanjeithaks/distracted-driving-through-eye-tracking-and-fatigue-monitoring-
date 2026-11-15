@echo off
title Driver Monitoring - Automated Demo
cd /d "%~dp0"
echo ============================================================
echo  Driver Fatigue and Distraction Monitoring - Demo
echo ============================================================
echo.
echo Running the 6-scenario demo (no webcam needed)...
echo.
python demo.py
echo.
echo Open docs\demo_report.html to view the session report.
pause
