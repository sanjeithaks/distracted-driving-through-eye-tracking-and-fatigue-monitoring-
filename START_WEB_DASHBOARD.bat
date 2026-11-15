@echo off
title Driver Monitoring - Web Dashboard
cd /d "%~dp0"
echo ============================================================
echo  Driver Fatigue and Distraction Monitoring - Web Dashboard
echo ============================================================
echo.
echo Opening dashboard at http://localhost:8501
echo Keep this window open. Press Ctrl+C to stop.
echo.
python -m streamlit run app.py
pause
