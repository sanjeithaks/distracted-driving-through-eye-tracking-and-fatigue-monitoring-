@echo off
title Driver Monitoring - Open Output Files
cd /d "%~dp0"
start "" "docs\OUTPUTS.html"
echo Opened the output index in your browser.
echo.
echo From there you can open:
echo   - the session report
echo   - screenshots and evidence snapshots
echo   - the annotated demo video
echo   - live webcam recordings (local only)
echo   - the GitHub repository
echo.
timeout /t 5 >nul
