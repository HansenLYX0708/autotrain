@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Schedule-AutoTrainUpdate.ps1"
echo.
pause
