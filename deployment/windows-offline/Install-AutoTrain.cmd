@echo off
net session >nul 2>&1
if errorlevel 1 (
  powershell.exe -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)
set /p INSTALL_DIR=Install directory [C:\AutoTrain]: 
if "%INSTALL_DIR%"=="" set "INSTALL_DIR=C:\AutoTrain"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install.ps1" -InstallRoot "%INSTALL_DIR%" -RequireGpu
if errorlevel 1 pause
