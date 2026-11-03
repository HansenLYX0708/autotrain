@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0New-UpdatePackage.ps1" %*
if "%~1"=="" pause
