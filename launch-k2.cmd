@echo off
title K2 Horizon Agent
cd /d "%~dp0"
echo Starting K2 Horizon agent...
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch-k2.ps1"
if errorlevel 1 pause
