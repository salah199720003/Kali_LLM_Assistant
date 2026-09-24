@echo off
title DeepHat Agent
cd /d "%~dp0"
echo Starting DeepHat agent...
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch-deephat.ps1"
if errorlevel 1 pause
