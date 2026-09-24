@echo off
title Bonsai 2 27B Agent
cd /d "%~dp0"
echo Starting Bonsai 2 27B agent...
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch-bonsai.ps1"
if errorlevel 1 pause
