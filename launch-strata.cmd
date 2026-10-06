@echo off
title Qwen Coder Agent
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch-strata.ps1" %*
if errorlevel 1 pause
