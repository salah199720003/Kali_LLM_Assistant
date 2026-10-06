@echo off
title Qwen3.8 27B Q4_K_S Agent
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch-27b-q4ks.ps1"
if errorlevel 1 pause
