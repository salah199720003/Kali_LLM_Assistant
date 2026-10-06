@echo off
title Qwen3.8 27B Agent
cd /d "%~dp0"
echo Starting Qwen3.8 27B agent (CUDA)...
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch-27b.ps1"
if errorlevel 1 pause
