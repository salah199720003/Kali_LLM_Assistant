$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& (Join-Path $PSScriptRoot 'start-deephat-server.ps1') -ContextSize 102400
$env:DEEP_AGENT_BACKEND = 'llama'
$env:DEEP_AGENT_BASE_URL = 'http://127.0.0.1:8080/v1'
$env:DEEP_AGENT_MODEL = 'deephat-v1-7b'
Write-Host "Agent backend: DeepHat at $env:DEEP_AGENT_BASE_URL"
uv run --with-requirements requirements.txt python deep_agent.py
