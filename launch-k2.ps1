$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

& (Join-Path $PSScriptRoot 'start-k2-server.ps1') -ContextSize 102400

$env:DEEP_AGENT_BACKEND = 'llama'
$env:DEEP_AGENT_BASE_URL = 'http://127.0.0.1:8080/v1'
$env:DEEP_AGENT_MODEL = 'k2-horizon'
Write-Host "Agent backend: K2 Horizon at $env:DEEP_AGENT_BASE_URL"
uv run --with-requirements requirements.txt python deep_agent.py
