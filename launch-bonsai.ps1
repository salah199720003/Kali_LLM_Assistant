param([ValidateRange(1024, 262144)][int]$ContextSize = 102400)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& (Join-Path $PSScriptRoot 'start-bonsai-server.ps1') -ContextSize $ContextSize

$env:DEEP_AGENT_BACKEND = 'llama'
$env:DEEP_AGENT_BASE_URL = 'http://127.0.0.1:8080/v1'
$env:DEEP_AGENT_MODEL = 'bonsai-2-27b'
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot 'runtime\uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot 'runtime\python'
Write-Host "Agent backend: Bonsai 2 27B at $env:DEEP_AGENT_BASE_URL"
& uv run --with-requirements requirements.txt python deep_agent.py
exit $LASTEXITCODE
