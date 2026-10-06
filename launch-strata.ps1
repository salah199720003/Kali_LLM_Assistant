[CmdletBinding()]
param(
    [ValidateSet('on', 'off')][string]$Thinking = 'on',
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$url = 'http://127.0.0.1:8082'
$model = 'qwen3.8-flash-next-coder-iq1_m'
try {
    $health = Invoke-RestMethod -Uri "$url/health" -TimeoutSec 5
    $models = Invoke-RestMethod -Uri "$url/v1/models" -TimeoutSec 5
} catch {
    throw "Cannot reach Coder at $url. Open Documents\Codex\Strata\launch-coder.cmd first."
}
if ($health.service -ne 'strata' -or $health.model -ne $model -or $models.data.id -notcontains $model) {
    throw 'Port 8082 is serving a different model. Start the Strata Coder launcher first.'
}

# Agent requests set their own thinking and sampling; no browser settings are changed.
$env:DEEP_AGENT_BACKEND = 'openai'
$env:DEEP_AGENT_BASE_URL = "$url/v1"
$env:DEEP_AGENT_MODEL = $model
$env:DEEP_AGENT_API_KEY = ''
$env:DEEP_AGENT_THINKING = $Thinking
$env:DEEP_AGENT_ACTION_THINKING_TOKENS = '8192'
$env:DEEP_AGENT_EXECUTION_MODE = 'unrestricted'
$env:DEEP_AGENT_COMMAND_TIMEOUT = '300'
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot 'runtime\uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot 'runtime\python'
Write-Host "Agent: Qwen Flash-Next Coder | Context: $($health.max_context) | Thinking: $Thinking"
Write-Host 'Use reasoning:low, reasoning:medium, reasoning:high or reasoning:off in the agent.'
if ($CheckOnly) { return }
& uv run --with-requirements requirements.txt python deep_agent.py
exit $LASTEXITCODE
