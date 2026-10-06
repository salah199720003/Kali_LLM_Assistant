$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

# Local alias of hf.co/empero-ai/Qwen3.8-27B-Ridge-GGUF:latest (3.69 bpw).
$model = 'ridge:3.7'
$ollama = Get-Command ollama -ErrorAction Stop
$modelInfo = & $ollama.Source show $model 2>&1
if ($LASTEXITCODE -ne 0) {
    $details = ($modelInfo | Out-String).Trim()
    throw "Ollama could not open the installed model '$model'. $details"
}

$env:DEEP_AGENT_BACKEND = 'ollama'
$env:DEEP_AGENT_MODEL = $model
$env:DEEP_AGENT_OLLAMA_NUM_CTX = '32768'
# true enables the model template's default xhigh reasoning level.
$env:DEEP_AGENT_OLLAMA_THINK = 'true'
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot 'runtime/uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot 'runtime/python'
Write-Host "Agent backend: Ridge 3.7 via Ollama ($model, xhigh reasoning, 32K context)"
& uv run --with-requirements requirements.txt python deep_agent.py
exit $LASTEXITCODE
