$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

# Bound temporary processing buffers so desktop apps have more VRAM headroom.
# At 96K, the extra MTP weights and buffers caused severe input-processing
# delays on this 12 GB GPU. Standard decoding retains the same model weights.
& (Join-Path $PSScriptRoot 'start-llm-server.ps1') -ServerPath 'C:\Models\llama-cuda\bin\llama-server.exe' -ModelPath 'C:\Models\Qwen38-27B\Qwen3.8-27B-UD-IQ3_XXS.gguf' -Alias agent-27b -Port 8081 -ContextSize 96000 -ExtraArgs '--spec-type none --parallel 1 -b 512 -ub 128'

$env:DEEP_AGENT_BACKEND = 'llama'
$env:DEEP_AGENT_BASE_URL = 'http://127.0.0.1:8081/v1'
$env:DEEP_AGENT_MODEL = 'agent-27b'
$env:DEEP_AGENT_EXECUTION_MODE = 'unrestricted'
# Agent request default only; reasoning:high/off overrides this in the terminal.
$env:DEEP_AGENT_THINKING = 'off'
# Metasploit boots in 60-120s; the default 90s command timeout killed PoCs mid-startup.
$env:DEEP_AGENT_COMMAND_TIMEOUT = '300'
Write-Host "Agent backend: agent-27b (Qwen3.8 27B IQ3_XXS) at $env:DEEP_AGENT_BASE_URL"
uv run --with-requirements requirements.txt python deep_agent.py
