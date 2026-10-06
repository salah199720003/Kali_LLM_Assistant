$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
uv run --with-requirements requirements.txt python experiments/q4ks_agent.py
exit $LASTEXITCODE
