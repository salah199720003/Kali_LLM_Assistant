param(
    [string]$ServerPath = 'C:\Models\K2\llama-k2\build-vulkan\bin\llama-server.exe',
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [string]$RuntimePath = 'C:\Models\K2\toolchain\msys64\ucrt64\bin',
    [string]$Alias = 'agent-14b',
    [int]$Port = 8081,
    [int]$ContextSize = 102400,
    [string]$KvCacheType = 'q4_0',
    [switch]$GpuKvCache,
    [ValidateSet('auto', 'on', 'off')][string]$Reasoning = 'auto',
    [string]$ExtraArgs = ''
)

$ErrorActionPreference = 'Stop'
$modelUrl = "http://127.0.0.1:$Port/v1/models"
$propsUrl = "http://127.0.0.1:$Port/props"
$logDir = Split-Path -Parent $ModelPath
$stdoutLog = Join-Path $logDir 'server-out.log'
$stderrLog = Join-Path $logDir 'server-err.log'

function Get-ServerModels {
    try {
        $response = Invoke-RestMethod -Uri $modelUrl -TimeoutSec 2
        return @($response.data | ForEach-Object { $_.id })
    } catch {
        return @()
    }
}

$existing = @(Get-ServerModels)
if ($existing.Count -gt 0) {
    if ($existing -contains $Alias) {
        # Reasoning sets only the startup default of a new server. Agent and
        # browser requests each select their own thinking mode.
        Write-Host "Server already running with $Alias at http://127.0.0.1:$Port/v1"
        exit 0
    }
    throw "Port $Port already serves: $($existing -join ', '). Stop it or choose another port."
}

if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "Model file missing: $ModelPath"
}

$originalPath = $env:Path
try {
    $env:Path = "$RuntimePath;$originalPath"
    $serverArguments = @(
        '-m', ('"{0}"' -f $ModelPath),
        '-ngl', '99', '-c', "$ContextSize", '-np', '1'
    )
    if ($GpuKvCache) {
        $serverArguments += '--kv-offload'
    } else {
        $serverArguments += '--no-kv-offload'
    }
    $serverArguments += @(
        '--flash-attn', 'on',
        '--cache-type-k', $KvCacheType, '--cache-type-v', $KvCacheType,
        '--jinja', '--alias', $Alias,
        '--reasoning', $Reasoning,
        '--host', '127.0.0.1', '--port', "$Port"
    )
    # Browser assets only; the terminal agent still selects effort per request.
    $qwenBrowserPath = Join-Path $PSScriptRoot 'runtime\qwen-webui'
    if ($ModelPath -match 'Qwen3[.]8-27B' -and
        (Test-Path -LiteralPath (Join-Path $qwenBrowserPath 'effort-ui.json')) -and
        (Test-Path -LiteralPath (Join-Path $qwenBrowserPath 'index.html'))) {
        $serverArguments += @('--path', ('"{0}"' -f $qwenBrowserPath))
    }
    if ($ExtraArgs) {
        $serverArguments += ($ExtraArgs -split '\s+')
    }
    $server = Start-Process -FilePath $ServerPath -ArgumentList $serverArguments `
        -WorkingDirectory (Split-Path -Parent $ServerPath) -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru
} finally {
    $env:Path = $originalPath
}

for ($attempt = 0; $attempt -lt 90; $attempt++) {
    Start-Sleep -Seconds 2
    if (@(Get-ServerModels) -contains $Alias) {
        Write-Host "$Alias ready with $ContextSize context at http://127.0.0.1:$Port/v1 (PID $($server.Id))."
        Write-Host "Server logs: $stdoutLog and $stderrLog"
        exit 0
    }
    $server.Refresh()
    if ($server.HasExited) {
        $errorTail = if (Test-Path -LiteralPath $stderrLog) {
            (Get-Content -LiteralPath $stderrLog -Tail 12) -join [Environment]::NewLine
        } else { '' }
        throw "llama-server exited with code $($server.ExitCode). $errorTail"
    }
}

throw "Server did not become ready within 180 seconds. Check $stderrLog"
