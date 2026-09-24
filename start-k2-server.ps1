param(
    [string]$ServerPath = 'C:\Models\K2\llama-k2\build-vulkan\bin\llama-server.exe',
    [string]$ModelPath = 'C:\Models\K2\K2-Horizon-7B-Q5_K_M.gguf',
    [string]$RuntimePath = 'C:\Models\K2\toolchain\msys64\ucrt64\bin',
    [int]$Port = 8080,
    [int]$ContextSize = 102400,
    [string]$KvCacheType = 'q4_0',
    [switch]$GpuKvCache
)

$ErrorActionPreference = 'Stop'
if ($GpuKvCache -and -not $PSBoundParameters.ContainsKey('ContextSize')) {
    $ContextSize = 32768
}
$modelUrl = "http://127.0.0.1:$Port/v1/models"
$propsUrl = "http://127.0.0.1:$Port/props"
$logDir = Split-Path -Parent $ModelPath
$stdoutLog = Join-Path $logDir 'server-out.log'
$stderrLog = Join-Path $logDir 'server-err.log'

function Get-K2ServerModel {
    try {
        $response = Invoke-RestMethod -Uri $modelUrl -TimeoutSec 2
        return @($response.data | ForEach-Object { $_.id })
    } catch {
        return @()
    }
}

$existingModels = @(Get-K2ServerModel)
if ($existingModels.Count -gt 0) {
    if ($existingModels -contains 'k2-horizon') {
        $existingProps = Invoke-RestMethod -Uri $propsUrl -TimeoutSec 5
        $activeContext = $existingProps.default_generation_settings.n_ctx
        if ($activeContext -ne $ContextSize) {
            throw "K2 is already running with context $activeContext; requested $ContextSize. Stop the current llama-server process and run this launcher again to change it."
        }
        if ($GpuKvCache) {
            throw "K2 is already running. Stop the current llama-server process before switching to the GPU KV cache mode."
        }
        Write-Host "K2 Horizon is already running with $activeContext context at http://127.0.0.1:$Port/v1"
        exit 0
    }
    throw "Port $Port already serves another model: $($existingModels -join ', ')"
}

if ($ContextSize -lt 1024 -or $ContextSize -gt 524288) {
    throw 'ContextSize must be between 1024 and the model maximum of 524288.'
}

foreach ($item in @($ServerPath, $ModelPath, $RuntimePath)) {
    if (-not (Test-Path -LiteralPath $item)) {
        throw "Required K2 file or directory is missing: $item"
    }
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
        '--jinja', '--alias', 'k2-horizon',
        '--host', '127.0.0.1', '--port', "$Port"
    )
    $server = Start-Process -FilePath $ServerPath -ArgumentList $serverArguments `
        -WorkingDirectory (Split-Path -Parent $ServerPath) -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru
} finally {
    $env:Path = $originalPath
}

for ($attempt = 0; $attempt -lt 60; $attempt++) {
    Start-Sleep -Seconds 2
    if (@(Get-K2ServerModel) -contains 'k2-horizon') {
        $kvLocation = if ($GpuKvCache) { 'GPU' } else { 'system RAM' }
        Write-Host "K2 Horizon is ready with $ContextSize context and $kvLocation KV cache at http://127.0.0.1:$Port/v1 (PID $($server.Id))."
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

throw "K2 did not become ready within 120 seconds. Check $stderrLog"
