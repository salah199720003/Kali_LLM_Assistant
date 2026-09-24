param(
    [ValidateRange(1024, 262144)][int]$ContextSize = 102400,
    [ValidateSet('f16', 'q8_0')][string]$KvCacheType = 'q8_0',
    [ValidateRange(1, 65535)][int]$Port = 8080,
    [ValidateRange(0, 99)][int]$GpuLayers = 99,
    [string]$ServerPath = (Join-Path $PSScriptRoot 'runtime\bonsai\bin\llama-server.exe'),
    [string]$ModelPath = (Join-Path $PSScriptRoot 'runtime\bonsai\models\Ternary-Bonsai-2-27B-PQ2_0.gguf')
)

$ErrorActionPreference = 'Stop'
$baseUrl = "http://127.0.0.1:$Port"
$logDir = Join-Path $PSScriptRoot 'runtime\bonsai'
$stdoutLog = Join-Path $logDir 'server-out.log'
$stderrLog = Join-Path $logDir 'server-err.log'

function Get-ServedModels {
    try {
        $response = Invoke-RestMethod -Uri "$baseUrl/v1/models" -TimeoutSec 2
        return @($response.data | ForEach-Object { $_.id })
    } catch { return @() }
}

$existing = @(Get-ServedModels)
if ($existing.Count -gt 0) {
    if ($existing -notcontains 'bonsai-2-27b') {
        throw "Port $Port serves $($existing -join ', '). Stop that model server before starting Bonsai."
    }
    $props = Invoke-RestMethod -Uri "$baseUrl/props" -TimeoutSec 5
    $activeContext = $props.default_generation_settings.n_ctx
    if ($activeContext -ne $ContextSize) {
        throw "Bonsai is running with context $activeContext; requested $ContextSize. Restart its server to change context."
    }
    Write-Host "Bonsai 2 27B is already running with $activeContext context at $baseUrl/v1"
    return
}

foreach ($item in @($ServerPath, $ModelPath)) {
    if (-not (Test-Path -LiteralPath $item -PathType Leaf)) {
        throw "Missing Bonsai file: $item. Run setup-bonsai.ps1 first."
    }
}
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
# PrismML's ternary kernels are required. The existing K2/Vulkan runtime is
# incompatible with these weights. Q8 KV reduces memory at 100K context
# without using the experimental KV4 path.
$serverArguments = @(
    '-m', ('"{0}"' -f $ModelPath),
    '-ngl', "$GpuLayers", '-c', "$ContextSize", '-np', '1',
    '-fa', 'on', '--cache-type-k', $KvCacheType, '--cache-type-v', $KvCacheType,
    '-b', '512', '-ub', '128',
    '--jinja', '--alias', 'bonsai-2-27b', '--reasoning-format', 'deepseek',
    '--temp', '1.0', '--top-p', '0.95', '--top-k', '20', '--min-p', '0.05',
    '--host', '127.0.0.1', '--port', "$Port"
)
$server = Start-Process -FilePath $ServerPath -ArgumentList $serverArguments `
    -WorkingDirectory (Split-Path -Parent $ServerPath) -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru

Write-Host "Loading Bonsai 2 27B with $ContextSize context (PID $($server.Id))..."
try {
    for ($attempt = 0; $attempt -lt 120; $attempt++) {
        $server.Refresh()
        if ($server.HasExited) {
            $tail = if (Test-Path -LiteralPath $stderrLog) {
                (Get-Content -LiteralPath $stderrLog -Tail 20) -join [Environment]::NewLine
            } else { '' }
            throw "Bonsai server exited with code $($server.ExitCode). $tail"
        }
        if (@(Get-ServedModels) -contains 'bonsai-2-27b') {
            $props = Invoke-RestMethod -Uri "$baseUrl/props" -TimeoutSec 5
            if ($props.default_generation_settings.n_ctx -ne $ContextSize) {
                throw 'Bonsai loaded with an unexpected context size.'
            }
            $server.Id | Set-Content -LiteralPath (Join-Path $logDir 'server.pid') -Encoding ascii
            Write-Host "Bonsai 2 27B is ready at $baseUrl/v1"
            Write-Host "Server logs: $stdoutLog and $stderrLog"
            return
        }
        Start-Sleep -Seconds 2
    }
    throw "Bonsai did not become ready. Check $stderrLog."
} catch {
    # Clean up only the process this invocation created, never other servers.
    $server.Refresh()
    if (-not $server.HasExited) { $server.Kill() }
    throw
}
