param(
    [string]$ServerPath = 'C:\Models\K2\llama-k2\build-vulkan\bin\llama-server.exe',
    [string]$ModelPath = (Join-Path $env:USERPROFILE '.ollama\models\blobs\sha256-42e0c2cf595cd9fa93d723b416d33a3a0ccb9b87866067d0f152a97e053a2419'),
    [string]$RuntimePath = 'C:\Models\K2\toolchain\msys64\ucrt64\bin',
    [int]$Port = 8080,
    [ValidateRange(1024, 131072)][int]$ContextSize = 102400
)

$ErrorActionPreference = 'Stop'
$modelUrl = "http://127.0.0.1:$Port/v1/models"
$propsUrl = "http://127.0.0.1:$Port/props"
function Get-DeepHatModels {
    try {
        $reply = Invoke-RestMethod -Uri $modelUrl -TimeoutSec 2
        return @($reply.data | ForEach-Object { $_.id })
    } catch { return @() }
}
$existing = @(Get-DeepHatModels)
if ($existing.Count) {
    if ($existing -notcontains 'deephat-v1-7b') {
        throw "Port $Port serves another model ($($existing -join ', ')). Stop that server before launching DeepHat."
    }
    $props = Invoke-RestMethod -Uri $propsUrl -TimeoutSec 5
    $activeContext = $props.default_generation_settings.n_ctx
    if ($activeContext -ne $ContextSize) {
        throw "DeepHat is running with $activeContext context. Restart its server to use $ContextSize."
    }
    Write-Host "DeepHat is already running with $activeContext context at http://127.0.0.1:$Port/v1"
    return
}
foreach ($required in @($ServerPath, $ModelPath, $RuntimePath)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required file or directory is missing: $required" }
}
$logDir = Join-Path $PSScriptRoot 'runtime'
[void](New-Item -ItemType Directory -Path $logDir -Force)
$stdoutLog = Join-Path $logDir 'deephat-out.log'
$stderrLog = Join-Path $logDir 'deephat-err.log'
$serverArguments = @(
    '-m', ('"{0}"' -f $ModelPath), '--alias', 'deephat-v1-7b',
    '-ngl', '99', '-c', "$ContextSize", '-np', '1', '--fit', 'off',
    '--flash-attn', 'on', '--kv-offload',
    '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0',
    '-b', '512', '-ub', '128', '--jinja',
    '--host', '127.0.0.1', '--port', "$Port"
)
if ($ContextSize -gt 32768) {
    $serverArguments += @('--rope-scaling', 'yarn', '--rope-scale', '4', '--yarn-orig-ctx', '32768')
    # This server caps slots at the GGUF context metadata even with YaRN.
    # Expose the documented extended limit, preserving YaRN's true original
    # context above. This is a runtime override; the model file is unchanged.
    $serverArguments += @('--override-kv', 'qwen2.context_length=int:131072')
}
$originalPath = $env:Path
try {
    $env:Path = "$RuntimePath;$originalPath"
    $server = Start-Process -FilePath $ServerPath -ArgumentList $serverArguments `
        -WorkingDirectory (Split-Path -Parent $ServerPath) -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru
} finally { $env:Path = $originalPath }
for ($attempt = 0; $attempt -lt 90; $attempt++) {
    Start-Sleep -Seconds 2
    $server.Refresh()
    if ($server.HasExited) {
        $tail = (Get-Content -LiteralPath $stderrLog -Tail 15 -ErrorAction SilentlyContinue) -join [Environment]::NewLine
        throw "DeepHat server exited. $tail"
    }
    if (@(Get-DeepHatModels) -contains 'deephat-v1-7b') {
        $props = Invoke-RestMethod -Uri $propsUrl -TimeoutSec 5
        if ($props.default_generation_settings.n_ctx -ne $ContextSize) {
            throw 'DeepHat loaded with an unexpected context size. Check the server log.'
        }
        Write-Host "DeepHat is ready: $ContextSize context, Q8 GPU KV cache, Flash Attention (PID $($server.Id))."
        Write-Host "Server log: $stderrLog"
        return
    }
}
throw "DeepHat did not become ready within the startup window. Check $stderrLog"
