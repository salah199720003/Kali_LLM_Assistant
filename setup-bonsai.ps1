param([switch]$SkipModel)

$ErrorActionPreference = 'Stop'
$bonsaiProfile = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'bonsai-profile.json') -Raw | ConvertFrom-Json
$root = Join-Path $PSScriptRoot 'runtime\bonsai'
$downloads = Join-Path $root 'downloads'
$bin = Join-Path $root 'bin'
$models = Join-Path $root 'models'
New-Item -ItemType Directory -Force -Path $downloads, $bin, $models | Out-Null

function Test-Sha256([string]$Path, [string]$Expected) {
    return (Test-Path -LiteralPath $Path -PathType Leaf) -and
        ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -eq $Expected)
}

foreach ($asset in $bonsaiProfile.runtime_assets) {
    $archive = Join-Path $downloads $asset.filename
    if (-not (Test-Sha256 $archive $asset.sha256)) {
        Write-Host "Downloading PrismML runtime: $($asset.filename)"
        $url = "https://github.com/PrismML-Eng/llama.cpp/releases/download/$($bonsaiProfile.runtime_release)/$($asset.filename)"
        $client = New-Object System.Net.WebClient
        try {
            $client.Headers['User-Agent'] = 'deep-vm-agent-model-setup'
            $client.DownloadFile($url, $archive)
        } finally {
            $client.Dispose()
        }
        if (-not (Test-Sha256 $archive $asset.sha256)) { throw "Runtime SHA-256 mismatch: $archive" }
    }
    Write-Host "Extracting verified runtime: $($asset.filename)"
    Expand-Archive -LiteralPath $archive -DestinationPath $bin -Force
}

if (-not (Test-Path -LiteralPath (Join-Path $bin 'llama-server.exe'))) {
    throw "The runtime archive did not contain bin\llama-server.exe. Inspect $bin."
}
if ($SkipModel) { return }

$model = Join-Path $models $bonsaiProfile.filename
if (-not (Test-Sha256 $model $bonsaiProfile.model_sha256)) {
    if (-not (Get-Command hf -ErrorAction SilentlyContinue)) {
        throw 'Hugging Face CLI (hf) is required to download the pinned model.'
    }
    Write-Host "Downloading Bonsai 2 27B PQ2_0 (7.21 GB), revision $($bonsaiProfile.revision)"
    $oldHfHome = $env:HF_HOME
    $oldUtf8 = $env:PYTHONUTF8
    try {
        $env:HF_HOME = Join-Path $root 'hf-cache'
        $env:PYTHONUTF8 = '1'
        & hf download $bonsaiProfile.repository $bonsaiProfile.filename --revision $bonsaiProfile.revision --local-dir $models --format human
        if ($LASTEXITCODE -ne 0) { throw 'Bonsai model download failed.' }
    } finally {
        $env:HF_HOME = $oldHfHome
        $env:PYTHONUTF8 = $oldUtf8
    }
    Write-Host 'Verifying model SHA-256...'
    if (-not (Test-Sha256 $model $bonsaiProfile.model_sha256)) { throw 'Bonsai model SHA-256 mismatch.' }
}
$bonsaiProfile.revision | Set-Content -LiteralPath (Join-Path $root 'verified-revision.txt') -Encoding ascii
Write-Host 'Bonsai runtime and model are downloaded and verified.'
