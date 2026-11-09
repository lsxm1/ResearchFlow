$ErrorActionPreference = 'Stop'
$directory = Join-Path $PSScriptRoot '.cache\local-model'
New-Item -ItemType Directory -Force -Path $directory | Out-Null

function Get-VerifiedAsset($Url, $Name, $Expected) {
    $target = Join-Path $directory $Name
    if ((Test-Path -LiteralPath $target) -and (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -eq $Expected) {
        return $target
    }
    $part = $target + '.part'
    & curl.exe --fail --location --silent --show-error --connect-timeout 15 --max-time 1200 $Url -o $part
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $Name" }
    if ((Get-FileHash -LiteralPath $part -Algorithm SHA256).Hash -ne $Expected) { throw "Checksum mismatch: $Name" }
    Move-Item -LiteralPath $part -Destination $target -Force
    return $target
}

$archive = Get-VerifiedAsset 'https://github.com/ggml-org/llama.cpp/releases/download/b12223/llama-b12223-bin-win-cpu-x64.zip' 'runner.zip' '3d10402849209453c83d57c556ce84613fca7628b03c93efaddcd9715206a904'
$runner = Join-Path $directory 'runner'
Expand-Archive -LiteralPath $archive -DestinationPath $runner -Force
if (-not (Test-Path -LiteralPath (Join-Path $runner 'llama-server.exe'))) { throw 'llama-server executable is missing.' }
Get-VerifiedAsset 'https://registry.ollama.ai/v2/library/qwen2.5/blobs/sha256:c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515' 'qwen-0.5b.gguf' 'c5396e06af294bd101b30dce59131a76d2b773e76950acc870eda801d3ab0515' | Out-Null
Write-Output 'Local model is ready.'
