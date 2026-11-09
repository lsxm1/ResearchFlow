$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Force -Path '.cache\tmp' | Out-Null
$env:TEMP = Join-Path $PSScriptRoot '.cache\tmp'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $PSScriptRoot '.cache\pip'
$env:npm_config_cache = Join-Path $PSScriptRoot '.cache\npm'
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create Python virtual environment.' }
}
& '.\.venv\Scripts\python.exe' -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) { throw 'Failed to bootstrap pip.' }
& '.\.venv\Scripts\python.exe' -m pip install -r backend/requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Backend dependency installation failed.' }
Set-Location -LiteralPath (Join-Path $PSScriptRoot 'frontend')
& npm.cmd ci
if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
& (Join-Path $PSScriptRoot 'setup-ocr.ps1')
Write-Output 'Ready. Run start.ps1 for the single-server workbench.'
