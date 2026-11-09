$ErrorActionPreference = 'Stop'
$directory = Join-Path $PSScriptRoot 'data\tessdata'
New-Item -ItemType Directory -Force -Path $directory | Out-Null
$target = Join-Path $directory 'eng.traineddata'
if (-not (Test-Path -LiteralPath $target)) {
    $temporary = Join-Path $directory ('eng-' + [Guid]::NewGuid().ToString('N') + '.download')
    try {
        Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/4.1.0/eng.traineddata' -OutFile $temporary -UseBasicParsing -TimeoutSec 60
        Move-Item -LiteralPath $temporary -Destination $target
    } finally {
        if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary }
    }
}
Write-Output 'English OCR data is ready. Restart the ResearchFlow backend if it is running.'
