param([switch]$LocalModel)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe') -or -not (Test-Path -LiteralPath 'frontend\node_modules')) {
    & (Join-Path $PSScriptRoot 'setup.ps1')
    Set-Location -LiteralPath $PSScriptRoot
}
Push-Location -LiteralPath 'frontend'
try {
    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
} finally { Pop-Location }
$model = $null
try {
    if ($LocalModel) {
        $directory = Join-Path $PSScriptRoot '.cache\local-model'
        $runner = Join-Path $directory 'runner\llama-server.exe'
        $weights = Join-Path $directory 'qwen-0.5b.gguf'
        if (-not (Test-Path -LiteralPath $runner) -or -not (Test-Path -LiteralPath $weights)) {
            & (Join-Path $PSScriptRoot 'setup-local-model.ps1')
        }
        $arguments = @('-m', ('"' + $weights + '"'), '--alias', 'qwen-local', '--host', '127.0.0.1', '--port', '8091', '-c', '8192', '-t', '6', '-ngl', '0', '--parallel', '1', '--api-key', 'researchflow-local')
        $model = Start-Process -FilePath $runner -ArgumentList $arguments -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $directory 'model.out.log') -RedirectStandardError (Join-Path $directory 'model.err.log')
        $deadline = (Get-Date).AddSeconds(30)
        $ready = $false
        while ((Get-Date) -lt $deadline) {
            if ($model.HasExited) { break }
            try {
                $response = Invoke-WebRequest -Uri 'http://127.0.0.1:8091/health' -UseBasicParsing -TimeoutSec 2
                if ($response.StatusCode -eq 200) { $ready = $true; break }
            } catch { Start-Sleep -Milliseconds 400 }
        }
        if (-not $ready) { throw 'Local model failed to start. See .cache/local-model/model.err.log.' }
        $env:RF_MODEL_MODE = 'openai_compatible'
        $env:RF_MODEL_BASE_URL = 'http://127.0.0.1:8091/v1'
        $env:RF_MODEL_NAME = 'qwen-local'
        $env:RF_MODEL_API_KEY = 'researchflow-local'
        $env:RF_MODEL_JSON_MODE = '1'
        $env:RF_MODEL_TEXT_TRANSLATION = '1'
        $env:RF_MODEL_PROFILE = 'compact'
        $env:RF_MODEL_MAX_TOKENS = '1200'
    }
    & '.\.venv\Scripts\python.exe' -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
} finally {
    if ($model -and -not $model.HasExited) { Stop-Process -Id $model.Id -ErrorAction SilentlyContinue }
}
