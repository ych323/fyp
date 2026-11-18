$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvRoot = Join-Path $ProjectRoot '.venv'
$PythonExe = Join-Path $VenvRoot 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $PythonExe)) {
    python -m venv $VenvRoot
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
& $PythonExe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { throw 'CUDA torch installation failed.' }
& $PythonExe -m pip install -r (Join-Path $ProjectRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Write-Output 'Installed in repository-local .venv. Original environment record was not overwritten.'
