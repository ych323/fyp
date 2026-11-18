$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
Set-Location -LiteralPath $ProjectRoot
& $PythonExe -u run.py all --config configs/pilot.yaml
if ($LASTEXITCODE -ne 0) { throw 'Pilot stopped. Inspect logs and metadata; previous artifacts retained.' }
