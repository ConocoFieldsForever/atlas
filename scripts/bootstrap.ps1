# bootstrap.ps1 — one-command python setup for the pack-building kit (Tier B bundle).
# Creates .\venv, installs the base requirements, and checks the environment.
# GPU lighting bake (optional, NVIDIA CUDA only): pip install -r extraction\requirements-bake.txt
param([string]$Python = "")

$ErrorActionPreference = "Stop"
# Locate the kit root (the dir holding extraction\). In the shipped bundle this script sits AT that
# root beside extraction\; in the dev tree it lives in scripts\ and the kit is its parent.
if (Test-Path (Join-Path $PSScriptRoot "extraction")) {
    $here = $PSScriptRoot
} else {
    $here = Split-Path -Parent $PSScriptRoot
}
if ($here -notmatch "\S") { $here = "." }
Set-Location $here

if (-not $Python) {
    $cand = Get-Command py -ErrorAction SilentlyContinue
    if ($cand) { $Python = "py -3.10" } else { $Python = "python" }
}

Write-Host "[bootstrap] creating venv with: $Python"
Invoke-Expression "$Python -m venv venv"
& ".\venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\venv\Scripts\python.exe" -m pip install -r "extraction\requirements.txt"
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

& ".\venv\Scripts\python.exe" "extraction\check_env.py"
Write-Host "[bootstrap] done. The viewer menu will use .\venv automatically (or set EFT_PY)."
