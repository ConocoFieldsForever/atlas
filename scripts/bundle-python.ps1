#!/usr/bin/env pwsh
# Download + set up a self-contained, RELOCATABLE Windows embeddable Python for the pack-builder
# kit, so a non-dev user needs NO system Python installed. This ships Python + pip only (~30 MB);
# the heavy extraction deps (UnityPy/numpy/Pillow, ~130 MB) are installed on first use by the app's
# INSTALL DEPS button (tools/setup_deps.py) INTO this same Python. Validated: enabling site-packages
# via ._pth + get-pip makes `import UnityPy, numpy, PIL` work and the tree is relocatable.
#
#   .\scripts\bundle-python.ps1 -Dest dist\python
param(
    [Parameter(Mandatory)][string]$Dest,
    [string]$Version = "3.11.9"
)
$ErrorActionPreference = "Stop"
$parts = $Version.Split('.')
$tag = "python$($parts[0])$($parts[1])"          # e.g. python311
$tmp = Join-Path $env:TEMP ("pyembed-" + $Version)
if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
New-Item -ItemType Directory -Force $tmp | Out-Null

$zip = Join-Path $tmp "py.zip"
$url = "https://www.python.org/ftp/python/$Version/python-$Version-embed-amd64.zip"
Write-Host "[bundle-python] downloading $url"
Invoke-WebRequest -Uri $url -OutFile $zip

if (Test-Path $Dest) { Remove-Item -Recurse -Force $Dest }
Expand-Archive -Path $zip -DestinationPath $Dest -Force

# The embeddable dist disables site-packages by default (._pth has `#import site`). Rewrite it to
# enable site + add Lib\site-packages so pip-installed packages are importable. Paths are relative
# to python.exe, so the tree stays relocatable to the user's extraction folder.
$pth = Join-Path $Dest "$tag._pth"
# The '..' entry is the kit root (one dir above python\), so "python -m eft_pipeline..." finds the
# pipeline packages. The embeddable dist ignores the working directory, so without this a build
# fails at the assemble stage with ModuleNotFoundError: No module named eft_pipeline. "import site"
# enables the pip-installed deps (UnityPy/numpy/Pillow). (Keep this file ASCII-only.)
Set-Content -Encoding ascii -Path $pth -Value @(
    "$tag.zip"
    "."
    ".."
    "Lib\site-packages"
    ""
    "import site"
)

# Bootstrap pip (embeddable python has no ensurepip).
$getpip = Join-Path $tmp "get-pip.py"
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getpip
& (Join-Path $Dest "python.exe") $getpip --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw "get-pip failed (rc=$LASTEXITCODE)" }

# Sanity: python + pip run.
& (Join-Path $Dest "python.exe") -m pip --version
if ($LASTEXITCODE -ne 0) { throw "pip not runnable after bootstrap" }
$mb = [math]::Round((Get-ChildItem -Recurse $Dest | Measure-Object Length -Sum).Sum / 1MB, 1)
Write-Host "[bundle-python] ready at $Dest ($mb MB, Python $Version + pip; deps install on first INSTALL DEPS)"
