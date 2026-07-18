# release.ps1 — build the shippable Atlas bundles LOCALLY (redistribution PR5).
# PowerShell 5.1-safe (no &&). The GitHub workflow mirrors this; nothing runs in CI until
# this script has been proven locally (user directive: no wasted credits).
#
#   .\scripts\release.ps1              # Tier A: viewer-only zip
#   .\scripts\release.ps1 -Full       # Tier B: + python pipeline kit
#   .\scripts\release.ps1 -SkipBuild -SkipRenderSmoke   # CI mode (no GPU on runners)
#
# NEVER includes packs/*.eftpack or anything game-derived (see LICENSE-NOTES.md).

param(
    [switch]$Full,
    [switch]$SkipBuild,
    [switch]$SkipRenderSmoke,
    [string]$SmokePack = "packs\factory.eftpack"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# 0. A running viewer holds a lock on the exe -> stale-binary trap (project build-loop rule).
$running = Get-Process atlas -ErrorAction SilentlyContinue
if ($running -and -not $SkipBuild) {
    throw "atlas.exe is running (PID $($running.Id -join ',')) - close it first (locked exe = stale binary)"
}

# 1. Version = Cargo.toml package version + short git hash.
$verLine = Select-String -Path "viewer\Cargo.toml" -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1
$ver = $verLine.Matches[0].Groups[1].Value
$sha = (git rev-parse --short HEAD).Trim()
$name = "atlas-$ver-$sha-win64"
Write-Host "[release] $name"

# 2. Build (locked deps for reproducibility).
if (-not $SkipBuild) {
    $before = (Get-Item "target\release\atlas.exe" -ErrorAction SilentlyContinue).LastWriteTime
    # cmd does the stderr merge: PS 5.1's own 2>&1 on native exes wraps stderr lines in
    # ErrorRecords and $ErrorActionPreference=Stop turns cargo WARNINGS into a fatal throw.
    $buildOut = cmd /c "cargo build --release --locked 2>&1" | Out-String
    if ($LASTEXITCODE -ne 0) { Write-Host $buildOut; throw "cargo build failed ($LASTEXITCODE)" }
    # Only demand a fresh mtime when cargo actually recompiled the bin (an up-to-date build
    # legitimately leaves it untouched; the locked-exe trap is caught by the step-0 check).
    if ($buildOut -match "Compiling atlas") {
        $after = (Get-Item "target\release\atlas.exe").LastWriteTime
        if ($before -and $after -le $before) { throw "binary mtime did not advance - stale build?" }
    }
}

# 3. Version smoke (works on GPU-less machines; the only CI-safe check).
$verOut = & "target\release\atlas.exe" --version
if ($LASTEXITCODE -ne 0 -or -not ($verOut -match "atlas")) { throw "--version smoke failed: $verOut" }
Write-Host "[release] smoke: $verOut"

# 4. Assemble dist tree.
$dist = "dist\$name"
if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
New-Item -ItemType Directory -Force "$dist\packs\shared" | Out-Null
Copy-Item "target\release\atlas.exe" $dist
# wired shaders only (instanced/sh_gi/splat are dead - provenance audit)
New-Item -ItemType Directory -Force "$dist\assets\shaders" | Out-Null
foreach ($sh in "gpu_cull.wgsl","gpu_draw.wgsl","gpu_shadow.wgsl","ssao.wgsl","grade.wgsl","instancing_m0.wgsl") {
    Copy-Item "viewer\assets\shaders\$sh" "$dist\assets\shaders\"
}
# README.md is the friendly non-dev guide (first-run + SmartScreen "Run anyway" steps);
# README_DIST.md is the technical/env-toggle reference. Ship both.
Copy-Item "README.md" $dist -ErrorAction SilentlyContinue
Copy-Item "README_DIST.md" $dist -ErrorAction SilentlyContinue
Copy-Item "LICENSE-NOTES.md" $dist -ErrorAction SilentlyContinue

if ($Full) {
    # Tier B: the python pipeline kit ("build your own packs").
    foreach ($d in "extraction","eft_pipeline","tools") {
        Copy-Item -Recurse $d "$dist\$d"
    }
    # prune caches + anything game-derived that may sit in the workspace copies
    Get-ChildItem -Recurse "$dist" -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
    # The grade LUT source (lut_amidgen_bluegreen.png) is extracted from the game's
    # resources.assets and eft_grade_lut.bin is its derivative — non-redistributable
    # (LICENSE-NOTES.md). Pack builders regenerate them locally via make_grade_lut_game.py.
    Remove-Item "$dist\extraction\grade\lut_amidgen_bluegreen.png","$dist\extraction\grade\eft_grade_lut.bin" -Force -ErrorAction SilentlyContinue
    Copy-Item "extraction\requirements.txt" $dist
    Copy-Item "scripts\bootstrap.ps1" $dist
}

# Belt-and-braces: no pack/game data may ship.
$leaks = Get-ChildItem -Recurse "$dist\packs" -Filter "*.eftpack" -ErrorAction SilentlyContinue
if ($leaks) { throw "packs leaked into dist: $($leaks.FullName -join ', ')" }

# 5. Render smoke: full headless load + screenshot against a local pack (LOCAL ONLY - GPU).
if (-not $SkipRenderSmoke) {
    if (-not (Test-Path $SmokePack)) {
        Write-Warning "no smoke pack at $SmokePack - skipping render smoke"
    } else {
        $shot = Join-Path (Resolve-Path "dist") "smoke.png"
        if (Test-Path $shot) { Remove-Item $shot }
        $env:EFT_HIDDEN = "1"; $env:EFT_UNCAPPED = "1"; $env:EFT_SHOT = $shot
        $p = Start-Process -FilePath "$dist\atlas.exe" -ArgumentList (Resolve-Path $SmokePack) -PassThru -WindowStyle Hidden
        $deadline = (Get-Date).AddSeconds(120)
        while ((Get-Date) -lt $deadline -and -not (Test-Path $shot)) { Start-Sleep -Seconds 2 }
        try { Stop-Process -Id $p.Id -Force -ErrorAction Stop } catch {}
        Remove-Item Env:\EFT_HIDDEN, Env:\EFT_UNCAPPED, Env:\EFT_SHOT -ErrorAction SilentlyContinue
        if (-not (Test-Path $shot)) { throw "render smoke: no screenshot produced" }
        if ((Get-Item $shot).Length -lt 10kb) { throw "render smoke: screenshot suspiciously small" }
        Write-Host "[release] render smoke OK ($([math]::Round((Get-Item $shot).Length/1kb)) KB)"
    }
}

# 5b. The render smoke runs the dist exe, which writes game-derived BC texcache into
#     <dist>\packs\shared\texcache (paths.rs anchors the cache beside the exe). Purge everything
#     the exe generated under packs\ so only an empty shared\ ships.
if (Test-Path "$dist\packs") {
    Get-ChildItem "$dist\packs" -Recurse -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Force "$dist\packs\shared" | Out-Null
New-Item -ItemType File -Force "$dist\packs\shared\.keep" | Out-Null

# 5c. Final belt-and-braces: NOTHING game-derived may ship — packs, texcache, BC blobs, extracted
#     grade LUTs, or the tarkov.dev intel caches. Fail closed on any match anywhere in the tree.
$bad = Get-ChildItem -Recurse "$dist" -File -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -match '\.eftpack$|\.bc[0-9]|lut_amidgen|eft_grade_lut\.bin$' -or
    $_.FullName -match '\\texcache\\' -or
    $_.Name -in @('loot.json','tasks.json')
}
if ($bad) { throw "game-derived data leaked into dist: $($bad.FullName -join ', ')" }

# 6. Zip + checksum.
$zip = "dist\$name$(if ($Full) { '-full' }).zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path "$dist\*" -DestinationPath $zip
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash
"$hash  $(Split-Path -Leaf $zip)" | Out-File -Encoding ascii "$zip.sha256"
Write-Host "[release] $zip  SHA256=$hash"
