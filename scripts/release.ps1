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
    [switch]$SkipPython,   # -Full only: skip bundling the embeddable Python (fast local reruns)
    # A pack this repo actually builds. It used to name packs\factory.eftpack, which no longer
    # exists (the shipped map is the 1.0 rework, factory_rework), so the render smoke silently
    # took the "no smoke pack - skipping" branch and the release passed without ever rendering.
    [string]$SmokePack = "packs\factory_rework.eftpack"
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
# Source archives / delegated workspaces may not carry a root .git directory. Keep local release
# packaging usable there while preserving the commit suffix for normal clones and CI checkouts.
$sha = "local"
if (Test-Path (Join-Path $repo ".git")) {
    $shaOut = & git rev-parse --short HEAD
    if ($LASTEXITCODE -eq 0 -and $shaOut) { $sha = $shaOut.Trim() }
}
$name = "atlas-$ver-$sha-win64"
Write-Host "[release] $name"

# 2. Build (locked deps for reproducibility) — ALWAYS through build-clean.ps1, which remaps the
#    build machine's cargo-registry + workspace prefixes out of the binary. release.ps1 used to
#    call plain `cargo build` here, so LOCALLY-built zips shipped panic Location strings containing
#    the builder's C:\Users\<name>\.cargo\... paths (caught in a user's crash log — CI already
#    routed through build-clean; local was the leak).
if (-not $SkipBuild) {
    & "$PSScriptRoot\build-clean.ps1" -Locked
    if ($LASTEXITCODE -ne 0) { throw "build-clean.ps1 failed ($LASTEXITCODE)" }
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
foreach ($sh in "gpu_cull.wgsl","gpu_draw.wgsl","gpu_shadow.wgsl","ssao.wgsl","grade.wgsl","instancing_m0.wgsl","fpv_cam.wgsl") {
    # Fail closed: a shader the binary loads at runtime but missing from this allowlist ships a
    # broken feature (fpv_cam.wgsl was caught ONLY by a user's log — bevy_asset 404s at runtime).
    if (-not (Test-Path "viewer\assets\shaders\$sh")) { throw "release: shader $sh missing from workspace" }
    Copy-Item "viewer\assets\shaders\$sh" "$dist\assets\shaders\"
}
# One README now: the non-dev first-run guide (incl. the SmartScreen "Run anyway" steps) and the
# technical env-toggle reference were merged, so there is a single file to ship and to keep current.
Copy-Item "README.md" $dist -ErrorAction SilentlyContinue
Copy-Item "LICENSE-NOTES.md" $dist -ErrorAction SilentlyContinue
# Field-test protocol for the target-GPU testers (RTX 4060 / RX 6800 round): ships at the zip
# root so a tester can't miss it.
Copy-Item "docs\FIELD_TEST_GPU.md" "$dist\FIELD_TEST.md" -ErrorAction SilentlyContinue

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
    # Bundle a self-contained embeddable Python (+ pip, ~30 MB) so a non-dev needs NO system Python.
    # The heavy extraction deps (UnityPy/numpy/Pillow) install on first INSTALL DEPS into this same
    # Python (tools/setup_deps.py detects the embeddable interpreter). paths::python_exe prefers it.
    if (-not $SkipPython) {
        & "$PSScriptRoot\bundle-python.ps1" -Dest "$dist\python"
        if ($LASTEXITCODE -ne 0) { throw "bundle-python.ps1 failed (rc=$LASTEXITCODE)" }
    } else {
        Write-Host "[release] -SkipPython: no bundled Python in this dist"
    }
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
        $oldHidden = $env:EFT_HIDDEN
        $oldUncapped = $env:EFT_UNCAPPED
        $oldShot = $env:EFT_SHOT
        $p = $null
        try {
            $env:EFT_HIDDEN = "1"; $env:EFT_UNCAPPED = "1"; $env:EFT_SHOT = $shot
            $p = Start-Process -FilePath "$dist\atlas.exe" -ArgumentList (Resolve-Path $SmokePack) -PassThru -WindowStyle Hidden
            $deadline = (Get-Date).AddSeconds(120)
            while ((Get-Date) -lt $deadline -and -not (Test-Path $shot) -and -not $p.HasExited) {
                Start-Sleep -Seconds 2
            }
        } finally {
            if ($null -ne $p -and -not $p.HasExited) {
                try { Stop-Process -Id $p.Id -Force -ErrorAction Stop } catch {}
            }
            if ($null -eq $oldHidden) { Remove-Item Env:\EFT_HIDDEN -ErrorAction SilentlyContinue } else { $env:EFT_HIDDEN = $oldHidden }
            if ($null -eq $oldUncapped) { Remove-Item Env:\EFT_UNCAPPED -ErrorAction SilentlyContinue } else { $env:EFT_UNCAPPED = $oldUncapped }
            if ($null -eq $oldShot) { Remove-Item Env:\EFT_SHOT -ErrorAction SilentlyContinue } else { $env:EFT_SHOT = $oldShot }
        }
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
    $_.Name -in @('loot.json','tasks.json') -or
    # compiled-python bytecode embeds the build machine's absolute paths (username leak);
    # bundle-python.ps1 strips its own __pycache__, this fails closed on any stray .pyc
    $_.Name -match '\.pyc$'
}
if ($bad) { throw "game-derived data leaked into dist: $($bad.FullName -join ', ')" }

# 6. Zip + checksum.
$zip = "dist\$name$(if ($Full) { '-full' }).zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path "$dist\*" -DestinationPath $zip
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash
"$hash  $(Split-Path -Leaf $zip)" | Out-File -Encoding ascii "$zip.sha256"
Write-Host "[release] $zip  SHA256=$hash"
