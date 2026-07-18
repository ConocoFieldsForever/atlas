#!/usr/bin/env pwsh
# Release build with the build machine's absolute paths remapped OUT of the binary.
#
# Rust bakes CARGO_HOME\registry\...\<dep>\src\*.rs and the workspace dir into panic Location
# strings in .rdata (this is how a build previously leaked the builder's Windows username into a
# shipped exe). `trim-paths` would fix it path-independently but is not stabilized in the pinned
# cargo, so we remap those two prefixes to neutral tokens with --remap-path-prefix. The prefixes are
# read from the ENVIRONMENT (CARGO_HOME / USERPROFILE / this repo's path), so NO username is ever
# hardcoded in the repo. RUSTFLAGS overrides .cargo/config.toml entirely, so +crt-static is repeated
# here (keep it in lockstep with .cargo/config.toml).
#
#   .\scripts\build-clean.ps1            # local release build, remapped
#   .\scripts\build-clean.ps1 -Locked    # CI: also pass --locked
param([switch]$Locked)
$ErrorActionPreference = "Stop"

$reg = if ($env:CARGO_HOME) { Join-Path $env:CARGO_HOME "registry" } else { Join-Path $env:USERPROFILE ".cargo\registry" }
$ws  = (Resolve-Path "$PSScriptRoot\..").Path
$env:RUSTFLAGS = "-C target-feature=+crt-static --remap-path-prefix=$reg=/cargoreg --remap-path-prefix=$ws=/atlas"
Write-Host "build-clean: remapping `"$reg`" and `"$ws`" out of the binary"

$cargoArgs = @("build", "--release")
if ($Locked) { $cargoArgs += "--locked" }
& cargo @cargoArgs
exit $LASTEXITCODE
