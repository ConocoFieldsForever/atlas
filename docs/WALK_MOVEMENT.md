# Walk camera movement provenance

## What the implementation history showed

The original walk camera was added on July 17, 2026 (`5631fcf`, followed by collision and head-bob
commits). It used direct yaw-relative WASD, an immediate Shift multiplier, and a jump whose height
increased with the scroll-wheel speed. Later implementation work concentrated on the drone
simulator and agent link; it did not port EFT's on-foot velocity model.

The current game dump makes a useful partial port possible. The local dump is from
Escape from Tarkov `1.0.6.5-46221`:

- `%USERPROFILE%\EFTDump_1.0.6.5.46221\dump.cs`
- installed `EscapeFromTarkov_Data\resources.assets`, MonoBehaviour path ID `45760`, object
  `eftsettings` (`EFTHardSettings`)

Relevant values recovered from that asset and the IL2CPP type metadata:

| Game setting | Value | Viewer use |
| --- | ---: | --- |
| `MovementAccelerationRange` | `(0, 0, slope 3.3333)` to `(0.7053492, 1)` | Full walk velocity-change interval |
| `DecelerationSpeed` | `1.2` | Normalized stopping/slowing rate |
| `StartingSprintSpeed` | `0.5` | Fraction of selected walk speed required before sprint entry |
| `MovementContext::SPRINT_RESTART_DELAY` | `0.4 s` | Delay after leaving sprint |
| Normal character-controller step offset | `0.25 m` | Recorded for reference; the viewer retains its measured `0.5 m` map step allowance |
| Normal slope limit | `60 degrees` | Already matches the viewer's walkable-face threshold |
| Stand controller height | `1.6 m` | Recorded for reference; the viewer retains its established `1.8 m` capsule / `1.7 m` eye |

The walk camera now keeps a world-space XZ velocity. Input supplies a desired velocity and finite
acceleration/deceleration moves the existing velocity toward it. Reversing must first brake the old
vector; releasing input coasts to a stop; sprint has a run-up and restart delay; and wall collision
removes only inward velocity so tangential momentum survives. A jump preserves takeoff velocity and
permits only limited air steering. The speed wheel controls only normal walking speed. Jump height
is fixed instead of being coupled to the wheel, and head bob uses actual post-collision distance.

This is deliberately a mechanics port, not copied game code. EFT's final movement also depends on
profile/server configuration, carried weight, skills, stamina, injuries, pose, and animation root
motion. Those inputs and the player animation rig do not exist in an `.eftpack`, so absolute sprint
speed, stance transitions, stamina drain, and weight penalties remain viewer-side future work.

## How Atlas was launched and exercised

Development commands consistently used the repository as the working directory so relative
`packs\...` paths and `packs\logs\atlas_viewer.log` resolved correctly.

Development build and a direct Factory launch with the agent link:

```powershell
cargo build -p atlas
$env:EFT_AGENT = '7878'
& .\target\debug\atlas.exe packs\factory_rework.eftpack
```

For a detached interactive launch it used `Start-Process`, commonly minimized while testing:

```powershell
$env:EFT_AGENT = '7878'
Start-Process -FilePath ".\target\debug\atlas.exe" `
  -ArgumentList "packs\factory_rework.eftpack" `
  -WindowStyle Minimized
```

It exercised the same running simulation through the local TCP wrapper, then stopped the test
instance:

```powershell
& .\venv\Scripts\python.exe tools\drone_env.py
Stop-Process -Name atlas -Force
```

For normal app/menu use, the release build was launched with no map argument:

```powershell
cargo build --release
Start-Process -FilePath ".\target\release\atlas.exe" -PassThru
```

Passing a pack bypassed the menu for focused tests:

```powershell
Start-Process -FilePath ".\target\release\atlas.exe" `
  -ArgumentList "packs\interchange.eftpack" `
  -PassThru
```

Automated render smoke tests added `EFT_SHOT`, `EFT_HIDDEN=1`, and `EFT_UNCAPPED=1`, waited for the
screenshot file, and inspected `packs\logs\atlas_viewer.log`. Hidden captures now exit themselves
after the image is written; the wrapper still owns the exact PID as a fallback and restores its
environment in `finally`. For an interactive visible capture, set only `EFT_SHOT`; do not set
`EFT_HIDDEN` or pass PowerShell's `-WindowStyle Hidden`.

One release test initially launched the packaged executable with its `dist` directory as the
working directory, found that repository packs were not visible, and relaunched it with the
repository root as `-WorkingDirectory`; that working-directory detail is important when using an
unpacked release.
