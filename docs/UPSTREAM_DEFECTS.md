# Upstream defects worth reporting

Defects in our DEPENDENCIES, found while building Atlas. Each was verified against the actual
upstream source in `~/.cargo/registry` before being written down — two candidates that came from
memory were checked and **discarded** (see the bottom section), so the ones here are the ones that
survived.

---

## 1. Bevy — `prepare_windows` panics on recoverable surface errors  ⭐ strongest

**Crate**: `bevy_render` 0.17.3 · `src/view/window/mod.rs`, `prepare_windows`
**Related upstream reports**: bevy#13150, bevy#21753 (same panic string, no fix in 0.17.3)

### What upstream does

```rust
#[cfg(target_os = "linux")]
Err(wgpu::SurfaceError::Timeout) if may_erroneously_timeout() => {
    tracing::trace!("Couldn't get swap chain texture. This is probably a quirk of your Linux GPU driver...");
}
Err(err) => {
    panic!("Couldn't get swap chain texture, operation unrecoverable: {err}");
}
```

`Timeout` is tolerated **only** on Linux, and only when `enumerate_adapters` finds an adapter whose
name starts with `Radeon`/`AMD`/`Intel`. Everything else panics.

### Why that is wrong

Three of the four `SurfaceError` variants that reach the fallback arm are recoverable:

| variant | wgpu's own documentation | Bevy 0.17.3 |
|---|---|---|
| `Outdated` | reconfigure the surface | reconfigures ✅ |
| **`Lost`** | **"the swap chain has been lost, recreate it"** | **panics** ❌ |
| **`Timeout`** | acquire took too long; try again next frame | panics off-Linux ❌ |
| **`Other`** | generic/transient failure | panics ❌ |
| `OutOfMemory` | unrecoverable | panics ✅ correct |

`Lost` is the clearest: wgpu documents the remedy, and the arm immediately above already performs
exactly that remedy for `Outdated`. Falling through to `panic!` for `Lost` looks like an oversight
rather than a decision.

### Impact

A Windows app that enters the Win32 **modal move/size loop** on a GPU-bound machine stalls
presentation until the 1 s acquire budget expires, producing `Timeout` — and wgpu-hal's DXGI FIFO
pacing can surface the same stall as a spurious `Other`. Under the common release profile
`panic = "abort"` this is a **silent process death**: no backtrace, no log line, nothing for the user
to report.

**Field-reproduced**: RX 6800, ~85% GPU utilisation, crash on every window move/resize. Diagnosed
only after adding a file-logging layer, because the process left no evidence at all.

### Fix we ship (vendored)

Treat `Timeout | Other` as a skipped frame on every platform — which is what the Linux arm and the
post-reconfigure arm already do — and fold `Lost` into the `Outdated` reconfigure arm. Keep the panic
for `OutOfMemory` only. This also deletes the Linux-only adapter-name allowlist, since the general
case now covers it. ~15 lines; see `third_party/bevy_render/src/view/window/mod.rs`.

Suggested upstream shape: skip the frame and `warn!` (rate-limited — during a sustained window drag
this fires every frame).

---

## 2. Bevy — no device-loss recovery path

**Crate**: `bevy_render` 0.17.3

Once the render device is lost, every subsequent submit fails permanently; there is no way to rebuild
the device inside a running `App`. The only options left to an application are to freeze on a
skip-frame loop or exit. Atlas exits with a coded status and one log line, because both alternatives
strand the user.

Worth filing as a gap rather than a bug — but it is what makes defect #1 fatal instead of merely
noisy, so the two are related.

---

## 3. wgpu — default uncaptured-error handler panics

**Crate**: `wgpu` 26.0.1

`Device::on_uncaptured_error`'s default handler panics. Combined with `panic = "abort"`, any
validation error or device loss at runtime terminates the process with no diagnostic. Every wgpu
application that ships with `panic = "abort"` has to remember to install its own handler purely to
retain error text.

Suggestion: default to logging via `tracing` and continuing for the non-fatal classes, or document
the `panic = "abort"` interaction prominently. Weaker than #1 — this is arguably a deliberate design
choice — but the failure mode (no output whatsoever) is worth raising.

---

## Checked and NOT reportable

Recorded so nobody re-files them.

- **UnityPy `read_typetree()` truncating a partial dict.** Believed for a while that a MeshRenderer
  with no materials came back with the key list stopping at `m_RenderingLayerMask`. It does not —
  that was a `list(d.keys())[:12]` slice in our own probe script. Verified against UnityPy 1.25.0:
  every `MeshRenderer` returns all 27 keys, and a material-less one has `m_Materials: []`. The
  conclusion drawn from it (those renderers genuinely have zero materials) was correct and is
  independently confirmed by the 144-vs-156-byte payload delta.

- **`wgpu_core::handle_error_fatal` bypassing `on_uncaptured_error`.** No such function exists in
  wgpu-core 26.0.1. The note predates this dependency version.
