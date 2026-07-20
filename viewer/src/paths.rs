//! eft::paths — exe-anchored filesystem roots (redistribution PR1).
//!
//! Everything used to be cwd-relative (`packs/…`), which only works when the exe is launched
//! from the workspace root. A shipped bundle is `dist/atlas.exe` + `assets/` + `packs/`
//! side by side, launched from anywhere (Explorer double-click cwd = who knows). All path
//! lookups route through here:
//!   - `exe_dir()`      — directory containing the running exe.
//!   - `packs_root()`   — `<exe>/packs` when it exists, else `<cwd>/packs` (dev `cargo run`
//!                        from the workspace root; exe lives in target/release), else
//!                        `<exe>/packs` (created on demand by writers).
//!   - `shared_dir()`   — `<packs>/shared` (cross-map tier: tarkov.dev catalogs, icons,
//!                        texcache).
//!   - `config_path()`  — `<exe>/atlas.config.json`, falling back to the cwd copy when
//!                        only it exists (pre-portability configs keep working).
//!   - `repo_root()`    — first dir among [cwd, exe_dir] containing `tools/build_map.py`
//!                        (where the python kit lives; the menu build spawns from here).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub fn exe_dir() -> &'static Path {
    static D: OnceLock<PathBuf> = OnceLock::new();
    D.get_or_init(|| {
        std::env::current_exe()
            .ok()
            .and_then(|e| e.parent().map(Path::to_path_buf))
            .unwrap_or_else(|| PathBuf::from("."))
    })
}

pub fn packs_root() -> &'static Path {
    static D: OnceLock<PathBuf> = OnceLock::new();
    D.get_or_init(|| {
        // A `packs` dir only counts as the root if it holds REAL content — a `.eftpack` or the shared
        // intel — NOT merely a `shared/texcache` the BC cache may have created beside the exe. Without
        // this, a texcache-only <exe>/packs HIJACKS the root away from the actual packs dir: the menu's
        // INTEL strip then reads <exe>/packs/shared (empty) and shows "synced never / 0 icons" even
        // though loot.json is present and syncs write it to <cwd>/packs/shared (where the pack loader,
        // which resolves relative to the LOADED pack, correctly finds it). So the two disagreed.
        let has_content = |p: &Path| -> bool {
            p.join("shared").join("loot.json").is_file()
                || p.join("shared").join("tasks.json").is_file()
                || std::fs::read_dir(p)
                    .map(|rd| {
                        rd.flatten()
                            .any(|e| e.path().extension().map_or(false, |x| x == "eftpack"))
                    })
                    .unwrap_or(false)
        };
        let exe = exe_dir().join("packs");
        let cwd = PathBuf::from("packs");
        // Prefer whichever real packs dir has content; exe-beside wins ties (the release layout).
        if has_content(&exe) {
            return exe;
        }
        if has_content(&cwd) {
            return std::fs::canonicalize(&cwd).unwrap_or(cwd);
        }
        // Neither has content yet (fresh install, pre-sync): keep the old preference so first-run
        // writes land somewhere sane — exe-beside if present, else cwd/packs.
        if exe.is_dir() {
            return exe;
        }
        if cwd.is_dir() {
            return std::fs::canonicalize(&cwd).unwrap_or(cwd);
        }
        exe
    })
}

pub fn shared_dir() -> PathBuf {
    packs_root().join("shared")
}

/// `%APPDATA%\atlas\atlas.config.json` — the WRITABLE per-user config location (survives a
/// read-only / Program Files install dir, where a config beside the exe would silently fail to
/// save; finding 12). None when APPDATA is unset (non-Windows dev / stripped env).
pub fn appdata_config() -> Option<PathBuf> {
    std::env::var_os("APPDATA").map(|a| PathBuf::from(a).join("atlas").join("atlas.config.json"))
}

/// Writable, per-user task/key progress. Kept outside packs so changing maps or replacing a pack
/// never loses the user's checklist.
pub fn progress_path() -> PathBuf {
    std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| exe_dir().to_path_buf())
        .join("atlas")
        .join("progress.json")
}

/// Resolved config path. READ order prefers an EXISTING file so pre-portability installs (config
/// beside the exe / in the cwd) keep working; when none exists yet we point WRITES at the
/// user-profile location (`%APPDATA%\atlas`), which is writable even when the install dir is not —
/// so language / game-dir / assets settings actually persist across restarts (finding 12).
pub fn config_path() -> PathBuf {
    let appdata = appdata_config();
    for cand in [
        appdata.clone(),
        Some(exe_dir().join("atlas.config.json")),
        Some(PathBuf::from("atlas.config.json")),
    ]
    .into_iter()
    .flatten()
    {
        if cand.exists() {
            return cand;
        }
    }
    // Nothing on disk yet: write to the user profile (falling back to beside-the-exe only when
    // APPDATA is unavailable).
    appdata.unwrap_or_else(|| exe_dir().join("atlas.config.json"))
}

/// Where the python kit lives (tools/, extraction/, eft_pipeline/): dev = the workspace cwd;
/// shipped bundle = beside the exe. None when neither has the kit (menu greys out BUILD).
pub fn repo_root() -> Option<&'static Path> {
    static D: OnceLock<Option<PathBuf>> = OnceLock::new();
    D.get_or_init(|| {
        for cand in [std::env::current_dir().ok(), Some(exe_dir().to_path_buf())]
            .into_iter()
            .flatten()
        {
            if cand.join("tools").join("build_map.py").is_file() {
                return Some(cand);
            }
        }
        None
    })
    .as_deref()
}

/// Python for the menu build: EFT_PY env > bundled embeddable Python (shipped in the -Full release
/// so a non-dev needs NO system Python) > venv beside the kit (dev / system-python path) > PATH
/// "python". The bundled Python ships with pip but WITHOUT the heavy deps; INSTALL DEPS
/// (tools/setup_deps.py) installs UnityPy/numpy/Pillow into it on first use.
pub fn python_exe(root: &Path) -> PathBuf {
    if let Ok(p) = std::env::var("EFT_PY") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    let bundled = root.join("python").join("python.exe");
    if bundled.is_file() {
        return bundled;
    }
    let venv = root.join("venv").join("Scripts").join("python.exe");
    if venv.is_file() {
        return venv;
    }
    PathBuf::from("python")
}
