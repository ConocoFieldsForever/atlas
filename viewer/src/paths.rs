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
        let exe = exe_dir().join("packs");
        if exe.is_dir() {
            return exe;
        }
        let cwd = PathBuf::from("packs");
        if cwd.is_dir() {
            return std::fs::canonicalize(&cwd).unwrap_or(cwd);
        }
        exe
    })
}

pub fn shared_dir() -> PathBuf {
    packs_root().join("shared")
}

pub fn config_path() -> PathBuf {
    let exe = exe_dir().join("atlas.config.json");
    if exe.exists() {
        return exe;
    }
    let cwd = PathBuf::from("atlas.config.json");
    if cwd.exists() {
        return cwd;
    }
    exe
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
