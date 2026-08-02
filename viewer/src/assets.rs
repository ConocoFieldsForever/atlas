//! assets.rs — the ASSETS tab: an inspector for what the map is MADE OF, joined to the geometry.
//!
//! The viewer cannot read Unity bundles (UnityPy is Python), so everything here comes from JSON
//! written by `tools/asset_index.py` into `<pack>/assets/`. Cost decides what is built when:
//!
//!   catalog.json + search.bin   global object index + script/type histograms   once (~2 min)
//!   lv<N>.json                  one level's scene graph                        on demand (~1.5s)
//!   dump_<lv>_<pid>.json        one object's raw typetree                      on select
//!   asset_<file>_<pid>.json     a shared mesh/texture/material, + PNG thumb    on select
//!
//! Nothing is eager beyond the catalog, because the scale forbids it: streets references 238 level
//! bundles holding 4.2M objects, and level233 alone has 203,381. Every build runs on a worker
//! thread and arrives by channel — the panel never blocks a frame on Python.
//!
//! DESIGN. Bundle numbers are storage structure, not meaning: nobody knows what "level214" is, so
//! the old list of 238 of them was navigation into an implementation detail. A level is now a quiet
//! provenance badge on a row. The tab is a contextual inspector with three ways in, in order of how
//! often they are what you want:
//!
//!   1. THE PICKED OBJECT — double-click geometry, land on the exact source GameObject. The join is
//!      exact (`_fold32` of the parent Transform id + level, 100% of streets' 186,723 parented
//!      instances resolve), never a name match. This is the reason the tab exists.
//!   2. SEARCH — one query over every GameObject in the map, filterable by script/component.
//!   3. CATALOG — what kinds of things exist at all ("BallisticCollider 1,410"), each a saved query.
//!
//! Layout is fixed: header (search + context) / browser (results or hierarchy) / inspector. At
//! 430px a tree and an inspector cannot sit side by side, so the inspector is a pane BELOW the
//! browser with a draggable split.
//!
//! HONESTY ABOUT PARTIAL DATA: il2cpp stripping leaves many custom MonoBehaviours with only their
//! four base fields described. `BallisticCollider` (1,410 instances in level421) reads 32 of its 76
//! bytes. The inspector states the shortfall in bytes; it never shows four fields as the object.

use bevy::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Mutex;

/// Rows materialised for the tree in one frame (a single node can have 12,000 children).
const MAX_ROWS: usize = 20_000;
/// Children shown under one parent before a "… N more" row takes over.
const MAX_CHILDREN: usize = 400;
/// Search hits kept. Ranking runs over the whole index; only the best of them are shown.
const MAX_HITS: usize = 400;
const ROW_H: f32 = 17.0;
/// Recompute "near camera" only after the camera has actually moved this far (metres).
const NEAR_REFRESH_M: f32 = 8.0;
/// Radius for the near-camera list.
const NEAR_RADIUS_M: f32 = 120.0;

// ---------------------------------------------------------------------------
// The JSON contract (tools/asset_index.py). Keys are short because level233's node array has
// 203,381 entries and long names cost more than the data.
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize, Default)]
pub struct Catalog {
    #[serde(default)]
    pub count: u64,
    #[serde(rename = "recPad", default)]
    pub rec_pad: u32,
    #[serde(rename = "scriptNames", default)]
    pub script_names: Vec<String>,
    #[serde(rename = "compBits", default)]
    pub comp_bits: Vec<String>,
    /// (script class, count) across the whole map, most common first.
    #[serde(default)]
    pub scripts: Vec<(String, u64)>,
    /// (component type, count) across the whole map.
    #[serde(default)]
    pub components: Vec<(String, u64)>,
}

/// One indexed GameObject, decoded from `search.bin`'s fixed-stride records.
pub struct Entry {
    pub lv: u32,
    pub path_id: i64,
    pub fold: u32,
    pub name: String,
    pub lower: String,
    pub script: u16,
    pub comps: u32,
}

#[derive(serde::Deserialize)]
pub struct Node {
    #[serde(rename = "t")]
    pub ty: String,
    /// GameObject name, or the resolved MonoBehaviour script class.
    #[serde(rename = "n", default)]
    pub name: String,
    #[serde(rename = "p")]
    pub path_id: i64,
    #[serde(rename = "a", default)]
    pub active: u8,
    /// `_fold32` of this GameObject's Transform path_id — the pick join key. 0 on components.
    #[serde(rename = "f", default)]
    pub fold: u32,
    #[serde(rename = "c", default)]
    pub comps: Vec<u32>,
    #[serde(rename = "k", default)]
    pub kids: Vec<u32>,
    #[serde(rename = "sz", default)]
    pub size: u32,
    /// The one fact that makes a component row worth reading ("plywood_board_1_LOD", "3 levels").
    #[serde(rename = "v", default)]
    pub value: String,
    /// PPtr to the shared asset this component points at, for preview.
    #[serde(rename = "r", default)]
    pub asset: Option<AssetRef>,
}

#[derive(serde::Deserialize, Clone, PartialEq)]
pub struct AssetRef {
    /// File the pointer is relative to — `m_FileID` indexes THIS file's externals, not the level's.
    #[serde(rename = "o")]
    pub origin: String,
    #[serde(rename = "f")]
    pub file_id: i64,
    #[serde(rename = "p")]
    pub path_id: i64,
}

#[derive(serde::Deserialize)]
pub struct LevelIndex {
    pub lv: u32,
    #[serde(default)]
    pub counts: HashMap<String, u64>,
    #[serde(default)]
    pub nodes: Vec<Node>,
    #[serde(default)]
    pub roots: Vec<u32>,
    /// Derived on the worker thread (see `finish_level`), never present in the JSON.
    #[serde(skip)]
    pub parent: Vec<i32>,
    #[serde(skip)]
    pub by_fold: HashMap<u32, u32>,
    #[serde(skip)]
    pub by_pid: HashMap<i64, u32>,
}

/// One object's typetree read, plus how much of it the bundle actually described.
#[derive(serde::Deserialize)]
pub struct Dump {
    #[serde(rename = "pathId")]
    pub path_id: i64,
    #[serde(default)]
    pub script: String,
    pub complete: bool,
    #[serde(default)]
    pub size: u32,
    #[serde(default)]
    pub read: Option<u32>,
    #[serde(default)]
    pub undescribed: Option<u32>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub fields: Option<serde_json::Value>,
}

/// A resolved shared asset. `kind` selects which of the optional blocks below are populated.
#[derive(serde::Deserialize, Default)]
pub struct AssetView {
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub name: String,
    #[serde(rename = "type", default)]
    pub ty: String,
    #[serde(rename = "srcFile", default)]
    pub src_file: String,
    #[serde(rename = "pathId", default)]
    pub path_id: i64,
    #[serde(default)]
    pub error: Option<String>,
    // mesh
    #[serde(default)]
    pub tris: u64,
    #[serde(default)]
    pub verts: u64,
    #[serde(default)]
    pub submeshes: u32,
    #[serde(default)]
    pub readable: bool,
    #[serde(default)]
    pub bounds: Option<Bounds>,
    #[serde(default)]
    pub positions: Vec<[f32; 3]>,
    /// One per position — OBJ corners are de-duplicated per (v/vt) pair so seams do not tear.
    #[serde(default)]
    pub uvs: Vec<[f32; 2]>,
    #[serde(default)]
    pub indices: Vec<u32>,
    #[serde(rename = "trisShown", default)]
    pub tris_shown: u64,
    #[serde(rename = "geomError", default)]
    pub geom_error: Option<String>,
    // texture
    #[serde(default)]
    pub w: u32,
    #[serde(default)]
    pub h: u32,
    #[serde(default)]
    pub mips: u32,
    #[serde(default)]
    pub format: String,
    #[serde(default)]
    pub thumb: Option<String>,
    // material
    #[serde(default)]
    pub shader: String,
    #[serde(default)]
    pub slots: Vec<TexSlot>,
    #[serde(default)]
    pub colors: Vec<NamedColor>,
    #[serde(default)]
    pub floats: Vec<NamedFloat>,
    // physic material
    #[serde(rename = "dynamicFriction", default)]
    pub dyn_friction: f32,
    #[serde(rename = "staticFriction", default)]
    pub static_friction: f32,
    #[serde(default)]
    pub bounciness: f32,
    #[serde(rename = "frictionCombine", default)]
    pub friction_combine: String,
    #[serde(rename = "bounceCombine", default)]
    pub bounce_combine: String,
}

/// The base-colour texture of a GameObject, resolved renderer -> material -> _MainTex in one hop
/// so the mesh thumbnail can be skinned with what the object actually looks like.
#[derive(serde::Deserialize, Default)]
pub struct AlbedoView {
    #[serde(default)]
    pub material: String,
    #[serde(default)]
    pub texture: String,
    #[serde(default)]
    pub thumb: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub w: u32,
    #[serde(default)]
    pub h: u32,
    #[serde(default)]
    pub format: String,
}

#[derive(serde::Deserialize, Default)]
pub struct Bounds {
    #[serde(default)]
    pub c: [f32; 3],
    #[serde(default)]
    pub e: [f32; 3],
}

#[derive(serde::Deserialize)]
pub struct TexSlot {
    pub slot: String,
    #[serde(default)]
    pub tex: String,
    pub origin: String,
    #[serde(rename = "fileId")]
    pub file_id: i64,
    #[serde(rename = "pathId")]
    pub path_id: i64,
}

#[derive(serde::Deserialize)]
pub struct NamedColor {
    pub name: String,
    pub rgba: [f32; 4],
}

#[derive(serde::Deserialize)]
pub struct NamedFloat {
    pub name: String,
    pub v: f32,
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// What a background build is doing. `Missing` is a normal first-run state with a button, not a
/// failure — keeping it distinct from `Failed` is why the panel can offer the right next action.
#[derive(Default, PartialEq, Clone)]
pub enum Load {
    #[default]
    Missing,
    Running(String),
    Ready,
    Failed(String),
}

/// Which view the browser region is showing. A breadcrumb names it; there are no permanent tabs.
#[derive(Default, PartialEq, Clone)]
pub enum Mode {
    #[default]
    Landing,
    Search,
    Hierarchy,
}

enum Msg {
    Catalog(Result<Box<(Catalog, Vec<Entry>)>, String>),
    Level(u32, Result<Box<LevelIndex>, String>),
    Dump(Result<Box<Dump>, String>),
    /// Carries the request it answers so an out-of-order reply can be dropped rather than shown
    /// against whatever is selected now.
    Asset(AssetRef, Result<Box<AssetView>, String>),
    /// (level, GameObject path_id) it answers, for the same reason.
    Albedo((u32, i64), Result<Box<AlbedoView>, String>),
}

/// One row in the near-camera list: a source object, its distance, and how many instances it has.
pub struct NearRow {
    pub fold: u32,
    pub lv: u32,
    pub name: String,
    pub dist: f32,
    pub count: u32,
    pub pos: Vec3,
}

#[derive(Resource)]
pub struct AssetBrowser {
    tx: Mutex<Sender<Msg>>,
    rx: Mutex<Receiver<Msg>>,

    pub catalog: Option<Catalog>,
    pub entries: Vec<Entry>,
    pub catalog_state: Load,
    kicked: bool,

    pub mode: Mode,
    pub query: String,
    /// Last query the hit list was computed for. The recompute is gated on THIS, not on
    /// `is_changed()`: egui takes `&mut` on the resource every frame the text box is drawn, so
    /// change detection reports a change whether or not a character was typed.
    last_query: String,
    /// Restrict results to one script class (a catalog click), by index into `catalog.script_names`.
    pub filter_script: Option<u16>,
    /// Restrict results to objects carrying this component bit.
    pub filter_comp: Option<u8>,
    last_filters: (Option<u16>, Option<u8>),
    /// Only show objects that produced geometry in this pack.
    pub only_geometry: bool,
    last_only_geom: bool,
    hits: Vec<u32>,

    pub level: Option<Box<LevelIndex>>,
    pub level_state: Load,
    pending_level: Option<u32>,

    pub expanded: HashSet<u32>,
    pub selected: Option<u32>,

    pub dump: Option<Box<Dump>>,
    pub dump_state: Load,
    dump_text: String,

    pub asset: Option<Box<AssetView>>,
    pub asset_state: Load,
    /// Which component's asset the current `asset` belongs to, so a stale reply is ignored.
    asset_for: Option<AssetRef>,
    /// egui handle for the decoded texture thumbnail, plus the file it came from.
    tex: Option<(String, bevy_egui::egui::TextureHandle)>,
    /// Orbit angles for the mesh thumbnail (radians).
    pub orbit: (f32, f32),
    /// Thumbnail zoom multiplier (1.0 = fit to bounds).
    pub zoom: f32,
    pub wireframe: bool,
    /// Skin the thumbnail with the object's base-colour texture rather than flat shading.
    pub textured: bool,
    /// The GameObject whose albedo is loaded / in flight.
    albedo_for: Option<(u32, i64)>,
    pub albedo: Option<Box<AlbedoView>>,
    pub albedo_state: Load,
    /// Decoded albedo, kept with the file it came from so it is uploaded once.
    albedo_tex: Option<(String, bevy_egui::egui::TextureHandle)>,

    /// A reveal waiting for its level: (lv, par, par2).
    pending_reveal: Option<(u32, u32, u32)>,
    scroll_to: Option<usize>,
    note: String,
    /// Height of the inspector pane (draggable split).
    pub split: f32,

    pub near: Vec<NearRow>,
    near_at: Option<Vec3>,
    /// Show the full near list rather than the first handful. Collapsed by default so the catalog
    /// below it is reachable without scrolling past 60 rows.
    near_all: bool,
}

impl Default for AssetBrowser {
    fn default() -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        Self {
            tx: Mutex::new(tx),
            rx: Mutex::new(rx),
            catalog: None,
            entries: Vec::new(),
            catalog_state: Load::Missing,
            kicked: false,
            mode: Mode::Landing,
            query: String::new(),
            last_query: String::new(),
            filter_script: None,
            filter_comp: None,
            last_filters: (None, None),
            only_geometry: false,
            last_only_geom: false,
            hits: Vec::new(),
            level: None,
            level_state: Load::Missing,
            pending_level: None,
            expanded: HashSet::new(),
            selected: None,
            dump: None,
            dump_state: Load::Missing,
            dump_text: String::new(),
            asset: None,
            asset_state: Load::Missing,
            asset_for: None,
            tex: None,
            orbit: (0.6, -0.4),
            zoom: 1.0,
            wireframe: false,
            textured: true,
            albedo_for: None,
            albedo: None,
            albedo_state: Load::Missing,
            albedo_tex: None,
            pending_reveal: None,
            scroll_to: None,
            note: String::new(),
            // Tall enough that a mesh preview plus its stats and skin line fit without scrolling;
            // draggable from there.
            split: 380.0,
            near: Vec::new(),
            near_at: None,
            near_all: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Running the Python side
// ---------------------------------------------------------------------------

/// Run `tools/asset_index.py <args>` and return the text of the JSON file it wrote (the tool prints
/// that path as its last stdout line). Blocking — worker threads only.
fn run_tool(args: &[String]) -> Result<String, String> {
    let path = tool_path(args)?;
    std::fs::read_to_string(&path).map_err(|e| format!("reading {path}: {e}"))
}

/// As `run_tool`, but returns the PATH the tool wrote rather than its contents (binary sidecars).
fn tool_path(args: &[String]) -> Result<String, String> {
    let root = crate::paths::repo_root().ok_or_else(|| {
        "python kit not found (tools/asset_index.py must sit beside the exe or in the workspace)"
            .to_string()
    })?;
    let py = crate::paths::python_exe(root);
    let mut cmd = std::process::Command::new(&py);
    cmd.arg("tools/asset_index.py").args(args).current_dir(root);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW — no console flash
    }
    let out = cmd
        .output()
        .map_err(|e| format!("could not run {}: {e}", py.display()))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let tail: Vec<&str> = err.lines().rev().take(4).collect();
        return Err(tail.into_iter().rev().collect::<Vec<_>>().join(" / ").trim().to_string());
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    Ok(stdout
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| "the indexer wrote no output path".to_string())?
        .trim()
        .to_string())
}

/// Decode `search.bin` against `search_names.bin`.
///
/// Layout is `REC` in asset_index.py — `struct.Struct("<IqIIHHI")`. The `<` prefix means NO
/// alignment padding, so the i64 sits at byte 4, not at an 8-aligned 8. Fields:
///   lv u32 @0 | pathId i64 @4 | fold u32 @12 | nameOff u32 @16 | nameLen u16 @20 |
///   scriptId u16 @22 | compMask u32 @24        (28 bytes, stride `rec_pad`)
fn decode_entries(bin: &[u8], names: &[u8], rec_pad: usize) -> Vec<Entry> {
    const REC: usize = 28;
    let pad = rec_pad.max(REC);
    let mut out = Vec::with_capacity(bin.len() / pad);
    let g32 = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let g16 = |b: &[u8], o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
    for r in bin.chunks_exact(pad) {
        let lv = g32(r, 0);
        let path_id = i64::from_le_bytes([r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11]]);
        let fold = g32(r, 12);
        let noff = g32(r, 16) as usize;
        let nlen = g16(r, 20) as usize;
        let script = g16(r, 22);
        let comps = g32(r, 24);
        let name = names
            .get(noff..noff.saturating_add(nlen))
            .map(|s| String::from_utf8_lossy(s).into_owned())
            .unwrap_or_default();
        let lower = name.to_lowercase();
        out.push(Entry { lv, path_id, fold, name, lower, script, comps });
    }
    out
}

/// Fill in the links the JSON leaves out — a parent array (so a reveal can expand ancestors) and the
/// fold/pathId maps. Runs on the worker thread; on level233 this touches 203k nodes.
fn finish_level(mut li: LevelIndex) -> LevelIndex {
    li.parent = vec![-1; li.nodes.len()];
    for i in 0..li.nodes.len() {
        let (kids, comps) = {
            let n = &li.nodes[i];
            (n.kids.clone(), n.comps.clone())
        };
        for c in kids.into_iter().chain(comps) {
            if let Some(p) = li.parent.get_mut(c as usize) {
                *p = i as i32;
            }
        }
    }
    li.by_fold = li
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.fold != 0)
        .map(|(i, n)| (n.fold, i as u32))
        .collect();
    li.by_pid = li
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.path_id, i as u32))
        .collect();
    li
}

impl AssetBrowser {
    fn send(&self, f: impl FnOnce(Sender<Msg>) + Send + 'static) {
        let tx = match self.tx.lock() {
            Ok(g) => g.clone(),
            Err(_) => return,
        };
        let _ = std::thread::Builder::new()
            .name("assets-index".into())
            .spawn(move || f(tx));
    }

    fn load_catalog(&mut self, pack: PathBuf, build: bool) {
        self.catalog_state = Load::Running(if build {
            "indexing every level the map uses — a couple of minutes, once".into()
        } else {
            "loading index".into()
        });
        self.send(move |tx| {
            let dir = pack.join("assets");
            let cat_fp = dir.join("catalog.json");
            let res = (|| -> Result<Box<(Catalog, Vec<Entry>)>, String> {
                let text = if !build && cat_fp.is_file() {
                    std::fs::read_to_string(&cat_fp)
                        .map_err(|e| format!("reading {}: {e}", cat_fp.display()))?
                } else {
                    run_tool(&["catalog".into(), pack.to_string_lossy().into_owned()])?
                };
                let cat: Catalog = serde_json::from_str(&text)
                    .map_err(|e| format!("parsing catalog.json: {e}"))?;
                let bin = std::fs::read(dir.join("search.bin")).unwrap_or_default();
                let names = std::fs::read(dir.join("search_names.bin")).unwrap_or_default();
                let entries = decode_entries(&bin, &names, cat.rec_pad as usize);
                Ok(Box::new((cat, entries)))
            })();
            let _ = tx.send(Msg::Catalog(res));
        });
    }

    /// Open one level: read its cached scene graph, or build it. The cache is keyed to the installed
    /// game files, so a game UPDATE wants the rebuild button rather than a stale hit.
    fn load_level(&mut self, pack: PathBuf, lv: u32, build: bool) {
        if self.pending_level == Some(lv) {
            return;
        }
        self.pending_level = Some(lv);
        self.level_state = Load::Running(format!("reading level{lv}"));
        self.send(move |tx| {
            let fp = pack.join("assets").join(format!("lv{lv}.json"));
            let text = if !build && fp.is_file() {
                std::fs::read_to_string(&fp).map_err(|e| format!("reading {}: {e}", fp.display()))
            } else {
                run_tool(&["level".into(), pack.to_string_lossy().into_owned(), lv.to_string()])
            };
            let res = text
                .and_then(|t| {
                    serde_json::from_str::<LevelIndex>(&t)
                        .map_err(|e| format!("parsing lv{lv}.json: {e}"))
                })
                .map(|li| Box::new(finish_level(li)));
            let _ = tx.send(Msg::Level(lv, res));
        });
    }

    fn load_dump(&mut self, pack: PathBuf, lv: u32, path_id: i64) {
        self.dump_state = Load::Running("reading object".into());
        self.dump = None;
        self.dump_text.clear();
        self.send(move |tx| {
            let res = run_tool(&[
                "dump".into(),
                pack.to_string_lossy().into_owned(),
                lv.to_string(),
                path_id.to_string(),
            ])
            .and_then(|t| serde_json::from_str::<Dump>(&t).map_err(|e| format!("parsing the dump: {e}")))
            .map(Box::new);
            let _ = tx.send(Msg::Dump(res));
        });
    }

    fn load_asset(&mut self, pack: PathBuf, r: AssetRef) {
        if self.asset_for.as_ref() == Some(&r) && self.asset_state != Load::Missing {
            return;
        }
        self.asset_state = Load::Running("resolving asset".into());
        self.asset = None;
        self.asset_for = Some(r.clone());
        self.tex = None;
        self.send(move |tx| {
            let res = run_tool(&[
                "asset".into(),
                pack.to_string_lossy().into_owned(),
                r.origin.clone(),
                r.file_id.to_string(),
                r.path_id.to_string(),
            ])
            .and_then(|t| {
                serde_json::from_str::<AssetView>(&t).map_err(|e| format!("parsing the asset: {e}"))
            })
            .map(Box::new);
            let _ = tx.send(Msg::Asset(r, res));
        });
    }

    /// Resolve one GameObject's base-colour texture so the mesh thumbnail can be skinned with it.
    fn load_albedo(&mut self, pack: PathBuf, lv: u32, go_pid: i64) {
        if self.albedo_for == Some((lv, go_pid)) {
            return;
        }
        self.albedo_for = Some((lv, go_pid));
        self.albedo = None;
        self.albedo_tex = None;
        self.albedo_state = Load::Running("resolving texture".into());
        self.send(move |tx| {
            let res = run_tool(&[
                "albedo".into(),
                pack.to_string_lossy().into_owned(),
                lv.to_string(),
                go_pid.to_string(),
            ])
            .and_then(|t| {
                serde_json::from_str::<AlbedoView>(&t).map_err(|e| format!("parsing albedo: {e}"))
            })
            .map(Box::new);
            let _ = tx.send(Msg::Albedo((lv, go_pid), res));
        });
    }

    /// Select `idx`, open every ancestor so it is visible, and ask the tree to scroll to it.
    fn reveal_node(&mut self, idx: u32) {
        self.mode = Mode::Hierarchy;
        self.selected = Some(idx);
        self.dump = None;
        self.dump_state = Load::Missing;
        self.asset = None;
        self.asset_state = Load::Missing;
        self.asset_for = None;
        self.tex = None;
        if let Some(li) = self.level.as_ref() {
            let mut p = li.parent.get(idx as usize).copied().unwrap_or(-1);
            let mut guard = 0;
            while p >= 0 && guard < 512 {
                self.expanded.insert(p as u32);
                p = li.parent.get(p as usize).copied().unwrap_or(-1);
                guard += 1;
            }
        }
        self.scroll_to = None; // recomputed against the fresh row list in the panel
    }

    /// Jump to a global search entry: load its level if needed, then reveal it by path_id.
    fn open_entry(&mut self, pack: &Path, lv: u32, path_id: i64, fold: u32) {
        let here = self
            .level
            .as_ref()
            .filter(|li| li.lv == lv)
            .and_then(|li| li.by_pid.get(&path_id).copied());
        match here {
            Some(i) => self.reveal_node(i),
            None => {
                self.pending_reveal = Some((lv, fold, 0));
                self.note.clear();
                self.load_level(pack.to_path_buf(), lv, false);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

fn poll_assets(mut ab: ResMut<AssetBrowser>) {
    let msgs: Vec<Msg> = match ab.rx.lock() {
        Ok(rx) => rx.try_iter().collect(),
        Err(_) => return,
    };
    for m in msgs {
        match m {
            Msg::Catalog(Ok(b)) => {
                let (cat, entries) = *b;
                ab.catalog = Some(cat);
                ab.entries = entries;
                ab.catalog_state = Load::Ready;
                // The near list is NAMED from the index, and it is normally computed before the
                // index finishes loading (the camera is still). Without this it would sit there
                // reading "level302 · object" until the user happened to fly 8 m.
                ab.near_at = None;
            }
            Msg::Catalog(Err(e)) => ab.catalog_state = Load::Failed(e),
            Msg::Level(lv, res) => {
                // Ignore a reply the user has already navigated away from. Opening level A then B
                // before A finished would otherwise install A over B AND consume B's pending
                // reveal (the lv check inside would fail), so B would arrive with nothing revealed.
                if ab.pending_level.map(|p| p != lv).unwrap_or(false) {
                    continue;
                }
                ab.pending_level = None;
                match res {
                    Ok(li) => {
                        ab.level = Some(li);
                        ab.level_state = Load::Ready;
                        ab.expanded.clear();
                        ab.selected = None;
                        // A reveal that was waiting on this level can now resolve. `fold` first
                        // (the pick's parent transform), then the grandparent as a fallback.
                        if let Some((rlv, a, b)) = ab.pending_reveal.take() {
                            if rlv == lv {
                                let found = ab.level.as_ref().and_then(|li| {
                                    li.by_fold.get(&a).or_else(|| li.by_fold.get(&b)).copied()
                                });
                                match found {
                                    Some(i) => ab.reveal_node(i),
                                    None => {
                                        ab.mode = Mode::Hierarchy;
                                        ab.note = format!(
                                            "level{lv} holds no object with that transform id — \
                                             the pack instance may predate the ancestry capture"
                                        );
                                    }
                                }
                            }
                        } else {
                            ab.mode = Mode::Hierarchy;
                        }
                    }
                    Err(e) => ab.level_state = Load::Failed(e),
                }
            }
            Msg::Dump(Ok(d)) => {
                ab.dump_text = d
                    .fields
                    .as_ref()
                    .map(|f| serde_json::to_string_pretty(f).unwrap_or_else(|_| "<unprintable>".into()))
                    .unwrap_or_default();
                ab.dump = Some(d);
                ab.dump_state = Load::Ready;
            }
            Msg::Dump(Err(e)) => ab.dump_state = Load::Failed(e),
            // Drop a reply for a request the inspector has moved on from (selection changed, or a
            // texture slot was followed) — subprocesses can finish out of order.
            Msg::Asset(r, _) if ab.asset_for.as_ref() != Some(&r) => {}
            Msg::Asset(_, Ok(a)) => {
                ab.asset = Some(a);
                ab.asset_state = Load::Ready;
            }
            Msg::Asset(_, Err(e)) => ab.asset_state = Load::Failed(e),
            Msg::Albedo(k, _) if ab.albedo_for != Some(k) => {}
            Msg::Albedo(_, Ok(a)) => {
                ab.albedo = Some(a);
                ab.albedo_state = Load::Ready;
            }
            Msg::Albedo(_, Err(e)) => ab.albedo_state = Load::Failed(e),
        }
    }
}

/// Rebuild the near-camera list, but only after the camera has actually moved — this walks every
/// pack instance, which is 186,724 of them on streets and has no business running per frame.
fn update_near(
    mut ab: ResMut<AssetBrowser>,
    tab: Res<crate::ui::RightPanelTab>,
    pack: Option<Res<crate::render::LoadedPack>>,
    cams: Query<&GlobalTransform, With<crate::render::CullCamera>>,
) {
    if *tab != crate::ui::RightPanelTab::Assets {
        return;
    }
    let (Some(pack), Ok(cam)) = (pack, cams.single()) else {
        return;
    };
    let eye = cam.translation();
    if ab
        .near_at
        .map(|p| p.distance(eye) < NEAR_REFRESH_M)
        .unwrap_or(false)
    {
        return;
    }
    ab.near_at = Some(eye);
    // Group by SOURCE OBJECT (the fold), not by instance: twelve copies of one railing are one
    // answer to "what am I looking at", not twelve.
    let mut by_fold: HashMap<(u32, u32), (f32, u32, Vec3)> = HashMap::new();
    for i in &pack.0.instances {
        if i.par == 0 {
            continue;
        }
        let p = Vec3::new(i.affine[3], i.affine[7], i.affine[11]);
        let d = p.distance(eye);
        if d > NEAR_RADIUS_M {
            continue;
        }
        let e = by_fold.entry((i.lv, i.par)).or_insert((d, 0, p));
        e.1 += 1;
        if d < e.0 {
            e.0 = d;
            e.2 = p;
        }
    }
    let named: HashMap<(u32, u32), &str> = HashMap::new();
    let _ = named;
    let mut rows: Vec<NearRow> = by_fold
        .into_iter()
        .map(|((lv, fold), (dist, count, pos))| NearRow {
            fold,
            lv,
            name: String::new(),
            dist,
            count,
            pos,
        })
        .collect();
    rows.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
    rows.truncate(60);
    // Name them from the global index when it is loaded (the index is keyed by the same fold).
    if !ab.entries.is_empty() {
        let want: HashSet<(u32, u32)> = rows.iter().map(|r| (r.lv, r.fold)).collect();
        let mut found: HashMap<(u32, u32), String> = HashMap::new();
        for e in &ab.entries {
            if e.fold != 0 && want.contains(&(e.lv, e.fold)) {
                found.entry((e.lv, e.fold)).or_insert_with(|| e.name.clone());
            }
        }
        for r in rows.iter_mut() {
            if let Some(n) = found.get(&(r.lv, r.fold)) {
                r.name = n.clone();
            }
        }
    }
    ab.near = rows;
}

/// Map swap: the whole browse state belongs to the OLD pack (node indices, folds and levels are all
/// per-map), so drop it rather than let it describe the wrong map.
fn teardown_assets(mut ab: ResMut<AssetBrowser>) {
    *ab = AssetBrowser::default();
}

/// One visible tree row. `more` > 0 marks the "… N more" tail (and then `node` is meaningless).
struct Row {
    node: u32,
    depth: u8,
    more: u32,
}

/// Walk stack entry. `More` is its own variant rather than a re-pushed node id — pushing the parent
/// again as a placeholder would re-enter its expanded branch and spin until the row cap.
enum Item {
    Node(u32, u8),
    More(u32, u8),
}

/// Flatten the expanded tree into the rows to draw. Only expanded branches are walked, and a single
/// parent contributes at most `MAX_CHILDREN` rows.
fn build_rows(li: &LevelIndex, expanded: &HashSet<u32>, rows: &mut Vec<Row>) {
    rows.clear();
    let mut stack: Vec<Item> = li.roots.iter().rev().map(|&r| Item::Node(r, 0)).collect();
    while let Some(item) = stack.pop() {
        if rows.len() >= MAX_ROWS {
            break;
        }
        match item {
            Item::More(count, depth) => rows.push(Row { node: 0, depth, more: count }),
            Item::Node(idx, depth) => {
                rows.push(Row { node: idx, depth, more: 0 });
                if !expanded.contains(&idx) {
                    continue;
                }
                let Some(n) = li.nodes.get(idx as usize) else {
                    continue;
                };
                let hidden = (n.kids.len().saturating_sub(MAX_CHILDREN)
                    + n.comps.len().saturating_sub(MAX_CHILDREN)) as u32;
                // LIFO: push in REVERSE of the on-screen order — components first (they say what
                // the object IS), then child GameObjects, then the overflow tail.
                if hidden > 0 {
                    stack.push(Item::More(hidden, depth + 1));
                }
                for &k in n.kids.iter().take(MAX_CHILDREN).rev() {
                    stack.push(Item::Node(k, depth + 1));
                }
                for &c in n.comps.iter().take(MAX_CHILDREN).rev() {
                    stack.push(Item::Node(c, depth + 1));
                }
            }
        }
    }
}

pub struct AssetsPlugin;

impl Plugin for AssetsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<AssetBrowser>()
            .add_systems(Update, (poll_assets, update_near))
            .add_systems(
                Update,
                teardown_assets.run_if(resource_changed::<crate::render::MapEpoch>),
            );
        // EguiPrimaryContextPass, NOT Update: a panel drawn from Update still renders, but egui
        // never registers the area, so `is_pointer_over_area()` stays false and every click falls
        // through the panel into the scene pick raycast.
        #[cfg(feature = "egui")]
        app.add_systems(bevy_egui::EguiPrimaryContextPass, assets_panel);
    }
}

/// Everything the panel needs from the pack, gathered once per frame.
#[cfg(feature = "egui")]
struct Ctx<'a> {
    pack_root: PathBuf,
    pack: Option<&'a crate::eftpack::Pack>,
}

#[cfg(feature = "egui")]
#[allow(clippy::too_many_arguments)]
fn assets_panel(
    mut contexts: bevy_egui::EguiContexts,
    tab: Res<crate::ui::RightPanelTab>,
    menu: Option<Res<crate::menu::MenuState>>,
    mut ab: ResMut<AssetBrowser>,
    pack: Option<Res<crate::render::LoadedPack>>,
    last_pick: Res<crate::pick::LastPick>,
    mut cam_cmd: ResMut<crate::CameraCommand>,
) {
    use crate::ui_theme as theme;
    use bevy_egui::egui::{self, RichText};
    if menu.is_some() || *tab != crate::ui::RightPanelTab::Assets {
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else { return };
    let Some(pack_res) = pack.as_ref() else {
        egui::SidePanel::right("assets_panel")
            .default_width(470.0)
            .frame(theme::panel_frame())
            .show(ctx, |ui| {
                ui.label(theme::title("ASSETS"));
                ui.label(RichText::new("no map loaded").size(11.0).color(DIM));
            });
        return;
    };
    let c = Ctx {
        pack_root: pack_res.0.root.clone(),
        pack: Some(&pack_res.0),
    };
    let ctx2 = ctx.clone();

    egui::SidePanel::right("assets_panel")
        .default_width(470.0)
        .min_width(380.0)
        .frame(theme::panel_frame())
        .show(ctx, |ui| {
            // ---- header: identity + search ------------------------------------------
            ui.horizontal(|ui| {
                ui.label(theme::title("ASSETS"));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new(breadcrumb(&ab)).size(9.5).color(DIM));
                });
            });

            if !ab.kicked {
                ab.kicked = true;
                if c.pack_root.join("assets").join("catalog.json").is_file() {
                    ab.load_catalog(c.pack_root.clone(), false);
                }
            }

            match ab.catalog_state.clone() {
                Load::Running(what) => {
                    ui.add_space(theme::SP_MD);
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(RichText::new(what).size(10.5).color(DIM));
                    });
                    return;
                }
                Load::Missing => {
                    ui.add_space(theme::SP_MD);
                    ui.label(
                        RichText::new(
                            "This map has no asset index yet. Building one reads every level \
bundle the map uses and records what is inside — a couple of minutes, once.",
                        )
                        .size(10.5)
                        .color(DIM),
                    );
                    ui.add_space(theme::SP_SM);
                    if ui.button("build asset index").clicked() {
                        ab.load_catalog(c.pack_root.clone(), true);
                    }
                    return;
                }
                Load::Failed(e) => {
                    ui.add_space(theme::SP_MD);
                    ui.label(RichText::new("index unavailable").size(11.0).color(WARN));
                    ui.label(RichText::new(e).size(10.0).color(DIM));
                    ui.add_space(theme::SP_SM);
                    if ui.button("retry").clicked() {
                        ab.load_catalog(c.pack_root.clone(), true);
                    }
                    return;
                }
                Load::Ready => {}
            }

            draw_search_bar(ui, &mut ab, &c);
            draw_context_card(ui, &mut ab, &c, &last_pick, &mut cam_cmd);
            ui.separator();

            // ---- browser region ------------------------------------------------------
            let inspector_h = ab.split.clamp(0.0, (ui.available_height() - 90.0).max(0.0));
            let browser_h = (ui.available_height() - inspector_h - 8.0).max(60.0);
            ui.allocate_ui(egui::vec2(ui.available_width(), browser_h), |ui| {
                match ab.mode.clone() {
                    Mode::Landing => draw_landing(ui, &mut ab, &c, &mut cam_cmd),
                    Mode::Search => draw_results(ui, &mut ab, &c),
                    Mode::Hierarchy => draw_hierarchy(ui, &mut ab, &c),
                }
            });

            // ---- draggable split -----------------------------------------------------
            let (bar, resp) = ui.allocate_exact_size(
                egui::vec2(ui.available_width(), 6.0),
                egui::Sense::drag(),
            );
            let hot = resp.hovered() || resp.dragged();
            ui.painter().rect_filled(
                egui::Rect::from_center_size(bar.center(), egui::vec2(bar.width(), 2.0)),
                1.0,
                if hot { theme::BEIGE } else { theme::CARD_HOVER },
            );
            if resp.dragged() {
                ab.split = (ab.split - resp.drag_delta().y).clamp(0.0, 900.0);
            }
            if resp.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
            }

            draw_inspector(ui, &mut ab, &c, &ctx2, &mut cam_cmd);
        });
}

#[cfg(feature = "egui")]
const DIM: bevy_egui::egui::Color32 = crate::ui_theme::MUTED;
#[cfg(feature = "egui")]
const WARN: bevy_egui::egui::Color32 = bevy_egui::egui::Color32::from_rgb(226, 154, 94);

#[cfg(feature = "egui")]
fn breadcrumb(ab: &AssetBrowser) -> String {
    match ab.mode {
        Mode::Landing => "overview".into(),
        Mode::Search => format!("search \u{00B7} {} hits", ab.hits.len()),
        Mode::Hierarchy => ab
            .level
            .as_ref()
            .map(|l| format!("level{}", l.lv))
            .unwrap_or_default(),
    }
}

/// Search box + the active filter chips. Typing anything switches the browser to results.
#[cfg(feature = "egui")]
fn draw_search_bar(ui: &mut bevy_egui::egui::Ui, ab: &mut AssetBrowser, c: &Ctx) {
    use crate::ui_theme as theme;
    use bevy_egui::egui;
    ui.add_space(theme::SP_SM);
    let n = ab.catalog.as_ref().map(|c| c.count).unwrap_or(0);
    ui.add(
        egui::TextEdit::singleline(&mut ab.query)
            .desired_width(f32::INFINITY)
            .hint_text(format!("search {} objects", thousands(n))),
    );

    // Chips for the active filters, each clickable to clear.
    let script_name = ab
        .filter_script
        .and_then(|s| ab.catalog.as_ref().and_then(|c| c.script_names.get(s as usize)).cloned());
    let comp_name = ab
        .filter_comp
        .and_then(|b| ab.catalog.as_ref().and_then(|c| c.comp_bits.get(b as usize)).cloned());
    if script_name.is_some() || comp_name.is_some() || ab.only_geometry {
        ui.horizontal_wrapped(|ui| {
            if let Some(n) = script_name {
                if ui.small_button(format!("script: {n}  \u{00d7}")).clicked() {
                    ab.filter_script = None;
                }
            }
            if let Some(n) = comp_name {
                if ui.small_button(format!("has: {n}  \u{00d7}")).clicked() {
                    ab.filter_comp = None;
                }
            }
            if ab.only_geometry && ui.small_button("in this pack  \u{00d7}").clicked() {
                ab.only_geometry = false;
            }
        });
    }

    // Recompute hits ONLY when the query or filters actually changed. Gating on `is_changed()`
    // would re-rank 1.5M entries every frame the panel is open, because egui takes &mut on the
    // resource just to draw the text box.
    let filters = (ab.filter_script, ab.filter_comp);
    if ab.query != ab.last_query
        || filters != ab.last_filters
        || ab.only_geometry != ab.last_only_geom
    {
        ab.last_query = ab.query.clone();
        ab.last_filters = filters;
        ab.last_only_geom = ab.only_geometry;
        recompute_hits(ab, c);
        let active = !ab.query.trim().is_empty() || filters.0.is_some() || filters.1.is_some();
        ab.mode = if active { Mode::Search } else { Mode::Landing };
    }
    ui.add_space(theme::SP_SM);
}

/// Rank the global index against the query + filters. Scored so exact and prefix matches beat
/// incidental substring hits — with 1.5M objects an unranked `contains` is unusable.
#[cfg(feature = "egui")]
fn recompute_hits(ab: &mut AssetBrowser, c: &Ctx) {
    ab.hits.clear();
    let q = ab.query.trim().to_lowercase();
    let has_query = !q.is_empty();
    if !has_query && ab.filter_script.is_none() && ab.filter_comp.is_none() {
        return;
    }
    // Folds that actually produced geometry in this pack, for the "in this pack" filter.
    let geom: Option<HashSet<(u32, u32)>> = if ab.only_geometry {
        c.pack.map(|p| p.instances.iter().map(|i| (i.lv, i.par)).collect())
    } else {
        None
    };
    // The index holds one record per (object, script), so an object with two scripts would appear
    // twice in a plain text query. A script filter already selects a single record per object, so
    // the de-dup is only needed when no script filter is active.
    let mut seen: HashSet<(u32, i64)> = HashSet::new();
    let dedup = ab.filter_script.is_none();
    let mut scored: Vec<(u32, u32)> = Vec::new(); // (score, index)
    for (i, e) in ab.entries.iter().enumerate() {
        if let Some(s) = ab.filter_script {
            if e.script != s {
                continue;
            }
        }
        if let Some(b) = ab.filter_comp {
            if e.comps & (1u32 << b) == 0 {
                continue;
            }
        }
        if let Some(g) = geom.as_ref() {
            if e.fold == 0 || !g.contains(&(e.lv, e.fold)) {
                continue;
            }
        }
        let score = if has_query {
            match e.lower.find(&q) {
                None => continue,
                Some(0) if e.lower.len() == q.len() => 0,
                Some(0) => 1,
                Some(_) => 2,
            }
        } else {
            1
        };
        if dedup && !seen.insert((e.lv, e.path_id)) {
            continue;
        }
        scored.push((score, i as u32));
        if scored.len() > MAX_HITS * 40 {
            break; // enough to rank from; a query this broad is being refined anyway
        }
    }
    scored.sort_by_key(|&(s, i)| (s, ab.entries[i as usize].name.len(), i));
    ab.hits = scored.into_iter().take(MAX_HITS).map(|(_, i)| i).collect();
}

/// The picked-geometry card — the tab's primary context. Always the first thing under the search.
#[cfg(feature = "egui")]
fn draw_context_card(
    ui: &mut bevy_egui::egui::Ui,
    ab: &mut AssetBrowser,
    c: &Ctx,
    last_pick: &crate::pick::LastPick,
    cam_cmd: &mut crate::CameraCommand,
) {
    use crate::ui_theme as theme;
    use bevy_egui::egui::{self, RichText};
    let Some(p) = last_pick.0.clone() else {
        ui.label(
            RichText::new("double-click geometry in the scene to inspect its source")
                .size(10.0)
                .color(DIM),
        );
        return;
    };
    egui::Frame::NONE
        .fill(theme::CARD)
        .inner_margin(6.0)
        .show(ui, |ui| {
            ui.label(RichText::new("PICKED").size(9.5).color(DIM));
            ui.label(RichText::new(&p.mesh).size(11.5).color(theme::TEXT_BRIGHT));
            if !p.root.is_empty() {
                ui.label(RichText::new(&p.root).size(9.5).color(DIM));
            }
            ui.horizontal(|ui| {
                let can = p.par != 0 || p.par2 != 0;
                if ui
                    .add_enabled(can, egui::Button::new("reveal source"))
                    .on_hover_text(
                        "open the exact GameObject this instance came from — matched on the folded \
transform id the pack ships, not on the name",
                    )
                    .clicked()
                {
                    ab.note.clear();
                    let here = ab
                        .level
                        .as_ref()
                        .filter(|li| li.lv == p.lv)
                        .and_then(|li| {
                            li.by_fold.get(&p.par).or_else(|| li.by_fold.get(&p.par2)).copied()
                        });
                    match here {
                        Some(i) => ab.reveal_node(i),
                        None => {
                            ab.pending_reveal = Some((p.lv, p.par, p.par2));
                            ab.load_level(c.pack_root.clone(), p.lv, false);
                        }
                    }
                }
                if ui.button("fly here").clicked() {
                    cam_cmd.fly_to = Some(p.world);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new(format!("level{}", p.lv)).size(9.0).color(DIM));
                });
            });
        });
    if !ab.note.is_empty() {
        let n = ab.note.clone();
        ui.label(RichText::new(n).size(10.0).color(WARN));
    }
}

/// At rest: what is around the camera, then the catalog of what exists at all.
#[cfg(feature = "egui")]
fn draw_landing(
    ui: &mut bevy_egui::egui::Ui,
    ab: &mut AssetBrowser,
    c: &Ctx,
    cam_cmd: &mut crate::CameraCommand,
) {
    use crate::ui_theme as theme;
    use bevy_egui::egui::{self, RichText};
    let mut open: Option<(u32, u32)> = None; // (lv, fold)
    let mut pick_script: Option<u16> = None;
    let mut pick_comp: Option<u8> = None;
    let mut toggle_near = false;

    egui::ScrollArea::vertical()
        .id_salt("assets_landing")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.label(RichText::new("AROUND THE CAMERA").size(9.5).color(DIM));
            if ab.near.is_empty() {
                ui.label(
                    RichText::new("nothing within 120 m — fly closer to the map")
                        .size(10.0)
                        .color(DIM),
                );
            }
            let show = if ab.near_all { ab.near.len() } else { 8 };
            for r in ab.near.iter().take(show) {
                ui.horizontal(|ui| {
                    let label = if r.name.is_empty() {
                        format!("level{} \u{00B7} object", r.lv)
                    } else {
                        r.name.clone()
                    };
                    if ui
                        .add(egui::Button::new(RichText::new(label).size(10.5)).frame(false))
                        .on_hover_text("open this object in the hierarchy")
                        .clicked()
                    {
                        open = Some((r.lv, r.fold));
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add(egui::Button::new(RichText::new("fly").size(9.0)).frame(false))
                            .on_hover_text("move the camera to the nearest instance")
                            .clicked()
                        {
                            cam_cmd.fly_to = Some(r.pos);
                        }
                        ui.label(
                            RichText::new(format!("{:.0} m", r.dist)).size(9.5).color(DIM),
                        );
                        if r.count > 1 {
                            ui.label(RichText::new(format!("\u{00d7}{}", r.count)).size(9.0).color(DIM));
                        }
                    });
                });
            }
            if ab.near.len() > 8 {
                let label = if ab.near_all {
                    "show fewer".to_string()
                } else {
                    format!("show all {} nearby", ab.near.len())
                };
                if ui.small_button(label).clicked() {
                    toggle_near = true;
                }
            }

            ui.add_space(theme::SP_MD);
            ui.separator();
            ui.label(RichText::new("SCRIPTS IN THIS MAP").size(9.5).color(DIM));
            ui.label(
                RichText::new("the game's own behaviours — click one to list every object with it")
                    .size(9.0)
                    .color(DIM),
            );
            if let Some(cat) = ab.catalog.as_ref() {
                for (name, n) in cat.scripts.iter().take(40) {
                    let id = cat.script_names.iter().position(|s| s == name);
                    ui.horizontal(|ui| {
                        if ui
                            .add(egui::Button::new(RichText::new(name).size(10.5)).frame(false))
                            .clicked()
                        {
                            if let Some(i) = id {
                                pick_script = Some(i as u16);
                            }
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(RichText::new(thousands(*n)).size(9.5).color(DIM));
                        });
                    });
                }
                ui.add_space(theme::SP_MD);
                ui.separator();
                ui.label(RichText::new("COMPONENT TYPES").size(9.5).color(DIM));
                for (name, n) in cat.components.iter().take(24) {
                    let bit = cat.comp_bits.iter().position(|b| b == name);
                    ui.horizontal(|ui| {
                        let btn = egui::Button::new(RichText::new(name).size(10.5)).frame(false);
                        if ui.add_enabled(bit.is_some(), btn).clicked() {
                            if let Some(b) = bit {
                                pick_comp = Some(b as u8);
                            }
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.label(RichText::new(thousands(*n)).size(9.5).color(DIM));
                        });
                    });
                }
            }
        });

    if let Some((lv, fold)) = open {
        let pid = ab
            .entries
            .iter()
            .find(|e| e.lv == lv && e.fold == fold)
            .map(|e| e.path_id);
        match pid {
            Some(p) => ab.open_entry(&c.pack_root, lv, p, fold),
            None => {
                ab.pending_reveal = Some((lv, fold, 0));
                ab.load_level(c.pack_root.clone(), lv, false);
            }
        }
    }
    if toggle_near {
        ab.near_all = !ab.near_all;
    }
    if let Some(s) = pick_script {
        ab.filter_script = Some(s);
        ab.filter_comp = None;
    }
    if let Some(b) = pick_comp {
        ab.filter_comp = Some(b);
        ab.filter_script = None;
    }
}

/// Global search results. Rows carry name, what it is, geometry count and a quiet level badge.
#[cfg(feature = "egui")]
fn draw_results(ui: &mut bevy_egui::egui::Ui, ab: &mut AssetBrowser, c: &Ctx) {
    use bevy_egui::egui::{self, RichText};
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(if ab.hits.len() >= MAX_HITS {
                format!("best {MAX_HITS} matches")
            } else {
                format!("{} match(es)", ab.hits.len())
            })
            .size(10.0)
            .color(DIM),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let mut og = ab.only_geometry;
            if ui
                .checkbox(&mut og, RichText::new("in this pack").size(10.0))
                .on_hover_text("only objects that produced geometry in the loaded map")
                .changed()
            {
                ab.only_geometry = og;
            }
        });
    });
    if ab.hits.is_empty() {
        ui.label(RichText::new("nothing matched").size(10.0).color(DIM));
        return;
    }
    // Instance counts per (lv, fold) so a row can say "x12" without a scan per row.
    let mut open: Option<u32> = None;
    let hits = ab.hits.clone();
    egui::ScrollArea::vertical()
        .id_salt("assets_results")
        .auto_shrink([false, false])
        .show_rows(ui, ROW_H * 2.2, hits.len(), |ui, range| {
            for r in range {
                let Some(e) = ab.entries.get(hits[r] as usize) else {
                    continue;
                };
                let ninst = c
                    .pack
                    .map(|p| {
                        p.instances
                            .iter()
                            .filter(|i| i.lv == e.lv && (i.par == e.fold || i.par2 == e.fold))
                            .count()
                    })
                    .unwrap_or(0);
                let script = ab
                    .catalog
                    .as_ref()
                    .and_then(|c| c.script_names.get(e.script as usize))
                    .cloned()
                    .unwrap_or_default();
                // The whole two-line row is ONE click target, allocated with `Sense::click()` before
                // anything is drawn into it. Nested layouts wrapped in `scope(..).interact(..)` do
                // not reliably receive clicks inside a virtualised ScrollArea — the row simply did
                // not respond — so the rect is claimed up front and painted by hand.
                let name = if e.name.is_empty() { "(unnamed)" } else { e.name.as_str() };
                let mut bits: Vec<String> = Vec::new();
                if !script.is_empty() {
                    bits.push(script.clone());
                }
                if ninst > 0 {
                    bits.push(format!("\u{00d7}{ninst} in map"));
                }
                let sub = bits.join("  \u{00B7}  ");
                let badge = format!("level{}", e.lv);
                let (rect, resp) = ui.allocate_exact_size(
                    egui::vec2(ui.available_width(), ROW_H * 2.2),
                    egui::Sense::click(),
                );
                if resp.hovered() {
                    ui.painter().rect_filled(rect, 2.0, crate::ui_theme::CARD_HOVER);
                }
                let p = ui.painter();
                p.text(
                    rect.left_top() + egui::vec2(3.0, 3.0),
                    egui::Align2::LEFT_TOP,
                    name,
                    egui::FontId::proportional(11.0),
                    crate::ui_theme::TEXT_BRIGHT,
                );
                p.text(
                    rect.left_bottom() + egui::vec2(3.0, -4.0),
                    egui::Align2::LEFT_BOTTOM,
                    sub,
                    egui::FontId::proportional(9.0),
                    DIM,
                );
                p.text(
                    rect.right_bottom() + egui::vec2(-4.0, -4.0),
                    egui::Align2::RIGHT_BOTTOM,
                    badge,
                    egui::FontId::proportional(8.5),
                    DIM,
                );
                if resp.clicked() {
                    open = Some(hits[r]);
                }
            }
        });
    if let Some(i) = open {
        if let Some(e) = ab.entries.get(i as usize) {
            let (lv, pid, fold) = (e.lv, e.path_id, e.fold);
            ab.open_entry(&c.pack_root, lv, pid, fold);
        }
    }
}

/// A disclosure triangle: right when collapsed, down when expanded. PAINTED, not typed.
///
/// The UI font is bahnschrift, which has no U+25B8/U+25BE, and egui's fallbacks do not carry them
/// either — a literal ▸/▾ rendered as a tofu box, so collapsed and expanded rows looked identical.
/// egui's own `CollapsingHeader` paints its arrow for the same reason. Returns true when clicked.
#[cfg(feature = "egui")]
fn disclosure(ui: &mut bevy_egui::egui::Ui, open: bool) -> bool {
    use bevy_egui::egui;
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(12.0, ROW_H), egui::Sense::click());
    let col = if resp.hovered() { crate::ui_theme::BEIGE } else { DIM };
    let ctr = rect.center();
    let r = 3.4;
    let pts = if open {
        vec![
            egui::pos2(ctr.x - r, ctr.y - r * 0.55),
            egui::pos2(ctr.x + r, ctr.y - r * 0.55),
            egui::pos2(ctr.x, ctr.y + r * 0.85),
        ]
    } else {
        vec![
            egui::pos2(ctr.x - r * 0.55, ctr.y - r),
            egui::pos2(ctr.x - r * 0.55, ctr.y + r),
            egui::pos2(ctr.x + r * 0.85, ctr.y),
        ]
    };
    ui.painter()
        .add(egui::Shape::convex_polygon(pts, col, egui::Stroke::NONE));
    resp.clicked()
}

/// The revealed hierarchy for one level.
#[cfg(feature = "egui")]
fn draw_hierarchy(ui: &mut bevy_egui::egui::Ui, ab: &mut AssetBrowser, c: &Ctx) {
    use crate::ui_theme as theme;
    use bevy_egui::egui::{self, RichText};

    if let Load::Running(w) = ab.level_state.clone() {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(RichText::new(w).size(10.0).color(DIM));
        });
        return;
    }
    if let Load::Failed(e) = ab.level_state.clone() {
        ui.label(RichText::new(e).size(10.0).color(WARN));
        return;
    }
    let Some(lv) = ab.level.as_ref().map(|l| l.lv) else {
        ui.label(RichText::new("nothing open").size(10.0).color(DIM));
        return;
    };
    ui.horizontal(|ui| {
        // ASCII "<", not U+2190: bahnschrift has no arrows block either (this button showed a
        // tofu box). Every glyph in this panel is now Latin-1 or painted.
        if ui.small_button("< overview").clicked() {
            ab.mode = Mode::Landing;
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .small_button("rebuild")
                .on_hover_text("re-read this level from the bundle (after a game update)")
                .clicked()
            {
                ab.load_level(c.pack_root.clone(), lv, true);
            }
        });
    });
    // What this bundle is made of, as one quiet line — the level is provenance, so its population
    // belongs here rather than as a destination the user had to navigate through.
    if let Some(li) = ab.level.as_ref() {
        let mut counts: Vec<(&String, &u64)> = li.counts.iter().collect();
        counts.sort_by(|a, b| b.1.cmp(a.1));
        let head: Vec<String> = counts
            .iter()
            .take(5)
            .map(|(t, n)| format!("{t} {}", thousands(**n)))
            .collect();
        ui.label(RichText::new(head.join("  \u{00B7}  ")).size(9.0).color(DIM));
    }

    let mut rows: Vec<Row> = Vec::new();
    if let Some(li) = ab.level.as_ref() {
        build_rows(li, &ab.expanded, &mut rows);
    }

    // Scroll to the revealed node ONCE (latched), so dragging afterwards is not fought every frame.
    let mut scroll_off: Option<f32> = None;
    if let (Some(sel), None) = (ab.selected, ab.scroll_to) {
        if let Some(pos) = rows.iter().position(|r| r.node == sel && r.more == 0) {
            ab.scroll_to = Some(pos);
            scroll_off = Some((pos as f32 * ROW_H - 60.0).max(0.0));
        }
    }

    let mut toggle: Option<u32> = None;
    let mut select: Option<u32> = None;
    let mut area = egui::ScrollArea::vertical()
        .id_salt("assets_tree")
        .auto_shrink([false, false]);
    if let Some(off) = scroll_off {
        area = area.vertical_scroll_offset(off);
    }
    area.show_rows(ui, ROW_H, rows.len(), |ui, range| {
        let Some(li) = ab.level.as_ref() else { return };
        for r in range {
            let Some(row) = rows.get(r) else { continue };
            if row.more > 0 {
                ui.horizontal(|ui| {
                    ui.add_space(row.depth as f32 * 10.0);
                    ui.label(
                        RichText::new(format!("... {} more (use search)", row.more))
                            .size(9.5)
                            .color(DIM),
                    );
                });
                continue;
            }
            let Some(n) = li.nodes.get(row.node as usize) else {
                continue;
            };
            ui.horizontal(|ui| {
                ui.add_space(row.depth as f32 * 10.0);
                let branch = !n.kids.is_empty() || !n.comps.is_empty();
                if branch {
                    if disclosure(ui, ab.expanded.contains(&row.node)) {
                        toggle = Some(row.node);
                    }
                } else {
                    ui.add_space(12.0);
                }
                let is_go = n.ty == "GameObject";
                let label = if is_go {
                    if n.name.is_empty() { "(unnamed)".to_string() } else { n.name.clone() }
                } else if !n.name.is_empty() {
                    format!("{}  {}", n.ty, n.name)
                } else {
                    n.ty.clone()
                };
                let col = if is_go {
                    if n.active == 0 { DIM } else { theme::TEXT_BRIGHT }
                } else {
                    theme::SECTION
                };
                let sel = ab.selected == Some(row.node);
                if ui
                    .selectable_label(sel, RichText::new(label).size(10.5).color(col))
                    .clicked()
                {
                    select = Some(row.node);
                }
                // The component's one useful fact, dim and to the right of its type.
                if !n.value.is_empty() {
                    ui.label(RichText::new(&n.value).size(9.5).color(DIM));
                }
                if is_go && n.active == 0 {
                    ui.label(RichText::new("inactive").size(9.0).color(DIM));
                }
            });
        }
    });
    if let Some(t) = toggle {
        if !ab.expanded.remove(&t) {
            ab.expanded.insert(t);
        }
    }
    if let Some(s) = select {
        ab.selected = Some(s);
        ab.dump = None;
        ab.dump_state = Load::Missing;
        ab.asset = None;
        ab.asset_state = Load::Missing;
        ab.asset_for = None;
        ab.tex = None;
        ab.scroll_to = Some(0); // latch: a manual click must not re-trigger reveal scrolling
        // Selecting a component that points at a shared asset resolves it immediately — that is
        // the whole point of the inspector, and it is one cheap subprocess.
        let (r, is_mesh) = ab
            .level
            .as_ref()
            .and_then(|li| li.nodes.get(s as usize))
            .map(|n| (n.asset.clone(), n.ty == "MeshFilter"))
            .unwrap_or((None, false));
        if let Some(r) = r {
            ab.load_asset(c.pack_root.clone(), r);
        }
        // A mesh is previewed SKINNED, so the owning GameObject's base colour is resolved alongside
        // it. The owner is this component's parent in the tree.
        if is_mesh {
            let owner = ab.level.as_ref().and_then(|li| {
                let p = li.parent.get(s as usize).copied().unwrap_or(-1);
                (p >= 0).then(|| li.nodes.get(p as usize).map(|n| n.path_id))?
            });
            if let Some(go) = owner {
                ab.load_albedo(c.pack_root.clone(), lv, go);
            }
        }
    }
}

/// The selection inspector: semantic summary first, raw typetree behind an explicit action.
#[cfg(feature = "egui")]
#[allow(clippy::too_many_arguments)]
fn draw_inspector(
    ui: &mut bevy_egui::egui::Ui,
    ab: &mut AssetBrowser,
    c: &Ctx,
    ectx: &bevy_egui::egui::Context,
    cam_cmd: &mut crate::CameraCommand,
) {
    use crate::ui_theme as theme;
    use bevy_egui::egui::{self, RichText};

    let Some(sel) = ab.selected else {
        ui.label(RichText::new("select an object to inspect it").size(10.0).color(DIM));
        return;
    };
    let Some((ty, name, path_id, fold, value, size)) = ab.level.as_ref().and_then(|li| {
        li.nodes.get(sel as usize).map(|n| {
            (n.ty.clone(), n.name.clone(), n.path_id, n.fold, n.value.clone(), n.size)
        })
    }) else {
        return;
    };
    let lv = ab.level.as_ref().map(|l| l.lv).unwrap_or(0);

    ui.horizontal(|ui| {
        ui.label(
            RichText::new(if name.is_empty() { ty.clone() } else { name.clone() })
                .size(12.0)
                .color(theme::TEXT_BRIGHT),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new(format!("level{lv}")).size(9.0).color(DIM));
        });
    });
    let mut sub = if value.is_empty() { ty.clone() } else { format!("{ty}  \u{00B7}  {value}") };
    if size > 0 {
        sub.push_str(&format!("  \u{00B7}  {size} B"));
    }
    ui.label(RichText::new(sub).size(9.5).color(DIM));

    // Reverse join: this object's geometry in the loaded map.
    if fold != 0 {
        if let Some(p) = c.pack {
            let hits: Vec<usize> = p
                .instances
                .iter()
                .enumerate()
                .filter(|(_, i)| i.lv == lv && (i.par == fold || i.par2 == fold))
                .map(|(k, _)| k)
                .collect();
            ui.horizontal(|ui| {
                if hits.is_empty() {
                    // Truthful negative: plenty of GameObjects legitimately produce no geometry
                    // (triggers, logic nodes, culled shells). Saying so beats an inert button.
                    ui.label(RichText::new("no geometry in this pack").size(10.0).color(DIM));
                } else if ui
                    .button(format!("fly to geometry ({})", hits.len()))
                    .clicked()
                {
                    if let Some(i) = p.instances.get(hits[0]) {
                        cam_cmd.fly_to = Some(Vec3::new(i.affine[3], i.affine[7], i.affine[11]));
                    }
                }
            });
        }
    }
    ui.add_space(theme::SP_SM);

    egui::ScrollArea::vertical()
        .id_salt("assets_inspector")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            draw_asset_block(ui, ab, c, ectx);

            // ---- raw typetree: progressive disclosure, never the default view ----------
            ui.add_space(theme::SP_SM);
            ui.horizontal(|ui| {
                if ui
                    .small_button("read object")
                    .on_hover_text("the raw serialized fields, as Unity stored them")
                    .clicked()
                {
                    ab.load_dump(c.pack_root.clone(), lv, path_id);
                }
                match ab.dump_state.clone() {
                    Load::Running(_) => {
                        ui.spinner();
                    }
                    Load::Failed(e) => {
                        ui.label(RichText::new(e).size(9.5).color(WARN));
                    }
                    _ => {}
                }
            });

            let Some(d) = ab.dump.as_ref() else { return };
            if d.path_id != path_id {
                return; // a dump from a previous selection
            }
            if !d.script.is_empty() {
                ui.label(RichText::new(format!("script: {}", d.script)).size(10.5).color(theme::ACCENT));
            }
            // ---- THE HONESTY LINE ------------------------------------------------------
            // A stripped type tree describes the four base fields and nothing else. Printing those
            // alone would present a 76-byte object as a 4-field one, so state the shortfall.
            if !d.complete {
                if let (Some(read), Some(und)) = (d.read, d.undescribed) {
                    egui::Frame::NONE.fill(theme::CARD).inner_margin(6.0).show(ui, |ui| {
                        ui.label(RichText::new(format!("{und} BYTES UNDESCRIBED")).size(11.0).color(WARN));
                        ui.label(
                            RichText::new(format!(
                                "the bundle's type tree covers {read} of this object's {} bytes. \
The fields below are only the part Unity described — the script's own fields have no layout in the \
bundle (il2cpp stripping), so they are NOT shown and are NOT empty.",
                                d.size
                            ))
                            .size(9.5)
                            .color(DIM),
                        );
                    });
                } else if let Some(e) = d.error.as_ref() {
                    ui.label(RichText::new(format!("unreadable: {e}")).size(10.0).color(WARN));
                }
            }
            ui.add_space(theme::SP_SM);
            ui.label(
                RichText::new(&ab.dump_text)
                    .monospace()
                    .size(9.5)
                    .color(theme::TEXT_BRIGHT),
            );
        });
}

/// The resolved shared asset (mesh / texture / material / physic material), if the selected
/// component points at one.
#[cfg(feature = "egui")]
fn draw_asset_block(
    ui: &mut bevy_egui::egui::Ui,
    ab: &mut AssetBrowser,
    c: &Ctx,
    ectx: &bevy_egui::egui::Context,
) {
    use crate::ui_theme as theme;
    use bevy_egui::egui::{self, RichText};

    match ab.asset_state.clone() {
        Load::Running(w) => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new(w).size(10.0).color(DIM));
            });
            return;
        }
        Load::Failed(e) => {
            ui.label(RichText::new(format!("asset unavailable: {e}")).size(10.0).color(WARN));
            return;
        }
        Load::Missing => return,
        Load::Ready => {}
    }
    // Upload the albedo once it has arrived. REPEAT wrapping: these UVs run well past 1.0 (a
    // tiling material), and clamping would smear the edge texel across the whole model.
    if ab.albedo_tex.is_none() {
        if let Some(p) = ab.albedo.as_ref().and_then(|a| a.thumb.clone()) {
            if let Some(h) = load_texture_wrapped(ectx, &p, true) {
                ab.albedo_tex = Some((p, h));
            }
        }
    }
    let Some(a) = ab.asset.as_ref() else { return };
    let kind = a.kind.clone();
    let aname = a.name.clone();
    let src = a.src_file.clone();

    ui.separator();
    ui.horizontal(|ui| {
        ui.label(RichText::new(kind.to_uppercase()).size(9.5).color(DIM));
        ui.label(RichText::new(&aname).size(11.0).color(theme::TEXT_BRIGHT));
    });
    if let Some(e) = a.error.as_ref() {
        ui.label(RichText::new(e).size(10.0).color(WARN));
    }

    // A clicked texture slot re-targets the inspector, but `a` is borrowed out of `ab` for the whole
    // block — so the request is recorded here and issued once that borrow has ended.
    let mut follow: Option<AssetRef> = None;
    match kind.as_str() {
        "mesh" => {
            let (tris, verts, subs, shown, readable) =
                (a.tris, a.verts, a.submeshes, a.tris_shown, a.readable);
            let bounds = a.bounds.as_ref().map(|b| (b.c, b.e));
            let geom_err = a.geom_error.clone();
            let positions = &a.positions;
            let indices = &a.indices;
            let uvs = &a.uvs;
            // Thumbnail first: a mesh is a shape, and the numbers only mean something beside it.
            // Height follows the space the inspector actually has, leaving room for the stats and
            // skin line below — a fixed 210 pushed them off the bottom of the panel.
            let canvas_h = (ui.available_height() - 66.0).clamp(120.0, 280.0);
            let size = egui::vec2(ui.available_width(), canvas_h);
            let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::drag());
            ui.painter().rect_filled(rect, 2.0, theme::BG);
            if resp.dragged() {
                let d = resp.drag_delta();
                ab.orbit.0 += d.x * 0.01;
                ab.orbit.1 = (ab.orbit.1 + d.y * 0.01).clamp(-1.4, 1.4);
            }
            // Wheel zoom. The delta is CONSUMED (zeroed) so the surrounding ScrollArea — which
            // reads scroll after its content closure returns — does not also scroll the inspector.
            if resp.hovered() {
                let dy = ui.input_mut(|i| {
                    let d = i.smooth_scroll_delta.y;
                    if d != 0.0 {
                        i.smooth_scroll_delta.y = 0.0;
                        i.raw_scroll_delta.y = 0.0;
                    }
                    d
                });
                if dy != 0.0 {
                    ab.zoom = (ab.zoom * (1.0 + dy * 0.0025)).clamp(0.15, 25.0);
                }
            }
            let skin = if ab.textured {
                ab.albedo_tex.as_ref().map(|(_, h)| h.id())
            } else {
                None
            };
            draw_mesh_thumb(
                ui, rect, positions, uvs, indices, ab.orbit, ab.zoom, ab.wireframe, skin,
            );
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!(
                        "{} tris  \u{00B7}  {} verts  \u{00B7}  {subs} submesh{}",
                        thousands(tris),
                        thousands(verts),
                        if subs == 1 { "" } else { "es" }
                    ))
                    .size(9.5)
                    .color(DIM),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .small_button("reset")
                        .on_hover_text("drag to orbit \u{00B7} wheel to zoom")
                        .clicked()
                    {
                        ab.orbit = (0.6, -0.4);
                        ab.zoom = 1.0;
                    }
                    let mut wf = ab.wireframe;
                    if ui.checkbox(&mut wf, RichText::new("wire").size(9.5)).changed() {
                        ab.wireframe = wf;
                    }
                    let mut tx = ab.textured;
                    if ui
                        .checkbox(&mut tx, RichText::new("tex").size(9.5))
                        .on_hover_text("skin the mesh with its base-colour texture")
                        .changed()
                    {
                        ab.textured = tx;
                    }
                    if ab.zoom != 1.0 {
                        ui.label(
                            RichText::new(format!("{:.1}x", ab.zoom)).size(9.0).color(DIM),
                        );
                    }
                });
            });
            // Say which texture is on the model, or why there is none — an untextured grey model
            // otherwise looks like a preview bug rather than a shadow proxy with no _MainTex.
            match ab.albedo_state.clone() {
                Load::Running(_) => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(RichText::new("resolving texture").size(9.0).color(DIM));
                    });
                }
                Load::Ready => {
                    if let Some(al) = ab.albedo.as_ref() {
                        let line = match (al.error.as_ref(), al.texture.is_empty()) {
                            (Some(e), _) => format!("no skin: {e}"),
                            (None, false) => format!(
                                "skin: {}  \u{00B7}  {} \u{00d7} {} {}  \u{00B7}  {}",
                                al.texture, al.w, al.h, al.format, al.material
                            ),
                            _ => "no base-colour texture".to_string(),
                        };
                        ui.label(RichText::new(line).size(9.0).color(DIM));
                    }
                }
                _ => {}
            }
            if let Some((cc, ee)) = bounds {
                ui.label(
                    RichText::new(format!(
                        "bounds {:.2} \u{00d7} {:.2} \u{00d7} {:.2} m",
                        ee[0] * 2.0,
                        ee[1] * 2.0,
                        ee[2] * 2.0
                    ))
                    .size(9.0)
                    .color(DIM),
                );
                let _ = cc;
            }
            // Never let the preview imply it is the whole mesh.
            if shown > 0 && shown < tris {
                ui.label(
                    RichText::new(format!(
                        "preview shows {} of {} triangles",
                        thousands(shown),
                        thousands(tris)
                    ))
                    .size(9.0)
                    .color(WARN),
                );
            }
            if let Some(e) = geom_err {
                ui.label(RichText::new(format!("geometry not decodable: {e}")).size(9.0).color(WARN));
            }
            if !readable {
                ui.label(
                    RichText::new("m_IsReadable = false (the game cannot read this back at runtime)")
                        .size(9.0)
                        .color(DIM),
                );
            }
        }
        "texture" => {
            let (w, h, mips, fmt) = (a.w, a.h, a.mips, a.format.clone());
            let thumb = a.thumb.clone();
            if let Some(t) = thumb {
                // Decode once and keep the handle; egui uploads it to the GPU on first use.
                if ab.tex.as_ref().map(|(p, _)| p != &t).unwrap_or(true) {
                    if let Some(h) = load_texture(ectx, &t) {
                        ab.tex = Some((t.clone(), h));
                    }
                }
                if let Some((_, handle)) = ab.tex.as_ref() {
                    let avail = ui.available_width().min(240.0);
                    ui.add(
                        egui::Image::new((handle.id(), egui::vec2(avail, avail)))
                            .maintain_aspect_ratio(true)
                            .max_size(egui::vec2(avail, avail)),
                    );
                }
            }
            ui.label(
                RichText::new(format!("{w} \u{00d7} {h}  \u{00B7}  {fmt}  \u{00B7}  {mips} mips"))
                    .size(9.5)
                    .color(DIM),
            );
        }
        "material" => {
            let shader = a.shader.clone();
            ui.label(RichText::new(format!("shader  {shader}")).size(10.0).color(theme::ACCENT));
            for s in &a.slots {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&s.slot).size(9.5).color(DIM));
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new(if s.tex.is_empty() { "(unnamed)" } else { &s.tex })
                                    .size(10.0)
                                    .color(theme::TEXT_BRIGHT),
                            )
                            .frame(false),
                        )
                        .on_hover_text("open this texture")
                        .clicked()
                    {
                        follow = Some(AssetRef {
                            origin: s.origin.clone(),
                            file_id: s.file_id,
                            path_id: s.path_id,
                        });
                    }
                });
            }
            // Colour swatches read faster than four floats.
            for col in a.colors.iter().take(6) {
                ui.horizontal(|ui| {
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(14.0, 10.0), egui::Sense::hover());
                    ui.painter().rect_filled(
                        rect,
                        1.0,
                        egui::Color32::from_rgb(
                            (col.rgba[0].clamp(0.0, 1.0) * 255.0) as u8,
                            (col.rgba[1].clamp(0.0, 1.0) * 255.0) as u8,
                            (col.rgba[2].clamp(0.0, 1.0) * 255.0) as u8,
                        ),
                    );
                    ui.label(RichText::new(&col.name).size(9.0).color(DIM));
                });
            }
            // Scalars people actually reason about first, then whatever else the shader defines.
            const NOTABLE: [&str; 6] = [
                "_Metallic", "_Glossiness", "_Smoothness", "_BumpScale", "_Cutoff", "_Parallax",
            ];
            let mut floats: Vec<&NamedFloat> = a.floats.iter().collect();
            floats.sort_by_key(|f| {
                NOTABLE.iter().position(|n| *n == f.name).unwrap_or(NOTABLE.len())
            });
            for f in floats.iter().take(8) {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&f.name).size(9.0).color(DIM));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(format!("{:.3}", f.v)).size(9.0).color(theme::SECTION),
                        );
                    });
                });
            }
            if a.floats.len() > 8 {
                ui.label(
                    RichText::new(format!("+{} more shader properties", a.floats.len() - 8))
                        .size(9.0)
                        .color(DIM),
                );
            }
        }
        "physicMaterial" => {
            for (k, v) in [
                ("dynamic friction", format!("{:.2}", a.dyn_friction)),
                ("static friction", format!("{:.2}", a.static_friction)),
                ("bounciness", format!("{:.2}", a.bounciness)),
                ("friction combine", a.friction_combine.clone()),
                ("bounce combine", a.bounce_combine.clone()),
            ] {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(k).size(9.5).color(DIM));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new(v).size(9.5).color(theme::TEXT_BRIGHT));
                    });
                });
            }
        }
        _ => {}
    }
    // Provenance last and quiet: which shared file this came out of, and its id in it.
    let ty = a.ty.clone();
    let pid = a.path_id;
    if !src.is_empty() {
        ui.label(
            RichText::new(format!("{ty}  \u{00B7}  pathID {pid}  \u{00B7}  {src}"))
                .size(9.0)
                .color(DIM),
        );
    }
    if let Some(r) = follow {
        ab.load_asset(c.pack_root.clone(), r);
    }
}

/// Decode a PNG off disk into an egui texture. Returns None on any failure — a preview that cannot
/// be shown must not take the panel down with it.
#[cfg(feature = "egui")]
fn load_texture(
    ctx: &bevy_egui::egui::Context,
    path: &str,
) -> Option<bevy_egui::egui::TextureHandle> {
    load_texture_wrapped(ctx, path, false)
}

/// As `load_texture`, but `repeat` selects REPEAT wrapping — required for mesh skins, whose UVs
/// routinely exceed [0,1] on tiling materials.
#[cfg(feature = "egui")]
fn load_texture_wrapped(
    ctx: &bevy_egui::egui::Context,
    path: &str,
    repeat: bool,
) -> Option<bevy_egui::egui::TextureHandle> {
    use bevy_egui::egui;
    let bytes = std::fs::read(path).ok()?;
    let img = image::load_from_memory(&bytes).ok()?.to_rgba8();
    let (w, h) = img.dimensions();
    let color = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], img.as_raw());
    let opts = egui::TextureOptions {
        wrap_mode: if repeat {
            egui::TextureWrapMode::Repeat
        } else {
            egui::TextureWrapMode::ClampToEdge
        },
        ..egui::TextureOptions::LINEAR
    };
    Some(ctx.load_texture(path, color, opts))
}

/// Paint a mesh thumbnail with the egui painter: fit to bounds, orbit, flat-shade by facing.
/// Software-projected on purpose — it needs no render-graph node and no GPU resources for what is
/// a 150px thumbnail of a few thousand triangles.
#[cfg(feature = "egui")]
#[allow(clippy::too_many_arguments)]
fn draw_mesh_thumb(
    ui: &mut bevy_egui::egui::Ui,
    rect: bevy_egui::egui::Rect,
    pos: &[[f32; 3]],
    uvs: &[[f32; 2]],
    idx: &[u32],
    orbit: (f32, f32),
    zoom: f32,
    wireframe: bool,
    skin: Option<bevy_egui::egui::TextureId>,
) {
    use bevy_egui::egui::{self, Color32, Stroke};
    if pos.is_empty() || idx.len() < 3 {
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "no geometry",
            egui::FontId::proportional(10.0),
            DIM,
        );
        return;
    }
    let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
    for p in pos {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let ctr = Vec3::new(
        (lo[0] + hi[0]) * 0.5,
        (lo[1] + hi[1]) * 0.5,
        (lo[2] + hi[2]) * 0.5,
    );
    let extent = ((hi[0] - lo[0]).max(hi[1] - lo[1]).max(hi[2] - lo[2])).max(1e-4);
    let (sy, cy) = orbit.0.sin_cos();
    let (sp, cp) = orbit.1.sin_cos();
    let scale = rect.height().min(rect.width()) * 0.42 * zoom.max(0.01) / extent;
    let project = |p: &[f32; 3]| -> (egui::Pos2, f32) {
        let v = Vec3::new(p[0], p[1], p[2]) - ctr;
        // yaw about Y, then pitch about X — a fixed three-quarter view the drag rotates.
        let x = v.x * cy + v.z * sy;
        let z = -v.x * sy + v.z * cy;
        let y = v.y * cp - z * sp;
        let depth = v.y * sp + z * cp;
        (
            egui::pos2(rect.center().x + x * scale, rect.center().y - y * scale),
            depth,
        )
    };
    // View-space position per vertex: (screen x, screen y, depth). Depth drives the back-to-front
    // sort; the 3-D delta drives real per-face lighting.
    let tri_count = idx.len() / 3;
    let mut proj: Vec<(egui::Pos2, f32, Vec3)> = Vec::with_capacity(pos.len());
    for p in pos {
        let (sp2, depth) = project(p);
        proj.push((sp2, depth, Vec3::new(p[0], p[1], p[2])));
    }
    let mut order: Vec<(f32, usize)> = Vec::with_capacity(tri_count);
    for t in 0..tri_count {
        let (a, b, cc) = (idx[t * 3] as usize, idx[t * 3 + 1] as usize, idx[t * 3 + 2] as usize);
        if a >= proj.len() || b >= proj.len() || cc >= proj.len() {
            continue;
        }
        order.push(((proj[a].1 + proj[b].1 + proj[cc].1) / 3.0, t));
    }
    order.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

    let painter = ui.painter_at(rect);
    if wireframe {
        let s = Stroke::new(0.6, Color32::from_rgb(150, 190, 150));
        for &(_, t) in &order {
            let (a, b, cc) =
                (idx[t * 3] as usize, idx[t * 3 + 1] as usize, idx[t * 3 + 2] as usize);
            let (p0, p1, p2) = (proj[a].0, proj[b].0, proj[cc].0);
            painter.line_segment([p0, p1], s);
            painter.line_segment([p1, p2], s);
            painter.line_segment([p2, p0], s);
        }
        return;
    }

    // ONE mesh rather than a Shape per triangle. `Shape::convex_polygon` anti-aliases by feathering
    // its outline, and on a degenerate/sliver triangle — of which a game mesh has many — that
    // feathering shoots long spikes out of the shape. Raw mesh triangles carry no feathering, so
    // the artifact cannot occur, and 1,589 triangles become one draw command instead of 1,589.
    let light = Vec3::new(0.35, 0.75, 0.55).normalize();
    let (sy2, cy2) = orbit.0.sin_cos();
    let mut mesh = egui::epaint::Mesh::default();
    // A textured mesh carries the albedo's TextureId; the vertex colour then MODULATES it, so the
    // same lambert term shades the texture instead of replacing it.
    let textured = skin.is_some() && uvs.len() == pos.len();
    if let Some(t) = skin {
        if textured {
            mesh.texture_id = t;
        }
    }
    mesh.vertices.reserve(order.len() * 3);
    mesh.indices.reserve(order.len() * 3);
    for &(_, t) in &order {
        let (a, b, cc) = (idx[t * 3] as usize, idx[t * 3 + 1] as usize, idx[t * 3 + 2] as usize);
        let (p0, p1, p2) = (proj[a].0, proj[b].0, proj[cc].0);
        // Drop sub-pixel triangles outright: they cannot contribute a visible pixel and only cost
        // tessellation work.
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        if (e1.x * e2.y - e1.y * e2.x).abs() < 0.05 {
            continue;
        }
        // Lambert from the true face normal, rotated by the same yaw the projection uses.
        let n = (proj[b].2 - proj[a].2).cross(proj[cc].2 - proj[a].2);
        let n = if n.length_squared() > 1e-12 { n.normalize() } else { Vec3::Y };
        let nr = Vec3::new(n.x * cy2 + n.z * sy2, n.y, -n.x * sy2 + n.z * cy2);
        let lam = nr.dot(light).abs(); // abs(): these meshes are not reliably wound
        let col = if textured {
            // Grey multiplier: keeps the texture's own colour and just lights it. Biased brighter
            // than the flat-shaded case because it is multiplying, not replacing.
            let v = ((0.55 + 0.45 * lam) * 255.0).clamp(0.0, 255.0) as u8;
            Color32::from_rgb(v, v, v)
        } else {
            let v = ((0.30 + 0.70 * lam) * 210.0).clamp(0.0, 255.0) as u8;
            Color32::from_rgb(v, (v as f32 * 1.02).min(255.0) as u8, (v as f32 * 0.92) as u8)
        };
        let base = mesh.vertices.len() as u32;
        for (p, vi) in [(p0, a), (p1, b), (p2, cc)] {
            // OBJ's V axis points up, egui's points down — flip, or the skin renders mirrored.
            let uv = if textured {
                egui::pos2(uvs[vi][0], 1.0 - uvs[vi][1])
            } else {
                egui::epaint::WHITE_UV
            };
            mesh.vertices.push(egui::epaint::Vertex { pos: p, uv, color: col });
        }
        mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
    if !mesh.is_empty() {
        painter.add(egui::Shape::mesh(mesh));
    }
    let _ = (extent, scale);
}

/// 1234567 -> "1,234,567". The counts here run to seven figures and are unreadable raw.
fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}
