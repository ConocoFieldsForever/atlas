//! Import completed map packs from an older Atlas installation.
//!
//! Imports are deliberately copies: the old installation stays usable and later rebuild/delete
//! operations in either installation cannot affect the other. A map is copied into a private
//! `.importing` directory and renamed only after every file succeeds, so the normal pack scanner
//! never mistakes an interrupted import for a ready map.

use anyhow::{anyhow, bail, Context, Result};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

const COPY_BUFFER_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug)]
pub struct ImportMap {
    pub key: String,
    source: PathBuf,
    pub size_bytes: u64,
    pub already_installed: bool,
}

#[derive(Clone, Debug)]
struct SharedFile {
    source: PathBuf,
    relative: PathBuf,
    size_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct ImportPreview {
    /// The folder selected by the user. It is shown in the confirmation dialog but never logged or
    /// persisted, because Windows account/folder names can be personal information.
    pub selected_folder: PathBuf,
    pub maps: Vec<ImportMap>,
    shared_files: Vec<SharedFile>,
    pub copy_bytes: u64,
}

impl ImportPreview {
    pub fn new_map_count(&self) -> usize {
        self.maps
            .iter()
            .filter(|map| !map.already_installed)
            .count()
    }

    pub fn installed_map_count(&self) -> usize {
        self.maps.iter().filter(|map| map.already_installed).count()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImportMapStatus {
    Queued,
    Importing,
    Imported,
    Skipped,
    Failed(String),
    Cancelled,
}

#[derive(Clone, Debug)]
pub struct ImportMapProgress {
    pub key: String,
    pub status: ImportMapStatus,
}

#[derive(Clone, Debug)]
pub struct ImportProgress {
    pub maps: Vec<ImportMapProgress>,
    pub copied_bytes: u64,
    pub total_bytes: u64,
    pub finished: bool,
    pub cancelled: bool,
    pub shared_files_copied: usize,
}

impl ImportProgress {
    pub fn imported_count(&self) -> usize {
        self.maps
            .iter()
            .filter(|map| map.status == ImportMapStatus::Imported)
            .count()
    }

    pub fn skipped_count(&self) -> usize {
        self.maps
            .iter()
            .filter(|map| map.status == ImportMapStatus::Skipped)
            .count()
    }

    pub fn failed_count(&self) -> usize {
        self.maps
            .iter()
            .filter(|map| matches!(map.status, ImportMapStatus::Failed(_)))
            .count()
    }
}

pub struct ImportJob {
    progress: Arc<Mutex<ImportProgress>>,
    cancel: Arc<AtomicBool>,
}

impl ImportJob {
    pub fn start(preview: ImportPreview, destination_packs: PathBuf) -> Self {
        let progress = Arc::new(Mutex::new(ImportProgress {
            maps: preview
                .maps
                .iter()
                .map(|map| ImportMapProgress {
                    key: map.key.clone(),
                    status: if map.already_installed {
                        ImportMapStatus::Skipped
                    } else {
                        ImportMapStatus::Queued
                    },
                })
                .collect(),
            copied_bytes: 0,
            total_bytes: preview.copy_bytes,
            finished: false,
            cancelled: false,
            shared_files_copied: 0,
        }));
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_progress = Arc::clone(&progress);
        let worker_cancel = Arc::clone(&cancel);
        std::thread::spawn(move || {
            run_import(
                preview,
                &destination_packs,
                &worker_progress,
                &worker_cancel,
            );
        });
        Self { progress, cancel }
    }

    pub fn snapshot(&self) -> ImportProgress {
        self.progress
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// Inspect a user-selected old Atlas folder (or its `packs` folder) without modifying either tree.
pub fn preview(selected: &Path, destination_packs: &Path) -> Result<ImportPreview> {
    let source_packs = resolve_packs_dir(selected)?;
    if same_existing_path(&source_packs, destination_packs) {
        bail!("That is the packs folder Atlas is already using");
    }

    let mut maps = Vec::new();
    for entry in fs::read_dir(&source_packs)
        .with_context(|| format!("Could not read {}", source_packs.display()))?
        .flatten()
    {
        let file_type = match entry.file_type() {
            Ok(kind) => kind,
            Err(_) => continue,
        };
        if !file_type.is_dir() || file_type.is_symlink() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(key) = name.strip_suffix(".eftpack") else {
            continue;
        };
        if key.is_empty() || !valid_pack(&entry.path()) {
            continue;
        }
        let already_installed = destination_packs.join(&name).exists();
        maps.push(ImportMap {
            key: key.to_string(),
            source: entry.path(),
            size_bytes: directory_size(&entry.path()),
            already_installed,
        });
    }
    maps.sort_by(|a, b| a.key.cmp(&b.key));
    if maps.is_empty() {
        bail!("No completed Atlas map packs were found in that folder");
    }

    let mut shared_files = Vec::new();
    let source_shared = source_packs.join("shared");
    if source_shared.is_dir() {
        collect_shared_files(
            &source_shared,
            &source_shared,
            &destination_packs.join("shared"),
            &mut shared_files,
        )?;
    }
    let map_bytes: u64 = maps
        .iter()
        .filter(|map| !map.already_installed)
        .map(|map| map.size_bytes)
        .sum();
    let shared_bytes: u64 = shared_files.iter().map(|file| file.size_bytes).sum();
    Ok(ImportPreview {
        selected_folder: selected.to_path_buf(),
        maps,
        shared_files,
        copy_bytes: map_bytes.saturating_add(shared_bytes),
    })
}

fn resolve_packs_dir(selected: &Path) -> Result<PathBuf> {
    if !selected.is_dir() {
        bail!("The selected folder does not exist");
    }
    let nested = selected.join("packs");
    if nested.is_dir() {
        return Ok(nested);
    }
    Ok(selected.to_path_buf())
}

fn same_existing_path(left: &Path, right: &Path) -> bool {
    match (fs::canonicalize(left), fs::canonicalize(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => left == right,
    }
}

fn valid_pack(pack: &Path) -> bool {
    let manifest_ok = fs::read_to_string(pack.join("manifest.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .is_some();
    manifest_ok && pack.join("meshes.bin").is_file() && pack.join("instances.bin").is_file()
}

fn directory_size(path: &Path) -> u64 {
    let mut total = 0u64;
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    for entry in entries.flatten() {
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_symlink() {
            continue;
        }
        if kind.is_dir() {
            total = total.saturating_add(directory_size(&entry.path()));
        } else if kind.is_file() {
            total = total.saturating_add(entry.metadata().map(|m| m.len()).unwrap_or(0));
        }
    }
    total
}

fn collect_shared_files(
    root: &Path,
    current: &Path,
    destination_shared: &Path,
    out: &mut Vec<SharedFile>,
) -> Result<()> {
    for entry in fs::read_dir(current)
        .with_context(|| format!("Could not inspect shared data in {}", current.display()))?
        .flatten()
    {
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            continue;
        }
        let relative = entry
            .path()
            .strip_prefix(root)
            .map(Path::to_path_buf)
            .map_err(|_| anyhow!("Shared-data path escaped its source folder"))?;
        let first = relative
            .components()
            .next()
            .and_then(|part| part.as_os_str().to_str());
        if matches!(first, Some("logs" | "texcache")) {
            continue;
        }
        if kind.is_dir() {
            collect_shared_files(root, &entry.path(), destination_shared, out)?;
        } else if kind.is_file() && !destination_shared.join(&relative).exists() {
            out.push(SharedFile {
                source: entry.path(),
                relative,
                size_bytes: entry.metadata()?.len(),
            });
        }
    }
    Ok(())
}

fn run_import(
    preview: ImportPreview,
    destination_packs: &Path,
    progress: &Arc<Mutex<ImportProgress>>,
    cancel: &AtomicBool,
) {
    if let Err(error) = fs::create_dir_all(destination_packs) {
        finish_with_global_error(progress, format!("Could not create packs folder: {error}"));
        return;
    }
    for (index, map) in preview.maps.iter().enumerate() {
        if map.already_installed {
            continue;
        }
        if cancel.load(Ordering::Relaxed) {
            mark_remaining_cancelled(progress, index);
            finish(progress, true);
            return;
        }
        set_map_status(progress, index, ImportMapStatus::Importing);
        match copy_map(map, destination_packs, progress, cancel) {
            Ok(CopyOutcome::Copied) => set_map_status(progress, index, ImportMapStatus::Imported),
            Ok(CopyOutcome::Skipped) => set_map_status(progress, index, ImportMapStatus::Skipped),
            Err(CopyError::Cancelled) => {
                set_map_status(progress, index, ImportMapStatus::Cancelled);
                mark_remaining_cancelled(progress, index + 1);
                finish(progress, true);
                return;
            }
            Err(CopyError::Failed(error)) => {
                set_map_status(progress, index, ImportMapStatus::Failed(error));
            }
        }
    }

    if !cancel.load(Ordering::Relaxed) {
        for file in &preview.shared_files {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            match copy_shared_file(file, destination_packs, progress, cancel) {
                Ok(true) => {
                    with_progress(progress, |p| p.shared_files_copied += 1);
                }
                Ok(false) => {}
                Err(CopyError::Cancelled) => break,
                // Shared metadata is useful but not required to load a map. A failed shared file
                // must not downgrade successfully imported packs.
                Err(CopyError::Failed(_)) => {}
            }
        }
    }
    finish(progress, cancel.load(Ordering::Relaxed));
}

enum CopyOutcome {
    Copied,
    Skipped,
}

enum CopyError {
    Cancelled,
    Failed(String),
}

fn copy_map(
    map: &ImportMap,
    destination_packs: &Path,
    progress: &Arc<Mutex<ImportProgress>>,
    cancel: &AtomicBool,
) -> std::result::Result<CopyOutcome, CopyError> {
    let destination = destination_packs.join(format!("{}.eftpack", map.key));
    if destination.exists() {
        return Ok(CopyOutcome::Skipped);
    }
    let staging = destination_packs.join(format!("{}.eftpack.importing", map.key));
    if staging.exists() {
        fs::remove_dir_all(&staging).map_err(|e| {
            CopyError::Failed(format!("Could not clear an earlier incomplete import: {e}"))
        })?;
    }
    let result = copy_directory(&map.source, &staging, progress, cancel).and_then(|_| {
        if cancel.load(Ordering::Relaxed) {
            return Err(CopyError::Cancelled);
        }
        if destination.exists() {
            return Ok(CopyOutcome::Skipped);
        }
        fs::rename(&staging, &destination)
            .map(|_| CopyOutcome::Copied)
            .map_err(|e| CopyError::Failed(format!("Could not finish the imported pack: {e}")))
    });
    if staging.exists() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

fn copy_directory(
    source: &Path,
    destination: &Path,
    progress: &Arc<Mutex<ImportProgress>>,
    cancel: &AtomicBool,
) -> std::result::Result<(), CopyError> {
    fs::create_dir_all(destination)
        .map_err(|e| CopyError::Failed(format!("Could not create import folder: {e}")))?;
    let entries = fs::read_dir(source)
        .map_err(|e| CopyError::Failed(format!("Could not read old map files: {e}")))?;
    for entry in entries {
        if cancel.load(Ordering::Relaxed) {
            return Err(CopyError::Cancelled);
        }
        let entry =
            entry.map_err(|e| CopyError::Failed(format!("Could not read a map file: {e}")))?;
        let kind = entry
            .file_type()
            .map_err(|e| CopyError::Failed(format!("Could not inspect a map file: {e}")))?;
        if kind.is_symlink() {
            continue;
        }
        let target = destination.join(entry.file_name());
        if kind.is_dir() {
            copy_directory(&entry.path(), &target, progress, cancel)?;
        } else if kind.is_file() {
            copy_file(&entry.path(), &target, progress, cancel, false)?;
        }
    }
    Ok(())
}

fn copy_shared_file(
    file: &SharedFile,
    destination_packs: &Path,
    progress: &Arc<Mutex<ImportProgress>>,
    cancel: &AtomicBool,
) -> std::result::Result<bool, CopyError> {
    let destination = destination_packs.join("shared").join(&file.relative);
    if destination.exists() {
        return Ok(false);
    }
    let parent = destination
        .parent()
        .ok_or_else(|| CopyError::Failed("Invalid shared-data destination".into()))?;
    fs::create_dir_all(parent)
        .map_err(|e| CopyError::Failed(format!("Could not create shared-data folder: {e}")))?;
    let file_name = destination
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| CopyError::Failed("Invalid shared-data file name".into()))?;
    let staging = parent.join(format!("{file_name}.atlas-importing"));
    if staging.exists() {
        let _ = fs::remove_file(&staging);
    }
    let result = copy_file(&file.source, &staging, progress, cancel, true).and_then(|_| {
        if cancel.load(Ordering::Relaxed) {
            return Err(CopyError::Cancelled);
        }
        if destination.exists() {
            return Ok(false);
        }
        fs::rename(&staging, &destination)
            .map(|_| true)
            .map_err(|e| CopyError::Failed(format!("Could not finish shared-data file: {e}")))
    });
    if staging.exists() {
        let _ = fs::remove_file(staging);
    }
    result
}

fn copy_file(
    source: &Path,
    destination: &Path,
    progress: &Arc<Mutex<ImportProgress>>,
    cancel: &AtomicBool,
    create_new: bool,
) -> std::result::Result<(), CopyError> {
    let mut input = File::open(source)
        .map_err(|e| CopyError::Failed(format!("Could not open an old map file: {e}")))?;
    let mut options = OpenOptions::new();
    options.write(true);
    if create_new {
        options.create_new(true);
    } else {
        options.create(true).truncate(true);
    }
    let mut output = options
        .open(destination)
        .map_err(|e| CopyError::Failed(format!("Could not create an imported file: {e}")))?;
    let mut buffer = vec![0u8; COPY_BUFFER_BYTES];
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(CopyError::Cancelled);
        }
        let count = input
            .read(&mut buffer)
            .map_err(|e| CopyError::Failed(format!("Could not read an old map file: {e}")))?;
        if count == 0 {
            break;
        }
        output
            .write_all(&buffer[..count])
            .map_err(|e| CopyError::Failed(format!("Could not write an imported file: {e}")))?;
        with_progress(progress, |p| {
            p.copied_bytes = p.copied_bytes.saturating_add(count as u64);
        });
    }
    output
        .sync_all()
        .map_err(|e| CopyError::Failed(format!("Could not finish writing an imported file: {e}")))
}

fn with_progress(progress: &Arc<Mutex<ImportProgress>>, f: impl FnOnce(&mut ImportProgress)) {
    let mut guard = progress
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    f(&mut guard);
}

fn set_map_status(progress: &Arc<Mutex<ImportProgress>>, index: usize, status: ImportMapStatus) {
    with_progress(progress, |p| {
        if let Some(map) = p.maps.get_mut(index) {
            map.status = status;
        }
    });
}

fn mark_remaining_cancelled(progress: &Arc<Mutex<ImportProgress>>, start: usize) {
    with_progress(progress, |p| {
        for map in p.maps.iter_mut().skip(start) {
            if map.status == ImportMapStatus::Queued {
                map.status = ImportMapStatus::Cancelled;
            }
        }
    });
}

fn finish(progress: &Arc<Mutex<ImportProgress>>, cancelled: bool) {
    with_progress(progress, |p| {
        p.finished = true;
        p.cancelled = cancelled;
    });
}

fn finish_with_global_error(progress: &Arc<Mutex<ImportProgress>>, error: String) {
    with_progress(progress, |p| {
        for map in &mut p.maps {
            if map.status == ImportMapStatus::Queued {
                map.status = ImportMapStatus::Failed(error.clone());
            }
        }
        p.finished = true;
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn scratch(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("atlas-import-{name}-{nonce}"))
    }

    fn make_pack(root: &Path, key: &str, valid: bool) {
        let pack = root.join("packs").join(format!("{key}.eftpack"));
        fs::create_dir_all(&pack).unwrap();
        fs::write(pack.join("manifest.json"), if valid { "{}" } else { "{" }).unwrap();
        fs::write(pack.join("meshes.bin"), b"mesh").unwrap();
        fs::write(pack.join("instances.bin"), b"instances").unwrap();
    }

    fn wait(job: &ImportJob) -> ImportProgress {
        for _ in 0..200 {
            let snapshot = job.snapshot();
            if snapshot.finished {
                return snapshot;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("import did not finish");
    }

    #[test]
    fn preview_accepts_release_or_packs_folder_and_ignores_incomplete_maps() {
        let old = scratch("preview-old");
        let destination = scratch("preview-new").join("packs");
        make_pack(&old, "customs", true);
        make_pack(&old, "broken", false);
        fs::create_dir_all(old.join("packs").join("woods.eftpack.building")).unwrap();

        let from_release = preview(&old, &destination).unwrap();
        let from_packs = preview(&old.join("packs"), &destination).unwrap();
        assert_eq!(from_release.maps.len(), 1);
        assert_eq!(from_release.maps[0].key, "customs");
        assert_eq!(from_packs.maps.len(), 1);

        fs::remove_dir_all(old).unwrap();
    }

    #[test]
    fn import_is_staged_and_never_overwrites_an_installed_pack() {
        let old = scratch("copy-old");
        let new = scratch("copy-new");
        make_pack(&old, "customs", true);
        make_pack(&old, "woods", true);
        let installed = new.join("packs").join("customs.eftpack");
        fs::create_dir_all(&installed).unwrap();
        fs::write(installed.join("keep.txt"), b"newer").unwrap();
        fs::create_dir_all(old.join("packs").join("shared")).unwrap();
        fs::write(old.join("packs").join("shared").join("loot.json"), b"{}").unwrap();

        let preview = preview(&old, &new.join("packs")).unwrap();
        assert_eq!(preview.new_map_count(), 1);
        assert_eq!(preview.installed_map_count(), 1);
        let job = ImportJob::start(preview, new.join("packs"));
        let finished = wait(&job);

        assert_eq!(finished.imported_count(), 1);
        assert_eq!(finished.skipped_count(), 1);
        assert_eq!(fs::read(installed.join("keep.txt")).unwrap(), b"newer");
        assert!(new
            .join("packs")
            .join("woods.eftpack")
            .join("manifest.json")
            .is_file());
        assert!(!new.join("packs").join("woods.eftpack.importing").exists());
        assert!(new.join("packs").join("shared").join("loot.json").is_file());

        fs::remove_dir_all(old).unwrap();
        fs::remove_dir_all(new).unwrap();
    }
}
