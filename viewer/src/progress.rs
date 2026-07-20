//! Persistent task, objective, and key ownership state shared by Tasks, Map Intel, and Planner.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Resource, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PlayerProgress {
    pub tracked: HashSet<String>,
    pub done: HashSet<String>,
    pub owned_keys: HashSet<String>,
    #[serde(skip)]
    pub loaded: bool,
}

impl PlayerProgress {
    pub fn owns_key(&self, name: &str) -> bool {
        let needle = name.trim().to_lowercase();
        self.owned_keys.iter().any(|key| key.trim().to_lowercase() == needle)
    }
}

pub struct ProgressPlugin;

impl Plugin for ProgressPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PlayerProgress>()
            .add_systems(Startup, load_progress)
            .add_systems(Update, save_progress);
    }
}

fn load_progress(
    mut progress: ResMut<PlayerProgress>,
    mut tracker: ResMut<crate::ui::QuestTracker>,
) {
    let path = crate::paths::progress_path();
    if let Ok(text) = std::fs::read_to_string(&path) {
        match serde_json::from_str::<PlayerProgress>(&text) {
            Ok(mut saved) => {
                saved.loaded = true;
                tracker.active = saved.tracked.clone();
                *progress = saved;
                info!("progress: loaded {}", path.display());
                return;
            }
            Err(err) => warn!("progress: could not parse {}: {err}", path.display()),
        }
    }
    progress.loaded = true;
}

fn save_progress(progress: Res<PlayerProgress>) {
    if !progress.loaded || !progress.is_changed() {
        return;
    }
    let path = crate::paths::progress_path();
    if let Some(parent) = path.parent() {
        if let Err(err) = std::fs::create_dir_all(parent) {
            warn!("progress: could not create {}: {err}", parent.display());
            return;
        }
    }
    match serde_json::to_string_pretty(&*progress) {
        Ok(text) => {
            if let Err(err) = std::fs::write(&path, text) {
                warn!("progress: could not write {}: {err}", path.display());
            }
        }
        Err(err) => warn!("progress: could not serialize: {err}"),
    }
}
