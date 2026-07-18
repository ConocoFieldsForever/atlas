//! Minimal EN/RU localization. A compile-time `t(lang, key)` catalog + a `Lang` resource. egui is
//! immediate-mode, so flipping `Lang` re-renders the whole menu next frame — no restart, no
//! relayout. The default is seeded from the system UI locale; a `"lang"` key saved in
//! atlas.config.json (set by the menu's language toggle) overrides that automatic detection.

use bevy::prelude::*;

#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lang {
    En,
    Ru,
}

impl Lang {
    /// Two-letter badge shown on the toggle.
    pub fn code(self) -> &'static str {
        match self {
            Lang::En => "EN",
            Lang::Ru => "RU",
        }
    }
    /// The config tag persisted in atlas.config.json.
    pub fn tag(self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::Ru => "ru",
        }
    }
    pub fn toggled(self) -> Lang {
        match self {
            Lang::En => Lang::Ru,
            Lang::Ru => Lang::En,
        }
    }
}

/// Startup language: a saved override wins, else the system locale (`ru*` -> Ru), else English.
pub fn detect_lang(saved: Option<&str>) -> Lang {
    match saved {
        Some("ru") => return Lang::Ru,
        Some("en") => return Lang::En,
        _ => {}
    }
    if let Some(loc) = sys_locale::get_locale() {
        if loc.to_ascii_lowercase().starts_with("ru") {
            return Lang::Ru;
        }
    }
    Lang::En
}

/// Localized display name for a map, keyed by its dataset key (matches menu::KNOWN_MAPS). Falls
/// back to the English title for unknown keys / the En language.
pub fn map_title(lang: Lang, key: &str, en_title: &str) -> String {
    if lang == Lang::En {
        return en_title.to_string();
    }
    let ru = match key {
        "lighthouse" => "Маяк",
        "interchange" => "Развязка",
        "factory" => "Завод",
        "customs" => "Таможня",
        "woods" => "Лес",
        "shoreline" => "Берег",
        "reserve" => "Резерв",
        "labs" => "Лаборатория",
        "ground_zero" => "Эпицентр",
        "streets" => "Улицы Таркова",
        "labyrinth" => "Лабиринт",
        _ => return en_title.to_string(),
    };
    ru.to_string()
}

/// UI-string keys. Add an arm to `pair()` for each; a missing arm won't compile.
#[derive(Clone, Copy)]
pub enum K {
    Map,
    SelectLocation,
    PacksOnDisk,
    Intel,
    SyncNow,
    Synced,
    TasksLabel,
    Icons,
    Never,
    NotInstalled,
    Ready,
    ReadyUnstamped,
    GameFilesUpdated,
    Build,
    Building,
    Play,
    Delete,
    Update,
    Confirm,
    TickLight,
    TickGrass,
    TickZones,
    TickIcons,
    GameInstall,
    GameNotFound,
    ExtractedAssets,
    Choose,
    Set,
    IsSet,
    UsingDefault,
    FirstRunBanner,
    LanguageTip,
}

/// The catalog: `[english, russian]` per key.
fn pair(k: K) -> [&'static str; 2] {
    use K::*;
    match k {
        Map => ["MAP", "КАРТА"],
        SelectLocation => ["SELECT LOCATION", "ВЫБОР ЛОКАЦИИ"],
        PacksOnDisk => ["PACKS ON DISK", "ПАКЕТЫ НА ДИСКЕ"],
        Intel => ["INTEL", "ДАННЫЕ"],
        SyncNow => ["SYNC NOW", "ОБНОВИТЬ"],
        Synced => ["tarkov.dev synced", "tarkov.dev обновлён"],
        TasksLabel => ["tasks", "задачи"],
        Icons => ["icons", "иконок"],
        Never => ["never", "никогда"],
        NotInstalled => ["NOT INSTALLED", "НЕ УСТАНОВЛЕНО"],
        Ready => ["READY", "ГОТОВО"],
        ReadyUnstamped => ["READY (unstamped)", "ГОТОВО (без отметки)"],
        GameFilesUpdated => ["GAME FILES UPDATED", "ФАЙЛЫ ИГРЫ ОБНОВЛЕНЫ"],
        Build => ["BUILD", "СОБРАТЬ"],
        Building => ["BUILDING", "СБОРКА"],
        Play => ["PLAY", "ИГРАТЬ"],
        Delete => ["DELETE", "УДАЛИТЬ"],
        Update => ["UPDATE", "ОБНОВИТЬ"],
        Confirm => ["CONFIRM", "ПОДТВЕРДИТЬ"],
        TickLight => ["light", "свет"],
        TickGrass => ["grass", "трава"],
        TickZones => ["zones", "зоны"],
        TickIcons => ["icons", "иконки"],
        GameInstall => ["GAME INSTALL", "ПАПКА ИГРЫ"],
        GameNotFound => [
            "NOT FOUND - set the EscapeFromTarkov_Data path",
            "НЕ НАЙДЕНО — укажите путь к EscapeFromTarkov_Data",
        ],
        ExtractedAssets => ["EXTRACTED ASSETS", "РАСПАКОВАННЫЕ ДАННЫЕ"],
        Choose => ["CHOOSE\u{2026}", "ВЫБРАТЬ\u{2026}"],
        Set => ["SET", "ЗАДАТЬ"],
        IsSet => ["[set]", "[задано]"],
        UsingDefault => ["using default - CHOOSE to set", "по умолчанию — нажмите ВЫБРАТЬ"],
        FirstRunBanner => [
            "First run: choose a folder for EXTRACTED ASSETS. The first BUILD of a map runs a \
             one-time extraction from your game files into it (close the game first; ~1-6 GB per \
             map, can take a while); later builds are quick.",
            "Первый запуск: выберите папку для РАСПАКОВАННЫХ ДАННЫХ. Первая СБОРКА карты запускает \
             однократную распаковку из файлов игры в эту папку (сначала закройте игру; ~1-6 ГБ на \
             карту, может занять время); последующие сборки быстрые.",
        ],
        LanguageTip => ["Language / Язык (override auto-detect)", "Язык / Language (переопределить)"],
    }
}

/// Translate a UI-string key for the given language.
pub fn t(lang: Lang, k: K) -> &'static str {
    let [en, ru] = pair(k);
    match lang {
        Lang::En => en,
        Lang::Ru => ru,
    }
}
