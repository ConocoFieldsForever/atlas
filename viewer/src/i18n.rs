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

/// Localized display name for a map, keyed by its dataset id. The Russian names come from the
/// game-derived roster manifest (`crate::maps`), keyed by the SAME id as the English roster, so
/// EN/RU can't drift apart (the old hardcoded table was keyed "factory" while the roster shipped
/// "factory_rework", silently dropping Russian). Falls back to the English title for the En
/// language or any id not in the roster (e.g. an on-disk extra pack).
pub fn map_title(lang: Lang, key: &str, en_title: &str) -> String {
    if lang == Lang::En {
        return en_title.to_string();
    }
    crate::maps::ru_title(key)
        .map(str::to_string)
        .unwrap_or_else(|| en_title.to_string())
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
    Damaged,
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
    FirstBuildHint,
    LanguageTip,
    ProcessInBackground,
    ProcessInBackgroundTip,
    BuildDeps,
    DepsReady,
    DepsMissing,
    InstallDeps,
    BuildNeedsSetup,
    Installing,
    // Build / loading panel
    InstallingDeps,
    Done,
    Failed,
    Close,
    Cancel,
    ShowLog,
    HideLog,
    CopyLog,
    BuildFailed,
    BuildComplete,
    DepsDone,
    Starting,
    EstimatedTime,
    // INTEL strip
    IntelRefreshed,
    SyncFailed,
    Syncing,
    CancelLower,
    SyncTip,
    // card labels + tooltips
    BuiltLabel,
    IntelLabel,
    Today,
    DAgo,
    UpdateTip,
    // footer
    InstallDepsTip,
    FolderTitle,
    LangLabel,
    // update check (menu-only): version indicator + "update available" modal
    UpdateAvailable,
    UpdateTitle,
    UpdateBody, // parameterized: a single `{}` is filled with the new version tag
    UpdateWarn,
    UpdateLater,
}

/// Localized display for a build STAGE name (the Python log text, already uppercased + ASCII). We
/// map the English stage to Russian rather than pass Cyrillic through the ASCII whitelist. Prefix
/// match so truncated / suffixed variants ("GRASS: BUILD ...", "INSTALL PACKAGES (...)") still hit.
/// None => keep the (English) text as-is.
pub fn stage_ru(lang: Lang, en_upper: &str) -> Option<&'static str> {
    if lang == Lang::En {
        return None;
    }
    let s = en_upper.trim();
    let ru = if s.starts_with("CHECK DATASET") {
        "ПРОВЕРКА ДАННЫХ"
    } else if s.starts_with("EXTRACT DATASET") {
        "РАСПАКОВКА (ГЕО + ТЕКСТУРЫ)"
    } else if s.starts_with("EXTRACT GRASS") {
        "РАСПАКОВКА ТРАВЫ"
    } else if s.starts_with("EXTRACT LIGHTS") {
        "РАСПАКОВКА СВЕТА"
    } else if s.starts_with("BAKE LIGHTING") {
        "ЗАПЕКАНИЕ СВЕТА (GPU)"
    } else if s.starts_with("ASSEMBLE PACK") {
        "СБОРКА ПАКЕТА"
    } else if s.starts_with("GRASS") {
        "ТРАВА"
    } else if s.starts_with("GAMEPLAY ZONES") {
        "ИГРОВЫЕ ЗОНЫ"
    } else if s.starts_with("ITEM ICONS") {
        "ИКОНКИ ПРЕДМЕТОВ"
    } else if s.starts_with("NAV") {
        "НАВИГАЦИЯ"
    } else if s.starts_with("STAMP") {
        "ОТПЕЧАТОК ИГРЫ"
    } else if s.starts_with("CREATE VIRTUAL") {
        "СОЗДАНИЕ ОКРУЖЕНИЯ"
    } else if s.starts_with("INSTALL PACKAGES") {
        "УСТАНОВКА ПАКЕТОВ"
    } else if s.starts_with("VERIFY") {
        "ПРОВЕРКА"
    } else {
        return None;
    };
    Some(ru)
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
        Damaged => ["DAMAGED - REBUILD", "ПОВРЕЖДЁН - ПЕРЕСОБРАТЬ"],
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
        FirstBuildHint => [
            "First BUILD of a map runs a one-time ~1-6 GB extraction - close the game first.",
            "Первая СБОРКА карты запускает однократную распаковку ~1-6 ГБ - сначала закройте игру.",
        ],
        LanguageTip => ["Language / Язык (override auto-detect)", "Язык / Language (переопределить)"],
        ProcessInBackground => ["Process in background", "Обрабатывать в фоне"],
        ProcessInBackgroundTip => [
            "Builds keep running even if you close Atlas - reopen it later to see the progress or the finished map.",
            "Сборка продолжается, даже если закрыть Atlas - откройте его позже, чтобы увидеть прогресс или готовую карту.",
        ],
        BuildDeps => ["BUILD DEPS", "ЗАВИСИМОСТИ"],
        DepsReady => ["ready", "готово"],
        DepsMissing => [
            "Python packages missing (UnityPy) - required to build maps",
            "Не хватает пакетов Python (UnityPy) — нужны для сборки карт",
        ],
        InstallDeps => ["INSTALL DEPS", "УСТАНОВИТЬ"],
        BuildNeedsSetup => [
            "Install the build deps and set GAME INSTALL (below) first",
            "Сначала установите зависимости и укажите GAME INSTALL (ниже)",
        ],
        Installing => ["installing\u{2026}", "установка\u{2026}"],
        InstallingDeps => ["INSTALLING DEPENDENCIES", "УСТАНОВКА ЗАВИСИМОСТЕЙ"],
        Done => ["DONE", "ГОТОВО"],
        Failed => ["FAILED", "ОШИБКА"],
        Close => ["CLOSE", "ЗАКРЫТЬ"],
        Cancel => ["CANCEL", "ОТМЕНА"],
        ShowLog => ["SHOW LOG", "ПОКАЗАТЬ ЛОГ"],
        HideLog => ["HIDE LOG", "СКРЫТЬ ЛОГ"],
        CopyLog => ["COPY LOG", "КОПИРОВАТЬ ЛОГ"],
        BuildFailed => ["BUILD FAILED", "СБОРКА НЕ УДАЛАСЬ"],
        BuildComplete => ["BUILD COMPLETE", "СБОРКА ЗАВЕРШЕНА"],
        DepsDone => ["DEPENDENCIES INSTALLED", "ЗАВИСИМОСТИ УСТАНОВЛЕНЫ"],
        Starting => ["STARTING", "ЗАПУСК"],
        EstimatedTime => ["ESTIMATED TIME", "ОЦЕНКА ВРЕМЕНИ"],
        IntelRefreshed => ["intel refreshed", "данные обновлены"],
        SyncFailed => ["sync FAILED (see log)", "ошибка обновления (см. лог)"],
        Syncing => ["syncing\u{2026}", "обновление\u{2026}"],
        CancelLower => ["cancel", "отмена"],
        SyncTip => [
            "re-pull loot values, tasks and item icons from tarkov.dev (network)",
            "заново загрузить цены, задачи и иконки с tarkov.dev (сеть)",
        ],
        BuiltLabel => ["built", "собран"],
        IntelLabel => ["intel", "данные"],
        Today => ["today", "сегодня"],
        DAgo => ["d ago", "д назад"],
        UpdateTip => [
            "game files changed since this pack was built - run the pipeline again (data may be out of date)",
            "файлы игры изменились после сборки этого пакета — запустите сборку заново (данные могли устареть)",
        ],
        InstallDepsTip => [
            "creates a local venv and pip-installs UnityPy, numpy and Pillow",
            "создаёт локальный venv и ставит UnityPy, numpy и Pillow",
        ],
        FolderTitle => ["Choose a folder for extracted map assets", "Выберите папку для распакованных данных карт"],
        LangLabel => ["LANG", "ЯЗЫК"],
        UpdateAvailable => ["update available", "доступно обновление"],
        UpdateTitle => ["UPDATE AVAILABLE", "ДОСТУПНО ОБНОВЛЕНИЕ"],
        // `{}` = the new version tag (e.g. v0.1.0-15061f1); filled with format! at the call site.
        UpdateBody => [
            "A new version ({}) of Atlas is available.",
            "Доступна новая версия Atlas ({}).",
        ],
        UpdateWarn => [
            "Atlas may not work as intended if you don't update.",
            "Без обновления Atlas может работать некорректно.",
        ],
        UpdateLater => ["LATER", "ПОЗЖЕ"],
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
