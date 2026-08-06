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
    PlayNeedsSync,
    RebuildPack,
    DeleteConfirmTip,
    TickGrassTip,
    TickLightTip,
    TickZonesTip,
    TickIconsTip,
    ConfigSaveFailed,
    OverlayFpsCap,
    OverlayFpsCapTip,
    LinkHealth,
    LinkWatcher,
    LinkGameDir,
    LinkLogsDir,
    LinkAppLog,
    LinkShotsDir,
    LinkEvents,
    LinkNoEventsHint,
    PlayBusyBuilding,
    LanguageTip,
    ProcessInBackground,
    ProcessInBackgroundTip,
    ForceCpuProcess,
    ForceCpuProcessTip,
    ScreenshotLocate,
    ScreenshotLocateTip,
    OverlayEnable,
    OverlayEnableTip,
    Settings,
    SettingsTabOverlay,
    SettingsTabLive,
    SettingsTabGeneral,
    OverlayKeepAbove,
    OverlayBorderlessShown,
    OverlayPresentationLabel,
    OverlayModeWindowed,
    OverlayModeWindowedTip,
    OverlayModeBorderless,
    OverlayModeBorderlessTip,
    OverlayModeTransparent,
    OverlayModeTransparentTip,
    OverlayNextLaunchNote,
    EspModeLabel,
    EspModeTip,
    OverlayPanelSize,
    OverlayPanelPos,
    OverlayPerf,
    OverlayIdleHidden,
    OverlayIdleHiddenTip,
    OverlayOpenOnShot,
    OverlayOpenOnShotTip,
    OverlayReturnFocus,
    OverlayReturnFocusTip,
    DeleteShots,
    DeleteShotsTip,
    OverlayBorderlessNote,
    LiveLinkNote,
    BackToTarkov,
    OverlayReopenHint,
    OverlayExitWindow,
    OverlayExitHotkey,
    OverlayExitHint,
    UnbuiltNotProcessed,
    UnbuiltBody,
    UnbuiltProcess,
    UnbuiltProcessTip,
    UnbuiltCancel,
    Quality,
    QualityTip,
    QualityLowSum,
    QualityMediumSum,
    QualityHighSum,
    QualityUltraSum,
    QualityCustomSum,
    TexQuality,
    TexQualityTip,
    TexQualityFull,
    TexQualityHalf,
    TexQualityQuarter,
    TexQualityNote,
    BuildDeps,
    DepsReady,
    DepsMissing,
    InstallDeps,
    BuildNeedsSetup,
    BuildKitMissing,
    DepsFirst,
    SetGameFirst,
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
        SyncNow => ["SYNC NOW", "СИНХРОНИЗАЦИЯ"],
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
        // The MAP-ROW action: re-run this pack's build against newer game files.
        // Distinct from K::Update (new Atlas version) and K::SyncNow (tarkov.dev data)
        // — all three used to read "ОБНОВИТЬ" in RU.
        RebuildPack => ["REBUILD", "ПЕРЕСОБРАТЬ"],
        Confirm => ["CONFIRM", "ПОДТВЕРДИТЬ"],
        LinkHealth => ["LIVE LINK STATUS", "СОСТОЯНИЕ СВЯЗИ"],
        LinkWatcher => ["watcher running", "наблюдатель работает"],
        LinkGameDir => ["game install found", "установка игры найдена"],
        LinkLogsDir => ["log folder found", "папка логов найдена"],
        LinkAppLog => ["reading the game log", "читаем лог игры"],
        LinkShotsDir => ["screenshots folder found", "папка скриншотов найдена"],
        LinkEvents => ["recognized log events:", "распознано событий:"],
        LinkNoEventsHint => [
            "Reading the log but recognizing nothing - if you have played a raid since Atlas started, the game log format may have changed. Please report it.",
            "Лог читается, но ничего не распознано - если вы играли рейд после запуска Atlas, формат лога мог измениться. Сообщите об этом.",
        ],
        TickGrassTip => [
            "Procedural grass field (grass.bin). The map's own bushes and foliage are scene geometry and render regardless of this.",
            "Процедурное поле травы (grass.bin). Собственная растительность карты - это геометрия сцены и рисуется в любом случае.",
        ],
        TickLightTip => [
            "Baked SH irradiance volume. Without it the map falls back to flat realtime lighting.",
            "Запечённый SH-объём освещения. Без него карта использует плоское освещение.",
        ],
        TickZonesTip => [
            "Game-file gamedata.json: extracts, spawns, doors, loot containers and quest zones.",
            "gamedata.json из файлов игры: выходы, спавны, двери, контейнеры и квестовые зоны.",
        ],
        TickIconsTip => [
            "Cached item icons for the loot cards.",
            "Кэш иконок предметов для карточек лута.",
        ],
        ConfigSaveFailed => [
            "Settings could not be saved (read-only folder?) - this change will not survive a restart.",
            "Не удалось сохранить настройки (папка только для чтения?) - изменение не сохранится после перезапуска.",
        ],
        OverlayFpsCap => ["fps cap (0 = uncapped)", "лимит fps (0 = без лимита)"],
        OverlayFpsCapTip => [
            "Frame-rate ceiling while the overlay is up, so Atlas leaves the game headroom on the shared GPU. 0 removes the cap.",
            "Ограничение частоты кадров, пока оверлей открыт, чтобы оставить ресурс GPU игре. 0 - без ограничения.",
        ],
        DeleteConfirmTip => [
            "Deletes this map's built pack from disk. Rebuilding it takes the full processing time again.",
            "Удаляет собранный пакет этой карты с диска. Повторная сборка займёт всё время обработки заново.",
        ],
        TickLight => ["light", "свет"],
        // NOT "does this map show grass": the map's own foliage is ordinary scene geometry and
        // always renders. This flag is the PROCEDURAL grass field (grass.bin density grids), which
        // only some packs carry - the old bare "grass" label read as a missing-vegetation warning
        // on maps that visibly have plenty.
        TickGrass => ["grass sim", "трава (симуляция)"],
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
        PlayNeedsSync => [
            "No tarkov.dev data yet - maps open, but loot prices and task intel stay empty. Use SYNC NOW (top) when you are online.",
            "Данных tarkov.dev пока нет - карты откроются, но цены лута и данные квестов будут пустыми. Нажмите SYNC NOW (сверху), когда будете онлайн.",
        ],
        PlayBusyBuilding => [
            "A map is building - the lighting bake needs the GPU. PLAY unlocks when it finishes.",
            "Идёт сборка карты - запеканию света нужен GPU. PLAY разблокируется после её окончания.",
        ],
        LanguageTip => ["Language / Язык (override auto-detect)", "Язык / Language (переопределить)"],
        ProcessInBackground => ["Process in background", "Обрабатывать в фоне"],
        ProcessInBackgroundTip => [
            "Builds keep running even if you close Atlas - reopen it later to see the progress or the finished map.",
            "Сборка продолжается, даже если закрыть Atlas - откройте его позже, чтобы увидеть прогресс или готовую карту.",
        ],
        ForceCpuProcess => [
            "Force CPU processing (no GPU bakes)",
            "Принудительная обработка на CPU (без GPU)",
        ],
        ForceCpuProcessTip => [
            "Map builds bake lighting and terrain on the CPU instead of the GPU. Slower, but avoids GPU driver crashes/hangs during processing. Applies to the next build.",
            "Сборка карт запекает свет и террейн на CPU вместо GPU. Медленнее, но обходит сбои/зависания драйвера GPU при обработке. Действует со следующей сборки.",
        ],
        ScreenshotLocate => [
            "Screenshot to locate current position",
            "Скриншот определяет текущую позицию",
        ],
        ScreenshotLocateTip => [
            "Take a screenshot in raid and Atlas moves the camera to exactly where you are standing, looking the way you look - EFT writes your position and view angle into the screenshot filename.

Check Settings > Controls in Tarkov for your screenshot key. You may need to REBIND it: the default can collide with the Windows Snipping Tool or another screenshot app, which grabs the key first so EFT never writes the file.",
            "Сделайте скриншот в рейде, и Atlas переместит камеру точно туда, где вы стоите, и в ту же сторону - EFT записывает позицию и угол обзора в имя файла скриншота.

Проверьте клавишу скриншота в Настройках > Управление в Таркове. Возможно, её придётся ПЕРЕНАЗНАЧИТЬ: стандартная клавиша может конфликтовать с «Ножницами» Windows или другой программой скриншотов, которая перехватывает нажатие, и EFT не создаёт файл.",
        ],
        OverlayEnable => [
            "Overlay mode (your screenshot key opens the map over the game)",
            "Режим оверлея (клавиша скриншота открывает карту поверх игры)",
        ],
        OverlayEnableTip => [
            "Take an IN-GAME screenshot and Atlas rises over the game as a borderless always-on-top panel, standing exactly where you are. The big BACK TO TARKOV button (or ~ while Atlas is focused) dismisses it and hands focus back to the game. WASD and the mouse fly the map as usual.

Tarkov MUST be running in BORDERLESS (not exclusive fullscreen) - no window can appear over exclusive fullscreen.

Atlas only reads files the game already wrote; it never touches the game process. Overlaying a game is still your call, so this is off by default.",
            "Сделайте скриншот В ИГРЕ - и Atlas поднимется поверх игры панелью без рамки, точно там, где вы стоите. Большая кнопка ВЕРНУТЬСЯ В ТАРКОВ (или ~, пока Atlas в фокусе) убирает её и возвращает фокус игре. WASD и мышь управляют картой как обычно.

Тарков ДОЛЖЕН работать в РЕЖИМЕ ОКНА БЕЗ РАМКИ (не в полноэкранном) - поверх полноэкранного режима окно показать нельзя.

Atlas читает только файлы, которые игра уже записала, и не трогает процесс игры. Решение использовать оверлей остаётся за вами, поэтому по умолчанию он выключен.",
        ],
        Settings => ["SETTINGS", "НАСТРОЙКИ"],
        SettingsTabOverlay => ["Overlay", "Оверлей"],
        SettingsTabLive => ["Live link", "Связь с игрой"],
        SettingsTabGeneral => ["General", "Общие"],
        OverlayKeepAbove => ["Keep above the game", "Держать поверх игры"],
        OverlayBorderlessShown => ["Borderless while shown", "Без рамки, когда показан"],
        OverlayPresentationLabel => ["Window mode", "Режим окна"],
        OverlayModeWindowed => ["Windowed", "Оконный"],
        OverlayModeWindowedTip => [
            "A normal window. The game covers it when it takes focus — right for a second monitor.",
            "Обычное окно. Игра перекрывает его при фокусе — подходит для второго монитора.",
        ],
        OverlayModeBorderless => ["Borderless panel", "Панель без рамки"],
        OverlayModeBorderlessTip => [
            "An opaque panel held above the game. The default.",
            "Непрозрачная панель поверх игры. По умолчанию.",
        ],
        OverlayModeTransparent => ["Transparent", "Прозрачный"],
        OverlayModeTransparentTip => [
            "The game shows through wherever Atlas draws nothing. Fixed size, always on top.",
            "Игра видна там, где Atlas ничего не рисует. Фиксированный размер, всегда сверху.",
        ],
        OverlayNextLaunchNote => [
            "applies when a map is next opened",
            "применяется при следующем открытии карты",
        ],
        EspModeLabel => ["ESP mode — markers only, no map", "Режим ESP — только маркеры, без карты"],
        EspModeTip => [
            "Skips loading and drawing the 3D world entirely and shows routes, loot and markers as an overlay matched to the game camera. Loads in seconds. Applies when a map is next opened.",
            "Пропускает загрузку и отрисовку 3D-мира и показывает маршруты, лут и маркеры как оверлей, совмещённый с камерой игры. Загружается за секунды. Применяется при следующем открытии карты.",
        ],
        OverlayPanelSize => [
            "Panel size (fraction of monitor)",
            "Размер панели (доля монитора)",
        ],
        OverlayPanelPos => [
            "Position (0 = left/top, 1 = right/bottom)",
            "Положение (0 = слева/сверху, 1 = справа/снизу)",
        ],
        OverlayPerf => [
            "Performance while the game has focus",
            "Производительность, пока игра в фокусе",
        ],
        OverlayIdleHidden => ["Idle when hidden", "Простаивать, когда скрыт"],
        OverlayIdleHiddenTip => [
            "Stop redrawing while the overlay is dismissed, so Atlas costs the game nothing.",
            "Не перерисовывать карту, пока оверлей скрыт, - Atlas не отнимает ресурсы у игры.",
        ],
        OverlayOpenOnShot => [
            "Open on screenshot (recommended)",
            "Открывать по скриншоту (рекомендуется)",
        ],
        OverlayOpenOnShotTip => [
            "Press your in-game SCREENSHOT key in a raid and Atlas opens here, standing exactly where you are. Tarkov takes its own screenshot - Atlas never intercepts or injects a key; it only reads the file EFT writes.",
            "Нажмите клавишу СКРИНШОТА в рейде - и Atlas откроется точно там, где вы стоите. Тарков сам делает скриншот: Atlas не перехватывает и не эмулирует клавиши, а только читает файл, который записывает EFT.",
        ],
        OverlayReturnFocus => [
            "Give focus back to the game on close",
            "Возвращать фокус игре при закрытии",
        ],
        OverlayReturnFocusTip => [
            "Dismissing the overlay minimises Atlas, which makes Windows activate the game behind it - so your keys go back to Tarkov. Off = Atlas stays on the desktop.",
            "При закрытии оверлея Atlas сворачивается, и Windows активирует игру позади него - клавиши снова идут в Тарков. Выкл = Atlas остаётся на рабочем столе.",
        ],
        DeleteShots => [
            "Delete processed screenshots (recommended)",
            "Удалять обработанные скриншоты (рекомендуется)",
        ],
        DeleteShotsTip => [
            "EFT writes a full-resolution PNG every time you press the screenshot key and never cleans them up. Atlas deletes ONLY the screenshots it has already read a position out of; anything it could not parse is left alone.",
            "EFT записывает полноразмерный PNG при каждом нажатии клавиши скриншота и никогда их не удаляет. Atlas удаляет ТОЛЬКО те скриншоты, из которых уже прочитал позицию; всё, что не удалось разобрать, остаётся на месте.",
        ],
        OverlayBorderlessNote => [
            "Works with Tarkov in WINDOWED or BORDERLESS - nothing can draw over exclusive fullscreen.",
            "Работает, если Тарков запущен В ОКНЕ или БЕЗ РАМКИ - поверх полноэкранного режима ничего показать нельзя.",
        ],
        LiveLinkNote => [
            "Atlas follows your raid from the game's own logs and turns each in-raid screenshot into a position fix. It only reads files the game already wrote.",
            "Atlas следит за рейдом по логам самой игры и превращает каждый скриншот в рейде в отметку позиции. Он читает только файлы, которые игра уже записала.",
        ],
        BackToTarkov => ["\u{21a9}  BACK TO TARKOV", "\u{21a9}  ВЕРНУТЬСЯ В ТАРКОВ"],
        OverlayReopenHint => [
            "Take an in-game screenshot to reopen the overlay",
            "Чтобы снова открыть оверлей, сделайте скриншот в игре",
        ],
        OverlayExitWindow => [
            "EXIT OVERLAY / WINDOW MODE",
            "ВЫЙТИ ИЗ ОВЕРЛЕЯ / ОКОННЫЙ РЕЖИМ",
        ],
        OverlayExitHotkey => [
            "Hide-overlay hotkey",
            "Клавиша скрытия оверлея",
        ],
        OverlayExitHint => [
            "Hides the overlay and returns focus to the game",
            "Скрывает оверлей и возвращает фокус игре",
        ],
        UnbuiltNotProcessed => ["is not processed yet", "ещё не обработана"],
        UnbuiltBody => [
            "Your raid is on this map, but Atlas has no pack for it - what you are looking at is NOT where you are.",
            "Ваш рейд идёт на этой карте, но у Atlas нет её сборки - то, что вы видите, НЕ то место, где вы находитесь.",
        ],
        UnbuiltProcess => ["PROCESS THIS MAP", "ОБРАБОТАТЬ КАРТУ"],
        UnbuiltProcessTip => [
            "Runs the full build for this map. It takes several minutes and uses the CPU/GPU hard - the bake stays off the GPU while Atlas is rendering, but the game will still feel it. You can keep playing; the build survives closing Atlas.",
            "Запускает полную сборку карты. Это занимает несколько минут и сильно нагружает CPU/GPU - запекание не трогает GPU, пока Atlas рисует, но игра всё равно это почувствует. Можно продолжать играть: сборка переживёт закрытие Atlas.",
        ],
        UnbuiltCancel => ["Cancel", "Отмена"],
        Quality => ["Quality preset", "Пресет качества"],
        QualityTip => [
            "Sets the options that actually cost performance, measured on this build. Chosen here rather than in-raid because texture quality is applied while a map loads.",
            "Задаёт параметры, которые действительно влияют на производительность. Выбирается здесь, так как качество текстур применяется при загрузке карты.",
        ],
        // EN MUST stay byte-identical to `QualityPreset::summary()` — these copies had drifted
        // until Ultra advertised "~2% slower" for a preset measured at ~30%, i.e. off by ~15x, and
        // the menu is where the choice is actually made. `quality_summaries_match_render_source`
        // (below) fails the build if they part again.
        QualityLowSum => [
            "~30% faster • ~1.6 GB VRAM — no foliage, shadows or bloom",
            "~30% быстрее • ~1.6 ГБ — без травы, теней и свечения",
        ],
        QualityMediumSum => [
            "~25% faster • ~2.2 GB VRAM — thinned foliage to 150 m, no shadows",
            "~25% быстрее • ~2.2 ГБ — трава прорежена до 150 м, без теней",
        ],
        QualityHighSum => [
            "baseline • ~2.3 GB VRAM — the shipped look",
            "базовый • ~2.3 ГБ — стандартный вид",
        ],
        QualityUltraSum => [
            "~30% slower • ~4.5 GB VRAM — full-res textures, SSAO + volumetric sun shafts",
            "~30% медленнее • ~4.5 ГБ — текстуры полного разрешения, SSAO и объёмные лучи солнца",
        ],
        QualityCustomSum => [
            "your own mix — tune it in-raid under Graphics",
            "ваш набор — настройте в рейде",
        ],
        TexQuality => ["Texture quality", "Качество текстур"],
        TexQualityTip => [
            "How sharply map textures load. Half keeps the map perfectly readable and reclaims most of the texture memory - the right choice when Tarkov and Atlas share the GPU. Textures already at or below 128 px are never reduced.",
            "Насколько чёткими загружаются текстуры карты. «Половина» оставляет карту полностью читаемой и освобождает большую часть текстурной памяти - правильный выбор, когда Тарков и Atlas делят GPU. Текстуры размером 128 пикселей и меньше не уменьшаются.",
        ],
        TexQualityFull => ["Full", "Полное"],
        TexQualityHalf => ["Half (recommended with overlay)", "Половина (рекомендуется с оверлеем)"],
        TexQualityQuarter => ["Quarter", "Четверть"],
        TexQualityNote => [
            "Applies when a map is (re)loaded.",
            "Применяется при следующей загрузке карты.",
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
        BuildKitMissing => [
            "This viewer-only bundle has no build kit - use the full bundle to build maps",
            "В этой сборке нет комплекта для сборки карт — используйте полную версию",
        ],
        DepsFirst => [
            "Install the build deps (below) first",
            "Сначала установите зависимости (ниже)",
        ],
        SetGameFirst => [
            "Set GAME INSTALL to your Escape from Tarkov folder (below) first",
            "Сначала укажите папку установки Escape from Tarkov (GAME INSTALL, ниже)",
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

#[cfg(test)]
mod quality_summary_tests {
    use super::*;

    /// The menu's quality tooltips and `QualityPreset::summary()` are two copies of the SAME
    /// measured numbers, and they silently drifted: the menu advertised Ultra as "~2% slower"
    /// long after volumetric shafts made it ~30%, so the one place users actually choose a preset
    /// was off by a factor of ~15. Nothing detected it because nothing compared them. This does.
    #[test]
    fn quality_summaries_match_render_source() {
        for p in crate::render::QualityPreset::ALL {
            let key = match p {
                crate::render::QualityPreset::Low => K::QualityLowSum,
                crate::render::QualityPreset::Medium => K::QualityMediumSum,
                crate::render::QualityPreset::High => K::QualityHighSum,
                crate::render::QualityPreset::Ultra => K::QualityUltraSum,
                crate::render::QualityPreset::Custom => continue, // free-form copy, no numbers
            };
            assert_eq!(
                t(Lang::En, key),
                p.summary(),
                "menu tooltip for {:?} has drifted from QualityPreset::summary()",
                p
            );
        }
    }
}
