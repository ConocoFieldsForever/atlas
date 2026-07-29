# Game Data Sources for Atlas Live Link

What else Atlas can pull from EFT's own files and logs, beyond what we already consume.
Everything here is **passive file reading only** — no process memory, no injection, no network interception.

Verification legend:
- **[local]** — pattern confirmed in this machine's logs, game version 1.0.6.5.46221 (and 1.0.5/1.0.6 folders), July 2026.
- **[TM]** — pattern taken from TarkovMonitor source (`TarkovMonitor/GameWatcher.cs`, `LogMessageTypes.cs`, `MainBlazorUI.cs`, github.com/the-hideout/TarkovMonitor) but not observed locally.
- Most rows are both.

Paths use `<EFT_INSTALL>` (game install root) and `<LOG_DIR>` = `<EFT_INSTALL>\Logs\log_<yyyy.MM.dd_H-mm-ss>_<version>\`. Each channel file is named `<timestamp>_<version> <channel>_000.log`.

Line format for all channels: `YYYY-MM-DD HH:MM:SS.mmm|<version>|<level>|<channel>|<message>`, optionally followed by a multi-line JSON block starting at column 0 (`{` ... `}`). TarkovMonitor's master regex (verified to work on 1.0.6.5 lines):

```
(?<date>^\d{4}-\d{2}-\d{2}) (?<time>\d{2}:\d{2}:\d{2}\.\d{3})(?<tzoffset> [+-]\d{2}:\d{2})?\|(?<message>.+$)\s*(?<json>^{[\s\S]+?^})?
```

---

## 1. What we already consume (baseline)

| Source | Signal |
|---|---|
| `application_*.log` | `scene preset path:maps/<bundle>.bundle` → which map is loading |
| `push-notifications_*.log` | `ChatMessageReceived` type 10/11/12 (+templateId) → task started/failed/finished; `GroupMatchRaidSettings.raidSettings.side` → local PMC/Scav when present; `UserMatchOver` → raid end |
| `Documents\Escape From Tarkov\Screenshots` | filename → position + view quaternion |
| Build-time UnityPy extraction + tarkov.dev API | geometry, doors, interactables, lights, tasks/loot/icons |

## 2. Log channel inventory (1.0.6.5, verified locally)

Channels that appear in `<LOG_DIR>` (count = folders they appeared in across ~260 local sessions):

| Channel | Always present | Content (sampled) |
|---|---|---|
| `application` | yes | The gold mine: session mode, profile select, map/raid lifecycle, settings JSON dumps, GC noise |
| `output` | yes | Unity player log; superset of application plus stack traces (16 MB/session — avoid tailing) |
| `push-notifications` | yes | `Got notification \| <Type>` + JSON blocks (full inventory in §3.2) |
| `backend` | yes | Every HTTPS request/response **URL + timing only, no bodies** |
| `errors` | yes | Unity/NLog exceptions |
| `files-checker` | yes | "Consistency ensurance" runs before each matching |
| `network-connection` | in-raid | `Connect/Disconnect (address: ip:port)` + end-of-raid `Statistics (rtt, lose, sent, received)` |
| `network-messages` | in-raid | one metrics line (`rpi:\|rwi:\|...`) every 30 s while connected |
| `spatial-audio` | yes | audio init lines only |
| `aiData` / `aiErrors` | PVE/local raids | bot pathing errors (Error level only), no positions of value |
| `inventory`, `insurance`, `player`, `health-system`, `maperrors`, `objectPool`, `speaker` | rare | error-level only, nothing structured |

`<EFT_INSTALL>\Logging.config` lists ~50 channels with per-channel `minLevel`. Channels like `spawns`, `exfiltration`, `quests`, `traffic`, `ping`, `seasons` exist but are set to `Error` so they never produce useful files. **Logging.config is checksummed in `<EFT_INSTALL>\ConsistencyInfo`** — editing it to raise verbosity is detectable/reverted and violates our passive constraint. Consciously skipped (§5).

## 3. Discoverable live signals

### 3.1 `application_*.log` (we already tail this file)

| # | Trigger line (verified sample) | Data carried | Latency | Atlas feature idea |
|---|---|---|---|---|
| A1 | `Session mode: Pve` / `Session mode: Regular` **[local, TM]** | PVE vs PVP session | at login / profile switch | Auto-select PVE/PVP task-state profile; label the HUD |
| A2 | `CompleteSelectedProfile ProfileId:6a1330...b645 AccountId:11243037` (also `PrepareSelectedProfileLocally ...`) **[local, TM]** | profile id + account id (PMC and scav have distinct ids) | at login / profile switch | Multi-account/profile separation of task state; TM also uses re-select mid-session as a raid-ended fallback |
| A3 | `Matching with group id: ` **[local]** | matching began | instant on queue start | "Queuing…" state on HUD |
| A4 | `MatchingCompleted:27.1 real:33.98 diff:...` **[local, TM]** | queue time (seconds, `real:` value; `,` decimal in some locales) | instant when server found | Queue-time toast + history stats |
| A5 | `TRACE-NetworkGameCreate profileStatus: 'Profileid: ..., Status: Busy, RaidMode: Online, Ip: 173.201.39.87, Port: 17011, Location: TarkovStreets, Sid: US-STL01G108_..., GameMode: deathmatch, shortId: LYCQYF'` **[local, TM]** | authoritative map `nameId`, raid shortId, server IP/port, region code (Sid prefix), online/offline | right after matching | Confirm/auto-switch loaded map (more reliable than bundle name); raid ID + server region badge; dedupe reconnects by shortId (TM does) |
| A6 | `LocationLoaded:34.47 real:44.36 diff:...` **[local, TM]** | map load time | after load | Load-time stat |
| A7 | `GameStarting:73.66(27.64) real:86.85(...)` **[local, TM]** | PMC countdown began | instant | "Raid starting" state; PMC vs scav heuristic (TM: gap >3 s between Starting and Started ⇒ PMC) |
| A8 | `GameStarted:85.06(11.39) real:98.9(...)` **[local, TM]** | raid clock zero | instant | **Raid timer HUD** (count up / down from map duration); run-through timer; scav-cooldown timer on end |
| A9 | `PlayerSpawnEvent:...`, `GameSpawned:`, `GameRunned:`, `GamePrepared:`, `GamePooled:`, `GameCreated:` **[local]** | fine-grained load phases | during load | Loading progress indicator |
| A10 | `[Transit] `6a1330...` Count:0, EventPlayer:False` **[local]** | transit count for profile at raid start | at spawn | Transit chain indicator (Count>0 ⇒ arrived via transit) |
| A11 | `Reason:PacketsQueue, Position:(-24.995, 21.394, 109.302), SpeedLimit:4.6, CurrentState:Run, StrengthSummaryLevel:1, WalkSpeedLimit:1` **[local]** | **in-raid player world position** + move state, no screenshot needed | sporadic (netcode speed-limit events, a few per raid) | Opportunistic position pings between screenshots — free extra breadcrumbs on the map trail |
| A12 | `Network game matching aborted` / `...cancelled` **[TM]** | user cancelled queue | instant | Reset "Queuing…" state |
| A13 | `Init: pstrGameVersion: Escape from Tarkov 1.0.6.5.46221, uiAddress: ...` (BattlEye re-init) **[local, TM]** | back at main menu | on menu return | "Exited post-raid menus" → clear raid HUD (TM fires ExitedPostRaidMenus) |
| A14 | `Game settings:` + JSON block (`"FieldOfView": 50`, `"Language": "en"`, `"StreamerModeEnabled": false`, ...) **[local]** | client settings incl. **FOV** | at boot & settings change | Use the player's real FOV for the view cone drawn from screenshot quaternions (we currently have to guess) |
| A15 | `Control settings:` + JSON block (`keyBindings` array) **[local, TM]** | full keybinds; TM checks `MakeScreenshot` binding | at boot | Warn user if screenshot key unbound/`SysReq` (position updates silently dead otherwise) — TM does exactly this |
| A16 | `Sound settings:` / `PostFx settings:` + JSON **[local]** | audio/postfx config | at boot | (low value) |

### 3.2 `push-notifications_*.log` (we already tail this file)

Full inventory of `Got notification | <Type>` seen locally in July 2026 with counts:
`UserConfirmed` 113, `GroupMatchRaidNotReady` 108, `GroupMatchRaidReady` 105, `GroupMatchStartGame` 89, `ChatMessageReceived` 71, `GroupMatchRaidSettings` 55, `GroupMatchWasRemoved` 19, `GroupMatchInviteSend` 19, `GroupMatchInviteAccept` 17, `UserMatchCreated` 12, `UserMatchOver` 5, `GroupMatchInviteDecline` 5, `GroupMatchUserLeave` 4, `RagfairOfferSold` 3, `GroupMatchLeaderChanged` 2, `GroupMatchInviteCancel` 2, `GroupMatchInviteExpired` 1, `GroupMatchAbort` 1, `FriendsListAccept` 1.

| # | Trigger | Data carried (verified JSON fields) | Latency | Atlas feature idea |
|---|---|---|---|---|
| N1 | `Got notification \| UserConfirmed` **[local]** | `location` ("TarkovStreets"), `raidMode` ("Online"), `mode` ("deathmatch"), `shortId`, `sid` (server/region string), `ip`/`port`, `status:"Busy"`, `profileid` | earliest server-assignment signal, before load finishes | Earliest map pre-switch + raid id + region badge; complements A5 |
| N2 | `Got notification \| UserMatchCreated` **[local]** | `blockExitButton`, eventId | when match created | minor: "match locked" state |
| N3 | `Got notification \| UserMatchOver` **[local, TM]** *(already used)* | `location`, `status:"Free"`, `shortId` (may be null) — **no survived/died flag** | raid end | (already: raid end) |
| N4 | `Got notification \| GroupMatchRaidReady` **[local, TM]** | `extendedProfile`: `Info.Nickname`, `Side`, `Level`, `MemberCategory`, `SavageNickname`, per-body-part `Health`, `Equipment`/`Customization` (full loadout item tree) **[TM parses Info only]** | lobby, per member | **Group roster panel**: member names, sides, levels, ready states, even loadout summary |
| N5 | `Got notification \| GroupMatchRaidNotReady` **[local]** (TM ignores) | member un-readied | lobby | roster ready-state toggle off |
| N6 | `Got notification \| GroupMatchRaidSettings` **[local, TM]** | `raidSettings.location`, `timeVariant` (CURR/PAST), full `timeAndWeatherSettings` (cloudiness, rain, fog, wind, hourOfDay), `botSettings`, `side` ("Pmc"/"Savage"), `onlinePveRaidStates` per map | when settings are logged; absent in many solo sessions | Authoritative local PMC/scav side before spawn when present. Never substitute `GroupMatchRaidReady.extendedProfile.Info.Side`: that record describes a group member/other profile. Unknown must remain unknown. |
| N7 | `Got notification \| GroupMatchStartGame` **[local]** (TM ignores) | `groupId`, `estimate` (queue estimate, seconds) | group queue start | Queue ETA toast |
| N8 | `GroupMatchInviteAccept`/`InviteSend`/`InviteDecline`/`InviteCancel`/`InviteExpired`/`UserLeave` (`Nickname`)/`WasRemoved`/`LeaderChanged`/`Abort` **[local; TM handles Accept/UserLeave/WasRemoved]** | group membership churn | instant | Group member count / join-leave toasts |
| N9 | `ChatMessageReceived` `message.type == 4` + `templateId "5bdabfb886f7743e152e867e 0"` **[local, TM]** | flea sold: `systemData.buyerNickname`, `soldItem` (tpl id), `itemCount`; profit fields in attached items | instant | Flea sale toast (TM plays sound + stats). `templateId "5bdabfe486f7743e1665df6e 0"` = offer expired |
| N10 | `Got notification \| RagfairOfferSold` **[local]** (TM ignores — newer, cleaner than N9) | `offerId`, `handbookId`, `count` | instant | same as N9 without text-template parsing |
| N11 | `ChatMessageReceived` `type == 2` **[local, TM enum Insurance]** | insurance ack: `systemData.date/time/location` | post-raid | — |
| N12 | `ChatMessageReceived` `type == 8` **[local, TM enum InsuranceReturn]** | insured items returned: full `items.data[]` list (tpl ids, durability) | hours later | "Insurance back" toast with item icons (we already have tarkov.dev icons) |
| N13 | `ChatMessageReceived` `type == 15` **[local]** (TM ignores) | in-raid item transfer/BTR delivery: `items.data[]` with `SpawnedInSession` | post-raid | "Delivered items" toast |
| N14 | `ChatMessageReceived` `type == 13` **[TM enum TwitchDrop]** | Twitch drop | — | (skip) |

### 3.3 Other channels (not currently tailed)

| # | Source | Trigger | Data | Latency | Atlas feature idea |
|---|---|---|---|---|---|
| O1 | `network-connection_*.log` **[local]** | `Connect (address: 173.201.39.87:17011)` / `Enter to the 'Connected' state` / `Disconnect (address: ...)` | raid session bounds by server socket | instant | Robust in-raid/out-of-raid state even when app log is ambiguous (reconnects) |
| O2 | `network-connection_*.log` **[local]** | `Statistics (address: ..., rtt: 30.75, lose: 0, sent: 32233, received: 41001)` | RTT + packet loss for the raid | at disconnect | Post-raid ping/loss stat on raid summary |
| O3 | `network-messages_*.log` **[local]** | one `rpi:...\|rwi:...\|ui:...` line every 30 s while connected | liveness heartbeat | 30 s | "Still in raid" watchdog (detects crashed game vs. in-raid idle) |
| O4 | `backend_*.log` **[local]** | `---> Request HTTPS ... URL: https://gw-pve.escapefromtarkov.com/...` vs `gw-pvp.` | game mode from gateway host | first request after login | Instant PVE/PVP detection independent of A1 |
| O5 | `backend_*.log` **[local]** | `URL: .../client/match/local/start` and `/client/match/local/end` | local/PVE raid start/end markers | instant | PVE raid bounds (fires even when notifications are quiet); also `/client/match/group/*`, `/client/achievement/list`, `/client/weather` — **URLs only, bodies are never logged** |
| O6 | `files-checker_*.log` **[local]** | `Consistency ensurance is launched` | pre-matching integrity pass | ~1 s before queue | early "about to queue" hint (fires right before matching) |
| O7 | Screenshots dir **[local]** *(already used)* | `2026-07-24[21-39]_-135.39, 28.62, 86.99_0.00835, 0.92994, -0.02115, 0.36702_14.55 (0).png` | pos + quaternion (+ trailing number; menu screenshots carry no position: `2026-07-24[21-55]_6.41 (1).png`) | on keypress | (already) — but see A15 keybind check and A14 FOV |

### 3.4 Static / semi-live files under `<EFT_INSTALL>`

| File | Data | Atlas use |
|---|---|---|
| `ConsistencyInfo` (JSON) **[local]** | `"Version":"1.0.6.5.46221"` + every bundle path/size/checksum | **Game-update detector**: watch Version/checksums → prompt "map data stale, re-run extraction". Also a manifest to diff exactly *which* bundles changed (re-extract only those) |
| `Logging.config` (JSON) **[local]** | all 50 log channels + minLevel | know which channels can exist; do NOT edit (checksummed, see §5) |
| `EscapeFromTarkov_Data\app.info`, `boot.config` **[local]** | product name, `build-guid` | secondary update fingerprint |
| `EscapeFromTarkov_Data\StreamingAssets\` (`Acoustics`, `AudioBakeData`, `Culling_Data`, `Grass`, ...) **[local]** | baked culling/acoustics/grass data per map | potential build-time extraction inputs (e.g. Culling_Data for occlusion tuning) — not live signals |
| `EscapeFromTarkov_Data\ScriptingAssemblies.json`, `il2cpp_data`, `GameAssembly.dll` **[local]** | code metadata | already mined via `tools/il2cpp_explore.py` at build time |
| `EscapeFromTarkov_Data\resources.assets` → `TestBackendLocaleEn/Ru` **[local]** | exact client locale strings keyed by serialized exfil ids (`NW Exfil` → `Railway Exfil`, `E1` → `Stylobate Building Elevator`) | `extract_gamedata.py` stores `display_name_en/ru`; the viewer keeps the raw key for joins and never proximity-renames from community data |
| Registry `HKLM\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\EscapeFromTarkov` → `InstallLocation` (TM also checks `...\Steam App 3932890`) **[TM]** | install path | auto-discover `<EFT_INSTALL>` instead of configuring it |
| `Documents\Escape From Tarkov\` **[local]** | contains only `Screenshots` on this machine — no local profile cache exists | — |

Extract authority is the map logic scene itself: `ExfiltrationPoint` = PMC,
`ScavExfiltrationPoint` = Scav, `SharedExfiltrationPoint` = both, and
`SecretExfiltrationPoint` = secret/unknown-side. The component's `Settings.Name` is the locale
key and its `BoxCollider` is the selectable footprint. IL2CPP metadata proves `CarExtraction`
derives from `ExfiltrationSubscriber`; it animates a car subscribed to a real extraction point
and is not itself a selectable extract.

---

## 4. TOP 5 recommendations (impact × ease)

All of these read files we already know how to tail; 1–3 need zero new file watchers.

### 1. Raid lifecycle state machine from `application_*.log`
We already tail this file for `scene preset path`. Adding five substring checks gives queue → match → countdown → raid → menu states, a **raid timer HUD**, and a **queue-time toast**. Exact patterns (all verified [local]):

```
Matching with group id:                       → queuing
MatchingCompleted:[\d.,]+ real:(?<q>[\d.,]+)  → queue time (s)
TRACE-NetworkGameCreate profileStatus:        → parse: Location: (?<map>[^,]+), RaidMode: (?<mode>\w+),
                                                shortId: (?<id>[A-Z0-9]{6}), Sid: (?<sid>[^,]+)
GameStarting:                                 → countdown (PMC)
GameStarted:                                  → raid clock zero (timer HUD; PMC if >3 s after GameStarting)
Init: pstrGameVersion:                        → back at main menu (clear HUD)
```
Also dedupe reconnects by `shortId` like TM does (`Raids` dictionary).

### 2. PVE/PVP + profile identity from `application_*.log`
Two lines, huge QoL: auto-select the right Atlas task profile and never mix PVE/PVP progress.

```
Session mode: (?<mode>Pve|Regular)
CompleteSelectedProfile ProfileId:(?<pid>[a-f0-9]+) AccountId:(?<aid>\d+)
```
(TM's live regex expects `SelectedProfile ProfileId:` — which still substring-matches `CompleteSelectedProfile...` — but match on `CompleteSelectedProfile` directly on 1.0.6.x.) Cross-check: backend log host `gw-pve.` vs `gw-pvp.` (O4).

### 3. `UserConfirmed` + group notifications from `push-notifications_*.log`
Already tailing this file. `UserConfirmed` is the earliest map/raid/server signal (fires before the scene even loads) — switch the map view sooner than the bundle line allows, and show raid `shortId` + region (`sid` prefix, e.g. `US-STL01G108`). The `GroupMatch*` family adds a group roster (nickname/side/level/ready from `GroupMatchRaidReady.extendedProfile.Info`) and pre-raid weather/side from `GroupMatchRaidSettings.raidSettings`. Patterns (verified [local]):

```
Got notification | UserConfirmed          → JSON: location, raidMode, mode, shortId, sid
Got notification | GroupMatchRaidReady    → JSON: extendedProfile.Info.{Nickname,Side,Level}
Got notification | GroupMatchRaidNotReady
Got notification | GroupMatchRaidSettings → JSON: raidSettings.{location,side,timeAndWeatherSettings}
Got notification | GroupMatchStartGame    → JSON: estimate (queue ETA s)
Got notification | GroupMatchUserLeave / GroupMatchWasRemoved / GroupMatchInviteAccept
```

### 4. FOV + screenshot-keybind sanity from the boot settings dumps
One-shot parse at game start, directly improves an existing feature: draw the view cone with the player's real `FieldOfView`, and warn when position tracking can't work.

```
Game settings:      → JSON block → "FieldOfView": 50
Control settings:   → JSON block → keyBindings[] where keyName == "MakeScreenshot"
                      (unbound or "SysReq" ⇒ warn: map position updates won't fire)
```
Both blocks verified [local]; the keybind check is lifted from TM's `Eft_ControlSettings`.

### 5. Raid-bounds + ping fallback from `network-connection_*.log`
New (tiny) file watcher; makes in-raid detection bulletproof for PVE/local and reconnects, plus a free post-raid ping stat:

```
Connect \(address: (?<ip>[\d.]+):(?<port>\d+)\)          → entering raid server
Disconnect \(address: ...\)                               → leaving
Statistics \(address: ..., rtt: (?<rtt>[\d.]+), lose: (?<loss>[\d.eE-]+), sent: \d+, received: \d+\)
```
Verified [local]. Backend `/client/match/local/start|end` (O5) is an equivalent PVE-only cross-check.

**Honorable mentions:** opportunistic position lines `Reason:PacketsQueue, Position:(x, y, z)` (A11 — free breadcrumbs, but only a few per raid); `RagfairOfferSold` + insurance-return toasts (N10/N12); `ConsistencyInfo` version watch to prompt re-extraction after patches.

---

## 5. What TarkovMonitor gets that Atlas cannot (or should consciously skip)

Passive-impossible (no file EFT writes carries it):
- **Continuous player position** — only screenshots (user keypress) and the sporadic A11 lines. TM has the same limit; its map position also relies on the user pressing screenshot.
- **Raid outcome (Survived/KIA/MIA), killer, XP** — not written to any log. `UserMatchOver` carries only `status:"Free"` + location. TM doesn't have it either (its RaidEnded is inferred from profile re-select/menu return).
- **Other players/bots, boss spawns, loot instances, extract states in-raid** — server-side; never logged. (Log channels `spawns`/`exfiltration`/`quests` exist but are pinned to Error in `Logging.config`, which is checksummed in `ConsistencyInfo` — editing it is tamper-adjacent and out of scope.)
- **Backend response bodies** (weather, trader stock, achievements content) — backend log records URLs/timings only.
- **In-raid health/ammo/inventory state** — requires memory reading; out of scope by hard constraint.

TM features that are active/external, not passive-file-based — skip or substitute:
- **Goon sighting reports & queue-time submission** — TM *submits* to `manager.tarkov.dev` API (POST /goons, /queue-times); the *sighting itself is manual user input*, not log-derived. Atlas could add the same voluntary submit later, but there is nothing to parse.
- **TarkovTracker progress sync** — external API with user token (TM pushes task completions there). Orthogonal to log parsing; Atlas already has its own task state.
- **Scav cooldown / run-through thresholds** — TM pulls constants from tarkov.dev GraphQL, not from game files (we already use tarkov.dev at build time; fine to fetch these values the same way).
- **Process watching** (`Process.GetProcessesByName("EscapeFromTarkov")`) — TM polls the process list every 30 s to detect game start/exit. Not file-based; Atlas can infer the same from log-folder creation (a new `log_*` folder appears at launch) to stay purely file-passive.
- **Media pause / sound alerts / air-filter reminders** — TM app UX built on the same events; nothing extra to parse (air filter state comes from TarkovTracker hideout data, not logs).
