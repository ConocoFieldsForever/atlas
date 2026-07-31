#!/usr/bin/env python
"""Extract TYPED gameplay data from an EFT map's Unity scenes -> out/<map>/gamedata.json.

Ground truth for the tactical overlay: instead of name-classifying GameObjects (the
extract_semantics.py heuristic — 71% false-positive rate on extracts), this reads the TYPED
MonoBehaviours the game itself uses: ExfiltrationPoint / ScavExfiltrationPoint /
SharedExfiltrationPoint / SecretExfiltrationPoint (faction = component TYPE), Minefield,
SniperFiringZone, Door / Trunk (KeyId + DoorState), TransitPoint, StationaryWeapon,
SpawnPointMarker.

EFT is IL2CPP with an ENCRYPTED global-metadata.dat, so script typetrees CANNOT be generated;
each MonoBehaviour parses only its 32-byte header (m_GameObject/m_Enabled/m_Script/m_Name).
The script class comes from the MonoScript (an engine type, typetree intact); the script
FIELDS are decoded from the raw payload with layouts recovered empirically on lighthouse
level524 + all-level door dumps (column statistics; same method as the light-controller
decode). Layouts are validated defensively — a field that doesn't look right degrades to null
instead of shipping garbage.

Zone footprints come from the BoxCollider on the same GameObject: 4 bottom-face corners
through the full world TRS chain (colliders are often unit boxes scaled by the transform).
The Unity->viewer bridge is the map config's coordinates.global_matrix conjugation reduced to
points (G3 @ p, the diag(-1,1,1) X-flip) — identical rule as the geometry pipeline; corner
order is reversed after the flip so outlines stay CCW. "Anything with a zone" ships one:
MineDirectional blast boxes (largest CHILD BoxCollider — the mine GO itself has none),
quest/visit trigger zones (PlaceItemTrigger / ExperienceTrigger / FlareShootDetectorZone /
QuestTrigger, zone id = first script string), the LighthouseTraderZone compound, plus the
original minefields / sniper zones / exfils / transits.

LOOSE LOOT (first-party): LootPoint MonoBehaviours are the ONLY loose-loot positions the
client ships — a small curated set (lighthouse: gun racks / gun safes / food piles / car
trunks; factory: none). The bulk of loose loot is SERVER data (resources.assets carries only
"err"-wrapped Test*/LootData mocks of that exchange). A LootPoint payload DOES carry its item
pool: dword flags(=1); Id GUID string @4; 28-byte fixed block; dword N; N length-prefixed
24-hex item/category TEMPLATE ids; dword tail (validated on all 4 lighthouse variants). No
weights are serialized. Template ids resolve to names/prices via tarkov.dev (items +
itemCategories); each point is also nearest-neighbor-joined to tarkov.dev lootLoose for a
match-distance report. Both net steps degrade gracefully offline (ids ship un-named).

TERRAIN DRAPING: outline verts sit at the collider's BOTTOM face, which floats/sinks on
hills. When the map's .eftpack is present (EFT_PACK_DIR override, default <repo>/packs/
<map>.eftpack), a world heightfield is built from the pack's FLAG_TERRAIN instances (same
uv->world idea as eft_pipeline/build_grass.py, binned to a 2 m world-XZ grid), every outline
edge is subdivided ~4 m and each vert lifted to max(terrain+0.3, collider_base_y) — lines
follow the ground and never sink below the collider. Verts off the terrain grid keep the
collider Y. Zones keep a pre-subdivision "extent" [w, d] for the cards; the file gets a
top-level "draped" flag so the viewer can drop its own lift.

  python extraction/intel/extract_gamedata.py <map> [--levels a,b,c] [--out FILE]
      (requires EFT_TARKMAP_ROOT; levels default to the map config's source.levels;
       default output <EFT_TARKMAP_ROOT>/out/<map>/gamedata.json)
"""
import os, re, sys, json, gc, time, math, struct, functools
from collections import Counter

import numpy as np
print = functools.partial(print, flush=True)
import UnityPy

# portable kit: paths come from the environment (see README.md)
HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/extraction/intel
KIT = os.path.dirname(HERE)                                # <repo>/extraction
DATA = os.environ.get("EFT_GAME_DATA",
                      r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data")
TK = os.environ.get("EFT_TARKMAP_ROOT")
if not TK:
    raise SystemExit("extract_gamedata: EFT_TARKMAP_ROOT is not set. Point it at your workspace "
                     "tarkmap dir (the one holding maps/ and out/), "
                     "e.g.  setx EFT_TARKMAP_ROOT D:\\eft_work\\tarkmap")

args = [a for a in sys.argv[1:] if not a.startswith("--")]
MAP = args[0] if args else "lighthouse"
LEVELS = None
OUT = None
for a in sys.argv[1:]:
    if a.startswith("--levels="):
        LEVELS = [int(x) for x in a.split("=", 1)[1].split(",")]
    elif a.startswith("--out="):
        OUT = a.split("=", 1)[1]

_cfg_p = os.path.join(TK, "maps", MAP, "config.json")
if not os.path.exists(_cfg_p):
    _cfg_p = os.path.join(KIT, "maps", MAP, "config.json")
_cfg = json.load(open(_cfg_p, encoding="utf-8"))
if LEVELS is None:
    LEVELS = [int(x) for x in (_cfg["source"].get("levels") or [])]
if OUT is None:
    OUT = os.path.join(TK, "out", MAP, "gamedata.json")
# global_matrix is a constant X-flip and is no longer stored per-config; default to it when absent.
_gm = (_cfg.get("coordinates") or {}).get("global_matrix")
G3 = (np.array(_gm, np.float64).reshape(4, 4) if _gm
      else np.diag([-1.0, 1.0, 1.0, 1.0]))[:3, :3]

# faction from the component TYPE — the whole point of this extractor.
EXFIL_CLASSES = {
    "ExfiltrationPoint": "pmc",
    "ScavExfiltrationPoint": "scav",
    "SharedExfiltrationPoint": "shared",
    "SecretExfiltrationPoint": "secret",
}


@functools.lru_cache(maxsize=1)
def game_locale_tables():
    """First-party EN/RU locale dictionaries embedded in resources.assets.

    ExfiltrationPoint.Settings.Name is a locale KEY ("NW Exfil", "E1", ...), not necessarily
    the text EFT puts on screen. The old viewer renamed it by choosing the nearest tarkov.dev
    extract within 60 m, which could silently attach the wrong name. resources.assets ships the
    exact client locale snapshot, so carry that string beside the raw key instead.

    Keys are case-folded because one scene serializes "factory gate" while the locale table uses
    "Factory gate". Values are left byte-for-byte as authored by the game.
    """
    path = os.path.join(DATA, "resources.assets")
    wanted = {"TestBackendLocaleEn": "en", "TestBackendLocaleRu": "ru"}
    tables = {}
    try:
        env = UnityPy.load(path)
        for o in env.objects:
            if o.type.name != "TextAsset":
                continue
            d = o.read()
            tag = wanted.get(getattr(d, "m_Name", ""))
            if not tag:
                continue
            root = json.loads(getattr(d, "m_Script", "") or "{}")
            data = root.get("data") if isinstance(root, dict) else None
            if isinstance(data, dict):
                tables[tag] = {
                    str(k).casefold(): v for k, v in data.items()
                    if isinstance(v, str) and v.strip()
                }
            if len(tables) == len(wanted):
                break
    except Exception as ex:
        print(f"[exfils] game locale unavailable ({type(ex).__name__}: {ex}) - raw ids only")
    return tables


def localize_exfils(exfils):
    """Attach exact in-game display names without changing the serialized identity key.

    `CarExtraction` is deliberately not accepted here: IL2CPP metadata proves it derives from
    ExfiltrationSubscriber and only animates a car subscribed to the real ExfiltrationPoint. It is
    not a selectable extract and used to create duplicate/stray markers.
    """
    tables = game_locale_tables()
    if not tables:
        return
    for e in exfils:
        raw = (e.get("name") or "").casefold()
        for tag, table in tables.items():
            if value := table.get(raw):
                e[f"display_name_{tag}"] = value
    named_en = sum(bool(e.get("display_name_en")) for e in exfils)
    named_ru = sum(bool(e.get("display_name_ru")) for e in exfils)
    print(f"[exfils] exact game locale: EN {named_en}/{len(exfils)}, "
          f"RU {named_ru}/{len(exfils)}")

DOOR_CLASSES = {"Door": "door", "Trunk": "trunk", "KeycardDoor": "door", "SlidingDoor": "door",
                "ExfiltrationDoor": "exfil_door", "DoorSwitch": "door"}
# Swing doors we can open by rotating about the owner's local Z (Codex audit): Trunk / sliding /
# exfil doors need different motion and are marked non-swing so the viewer doesn't rotate them.
SWING_DOOR_CLASSES = {"Door", "KeycardDoor", "DoorSwitch"}
# EDoorState (EFT.Interactive) — flags; scenes serialize a single initial state.
DOOR_STATE = {0: "none", 1: "locked", 2: "shut", 4: "open", 8: "interacting", 16: "breach"}
# EPlayerSideMask
SIDE_MASK = {1: "usec", 2: "bear", 3: "pmc", 4: "savage", 5: "usec+savage", 6: "bear+savage", 7: "all"}
# SpawnPointMarker Categories mask. 1/2/4/64 CONFIRMED (AI-scene audit); bits 8/16/32 appear only
# on player-scene markers (lv520 masks 8/24/40 — coop/group/op is PLAUSIBLE but unproven), so they
# decode to neutral bitN tokens and the RAW mask always ships alongside.
CAT_BITS = {1: "player", 2: "bot", 4: "boss", 64: "botpmc"}


def cat_names(mask):
    return [CAT_BITS.get(b, f"bit{b}") for b in (1, 2, 4, 8, 16, 32, 64) if mask & b]
# quest/visit trigger MonoBehaviours ("anything with a zone gets extracted"): each carries its
# BoxCollider on the SAME GameObject and serializes the quest ZONE ID as the first script
# field (validated: lighthouse level524 x110, factory level68 x42).
QUEST_TRIGGER_CLASSES = {"PlaceItemTrigger": "place_item", "ExperienceTrigger": "visit",
                         "FlareShootDetectorZone": "flare", "QuestTrigger": "quest"}
BUFFER_ZONE_CLASSES = {
    "BufferGates": "buffer_gate", "BufferGate": "buffer_gate", "BufferZone": "buffer",
    "IgnorePlayerInputZone": "input_lock", "LighthouseKeeperZone": "lightkeeper",
    "EventObjectInteractive": "event_interactive",
    "InteractiveObjectCutsceneTrigger": "cutscene",
    # AI-guarded no-fire zone around an event transit (shoreline Terminal: GuardedTrasitZone +
    # GuardedZoneGates). Zone footprint + marker ride the existing buffer_zones sink.
    "GuardedZone": "guarded",
}
# Typed DAMAGE zones: a FlameDamageTrigger is an empty marker component (payload 0 B) whose
# BoxCollider IS the burn zone (shoreline gas fires, factory furnaces). kind stays per-class so
# future damage trigger types slot in without a format change.
DAMAGE_ZONE_CLASSES = {"FlameDamageTrigger": "flame"}


def read_cstr(buf, off):
    """length-prefixed utf8 string + 4-aligned end offset; (None, off) when implausible."""
    if off + 4 > len(buf):
        return None, off
    ln = int.from_bytes(buf[off:off + 4], "little")
    if ln < 0 or ln > 4096 or off + 4 + ln > len(buf):
        return None, off
    try:
        s = buf[off + 4:off + 4 + ln].decode("utf8")
    except UnicodeDecodeError:
        return None, off
    return s, (off + 4 + ln + 3) & ~3


def payload_of(o, hdr):
    """raw script fields after the 32(+name)-byte MonoBehaviour header."""
    raw = o.get_raw_data()
    nm = hdr.get("m_Name") or ""
    hsize = (12 + 4 + 12 + 4 + len(nm.encode("utf8")) + 3) & ~3
    return raw[hsize:]


# ---- payload decoders (layouts: see module docstring) ----
def dec_exfil_name(pl):
    """all four exfil classes: 48 fixed bytes, then the settings Name string."""
    s, _ = read_cstr(pl, 48)
    return (s or "").strip() or None


def dec_door(pl):
    """WorldInteractiveObject: [20B zeros][dword N = interaction-trigger count][N x (dword kind,
    trigger str)][dword 0x0F layer][KeyId str][12B][Id str][tail rel. Id end: open angle @+56,
    state @+92]. Classic doors (every pre-Icebreaker map) serialize N=0, which collapses to the
    original fixed layout (KeyId @28) validated on 299 lighthouse doors (state column {1,2,4,16},
    keyed doors all 1) + 97 open Interchange doors (angle within 0.15deg of the authored pose).
    Icebreaker+ doors driven by handles/switches/quests serialize N=1..2 trigger-name strings
    ("Open_01_<hash>"/"Close_01_<hash>"/"Quest_Complete_<hash>") BEFORE that block — the old
    fixed-offset read returned the trigger name as KeyId and lost state+angle (the 15 dead doors
    on the Icebreaker deck). The 0x0F layer dword anchors both layouts; if it isn't where the
    trigger walk says it should be, degrade to Nones rather than shipping garbage.
    Returns (key, id, state, angle, triggers) — `triggers` are the interaction trigger names
    (e.g. "Open_01_722179887"); their trailing digit-hash links a door to the Switch interactable
    that drives it (the switch serializes the same hash in ITS trigger string), letting the merge
    wire switch->door target edges with zero name matching. Classic doors: []."""
    import struct as _st

    def u32(off):
        return int.from_bytes(pl[off:off + 4], "little") if off + 4 <= len(pl) else None

    def tail(kend):
        """KeyId ended at `kend`: read Id + the fixed-offset tail relative to the Id end."""
        did, iend = read_cstr(pl, kend + 12)
        st = None
        if iend + 96 <= len(pl):
            v = int.from_bytes(pl[iend + 92:iend + 96], "little")
            st = DOOR_STATE.get(v)                          # unknown value -> None, not garbage
        ang = None
        if iend + 60 <= len(pl):
            a = _st.unpack_from("<f", pl, iend + 56)[0]
            if a == a and 0.0 < abs(a) <= 180.0:            # finite, door-scale
                ang = round(float(a), 2)
        return (did or None), st, ang

    def is_key(s):
        """A serialized KeyId is always empty or a 24-hex item template id — a trigger name
        ("Open_01_<hash>") is neither, which is how the two layouts are told apart."""
        return s == "" or (len(s) == 24 and all(c in "0123456789abcdef" for c in s))

    # Classic layout first (KeyId @28) — the exact pre-Icebreaker read, validated on every old map.
    key, kend = read_cstr(pl, 28)
    if key is not None and is_key(key):
        did, st, ang = tail(kend)
        return (key or None), did, st, ang, []
    # Trigger-block layout: [dword N @20][N x (kind dword, trigger str)][0x0F][KeyId][12B][Id][tail].
    # The 0x0F anchor is required HERE only (validated on every Icebreaker door variant) — classic
    # payloads never reach this path, so older maps keep their anchor-free read above.
    n_trig = u32(20)
    if n_trig is None or not 0 < n_trig <= 8:               # defensive: real doors carry 1-2
        return None, None, None, None, []
    off = 24
    triggers = []
    for _ in range(n_trig):
        off += 4                                            # trigger kind dword (4=open, 2=close)
        s, off = read_cstr(pl, off)
        if s is None:
            return None, None, None, None, []
        triggers.append(s)
    if u32(off) != 0x0F:                                    # layer anchor must line up
        return None, None, None, None, []
    key, kend = read_cstr(pl, off + 4)
    if key is None or not is_key(key):
        return None, None, None, None, []
    did, st, ang = tail(kend)
    return (key or None), did, st, ang, triggers


def dec_spawn(pl):
    """Id str, Name str, Vector3 pos, Quaternion rot, Sides mask, Categories mask, Infil str,
    then (AI-scene audit): PPtr BotZone | f32 (40.0 AI scenes / 4.0 player scenes) | i32
    CorePointId | PPtr SphereCollider | 16B const. Tail validated on interchange level66
    (102/102, collider radius == every `rad:` name token) and level520 (177/177, null BotZone,
    default 50 m collider). Both PPtrs are always in-scene (fid 0); a short payload or an
    external fid degrades the tail to Nones instead of shipping garbage."""
    sid, off = read_cstr(pl, 0)
    name, off = read_cstr(pl, off)
    if name is None or off + 36 > len(pl):
        return None
    pos = struct.unpack_from("<3f", pl, off)
    sides = int.from_bytes(pl[off + 28:off + 32], "little")
    cats = int.from_bytes(pl[off + 32:off + 36], "little")
    inf, end = read_cstr(pl, off + 36)
    if not all(math.isfinite(v) and abs(v) < 1e5 for v in pos):
        return None
    bz_pid = core = sph_pid = None
    if inf is not None and end + 32 <= len(pl):
        f0, p0 = struct.unpack_from("<iq", pl, end)           # PPtr BotZone
        core = int.from_bytes(pl[end + 16:end + 20], "little", signed=True)
        f1, p1 = struct.unpack_from("<iq", pl, end + 20)      # PPtr SphereCollider
        if f0 == 0 and p0:
            bz_pid = p0
        if f1 == 0 and p1:
            sph_pid = p1
    return sid, name, pos, sides, cats, inf, bz_pid, core, sph_pid


def dec_patrol_way(pl):
    """PatrolWay / PatrolWayWithName / PatrolWayWithConditions: u32 type @0, u32 N @4, then
    N x 12B PPtr PatrolPoint IN ROUTE ORDER (the trailing index in each point's GameObject
    name matches its array index), 0xFFFFFFFF sentinel, 1.0f, then the route name string
    (WithName only). Validated on interchange level66: 30/30 ways, every PPtr an in-scene
    PatrolPoint, sentinel on all. Returns (point_pids, name) or None when the shape doesn't
    hold (WithConditions variants that serialize extra state degrade here, not to garbage)."""
    if len(pl) < 16:
        return None
    n = int.from_bytes(pl[4:8], "little")
    if n == 0 or n > 4096 or 8 + 12 * n + 8 > len(pl):
        return None
    pids, off = [], 8
    for _ in range(n):
        fid = int.from_bytes(pl[off:off + 4], "little", signed=True)
        pid = int.from_bytes(pl[off + 4:off + 12], "little", signed=True)
        if fid != 0 or not pid:
            return None
        pids.append(pid)
        off += 12
    if pl[off:off + 4] != b"\xff\xff\xff\xff":
        return None
    # STRICT read: WithConditions serializes condition state after the sentinel, and the
    # lenient reader turned it into a "\x12" name on labs. Printable-or-nothing.
    name, _ = read_cstr_strict(pl, off + 8)
    return pids, (name or "").strip() or None


def locate_pptr_arrays(pl, targets):
    """[(kind, [pids])] for every [u32 N][N x 12B in-scene PPtr] run in `pl` whose pids ALL
    fall inside one of the `targets` pid-sets ({kind: set}). BotZone serializes its PatrolWay
    array @20 immediately followed by its SpawnPointMarker array (validated on every
    interchange zone), but WALKING the <400 B payload is cheap and survives a per-map layout
    wobble — an array that doesn't validate simply isn't found, and the caller degrades."""
    out, off, ln = [], 0, len(pl or b"")
    while off + 4 <= ln:
        n = int.from_bytes(pl[off:off + 4], "little")
        if 0 < n <= 2000 and off + 4 + 12 * n <= ln:
            pids = []
            for i in range(n):
                base = off + 4 + 12 * i
                fid = int.from_bytes(pl[base:base + 4], "little", signed=True)
                pid = int.from_bytes(pl[base + 4:base + 12], "little", signed=True)
                if fid != 0 or not pid:
                    pids = None
                    break
                pids.append(pid)
            if pids:
                kind = next((k for k, ts in targets.items() if all(p in ts for p in pids)), None)
                if kind:
                    out.append((kind, pids))
                    off += 4 + 12 * n
                    continue
        off += 4
    return out


def hull_xz(pts):
    """Convex hull (monotone chain) over the XZ plane; each hull vert keeps its own Y (member
    points already sit at ground height). [] when fewer than 3 distinct XZ positions or the
    set is collinear — a 1-marker zone has no footprint, only its `pos`."""
    uniq = sorted(set((round(p[0], 2), round(p[2], 2), round(p[1], 2)) for p in pts))
    if len(uniq) < 3:
        return []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lo, hi = [], []
    for p in uniq:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(uniq):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    ring = lo[:-1] + hi[:-1]
    return [[x, y, z] for x, z, y in ring] if len(ring) >= 3 else []


def dec_tod_sky(pl):
    """TOD_Sky (the *_Scripts scene's sun model): [5 ints][f Hour @20][i Day][i Month][i Year]
    [f Latitude][f Longitude] ... Validated on interchange level53 (6.4 h, 1/8/2018, 46.0 N
    84.0 E). Every field range-checked; anything implausible -> None."""
    if len(pl) < 44:
        return None
    hour = struct.unpack_from("<f", pl, 20)[0]
    day, month, year = struct.unpack_from("<3i", pl, 24)
    lat, lon = struct.unpack_from("<2f", pl, 36)
    ok = (math.isfinite(hour) and 0.0 <= hour < 24.0 and 1 <= day <= 31 and 1 <= month <= 12
          and 2000 <= year <= 2100 and math.isfinite(lat) and abs(lat) <= 90.0
          and math.isfinite(lon) and abs(lon) <= 180.0)
    return ({"hour": round(hour, 2), "day": day, "month": month, "year": year,
             "lat": round(float(lat), 3), "lon": round(float(lon), 3)} if ok else None)


def dec_level_border(pl):
    """LevelBorder (the *_Culling scene): u32 N then N x float3 Unity verts — the game's own
    playable-area polygon (interchange: 37 verts at fixed Y). Bridged; vert order reversed
    after the X-flip so the ring stays CCW like every other outline."""
    if len(pl) < 16:
        return None
    n = int.from_bytes(pl[0:4], "little")
    if not 3 <= n <= 4096 or 4 + 12 * n > len(pl):
        return None
    v = struct.unpack_from(f"<{3 * n}f", pl, 4)
    if not all(math.isfinite(x) and abs(x) < 1e5 for x in v):
        return None
    return [bridge(v[i * 3:i * 3 + 3]) for i in range(n)][::-1]


def dec_door_link(pl):
    """NavMeshDoorLink (AI scene): u32 link id, door id string (the SAME door_… id gamedata's
    doors[] carry), 12B zeros, float3 A, float3 B (then B again). Free nav-graph traversal
    edges keyed to already-extracted doors; validated on interchange x219."""
    did, e = read_cstr(pl, 4)
    if not did or e + 36 > len(pl):
        return None
    a = struct.unpack_from("<3f", pl, e + 12)
    b = struct.unpack_from("<3f", pl, e + 24)
    if not all(math.isfinite(x) and abs(x) < 1e5 for x in a + b):
        return None
    return did, a, b


def read_cstr_strict(buf, off):
    """extract_interact's PRINTABLE length-prefixed string read (len 1..256, every char in
    0x20..0x7e). The lenient module `read_cstr` is wrong for blind payload WALKS: arbitrary
    binary bytes < 0x80 decode as valid utf8, so a garbage dword can masquerade as a long
    "string" and carry the walk past the real field."""
    if off + 4 > len(buf):
        return None, off
    ln = int.from_bytes(buf[off:off + 4], "little")
    if ln <= 0 or ln > 256 or off + 4 + ln > len(buf):
        return None, off
    try:
        s = buf[off + 4:off + 4 + ln].decode("utf8")
    except UnicodeDecodeError:
        return None, off
    if not all(31 < ord(c) < 127 for c in s):
        return None, off
    return s, (off + 4 + ln + 3) & ~3


def walk_strings(pl, off):
    """(offset, string) pairs from `off` on — the defensive dword-step walk extract_interact
    uses, with the strict printable reader above."""
    out = []
    n = len(pl or b"")
    while off + 4 <= n:
        s, e = read_cstr_strict(pl, off)
        if s and len(s) >= 3:
            out.append((off, s))
            off = e
        else:
            off += 4
    return out


def hex24_strings(pl, off):
    """every length-prefixed 24-hex template id from `off` on, in serialized order."""
    return [s for _, s in walk_strings(pl, off)
            if len(s) == 24 and all(c in "0123456789abcdef" for c in s)]


def dec_stationary(pl):
    """StationaryWeapon: [20B float block][Name str @20][dword N][N x 12B mount PPtrs]
    [3 x 12B fixed PPtrs][7 floats: default yaw, 0, pitch min, pitch max, yaw min, yaw max, 0]
    [...][24-hex weapon template str][...]. Validated on shoreline lv29 / streets lv396 /
    ground_zero lv505 (14 components; NSV "Utes" 452-488 B and AGS 296-332 B layouts agree).
    Angles are Unity-world degrees. Everything is defensive: a field that doesn't look right
    ships as None instead of garbage. Returns (name, weapon_id, aim) with aim =
    {yaw, yaw_range, pitch_range} or None."""
    name, nend = read_cstr(pl, 20)
    name = (name or "").strip() or None
    ids = hex24_strings(pl, nend if name else 20)
    wid = ids[0] if len(ids) == 1 else None                 # exactly one id serialized today
    aim = None
    n_ptr = int.from_bytes(pl[nend:nend + 4], "little") if name and nend + 4 <= len(pl) else -1
    if 0 <= n_ptr <= 64:
        p = nend + 4 + 12 * n_ptr + 36                      # mount PPtr array + 3 fixed PPtrs
        if p + 28 <= len(pl):
            f = struct.unpack_from("<7f", pl, p)
            ok = (all(math.isfinite(v) and abs(v) <= 720.0 for v in f)
                  and f[2] < f[3] and f[4] < f[5] and f[4] - 1.0 <= f[0] <= f[5] + 1.0)
            if ok:
                aim = {"yaw": round(f[0], 2),
                       "yaw_range": [round(f[4], 2), round(f[5], 2)],
                       "pitch_range": [round(f[2], 2), round(f[3], 2)]}
    return name, wid, aim


def dec_lootgroup(pl):
    """LootableContainersGroup -> (id, min_spawn, max_spawn).

    Layout after the 28-byte MonoBehaviour header: m_Name (int32 length, always 0 here), then a
    length-prefixed 4-ALIGNED id string, then two int32s.

        Goshan          -> 00000000 06000000 "Goshan"..  11000000 15000000  -> (17, 21)
        ClothingShops   -> 00000000 0d000000 "ClothingShops"  0b000000 0f000000 -> (11, 15)

    The pair is how many of the group's containers actually spawn in a raid. Verified on all 19
    interchange groups: 0 <= min <= max <= (number of descendant LootableContainers) held every
    time, and the ratio varies enormously by area - Kiba Arms 2-3 of 3 (~83%) vs the mall stashes
    18-21 of 104 (~19%). That is the GAME's own per-location spawn odds, where the value model
    otherwise applies one type-average fill rate everywhere.
    """
    # NOTE the offsets: `payload_of` already consumes the MonoBehaviour header AND m_Name, so the
    # group-id length is at 0 here, not 4 (raw[28:] would put it at 4 -- that is where the probe saw
    # it, and reading the probe offsets straight into this decoder is what made every group decode
    # to nothing while still emitting a record).
    if len(pl) < 12:
        return None, None, None
    n = int.from_bytes(pl[0:4], "little")
    if not 0 <= n <= 64 or 4 + n > len(pl):
        return None, None, None
    gid = pl[4:4 + n].decode("utf-8", "replace")
    off = 4 + n
    off += (-off) % 4                      # Unity 4-aligns after a string
    if off + 8 > len(pl):
        return gid or None, None, None
    lo = int.from_bytes(pl[off:off + 4], "little", signed=True)
    hi = int.from_bytes(pl[off + 4:off + 8], "little", signed=True)
    if not (0 <= lo <= hi <= 4096):
        return gid or None, None, None     # decode did not hold: report the id, no odds
    return (gid or None), lo, hi


def dec_container(pl):
    """LootableContainer: [44B fixed][Id str @44 ("container_<zone>_00001")][...][24-hex
    container TEMPLATE id (tarkov.dev lootContainers: jacket / duffle-bag / dead-scav / ...)]
    [optional group tag]. Validated on icebreaker 704/705, shoreline 28, factory_rework 533
    (424-488 B payloads). Returns (id, template) — either may be None."""
    cid, e = read_cstr(pl, 44)
    cid = (cid or "").strip() or None
    ids = hex24_strings(pl, e if cid else 44)
    return cid, (ids[0] if ids else None)


def dec_card_reader(pl):
    """CardReader: an id string, then PAIRS of (24-hex ACCEPTED-card item id, event name), then
    a fallback event ("on_unknown_card_used"). Validated on the shoreline Terminal reader
    (lv29, 648 B: 5 keycards). Returns the ordered unique accepted-card template ids."""
    ids = hex24_strings(pl, 0)
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def dec_dialog(pl):
    """RaidDialogEntryPoint: [..][localization key str ("Dialog/EntryPoint/<map>/ActionName")]
    [..][dialog id str ("raid_dialog_scientist")]. Validated on icebreaker lv704."""
    strs = [s for _, s in walk_strings(pl, 0)]
    key = next((s for s in strs if "/" in s), None)
    did = next((s for s in reversed(strs) if "/" not in s), None)
    return key, did


def dec_zone_id(pl):
    """quest-trigger classes serialize the zone id as the FIRST script field (string @0)."""
    s, _ = read_cstr(pl, 0)
    return (s or "").strip() or None


def poly_area_xz(pts):
    """shoelace area of a polygon projected to XZ (picks the mine's real blast box)."""
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, z1 = pts[i][0], pts[i][2]
        x2, z2 = pts[(i + 1) % n][0], pts[(i + 1) % n][2]
        a += x1 * z2 - x2 * z1
    return abs(a) / 2.0


def dec_lootpoint(pl):
    """LootPoint: dword flags(=1); Id GUID str @4; 28-byte fixed block (two variant dwords +
    zeros); dword N; N x length-prefixed 24-hex item/category template ids; dword tail.
    Recovered empirically on lighthouse levels 185-207 (all 4 GameObject variants agree). The
    array offset is SCANNED over a small window past the GUID instead of hardcoded, so a
    fixed-block size change degrades to (guid, []) rather than garbage."""
    guid, e = read_cstr(pl, 4)
    if guid is None or len(guid) < 8:
        return None, []
    for off in range(e, min(len(pl) - 4, e + 64), 4):
        n = int.from_bytes(pl[off:off + 4], "little")
        if not 1 <= n <= 64:
            continue
        tps, p, ok = [], off + 4, True
        for _ in range(n):
            s, p2 = read_cstr(pl, p)
            if s is None or len(s) != 24 or not all(c in "0123456789abcdef" for c in s):
                ok = False
                break
            tps.append(s)
            p = p2
        if ok:
            return guid, tps
    return guid, []


def trs_mat(t, q, s):
    x, y, z, w = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], np.float64)
    M = np.eye(4)
    M[:3, :3] = R * np.array(s, np.float64)
    M[:3, 3] = t
    return M


def bridge(p):
    """Unity world point -> viewer/pack space (global_matrix conjugation reduced to a point)."""
    return [round(float(v), 2) for v in (G3 @ np.asarray(p, np.float64))]


# MonoScript name cache is global across levels (externals repeat).
_ms_idx = {}

def monoscript_index(path):
    if path not in _ms_idx:
        idx = {}
        if os.path.exists(path):
            e = UnityPy.load(path)
            for o in e.objects:
                if o.type.name == "MonoScript":
                    try:
                        idx[o.path_id] = o.read_typetree().get("m_ClassName")
                    except Exception:
                        pass
            del e
        _ms_idx[path] = idx
    return _ms_idx[path]


def scan_level(lv, sink, ai=False):
    p = os.path.join(DATA, f"level{lv}")
    if not os.path.exists(p):
        print(f"[level{lv}] missing - skip")
        return
    t0 = time.time()
    env = UnityPy.load(p)
    sf = next((f for f in env.files.values() if hasattr(f, "objects")), None)
    externals = list(getattr(sf, "externals", []) or [])
    objs = env.objects

    local_scripts = None

    def resolve(fid, pid):
        nonlocal local_scripts
        if fid == 0:
            if local_scripts is None:
                local_scripts = {}
                for o in objs:
                    if o.type.name == "MonoScript":
                        try:
                            local_scripts[o.path_id] = o.read_typetree().get("m_ClassName")
                        except Exception:
                            pass
            return local_scripts.get(pid)
        base = os.path.basename(getattr(externals[fid - 1], "path", "").replace("\\", "/"))
        return monoscript_index(os.path.join(DATA, base)).get(pid)

    go_obj, tr_obj, col_obj = {}, {}, {}
    sph_obj = {}                                  # SphereCollider pid -> obj (spawn radii)
    mb_obj = {}                                   # MonoBehaviour pid -> obj (PPtr resolution)
    mf_obj = {}                                   # GameObject pid -> its MeshFilter (door parts)
    mr_go = {}                                    # GameObject pid -> its MeshRenderer (door parts)
    mbs = []
    for o in objs:
        tn = o.type.name
        if tn == "GameObject":
            go_obj[o.path_id] = o
        elif tn in ("Transform", "RectTransform"):
            tr_obj[o.path_id] = o
        elif tn == "BoxCollider":
            col_obj[o.path_id] = o
        elif tn == "SphereCollider":
            sph_obj[o.path_id] = o
        elif tn == "MeshFilter":
            try:
                mf_obj[o.read(check_read=False).m_GameObject.path_id] = o
            except Exception:
                pass
        elif tn == "MeshRenderer":
            try:
                mr_go[o.read(check_read=False).m_GameObject.path_id] = o
            except Exception:
                pass
        elif tn == "MonoBehaviour":
            mbs.append(o)
            mb_obj[o.path_id] = o

    # lazy per-object typetree caches (engine types — typetrees intact)
    tt_cache = {}

    def tt(pid, table):
        if pid not in tt_cache:
            o = table.get(pid)
            try:
                tt_cache[pid] = o.read_typetree() if o else None
            except Exception:
                tt_cache[pid] = None
        return tt_cache[pid]

    def vec(d, k, dft):
        v = d.get(k) or {}
        return [v.get("x", dft[0]), v.get("y", dft[1]), v.get("z", dft[2])]

    go_tt_cache = {}

    def go_tt(pid):
        if pid not in go_tt_cache:
            o = go_obj.get(pid)
            try:
                go_tt_cache[pid] = o.read_typetree() if o else None
            except Exception:
                go_tt_cache[pid] = None
        return go_tt_cache[pid]

    wm_cache = {}

    def world_mat(tpid):
        """full TRS world matrix of a Transform (father-chain product), memoized per node."""
        if tpid in wm_cache:
            return wm_cache[tpid]
        d = tt(tpid, tr_obj)
        if not d:
            wm_cache[tpid] = np.eye(4)
            return wm_cache[tpid]
        q = [d.get("m_LocalRotation", {}).get(a, b) for a, b in zip("xyzw", (0, 0, 0, 1))]
        L = trs_mat(vec(d, "m_LocalPosition", (0, 0, 0)), q, vec(d, "m_LocalScale", (1, 1, 1)))
        f = (d.get("m_Father") or {}).get("m_PathID", 0)
        M = world_mat(f) @ L if f else L
        wm_cache[tpid] = M
        return M

    act_cache = {}

    def active_chain(tpid, go_pid):
        """GO m_IsActive AND every ancestor's — inactive content still ships, flag it."""
        key = (tpid, go_pid)
        if key in act_cache:
            return act_cache[key]
        gd = go_tt(go_pid)
        ok = bool(gd.get("m_IsActive", True)) if gd else True
        if ok and tpid:
            d = tt(tpid, tr_obj)
            f = (d.get("m_Father") or {}).get("m_PathID", 0) if d else 0
            if f:
                fd = tt(f, tr_obj)
                fgo = (fd.get("m_GameObject") or {}).get("m_PathID", 0) if fd else 0
                ok = active_chain(f, fgo)
        act_cache[key] = ok
        return ok

    group_tf = {}          # Transform pid of a LootableContainersGroup -> its record
    group_of_cache = {}

    def group_of(tpid):
        """Nearest ancestor LootableContainersGroup id for a Transform, or None.

        Memoized PER NODE (not per leaf): the same chain is walked by every container under a
        group, and a per-leaf memo is the quadratic pattern that has cost hours on deep hierarchies
        before. Groups are discovered as the scan goes, so a container seen BEFORE its group is
        resolved in the post-pass below instead.
        """
        if not tpid:
            return None
        if tpid in group_of_cache:
            return group_of_cache[tpid]
        chain, cur, guard = [], tpid, 0
        found = None
        while cur and guard < 256:
            guard += 1
            if cur in group_of_cache:
                found = group_of_cache[cur]
                break
            g = group_tf.get(cur)
            if g is not None:
                found = g.get("gid")
                break
            chain.append(cur)
            d = tt(cur, tr_obj)
            cur = (d.get("m_Father") or {}).get("m_PathID", 0) if d else 0
        for c in chain:
            group_of_cache[c] = found
        return found

    def go_info(go_pid):
        """(name, transform pid, [BoxCollider tt, ...]) of a GameObject."""
        gd = go_tt(go_pid)
        if not gd:
            return None, None, []
        tpid, cols = None, []
        for c in gd.get("m_Component", []):
            pp = (c.get("component") if isinstance(c, dict) else None) or c
            cid = pp.get("m_PathID") if isinstance(pp, dict) else None
            if cid is None:
                continue
            if cid in tr_obj and tpid is None:
                tpid = cid
            elif cid in col_obj:
                d = tt(cid, col_obj)
                if d:
                    cols.append(d)
        return gd.get("m_Name"), tpid, cols

    # ---- AI-scene helpers (SpawnPointMarker tail + patrol ways; see dec_spawn) ----
    def mb_hdr(pid):
        o = mb_obj.get(pid)
        try:
            return o.read_typetree(check_read=False) if o else None
        except Exception:
            return None

    def mb_go_name(pid):
        """A MonoBehaviour PPtr -> its GameObject's name (BotZone pid -> "ZoneTagilla")."""
        h = mb_hdr(pid) if pid else None
        return go_info((h.get("m_GameObject") or {}).get("m_PathID"))[0] if h else None

    def sphere_radius(pid):
        """SphereCollider m_Radius (raw — it matched every `rad:` name token; the marker
        transforms carry unit scale). None when the PPtr doesn't land on a sphere."""
        o = sph_obj.get(pid) if pid else None
        try:
            r = o.read_typetree().get("m_Radius") if o else None
        except Exception:
            r = None
        ok = isinstance(r, (int, float)) and math.isfinite(r) and 0.0 < r < 1e4
        return round(float(r), 2) if ok else None

    _mesh_name = {}
    _drawn_cache = {}

    def _drawn(gpid):
        """Does this GameObject's MeshRenderer actually DRAW? Same Unity-visibility rule the
        geometry extractor culls by (ShadowsOnly cast==3 / renderer disabled): a door's subtree
        also holds invisible BALLISTIC panels (Icebreaker's Box001..Box016) which are never in
        the pack, and whose 3ds-max-default names ('Box001') could otherwise false-match an
        unrelated instance a metre away."""
        if gpid not in _drawn_cache:
            ok = False
            mr = mr_go.get(gpid)
            if mr is not None:
                try:
                    d = mr.read_typetree()
                    ok = bool(d.get("m_Enabled", 1)) and d.get("m_CastShadows", 1) != 3
                except Exception:
                    ok = True                     # unreadable -> keep (match still needs name+pos)
            _drawn_cache[gpid] = ok
        return _drawn_cache[gpid]

    def door_parts(tpid, depth=0):
        """Every RENDERER in a door's transform SUBTREE, as [mesh name, bridged world pos].

        A door is not one mesh: the Door component sits on the swinging LEAF GameObject, whose
        subtree holds the panel, its glass, wood/metal inlays and the shadow proxy — while the
        FRAME is a SIBLING that must stay put (verified on streets Inside_Door_Wood_23: leaf
        subtree = door_L_LOD0 + door_L_glass_LOD0 + door_L_wood_LOD0; siblings = the frame
        `_LOD0` and its own `_glass_LOD0`). The viewer matched only the ONE nearest instance to
        the pivot, so a door's glass stayed behind when the panel swung. Shipping the subtree
        makes the part set the GAME's own grouping instead of a proximity guess. Mesh names
        match the pack's (the exporter sanitizes the same Unity mesh name)."""
        out = []
        d = tt(tpid, tr_obj)
        if not d:
            return out
        gpid = (d.get("m_GameObject") or {}).get("m_PathID")
        # A door part must actually RENDER: ballistic/collision proxies carry a MeshFilter with
        # no MeshRenderer (Icebreaker's Box001..Box016) and would only add noise to the match.
        mf = mf_obj.get(gpid) if _drawn(gpid) else None
        if mf is not None:
            if gpid not in _mesh_name:
                try:
                    _mesh_name[gpid] = mf.read(check_read=False).m_Mesh.read().m_Name
                except Exception:
                    _mesh_name[gpid] = None
            nm = _mesh_name[gpid]
            if nm:
                out.append([nm, bridge(world_mat(tpid)[:3, 3])])
        if depth < 6:
            for ch in d.get("m_Children", []) or []:
                cp = ch.get("m_PathID") if isinstance(ch, dict) else None
                if cp in tr_obj:
                    out.extend(door_parts(cp, depth + 1))
        return out

    def footprint(M, col):
        """4 world bottom-face corners of a BoxCollider under world matrix M, bridged.
        Corner order reversed after the X-flip so the outline stays CCW."""
        c = vec(col, "m_Center", (0, 0, 0))
        s = vec(col, "m_Size", (1, 1, 1))
        hx, hz = s[0] / 2.0, s[2] / 2.0
        y = c[1] - s[1] / 2.0
        loc = [(c[0] - hx, y, c[2] - hz), (c[0] + hx, y, c[2] - hz),
               (c[0] + hx, y, c[2] + hz), (c[0] - hx, y, c[2] + hz)]
        out = [bridge((M @ np.array([*l, 1.0]))[:3]) for l in loc]
        return [out[0], out[3], out[2], out[1]]

    def col_center(M, col):
        c = vec(col, "m_Center", (0, 0, 0))
        return bridge((M @ np.array([*c, 1.0]))[:3])

    # father -> [child transform pids], built LAZILY (reads every Transform typetree, so only
    # levels that actually hold child-collider zones — MineDirectional — pay for it).
    kid_map = None

    def child_transforms(tpid):
        nonlocal kid_map
        if kid_map is None:
            kid_map = {}
            for pid in tr_obj:
                d = tt(pid, tr_obj)
                f = (d.get("m_Father") or {}).get("m_PathID", 0) if d else 0
                if f:
                    kid_map.setdefault(f, []).append(pid)
        return kid_map.get(tpid, [])

    def largest_child_box(tpid):
        """(outline, center, child GO name) of the LARGEST child BoxCollider footprint — a
        MineDirectional hangs its blast/trigger boxes on child GOs (MON-50_MineTrigger x3 + a
        small body collider); the largest box IS the danger zone."""
        best = None
        for cpid in child_transforms(tpid):
            cd = tt(cpid, tr_obj)
            cgo = (cd.get("m_GameObject") or {}).get("m_PathID", 0) if cd else 0
            cname, _, ccols = go_info(cgo)
            M2 = world_mat(cpid)
            for col in ccols:
                fp = footprint(M2, col)
                area = poly_area_xz(fp)
                if best is None or area > best[0]:
                    best = (area, fp, col_center(M2, col), cname)
        return (best[1], best[2], best[3]) if best else (None, None, None)

    n_hit = 0
    ways_raw, zones_raw = [], []      # AI-scene raws, resolved in the post-pass below
    for o in mbs:
        try:
            hdr = o.read_typetree(check_read=False)
        except Exception:
            continue
        s = hdr.get("m_Script") or {}
        try:
            cls = resolve(s.get("m_FileID", 0), s.get("m_PathID", 0))
        except Exception:
            cls = None
        if cls not in EXFIL_CLASSES and cls not in DOOR_CLASSES \
                and cls not in QUEST_TRIGGER_CLASSES and cls not in (
                "Minefield", "SniperFiringZone", "TransitPoint", "StationaryWeapon",
                "SpawnPointMarker", "MineDirectional", "LootPoint", "LootPointsGroup",
                "LighthouseTraderZone", "BufferGateSwitcher", "LootableContainer",
                "LootableContainersGroup", "BarbedWire", "WindowBreaker",
                "ShootableQuestLocationObject",
                "CardReader", "RaidDialogEntryPoint",
                "BotZone", "PatrolWay", "PatrolWayWithName", "PatrolWayWithConditions",
                "AirdropPoint", "IndoorTrigger", "TOD_Sky", "LevelBorder",
                "NavMeshDoorLink", "AICorePoint", "AIPlaceInfo", "CultistSignEffect",
                "SpatialAudioRoom", "SpatialAudioPortal") \
                and cls not in BUFFER_ZONE_CLASSES and cls not in DAMAGE_ZONE_CLASSES:
            continue
        go_pid = (hdr.get("m_GameObject") or {}).get("m_PathID")
        if not go_pid:
            continue
        name, tpid, cols = go_info(go_pid)
        M = world_mat(tpid) if tpid else np.eye(4)
        tpos = bridge(M[:3, 3])
        active = active_chain(tpid, go_pid) and bool(hdr.get("m_Enabled", 1))
        pl = payload_of(o, hdr)
        n_hit += 1

        if cls in EXFIL_CLASSES:
            box = cols[0] if cols else None
            sink["exfils"].append({
                "name": dec_exfil_name(pl) or name or "Extract",
                "faction": EXFIL_CLASSES[cls],
                "pos": col_center(M, box) if box else tpos,
                "outline": footprint(M, box) if box else [],
                "go": name, "active": active, "lv": lv,
            })
        elif cls == "Minefield":
            for box in (cols or [None]):
                sink["minefields"].append({
                    "pos": col_center(M, box) if box else tpos,
                    "outline": footprint(M, box) if box else [],
                    "name": name, "active": active, "lv": lv,
                })
        elif cls == "SniperFiringZone":
            box = cols[0] if cols else None
            sink["sniper_zones"].append({
                "pos": col_center(M, box) if box else tpos,
                "outline": footprint(M, box) if box else [],
                "name": name, "active": active, "lv": lv,
            })
        elif cls in DOOR_CLASSES:
            key, did, st, ang, triggers = dec_door(pl)
            rec = {
                "pos": tpos, "key_id": key, "state": st, "kind": DOOR_CLASSES[cls],
                "id": did, "name": name, "active": active, "lv": lv,
            }
            # Swing doors (Door/KeycardDoor/DoorSwitch) carry the open angle so the viewer can
            # animate them about their pivot; trunks/sliding/exfil doors move differently (no swing).
            if cls in SWING_DOOR_CLASSES:
                rec["swing"] = True
                if ang is not None:
                    rec["open_angle"] = ang
                # The parts that swing WITH the panel (glass/inlays) — the game's own subtree,
                # so the viewer stops guessing by proximity. See `door_parts`.
                if tpid:
                    parts = door_parts(tpid)
                    if parts:
                        rec["parts"] = parts
            # Trigger-hash links (newer maps): the trailing digit-hash of each trigger name joins
            # this door to the Switch interactable that drives it (build_map's stage-6 merge turns
            # the join into switch->door target edges). Omitted when absent -> classic maps' output
            # stays byte-identical.
            links = [t.rsplit("_", 1)[1] for t in triggers
                     if "_" in t and t.rsplit("_", 1)[1].isdigit() and len(t.rsplit("_", 1)[1]) >= 6]
            if links:
                rec["links"] = sorted(set(links))
            sink["doors"].append(rec)
        elif cls == "TransitPoint":
            box = cols[0] if cols else None
            sink["transit_points"].append({
                "pos": col_center(M, box) if box else tpos, "name": name,
                "outline": footprint(M, box) if box else [], "active": active, "lv": lv,
            })
        elif cls == "StationaryWeapon":
            snm, wid, aim = dec_stationary(pl)
            rec = {
                "pos": tpos, "name": snm or name or "Stationary weapon",
                "active": active, "lv": lv,
            }
            # The mounted WEAPON's serialized template id + firing arc — optional fields so a
            # payload that fails the defensive decode ships the classic record unchanged.
            if wid:
                rec["weapon_id"] = wid
            if aim:
                rec.update(aim)
            sink["stationary"].append(rec)
        elif cls == "BarbedWire":
            # A hard movement obstacle the nav bake has no other way to see: the wire's own collider
            # is thin and often sits on a non-nav layer, so routes cross it freely today.
            sink["barbed_wire"].append({"pos": tpos, "name": name, "active": active, "lv": lv})
        elif cls == "WindowBreaker":
            # A breakable window is a SHORTCUT the nav grid treats as solid wall. Shipping the
            # positions lets a route explain "you can go through here", and eventually lets the bake
            # punch a door-like hole. The payload carries a scene id string; keep it for joins.
            wid, _ = read_cstr(pl, 0)
            rec = {"pos": tpos, "name": name, "active": active, "lv": lv}
            if wid:
                rec["id"] = wid
            sink["windows"].append(rec)
        elif cls == "ShootableQuestLocationObject":
            # Quest targets you SHOOT. quest_triggers already carries visit/place_item/flare; this
            # is the missing fourth kind, so it goes in the same list rather than a parallel one.
            sink["quest_triggers"].append({
                "pos": tpos, "name": name, "kind": "shoot", "outline": [],
                "active": active, "lv": lv,
            })
        elif cls == "LootableContainersGroup":
            gid, lo, hi = dec_lootgroup(pl)
            rec = {"pos": tpos, "name": name, "active": active, "lv": lv}
            if gid:
                rec["gid"] = gid
            if lo is not None:
                rec["min"] = lo
                rec["max"] = hi
            # tpid identifies the group's Transform; containers underneath it are its members.
            if tpid:
                group_tf[tpid] = rec
            sink["loot_groups"].append(rec)
        elif cls == "LootableContainer":
            cid, tpl = dec_container(pl)
            rec = {"pos": tpos, "name": name, "active": active, "lv": lv}
            if cid:
                rec["id"] = cid
            if tpl:
                rec["template"] = tpl
            # AUTHORITATIVE model join (loot glow): the container's own transform chain
            # (self, parent, grandparent), folded to u32 exactly like the geometry extractor
            # folds renderer `par`/`par2`. The assembler attaches the prefab's renderer
            # instances by ANCESTRY INTERSECTION — never by name or radius, which both missed
            # prefab parts with offset pivots and lit same-mesh DECORATIVE neighbours on
            # shelves (the streets weapon-box stack).
            _ch, _t = [], tpid
            for _ in range(3):
                if not _t:
                    break
                _ch.append(int((_t ^ (_t >> 32)) & 0xFFFFFFFF))
                _d = tt(_t, tr_obj)
                _t = (_d.get("m_Father") or {}).get("m_PathID", 0) if _d else 0
            if _ch:
                rec["tf"] = _ch
            # Attribute the container to the nearest ANCESTOR group. Membership is hierarchical in
            # the scene (a group is the parent GameObject), so walking up beats any spatial guess.
            g = group_of(tpid)
            if g is not None:
                rec["grp"] = g
            elif tpid:
                rec["_tpid"] = tpid   # resolved in the post-pass (its group may not be scanned yet)
            sink["containers"].append(rec)
        elif cls in DAMAGE_ZONE_CLASSES:
            box = cols[0] if cols else None
            sink["damage_zones"].append({
                "pos": col_center(M, box) if box else tpos, "name": name,
                "kind": DAMAGE_ZONE_CLASSES[cls],
                "outline": footprint(M, box) if box else [], "active": active, "lv": lv,
            })
        elif cls == "CardReader":
            rec = {"pos": tpos, "name": name or "Card reader", "active": active, "lv": lv}
            items = dec_card_reader(pl)
            if items:
                rec["item_ids"] = items
            sink["card_readers"].append(rec)
        elif cls == "RaidDialogEntryPoint":
            key, did = dec_dialog(pl)
            rec = {"pos": tpos, "name": name or "Dialog", "active": active, "lv": lv}
            if did:
                rec["id"] = did
            if key:
                rec["loc_key"] = key
            sink["dialogs"].append(rec)
        elif cls == "SpawnPointMarker":
            d = dec_spawn(pl)
            if d:
                sid, sname, pos, sides, cats, inf, bz_pid, core, sph_pid = d
                rec = {
                    "pos": bridge(pos), "name": sname, "side": SIDE_MASK.get(sides, str(sides)),
                    "categories_mask": cats, "infiltration": inf or None, "lv": lv,
                }
                # AI-scene tail fields, all omitted when absent so pre-audit consumers and
                # payloads that failed the defensive tail read see the classic record.
                if sid:
                    rec["id"] = sid
                if cats:
                    rec["categories"] = cat_names(cats)
                zone = mb_go_name(bz_pid)
                if zone:
                    rec["zone"] = zone
                r = sphere_radius(sph_pid)
                if r is not None:
                    rec["radius"] = r
                if core:
                    rec["core"] = core
                if ai:
                    rec["ai"] = True
                sink["spawn_points"].append(rec)
            else:
                rec = {"pos": tpos, "name": name, "side": None,
                       "categories_mask": None, "infiltration": None, "lv": lv}
                if ai:
                    rec["ai"] = True
                sink["spawn_points"].append(rec)
        elif cls in ("PatrolWay", "PatrolWayWithName", "PatrolWayWithConditions"):
            ways_raw.append((o.path_id, cls, name, pl))
        elif cls == "BotZone":
            zones_raw.append((o.path_id, name, pl))
        # ---- service-scene singles (Scripts / Culling / Sound; see the aux scan in main) ----
        elif cls == "AirdropPoint":
            # payload is empty — the Transform IS the data.
            sink["airdrop_points"].append({"pos": tpos, "name": name, "lv": lv})
        elif cls == "IndoorTrigger":
            box = cols[0] if cols else None
            sink["indoor_volumes"].append({
                "pos": col_center(M, box) if box else tpos, "name": name,
                "outline": footprint(M, box) if box else [], "lv": lv,
            })
        elif cls == "TOD_Sky":
            sink["_sun"].append(dec_tod_sky(pl))
        elif cls == "LevelBorder":
            lb = dec_level_border(pl)
            if lb:
                sink["_level_border"].append(lb)
        elif cls == "NavMeshDoorLink":
            d = dec_door_link(pl)
            if d:
                did, a, b = d
                sink["door_links"].append({"door": did, "a": bridge(a), "b": bridge(b), "lv": lv})
        elif cls == "AICorePoint":
            # id + connectivity group also sit in the GO name ("AICore ID:14 CG:27") — the
            # payload ints are the same values, byte-derived. CG = the game's own reachability
            # partition (nav-island ground truth).
            cid = int.from_bytes(pl[0:4], "little") if len(pl) >= 8 else None
            cg = int.from_bytes(pl[4:8], "little") if len(pl) >= 8 else None
            sink["core_points"].append({"pos": tpos, "id": cid, "cg": cg, "lv": lv})
        elif cls == "AIPlaceInfo":
            aid, _ = read_cstr(pl, 4)
            box = cols[0] if cols else None
            sink["ai_places"].append({
                "pos": col_center(M, box) if box else tpos, "id": (aid or "").strip() or None,
                "name": name, "outline": footprint(M, box) if box else [], "lv": lv,
            })
        elif cls == "CultistSignEffect":
            # Event ritual signs (HalloweenCultisSign / EventSectants GOs) — typed, so no name
            # guessing; interchange ships 92, woods 27. Position from the Transform.
            sink["cultist_signs"].append({"pos": tpos, "name": name, "active": active, "lv": lv})
        elif cls == "SpatialAudioRoom":
            box = cols[0] if cols else None
            sink["rooms"].append({
                "pos": col_center(M, box) if box else tpos, "name": name,
                "outline": footprint(M, box) if box else [], "lv": lv,
            })
        elif cls == "SpatialAudioPortal":
            # The edge is IN the GO name ("AudioPortal_FROM_<room>_TO_<room>") — no payload
            # decode needed for the room-and-doorway graph.
            mm = re.match(r"AudioPortal_FROM_(.+?)_TO_(.+)$", name or "")
            rec = {"pos": tpos, "lv": lv}
            if mm:
                rec["from"], rec["to"] = mm.group(1), mm.group(2)
            else:
                rec["name"] = name
            sink["room_portals"].append(rec)
        elif cls == "MineDirectional":
            # blast/trigger zone = the largest CHILD BoxCollider footprint (the mine GO itself
            # has none). Kind from the child name ("MON-50_MineTrigger" -> "MON-50").
            ol, cen, cname = largest_child_box(tpid) if tpid else (None, None, None)
            kind = (cname or "").split("_MineTrigger")[0] if cname and "_MineTrigger" in cname else None
            sink["mines_directional"].append({
                "pos": cen or tpos, "name": name, "kind": kind,
                "outline": ol or [], "active": active, "lv": lv,
            })
        elif cls in QUEST_TRIGGER_CLASSES:
            box = cols[0] if cols else None
            sink["quest_triggers"].append({
                "pos": col_center(M, box) if box else tpos,
                "name": dec_zone_id(pl) or name,
                "kind": QUEST_TRIGGER_CLASSES[cls],
                "outline": footprint(M, box) if box else [],
                "active": active, "lv": lv,
            })
        elif cls == "LighthouseTraderZone":
            box = cols[0] if cols else None
            sink["trader_zones"].append({
                "pos": col_center(M, box) if box else tpos, "name": name,
                "outline": footprint(M, box) if box else [], "active": active, "lv": lv,
            })
        elif cls == "BufferGateSwitcher":
            sink["buffer_switches"].append({
                "pos": tpos, "name": name or "Buffer gate switch", "kind": cls,
                "active": active, "lv": lv,
            })
        elif cls in BUFFER_ZONE_CLASSES:
            box = cols[0] if cols else None
            sink["buffer_zones"].append({
                "pos": col_center(M, box) if box else tpos, "name": name,
                "kind": BUFFER_ZONE_CLASSES[cls],
                "outline": footprint(M, box) if box else [], "active": active, "lv": lv,
            })
        elif cls == "LootPointsGroup":
            box = cols[0] if cols else None
            sink["loot_groups"].append({
                "pos": col_center(M, box) if box else tpos, "name": name or "Loot points group",
                "outline": footprint(M, box) if box else [], "active": active, "lv": lv,
            })
        elif cls == "LootPoint":
            guid, tps = dec_lootpoint(pl)
            if guid:
                sink["loose_points"].append({
                    "pos": tpos, "name": name, "guid": guid, "templates": tps,
                    "active": active, "lv": lv,
                })

    # ---- loot-group post-pass ---------------------------------------------------------------
    # A container can be scanned BEFORE the group that owns it, so anything still unattributed is
    # re-walked now that every group on this level is known. The memo is cleared first: entries
    # cached during the main loop may have concluded "no group" only because it had not been seen.
    group_of_cache.clear()
    for _rec in sink["containers"]:
        _tp = _rec.pop("_tpid", None)
        if _tp is not None and "grp" not in _rec:
            _g = group_of(_tp)
            if _g is not None:
                _rec["grp"] = _g
    # Turn (min, max, member count) into the per-container spawn probability the value model wants.
    # Members are counted from the attribution above rather than trusted from the payload, so the
    # number always matches the containers actually shipped for this level.
    _members = Counter(r["grp"] for r in sink["containers"] if r.get("grp"))
    for _g in sink["loot_groups"]:
        _gid = _g.get("gid")
        if not _gid or _g.get("min") is None:
            continue
        _n = _members.get(_gid, 0)
        _g["members"] = _n
        if _n:
            _g["p"] = round(min(1.0, ((_g["min"] + _g["max"]) / 2.0) / _n), 4)
    _p = {g["gid"]: g["p"] for g in sink["loot_groups"] if g.get("gid") and g.get("p") is not None}
    for _rec in sink["containers"]:
        _pp = _p.get(_rec.get("grp"))
        if _pp is not None:
            _rec["grp_p"] = _pp

    # ---- AI-scene post-pass: patrol ways + the bot-zone registry -----------------------------
    # PatrolPoint payloads are all zero — each point's POSITION is its Transform. A way's zone
    # comes from the BotZone side (its serialized PatrolWay array); a MARKER's zone came from
    # its own BotZone PPtr above. Counts/hulls are rebuilt in main() AFTER the Id dedupe so
    # variant AI scenes (Sandbox_AI + _high) union instead of first-scene-wins.
    if ways_raw or zones_raw:
        def point_pos(pid):
            h = mb_hdr(pid)
            if not h:
                return None
            _, ptp, _ = go_info((h.get("m_GameObject") or {}).get("m_PathID"))
            return bridge(world_mat(ptp)[:3, 3]) if ptp else None

        parsed, n_bad = {}, 0
        for wpid, wcls, gname, wpl in ways_raw:
            d = dec_patrol_way(wpl)
            pts = [p for p in (point_pos(pp) for pp in d[0])] if d else []
            pts = [p for p in pts if p]
            # 1-point ways are REAL data (Patrol_Killa_alarm1..6 — alarm response posts, not
            # routes); the viewer renders them as a dot with no polyline.
            if not d or not pts:
                n_bad += 1
                continue
            rec = {
                "name": d[1] or gname,
                "kind": {"PatrolWay": "patrol", "PatrolWayWithName": "named",
                         "PatrolWayWithConditions": "conditional"}[wcls],
                "points": pts, "lv": lv,
            }
            # The serialized route id repeats (KILLA_PATROL_ALT x6 alarm posts) while the GO
            # name stays distinct (Patrol_Killa_alarm1..6) — ship both when they differ.
            if d[1] and gname and gname != d[1]:
                rec["go"] = gname
            parsed[wpid] = rec
        way_zone = {}
        way_pids = {w[0] for w in ways_raw}
        for _zpid, zname, zpl in zones_raw:
            for kind, pids in locate_pptr_arrays(zpl, {"ways": way_pids}):
                for wp in pids:
                    way_zone[wp] = zname
            sink["_zones_reg"].append({"name": zname, "lv": lv})
        for wpid, rec in parsed.items():
            rec["zone"] = way_zone.get(wpid)
            sink["patrol_ways"].append(rec)
        if n_bad:
            print(f"[level{lv}] {n_bad} patrol way(s) failed the defensive decode - skipped")

    print(f"[level{lv}] {len(objs)} objs, {len(mbs)} MBs -> {n_hit} typed hits ({time.time()-t0:.0f}s)")
    del env, objs, go_obj, tr_obj, col_obj, mbs, tt_cache, go_tt_cache, wm_cache
    gc.collect()


def dedupe(rows, keyf):
    """cross-level dedupe; an ACTIVE row wins over an inactive twin."""
    best = {}
    for r in rows:
        k = keyf(r)
        if k not in best or (r.get("active") and not best[k].get("active")):
            best[k] = r
    return list(best.values())


# ---------------------------------------------------------------------------
# TERRAIN HEIGHTFIELD (Task: drape zone outlines on the ground)
# ---------------------------------------------------------------------------
class TerrainField:
    """World-XZ heightfield from the map's .eftpack FLAG_TERRAIN instances (pack space =
    viewer space, so outline verts sample it directly). Vertices are binned to a CELL-metre
    grid (mean Y per cell — terrain slices overlap at seams); sampling is a bilinear blend of
    the 4 surrounding cell centres, degrading to the plain cell mean where neighbours are
    missing and to None off the grid. Same data as eft_pipeline/build_grass.py's uv->world
    grids, but keyed by world XZ (we need height AT a point, not point AT a uv)."""
    CELL = 2.0  # instances are filtered on FLAG_TERRAIN = 1<<1 (build_grass.py's contract)

    def __init__(self, pack_dir):
        mani = json.load(open(os.path.join(pack_dir, "manifest.json")))
        mb = open(os.path.join(pack_dir, "meshes.bin"), "rb").read()
        ib = open(os.path.join(pack_dir, "instances.bin"), "rb").read()
        vl = mani["vertex"]
        vs = vl["stride"]
        poff = next(a for a in vl["attrs"] if a["name"] == "position")["offset"]
        inst = mani["instance"]
        istride = inst["stride"]
        fo = {f["name"]: f["offset"] for f in inst["fields"]}
        id2mesh = {m["id"]: m for m in mani["meshes"]}
        pts = []
        for i in range(len(ib) // istride):
            b = i * istride
            if not struct.unpack_from("<I", ib, b + fo["flags"])[0] & 2:
                continue
            a = np.array(struct.unpack_from("<12f", ib, b + fo["affine"]),
                         np.float64).reshape(3, 4)
            me = id2mesh[struct.unpack_from("<I", ib, b + fo["meshId"])[0]]
            n, off = me["vtxCount"], me["vtxOffset"]
            vb = np.frombuffer(mb, np.uint8, count=n * vs, offset=off).reshape(n, vs)
            loc = vb[:, poff:poff + 12].copy().view("<f4").astype(np.float64)
            pts.append(loc @ a[:, :3].T + a[:, 3])
        if not pts:
            raise ValueError("no FLAG_TERRAIN instances")
        w = np.concatenate(pts)
        c = self.CELL
        ix = np.floor(w[:, 0] / c).astype(np.int64)
        iz = np.floor(w[:, 2] / c).astype(np.int64)
        self.x0, self.z0 = int(ix.min()), int(iz.min())
        nx, nz = int(ix.max()) - self.x0 + 1, int(iz.max()) - self.z0 + 1
        s = np.zeros((nx, nz), np.float64)
        n = np.zeros((nx, nz), np.int64)
        np.add.at(s, (ix - self.x0, iz - self.z0), w[:, 1])
        np.add.at(n, (ix - self.x0, iz - self.z0), 1)
        self.h = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        self.n_verts, self.n_cells = len(w), int((n > 0).sum())

    def sample(self, x, z):
        """terrain height at world (x, z) or None when off-grid."""
        c = self.CELL
        fx, fz = x / c - 0.5 - self.x0, z / c - 0.5 - self.z0
        x0, z0 = int(np.floor(fx)), int(np.floor(fz))
        tx, tz = fx - x0, fz - z0
        acc = wsum = 0.0
        for dx, dz, wgt in ((0, 0, (1 - tx) * (1 - tz)), (1, 0, tx * (1 - tz)),
                            (0, 1, (1 - tx) * tz), (1, 1, tx * tz)):
            xi, zi = x0 + dx, z0 + dz
            if 0 <= xi < self.h.shape[0] and 0 <= zi < self.h.shape[1]:
                v = self.h[xi, zi]
                if np.isfinite(v) and wgt > 0.0:
                    acc += v * wgt
                    wsum += wgt
        return acc / wsum if wsum > 1e-6 else None


def load_terrain_field():
    """the map's pack heightfield, or None (indoor maps / pack absent) — draping is optional."""
    pack = os.environ.get("EFT_PACK_DIR") or os.path.join(
        os.path.dirname(KIT), "packs", f"{MAP}.eftpack")
    if not os.path.exists(os.path.join(pack, "manifest.json")):
        print(f"[drape] no pack at {pack} - outlines stay at collider Y")
        return None
    try:
        tf = TerrainField(pack)
        print(f"[drape] heightfield from {pack}: {tf.n_verts} terrain verts -> "
              f"{tf.n_cells} cells @ {tf.CELL} m")
        return tf
    except ValueError:
        print(f"[drape] pack has no terrain instances (indoor map) - outlines stay at collider Y")
        return None
    except Exception as ex:
        print(f"[drape] heightfield failed ({type(ex).__name__}: {ex}) - outlines stay at collider Y")
        return None


def drape_outline(outline, field, step=4.0, lift=0.3):
    """subdivide each closed-outline edge every ~`step` m and set vert Y to
    max(terrain + lift, collider_base_y); base Y interpolates along the edge and is kept
    wherever the terrain grid has no data. Returns the new vert list."""
    n = len(outline)
    if n < 3 or field is None:
        return outline
    out = []
    for i in range(n):
        a, b = outline[i], outline[(i + 1) % n]
        seg = math.hypot(b[0] - a[0], b[2] - a[2])
        k = max(1, int(math.ceil(seg / step)))
        for j in range(k):
            t = j / k
            x = a[0] + (b[0] - a[0]) * t
            y0 = a[1] + (b[1] - a[1]) * t
            z = a[2] + (b[2] - a[2]) * t
            ty = field.sample(x, z)
            y = max(ty + lift, y0) if ty is not None else y0
            out.append([round(x, 2), round(y, 2), round(z, 2)])
    return out


def outline_extent(outline):
    """[w, d] metres of the (pre-subdivision) rectangular footprint, for the viewer cards."""
    if len(outline) < 3:
        return None
    d = lambda a, b: math.dist(a, b)
    w, h = d(outline[0], outline[1]), d(outline[1], outline[2])
    return [round(w, 1), round(h, 1)] if w > 0.05 and h > 0.05 else None


# Ground-hugging zones (colliders that genuinely sit on the terrain): drape to the ground so
# the outline follows undulating terrain instead of floating/sinking at the flat collider face.
# damage_zones (gas fires / furnace mouths) sit on the ground like the rest.
DRAPE_KEYS = ("exfils", "transit_points", "quest_triggers", "trader_zones",
              "buffer_zones", "loot_groups", "damage_zones")
# Elevated collider zones (minefields, sniper zones, directional mines): keep the collider's own
# world height. Their trigger boxes are frequently TALL volumes whose bottom face reaches the base
# terrain far below a raised platform (e.g. ground_zero Minefield_LowPower: collider center Y=15.65
# on a train platform, but bottom face Y<=-0.41 at ground). Draping to terrain snapped the whole
# zone to the ground far below where the mines actually are. USER PREFERENCE: use the collider's
# actual height (its center, == the marker `pos`), NOT the terrain drape.
COLLIDER_HEIGHT_KEYS = ("minefields", "sniper_zones", "mines_directional")


def drape_zones(sink):
    """Place every zone outline (and stamp its pre-subdivision extent):
      * DRAPE_KEYS          -> subdivide ~4 m and lift to the terrain (ground-hugging zones).
      * COLLIDER_HEIGHT_KEYS -> keep the collider's OWN height (the marker `pos` Y), never terrain.
    Returns (terrain_field_loaded?, verts_before, verts_after) for the drape group's logging."""
    field = load_terrain_field()
    before = after = 0
    for k in DRAPE_KEYS:
        for r in sink[k]:
            ol = r.get("outline") or []
            if len(ol) < 3:
                continue
            ext = outline_extent(ol)
            if ext:
                r["extent"] = ext
            before += len(ol)
            r["outline"] = drape_outline(ol, field)
            after += len(r["outline"])
    # Bot-zone hulls are SYNTHETIC boundaries (convex hull of spawn markers + patrol points,
    # all at ground height) — drape them so a long hull edge follows the terrain instead of
    # cutting through a hill. Patrol way POINTS are the game's own ordered route verts and are
    # deliberately NOT draped/subdivided (synthetic verts would masquerade as real points).
    for r in sink["bot_zones"]:
        hl = r.get("hull") or []
        if len(hl) >= 3:
            before += len(hl)
            r["hull"] = drape_outline(hl, field)
            after += len(r["hull"])
    # The LevelBorder polygon sits at ONE fixed authored Y (interchange: 14.9 across the whole
    # 1 km map) — drape it so the boundary ring rides the terrain instead of underground.
    sink["_level_border"] = [drape_outline(v, field) if len(v) >= 3 else v
                             for v in sink["_level_border"]]
    # Elevated collider zones: FLATTEN each outline to the collider center height (`pos` Y), which
    # is exactly where the marker sphere sits, so the ring/wall/marker all read at the platform
    # level. NO terrain sampling, NO subdivision (the footprint is already a horizontal rectangle).
    for k in COLLIDER_HEIGHT_KEYS:
        for r in sink[k]:
            ol = r.get("outline") or []
            if len(ol) < 3:
                continue
            ext = outline_extent(ol)
            if ext:
                r["extent"] = ext
            raw_ys = [p[1] for p in ol]
            y = round(r["pos"][1], 2)  # collider center height == the marker position
            print(f"[collider-height] {k} {r.get('name')!r:36s} pos_y={r['pos'][1]:.2f} "
                  f"footprint_y=[{min(raw_ys):.2f},{max(raw_ys):.2f}] -> outline_y={y}")
            r["outline"] = [[round(p[0], 2), y, round(p[2], 2)] for p in ol]
    return field is not None, before, after


# ---------------------------------------------------------------------------
# tarkov.dev RESOLUTION + lootLoose JOIN (loose_points) — all failures degrade to offline.
# Source is json.tarkov.dev's pre-generated catalogs via tarkov_static (ETag disk cache),
# NOT the GraphQL API — GraphQL 503s routinely and the static dumps are the supported feed
# tarkov.dev's own apps consume.
# ---------------------------------------------------------------------------
# tarkov.dev display name for the per-map lootLoose lookup. Unlisted maps fall back to a
# title-cased key (tarkov_static matches display name OR its normalizedName slug).
DEV_NAME = {"lighthouse": "Lighthouse", "factory": "Factory", "factory_rework": "Factory",
            "labs": "The Lab",
            "streets": "Streets of Tarkov", "ground_zero": "Ground Zero", "labyrinth": "The Labyrinth"}


def resolve_templates(loose):
    """template id -> {'n','s','pr','cat'} via the static items catalog (items + itemCategories).
    cat=1 marks a CATEGORY template ('Food and drink' pool slot, no price/icon)."""
    ids = sorted({t for r in loose for t in r["templates"]})
    if not ids or os.environ.get("EFT_GAMEDATA_OFFLINE"):
        return {}
    idx = {}
    try:
        import tarkov_static
        idx = tarkov_static.load_static_item_index(ids)
        print(f"[loose] resolved {len(idx)}/{len(ids)} template ids via json.tarkov.dev "
              f"({sum(1 for v in idx.values() if v['cat'])} categories)")
    except (SystemExit, Exception) as ex:                        # no cache + no network
        print(f"[loose] template resolution OFFLINE ({ex}) - shipping raw template ids")
    return idx


def join_dev_loose(loose):
    """nearest tarkov.dev lootLoose point per first-party point -> r['dev_d'] (m) + items for
    template-less points within 2.5 m. Prints the match-distance distribution."""
    if not loose or os.environ.get("EFT_GAMEDATA_OFFLINE"):
        return
    name = DEV_NAME.get(MAP, MAP.replace("_", " ").title())
    try:
        import tarkov_static
        rows = tarkov_static.load_static_loose(name)
    except (SystemExit, Exception) as ex:
        print(f"[loose] lootLoose join OFFLINE ({ex})")
        return
    pts = []
    for ll in rows:
        p = ll.get("position") or {}
        if all(k in p for k in "xyz"):
            pts.append(([-p["x"], p["y"], p["z"]], ll.get("items") or []))  # dev -> viewer bridge
    if not pts:
        print(f"[loose] tarkov.dev lootLoose('{name}') returned 0 points - no join")
        return
    P = np.array([p for p, _ in pts])
    ds = []
    for r in loose:
        q = np.array(r["pos"])
        i = int(np.argmin(((P - q) ** 2).sum(axis=1)))
        dist = float(np.linalg.norm(P[i] - q))
        r["dev_d"] = round(dist, 2)
        ds.append(dist)
        # a point whose payload had NO pool still gets the snapshot's items when co-located
        if not r.get("items") and dist <= 2.5:
            best = sorted(pts[i][1], key=lambda it: -(it.get("avg24hPrice") or 0))[:4]
            r["items"] = [{"n": it.get("name"), "s": it.get("shortName"),
                           "pr": it.get("avg24hPrice"), "cat": 0} for it in best]
            r["items_src"] = "tarkov.dev"
    ds = np.array(ds)
    print(f"[loose] join vs {len(pts)} tarkov.dev lootLoose points: "
          f"median {np.median(ds):.1f} m, p90 {np.percentile(ds, 90):.1f} m, max {ds.max():.1f} m; "
          f"<=1m {(ds <= 1).sum()}, <=2m {(ds <= 2).sum()}, <=5m {(ds <= 5).sum()} of {len(ds)}")


def finalize_loose(sink):
    """guid-dedupe (scene variants re-serialize the same rack), then merge same-name points
    within 0.5 m into one map point (a rack has several spawn SLOTS at ~one spot): n = slot
    count, templates = union. Then resolve templates + join tarkov.dev."""
    best = {}
    for r in sink["loose_points"]:
        k = r["guid"]
        if k not in best or (r.get("active") and not best[k].get("active")):
            best[k] = r
    merged = []
    for r in best.values():
        hit = None
        for m in merged:
            if m["name"] == r["name"] and math.dist(m["pos"], r["pos"]) <= 0.5:
                hit = m
                break
        if hit is None:
            merged.append({"pos": r["pos"], "name": r["name"], "n": 1,
                           "templates": list(r["templates"]),
                           "active": r["active"], "lv": r["lv"]})
        else:
            hit["n"] += 1
            hit["active"] = hit["active"] or r["active"]
            for t in r["templates"]:
                if t not in hit["templates"]:
                    hit["templates"].append(t)
    idx = resolve_templates(merged)
    for r in merged:
        items = []
        for t in r["templates"]:
            e = idx.get(t)
            items.append({"tpl": t, **e} if e else {"tpl": t})
        # priced real items first, categories last — the viewer titles the card off items[0]
        items.sort(key=lambda it: (it.get("cat", 0), -(it.get("pr") or 0)))
        if items:
            r["items"] = items
            r["items_src"] = "game files"  # the POOL is client data; names/prices are lookups
        del r["templates"]
    join_dev_loose(merged)
    sink["loose_points"] = merged


def resolve_new_intel(sink):
    """Display-name resolution for the typed additions, all via the tarkov_static disk-cached
    dumps (zero GraphQL) and all optional — offline or unresolvable ids just ship raw:
      * containers:   template -> `tpl_name` (static lootContainers: jacket / duffle-bag / ...)
      * card_readers: accepted-card `item_ids` -> `items` [{id, n}] (static items)
      * stationary:   weapon name, id-first then a NEAREST-POSITION join (<= 3 m) against the
        static per-map stationaryWeapons (same convention as the door<->lock 2 m and
        LootPoint<->lootLoose joins) -> `weapon_name`. The game serializes a PRESET-style id
        tarkov.dev does not index, so the position join is what actually names the mounts.
    """
    if os.environ.get("EFT_GAMEDATA_OFFLINE"):
        return
    if not (sink["containers"] or sink["card_readers"] or sink["stationary"]):
        return
    try:
        sys.path.insert(0, HERE)
        import tarkov_static
        if sink["containers"]:
            tbl = tarkov_static.load_static_containers()
            n = 0
            for r in sink["containers"]:
                nm = tbl.get(r.get("template") or "")
                if nm:
                    r["tpl_name"] = nm
                    n += 1
            print(f"[containers] {n}/{len(sink['containers'])} templates named "
                  f"(static lootContainers)")
        ids = sorted({i for r in sink["card_readers"] for i in r.get("item_ids") or []})
        if ids:
            got = {it["id"]: it["name"]
                   for it in tarkov_static.load_static_items(ids=ids).get("items") or []
                   if it.get("id") and it.get("name")}
            for r in sink["card_readers"]:
                items = [{"id": i, **({"n": got[i]} if i in got else {})}
                         for i in r.get("item_ids") or []]
                if items:
                    r["items"] = items
                    del r["item_ids"]
            print(f"[card_readers] {len(got)}/{len(ids)} accepted-card ids named")
        if sink["stationary"]:
            sw = tarkov_static.load_static_stationary()
            dev = sw["maps"].get(tarkov_static.map_slug(DEV_NAME.get(MAP, MAP))) or []
            n = 0
            for r in sink["stationary"]:
                nm = sw["weapons"].get(r.get("weapon_id") or "")
                if not nm and dev:
                    # dev positions are raw Unity; bridge to viewer space like everything else.
                    best = min(dev, key=lambda w: math.dist(r["pos"], bridge(w["pos"])))
                    if math.dist(r["pos"], bridge(best["pos"])) <= 3.0:
                        nm = sw["weapons"].get(best["id"])
                if nm:
                    r["weapon_name"] = nm
                    n += 1
            print(f"[stationary] {n}/{len(sink['stationary'])} mounts named "
                  f"({len(dev)} tarkov.dev positions on this map)")
    except Exception as ex:
        print(f"[resolve] static resolution OFFLINE/failed ({type(ex).__name__}: {ex}) "
              f"- shipping raw ids")


def sibling_levels(scanned):
    """AUTO-PROBE for the gameplay-logic scene: the map config's level list is the GEOMETRY
    set and may not include it (factory: exfils live in Factory_DesignStuff = level 68, not in
    levels 2/69/70/177). Candidates = every BuildSettings scene in the SAME directory as the
    already-scanned levels' scenes (data-driven, no per-map constants)."""
    try:
        env = UnityPy.load(os.path.join(DATA, "globalgamemanagers"))
        scenes = None
        for o in env.objects:
            if o.type.name == "BuildSettings":
                d = o.read_typetree()
                scenes = d.get("scenes") or d.get("m_Scenes") or []
                break
        if not scenes:
            return []
        dirs = {os.path.dirname(scenes[lv]) for lv in scanned if 0 <= lv < len(scenes)}
        cand = [i for i, s in enumerate(scenes)
                if os.path.dirname(s) in dirs and i not in scanned]
        print(f"[auto-probe] no exfils in the config levels; probing {len(cand)} sibling "
              f"scenes: {cand}")
        return cand
    except Exception as ex:
        print(f"[auto-probe] failed: {type(ex).__name__}: {ex}")
        return []


def service_levels(*tokens):
    """SERVICE-scene level indices for this map: every BuildSettings scene in the config's
    `source.unity_location` folder whose basename has one of `tokens` as an underscore-token
    (Shopping_Mall_AI = level 66, Shopping_Mall_Scripts = 53, Shopping_Mall_Culling = 521).
    The geometry configs exclude these ON PURPOSE (placeholder cubes / cultist-sign quads —
    gen_maps SERVICE_TOKENS), so they are scanned IN ADDITION to LEVELS, never merged into
    them. Duplicate BuildSettings rows (Terminal_AI at 635 AND 687 — same scene path) collapse
    to the first; genuine VARIANT scenes (Sandbox_AI + Sandbox_AI_high = Ground Zero 21+,
    Laboratory_dark_AI = event Labs) ALL scan — the spawn Id dedupe unions them and each
    record's `lv` keeps the variant it came from."""
    folder = ((_cfg.get("source") or {}).get("unity_location") or "").lower()
    if not folder:
        return []
    want = {t.lower() for t in tokens}
    try:
        env = UnityPy.load(os.path.join(DATA, "globalgamemanagers"))
        scenes = []
        for o in env.objects:
            if o.type.name == "BuildSettings":
                d = o.read_typetree()
                scenes = d.get("scenes") or d.get("m_Scenes") or []
                break
        marker = "Assets/Content/Locations/"
        out, seen = [], set()
        for i, s in enumerate(scenes):
            p = s.replace("\\", "/")
            j = p.find(marker)
            if j < 0:
                continue
            rest = p[j + len(marker):]
            f = rest.split("/", 1)[0] if "/" in rest else rest.rsplit(".", 1)[0]
            base = os.path.basename(p).rsplit(".", 1)[0]
            if (f.lower() == folder and want & {t.lower() for t in base.split("_")}
                    and i not in LEVELS and p not in seen):
                seen.add(p)
                out.append(i)
        return out
    except Exception as ex:
        print(f"[service-levels] failed: {type(ex).__name__}: {ex}")
        return []


def main():
    print(f"[cfg] map={MAP} levels={LEVELS} G3={G3.round(2).tolist()}")
    sink = {k: [] for k in ("exfils", "minefields", "sniper_zones", "doors",
                            "transit_points", "stationary", "spawn_points",
                            "mines_directional", "loose_points",
                            "quest_triggers", "trader_zones", "buffer_switches",
                            "buffer_zones", "loot_groups",
                            # typed additions (2026-07 audit); OMITTED from the output when
                            # empty so maps without them keep byte-identical gamedata.json.
                            "containers", "damage_zones", "card_readers", "dialogs",
                            # movement/progression additions (2026-07): barbed wire + breakable
                            # windows shape ROUTES; xp triggers are progression value.
                            "barbed_wire", "windows",
                            # AI-scene additions (spawn/patrol audit): _zones_reg is internal
                            # (name registry per scan), consumed by the bot_zones build below.
                            "patrol_ways", "bot_zones", "_zones_reg",
                            # service-scene additions (2026-07 second audit): airdrop landing
                            # candidates + indoor volumes + door-traversal nav links + AI core
                            # graph + bot home zones + event cultist signs + the audio room
                            # graph. _sun/_level_border collapse to top-level scalars below.
                            "airdrop_points", "indoor_volumes", "door_links", "core_points",
                            "ai_places", "cultist_signs", "rooms", "room_portals",
                            "_sun", "_level_border")}
    t0 = time.time()
    scanned = list(LEVELS)
    for lv in LEVELS:
        scan_level(lv, sink)
    # the AI scene (scav/boss spawn markers, patrol ways, bot zones) is a SERVICE scene the
    # geometry level list excludes — scan it additionally (0.1-0.4 s per scene).
    ai = service_levels("ai")
    if ai:
        print(f"[ai-levels] scanning AI scene(s): {ai}")
    for lv in ai:
        scan_level(lv, sink, ai=True)
    scanned += ai
    # More service scenes, same rule: Scripts (airdrop points / indoor volumes / TOD_Sky sun
    # model), Culling (LevelBorder playable-area polygon), Sound (audio room graph — SOME
    # configs already list the Sound scene as a geometry level; the `not in LEVELS` filter
    # inside service_levels keeps those from double-scanning).
    aux = (service_levels("scripts") + service_levels("culling", "levelborders")
           + service_levels("sound"))
    aux = [lv for lv in aux if lv not in scanned]
    if aux:
        print(f"[aux-levels] scanning service scene(s): {aux}")
    for lv in aux:
        scan_level(lv, sink)
    scanned += aux
    # the logic scene (the one carrying the exfil MBs) may sit outside the config's
    # geometry-level list — probe the sibling scenes for it.
    if not sink["exfils"]:
        extra = sibling_levels(scanned)
        for lv in extra:
            scan_level(lv, sink)
        scanned += extra

    sink["exfils"] = dedupe(sink["exfils"], lambda r: (r["faction"], r["name"]))
    localize_exfils(sink["exfils"])
    sink["doors"] = dedupe(sink["doors"], lambda r: r["id"] or (r["name"], tuple(r["pos"])))
    for k in ("minefields", "sniper_zones", "transit_points", "stationary", "mines_directional",
              "quest_triggers", "trader_zones", "buffer_switches", "buffer_zones", "loot_groups",
              "damage_zones", "card_readers", "dialogs",
              "barbed_wire", "windows"):
        sink[k] = dedupe(sink[k], lambda r: (r.get("name"), tuple(r["pos"])))
    # containers re-serialize across scene variants; the Id string is the stable key.
    sink["containers"] = dedupe(sink["containers"],
                                lambda r: r.get("id") or (r.get("name"), tuple(r["pos"])))
    # spawn markers: the serialized GUID is the stable key (Terminal's AI scene sits TWICE in
    # BuildSettings; Ground Zero ships two variant AI scenes sharing 83 of ~100 markers).
    sink["spawn_points"] = dedupe(sink["spawn_points"],
                                  lambda r: r.get("id") or (r.get("name"), tuple(r["pos"])))
    sink["patrol_ways"] = dedupe(sink["patrol_ways"],
                                 lambda r: (r.get("name"), r.get("zone"),
                                            tuple(tuple(p) for p in r["points"])))
    for k in ("airdrop_points", "indoor_volumes", "cultist_signs", "rooms", "ai_places"):
        sink[k] = dedupe(sink[k], lambda r: (r.get("name"), tuple(r["pos"])))
    sink["door_links"] = dedupe(sink["door_links"], lambda r: (r["door"], tuple(r["a"])))
    sink["core_points"] = dedupe(sink["core_points"],
                                 lambda r: (r.get("id"), r.get("cg"), tuple(r["pos"])))
    sink["room_portals"] = dedupe(
        sink["room_portals"],
        lambda r: (r.get("from"), r.get("to"), r.get("name"), tuple(r["pos"])))
    # ---- bot zones: names registered per scan; counts/hulls/centroid rebuilt HERE, after the
    # Id dedupe, so a zone spanning two variant AI scenes gets the UNION of its members. The
    # hull is the convex hull of the zone's spawn markers + patrol points (BotZone itself has
    # no collider — verified on all 12 interchange zones).
    reg = {}
    for z in sink.pop("_zones_reg"):
        reg.setdefault(z["name"], z)
    for name in sorted(reg):
        zs = [s for s in sink["spawn_points"] if s.get("zone") == name]
        zw = [w for w in sink["patrol_ways"] if w.get("zone") == name]
        pts = [tuple(s["pos"]) for s in zs] + [tuple(p) for w in zw for p in w["points"]]
        if not pts:
            continue   # a zone with no members has no anchor to show
        cen = [round(sum(p[i] for p in pts) / len(pts), 2) for i in range(3)]
        sink["bot_zones"].append({
            "name": name, "pos": cen, "hull": hull_xz(pts),
            "n_spawns": len(zs), "n_ways": len(zw), "lv": reg[name]["lv"],
        })
    # Friendly zone display names from the static maps_en catalog — the SAME table tarkov.dev
    # renders its bosses' spawnLocations from ("ZoneCenterBot" -> "Center", "ZoneWoodCutter" ->
    # "Lumber Mill"), so the viewer's boss->zone join becomes an EXACT string match instead of
    # a substring heuristic. Omitted when offline with no cache; everything degrades.
    if sink["bot_zones"] and not os.environ.get("EFT_GAMEDATA_OFFLINE"):
        try:
            import tarkov_static
            zen = tarkov_static.load_static_zone_names()
            hit = 0
            for z in sink["bot_zones"]:
                en = zen.get(z["name"])
                if en:
                    z["en"] = en
                    hit += 1
            print(f"[zones] {hit}/{len(sink['bot_zones'])} bot zones named via json.tarkov.dev")
        except (SystemExit, Exception) as ex:
            print(f"[zones] zone names OFFLINE ({ex}) - raw ids only")
    # display names for the typed additions (static dumps; degrades to raw ids offline).
    resolve_new_intel(sink)

    # first-party loose loot: guid-dedupe + slot-merge, then tarkov.dev resolution + join.
    finalize_loose(sink)
    # terrain-drape every zone outline (subdivide ~4 m; Y = max(terrain+0.3, collider base)).
    draped, ol_before, ol_after = drape_zones(sink)
    if draped:
        print(f"[drape] outline verts {ol_before} -> {ol_after}")

    logic_levels = sorted({e["lv"] for e in sink["exfils"]})
    # New sinks are dropped entirely (data AND count) when empty: a map without them keeps a
    # byte-identical gamedata.json across this extractor change.
    NEW_SINKS = ("containers", "damage_zones", "card_readers", "dialogs",
                 "barbed_wire", "windows",
                 "patrol_ways", "bot_zones",
                 "airdrop_points", "indoor_volumes", "door_links", "core_points",
                 "ai_places", "cultist_signs", "rooms", "room_portals")
    # Top-level scalars from the service scenes (omitted entirely when absent).
    sun = next((s for s in sink.pop("_sun") if s), None)
    borders = [v for v in sink.pop("_level_border") if v]
    ship = {k: v for k, v in sink.items() if v or k not in NEW_SINKS}
    counts = {k: len(v) for k, v in ship.items()}
    counts["exfils_by_faction"] = dict(Counter(e["faction"] for e in sink["exfils"]))
    counts["doors_with_key"] = sum(1 for d in sink["doors"] if d.get("key_id"))
    out = {"map": MAP, "generated_levels": scanned, "logic_levels": logic_levels,
           "draped": draped, "counts": counts, **ship}
    if sun:
        out["sun"] = sun
        print(f"  sun: {sun['hour']:.1f}h {sun['day']}/{sun['month']}/{sun['year']} "
              f"lat {sun['lat']} lon {sun['lon']}")
    if borders:
        out["level_border"] = max(borders, key=len)   # variant scenes: keep the fullest ring
        counts["level_border"] = len(out["level_border"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), separators=(",", ":"))
    print(f"\n[out] {OUT}  ({os.path.getsize(OUT)/1e3:.0f} kB, {time.time()-t0:.0f}s)")
    print("[counts]", json.dumps(counts))
    for e in sink["exfils"]:
        print(f"  exfil [{e['faction']:6s}] {e['name']:34s} pos={e['pos']} outline_pts={len(e['outline'])} active={e['active']}")
    for t in sink["transit_points"]:
        print(f"  transit {t['name']:24s} pos={t['pos']}")
    for s in sink["stationary"]:
        arc = (f" yaw={s['yaw']} arc={s['yaw_range']} pitch={s['pitch_range']}"
               if "yaw_range" in s else "")
        print(f"  stationary {s['name']:12s} pos={s['pos']} "
              f"weapon={s.get('weapon_name') or s.get('weapon_id') or '?'}{arc}")
    if sink["containers"]:
        ck = Counter(c.get("tpl_name") or c.get("template") or "?" for c in sink["containers"])
        print(f"  containers: {len(sink['containers'])} typed lootable(s) {dict(ck)}")
    for z in sink["damage_zones"]:
        print(f"  damage_zone [{z.get('kind')}] {str(z.get('name'))[:32]:32s} pos={z['pos']} "
              f"outline_pts={len(z['outline'])} active={z['active']}")
    for r in sink["card_readers"]:
        cards = [it.get("n") or it["id"] for it in r.get("items") or []] or r.get("item_ids") or []
        print(f"  card_reader {r['name']} pos={r['pos']} accepts={cards}")
    for r in sink["dialogs"]:
        print(f"  dialog {r['name']} pos={r['pos']} id={r.get('id')}")
    if sink["bot_zones"]:
        print(f"  bot_zones: {len(sink['bot_zones'])}  " + "  ".join(
            f"{z['name']}({z['n_spawns']}s/{z['n_ways']}w{'' if z['hull'] else ',no-hull'})"
            for z in sink["bot_zones"]))
    if sink["patrol_ways"]:
        pk = Counter(w["kind"] for w in sink["patrol_ways"])
        zoned = sum(1 for w in sink["patrol_ways"] if w.get("zone"))
        print(f"  patrol_ways: {len(sink['patrol_ways'])} kinds={dict(pk)} zoned={zoned} "
              f"pts={sum(len(w['points']) for w in sink['patrol_ways'])}")
    sc_ai = Counter((s.get("side"), bool(s.get("ai"))) for s in sink["spawn_points"])
    print(f"  spawn_points: {len(sink['spawn_points'])} by (side, ai): {dict(sc_ai)}")
    aux_counts = {k: len(sink[k]) for k in
                  ("airdrop_points", "indoor_volumes", "door_links", "core_points",
                   "ai_places", "cultist_signs", "rooms", "room_portals") if sink[k]}
    if aux_counts:
        print(f"  service-scene intel: {aux_counts}")
    st = Counter(d["state"] for d in sink["doors"])
    print(f"  doors: {len(sink['doors'])} states={dict(st)} with_key={counts['doors_with_key']}")
    mk = Counter(m.get("kind") for m in sink["mines_directional"])
    qk = Counter(q.get("kind") for q in sink["quest_triggers"])
    print(f"  mines_directional: {len(sink['mines_directional'])} kinds={dict(mk)} "
          f"with_outline={sum(1 for m in sink['mines_directional'] if m.get('outline'))}")
    print(f"  quest_triggers: {len(sink['quest_triggers'])} kinds={dict(qk)}")
    for z in sink["trader_zones"]:
        print(f"  trader_zone {z['name']} pos={z['pos']} outline_pts={len(z['outline'])}")
    for r in sink["loose_points"]:
        top = (r.get("items") or [{}])[0]
        print(f"  loose {str(r['name'])[:28]:28s} pos={r['pos']} slots={r['n']} "
              f"pool={len(r.get('items') or [])} top={top.get('s') or top.get('n') or top.get('tpl')} "
              f"dev_d={r.get('dev_d')}")
    print("  copy next to the pack:  copy \"%s\" packs\\%s.eftpack\\" % (OUT, MAP))


if __name__ == "__main__":
    main()
