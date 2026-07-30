"""Data-driven structural culls for the instance assembler.

Replaces the old hardcoded, Interchange-only rule ("keep terrain or roots starting with SBG") that silently
nuked every other map (Factory's roots are 'Office'/'Area_01'/'bunker'/'factory_tunel' — no 'SBG' anywhere).

Every map's `config.json` carries a `cull` block; all fields optional with sensible defaults:
  keep_root_prefix      : allowlist (e.g. "SBG" for Interchange). Roots with this prefix are PROTECTED from
                          drop_root_regex (declared content); all other roots still flow through the same
                          structural denylist as on any map (terrain is always kept). Most maps leave this null.
  drop_root_regex       : denylist of junk roots (decals / lights / audio / triggers / blockers / probes /
                          skybox). DEFAULT_DROP_ROOT_RE covers every map seen so far; override per map.
  drop_shadow_submeshes : drop submeshes whose shader starts with 'shadow' (default True; applied in assemble).

`root` is the top-most ancestor GameObject name in the Unity scene tree (extractor records it). It is a
per-map grouping name, NOT a universal convention — hence allow/deny by data, never a literal in code.
"""
import re, os

# Junk roots that carry no renderable map geometry. Validated against Factory (keeps 10,588 geometry roots,
# drops 3,212: Decals/Day_light/Night_light/SpatialAudioSystem/BLOCKER...) and reproduces Interchange.
# NOTE the light term is (day|night)_?light, NOT the old `.*light`: now that *_Light SCENES are extracted
# as geometry (they hold the lamp fixtures + vehicles whose headlights are the light sources — reserve's
# KamAZ under the warehouse crates), a bare `.*light` would re-cull those roots at assemble on any map
# without a keep_root_prefix allowlist. Interchange's 'SBG_Shopping_Mall_Light' (Gazel van + Ford Focus)
# already documented this over-match; reserve's 'SBG_Reserve_Base_Lights' only escaped by being plural.
# Factory's validated junk (Day_light/Night_light sun/moon glow rigs) still matches.
DEFAULT_DROP_ROOT_RE = (
    r"(?i)^(decals?|.*_decals?|.*(?:day|night)_?light|.*audio.*|trig?ger.*|blocker.*|justplane.*|"
    r".*event.*|.*volume|reflectionprobe.*|lightprobe.*|skybox.*|spatialaudio.*)$"
)

# Unity occlusion/collision/stencil PROXY volumes are exported with NO albedo texture AND the engine-default
# 'Standard' (or empty) shader — real EFT visual geometry always carries a game shader (p0/*, Nature/*, Custom/*,
# Legacy/*). Such submeshes are invisible in-game but, if kept, render as solid white/grey boxes that OCCLUDE the
# real geometry behind them (e.g. the concrete columns under an overpass). Structural, name-free, map-agnostic.
PROXY_SHADERS = {"", "?", "standard"}

# SpeedTree BILLBOARD-LOD geometry (Unity's synthesized camera-facing impostor). We render static geometry, not
# camera-facing billboards, so this ships as an untextured/white cross or cylinder standing around every tree. The
# real tree LODs (Nature/SpeedTreeEFT) are kept; only the impostor is dropped. Map-agnostic (a Unity built-in shader).
BILLBOARD_SHADERS = {"hidden/tree billboard lod"}

# Atmospheric FOG-SHEET billboards (e.g. Custom/Billboard_FogSheet_Simple): large camera-facing translucent planes
# that fake volumetric fog in-engine. We render static geometry, so they ship as intrusive see-through "walls" of
# haze. Drop them (data-driven substring — catches _Simple/_Soft/etc variants). Map-agnostic.
FOG_SUBSTR = ("fogsheet", "billboard_fog")


class Culls:
    """Compiled, reusable cull predicate built from a config `cull` block."""

    def __init__(self, cull_cfg=None):
        c = cull_cfg or {}
        self.keep_prefix = (c.get("keep_root_prefix") or None)
        self.drop_re = re.compile(c.get("drop_root_regex") or DEFAULT_DROP_ROOT_RE)
        self.drop_shadow = c.get("drop_shadow_submeshes", True)
        self.drop_proxies = c.get("drop_untextured_proxies", True)   # occluder/collision/stencil boxes (see PROXY_SHADERS)
        # Unity in-game-camera visibility: drop renderers Unity never draws (ShadowsOnly / disabled / inactive-in-
        # hierarchy), recorded per-instance by the extractor as it['drop']. Default ON; TARKMAP_KEEP_HIDDEN=1 keeps
        # them for A/B inspection. Map-agnostic (pure Unity flags). See tarkmap/UNITY_VISIBILITY_GATE.md.
        self.drop_hidden = c.get("drop_unity_hidden", True) and os.environ.get("TARKMAP_KEEP_HIDDEN") != "1"
        # inactive-in-hierarchy (aih==False) is NOT "never drawn": EFT activates much of this geometry at
        # RAID LOAD (checkout counters/registers, per-lane units, alt-state props), so the static-scene
        # 'inactive' flag hides real in-raid geometry -> floating baskets over vanished counters, missing
        # registers. KEEP inactive geometry by default; only ShadowsOnly/disabled are truly never drawn.
        # TARKMAP_DROP_INACTIVE=1 restores the old aggressive drop for A/B.
        self.drop_inactive = os.environ.get("TARKMAP_DROP_INACTIVE") == "1"
        # ...but the keep-exemption is for LOOT-SCALE objects (crates/registers/counters). Artist-parked
        # scenery duplicates are ALSO aih==False (shoreline ships a disabled 140 m Lighthouse_Mountain
        # copy floating mid-air, tipped over) and must not ride the exemption: an inactive instance whose
        # world bounding DIAMETER exceeds this keeps only via TARKMAP_KEEP_HIDDEN=1. The caller supplies
        # the mesh-size lookup (filter(..., mesh_diam=...)); without one the gate is inert (old behavior).
        self.inactive_keep_max_m = float(c.get("inactive_keep_max_m", 10.0))
        self._mesh_diam = None
        self._n_oversize = 0
        # OFF-MAP BACKDROP cull: EFT scenes carry a distant city-skyline cluster (e.g. Interchange's build03_part*/
        # bulding_city_LOD0/concrete2 under SBG_Shopping_Mall_2) placed ~1-2km OUTSIDE the terrain footprint. Nothing culls
        # them by name (they sit under an allowlisted root), so they render as giant intrusive silhouettes over the map.
        # Data-driven + map-agnostic: the TERRAIN tiles define the playable footprint; drop non-terrain instances whose
        # translation sits well beyond it (measured on the RAW scene.json translation, which is invariant to the later
        # handedness conjugation — a rigid flip moves terrain + instances together). Verified on Interchange: real geometry
        # is <100u outside the padded footprint, the 10 backdrop instances are ~780u outside — a clean, unambiguous gap.
        self.drop_offmap = c.get("drop_offmap_backdrops", True)
        self.offmap_pad_m = float(c.get("offmap_pad_m", 700.0))       # pad the terrain tile-ORIGIN bbox by ~one tile so tile geometry + near-map structures are safely inside
        self.offmap_margin_m = float(c.get("offmap_margin_m", 300.0)) # only drop instances THIS far BEYOND the padded footprint (well inside the 100u..780u gap)

    def keep_instance(self, it):
        if it.get("kind") not in ("mesh", "terrain") or not it.get("m") or not it.get("subs"):
            return False
        if it.get("kind") == "terrain":
            return True
        if self.drop_hidden:
            # ShadowsOnly (cast==3) + disabled (renON==False) renderers are NEVER drawn by Unity -> always drop.
            # inactive-in-hierarchy (aih==False) -> KEEP (EFT raid-activates it) unless TARKMAP_DROP_INACTIVE=1.
            if it.get("cast") == 3 or it.get("renON") is False:
                return False
            if it.get("aih") is False:
                # Keep inactive geometry (EFT raid-activates loot boxes / crates / containers /
                # counters), EXCEPT redundant purely-numeric lane-number labels ("1".."20") that
                # duplicate the active IDEA_checkout_numbers signage and z-fight it. Full drop via
                # TARKMAP_DROP_INACTIVE=1.
                mname = str(it.get("mesh") or "").split("/")[-1].split("__")[0]
                if self.drop_inactive or mname.isdigit():
                    return False
                # Size gate (see __init__): raid-activated objects are loot-scale; oversized
                # inactive geometry is parked/disabled scenery the game never draws.
                if self._mesh_diam is not None:
                    d = self._mesh_diam(it)
                    if d is not None and d > self.inactive_keep_max_m:
                        self._n_oversize += 1
                        return False
        # WATER LAYER OVERRIDE. `wlayer` is set by eft_extract_v2 for geometry on Unity's built-in
        # Water layer (4) — the layer EFT's own ballistics reads. It must outrank the root denylist
        # below, because BSG parks woods' `WATER_LEVEL` surface under a `BLOCKERS` scene root, and
        # BLOCKER is on that denylist: without this the lake extracts correctly and is then silently
        # discarded here, which is indistinguishable from never extracting it. The engine LAYER is the
        # authoritative statement that this is water; which root it happens to hang under is not.
        if it.get("wlayer"):
            return True
        root = it.get("root") or ""
        # ALLOWLIST = PROTECTION, NOT EXCLUSION (2026-07-13). The allowlist's one legitimate job is to shield
        # declared-content roots from the generic name denylist (DEFAULT_DROP_ROOT_RE's '.*light' would nuke
        # 'SBG_Shopping_Mall_Light' -- the Gazel van, Ford Focus, and every light-fixture mesh, ~6,150 instances).
        # The OLD semantics ("keep ONLY allowlisted roots") also silently deleted every real root that simply
        # lacked the prefix: on Interchange that was New_mechanics (828 inst -- the power SWITCH
        # reserve_electric_switcher_lever + the mechanics-update lamp fixtures map-wide), STATIONARY (the
        # substation MG nest), DUCK, NEW_CONTAINER_2, Power Plant, SniperBorders_ALERTS_TEMP. Non-allowlisted
        # roots now flow through the SAME structural denylist as on maps with no allowlist at all -- one rule
        # set, every map, no per-name patches. (Noise still dies structurally: SpatialAudioSystem '.*audio.*',
        # Triger_Zone 'trig?ger.*', BLOCKER, Event_swithcer '.*event.*'.)
        if self.keep_prefix and root.upper().startswith(self.keep_prefix.upper()):
            return True
        return not self.drop_re.match(root)

    def keep_submesh(self, sb):
        sh = (sb.get("sh") or "").strip().lower()
        if self.drop_shadow and sh.startswith("shadow"):
            return False
        if sh in BILLBOARD_SHADERS:
            return False                                              # SpeedTree impostor -> white cylinder if kept
        if any(f in sh for f in FOG_SUBSTR):
            return False                                              # fog-sheet billboard -> see-through haze "wall" if kept
        if self.drop_proxies and not sb.get("tex") and sh in PROXY_SHADERS:
            # ...EXCEPT water. An untextured empty-shader sub with role='water' is a real surface,
            # not a proxy box: shoreline's ocean tiles have always matched this filter and died here
            # (the visible ocean is the seaLevel quad the viewer synthesizes instead), which was fine
            # for the SEA. But enclosed water gets no synthesized quad (derive_sea_level correctly
            # rejects lakes), so woods' layer-4 WATER_LEVEL plane must ship as geometry or the lake
            # simply does not exist. The viewer shades role='water' + no albedo as opaque deep water
            # (same branch as the sea quad), so no material/texture is needed. Discovered the hard
            # way: the lake extracted perfectly and this line silently merged it out of the pack.
            if sb.get("role") != "water":
                return False                                          # untextured + default-shader = invisible Unity proxy box
        return True

    def _offmap_backdrop_filter(self, kept, dropped_roots):
        """Drop non-terrain instances placed far OUTSIDE the terrain footprint (distant skyline backdrops). Returns
        (kept2, offmap_count, offmap_examples). No-op unless there's enough terrain to define a footprint."""
        if not self.drop_offmap:
            return kept, 0, []
        ter = [it for it in kept if it.get("kind") == "terrain" and it.get("m")]
        if len(ter) < 4:                       # too little terrain to define a playable footprint (interior-only map) -> skip
            return kept, 0, []
        xs = [it["m"][3] for it in ter]; zs = [it["m"][11] for it in ter]
        xlo, xhi = min(xs) - self.offmap_pad_m, max(xs) + self.offmap_pad_m
        zlo, zhi = min(zs) - self.offmap_pad_m, max(zs) + self.offmap_pad_m
        m = self.offmap_margin_m
        def _outdist(it):
            x, z = it["m"][3], it["m"][11]
            dx = max(xlo - x, 0.0, x - xhi); dz = max(zlo - z, 0.0, z - zhi)
            return (dx * dx + dz * dz) ** 0.5
        keep2, off, examples = [], 0, []
        for it in kept:
            if it.get("kind") != "terrain" and it.get("m") and _outdist(it) > m:
                off += 1
                r = it.get("root") or "?"; dropped_roots[r] = dropped_roots.get(r, 0) + 1
                if len(examples) < 12: examples.append((str(it.get("mesh", "?"))[:40], round(_outdist(it))))
            else:
                keep2.append(it)
        return keep2, off, examples

    def filter(self, instances, mesh_diam=None):
        """Return (kept, report). report = {'raw','kept','dropped','top_dropped_roots',...} for fail-loud logging.
        `mesh_diam(it) -> world bounding diameter (m) or None` arms the oversized-inactive gate (see __init__);
        callers that can't size meshes just omit it and keep the old behavior."""
        kept, dropped_roots = [], {}; hidden = 0
        self._mesh_diam = mesh_diam
        self._n_oversize = 0
        for it in instances:
            if self.keep_instance(it):
                kept.append(it)
            elif it.get("kind") in ("mesh", "terrain"):
                if self.drop_hidden and it.get("drop"): hidden += 1
                r = it.get("root") or "?"
                dropped_roots[r] = dropped_roots.get(r, 0) + 1
        kept, offmap, offmap_examples = self._offmap_backdrop_filter(kept, dropped_roots)
        top = sorted(dropped_roots.items(), key=lambda kv: -kv[1])[:12]
        return kept, {"raw": len(instances), "kept": len(kept), "hidden_unity": hidden,
                      "inactive_oversize": self._n_oversize,
                      "offmap_backdrop": offmap, "offmap_examples": offmap_examples,
                      "dropped": sum(dropped_roots.values()), "top_dropped_roots": top}
