#!/usr/bin/env python
r"""Audit installed EFT files as possible authoritative quest/task inputs.

This is a read-only proof of concept.  It opens Unity asset files and optional
Il2CppDumper output, but never writes into the game installation, launches the
game, attaches to a process, or talks to a backend.

The audit deliberately separates three kinds of evidence:

* INSTANCE DATA: a structurally complete quest catalog (ids + conditions +
  rewards).  Only this can make the installed client authoritative for the
  production quest graph.
* FIRST-PARTY SUPPORTING DATA: locale strings, item templates, profile fixtures,
  and scene components that carry task/condition ids or exact world geometry.
* SCHEMA: IL2CPP class/field declarations and request endpoints.  Schema proves
  how a server response is consumed; it does not prove that the response's
  production instances are shipped on disk.

Typical use (UnityPy 1.25.0 is pinned in extraction/requirements.txt):

  .\venv\Scripts\python.exe extraction\intel\audit_game_quests.py

Useful overrides:

  --game-data D:\...\EscapeFromTarkov_Data
  --tasks packs\shared\tasks.json
  --il2cpp-dump C:\path\to\EFTDump_1.0.x
  --out out\quest_authority_report.json
  --no-scene-scan

The default scene pass only opens BuildSettings scenes whose names contain
"Quest" or "Cutscene".  It extracts class names and embedded 24-hex ids from
interesting MonoBehaviour payloads; it does not attempt a whole-install bake.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator


AUDIT_FORMAT_VERSION = 1
HEX24_RE = re.compile(r"^[0-9a-fA-F]{24}$")
HEX24_BYTES_RE = re.compile(rb"(?<![0-9a-fA-F])([0-9a-fA-F]{24})(?![0-9a-fA-F])")
FIXTURE_NAME_RE = re.compile(r"(test|mock|fixture|sample|debug|bigprofile)", re.I)
QUEST_NAME_RE = re.compile(r"(quest|task|locale|profile|condition)", re.I)

# A candidate must look like the response consumed by QuestTemplate, not merely
# contain a quest-shaped example or player progress rows.
MIN_CATALOG_RECORDS = 100
MIN_ID_FRACTION = 0.90
MIN_CONDITIONS_FRACTION = 0.50
MIN_REWARDS_FRACTION = 0.50

METADATA_CLASSES = (
    "QuestTemplate",
    "LocalDisableIfNoQuest",
    "CutsceneActionManageTimelineAssetsByExistQuest",
    "ManageAssetsByQuestId",
    "QuestLocationObject",
    "ShootableQuestLocationObject",
)

SCENE_CLASSES = {
    "LocalDisableIfNoQuest",
    "CutsceneActionManageTimelineAssetsByExistQuest",
    "QuestLocationObject",
    "ShootableQuestLocationObject",
    "InteractiveObjectCutsceneTrigger",
    "QuestTrigger",
    "PlaceItemTrigger",
    "ExperienceTrigger",
    "FlareShootDetectorZone",
}


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_stamp(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    stamp: dict[str, Any] = {"path": str(path.resolve()), "exists": path.is_file()}
    if not path.is_file():
        return stamp
    stat = path.stat()
    stamp.update(
        {
            "bytes": stat.st_size,
            "mtime_utc": _dt.datetime.fromtimestamp(
                stat.st_mtime, _dt.timezone.utc
            ).isoformat(timespec="seconds"),
        }
    )
    if hash_file:
        stamp["sha256"] = sha256_file(path)
    return stamp


def decode_text_asset(script: Any) -> bytes:
    if isinstance(script, bytes):
        return script
    if isinstance(script, bytearray):
        return bytes(script)
    if isinstance(script, memoryview):
        return script.tobytes()
    if isinstance(script, str):
        return script.encode("utf-8")
    return bytes(script)


def parse_json_bytes(data: bytes) -> tuple[Any | None, str | None]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return None, f"utf8: {exc}"
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"json: line {exc.lineno}, column {exc.colno}: {exc.msg}"


def unwrap_backend_data(value: Any) -> Any:
    """Unwrap common BSG backend envelopes without assuming every dict is one."""
    if (
        isinstance(value, dict)
        and "data" in value
        and ("err" in value or "errmsg" in value)
    ):
        return value["data"]
    return value


def iter_record_collections(
    value: Any, path: str = "$", depth: int = 0
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Yield plausible record collections up to four levels below a JSON root."""
    if depth > 4:
        return
    if isinstance(value, list):
        records = [item for item in value if isinstance(item, dict)]
        if records:
            yield path, records
        for index, child in enumerate(value[:20]):
            if isinstance(child, (dict, list)):
                yield from iter_record_collections(
                    child, f"{path}[{index}]", depth + 1
                )
    elif isinstance(value, dict):
        dict_values = list(value.values())
        records = [item for item in dict_values if isinstance(item, dict)]
        if records and len(records) >= max(1, len(dict_values) // 2):
            yield f"{path}{{values}}", records
        for key, child in list(value.items())[:200]:
            if isinstance(child, (dict, list)):
                yield from iter_record_collections(
                    child, f"{path}.{key}", depth + 1
                )


def _keymap(record: dict[str, Any]) -> dict[str, str]:
    return {str(key).lower(): str(key) for key in record}


def _record_id(record: dict[str, Any]) -> str | None:
    keys = _keymap(record)
    for alias in ("_id", "id", "templateid", "qid"):
        original = keys.get(alias)
        value = record.get(original) if original else None
        if isinstance(value, str) and HEX24_RE.fullmatch(value):
            return value.lower()
    return None


def _has_nonempty(record: dict[str, Any], aliases: Iterable[str]) -> bool:
    keys = _keymap(record)
    for alias in aliases:
        original = keys.get(alias.lower())
        if original is None:
            continue
        value = record.get(original)
        if value not in (None, "", [], {}):
            return True
    return False


def classify_catalog_records(
    records: list[dict[str, Any]],
    *,
    collection_path: str,
    asset_name: str,
    min_records: int = MIN_CATALOG_RECORDS,
) -> dict[str, Any]:
    count = len(records)
    ids = sum(_record_id(record) is not None for record in records)
    conditions = sum(_has_nonempty(record, ("conditions",)) for record in records)
    rewards = sum(
        _has_nonempty(
            record,
            ("rewards", "finishRewards", "startRewards", "successRewards"),
        )
        for record in records
    )
    task_shape = sum(
        _record_id(record) is not None
        and _has_nonempty(
            record,
            (
                "conditions",
                "rewards",
                "traderId",
                "location",
                "min_level",
                "name",
                "QuestName",
            ),
        )
        for record in records
    )

    def fraction(value: int) -> float:
        return round(value / count, 4) if count else 0.0

    fixture_name = bool(FIXTURE_NAME_RE.search(asset_name))
    reasons = []
    if count < min_records:
        reasons.append(f"record_count<{min_records}")
    if fraction(ids) < MIN_ID_FRACTION:
        reasons.append(f"id_fraction<{MIN_ID_FRACTION:.2f}")
    if fraction(conditions) < MIN_CONDITIONS_FRACTION:
        reasons.append(f"conditions_fraction<{MIN_CONDITIONS_FRACTION:.2f}")
    if fraction(rewards) < MIN_REWARDS_FRACTION:
        reasons.append(f"rewards_fraction<{MIN_REWARDS_FRACTION:.2f}")
    if fixture_name:
        reasons.append("fixture_or_test_name")
    accepted = not reasons
    return {
        "collection_path": collection_path,
        "record_count": count,
        "records_with_24hex_id": ids,
        "id_fraction": fraction(ids),
        "records_with_conditions": conditions,
        "conditions_fraction": fraction(conditions),
        "records_with_rewards": rewards,
        "rewards_fraction": fraction(rewards),
        "records_with_task_shape": task_shape,
        "task_shape_fraction": fraction(task_shape),
        "fixture_or_test_name": fixture_name,
        "authoritative_production_catalog": accepted,
        "rejection_reasons": reasons,
    }


def classify_catalog_document(
    document: Any, asset_name: str, min_records: int = MIN_CATALOG_RECORDS
) -> dict[str, Any]:
    unwrapped = unwrap_backend_data(document)
    candidates = [
        classify_catalog_records(
            records,
            collection_path=path,
            asset_name=asset_name,
            min_records=min_records,
        )
        for path, records in iter_record_collections(unwrapped)
    ]
    if not candidates:
        return {
            "collection_path": None,
            "record_count": 0,
            "records_with_24hex_id": 0,
            "id_fraction": 0.0,
            "records_with_conditions": 0,
            "conditions_fraction": 0.0,
            "records_with_rewards": 0,
            "rewards_fraction": 0.0,
            "records_with_task_shape": 0,
            "task_shape_fraction": 0.0,
            "fixture_or_test_name": bool(FIXTURE_NAME_RE.search(asset_name)),
            "authoritative_production_catalog": False,
            "rejection_reasons": ["no_record_collection"],
        }
    # Prefer accepted candidates, then the one with the most complete task shape.
    return max(
        candidates,
        key=lambda item: (
            item["authoritative_production_catalog"],
            item["task_shape_fraction"],
            item["conditions_fraction"] + item["rewards_fraction"],
            item["record_count"],
        ),
    )


def load_task_baseline(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    tasks = document.get("tasks") if isinstance(document, dict) else None
    if not isinstance(tasks, list):
        raise ValueError(f"{path} has no tasks[] list")
    task_by_id = {
        str(task["id"]).lower(): task
        for task in tasks
        if isinstance(task, dict) and HEX24_RE.fullmatch(str(task.get("id", "")))
    }
    objective_ids = {
        str(obj["id"]).lower()
        for task in tasks
        if isinstance(task, dict)
        for obj in (task.get("objectives") or [])
        if isinstance(obj, dict) and HEX24_RE.fullmatch(str(obj.get("id", "")))
    }
    return document, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "tasks": len(tasks),
        "task_ids": len(task_by_id),
        "objective_ids": len(objective_ids),
        "_task_by_id": task_by_id,
        "_objective_ids": objective_ids,
    }


def locale_coverage(
    locale_document: Any, task_by_id: dict[str, dict[str, Any]], objective_ids: set[str]
) -> dict[str, Any]:
    data = unwrap_backend_data(locale_document)
    if not isinstance(data, dict):
        return {"usable": False, "reason": "locale data is not a dictionary"}
    strings = {str(key): value for key, value in data.items() if isinstance(value, str)}
    task_name_keys = {
        key[:24].lower(): value
        for key, value in strings.items()
        if len(key) == 29
        and key[24:] == " name"
        and HEX24_RE.fullmatch(key[:24])
        and value
    }
    present_ids = set(task_by_id) & set(task_name_keys)
    exact = sum(
        str(task_by_id[task_id].get("name") or "") == task_name_keys[task_id]
        for task_id in present_ids
    )
    objective_key_ids = set()
    for key in strings:
        first = key.split(" ", 1)[0].lower()
        if HEX24_RE.fullmatch(first) and first in objective_ids:
            objective_key_ids.add(first)
    return {
        "usable": True,
        "total_string_keys": len(strings),
        "24hex_lowercase_name_keys": len(task_name_keys),
        "baseline_task_names_present": len(present_ids),
        "baseline_task_names_exact": exact,
        "baseline_task_names_missing": sorted(set(task_by_id) - present_ids),
        "baseline_objective_ids_present_as_locale_key": len(objective_key_ids),
        "baseline_objective_ids_total": len(objective_ids),
    }


def profile_evidence(
    document: Any, baseline_task_ids: set[str], baseline_objective_ids: set[str]
) -> dict[str, Any]:
    root = unwrap_backend_data(document)
    if not isinstance(root, dict):
        return {"usable": False}
    quest_rows = root.get("Quests") or root.get("quests") or []
    counters = root.get("TaskConditionCounters") or root.get("taskConditionCounters") or {}
    task_ids: set[str] = set()
    condition_ids: set[str] = set()
    typed_conditions: Counter[str] = Counter()
    if isinstance(quest_rows, list):
        for row in quest_rows:
            if not isinstance(row, dict):
                continue
            qid = str(row.get("qid") or row.get("_id") or "").lower()
            if HEX24_RE.fullmatch(qid):
                task_ids.add(qid)
            for condition_id in row.get("completedConditions") or []:
                condition_id = str(condition_id).lower()
                if HEX24_RE.fullmatch(condition_id):
                    condition_ids.add(condition_id)
    if isinstance(counters, dict):
        counter_rows = counters.items()
    elif isinstance(counters, list):
        counter_rows = ((str(row.get("id", "")), row) for row in counters if isinstance(row, dict))
    else:
        counter_rows = ()
    for key, row in counter_rows:
        condition_id = str(key).lower()
        if isinstance(row, dict):
            condition_id = str(row.get("id") or condition_id).lower()
            source_id = str(row.get("sourceId") or "").lower()
            if HEX24_RE.fullmatch(source_id):
                task_ids.add(source_id)
            condition_type = row.get("type")
            if isinstance(condition_type, str) and condition_type:
                typed_conditions[condition_type] += 1
        if HEX24_RE.fullmatch(condition_id):
            condition_ids.add(condition_id)
    return {
        "usable": bool(task_ids or condition_ids),
        "task_ids": len(task_ids),
        "task_ids_matching_baseline": len(task_ids & baseline_task_ids),
        "condition_ids": len(condition_ids),
        "condition_ids_matching_baseline": len(condition_ids & baseline_objective_ids),
        "condition_type_histogram": dict(typed_conditions.most_common()),
        "note": "profile progress/fixtures are evidence of ids and types, not a quest template catalog",
    }


def _looks_quest_related(document: Any) -> bool:
    root = unwrap_backend_data(document)
    if isinstance(root, dict):
        keys = {str(key).lower() for key in root}
        return bool(
            keys
            & {
                "quests",
                "taskconditioncounters",
                "conditions",
                "rewards",
                "questsettings",
            }
        )
    return False


def scan_text_assets(
    resources_path: Path,
    task_by_id: dict[str, dict[str, Any]],
    objective_ids: set[str],
) -> dict[str, Any]:
    try:
        import UnityPy  # type: ignore
    except ImportError as exc:
        return {
            "available": False,
            "error": f"UnityPy unavailable: {exc}",
            "remedy": f"{sys.executable} -m pip install -r extraction/requirements.txt",
        }

    env = UnityPy.load(str(resources_path))
    inventory: list[dict[str, Any]] = []
    documents: dict[str, Any] = {}
    total_text_assets = 0
    parsed_json_assets = 0
    for obj in env.objects:
        if obj.type.name != "TextAsset":
            continue
        total_text_assets += 1
        try:
            asset = obj.read()
            name = str(getattr(asset, "m_Name", "") or f"path_id_{obj.path_id}")
            payload = decode_text_asset(getattr(asset, "m_Script", b""))
        except Exception as exc:
            inventory.append(
                {
                    "path_id": obj.path_id,
                    "name": None,
                    "read_error": str(exc),
                }
            )
            continue
        document, parse_error = parse_json_bytes(payload)
        if document is not None:
            parsed_json_assets += 1
        related = bool(QUEST_NAME_RE.search(name)) or (
            document is not None and _looks_quest_related(document)
        )
        catalog = (
            classify_catalog_document(document, name)
            if document is not None
            else None
        )
        if catalog and catalog["record_count"] > 0:
            related = related or catalog["task_shape_fraction"] > 0
        if not related:
            continue
        record: dict[str, Any] = {
            "path_id": obj.path_id,
            "name": name,
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "json": document is not None,
            "fixture_or_test_name": bool(FIXTURE_NAME_RE.search(name)),
        }
        if parse_error:
            record["parse_error"] = parse_error
        if catalog:
            record["catalog_classification"] = catalog
        inventory.append(record)
        if document is not None:
            documents[name] = document

    locale_assets = []
    for name, document in documents.items():
        if "locale" not in name.lower():
            continue
        coverage = locale_coverage(document, task_by_id, objective_ids)
        locale_assets.append({"name": name, **coverage})

    profiles = []
    for name, document in documents.items():
        evidence = profile_evidence(document, set(task_by_id), objective_ids)
        if evidence.get("usable"):
            profiles.append({"name": name, **evidence})

    production = [
        item
        for item in inventory
        if (item.get("catalog_classification") or {}).get(
            "authoritative_production_catalog"
        )
    ]
    del env
    return {
        "available": True,
        "container": file_stamp(resources_path),
        "text_assets_total": total_text_assets,
        "json_text_assets_total": parsed_json_assets,
        "quest_related_inventory": sorted(
            inventory, key=lambda item: (str(item.get("name")), item.get("path_id", 0))
        ),
        "localization_coverage": locale_assets,
        "profile_progress_evidence": profiles,
        "production_catalog_candidates": production,
        "production_catalog_found": bool(production),
    }


def parse_consistency_info(game_data: Path) -> dict[str, Any]:
    install = game_data.parent
    path = install / "ConsistencyInfo"
    if not path.is_file():
        return file_stamp(path)
    stamp = file_stamp(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        stamp["game_version"] = document.get("Version")
        stamp["manifest_entries"] = len(document.get("Entries") or [])
    except Exception as exc:
        stamp["parse_error"] = str(exc)
    return stamp


def discover_il2cpp_dump(
    explicit: Path | None, consistency: dict[str, Any]
) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("EFT_IL2CPP_DUMP")
    if env:
        candidates.append(Path(env))
    version = consistency.get("game_version")
    if version:
        candidates.append(Path.home() / f"EFTDump_{version}")
    candidates.extend(
        sorted(
            Path.home().glob("EFTDump_*"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0,
            reverse=True,
        )
    )
    for candidate in candidates:
        root = candidate.parent if candidate.is_file() else candidate
        if (root / "dump.cs").is_file():
            return root
    return None


CLASS_DECL_RE = re.compile(
    r"^(?:public|private|protected|internal|sealed|abstract|static|\s)+"
    r"class\s+([A-Za-z0-9_.<>]+)"
)
FIELD_RE = re.compile(
    r"^\s*(?P<decl>(?:public|private|protected|internal)\s+.+?\s+"
    r"(?P<name>[A-Za-z0-9_<>]+));\s*//\s*(?P<offset>0x[0-9A-Fa-f]+)"
)
JSON_PROPERTY_RE = re.compile(r'\[JsonProperty\("([^"]+)"\)\]')


def parse_dump_classes(text: str, class_names: Iterable[str]) -> dict[str, Any]:
    wanted = set(class_names)
    results: dict[str, Any] = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        match = CLASS_DECL_RE.match(lines[index])
        if not match or match.group(1).split(".")[-1] not in wanted:
            index += 1
            continue
        raw_name = match.group(1)
        name = raw_name.split(".")[-1]
        block = []
        depth = 0
        opened = False
        cursor = index
        while cursor < len(lines):
            line = lines[cursor]
            block.append(line)
            if "{" in line:
                opened = True
            depth += line.count("{") - line.count("}")
            cursor += 1
            if opened and depth <= 0:
                break
        fields = []
        json_property = None
        in_fields = False
        for line in block:
            if line.strip() == "// Fields":
                in_fields = True
                continue
            if in_fields and line.strip() in ("// Properties", "// Methods"):
                in_fields = False
            property_match = JSON_PROPERTY_RE.search(line)
            if property_match:
                json_property = property_match.group(1)
                continue
            if not in_fields:
                continue
            field_match = FIELD_RE.match(line)
            if not field_match:
                continue
            field = {
                "name": field_match.group("name"),
                "declaration": field_match.group("decl").strip(),
                "offset": field_match.group("offset"),
            }
            if json_property:
                field["json_property"] = json_property
            json_property = None
            fields.append(field)
        results[name] = {
            "declaration": block[0].strip(),
            "fields": fields,
        }
        index = cursor
    return results


def scan_il2cpp_dump(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {
            "available": False,
            "reason": "no dump.cs found; set EFT_IL2CPP_DUMP or pass --il2cpp-dump",
        }
    dump_cs = root / "dump.cs"
    strings = root / "stringliteral.json"
    dummy = root / "DummyDll" / "Assembly-CSharp.dll"
    try:
        text = dump_cs.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return {"available": False, "path": str(root), "error": str(exc)}
    classes = parse_dump_classes(text, METADATA_CLASSES)
    request_methods = sorted(
        set(
            re.findall(
                r"\b(?:RequestQuestsTemplates|GetQuestsByCompletableItem|"
                r"GetNewQuestTemplates|ConfirmQuestTemplates)\b",
                text,
            )
        )
    )
    endpoints = []
    if strings.is_file():
        try:
            raw = strings.read_text(encoding="utf-8", errors="replace")
            endpoints = sorted(
                set(re.findall(r'"/client/quest[^"]*"', raw, re.I))
            )
        except OSError:
            pass
    return {
        "available": True,
        "root": str(root.resolve()),
        "inputs": {
            "dump_cs": file_stamp(dump_cs),
            "string_literals": file_stamp(strings) if strings.is_file() else file_stamp(strings),
            "dummy_assembly_csharp": file_stamp(dummy) if dummy.is_file() else file_stamp(dummy),
        },
        "classes": classes,
        "request_methods": request_methods,
        "quest_endpoints": [value.strip('"') for value in endpoints],
        "interpretation": (
            "current-client schema and request paths; not evidence that production "
            "QuestTemplate instances are stored in the install"
        ),
    }


def _monoscript_index(UnityPy: Any, path: Path, cache: dict[Path, dict[int, str]]) -> dict[int, str]:
    if path in cache:
        return cache[path]
    result: dict[int, str] = {}
    if path.is_file():
        env = UnityPy.load(str(path))
        for obj in env.objects:
            if obj.type.name != "MonoScript":
                continue
            try:
                tree = obj.read_typetree()
                name = tree.get("m_ClassName")
                if name:
                    result[obj.path_id] = str(name)
            except Exception:
                continue
        del env
    cache[path] = result
    return result


def _payload_after_header(obj: Any, header: dict[str, Any]) -> bytes:
    raw = obj.get_raw_data()
    name = str(header.get("m_Name") or "")
    header_size = (12 + 4 + 12 + 4 + len(name.encode("utf-8")) + 3) & ~3
    return raw[header_size:]


def _embedded_ids(payload: bytes) -> list[str]:
    return sorted(
        {
            match.group(1).decode("ascii").lower()
            for match in HEX24_BYTES_RE.finditer(payload)
        }
    )


def scan_quest_scenes(
    game_data: Path,
    baseline_task_ids: set[str],
    baseline_objective_ids: set[str],
) -> dict[str, Any]:
    try:
        import UnityPy  # type: ignore
    except ImportError as exc:
        return {"available": False, "error": f"UnityPy unavailable: {exc}"}
    managers = game_data / "globalgamemanagers"
    if not managers.is_file():
        return {"available": False, "error": f"missing {managers}"}
    manager_env = UnityPy.load(str(managers))
    scenes = []
    for obj in manager_env.objects:
        if obj.type.name != "BuildSettings":
            continue
        try:
            tree = obj.read_typetree()
            scenes = tree.get("scenes") or tree.get("m_Scenes") or []
        except Exception:
            pass
        break
    del manager_env
    selected = [
        (index, str(path))
        for index, path in enumerate(scenes)
        if re.search(r"(quest|cutscene)", str(path), re.I)
        and (game_data / f"level{index}").is_file()
    ]
    script_cache: dict[Path, dict[int, str]] = {}
    class_counts: Counter[str] = Counter()
    records = []
    errors = []
    seen_scene_paths = set()
    for level, scene_path in selected:
        normalized = scene_path.replace("\\", "/").lower()
        if normalized in seen_scene_paths:
            continue
        seen_scene_paths.add(normalized)
        path = game_data / f"level{level}"
        try:
            env = UnityPy.load(str(path))
            serialized = next(
                (file for file in env.files.values() if hasattr(file, "objects")), None
            )
            externals = list(getattr(serialized, "externals", []) or [])
            objects = env.objects
            local_scripts: dict[int, str] | None = None

            def resolve(file_id: int, path_id: int) -> str | None:
                nonlocal local_scripts
                if file_id == 0:
                    if local_scripts is None:
                        local_scripts = {}
                        for candidate in objects:
                            if candidate.type.name != "MonoScript":
                                continue
                            try:
                                name = candidate.read_typetree().get("m_ClassName")
                                if name:
                                    local_scripts[candidate.path_id] = str(name)
                            except Exception:
                                continue
                    return local_scripts.get(path_id)
                if file_id - 1 >= len(externals):
                    return None
                external_name = os.path.basename(
                    str(getattr(externals[file_id - 1], "path", "")).replace("\\", "/")
                )
                return _monoscript_index(
                    UnityPy, game_data / external_name, script_cache
                ).get(path_id)

            for obj in objects:
                if obj.type.name != "MonoBehaviour":
                    continue
                try:
                    header = obj.read_typetree(check_read=False)
                    script = header.get("m_Script") or {}
                    class_name = resolve(
                        int(script.get("m_FileID", 0)),
                        int(script.get("m_PathID", 0)),
                    )
                except Exception:
                    continue
                if not class_name:
                    continue
                if class_name not in SCENE_CLASSES and "QuestLocationObject" not in class_name:
                    continue
                class_counts[class_name] += 1
                try:
                    payload = _payload_after_header(obj, header)
                    ids = _embedded_ids(payload)
                except Exception:
                    ids = []
                records.append(
                    {
                        "level": level,
                        "scene": scene_path,
                        "path_id": obj.path_id,
                        "class": class_name,
                        "object_name": header.get("m_Name") or None,
                        "embedded_24hex_ids": ids,
                        "task_id_matches": sorted(set(ids) & baseline_task_ids),
                        "objective_id_matches": sorted(set(ids) & baseline_objective_ids),
                    }
                )
            del env
        except Exception as exc:
            errors.append({"level": level, "scene": scene_path, "error": str(exc)})
    all_ids = {
        value
        for record in records
        for value in record.get("embedded_24hex_ids", [])
    }
    return {
        "available": True,
        "selection": "unique BuildSettings scenes whose path contains Quest or Cutscene",
        "scenes_selected": len(selected),
        "unique_scenes_scanned": len(seen_scene_paths),
        "class_histogram": dict(class_counts.most_common()),
        "components": records,
        "embedded_ids": len(all_ids),
        "task_id_matches": len(all_ids & baseline_task_ids),
        "objective_id_matches": len(all_ids & baseline_objective_ids),
        "errors": errors,
    }


def scan_existing_gamedata(repo: Path, tasks_document: dict[str, Any]) -> dict[str, Any]:
    paths = sorted((repo / "packs").glob("*.eftpack/gamedata.json"))
    tarkmap = os.environ.get("EFT_TARKMAP_ROOT")
    if tarkmap:
        paths.extend(sorted((Path(tarkmap) / "out").glob("*/gamedata.json")))
    seen_maps = set()
    inputs = []
    trigger_count = 0
    trigger_names: set[tuple[str, str]] = set()
    for path in paths:
        map_name = path.parent.name.replace(".eftpack", "")
        if map_name in seen_maps:
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        seen_maps.add(map_name)
        triggers = document.get("quest_triggers") or []
        trigger_count += len(triggers)
        trigger_names.update(
            (map_name, str(row.get("name")))
            for row in triggers
            if isinstance(row, dict) and row.get("name")
        )
        inputs.append(file_stamp(path))
    claimed = {
        (str(zone.get("map")), str(zone.get("game") or zone.get("zid")))
        for task in tasks_document.get("tasks") or []
        if isinstance(task, dict)
        for objective in task.get("objectives") or []
        if isinstance(objective, dict)
        for zone in objective.get("zones") or []
        if isinstance(zone, dict) and (zone.get("game") or zone.get("zid"))
    }
    unclaimed = trigger_names - claimed
    return {
        "inputs": inputs,
        "maps": len(seen_maps),
        "quest_trigger_instances": trigger_count,
        "distinct_map_trigger_names": len(trigger_names),
        "claimed_by_tasks_json": len(trigger_names & claimed),
        "unclaimed_map_trigger_names": len(unclaimed),
        "unclaimed_sample": [
            {"map": map_name, "trigger": trigger}
            for map_name, trigger in sorted(unclaimed)[:100]
        ],
    }


def sanitize_baseline(stamp: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in stamp.items() if not key.startswith("_")}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    game_data = Path(args.game_data).expanduser().resolve()
    install = game_data.parent
    resources = (
        Path(args.resources).expanduser().resolve()
        if args.resources
        else game_data / "resources.assets"
    )
    tasks_path = Path(args.tasks).expanduser().resolve()
    tasks_document, baseline = load_task_baseline(tasks_path)
    task_by_id = baseline["_task_by_id"]
    objective_ids = baseline["_objective_ids"]
    consistency = parse_consistency_info(game_data)
    dump_root = discover_il2cpp_dump(
        Path(args.il2cpp_dump).expanduser().resolve() if args.il2cpp_dump else None,
        consistency,
    )
    assets = (
        scan_text_assets(resources, task_by_id, objective_ids)
        if resources.is_file()
        else {"available": False, "error": f"missing {resources}"}
    )
    il2cpp = scan_il2cpp_dump(dump_root)
    scenes = (
        {"available": False, "reason": "disabled by --no-scene-scan"}
        if args.no_scene_scan
        else scan_quest_scenes(game_data, set(task_by_id), objective_ids)
    )
    world = scan_existing_gamedata(repo, tasks_document)
    catalog_found = bool(assets.get("production_catalog_found"))
    return {
        "format_version": AUDIT_FORMAT_VERSION,
        "generated_at_utc": utc_now(),
        "mode": {
            "read_only": True,
            "launched_game": False,
            "attached_to_process": False,
            "network_requests": False,
        },
        "inputs": {
            "game_data": str(game_data),
            "consistency_info": consistency,
            "game_assembly": file_stamp(install / "GameAssembly.dll"),
            "global_metadata": file_stamp(
                game_data / "il2cpp_data" / "Metadata" / "global-metadata.dat"
            ),
            "tasks_baseline": sanitize_baseline(baseline),
        },
        "audit_scope": {
            "text_asset_containers": [str(resources)],
            "all_install_bundles_scanned": False,
            "scene_selection": (
                "disabled"
                if args.no_scene_scan
                else "unique BuildSettings scenes whose path contains Quest or Cutscene"
            ),
            "limitation": (
                "A negative result means no production catalog was demonstrated in the scanned "
                "static inputs; it is not mathematical proof that no unscanned bundle contains one."
            ),
        },
        "criteria": {
            "minimum_records": MIN_CATALOG_RECORDS,
            "minimum_24hex_id_fraction": MIN_ID_FRACTION,
            "minimum_conditions_fraction": MIN_CONDITIONS_FRACTION,
            "minimum_rewards_fraction": MIN_REWARDS_FRACTION,
            "fixture_or_test_names_rejected": True,
        },
        "installed_text_assets": assets,
        "il2cpp_metadata": il2cpp,
        "quest_scene_components": scenes,
        "existing_game_geometry": world,
        "verdict": {
            "installed_client_is_authoritative_for_production_quest_graph": catalog_found,
            "production_catalog_found": catalog_found,
            "production_catalog_interpretation": (
                "a structurally complete, non-fixture catalog was found in the scanned inputs"
                if catalog_found
                else "not demonstrated in the scanned static inputs"
            ),
            "installed_client_is_authoritative_for_schema": bool(il2cpp.get("available")),
            "installed_client_is_authoritative_for_observed_world_components": bool(
                scenes.get("available") or world.get("quest_trigger_instances")
            ),
            "recommended_source_split": {
                "quest_graph_conditions_rewards": (
                    "installed game files"
                    if catalog_found
                    else "official backend response; no structurally complete production "
                    "catalog passed the on-disk audit"
                ),
                "schema_and_field_meanings": "current-version IL2CPP dump",
                "task_names_and_some_first_party_labels": "embedded locale snapshot",
                "world_geometry_and_object_links": "installed Unity scenes",
                "player_progress_and_repeatables": "profile/backend only",
            },
        },
    }


def parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parents[2]
    default_game = os.environ.get(
        "EFT_GAME_DATA",
        r"C:\Battlestate Games\Escape from Tarkov\EscapeFromTarkov_Data",
    )
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--game-data", default=default_game)
    result.add_argument("--resources", help="override resources.assets path")
    result.add_argument("--tasks", default=str(repo / "packs" / "shared" / "tasks.json"))
    result.add_argument("--il2cpp-dump", help="directory containing dump.cs/DummyDll")
    result.add_argument(
        "--out", default=str(repo / "out" / "quest_authority_report.json")
    )
    result.add_argument("--no-scene-scan", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = build_report(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[quest-audit] fatal: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    assets = report["installed_text_assets"]
    metadata = report["il2cpp_metadata"]
    scenes = report["quest_scene_components"]
    print(f"[quest-audit] wrote {out}")
    print(
        "[quest-audit] production catalog in scanned static inputs: "
        + ("FOUND" if assets.get("production_catalog_found") else "not found")
    )
    print(
        f"[quest-audit] IL2CPP schema: {'available' if metadata.get('available') else 'unavailable'}"
    )
    if scenes.get("available"):
        print(
            "[quest-audit] quest/cutscene components: "
            f"{len(scenes.get('components') or [])} across "
            f"{scenes.get('unique_scenes_scanned', 0)} scene(s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
