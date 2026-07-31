"""Writes the `.eftchar` pack: manifest.json + skin.bin + anim.bin + textures/.

Same contract as `.eftpack`: the manifest declares EVERY stride, byte offset and count, and the Rust
loader reads the layout from it. Neither side hardcodes a number the other could change.

  skin.bin  = [ all meshes' interleaved vertices ][ all meshes' u32 indices ]
  anim.bin  = [ per clip, per bone track: position f32x3*F, rotation f32x4*F, scale f32x3*F ]

Only the channels a track actually has are written, and the manifest says which. Byte offsets are
absolute within the blob so a consumer can memory-map and slice without walking anything.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import coords
from .clips import Clip
from .controller import ControllerTable
from .skeleton import Skeleton
from .skin import Attachment, Material, SkinMesh, vertex_layout_manifest

PACK_VERSION = 1


@dataclass
class BuildInfo:
    """Provenance. A pack that cannot say which build it came from is a pack you cannot trust."""

    character: str
    display_name: str
    game_build: str
    bundles: List[str]


def _controller_manifest(
    controller: Optional[ControllerTable], clip_index_by_id: Optional[Sequence[int]]
) -> Optional[dict]:
    """Controller table + the id->clip-index map that makes clip lookup unambiguous.

    `clipNames` is for humans and is NOT unique: Tagilla's graph contains two distinct assets both
    called `crouch_run_aim_0`, only one of which is an absolute-pose clip. `clipIndexById[id]` is the
    authoritative resolution from a blend-tree leaf to a clip in this pack (-1 = not extracted).
    """
    if controller is None:
        return None
    out = controller.to_manifest()
    out["clipIndexById"] = list(clip_index_by_id) if clip_index_by_id else []
    return out


def write_pack(
    out_dir: str,
    info: BuildInfo,
    skel: Skeleton,
    meshes: Sequence[SkinMesh],
    materials: Sequence[Material],
    clips: Sequence[Clip],
    controller: Optional[ControllerTable],
    images: Dict[str, object],
    default_lod: int = 0,
    extra: Optional[dict] = None,
    clip_index_by_id: Optional[Sequence[int]] = None,
    attachments: Optional[Sequence[Attachment]] = None,
) -> dict:
    """Emit the pack. Returns the manifest dict that was written."""
    os.makedirs(out_dir, exist_ok=True)
    tex_dir = os.path.join(out_dir, "textures")

    # ---- textures ----
    written_textures: List[str] = []
    if images:
        os.makedirs(tex_dir, exist_ok=True)
        for name, img in sorted(images.items()):
            rel = f"textures/{name}.png"
            path = os.path.join(out_dir, rel)
            try:
                # NORMAL MAPS: Unity stores these in the DXT5nm/BC5 convention — X in ALPHA,
                # Y in green, red a constant 1.0 (unused), Z reconstructed. Written raw, a
                # consumer that reads X from RED (Bevy, glTF, every standard PBR shader) gets a
                # tangent normal of about (1, y, z): pointing along the tangent instead of out
                # of the surface, so shading flipped between lit and black as a head turned.
                # Repack to the standard RGB layout here, where the convention is known: X from
                # alpha, Y as-is, Z reconstructed = sqrt(1 - x^2 - y^2).
                if name.lower().endswith(("_n", "_normal", "_nrm")):
                    img = _repack_normal_map(img, name)
                img.save(path)
                written_textures.append(rel)
            except Exception as exc:
                print(f"  [warn] texture {name} failed to save: {exc}")

    # ---- skin.bin ----
    mesh_manifest: List[dict] = []
    vert_blobs: List[bytes] = []
    idx_blobs: List[bytes] = []
    vert_cursor = 0
    idx_cursor = 0
    for m in meshes:
        vb = np.ascontiguousarray(m.vertices).tobytes()
        ib = np.ascontiguousarray(m.indices.astype(np.uint32)).tobytes()
        mesh_manifest.append(
            {
                "name": m.name,
                "part": m.part,
                "lod": m.lod,
                "vertexCount": m.vertex_count,
                "vertexByteOffset": vert_cursor,
                "vertexByteLength": len(vb),
                "indexCount": int(m.indices.size),
                # Filled in after the vertex block length is known.
                "indexByteOffset": idx_cursor,
                "indexByteLength": len(ib),
                "boundBones": m.bound_bones,
                "inverseBindposes": m.inverse_bindposes.reshape(len(skel), 16).tolist(),
                "submeshes": [
                    {
                        "material": s.material,
                        "indexStart": s.index_start,
                        "indexCount": s.index_count,
                    }
                    for s in m.submeshes
                ],
            }
        )
        vert_blobs.append(vb)
        idx_blobs.append(ib)
        vert_cursor += len(vb)
        idx_cursor += len(ib)

    # Rigid attachments ride the same blob and the same vertex layout, so the loader has one parser.
    attach_manifest: List[dict] = []
    for a in attachments or []:
        vb = np.ascontiguousarray(a.vertices).tobytes()
        ib = np.ascontiguousarray(a.indices.astype(np.uint32)).tobytes()
        attach_manifest.append(
            {
                "name": a.name,
                "bone": a.bone,
                "lod": a.lod,
                "localPos": a.local_pos,
                "localRot": a.local_rot,
                "localScale": a.local_scale,
                "vertexCount": a.vertex_count,
                "vertexByteOffset": vert_cursor,
                "vertexByteLength": len(vb),
                "indexCount": int(a.indices.size),
                "indexByteOffset": idx_cursor,
                "indexByteLength": len(ib),
                "submeshes": [
                    {"material": s.material, "indexStart": s.index_start, "indexCount": s.index_count}
                    for s in a.submeshes
                ],
            }
        )
        vert_blobs.append(vb)
        idx_blobs.append(ib)
        vert_cursor += len(vb)
        idx_cursor += len(ib)

    vertex_block_len = vert_cursor
    for entry in mesh_manifest:
        entry["indexByteOffset"] += vertex_block_len
    for entry in attach_manifest:
        entry["indexByteOffset"] += vertex_block_len

    with open(os.path.join(out_dir, "skin.bin"), "wb") as fh:
        for b in vert_blobs:
            fh.write(b)
        for b in idx_blobs:
            fh.write(b)

    # ---- anim.bin ----
    clip_manifest: List[dict] = []
    anim_cursor = 0
    with open(os.path.join(out_dir, "anim.bin"), "wb") as fh:
        for c in clips:
            tracks: List[dict] = []
            for t in c.tracks:
                entry: dict = {"bone": t.bone}
                for chan, arr, width in (
                    ("position", t.position, 3),
                    ("rotation", t.rotation, 4),
                    ("scale", t.scale, 3),
                ):
                    if arr is None:
                        continue
                    blob = np.ascontiguousarray(arr.astype(np.float32)).tobytes()
                    fh.write(blob)
                    entry[chan] = {
                        "byteOffset": anim_cursor,
                        "byteLength": len(blob),
                        "components": width,
                    }
                    anim_cursor += len(blob)
                if len(entry) > 1:
                    tracks.append(entry)
            # Root motion, stripped off the bone track so no consumer can double-apply it.
            root_motion = None
            if c.root_motion is not None and c.root_motion_bone is not None:
                blob = np.ascontiguousarray(c.root_motion.astype(np.float32)).tobytes()
                fh.write(blob)
                root_motion = {
                    "bone": c.root_motion_bone,
                    "byteOffset": anim_cursor,
                    "byteLength": len(blob),
                    "components": 3,
                }
                anim_cursor += len(blob)
            clip_manifest.append(
                {
                    "rootMotion": root_motion,
                    "name": c.name,
                    "duration": c.duration,
                    "sampleRate": c.sample_rate,
                    "frameCount": c.frame_count,
                    "loop": c.loop,
                    "averageSpeed": c.average_speed,
                    "unresolvedBindings": len(c.unresolved),
                    "tracks": tracks,
                }
            )

    # ---- manifest ----
    manifest = {
        "version": PACK_VERSION,
        "id": info.character,
        "displayName": info.display_name,
        "source": {"gameBuild": info.game_build, "bundles": info.bundles},
        "conventions": coords.conventions(),
        "skeleton": skel.to_manifest(),
        "vertexLayout": vertex_layout_manifest(),
        "indexFormat": "u32",
        "defaultLod": default_lod,
        "meshes": mesh_manifest,
        "attachments": attach_manifest,
        "materials": [
            {"name": m.name, "textures": m.textures, "floats": m.floats, "colors": m.colors}
            for m in materials
        ],
        "textures": written_textures,
        "clips": clip_manifest,
        "controller": _controller_manifest(controller, clip_index_by_id),
        "blobs": {
            "skin": {
                "file": "skin.bin",
                "vertexBlockByteLength": vertex_block_len,
                "totalByteLength": vertex_block_len + idx_cursor,
            },
            "anim": {"file": "anim.bin", "totalByteLength": anim_cursor},
        },
    }
    if extra:
        manifest.update(extra)

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)

    return manifest


def _repack_normal_map(img, name: str):
    """DXT5nm/BC5 (X in alpha, red unused) -> standard RGB normal map. Pass-through when the
    image is already standard (a real red channel with variation), so nothing is corrupted
    twice and non-Unity-convention maps stay untouched."""
    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return img
    a = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    r, g, b, al = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    # A standard map has a varying red channel; DXT5nm pins red to ~1 and carries X in alpha.
    if r.std() > 0.02:
        return img
    x = al * 2.0 - 1.0
    y = g * 2.0 - 1.0
    z = np.sqrt(np.clip(1.0 - x * x - y * y, 0.0, 1.0))
    out = np.stack([(x + 1.0) * 0.5, (y + 1.0) * 0.5, (z + 1.0) * 0.5], axis=-1)
    print(f"  [normal] {name}: DXT5nm (X in alpha) -> standard RGB")
    return Image.fromarray(np.clip(out * 255.0, 0, 255).astype("uint8"), "RGB")
