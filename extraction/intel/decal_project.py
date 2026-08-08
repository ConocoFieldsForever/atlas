"""decal_project — clip receiving geometry against a decal's projection box.

Unity's StaticDeferredDecal is a PROJECTOR: it paints whatever geometry lies inside its box, so a
sign spanning two staggered plates paints both. Emitting one flat quad at the box centre cannot do
that -- the nearer surface occludes the quad and the far half of the word vanishes (field report:
"the second part of Untar is missing ... its still behind the other metal plate").

These decals are STATIC by name, on static geometry, so the projection is evaluated ONCE here
instead of per-frame against a depth buffer: for every triangle inside the box, clip it to the box
and emit the clipped polygon with UVs taken from the box's own local X/Z. The pixels match what
Unity's deferred pass would produce, with no runtime cost and no renderer change.

Public entry: `project_decals(dataset, decals, log)` -> rewrites each decal instance in place to
reference a baked mesh, dropping decals whose box contains no geometry.
"""
import os

import numpy as np

# The map's handedness flip -- the same constant the geometry extractor and assembler use.
G3 = np.diag([-1.0, 1.0, 1.0])

# A decal is nudged this far along the receiving surface's own normal so it wins the depth test
# against the surface it is painted on. Small enough to stay invisible at grazing angles.
SURFACE_OFFSET_M = 0.012
# Skip absurd boxes (a few authored decals have degenerate or kilometre-scale extents).
MAX_BOX_M = 200.0
# A surface is painted when its normal opposes the projection axis by at least this much (cos of
# the cutoff angle): 0.5 = 60 degrees. Unity fades decals out past a similar angle, and the cutoff
# is what stops a decal SMEARING down a surface it only grazes -- at 0.2 (78 degrees) the artwork
# stretched into long streaks along angled plates and their legs.
FACING_MIN = 0.50
# Runaway guard: a projector enclosing very dense geometry (terrain, foliage) contributes an
# unbounded triangle count. Anything past this is reported and skipped rather than silently
# doubling the pack.
MAX_TRIS_PER_DECAL = 4000
# How far past the authored box depth a decal may follow a surface it is already touching.
# Authored boxes are routinely a little thinner than the slanted surface they paint: the
# checkpoint plate sits 24 degrees off perpendicular, so its face sweeps 1.44 m through a 0.63 m
# box and a hard clip cut the lettering mid-glyph. 2.5x covers that without letting a decal run
# down a wall (4x did, and it showed).
DEPTH_REACH = 2.5


def _load_obj(path, cache):
    """(verts Nx3 float32, faces Mx3 int32) with the extractor's own X-negated local frame."""
    if path in cache:
        return cache[path]
    verts = []
    faces = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    a = line.split()
                    verts.append((float(a[1]), float(a[2]), float(a[3])))
                elif line.startswith("f "):
                    idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
                    for k in range(1, len(idx) - 1):
                        faces.append((idx[0], idx[k], idx[k + 1]))
    except OSError:
        cache[path] = None
        return None
    if not verts or not faces:
        cache[path] = None
        return None
    out = (np.asarray(verts, np.float32), np.asarray(faces, np.int32))
    cache[path] = out
    return out


def _clip_poly(poly, normal, dist):
    """Sutherland-Hodgman: keep the half-space `dot(p, normal) <= dist`."""
    if len(poly) == 0:
        return poly
    out = []
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        da = float(np.dot(a, normal)) - dist
        db = float(np.dot(b, normal)) - dist
        if da <= 0:
            out.append(a)
        if (da > 0) != (db > 0):
            t = da / (da - db)
            out.append(a + (b - a) * t)
    return out


def _tri_box_overlap(bx, by, bz, F, hx, hy, hz):
    """Exact triangle-vs-box overlap (Akenine-Moller separating axes), vectorised over all faces.

    bx/by/bz are the PER-VERTEX coordinates in the projector's own frame that the caller already
    computes (box centred on the origin, half-extents hx/hy/hz); F is the (n,3) face table.
    Returns a bool (n,): true when the triangle overlaps the box.

    Axes 1-3 are the box's own face normals and reduce to an interval test on numbers already in
    hand; they reject the overwhelming majority of a receiver's triangles. The remaining ten (the
    triangle's plane, plus the nine box-axis x triangle-edge crosses) are evaluated ONLY on the
    survivors, so the exact test costs barely more than a conservative one.
    """
    tx, ty, tz = bx[F], by[F], bz[F]                        # (n, 3) per-triangle vertex coords
    ok = ((tx.min(1) <= hx) & (tx.max(1) >= -hx)
          & (ty.min(1) <= hy) & (ty.max(1) >= -hy)
          & (tz.min(1) <= hz) & (tz.max(1) >= -hz))
    idx = np.nonzero(ok)[0]
    if idx.size == 0:
        return ok
    h = np.array([hx, hy, hz], np.float64)
    v = np.stack((tx[idx], ty[idx], tz[idx]), axis=2)       # (m, 3 verts, 3 box axes)
    e = (v[:, 1] - v[:, 0], v[:, 2] - v[:, 1], v[:, 0] - v[:, 2])
    good = np.ones(idx.size, bool)
    n = np.cross(e[0], e[1])                                # axis 13: the triangle's own plane
    good &= np.abs(np.einsum("ij,ij->i", n, v[:, 0])) <= np.abs(n) @ h
    for f in e:                                             # axes 4-12: box axis x triangle edge
        for k in range(3):
            i1, i2 = (k + 1) % 3, (k + 2) % 3
            a = np.zeros_like(f)
            a[:, i1] = -f[:, i2]                            # a = e_k x f, written out
            a[:, i2] = f[:, i1]
            pr = np.einsum("ijk,ik->ij", v, a)
            r = np.abs(a) @ h
            good &= (pr.min(1) <= r) & (pr.max(1) >= -r)
    ok[idx] = good
    return ok


def project_decals(dataset, decals, log=print):
    """Bake every decal against the dataset's own geometry. Returns the surviving instances."""
    import json

    scene_p = os.path.join(dataset, "scene.json")
    if not os.path.isfile(scene_p):
        log("  [decals] no scene.json — cannot project; keeping flat quads")
        return decals
    scene = json.load(open(scene_p, encoding="utf-8"))
    inst = scene.get("instances") or []

    # World AABB per instance, from the mesh's local AABB through its matrix. Cheap and only needs
    # the OBJ's extremes, so the per-mesh bbox is cached and most OBJs are never fully parsed.
    bbox_cache = {}

    def local_bbox(mesh):
        if mesh in bbox_cache:
            return bbox_cache[mesh]
        mn = np.full(3, np.inf, np.float64)
        mx = np.full(3, -np.inf, np.float64)
        try:
            with open(os.path.join(dataset, "meshes", mesh), encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("v "):
                        a = line.split()
                        p = (float(a[1]), float(a[2]), float(a[3]))
                        mn = np.minimum(mn, p)
                        mx = np.maximum(mx, p)
        except OSError:
            bbox_cache[mesh] = None
            return None
        r = None if not np.isfinite(mn).all() else (mn, mx)
        bbox_cache[mesh] = r
        return r

    log("  [decals] projecting %d decal(s) against %d instance(s)" % (len(decals), len(inst)))
    centers = []
    radii = []
    keep_inst = []
    for it in inst:
        mesh = it.get("mesh")
        m = it.get("m")
        if not mesh or not m or it.get("drop"):
            continue
        bb = local_bbox(mesh)
        if bb is None:
            continue
        # CONJUGATE, exactly as the assembler will when it places this instance: M' = G.M.G and
        # T' = G.T. Clipping against the RAW matrix put the geometry in a MIRRORED frame, so the
        # decal landed on the right object leaning the opposite way to the surface it painted
        # ("the plates are leaning back and the decal is leaning forward") -- the documented
        # signature of a handedness flip applied to positions but not to orientations.
        M3 = G3 @ np.array([[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]], np.float64) @ G3
        T = G3 @ np.array([m[3], m[7], m[11]], np.float64)
        corners = np.array([[x, y, z] for x in bb[0][0:1].tolist() + bb[1][0:1].tolist()
                            for y in bb[0][1:2].tolist() + bb[1][1:2].tolist()
                            for z in bb[0][2:3].tolist() + bb[1][2:3].tolist()], np.float64)
        w = corners @ M3.T + T
        c = (w.min(axis=0) + w.max(axis=0)) * 0.5
        centers.append(c)
        radii.append(float(np.linalg.norm(w.max(axis=0) - c)))
        keep_inst.append((mesh, M3, T))
    if not keep_inst:
        log("  [decals] no usable instance geometry — keeping flat quads")
        return decals
    centers = np.asarray(centers, np.float64)
    radii = np.asarray(radii, np.float64)

    obj_cache = {}
    out = []
    baked = 0
    empty = 0
    heavy = 0
    tri_total = 0
    mesh_dir = os.path.join(dataset, "meshes")

    for di, dec in enumerate(decals):
        m = dec["m"]
        # The projector lives in the same conjugated frame as the geometry it paints.
        Md = G3 @ np.array([[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]], np.float64) @ G3
        C = G3 @ np.array([m[3], m[7], m[11]], np.float64)
        ax = Md[:, 0]   # image across
        ay = Md[:, 1]   # projection axis
        az = Md[:, 2]   # image down
        hx, hy, hz = np.linalg.norm(ax) / 2, np.linalg.norm(ay) / 2, np.linalg.norm(az) / 2
        if min(hx, hy, hz) < 1e-4 or max(hx, hy, hz) > MAX_BOX_M:
            continue
        ux, uy, uz = ax / (hx * 2), ay / (hy * 2), az / (hz * 2)
        reach = float(np.linalg.norm([hx, hy, hz]))

        near = np.nonzero(np.linalg.norm(centers - C, axis=1) <= (radii + reach))[0]
        verts_out = []
        uvs_out = []
        faces_out = []
        for ii in near:
            mesh, M3, T = keep_inst[ii]
            geo = _load_obj(os.path.join(mesh_dir, mesh), obj_cache)
            if geo is None:
                continue
            V, F = geo
            W = V.astype(np.float64) @ M3.T + T
            rel = W - C
            bx = rel @ ux
            by = rel @ uy
            bz = rel @ uz
            # Cheap conservative reject before any per-triangle work: if EVERY vertex of this
            # instance lies beyond one of the box's slabs, no triangle of it can reach the box.
            if (bx.min() > hx or bx.max() < -hx or by.min() > hy or by.max() < -hy
                    or bz.min() > hz or bz.max() < -hz):
                continue
            # GATE: this instance is a receiver only if a TRIANGLE genuinely overlaps the authored
            # box. The old test asked whether a VERTEX was inside it, which silently skipped every
            # triangle that CONTAINS or straddles the projector with all three corners outside --
            # the normal case for a small projector on a large plate. On Interchange that left 57
            # of 1,737 projectors painting nothing at all and under-painted 948 more. The clip
            # below was always able to handle those triangles; it was never handed them.
            # A vertex inside the box implies its triangles overlap it, so this is a strict
            # SUPERSET of the old gate: nothing that used to bake can stop baking.
            touch = _tri_box_overlap(bx, by, bz, F, hx, hy, hz)
            if not touch.any():
                continue
            cand_mask = touch
            # The LOOSE slab stays tied to the ORIGINAL vertex condition, deliberately. Once a
            # vertex is inside the authored box the clip may follow that same surface out to
            # DEPTH_REACH (the slanted checkpoint plates). Extending that licence to every
            # instance the new gate admits would let a large projector drag in everything within
            # 2.5x its depth and blow the density guard. Old set UNION new set, nothing wider.
            touches = (np.abs(bx) <= hx) & (np.abs(by) <= hy) & (np.abs(bz) <= hz)
            if touches.any():
                inside = ((np.abs(bx) <= hx * 1.2) & (np.abs(by) <= hy * DEPTH_REACH)
                          & (np.abs(bz) <= hz * 1.2))
                cand_mask = cand_mask | inside[F].any(axis=1)
            cand = np.nonzero(cand_mask)[0]
            for fi in cand:
                tri = [W[F[fi, k]] for k in range(3)]
                e1 = tri[1] - tri[0]
                e2 = tri[2] - tri[0]
                nrm = np.cross(e1, e2)
                ln = float(np.linalg.norm(nrm))
                if ln < 1e-12:
                    continue
                nrm = nrm / ln
                # FACING CULL, the same rule Unity's deferred decals apply: a decal paints the
                # surfaces its box projects ONTO, not every polygon it happens to enclose. Without
                # it a box sitting on dense terrain clipped thousands of ground triangles nobody
                # can see (1.42 M for one level's decals), and back faces got painted through.
                # SIDE: keep the faces the projector actually lands on. The sign follows the
                # conjugated axis -- conjugating the projector flipped uy, which silently moved
                # every decal to the far face of its surface ("its on the backside of the plate
                # now"). Determined by observation, and the symptom is unmistakable if it ever
                # flips again: the artwork disappears from the side you are standing on.
                if float(np.dot(nrm, uy)) < FACING_MIN:
                    continue
                poly = [np.asarray(p, np.float64) - C for p in tri]
                # Clip to the IMAGE rectangle (X and Z) exactly -- that is what frames the artwork.
                # The DEPTH axis is deliberately generous: the authored box is often thinner than
                # the surface it paints (the checkpoint plates' faces sweep 1.44 m through a 0.63 m
                # box because they sit at an angle to it), and a hard slab cuts the lettering
                # mid-glyph. Unity's runtime projection reaches the whole surface it lands on, so
                # the depth bound here only has to stop the decal leaking onto DIFFERENT geometry
                # further along the ray -- which the instance gate above already decided.
                for axis, half in ((ux, hx), (uz, hz), (uy, hy * DEPTH_REACH)):
                    poly = _clip_poly(poly, axis, half)
                    if len(poly) < 3:
                        break
                    poly = _clip_poly(poly, -axis, half)
                    if len(poly) < 3:
                        break
                if len(poly) < 3:
                    continue
                base = len(verts_out)
                for p in poly:
                    world = C + p + nrm * SURFACE_OFFSET_M
                    verts_out.append(world)
                    # UV from the box's own local frame: X across, Z down the image.
                    # U runs against the box's local X: conjugating the projector reversed that
                    # axis's handedness, so reading it directly rendered the artwork mirrored
                    # ("ЯATNU" instead of "UNTAR"). V is unaffected -- the flip is in X only.
                    uvs_out.append((0.5 - (float(np.dot(p, ux)) / (hx * 2)),
                                    (float(np.dot(p, uz)) / (hz * 2)) + 0.5))
                # WINDING: order each fan triangle so its own normal agrees with the surface it
                # was fitted to. Copying the source order is not enough (the clip can reverse a
                # polygon) and hard-coding a flip is what left the plates blank -- back faces,
                # culled. Deriving it from `nrm` is correct in either handedness.
                for k in range(1, len(poly) - 1):
                    tri_n = np.cross(poly[k] - poly[0], poly[k + 1] - poly[0])
                    if float(np.dot(tri_n, nrm)) >= 0.0:
                        faces_out.append((base, base + k, base + k + 1))
                    else:
                        faces_out.append((base, base + k + 1, base + k))

        if len(faces_out) == 0:
            empty += 1
            continue
        if len(faces_out) > MAX_TRIS_PER_DECAL:
            heavy += 1
            continue
        name = "decal_bake_%05d__gen.obj" % di
        with open(os.path.join(mesh_dir, name), "w", encoding="utf-8") as f:
            f.write("g decal_bake_%05d\n" % di)
            for v in verts_out:
                # ALREADY in the assembler's final frame (the receiver was conjugated before
                # clipping) and the instance matrix is identity, whose conjugation is identity.
                # Negating X here would mirror the decal straight back off the surface it was
                # just fitted to.
                f.write("v %.5f %.5f %.5f\n" % (v[0], v[1], v[2]))
            for u, vv in uvs_out:
                f.write("vt %.6f %.6f\n" % (u, vv))
            for a, b, c in faces_out:
                f.write("f %d/%d %d/%d %d/%d\n" % (a + 1, a + 1, b + 1, b + 1, c + 1, c + 1))
        sub = dict(dec["subs"][0])
        # The mesh carries WORLD positions and REAL uv coordinates now, so the instance is identity
        # and the sub's tiling only has to remap the unit square onto the atlas cell.
        dec = dict(dec)
        dec["mesh"] = name
        dec["m"] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        sub["n"] = len(faces_out)
        dec["subs"] = [sub]
        out.append(dec)
        baked += 1
        tri_total += len(faces_out)

    log("  [decals] projected %d decal(s) onto %d triangle(s); %d had no geometry in range, "
        "%d skipped as too dense (> %d tris)" % (baked, tri_total, empty, heavy, MAX_TRIS_PER_DECAL))
    return out
