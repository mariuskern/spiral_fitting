"""Spatial prefilter for the brute-force point->patch linking fallback.

Upstream links each PCL point to a patch by projecting it onto *every* candidate
patch (``point_collection._link_collection_to_patch_subset``), and each such
projection tests the point against *every* triangle of that patch
(``tifxyz.Patch.project``). That is the fallback for when the native
``vc.surface_index`` extension is missing -- which it is whenever the Python
scripts run without a VC3D CMake build, the normal state on a workstation.
Measured on the PHerc. Paris 4 z 4000-5000 fit (~2,800 ROI-filtered patches,
219,405 points): 10.1 s per point collection, a 41-hour tqdm ETA over the
14,658 collections before the first optimisation step. The patch count is not
the cost driver -- the next paragraph is.

Both levels need pruning, and measurements say the second dominates: the
verified patches are bands with millions of triangles spanning the whole fitted
region, so even a perfect patch-level filter still pays ~0.5 s per point just
to project onto the one band that wins. (A patch-box-only version measured a
mere 2.5x.)

The pruning argument, used twice, is the standard lower-bound one:

  * every triangle of a tile lies inside that tile's vertex bounding box, so
    point-to-box distance is a lower bound on point-to-triangle distance;
  * the linking loop discards any patch whose projection distance exceeds
    ``tolerance`` (strict ``>``), so a face -- or a whole patch -- whose box
    lower bound already exceeds ``tolerance`` cannot influence the outcome;
  * among the faces that survive, the *unchanged* upstream arithmetic runs
    (same formulas, same float32, same ``eps``, same first-of-ties ``argmin``
    with original relative face order preserved), so accepted links carry
    bit-identical distances and ij coordinates, and rejected patches are
    rejected in both versions. The box tests are padded by ``_PAD`` so float
    rounding in the lower bound can only ever let extra faces through, never
    drop a real candidate.

Tile boxes bound only the *valid* faces (invalid quads are excluded from the
box), which tightens the filter and stays sound because invalid faces are
excluded from upstream's search too. A valid quad with a non-finite vertex
would poison its tile box, so such tiles fall back to an infinite box --
always searched, never wrong.

Install with :func:`install`, which rebinds the upstream symbol; both callers
(the general all-patches fallback and the between-patches pair case) go
through it, and both use the tile-local projection (`_INDEX_MIN_PATCHES = 1`;
the pair case's verbatim-upstream projection was measured at ~50 s per pair
collection at full-scroll scale because the pairs are the same million-face
bands).
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

import point_collection as _pc

# Quads per tile side. Small enough to localise a band patch, large enough to
# keep tile counts and per-point box tests cheap.
_TILE = 16

# Width in z voxels of one bucket of the patch-level index.
_Z_BUCKET = 32.0

# Safety pad (voxels) on every box test. The box distance is a lower bound in
# exact arithmetic; padding keeps it one under float rounding as well.
_PAD = 0.05

# Candidate sets of any size go through the tile machinery. This used to be
# 64, keeping the between-patches pair case on upstream's own Patch.project
# for maximum fidelity -- but at full-scroll scale those pairs are band
# patches with millions of faces, and verbatim projection took ~50 s per
# pair collection (74 of them). The tile-local projection is covered by the
# same lower-bound theorem and the pair patches' face indexes are already
# cached from the global index, so the pair case costs microseconds instead.
_INDEX_MIN_PATCHES = 1

_PATCH_INDEX_CACHE: Dict[Any, Any] = {}
_FACE_INDEX_CACHE: Dict[int, Any] = {}
_PATCH_INDEX_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Per-patch face index: tile boxes over the valid-quad grid.
# ---------------------------------------------------------------------------

def _face_index_for(patch) -> Dict[str, Any]:
    key = id(patch)
    cached = _FACE_INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    zyxs = patch.zyxs
    arr = zyxs.detach().cpu().numpy() if hasattr(zyxs, 'detach') else np.asarray(zyxs)
    arr = np.asarray(arr, dtype=np.float32)
    h, w, _ = arr.shape
    qh, qw = h - 1, w - 1

    valid = patch.valid_quad_mask
    valid_np = valid.detach().cpu().numpy() if hasattr(valid, 'detach') else np.asarray(valid)
    valid_np = valid_np.astype(bool).reshape(max(qh, 0), max(qw, 0))

    if qh <= 0 or qw <= 0 or not valid_np.any():
        # No valid faces: upstream's project returns inf for such a patch and
        # the linking loop discards it, so it must never become a candidate.
        index = {
            'h': h, 'w': w, 'qh': qh, 'qw': qw, 'th': 0, 'tw': 0,
            'tile_lo': np.zeros((0, 3)), 'tile_hi': np.zeros((0, 3)),
            'valid': valid_np,
            'vertices': (zyxs if hasattr(zyxs, 'reshape') and hasattr(zyxs, 'to')
                         else torch.as_tensor(arr)).reshape(-1, 3),
            'patch_lo': np.full(3, np.inf),
            'patch_hi': np.full(3, -np.inf),
        }
        _FACE_INDEX_CACHE[key] = index
        return index

    # Per-quad boxes over the quad's four corners, in float64 so the reductions
    # below cannot themselves round a bound the wrong way.
    tl = arr[:-1, :-1].astype(np.float64)
    tr = arr[:-1, 1:].astype(np.float64)
    bl = arr[1:, :-1].astype(np.float64)
    br = arr[1:, 1:].astype(np.float64)
    quad_lo = np.minimum(np.minimum(tl, tr), np.minimum(bl, br))
    quad_hi = np.maximum(np.maximum(tl, tr), np.maximum(bl, br))

    # Invalid quads cannot win and must not widen tile boxes.
    quad_lo[~valid_np] = np.inf
    quad_hi[~valid_np] = -np.inf

    # A valid quad with a non-finite corner cannot be bounded: make its tile
    # unbounded so it is always searched.
    bad = valid_np & ~(np.isfinite(quad_lo).all(axis=-1) & np.isfinite(quad_hi).all(axis=-1))
    if bad.any():
        quad_lo[bad] = -np.inf
        quad_hi[bad] = np.inf

    th = -(-qh // _TILE)
    tw = -(-qw // _TILE)
    pad_lo = np.full((th * _TILE, tw * _TILE, 3), np.inf)
    pad_hi = np.full((th * _TILE, tw * _TILE, 3), -np.inf)
    pad_lo[:qh, :qw] = quad_lo
    pad_hi[:qh, :qw] = quad_hi
    tile_lo = pad_lo.reshape(th, _TILE, tw, _TILE, 3).min(axis=(1, 3)).reshape(-1, 3)
    tile_hi = pad_hi.reshape(th, _TILE, tw, _TILE, 3).max(axis=(1, 3)).reshape(-1, 3)

    index = {
        'h': h, 'w': w, 'qh': qh, 'qw': qw, 'th': th, 'tw': tw,
        'tile_lo': tile_lo, 'tile_hi': tile_hi,
        'valid': valid_np,
        'vertices': (zyxs if hasattr(zyxs, 'reshape') and hasattr(zyxs, 'to')
                     else torch.as_tensor(arr)).reshape(-1, 3),
        # Whole-patch box over valid quads, for the patch-level index.
        'patch_lo': tile_lo.min(axis=0),
        'patch_hi': tile_hi.max(axis=0),
    }
    _FACE_INDEX_CACHE[key] = index
    return index


def _box_dist2(point: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Squared distance from a point to axis-aligned boxes (0 inside)."""
    d = np.maximum(np.maximum(lo - point[None, :], point[None, :] - hi), 0.0)
    return (d * d).sum(axis=1)


def _project_local(patch, points_t: torch.Tensor, points_np: np.ndarray,
                   tolerance: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """``Patch.project`` for a batch of points, on the faces near them.

    Returns the same ``(ij, distance)`` per point as upstream would:
    bit-identical whenever the patch is accepted for that point (distance <=
    tolerance), and *some* distance greater than tolerance whenever upstream's
    would also be greater (possibly inf here where upstream computes a finite
    value -- both are then discarded by the caller's strict > test).

    The face subset is the union over the batch. For any single point that is
    a superset of its own nearby faces, and a superset can only add faces
    farther than tolerance from that point: they cannot beat an accepted
    minimum, and in the rejected case the value is discarded anyway. Ties among
    accepted minima all lie within tolerance, hence inside every subset
    involved, and face order is preserved -- so argmin picks the same face.
    """
    fi = _face_index_for(patch)
    device = points_t.device
    n = points_np.shape[0]

    def miss() -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.full((n, 2), -1.0, device=device),
                torch.full((n,), float('inf'), device=device, dtype=torch.float32))

    limit = (tolerance + _PAD) ** 2
    if fi['tile_lo'].shape[0] == 0:
        return miss()
    # (n_tiles, n_points) squared box distances, reduced over the batch.
    d = np.maximum(
        np.maximum(fi['tile_lo'][:, None, :] - points_np[None, :, :],
                   points_np[None, :, :] - fi['tile_hi'][:, None, :]),
        0.0)
    tiles = np.flatnonzero(((d * d).sum(axis=2) <= limit).any(axis=1))
    if tiles.size == 0:
        return miss()

    # Quad ids inside the surviving tiles, ascending; then valid ones only.
    ti = tiles // fi['tw']
    tj = tiles % fi['tw']
    qi = (ti[:, None] * _TILE + np.arange(_TILE)[None, :]).reshape(-1)
    qj = (tj[:, None] * _TILE + np.arange(_TILE)[None, :]).reshape(-1)
    qi = qi[qi < fi['qh']]
    qj = qj[qj < fi['qw']]
    quads = (np.unique(qi)[:, None] * fi['qw'] + np.unique(qj)[None, :]).reshape(-1)
    # np.unique(qi/qj) above deduplicates rows/cols across tiles but their cross
    # product covers tile *rectangles* jointly -- a superset of the surviving
    # tiles, which is sound (extra faces only), and keeps this fully vectorised.
    quads = quads[fi['valid'].reshape(-1)[quads]]
    if quads.size == 0:
        return miss()
    quads.sort()

    w = fi['w']
    i = quads // fi['qw']
    j = quads % fi['qw']
    v_tl = i * w + j
    v_tr = v_tl + 1
    v_bl = v_tl + w
    v_br = v_bl + 1

    # Upstream face order is [all type-0 faces in quad order] + [all type-1
    # faces in quad order]; build the subset in exactly that order so argmin
    # tie-breaking picks the same face.
    quads_t = torch.from_numpy(quads).to(device)
    faces = torch.cat([
        torch.stack([torch.from_numpy(v_bl), torch.from_numpy(v_tl), torch.from_numpy(v_tr)], dim=1),
        torch.stack([torch.from_numpy(v_bl), torch.from_numpy(v_tr), torch.from_numpy(v_br)], dim=1),
    ], dim=0).to(device)
    base_faces = torch.cat([
        torch.stack([torch.from_numpy(i), torch.from_numpy(j)], dim=1),
        torch.stack([torch.from_numpy(i), torch.from_numpy(j)], dim=1),
    ], dim=0).to(device)
    face_type = torch.cat([
        torch.zeros(quads_t.shape[0], dtype=torch.long, device=device),
        torch.ones(quads_t.shape[0], dtype=torch.long, device=device),
    ], dim=0)

    vertices = fi['vertices'].to(device=device, dtype=torch.float32)
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    ab = b - a
    ac = c - a

    # From here on this is upstream Patch.project's arithmetic, verbatim
    # (points broadcast over the face subset, like its chunk loop).
    eps = 1e-8
    P = points_t.reshape(-1, 1, 3).to(dtype=torch.float32)

    AP = P - a
    BP = P - b
    CP = P - c

    d1 = (AP * ab).sum(-1)
    d2 = (AP * ac).sum(-1)
    d3 = (BP * ab).sum(-1)
    d4 = (BP * ac).sum(-1)
    d5 = (CP * ab).sum(-1)
    d6 = (CP * ac).sum(-1)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    mask_a = (d1 <= 0) & (d2 <= 0)
    mask_b = (d3 >= 0) & (d4 <= d3)
    mask_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    mask_c = (d6 >= 0) & (d5 <= d6)
    mask_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    mask_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    mask_face = ~(mask_a | mask_b | mask_ab | mask_c | mask_ac | mask_bc)

    bary = torch.zeros((P.shape[0], faces.shape[0], 3), dtype=torch.float32, device=device)
    bary[..., 0][mask_a] = 1.0
    bary[..., 1][mask_b] = 1.0

    v_ab = d1 / (d1 - d3 + eps)
    bary[..., 0][mask_ab] = 1 - v_ab[mask_ab]
    bary[..., 1][mask_ab] = v_ab[mask_ab]

    bary[..., 2][mask_c] = 1.0

    w_ac = d2 / (d2 - d6 + eps)
    bary[..., 0][mask_ac] = 1 - w_ac[mask_ac]
    bary[..., 2][mask_ac] = w_ac[mask_ac]

    w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6) + eps)
    bary[..., 1][mask_bc] = 1 - w_bc[mask_bc]
    bary[..., 2][mask_bc] = w_bc[mask_bc]

    denom = (va + vb + vc) + eps
    v_face = vb / denom
    w_face = vc / denom
    bary[..., 1][mask_face] = v_face[mask_face]
    bary[..., 2][mask_face] = w_face[mask_face]
    bary[..., 0][mask_face] = 1 - v_face[mask_face] - w_face[mask_face]

    closest = (
        bary[..., 0:1] * a
        + bary[..., 1:2] * b
        + bary[..., 2:3] * c
    )
    dist2 = ((closest - P) ** 2).sum(-1)
    idx = dist2.argmin(dim=1)

    batch_idx = torch.arange(P.shape[0], device=device)
    best_bary = bary[batch_idx, idx]
    best_base = base_faces[idx]
    best_type = face_type[idx]

    i_off = torch.where(best_type == 0, best_bary[:, 0], best_bary[:, 0] + best_bary[:, 2])
    j_off = torch.where(best_type == 0, best_bary[:, 2], best_bary[:, 1] + best_bary[:, 2])

    ij = torch.stack([best_base[:, 0].float() + i_off,
                      best_base[:, 1].float() + j_off], dim=1)
    return ij, dist2[batch_idx, idx].sqrt()


# ---------------------------------------------------------------------------
# Patch-level index: whole-patch boxes bucketed by z.
# ---------------------------------------------------------------------------

def _patch_index_for(candidate_patches: Dict[str, Any], tolerance: float):
    key = (id(candidate_patches), len(candidate_patches), float(tolerance))
    cached = _PATCH_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    with _PATCH_INDEX_LOCK:
        return _patch_index_build(key, candidate_patches, tolerance)


def _patch_index_build(key, candidate_patches: Dict[str, Any], tolerance: float):
    cached = _PATCH_INDEX_CACHE.get(key)
    if cached is not None:  # built while this thread waited on the lock
        return cached

    # The candidate filter works on TILE boxes pooled across all patches, not
    # whole-patch boxes. Whole-patch boxes collapse at full-scroll scale: the
    # band patches' boxes cover most of the volume, so with ~46k patches
    # loaded every point ended up projecting onto dozens of bands and the
    # linking phase measured ~30x slower per collection than the 4000-slice
    # probe. A tile box only admits a patch whose *surface* actually comes
    # near the point, which is the minimum sound candidate set. Soundness is
    # the same lower-bound argument as inside _project_local: the distance to
    # a patch surface is >= the distance to the nearest of its tile boxes.
    patch_ids = list(candidate_patches.keys())

    tile_lo: List[np.ndarray] = []
    tile_hi: List[np.ndarray] = []
    owner: List[np.ndarray] = []
    for k, patch_id in enumerate(patch_ids):
        fi = _face_index_for(candidate_patches[patch_id])
        n = fi['tile_lo'].shape[0]
        if n == 0:
            continue
        tile_lo.append(fi['tile_lo'])
        tile_hi.append(fi['tile_hi'])
        owner.append(np.full(n, k, dtype=np.int64))

    if not tile_lo:
        index = {'patch_ids': patch_ids, 'lo': np.zeros((0, 3)),
                 'hi': np.zeros((0, 3)), 'owner': np.zeros(0, dtype=np.int64),
                 'base': 0.0, 'buckets': [], 'everywhere': np.zeros(0, dtype=np.int64)}
        _PATCH_INDEX_CACHE[key] = index
        return index

    lo = np.concatenate(tile_lo, axis=0)
    hi = np.concatenate(tile_hi, axis=0)
    owner_arr = np.concatenate(owner, axis=0)

    pad = tolerance + _PAD
    finite = np.isfinite(lo[:, 0]) & np.isfinite(hi[:, 0])
    everywhere = np.flatnonzero(~finite)  # unbounded tiles: always candidates
    buckets: List[np.ndarray] = []
    base = 0.0
    if finite.any():
        z_lo = lo[:, 0] - pad
        z_hi = hi[:, 0] + pad
        base = float(z_lo[finite].min())
        first = np.where(finite, np.floor((z_lo - base) / _Z_BUCKET), 0).astype(np.int64)
        last = np.where(finite, np.floor((z_hi - base) / _Z_BUCKET), 0).astype(np.int64)
        n_buckets = int(last[finite].max()) + 1

        # Counting sort; per-bucket python appends over millions of tiles
        # would cost more than the index saves.
        spans = np.where(finite, last - first + 1, 0)
        total = int(spans.sum())
        rows = np.repeat(np.arange(lo.shape[0]), spans)
        offsets = np.repeat(np.cumsum(spans) - spans, spans)
        bucket_of = np.repeat(first, spans) + (np.arange(total) - offsets)

        order = np.argsort(bucket_of, kind='stable')
        rows = rows[order]
        bounds = np.searchsorted(bucket_of[order], np.arange(n_buckets + 1))
        for bkt in range(n_buckets):
            buckets.append(rows[bounds[bkt]:bounds[bkt + 1]])

    index = {'patch_ids': patch_ids, 'lo': lo, 'hi': hi, 'owner': owner_arr,
             'base': base, 'buckets': buckets, 'everywhere': everywhere}
    _PATCH_INDEX_CACHE[key] = index
    return index


def _candidate_patches_for(point_np: np.ndarray, index, tolerance: float) -> np.ndarray:
    """Ascending patch indices with a tile box within tolerance of the point."""
    if not index['buckets']:
        if index['everywhere'].size:
            return np.unique(index['owner'][index['everywhere']])
        return np.arange(len(index['patch_ids']), dtype=np.int64)
    slab = int(np.floor((point_np[0] - index['base']) / _Z_BUCKET))
    tiles = (index['buckets'][slab] if 0 <= slab < len(index['buckets'])
             else np.zeros(0, dtype=np.int64))
    if index['everywhere'].size:
        tiles = np.concatenate([tiles, index['everywhere']])
    if tiles.size == 0:
        return tiles
    limit = (tolerance + _PAD) ** 2
    close = _box_dist2(point_np, index['lo'][tiles], index['hi'][tiles]) <= limit
    hit = tiles[close]
    if hit.size == 0:
        return hit
    # np.unique sorts, so candidates come back in patch dict order.
    return np.unique(index['owner'][hit])


# ---------------------------------------------------------------------------
# The linking loop itself: upstream's, with the two filters slotted in.
# ---------------------------------------------------------------------------

def link_collection_to_patch_subset(
    links: Dict[str, List[Any]],
    collection_id: int,
    collection: Dict[str, Any],
    candidate_patches: Dict[str, Any],
    tolerance: float,
    hit_policy: str = 'nearest',
) -> None:
    point_items = list(collection['points'].items())
    if not point_items or not candidate_patches:
        return

    device = torch.device('cpu')
    tolerance_t = torch.tensor(tolerance, device=device, dtype=torch.float32)

    use_index = len(candidate_patches) >= _INDEX_MIN_PATCHES
    if not use_index:
        # Few candidates (the between-patches pair case): upstream's own loop,
        # with upstream's own Patch.project.
        patch_ids = list(candidate_patches.keys())
        for point_id, point in point_items:
            point_t = torch.as_tensor(
                point.get('zyx', point['p'][::-1]), dtype=torch.float32, device=device)

            nearest_patch_id = None
            nearest_distance = torch.tensor(float('inf'), device=device)
            nearest_ij = None
            best_area = -1.0
            for patch_id in patch_ids:
                ij_coord, distance = candidate_patches[patch_id].project(point_t)
                if distance > tolerance_t:
                    continue
                if hit_policy == 'largest_area':
                    area = float(candidate_patches[patch_id].area)
                    is_better = area > best_area or (area == best_area and distance < nearest_distance)
                else:
                    is_better = distance < nearest_distance
                if is_better:
                    nearest_distance = distance
                    nearest_patch_id = patch_id
                    nearest_ij = ij_coord
                    best_area = float(candidate_patches[patch_id].area)
            if nearest_patch_id:
                _pc._record_point_patch_link(
                    links, collection, collection_id, point_id, point,
                    nearest_patch_id,
                    float(nearest_distance.cpu().item()),
                    nearest_ij.tolist())
        return

    index = _patch_index_for(candidate_patches, float(tolerance))
    patch_ids = index['patch_ids']

    points_np = np.stack([
        np.asarray(point.get('zyx', point['p'][::-1]), dtype=np.float32)
        for _, point in point_items
    ], axis=0).astype(np.float64, copy=False)
    points_t = torch.from_numpy(points_np.astype(np.float32))

    # Rows grouped by candidate patch, so each patch is projected once for all
    # of its points. Patch keys ascend, and per point the projections are then
    # visited in that same ascending order, reproducing upstream's per-point
    # candidate order exactly.
    rows_by_patch: Dict[int, List[int]] = {}
    for row in range(points_np.shape[0]):
        for k in _candidate_patches_for(points_np[row], index, float(tolerance)):
            rows_by_patch.setdefault(int(k), []).append(row)

    # results[row] = list of (candidate rank k, ij row tensor, distance tensor),
    # appended in ascending k because rows_by_patch is visited in sorted order.
    results: Dict[int, List[Tuple[int, torch.Tensor, torch.Tensor]]] = {}
    for k in sorted(rows_by_patch):
        rows = rows_by_patch[k]
        patch = candidate_patches[patch_ids[k]]
        sub_np = points_np[rows]
        sub_t = points_t[rows]
        ij_batch, dist_batch = _project_local(patch, sub_t, sub_np, float(tolerance))
        for local_i, row in enumerate(rows):
            results.setdefault(row, []).append(
                (k, ij_batch[local_i], dist_batch[local_i]))

    for row, (point_id, point) in enumerate(point_items):
        hits = results.get(row)
        if not hits:
            continue

        nearest_patch_id = None
        nearest_distance = torch.tensor(float('inf'), device=device)
        nearest_ij = None
        best_area = -1.0
        for k, ij_coord, distance in hits:
            if distance > tolerance_t:
                continue
            patch_id = patch_ids[k]
            if hit_policy == 'largest_area':
                area = float(candidate_patches[patch_id].area)
                is_better = area > best_area or (area == best_area and distance < nearest_distance)
            else:
                is_better = distance < nearest_distance
            if is_better:
                nearest_distance = distance
                nearest_patch_id = patch_id
                nearest_ij = ij_coord
                best_area = float(candidate_patches[patch_id].area)

        if nearest_patch_id:
            _pc._record_point_patch_link(
                links,
                collection,
                collection_id,
                point_id,
                point,
                nearest_patch_id,
                float(nearest_distance.cpu().item()),
                nearest_ij.tolist(),
            )


# # ---------------------------------------------------------------------------
# # Threaded execution: capture upstream's serial per-collection calls, run them
# # on a pool, merge in the original call order.
# # ---------------------------------------------------------------------------
# #
# # Upstream's fallback loops over 14k collections serially, one core busy out of
# # 32. Each captured call is independent (it writes only its own collection's
# # points plus a private local links dict), so they parallelise cleanly; merging
# # the local results in *capture order* reproduces the serial links structure
# # exactly -- same lists, same order -- so this changes wall-clock, not output.

# # Keep a direct reference so callers can select the fast implementation
# # without globally monkey-patching point_collection (resident fits may use
# # different linking backends in the same Python process).
# _UPSTREAM_LPTP = _pc.link_points_to_patches
# _ORIGINAL_LPTP = None


# def _worker_thread_count() -> int:
#     override = os.environ.get('FIT_SPIRAL_LINK_THREADS')
#     if override:
#         try:
#             if int(override) > 0:
#                 return int(override)
#         except ValueError:
#             pass
#     # Measured 2026-07-29 on the 600-patch/10-collection sample: 24 workers ran
#     # 7.6x SLOWER than serial (GIL thrash over fine-grained numpy/torch glue).
#     # Per-patch batching is where the speed comes from; keep this serial unless
#     # someone explicitly experiments via the env var.
#     return 1


# def link_points_to_patches(*args: Any, **kwargs: Any):
#     """Fast, output-equivalent point-to-patch linking.

#     Unlike :func:`install`, this is safe to call alongside the original
#     ``point_collection.link_points_to_patches`` in one Python process.
#     """
#     captured: List[Tuple] = []

#     def capture(links, collection_id, collection, candidate_patches,
#                 tolerance, hit_policy='nearest'):
#         captured.append((links, collection_id, collection, candidate_patches,
#                          tolerance, hit_policy))

#     previous = _pc._link_collection_to_patch_subset
#     _pc._link_collection_to_patch_subset = capture
#     try:
#         links = _UPSTREAM_LPTP(*args, **kwargs)
#     finally:
#         _pc._link_collection_to_patch_subset = previous

#     if not captured:
#         return links

#     # Prebuild the shared patch index once, not 24 times under a lock.
#     for _, _, _, candidate_patches, tolerance, _ in captured:
#         if len(candidate_patches) >= _INDEX_MIN_PATCHES:
#             _patch_index_for(candidate_patches, float(tolerance))
#             break

#     def run_one(item):
#         target_links, collection_id, collection, candidate_patches, tolerance, hit_policy = item
#         local: Dict[str, List[Any]] = {}
#         link_collection_to_patch_subset(
#             local, collection_id, collection, candidate_patches,
#             tolerance, hit_policy=hit_policy)
#         return target_links, local

#     workers = _worker_thread_count()
#     progress = getattr(_pc, 'tqdm', None)

#     # The per-call tensor work is tiny after prefiltering; intra-op torch
#     # threading would only thrash against the outer pool.
#     old_torch_threads = torch.get_num_threads()
#     torch.set_num_threads(1)
#     try:
#         with ThreadPoolExecutor(max_workers=workers) as pool:
#             results = pool.map(run_one, captured)
#             if progress is not None:
#                 results = progress(results, 'linking points to patches (threaded)',
#                                    total=len(captured))
#             for target_links, local in results:  # map() preserves capture order
#                 for patch_id, patch_links in local.items():
#                     target_links.setdefault(patch_id, []).extend(patch_links)
#     finally:
#         torch.set_num_threads(old_torch_threads)

#     return links


# # Compatibility name for launchers using install().
# _threaded_link_points_to_patches = link_points_to_patches


# def install() -> None:
#     """Rebind the upstream fallback to the prefiltered, threaded version."""
#     global _ORIGINAL_LPTP
#     for name in ('_link_collection_to_patch_subset', 'link_points_to_patches',
#                  '_record_point_patch_link'):
#         if not hasattr(_pc, name):
#             raise RuntimeError(
#                 f'point_collection.{name} is gone; upstream restructured the '
#                 'linking fallback -- re-check fast_link rather than silently '
#                 'running unpatched')
#     _pc._link_collection_to_patch_subset = link_collection_to_patch_subset
#     if _ORIGINAL_LPTP is None:
#         _ORIGINAL_LPTP = _pc.link_points_to_patches
#         _pc.link_points_to_patches = _threaded_link_points_to_patches
