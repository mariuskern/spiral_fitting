"""Equivalence + speed check for fast_link against upstream's brute force.

Loads real patches and real point collections (nothing synthetic: the whole
claim is about this dataset's geometry), links a sample of collections both
ways, and compares the resulting link structures exactly -- patch ids, point
ids, ij coordinates and distances, plus the ``on_patch`` annotation the linker
writes back into the point dicts.

Sampling is over collections, not points: a collection is the unit upstream
iterates, and each one is linked against the full patch set, so a sampled
collection exercises the full 45k-patch search.

Usage:
    python test_fast_link.py [--patches N] [--collections N]
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

SPIRAL = Path('/root/villa/volume-cartographer/scripts/spiral')
sys.path.insert(0, str(SPIRAL))
sys.path.insert(0, '/root')

DATASET = Path(os.environ.get('FIT_DATASET', '/root/spiral-dataset/PHercParis4'))

import point_collection as pc  # noqa: E402
from tifxyz import load_tifxyz  # noqa: E402

import fast_link  # noqa: E402


def load_patches(limit: int) -> dict:
    patches = {}
    for directory in (DATASET / 'verified_patches', DATASET / 'unverified_patches'):
        for entry in sorted(os.listdir(directory)):
            if len(patches) >= limit:
                return patches
            try:
                patches[entry] = load_tifxyz(str(directory / entry))
            except Exception:
                continue
    return patches


def load_collections(limit: int) -> dict:
    """Take collections from the same json fit_spiral loads, in its order."""
    collections = {}
    next_id = 0
    for name in ('patch-overlap-pcls.json', 'relative_windings.json',
                 'same_windings.json', 'abs_winding.json'):
        path = DATASET / name
        if not path.is_file():
            continue
        for pcl in (pc.load_point_collection(str(path)) or {}).values():
            pcl['source_file'] = str(path)
            pcl['sampling_group'] = str(path)
            pcl.setdefault('metadata', {})
            collections[next_id] = pcl
            next_id += 1
            if len(collections) >= limit:
                return collections
    return collections


def normalise(links: dict) -> dict:
    """Links as plain comparable data, sorted so dict/list order cannot mask a diff."""
    out = {}
    for patch_id, entries in links.items():
        rows = []
        for link in entries:
            rows.append((
                int(link.point_id),
                int(link.collection_id),
                str(link.collection_name),
                tuple(float(v) for v in link.point_zyx),
                tuple(float(v) for v in link.ij_coords),
                float(link.distance),
                float(link.winding_annotation),
            ))
        out[str(patch_id)] = sorted(rows)
    return out


def on_patch_of(collections: dict) -> dict:
    out = {}
    for cid, collection in collections.items():
        for pid, point in collection['points'].items():
            hit = point.get('on_patch')
            if hit is not None:
                out[(cid, pid)] = (str(hit['id']), float(hit['distance']),
                                   tuple(float(v) for v in hit['ij']))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--patches', type=int, default=3000)
    ap.add_argument('--collections', type=int, default=12)
    ap.add_argument('--tolerance', type=float, default=10.0)
    args = ap.parse_args()

    t0 = time.time()
    patches = load_patches(args.patches)
    collections = load_collections(args.collections)
    print(f'loaded {len(patches)} patches, {len(collections)} collections, '
          f'{sum(len(c["points"]) for c in collections.values())} points '
          f'in {time.time() - t0:.1f}s')
    if not patches or not collections:
        sys.exit('nothing to compare -- check the dataset path')

    for policy in ('nearest', 'largest_area'):
        slow_collections = copy.deepcopy(collections)
        fast_collections = copy.deepcopy(collections)

        slow_links: dict = {}
        t0 = time.time()
        for cid, collection in slow_collections.items():
            pc._link_collection_to_patch_subset(
                slow_links, cid, collection, patches, args.tolerance, hit_policy=policy)
        slow_seconds = time.time() - t0

        fast_links: dict = {}
        t0 = time.time()
        for cid, collection in fast_collections.items():
            fast_link.link_collection_to_patch_subset(
                fast_links, cid, collection, patches, args.tolerance, hit_policy=policy)
        fast_seconds = time.time() - t0

        same_links = normalise(slow_links) == normalise(fast_links)
        same_points = on_patch_of(slow_collections) == on_patch_of(fast_collections)
        linked = sum(len(v) for v in slow_links.values())
        speedup = slow_seconds / fast_seconds if fast_seconds > 0 else float('inf')
        print(f'[{policy:12s}] links {"IDENTICAL" if same_links else "DIFFER"}, '
              f'on_patch {"IDENTICAL" if same_points else "DIFFERS"}, '
              f'{linked} links over {len(slow_links)} patches | '
              f'upstream {slow_seconds:.1f}s vs fast {fast_seconds:.1f}s = {speedup:.1f}x')
        if not (same_links and same_points):
            sys.exit(f'{policy}: MISMATCH')

    # Phase 2: the threaded orchestration must reproduce the serial one.
    # Both sides use the fast per-collection function (its equivalence to
    # upstream was just proven above); what differs is only execution -- pool
    # vs loop -- so any difference here is an ordering or merge bug.
    fast_link.install()

    serial_collections = copy.deepcopy(collections)
    threaded_collections = copy.deepcopy(collections)

    lptp_serial = fast_link._ORIGINAL_LPTP
    pc._link_collection_to_patch_subset = fast_link.link_collection_to_patch_subset
    t0 = time.time()
    serial_links = lptp_serial(patches, serial_collections, tolerance=args.tolerance)
    serial_seconds = time.time() - t0

    t0 = time.time()
    threaded_links = pc.link_points_to_patches(
        patches, threaded_collections, tolerance=args.tolerance)
    threaded_seconds = time.time() - t0

    def ordered(links):
        return {
            str(pid): [(l.point_id, l.collection_id, tuple(l.point_zyx),
                        tuple(l.ij_coords), l.distance) for l in entries]
            for pid, entries in links.items()
        }

    same_order = ordered(serial_links) == ordered(threaded_links)
    same_points2 = on_patch_of(serial_collections) == on_patch_of(threaded_collections)
    print(f'[threaded    ] links+order {"IDENTICAL" if same_order else "DIFFER"}, '
          f'on_patch {"IDENTICAL" if same_points2 else "DIFFERS"} | '
          f'serial {serial_seconds:.1f}s vs threaded {threaded_seconds:.1f}s')
    if not (same_order and same_points2):
        sys.exit('threaded: MISMATCH')

    print('PASS: prefiltered linking reproduces upstream exactly')


if __name__ == '__main__':
    main()
