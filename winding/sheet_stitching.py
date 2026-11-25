"""Portable sheet pipeline: XYZ stitching with the original overlap voting.

The original adaptive_links API is sufficient. If the optional histogram helpers
exist, reuse their single-count fast path; otherwise collect metric histograms
locally and call the unchanged original adaptive_links for all link decisions.
"""

from pathlib import Path

import numpy as np

import vesuvius_sheet_tools.assemble_scroll as assemble_scroll
from sheet_io import save_npz, write_json


def overlap_histogram(a, b):
    """Compact pair counts for metrics when the original has no histogram API."""
    both = (a > 0) & (b > 0)
    if not both.any():
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64)
    av = a[both].astype(np.int64)
    bv = b[both].astype(np.int64)
    base = int(bv.max()) + 1
    keys, counts = np.unique(av * base + bv, return_counts=True)
    return np.column_stack((keys // base, keys % base)), counts


def overlap_links(a, b):
    """Use only original voting rules, with optional shared counting for metrics."""
    histogram = getattr(assemble_scroll, "overlap_histogram", None)
    vote = getattr(assemble_scroll, "links_from_histogram", None)
    if callable(histogram) and callable(vote):
        matched, counts = histogram(a, b)
        return matched, counts, vote(matched, counts)
    links = assemble_scroll.adaptive_links(a, b)
    matched, counts = overlap_histogram(a, b)
    return matched, counts, links


class Stitcher:
    """Stitch face neighbors, retaining only ID unions and on-disk pair histograms."""

    def __init__(self, out):
        self.out = out
        self.parent = {}
        self.pairs = []

    def find(self, key):
        parent = self.parent
        parent.setdefault(key, key)
        root = key
        while parent[root] != root:
            root = parent[root]
        while parent[key] != key:
            previous = parent[key]
            parent[key] = root
            key = previous
        return root

    def cache_faces(self, block, labels, grid, starts, overlap):
        """Persist only the forward-facing bands; consume each band once."""
        cache = self.out / "stitch_cache"
        cache.mkdir(exist_ok=True)
        for axis in range(3):
            if grid[axis] + 1 == len(starts[axis]):
                continue
            region = [slice(None)] * 3
            region[axis] = slice(-int(overlap[axis]), None)
            np.save(cache / f"{block['index']}_{axis}.npy", labels[tuple(region)])

    def link(self, previous, current, labels, axis):
        start = np.maximum(previous["origin_zyx"], current["origin_zyx"])
        end = np.minimum(np.add(previous["origin_zyx"], previous["shape_zyx"]),
                         np.add(current["origin_zyx"], current["shape_zyx"]))
        if np.any(end <= start):
            return
        def region(block):
            return tuple(slice(int(lo - o), int(hi - o))
                         for lo, hi, o in zip(start, end, block["origin_zyx"]))
        cached = self.out / "stitch_cache" / f"{previous['index']}_{axis}.npy"
        band = np.load(cached, mmap_mode="r", allow_pickle=False)
        aa = np.asarray(band)
        bb = labels[region(current)]
        matched, counts, links = overlap_links(aa, bb)
        i, j = previous["index"], current["index"]
        for a, b in links:
            root_a, root_b = self.find((i, a)), self.find((j, b))
            if root_a != root_b:
                self.parent[root_b] = root_a
        relative = Path("overlaps") / f"pair_{len(self.pairs):06d}.npz"
        (self.out / relative).parent.mkdir(exist_ok=True)
        save_npz(self.out / relative, matched=matched, counts=counts)
        self.pairs.append(dict(crop_a=i, crop_b=j, links=len(links),
                               overlap_origin_zyx=start.tolist(),
                               overlap_shape_zyx=(end - start).tolist(),
                               overlap_foreground_voxels=int(counts.sum()),
                               pair_histogram=relative.as_posix()))
        del aa, band
        cached.unlink()

    def finish(self, blocks):
        global_ids, table = {}, {}
        for block in blocks:
            mapping = {}
            for local in block.pop("ids"):
                root = self.find((block["index"], local))
                if root not in global_ids:
                    global_ids[root] = len(global_ids) + 1
                mapping[str(local)] = global_ids[root]
            z, y, x = block["origin_zyx"]
            table[f"z{z}/y{y}_x{x}"] = mapping
        for pair in self.pairs:
            with np.load(self.out / pair["pair_histogram"], allow_pickle=False) as data:
                agree = sum(int(n) for (a, b), n in zip(data["matched"], data["counts"])
                            if self.find((pair["crop_a"], int(a))) ==
                            self.find((pair["crop_b"], int(b))))
            total = pair["overlap_foreground_voxels"]
            pair["agreement_rate"] = agree / total if total else None
        metrics = dict(global_instances=len(global_ids), crop_count=len(blocks),
                       pairs=self.pairs,
                       note="Face-neighbor overlap votes; global IDs are not winding numbers.")
        write_json(self.out / "global_table.json", table)
        write_json(self.out / "assembly_metrics.json", metrics)
        return metrics
