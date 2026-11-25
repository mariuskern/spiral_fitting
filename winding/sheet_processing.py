"""Portable sheet pipeline: array processing assembled from original helpers.

Cleaning/seeding and the split driver originally live inside CLI main functions.
They are composed here without changing those entry points. Merge and centerline
consolidation always use the original implementations.
"""

import numpy as np
from scipy import ndimage
from skimage.exposure import equalize_adapthist
from skimage.segmentation import watershed

from vesuvius_sheet_tools.separate_sheets import merge_instances
from vesuvius_sheet_tools.split_stacked import (MEGA_FRAC, MIN_CENTER_CC, CLAHE_KERNEL, CLAHE_CLIP,
                           SIGMA_INT, NMS_STEP, P_MIN, structure_normals,
                           consolidate_centerlines)


def segment(mask, args, compute):
    """Original watershed/merge recipe, adding coverage for seedless components."""
    dist = compute.edt(mask)
    markers, components, n = compute.seeds(mask, dist, args.core_dist, args.min_core)
    seeded = np.unique(components[markers > 0])
    missing = np.setdiff1d(np.arange(1, n + 1), seeded)
    if missing.size:
        positions = ndimage.maximum_position(dist, components, missing)
        next_id = int(markers.max()) + 1
        for i, position in enumerate(positions):
            markers[tuple(position)] = next_id + i
    del components
    labels = watershed(-dist, markers, mask=mask).astype(np.int32)
    del markers
    before = int(labels.max())
    if not args.no_merge:
        labels, _, _, _ = merge_instances(labels, dist)
    return labels, dict(instances_before_merge=before, fallback_seeds=int(missing.size))


def split_labels(labels, ct, mega_fraction=MEGA_FRAC, min_center=MIN_CENTER_CC, compute=None):
    """Split each candidate independently, preserving foreground and other IDs."""
    result = labels.copy()
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    candidates = ids[counts >= mega_fraction * counts.sum()]
    report = []
    if not candidates.size:
        return result, report
    # Normalize CT to [0,1] for CLAHE, including float CT outside skimage's range.
    low, high = float(ct.min()), float(ct.max())
    eq = np.empty(ct.shape, dtype=np.float32)
    for z, image in enumerate(ct):
        normalized = (image.astype(np.float32) - low) / max(high - low, 1e-12)
        eq[z] = equalize_adapthist(normalized, kernel_size=CLAHE_KERNEL, clip_limit=CLAHE_CLIP)
    normals, planarity = (structure_normals(eq) if compute is None
                          else compute.structure_normals(eq))
    smooth = ndimage.gaussian_filter(eq, SIGMA_INT)
    del eq
    next_id = int(labels.max()) + 1
    for instance in candidates:
        region = labels == instance
        positions = np.array(np.nonzero(region)).T
        idx = tuple(positions.T)
        coords = positions.T.astype(np.float32)
        nrm = normals[idx].T
        val = smooth[idx]
        up = ndimage.map_coordinates(smooth, coords + NMS_STEP * nrm, order=1, mode="nearest")
        down = ndimage.map_coordinates(smooth, coords - NMS_STEP * nrm, order=1, mode="nearest")
        ridge = (val > up) & (val > down) & (planarity[idx] >= P_MIN)
        centers = np.zeros(labels.shape, dtype=bool)
        centers[tuple(positions[ridge].T)] = True
        center_lab, _ = ndimage.label(centers)
        if ridge.any():
            lut = consolidate_centerlines(center_lab, positions[ridge].astype(np.float32),
                                          nrm.T[ridge], center_lab[tuple(positions[ridge].T)])
            center_lab = lut[center_lab]
        sizes = np.bincount(center_lab.ravel())
        keep = sizes >= min_center
        keep[0] = False
        center_lab = np.where(keep[center_lab], center_lab, 0)
        seeds = np.unique(center_lab[center_lab > 0])
        changed = False
        if len(seeds) >= 2:
            assigned = watershed(-smooth, center_lab, mask=region)
            # Do not replace a label unless every original voxel is covered.
            if np.all(assigned[region] > 0):
                for seed in seeds:
                    result[assigned == seed] = next_id
                    next_id += 1
                changed = True
        report.append(dict(instance=int(instance), surviving_centerlines=len(seeds), split=changed))
    return result, report
