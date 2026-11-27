"""Rank fitted tifxyz windings by estimated area overlapping papyrus.

Samples are uniform over the triangulated 3D surface, not over its vertices.
A sample counts as material when its nearest volume voxel exceeds --threshold.
Use a papyrus mask or a CT with zero background; raw CT intensity alone does
not identify papyrus reliably. Volumes use ZYX order, meshes use XYZ voxels.
"""

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import sys

import numpy as np
import tifffile

from visualize_meshes import find_meshes, grid_triangles, positive_int


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meshes", type=Path, help="Fit run, meshes/, or meshes/fitted/")
    parser.add_argument("--ct", required=True, type=Path,
                        help="Papyrus mask or background-masked CT Zarr")
    parser.add_argument("--ct-dataset", default="0", help="Array in a Zarr group (default: 0)")
    parser.add_argument("--threshold", type=float, default=0,
                        help="Material means voxel value > threshold (default: 0)")
    parser.add_argument("--samples", type=positive_int, default=100_000,
                        help="Surface samples per mesh (default: 100000)")
    parser.add_argument("--seed", type=int, default=0, help="Reproducible sampling seed (default: 0)")
    parser.add_argument("--sort-by", choices=("coverage", "area"), default="coverage",
                        help="Sort by material percentage or material area (default: coverage)")
    parser.add_argument("--top", type=positive_int, help="Only print the first N results")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--meshes", dest="mesh_ids", type=int, nargs="+", metavar="N",
                           help="Only these winding numbers")
    selection.add_argument("--mesh-range", type=int, nargs=2, metavar=("MIN", "MAX"),
                           help="Inclusive winding-number range")
    parser.add_argument("--spliced", action="store_true", help="Evaluate wNNN_spliced meshes")
    parser.add_argument("--voxel-size-um", type=float,
                        help="Mesh voxel size; default: infer from meta.json area_cm2/area_vx2")
    parser.add_argument("--coordinate-scale", type=float, default=1,
                        help="Volume voxels per mesh voxel (default: 1; e.g. 4 for 9.6 to 2.4 um)")
    parser.add_argument("--volume-origin", type=float, nargs=3, default=(0, 0, 0),
                        metavar=("X", "Y", "Z"),
                        help="First volume voxel in mesh coordinates, for cropped volumes (default: 0 0 0)")
    args = parser.parse_args(argv)
    args.mesh_step = 1  # Same mesh discovery/selection as visualize_meshes.py.
    for name in ("coordinate_scale", "voxel_size_um"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if not np.isfinite(args.threshold) or not np.isfinite(args.volume_origin).all():
        parser.error("--threshold and --volume-origin must be finite")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.mesh_range is not None and args.mesh_range[0] > args.mesh_range[1]:
        parser.error("--mesh-range requires MIN <= MAX")
    if args.mesh_ids is not None and min(args.mesh_ids) < 0:
        parser.error("--meshes requires nonnegative winding numbers")
    return args


def infer_voxel_size(meshes):
    """Infer physical mesh voxel size from the fitter's area metadata."""
    sizes = []
    for _, path in meshes:
        metadata_path = path / "meta.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        area_cm2, area_vx2 = metadata.get("area_cm2"), metadata.get("area_vx2")
        if area_cm2 is not None and area_vx2 is not None and area_cm2 > 0 and area_vx2 > 0:
            sizes.append(np.sqrt(area_cm2 / area_vx2) * 10_000)
    if not sizes:
        return None
    if not np.isfinite(sizes).all() or not np.allclose(sizes, sizes[0], rtol=1e-4):
        raise ValueError("Inconsistent mesh voxel sizes in meta.json; supply --voxel-size-um")
    return float(sizes[0])


def load_triangles(path):
    grid = np.stack([tifffile.imread(path / f"{axis}.tif") for axis in "xyz"], axis=-1)
    grid = grid.astype(np.float64)
    mask_path = path / "mask.tif"
    if mask_path.exists():
        mask = tifffile.imread(mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != grid.shape[:2]:
            raise ValueError(f"mask.tif shape does not match coordinates in {path}")
        grid[mask == 0] = np.nan
    triangles = grid_triangles(grid)
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5
    keep = np.isfinite(areas) & (areas > 0)
    if not keep.any():
        raise ValueError(f"No valid, nondegenerate surface quads in {path}")
    return triangles[keep], areas[keep]


def sample_surface(triangles, areas, count, rng):
    """Draw independent, uniform samples over true 3D triangle area."""
    cdf = np.cumsum(areas)
    indices = np.searchsorted(cdf, rng.random(count) * cdf[-1], side="right")
    chosen = triangles[indices]
    u, v = rng.random((2, count))
    root = np.sqrt(u)
    return ((1 - root[:, None]) * chosen[:, 0]
            + (root * (1 - v))[:, None] * chosen[:, 1]
            + (root * v)[:, None] * chosen[:, 2])


class VolumeSampler:
    """Read only touched chunks, with a bounded cache shared across windings."""

    def __init__(self, volume, cache_bytes=256 * 1024**2):
        self.volume = volume
        self.shape = np.asarray(volume.shape)
        self.chunks = np.asarray(volume.chunks)
        self.cache_bytes = cache_bytes
        self.cache = OrderedDict()
        self.cached_bytes = 0

    def read_chunk(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        start = np.asarray(key) * self.chunks
        stop = np.minimum(start + self.chunks, self.shape)
        chunk = np.asarray(self.volume[tuple(slice(int(a), int(b)) for a, b in zip(start, stop))])
        if chunk.nbytes <= self.cache_bytes:
            while self.cache and self.cached_bytes + chunk.nbytes > self.cache_bytes:
                _, old = self.cache.popitem(last=False)
                self.cached_bytes -= old.nbytes
            self.cache[key] = chunk
            self.cached_bytes += chunk.nbytes
        return chunk

    def material_at(self, xyz, threshold, coordinate_scale=1, origin=(0, 0, 0)):
        zyx = ((xyz - np.asarray(origin)) * coordinate_scale)[:, ::-1]
        # Check floating-point bounds before conversion to avoid integer overflow.
        inside = np.isfinite(zyx).all(1) & ((zyx >= -0.5) & (zyx < self.shape - 0.5)).all(1)
        material = np.zeros(len(xyz), dtype=bool)
        selected = np.flatnonzero(inside)
        indices = np.floor(zyx[inside] + 0.5).astype(np.int64)
        if not len(indices):
            return material, inside
        chunk_indices = indices // self.chunks
        chunk_shape = (self.shape + self.chunks - 1) // self.chunks
        codes = np.ravel_multi_index(chunk_indices.T, chunk_shape)
        order = np.argsort(codes)
        boundaries = np.r_[0, np.flatnonzero(np.diff(codes[order])) + 1, len(order)]
        for lo, hi in zip(boundaries[:-1], boundaries[1:]):
            group = order[lo:hi]
            key = tuple(int(n) for n in chunk_indices[group[0]])
            chunk = self.read_chunk(key)
            local = indices[group] - np.asarray(key) * self.chunks
            values = chunk[tuple(local.T)]
            material[selected[group]] = np.isfinite(values) & (values > threshold)
        return material, inside


def rank_meshes(args):
    import zarr

    meshes = find_meshes(args)
    volume = zarr.open(str(args.ct), mode="r")
    if not isinstance(volume, zarr.Array):
        group = volume
        try:
            volume = group[args.ct_dataset]
        except KeyError as error:
            available = ", ".join(sorted(group.array_keys())) or "(none at group root)"
            raise ValueError(
                f"CT dataset {args.ct_dataset!r} not found in {args.ct}. "
                f"Available arrays: {available}. Choose one with --ct-dataset."
            ) from error
    if not isinstance(volume, zarr.Array) or volume.ndim != 3 or not all(volume.shape):
        raise ValueError("--ct must select a nonempty 3D Zarr array in ZYX order")
    if np.dtype(volume.dtype).kind not in "buif":
        raise ValueError("--ct must contain real numeric or boolean voxels")
    voxel_size = args.voxel_size_um or infer_voxel_size(meshes)
    factor = (voxel_size / 10_000) ** 2 if voxel_size else 1
    unit = "cm2" if voxel_size else "voxel2"
    sampler = VolumeSampler(volume)
    print(f"Material: voxel > {args.threshold:g}; {args.samples:,} area-uniform samples/mesh. "
          f"Volume ZYX: {volume.shape}; coordinate scale: {args.coordinate_scale:g}.", file=sys.stderr)
    print("Coverage is an estimate over the entire valid mesh area; outside-volume samples count as uncovered. "
          "In-volume % helps identify cropping or coordinate mismatches.", file=sys.stderr)
    if voxel_size:
        print(f"Mesh voxel size: {voxel_size:g} um; areas in {unit}.", file=sys.stderr)
    else:
        print("No physical voxel size found; areas in mesh voxel2. Use --voxel-size-um for cm2.", file=sys.stderr)
    results = []
    for number, path in meshes:
        triangles, areas = load_triangles(path)
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, number]))
        xyz = sample_surface(triangles, areas, args.samples, rng)
        material, inside = sampler.material_at(xyz, args.threshold, args.coordinate_scale, args.volume_origin)
        total = float(areas.sum()) * factor
        coverage = float(material.mean())
        results.append(dict(number=number, mesh=path.name, coverage=coverage,
                            area=coverage * total, total=total, inside=float(inside.mean())))
        print(f"{path.name}: {coverage:.1%} material, {inside.mean():.1%} in volume", file=sys.stderr, flush=True)
    results.sort(key=lambda row: (-row[args.sort_by], -row["area"], row["number"]))
    print(f"{'Rank':>4}  {'Mesh':<16}  {'Papyrus %':>10}  "
          f"{'Papyrus ' + unit:>14}  {'Mesh ' + unit:>14}  {'In-volume %':>11}")
    for rank, row in enumerate(results[:args.top], 1):
        print(f"{rank:4d}  {row['mesh']:<16}  {100 * row['coverage']:10.2f}  "
              f"{row['area']:14.3f}  {row['total']:14.3f}  {100 * row['inside']:11.2f}")
    return results


def main(argv=None):
    args = parse_args(argv)
    try:
        rank_meshes(args)
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
