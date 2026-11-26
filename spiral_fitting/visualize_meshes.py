"""Overlay colored tifxyz mesh cross-sections on slices of a CT Zarr volume.

Mesh XYZ coordinates and CT voxels must use the same origin and resolution.
Use --out for image export on machines without a graphical display.
"""

import argparse
from pathlib import Path
import re

import numpy as np
import tifffile


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meshes", type=Path, help="Fit run, meshes directory, or meshes/fitted directory")
    parser.add_argument("--ct", required=True, type=Path, help="CT Zarr array or group")
    parser.add_argument("--ct-dataset", help="Array inside the CT group (default: 0; ignored for a direct array)")
    positions = parser.add_mutually_exclusive_group()
    positions.add_argument("--num-slices", type=positive_int, default=6,
                           help="Number of evenly spaced sections (default: 6)")
    positions.add_argument("--slices", type=int, nargs="+", metavar="Z", help="Explicit CT Z indices")
    parser.add_argument("--z-range", type=int, nargs=2, metavar=("MIN", "MAX"),
                        help="Inclusive Z range (default: overlap of meshes and CT)")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--meshes", dest="mesh_ids", type=int, nargs="+", metavar="N",
                           help="Only these winding numbers, e.g. 10 27 33")
    selection.add_argument("--mesh-range", type=int, nargs=2, metavar=("MIN", "MAX"),
                           help="Inclusive winding-number range")
    parser.add_argument("--mesh-step", type=positive_int, default=1,
                        help="Show every Nth mesh in the selected, sorted list")
    parser.add_argument("--spliced", action="store_true", help="Use wNNN_spliced instead of wNNN meshes")
    parser.add_argument("--columns", type=positive_int, default=3, help="Maximum number of subplot columns")
    parser.add_argument("--cmap", default="turbo", help="Matplotlib mesh colormap (default: turbo)")
    parser.add_argument("--colorbar-step", type=positive_int,
                        help="Label every Nth selected mesh; default: at most 16 labels")
    parser.add_argument("--line-width", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.9, help="Mesh opacity (default: 0.9)")
    parser.add_argument("--ct-window", type=float, nargs=2, metavar=("MIN", "MAX"),
                        help="Grayscale limits; default: first slice's nonzero 1st/99.5th percentiles")
    parser.add_argument("--xy-range", type=float, nargs=4, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                        help="Zoom all panels to the same XY rectangle in CT voxels")
    parser.add_argument("--out", type=Path, help="Save PNG/PDF/etc. instead of opening a window")
    parser.add_argument("--dpi", type=positive_int, default=160, help="Export resolution (default: 160)")
    args = parser.parse_args(argv)
    if not np.isfinite(args.line_width) or args.line_width <= 0:
        parser.error("--line-width must be positive and finite")
    if not np.isfinite(args.alpha) or not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")
    for name in ("z_range", "mesh_range", "ct_window"):
        values = getattr(args, name)
        if values is not None and (not np.isfinite(values).all() or values[0] > values[1]):
            parser.error(f"--{name.replace('_', '-')} requires finite MIN <= MAX")
    if args.ct_window is not None and args.ct_window[0] == args.ct_window[1]:
        parser.error("--ct-window requires MIN < MAX")
    if args.xy_range is not None:
        x0, x1, y0, y1 = args.xy_range
        if not np.isfinite(args.xy_range).all() or x0 >= x1 or y0 >= y1:
            parser.error("--xy-range requires finite XMIN < XMAX and YMIN < YMAX")
    return args


def find_meshes(args):
    root = args.meshes
    for relative in ("meshes/fitted", "fitted"):
        if (root / relative).is_dir():
            root = root / relative
            break
    pattern = re.compile(r"w(\d+)" + (r"_spliced" if args.spliced else ""))
    found = []
    if not root.is_dir():
        raise ValueError(f"Mesh directory does not exist: {root}")
    for path in root.iterdir():
        match = pattern.fullmatch(path.name)
        if match and path.is_dir():
            found.append((int(match[1]), path))
    if args.mesh_ids is not None:
        missing = set(args.mesh_ids) - {number for number, _ in found}
        if missing:
            raise ValueError(f"Mesh numbers not found: {sorted(missing)}")
        found = [(number, path) for number, path in found if number in args.mesh_ids]
    if args.mesh_range is not None:
        lo, hi = args.mesh_range
        found = [(number, path) for number, path in found if lo <= number <= hi]
    found.sort()
    if len({number for number, _ in found}) != len(found):
        raise ValueError("Multiple directories have the same winding number")
    if not found:
        raise ValueError(f"No selected {'spliced ' if args.spliced else ''}wNNN tifxyz meshes in {root}")
    return found[::args.mesh_step]


def grid_triangles(grid):
    """Triangulate valid tifxyz quads using the fitter's diagonal convention."""
    if grid.ndim != 3 or grid.shape[-1] != 3 or min(grid.shape[:2]) < 2:
        raise ValueError(f"Expected an H x W x 3 grid with H,W >= 2, got {grid.shape}")
    valid = np.isfinite(grid).all(-1) & ~(grid == -1).all(-1)
    quads = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    tl, tr = grid[:-1, :-1][quads], grid[:-1, 1:][quads]
    bl, br = grid[1:, :-1][quads], grid[1:, 1:][quads]
    return np.concatenate((np.stack((bl, tl, tr), axis=1), np.stack((bl, tr, br), axis=1)))


def intersect_triangles(triangles, z):
    """Return XY line segments for every triangle intersecting z=constant.

    Includes multiple crossings of folded surfaces and edges on the plane.
    Fully coplanar triangles have no unique section line and are omitted.
    """
    triangles = np.asarray(triangles)
    if not len(triangles):
        return np.empty((0, 2, 2))
    zs = triangles[:, :, 2]
    active = (zs.min(1) <= z) & (zs.max(1) >= z) & ~np.all(zs == z, axis=1)
    triangles = triangles[active]
    if not len(triangles):
        return np.empty((0, 2, 2))
    starts, ends = triangles, triangles[:, [1, 2, 0]]
    dz = ends[:, :, 2] - starts[:, :, 2]
    crossings = ((starts[:, :, 2] <= z) & (ends[:, :, 2] >= z)
                 | (ends[:, :, 2] <= z) & (starts[:, :, 2] >= z)) & (dz != 0)
    t = np.divide(z - starts[:, :, 2], dz, out=np.zeros_like(dz, dtype=float), where=dz != 0)
    points = starts[:, :, :2] + t[:, :, None] * (ends[:, :, :2] - starts[:, :, :2])
    pairs = np.array([[0, 1], [0, 2], [1, 2]])
    segments = points[:, pairs]
    # A vertex on the plane can appear twice. Choose the farthest valid pair.
    distance2 = ((segments[:, :, 1] - segments[:, :, 0]) ** 2).sum(-1)
    distance2[~crossings[:, pairs].all(-1)] = -1
    best = distance2.argmax(1)
    keep = distance2[np.arange(len(best)), best] > 1e-16
    return segments[np.arange(len(best))[keep], best[keep]]


def choose_slices(args, depth, z_bounds):
    if args.z_range is None:
        lo = max(0, int(np.ceil(z_bounds[0])))
        hi = min(depth - 1, int(np.floor(z_bounds[1])))
    else:
        lo, hi = args.z_range
        if lo < 0 or hi >= depth:
            raise ValueError(f"--z-range must be within CT indices 0..{depth - 1}")
    if args.slices is not None:
        values = np.asarray(args.slices)
        if (values < 0).any() or (values >= depth).any():
            raise ValueError(f"--slices must be within CT indices 0..{depth - 1}")
        if len(np.unique(values)) != len(values):
            raise ValueError("--slices must contain distinct indices")
        if args.z_range is not None and ((values < lo).any() or (values > hi).any()):
            raise ValueError("--slices must lie inside --z-range")
        return values
    if lo > hi:
        raise ValueError("Selected meshes do not overlap the CT Z range")
    count = hi - lo + 1
    if args.num_slices > count:
        raise ValueError(f"Requested {args.num_slices} sections but only {count} CT slices are available")
    # Midpoints of equally sized bins give distinct indices even at full depth.
    return lo + np.floor((np.arange(args.num_slices) + 0.5) * count / args.num_slices).astype(int)


def visualize(args):
    import matplotlib
    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    import zarr

    meshes = []
    for number, path in find_meshes(args):
        grid = np.stack([tifffile.imread(path / f"{axis}.tif") for axis in "xyz"], axis=-1)
        triangles = grid_triangles(grid)
        if not len(triangles):
            raise ValueError(f"No valid mesh quads in {path}")
        meshes.append((number, triangles))
    ct = zarr.open(str(args.ct), mode="r")
    if not isinstance(ct, zarr.Array):
        ct = ct[args.ct_dataset if args.ct_dataset is not None else "0"]
    if not isinstance(ct, zarr.Array) or ct.ndim != 3 or not all(ct.shape):
        raise ValueError("CT must be a nonempty 3D Zarr array in ZYX order; use --ct-dataset for a group")
    bounds = (min(t[:, :, 2].min() for _, t in meshes), max(t[:, :, 2].max() for _, t in meshes))
    slices = choose_slices(args, ct.shape[0], bounds)
    numbers = np.array([n for n, _ in meshes])
    norm = Normalize(numbers.min() - (len(numbers) == 1) * 0.5,
                     numbers.max() + (len(numbers) == 1) * 0.5)
    cmap = matplotlib.colormaps[args.cmap]
    columns = min(args.columns, len(slices))
    rows = (len(slices) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns + 1, 5 * rows),
                             squeeze=False, layout="constrained")
    window = args.ct_window
    try:
        for ax, z in zip(axes.flat, slices):
            image = np.asarray(ct[int(z)])
            if window is None:
                finite = image[np.isfinite(image) & (image != 0)]
                window = list(np.percentile(finite, [1, 99.5])) if finite.size else [0, 1]
                if window[0] == window[1]:
                    window = [min(0, window[0]), max(1, window[1])]
            ax.imshow(image, cmap="gray", vmin=window[0], vmax=window[1], interpolation="nearest")
            segment_count = 0
            for number, triangles in meshes:
                segments = intersect_triangles(triangles, int(z))
                segment_count += len(segments)
                ax.add_collection(LineCollection(segments, colors=[cmap(norm(number))],
                                                 linewidths=args.line_width, alpha=args.alpha))
            if args.xy_range:
                x0, x1, y0, y1 = args.xy_range
                ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
            else:
                ax.set_xlim(-0.5, ct.shape[2] - 0.5); ax.set_ylim(ct.shape[1] - 0.5, -0.5)
            ax.set_title(f"Z = {z}" + (" (no mesh intersection)" if not segment_count else ""))
            ax.set_xlabel("X [voxel]"); ax.set_ylabel("Y [voxel]")
            print(f"Z={z}: {len(meshes)} meshes, {segment_count} intersection segments", flush=True)
        visible_axes = list(axes.flat)[:len(slices)]
        for ax in list(axes.flat)[len(slices):]:
            ax.set_visible(False)
        step = args.colorbar_step or max(1, int(np.ceil((len(numbers) - 1) / 15)))
        ticks = np.unique(np.r_[numbers[::step], numbers[-1]])
        bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                           ax=visible_axes, ticks=ticks, fraction=0.025, pad=0.02)
        bar.set_ticklabels([f"w{n:03d}" for n in ticks])
        bar.set_label("Mesh / winding number")
        fig.suptitle(f"CT and mesh cross-sections — {len(meshes)} windings")
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.out, dpi=args.dpi)
            print(f"Saved {args.out.resolve()}")
        else:
            plt.show()
    finally:
        plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    try:
        visualize(args)
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
