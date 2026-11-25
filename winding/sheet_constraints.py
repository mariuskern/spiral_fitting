"""Portable umbilicus, same-/relative-winding constraints from local_run.json.

Original numerical helpers are imported, not copied. Only manifest-based I/O,
bounded aggregation and coordinate adapters are new. Use --only-umbilicus for
axis estimation alone; the original make_umbilicus.py needs no modifications.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

if __package__:
    from vesuvius_sheet_tools import make_umbilicus as original_axis
else:
    import vesuvius_sheet_tools.make_umbilicus as original_axis


def winding_modules():
    if __package__:
        from vesuvius_sheet_tools import make_same_windings, make_relative_windings, validate_constraints
    else:
        import vesuvius_sheet_tools.make_same_windings as make_same_windings
        import vesuvius_sheet_tools.make_relative_windings as make_relative_windings
        import vesuvius_sheet_tools.validate_constraints as validate_constraints
    return make_same_windings, make_relative_windings, validate_constraints


def int3(value, name, minimum=0):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(type(v) is not int or v < minimum for v in value)):
        raise ValueError(f"{name} must contain three integers >= {minimum} in ZYX order")
    return np.asarray(value, dtype=np.int64)


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class LocalRun:
    """Composite arbitrary overlapping blocks in bounded batches of XY planes.

    Mask overlaps are ORed. Global-ID conflicts remain -1 and cannot produce
    trusted sample points or relative edges. No whole-volume dense array exists.
    """

    def __init__(self, run_dir, scale, full_shape, batch_slices=8):
        self.root = Path(run_dir).resolve()
        self.scale = np.asarray(scale, dtype=float)
        self.full_shape = int3(full_shape, "full-shape", 1)
        if self.scale.shape != (3,) or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("scale must contain three positive finite ZYX factors")
        if batch_slices < 1:
            raise ValueError("batch-slices must be >= 1")
        self.batch_slices = batch_slices
        status = self.root / "pipeline_status.json"
        if status.exists():
            state = json.loads(status.read_text(encoding="utf-8"))
            if not isinstance(state, dict) or not state.get("complete"):
                raise ValueError("Input pipeline run is incomplete; finish or resume it first")
        manifest = json.loads((self.root / "local_run.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("local_run.json must be an object")
        if manifest.get("axis_order", "ZYX") != "ZYX":
            raise ValueError("local_run.json must use ZYX axis order")
        self.blocks = []
        for entry in manifest["crops"]:
            origin = int3(entry["origin_zyx"], "block origin")
            shape = int3(entry["shape_zyx"], "block shape", 1)
            if np.any((origin + shape - 1) * self.scale >= self.full_shape):
                raise ValueError("Block lies outside full-shape after scaling")
            path = (self.root / entry["block"]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Block path must be inside the run directory")
            self.blocks.append((origin, shape, path))
        if not self.blocks:
            raise ValueError("No blocks in local_run.json")
        self.lower = np.min([o for o, _, _ in self.blocks], axis=0)
        self.upper = np.max([o + s for o, s, _ in self.blocks], axis=0)
        self.zmin, self.zmax = int(self.lower[0]), int(self.upper[0] - 1)
        if self.zmin == self.zmax:
            raise ValueError("Need at least two Z slices")
        self.tables = None

    def planes(self, zs, mapped=False):
        zs = list(zs)
        if mapped and self.tables is None:
            self.tables = json.loads((self.root / "global_table.json").read_text(encoding="utf-8"))
        for start in range(0, len(zs), self.batch_slices):
            batch = zs[start:start + self.batch_slices]
            canvases = np.zeros((len(batch), *(self.upper - self.lower)[1:]),
                                dtype=np.int64 if mapped else bool)
            for origin, shape, path in self.blocks:
                selected = [(i, z - int(origin[0])) for i, z in enumerate(batch)
                            if origin[0] <= z < origin[0] + shape[0]]
                if not selected:
                    continue
                with np.load(path, allow_pickle=False) as data:
                    labels = data["labels"]
                    if "origin" in data and not np.array_equal(data["origin"], origin):
                        raise ValueError(f"Block origin differs from manifest: {path}")
                if (labels.shape != tuple(shape) or not np.issubdtype(labels.dtype, np.integer)
                        or np.any(labels < 0)):
                    raise ValueError(f"Invalid block labels or shape: {path}")
                if mapped:
                    z, y, x = origin
                    table = self.tables[f"z{z}/y{y}_x{x}"]
                    ids = np.unique(labels)
                    values = [0 if v == 0 else table[str(int(v))] for v in ids]
                    if any(type(v) is not int or not 0 <= v <= np.iinfo(np.int64).max for v in values):
                        raise ValueError(f"Invalid global IDs: {path}")
                    lut = np.asarray(values, dtype=np.int64)
                    if np.any(lut[ids > 0] == 0):
                        raise ValueError(f"Foreground cannot map to background: {path}")
                y, x = origin[1:] - self.lower[1:]
                for i, local_z in selected:
                    region = canvases[i, y:y + shape[1], x:x + shape[2]]
                    if not mapped:
                        region |= labels[local_z] > 0
                    else:
                        incoming = lut[np.searchsorted(ids, labels[local_z])]
                        conflict = ((region < 0) | ((region > 0) & (incoming > 0)
                                                   & (region != incoming)))
                        take = (region == 0) & (incoming > 0)
                        region[take] = incoming[take]
                        region[conflict] = -1
                del labels
            for z, canvas in zip(batch, canvases):
                yield z, canvas


def axis_controls(run, args):
    """Original centroid recipe and clipped-edge running median, generic geometry."""
    zs = sorted(set(range(run.zmin, run.zmax + 1, args.z_step)) | {run.zmax})
    raw, previews = [], {}
    preview_zs = {zs[0], zs[len(zs) // 2], zs[-1]}
    for z, mask in tqdm(run.planes(zs), total=len(zs), desc="Umbilicus", unit="slice",
                        disable=args.no_progress):
        if z in preview_zs:
            previews[z] = mask.copy()
        if int(mask.sum()) >= args.min_pixels:
            # Row/column reductions avoid allocating coordinates of every voxel.
            count_y, count_x = mask.sum(axis=1), mask.sum(axis=0)
            raw.append([count_y @ np.arange(mask.shape[0]) / count_y.sum() + run.lower[1],
                        count_x @ np.arange(mask.shape[1]) / count_x.sum() + run.lower[2]])
        else:
            raw.append([np.nan, np.nan])
    raw = np.asarray(raw)
    valid = np.isfinite(raw).all(axis=1)
    if args.umbilicus:
        document = json.loads(args.umbilicus.read_text(encoding="utf-8"))
        controls = np.array([[p["z"], p["y"], p["x"]] for p in document["control_points"]], dtype=float)
        method = "provided_umbilicus"
    elif args.axis:
        controls = np.array([[run.zmin, args.axis[1], args.axis[0]],
                             [run.zmax, args.axis[1], args.axis[0]]]) * run.scale
        method = "provided_constant_axis"
    else:
        if valid.sum() < 2 or not valid[0] or not valid[-1]:
            raise ValueError("Need usable foreground on first and last slices and >=2 samples; "
                             "adjust min-pixels or supply an axis/umbilicus")
        smooth = np.column_stack([original_axis.running_median(
            np.interp(zs, np.asarray(zs)[valid], raw[valid, axis]), args.smooth_window)
            for axis in range(2)])
        controls = np.column_stack([zs, smooth]) * run.scale
        method = "foreground_mask_centroid"
    if (controls.ndim != 2 or controls.shape[1] != 3 or len(controls) < 2
            or not np.isfinite(controls).all() or np.any(controls < 0)
            or np.any(controls >= run.full_shape) or np.any(np.diff(controls[:, 0]) <= 0)):
        raise ValueError("Umbilicus requires >=2 finite in-bounds points with strictly increasing Z")
    if controls[0, 0] > run.zmin * run.scale[0] or controls[-1, 0] < run.zmax * run.scale[0]:
        raise ValueError("Umbilicus must cover all input Z slices")
    max_jump = float(np.linalg.norm(np.diff(controls[:, 1:], axis=0), axis=1).max())
    if max_jump >= args.max_jump:
        raise ValueError(f"Umbilicus jump {max_jump:g} exceeds max-jump {args.max_jump:g}")
    document = dict(control_points=[dict(z=p[0], y=p[1], x=p[2]) for p in controls.tolist()])
    missing_z = np.asarray(zs)[~valid].tolist()
    report = dict(method=method, missing_centroid_input_z=missing_z,
                  interpolated_input_z=missing_z if method == "foreground_mask_centroid" else [],
                  max_jump_fullres=max_jump, control_points=len(controls),
                  smoothing="make_umbilicus.running_median (clipped edges)",
                  note="Mask centroids are initial axis estimates, not verified winding cores.")
    return document, controls, report, (zs, raw, previews)


def axis_at(controls, run, z):
    return np.array([np.interp(z * run.scale[0], controls[:, 0], controls[:, axis]) / run.scale[axis]
                     for axis in (1, 2)])


def axis_figure(path, run, controls, plotting):
    zs, raw, previews = plotting
    plt = original_axis.plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    try:
        for axis, label, plot in ((1, "Y", axes[0, 0]), (2, "X", axes[0, 1])):
            plot.plot(np.asarray(zs) * run.scale[0], raw[:, axis - 1] * run.scale[axis], ".", label="Mask centroid")
            plot.plot(controls[:, 0], controls[:, axis], "-", label="Axis")
            plot.set(xlabel="Z [full-res voxels]", ylabel=f"{label} [full-res voxels]")
            plot.legend()
        axes[0, 2].axis("off")
        axes[0, 2].text(0, 1, "Centroid-based initial estimate\nReview the winding core against CT.",
                        color=original_axis.INK, va="top", transform=axes[0, 2].transAxes)
        for plot, z in zip(axes[1], sorted(previews)):
            plot.imshow(previews[z], cmap="gray", origin="lower",
                        extent=[(run.lower[2] - .5) * run.scale[2], (run.upper[2] - .5) * run.scale[2],
                                (run.lower[1] - .5) * run.scale[1], (run.upper[1] - .5) * run.scale[1]])
            center = axis_at(controls, run, z) * run.scale[1:]
            plot.plot(center[1], center[0], "r+")
            plot.set(title=f"Z={z * run.scale[0]:g}", xlabel="X [full-res voxels]", ylabel="Y [full-res voxels]")
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def reservoir(record, name, points, capacity, rng):
    """Uniform bounded candidate reservoir using independent random priorities."""
    if not len(points):
        return
    keys = rng.random(len(points))
    if len(points) > capacity:
        select = np.argpartition(keys, capacity - 1)[:capacity]
        points, keys = points[select], keys[select]
    if name in record:
        old_points, old_keys = record[name]
        points, keys = np.concatenate([old_points, points]), np.r_[old_keys, keys]
    if len(points) > capacity:
        select = np.argpartition(keys, capacity - 1)[:capacity]
        points, keys = points[select], keys[select]
    record[name] = points, keys


def same_windings(run, controls, args, original):
    records, total, conflicts = {}, 0, 0
    rng = np.random.default_rng(original.SEED)
    planes = iter(run.planes(range(run.zmin, run.zmax + 1), mapped=True))
    _, current = next(planes)
    previous = current
    for z in tqdm(range(run.zmin, run.zmax + 1), desc="Same windings", unit="slice", disable=args.no_progress):
        following = next(planes, (None, current))[1]
        deep = original.interior_mask(np.stack([previous != 0, current != 0, following != 0]))[1]
        total += int(np.count_nonzero(current))
        conflicts += int(np.count_nonzero(current < 0))
        yy, xx = np.nonzero(current > 0)
        ids = current[yy, xx]
        order = np.argsort(ids, kind="stable")
        ids, yy, xx = ids[order], yy[order], xx[order]
        starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]) if len(ids) else []
        ends = list(starts[1:]) + [len(ids)]
        center = axis_at(controls, run, z) - run.lower[1:]
        radii = np.hypot(yy - center[0], xx - center[1])
        angles = ((np.arctan2(yy - center[0], xx - center[1]) + np.pi) % (2 * np.pi))
        bins = np.minimum((angles * original.A_BINS / (2 * np.pi)).astype(int), original.A_BINS - 1)
        for start, end in zip(starts, ends):
            gid = int(ids[start])
            rec = records.setdefault(gid, dict(volume=0, deep=0, moments={}))
            rec["volume"] += int(end - start)
            inside = deep[yy[start:end], xx[start:end]]
            rec["deep"] += int(inside.sum())
            ab, rr = bins[start:end], radii[start:end]
            moments = rec["moments"].setdefault(z // args.z_step, np.zeros((3, original.A_BINS)))
            moments[0] += np.bincount(ab, minlength=original.A_BINS)
            moments[1] += np.bincount(ab, weights=rr, minlength=original.A_BINS)
            moments[2] += np.bincount(ab, weights=rr * rr, minlength=original.A_BINS)
            points = np.column_stack([np.full(end - start, z), yy[start:end] + run.lower[1],
                                      xx[start:end] + run.lower[2]])
            reservoir(rec, "all", points, args.candidates_per_instance, rng)
            reservoir(rec, "interior", points[inside], args.candidates_per_instance, rng)
        previous, current = current, following
    collections, rejected = {}, defaultdict(int)
    for gid, rec in records.items():
        if rec["volume"] < args.min_volume:
            rejected["volume"] += 1
            continue
        if args.max_instance_fraction < 1 and rec["volume"] >= args.max_instance_fraction * total:
            rejected["mega"] += 1
            continue
        if rec["deep"] / rec["volume"] > args.max_deep_frac:
            rejected["deep_fraction"] += 1
            continue
        moments = np.concatenate(list(rec["moments"].values()), axis=1)
        sigma = original.local_radial_sigma(dict(an=moments[0], ar=moments[1], ar2=moments[2]))
        if sigma > original.RADIAL_SIGMA_FACTOR * args.pitch_vox:
            rejected["radial"] += 1
            continue
        points, _ = rec.get("interior", rec["all"])
        registry = original.PointSet(min_dist=args.point_spacing)
        original.sample_instance(points, args.points_per_instance, registry, rng)
        if len(registry.points) < original.K_MIN:
            rejected["sampling"] += 1
            continue
        collections[str(gid)] = collection(f"inst_{gid}", registry.points, run.scale)
    return collections, dict(candidates=len(records), rejected=dict(rejected),
                             foreground_voxels=total, conflicting_voxels=conflicts)


def collection(name, points, scale, winds=None, color=None):
    entries = {}
    for i, point in enumerate(points):
        item = dict(p=(np.asarray(point) * scale)[::-1].tolist(), creation_time=0)
        if winds is not None:
            item["wind_a"] = float(winds[i])
        entries[str(i)] = item
    return dict(name=name, points=entries, metadata={}, color=color or [80, 180, 240])


class WindingGraph:
    """Weighted unions reject inconsistent winding offsets in near-linear time."""

    def __init__(self):
        self.parent, self.weight, self.rank = {}, {}, {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x], self.weight[x], self.rank[x] = x, 0, 0
        if self.parent[x] != x:
            root, delta = self.find(self.parent[x])
            self.weight[x] += delta
            self.parent[x] = root
        return self.parent[x], self.weight[x]

    def add(self, inner, outer):
        a, wa = self.find(inner)
        b, wb = self.find(outer)
        if a == b:
            return wb - wa == 1
        if self.rank[a] < self.rank[b]:
            self.parent[a], self.weight[a] = b, wb - wa - 1
        else:
            self.parent[b], self.weight[b] = a, 1 + wa - wb
            if self.rank[a] == self.rank[b]:
                self.rank[a] += 1
        return True


def ray_observations(canvas, cy, cx, theta, step, original):
    """Keep original ray runs, rejecting conflicts and mixed touching labels."""
    radius = np.arange(0, np.hypot(max(abs(cy), abs(canvas.shape[0] - cy)),
                                   max(abs(cx), abs(canvas.shape[1] - cx))) + step, step)
    fy, fx = cy + radius * np.sin(theta), cx + radius * np.cos(theta)
    valid = (fy >= 0) & (fx >= 0) & (fy < canvas.shape[0]) & (fx < canvas.shape[1])
    radius = radius[valid]
    ids = canvas[fy[valid].astype(int), fx[valid].astype(int)]
    if not len(ids) or np.any(ids < 0):
        return []
    edges = np.flatnonzero(np.r_[True, (ids[1:] > 0) != (ids[:-1] > 0), True])
    for start, end in zip(edges[:-1], edges[1:]):
        if ids[start] > 0 and np.any(ids[start:end] != ids[start]):
            return []
    return original.ray_runs(canvas, cy, cx, theta, radius)


def relative_windings(run, controls, same, args, original):
    zs = range(run.zmin, run.zmax + 1, args.z_step)
    observations = {}
    for z, canvas in tqdm(run.planes(zs, mapped=True), total=len(zs), desc="Relative windings",
                          unit="slice", disable=args.no_progress):
        cy, cx = axis_at(controls, run, z) - run.lower[1:]
        for ray in range(args.rays):
            runs = ray_observations(canvas, cy, cx, 2 * np.pi * ray / args.rays, args.ray_step, original)
            for a, b in zip(runs, runs[1:]):
                if a[1] == b[1] or str(a[1]) not in same or str(b[1]) not in same:
                    continue
                ratio = (b[0] - a[0]) * args.ray_step / args.pitch_vox
                if not original.GAP_LO_FRAC <= ratio <= original.GAP_HI_FRAC:
                    continue
                key = int(a[1]), int(b[1])
                rec = observations.setdefault(key, dict(support=set(), best=None))
                rec["support"].add((z, ray))
                if rec["best"] is None or abs(ratio - 1) < rec["best"][0]:
                    rec["best"] = abs(ratio - 1), ratio, [[z, a[2] + run.lower[1], a[3] + run.lower[2]],
                                                                       [z, b[2] + run.lower[1], b[3] + run.lower[2]]]
    graph, accepted, collections, ratios = WindingGraph(), [], {}, []
    rejected = defaultdict(int)
    for pair, rec in sorted(observations.items(), key=lambda item: (-len(item[1]["support"]), item[0])):
        if pair[::-1] in observations:
            rejected["orientation"] += 1
        elif len(rec["support"]) < args.min_support:
            rejected["support"] += 1
        elif not graph.add(*pair):
            rejected["cycle"] += 1
        else:
            accepted.append(pair)
            _, ratio, points = rec["best"]
            ratios.append(ratio)
            color = original.PALETTE[len(collections) % len(original.PALETTE)]
            collections[str(len(collections))] = collection(f"adjacent_{pair[0]}_{pair[1]}", points,
                                                          run.scale, [0, 1], color)
    violations, _ = original.cycle_check(accepted)
    if violations:
        raise ValueError("Original relative-winding cycle check failed")
    histogram = None
    if ratios:
        shape_ok, hard_ok, lines = original.ratio_histogram(ratios)
        histogram = dict(shape_ok=bool(shape_ok), hard_ok=bool(hard_ok), lines=lines)
    return collections, dict(candidate_pairs=len(observations), rejected=dict(rejected),
                             cycle_violations=len(violations),
                             ratio_diagnostic=histogram)


def generate(args):
    if (args.z_step < 1 or args.min_pixels < 1 or args.smooth_window < 1
            or args.smooth_window % 2 != 1 or not np.isfinite(args.max_jump) or args.max_jump <= 0):
        raise ValueError("Require positive sampling/jump parameters and an odd smooth-window")
    out = args.out if args.out is not None else args.run_dir / "portable_constraints"
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory must be empty: {out}")
    if not args.only_umbilicus:
        same_module, relative_module, validator = winding_modules()
        defaults = dict(min_volume=same_module.MIN_VOL, points_per_instance=same_module.K_MAX,
                        candidates_per_instance=same_module.CANDS_PER_TILE,
                        point_spacing=same_module.MIN_DIST, max_instance_fraction=same_module.MEGA_FRAC,
                        max_deep_frac=same_module.MAX_DEEP_FRAC,
                        rays=relative_module.N_RAYS, min_support=relative_module.MIN_SUPPORT)
        for key, default in defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, default)
        if (args.pitch_vox is None or not np.isfinite(args.pitch_vox) or args.pitch_vox <= 0
                or not np.isfinite(args.point_spacing) or args.point_spacing <= 0
                or not np.isfinite(args.ray_step) or args.ray_step <= 0
                or args.rays < 3 or args.min_support < 3 or args.min_volume < 1
                or args.points_per_instance < 3 or args.candidates_per_instance < args.points_per_instance
                or not 0 < args.max_instance_fraction <= 1 or not 0 <= args.max_deep_frac <= 1):
            raise ValueError("Invalid winding parameters; provide positive pitch, spacing and valid filters")
    run = LocalRun(args.run_dir, args.scale, args.full_shape, args.batch_slices)
    axis_doc, controls, axis_report, plotting = axis_controls(run, args)
    documents = {"umbilicus.json": axis_doc}
    report = dict(source=str(run.root), scale_zyx=run.scale.tolist(), full_shape_zyx=run.full_shape.tolist(),
                  axis=axis_report, note="Experimental constraints require geometric review; no seed patch or absolute windings inferred.")
    if not args.only_umbilicus:
        same, same_report = same_windings(run, controls, args, same_module)
        relative, relative_report = relative_windings(run, controls, same, args, relative_module)
        for name, content in (("same_windings.json", same), ("relative_windings.json", relative), ("abs_winding.json", {})):
            documents[name] = dict(vc_pointcollections_json_version="1", collections=content)
        errors, warnings = [], []
        validator.validate_umbilicus(axis_doc, errors, warnings, full_shape=run.full_shape)
        for name, document in documents.items():
            if name != "umbilicus.json":
                validator.validate_point_collection(document, errors, warnings, full_shape=run.full_shape)
        if errors:
            raise ValueError("Constraint validation failed: " + "; ".join(errors[:10]))
        report.update(same_winding_collections=len(same), relative_winding_collections=len(relative),
                      same=same_report, relative=relative_report, validation_warnings=warnings,
                      parameters={k: v for k, v in vars(args).items() if not isinstance(v, Path)})
    out.mkdir(parents=True, exist_ok=True)
    axis_figure(out / "umbilicus_check.png", run, controls, plotting)
    for name, document in documents.items():
        write_json(out / name, document)
    write_json(out / "umbilicus_report.json", axis_report)
    write_json(out / "generation_report.json", report)
    tqdm.write(f"Saved {len(controls)} umbilicus controls, "
               f"{report.get('same_winding_collections', 0)} same and "
               f"{report.get('relative_winding_collections', 0)} relative collections to {out.resolve()}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--scale", type=float, nargs=3, required=True, metavar=("Z", "Y", "X"))
    parser.add_argument("--full-shape", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--umbilicus", type=Path)
    source.add_argument("--axis", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--only-umbilicus", action="store_true", help="Create only axis JSON, report and preview")
    parser.add_argument("--pitch-vox", type=float, help="Expected wrap spacing in INPUT XY voxels; required for windings")
    parser.add_argument("--z-step", type=int, default=8)
    parser.add_argument("--smooth-window", type=int, default=original_axis.MEDIAN_WINDOW)
    parser.add_argument("--min-pixels", type=int, default=original_axis.MIN_PIXELS)
    parser.add_argument("--max-jump", type=float, default=original_axis.MAX_JUMP)
    parser.add_argument("--batch-slices", type=int, default=8)
    parser.add_argument("--min-volume", type=int, help="Default: original MIN_VOL")
    parser.add_argument("--points-per-instance", type=int, help="Default: original K_MAX")
    parser.add_argument("--candidates-per-instance", type=int, help="Default: original CANDS_PER_TILE")
    parser.add_argument("--point-spacing", type=float, help="Default: original MIN_DIST")
    parser.add_argument("--max-instance-fraction", type=float, help="Default: original MEGA_FRAC; 1 disables this filter")
    parser.add_argument("--max-deep-frac", type=float, help="Default: original MAX_DEEP_FRAC")
    parser.add_argument("--rays", type=int, help="Default: original N_RAYS")
    parser.add_argument("--ray-step", type=float, default=1)
    parser.add_argument("--min-support", type=int, help="Default: original MIN_SUPPORT")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    try:
        return generate(args)
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
