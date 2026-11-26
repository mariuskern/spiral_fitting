"""Show winding constraints in 3D and XY, optionally over a browsable CT slice."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Constraint output directory")
    parser.add_argument("--out", type=Path, help="Save the initial view instead of opening a window")
    parser.add_argument("--z-range", type=float, nargs=2, metavar=("MIN", "MAX"), help="Inclusive original Z range")
    parser.add_argument("--z-exaggeration", type=float, default=1)
    parser.add_argument("--hide-labels", action="store_true")
    parser.add_argument("--ct", nargs="?", const="auto",
                        help="Optional CT overlay; omit --ct for constraints only. With --ct but no path, infer CT from the source pipeline config")
    parser.add_argument("--no-ct", action="store_true",
                        help="Disable all CT loading and path inference; overrides --ct and ignores CT options")
    parser.add_argument("--ct-dataset", help="Array path inside a CT Zarr group")
    parser.add_argument("--ct-scale", type=float, nargs=3, metavar=("Z", "Y", "X"),
                        help="CT voxel to original voxel factors (explicit CT default: 1 1 1)")
    parser.add_argument("--ct-origin", type=float, nargs=3, default=[0, 0, 0], metavar=("Z", "Y", "X"),
                        help="Original voxel coordinate of the first CT voxel, for a cropped CT")
    parser.add_argument("--slice", type=int, help="Initial slice index in the supplied CT array")
    parser.add_argument("--slice-thickness", type=float, help="Original Z thickness of the point overlay; default: one CT slice")
    parser.add_argument("--ct-window", type=float, nargs=2, metavar=("MIN", "MAX"), help="Fixed grayscale window")
    parser.add_argument("--ct-alpha", type=float, default=1)
    args = parser.parse_args(argv)
    if not np.isfinite(args.z_exaggeration) or args.z_exaggeration <= 0:
        parser.error("--z-exaggeration must be positive and finite")
    if args.z_range is not None and (not np.isfinite(args.z_range).all() or args.z_range[0] > args.z_range[1]):
        parser.error("--z-range requires finite MIN <= MAX")
    if args.no_ct:
        args.ct = None
        return args
    if not args.ct and (args.ct_dataset is not None or args.ct_scale is not None or args.slice is not None or
                        args.ct_window is not None or args.slice_thickness is not None or any(args.ct_origin)):
        parser.error("CT options require --ct [PATH]")
    if not np.isfinite(args.ct_alpha) or not 0 <= args.ct_alpha <= 1:
        parser.error("--ct-alpha must be in [0,1]")
    if args.ct_window is not None and (not np.isfinite(args.ct_window).all() or args.ct_window[0] >= args.ct_window[1]):
        parser.error("--ct-window requires finite MIN < MAX")
    return args


def clip_axis(points, z_range):
    if not len(points):
        return points
    points = points[np.argsort(points[:, 2])]
    if not np.isfinite(points).all() or np.any(np.diff(points[:, 2]) <= 0):
        raise ValueError("Umbilicus requires finite control points with distinct Z coordinates")
    if z_range is None:
        return points
    lo, hi = max(z_range[0], points[0, 2]), min(z_range[1], points[-1, 2])
    if lo > hi:
        return np.empty((0, 3))
    zs = np.unique(np.r_[lo, points[(points[:, 2] >= lo) & (points[:, 2] <= hi), 2], hi])
    return np.column_stack([np.interp(zs, points[:, 2], points[:, 0]),
                            np.interp(zs, points[:, 2], points[:, 1]), zs])


class CTSlices:
    """Lazy slice reads and conversion into the constraints' original grid."""

    def __init__(self, args):
        scale, dataset = args.ct_scale, args.ct_dataset
        if args.ct == "auto":
            report = read_json(args.directory / "generation_report.json")
            run = Path(report["source"])
            if not run.is_absolute():
                run = (Path.cwd() / run).resolve()
            config = read_json(run / "pipeline_config.json")
            path = config.get("volume") or config.get("ct")
            if not path:
                raise ValueError("Source config has no CT path; supply --ct PATH explicitly")
            if dataset is None:
                dataset = config.get("ct_dataset")
            if scale is None:
                scale = report["scale_zyx"]
        else:
            path = args.ct
        self.scale = np.asarray([1, 1, 1] if scale is None else scale, dtype=float)
        self.origin = np.asarray(args.ct_origin, dtype=float)
        if not np.isfinite(self.scale).all() or np.any(self.scale <= 0) or not np.isfinite(self.origin).all() or np.any(self.origin < 0):
            raise ValueError("CT scale must be positive and CT origin nonnegative, all finite")
        scripts_dir = str(Path(__file__).resolve().parents[1])
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from sheet_io import Volume, read_region
        self.volume = Volume(path, dataset)
        self._read_region = read_region
        if not all(self.volume.shape):
            raise ValueError("CT array must be nonempty")
        self.low, self.high = 0, self.volume.shape[0] - 1
        if args.z_range is not None:
            lo, hi = (np.asarray(args.z_range) - self.origin[0]) / self.scale[0]
            self.low, self.high = max(0, int(np.ceil(lo))), min(self.high, int(np.floor(hi)))
        if self.low > self.high:
            raise ValueError("--z-range does not intersect any CT slice")
        self.thickness = self.scale[0] if args.slice_thickness is None else args.slice_thickness
        if not np.isfinite(self.thickness) or self.thickness <= 0:
            raise ValueError("--slice-thickness must be positive and finite")
        if args.slice is not None and not self.low <= args.slice <= self.high:
            raise ValueError(f"--slice must be in [{self.low}, {self.high}]")
        _, y, x = self.origin
        _, dy, dx = self.scale
        self.extent = [x - dx / 2, x + (self.volume.shape[2] - .5) * dx,
                       y - dy / 2, y + (self.volume.shape[1] - .5) * dy]
        self.cached_index, self.cached_image = None, None

    def read(self, index):
        if index != self.cached_index:
            image = self._read_region(
                self.volume,
                (index, 0, 0),
                (1, *self.volume.shape[1:]),
            )[0]
            if image.shape != self.volume.shape[1:]:
                raise ValueError("CT slice shape differs from the volume shape")
            self.cached_index, self.cached_image = index, image
        return self.cached_image

    def z(self, index):
        return self.origin[0] + index * self.scale[0]


class SlicePanel:
    def __init__(self, fig, ax, grid, help_ax, source, groups, relatives, axis, args):
        from matplotlib.widgets import CheckButtons, Slider
        self.fig, self.ax, self.source = fig, ax, source
        self.groups, self.relatives, self.axis, self.args = groups, relatives, axis, args
        self.index = args.slice if args.slice is not None else source.low
        self.points = np.concatenate([p for p, _ in groups]) if groups else np.empty((0, 3))
        self.colors = np.concatenate([np.tile(c, (len(p), 1)) for p, c in groups]) if groups else np.empty((0, 4))
        image = source.read(self.index)
        values = image[np.isfinite(image) & (image != 0)]
        window = args.ct_window
        if window is None:
            window = np.percentile(values, [.5, 99.5]) if values.size else [0, 1]
            if window[1] <= window[0]:
                window = [window[0], window[0] + 1]
        self.image = ax.imshow(np.ma.masked_invalid(image), cmap="gray", origin="lower", extent=source.extent,
                               vmin=window[0], vmax=window[1], alpha=args.ct_alpha)
        self.scatter = ax.scatter([], [], s=12, alpha=.9, edgecolors="none")
        self.pair_lines = [ax.plot([], [], "r.-", linewidth=1.5, markersize=7)[0] for _ in relatives]
        self.axis_marker, = ax.plot([], [], "cx", markersize=12, markeredgewidth=2, label="Umbilicus at slice")
        self.annotations = []
        self.visible = dict(CT=True, Points=True, Pairs=True, Umbilicus=True)
        ui = grid.subgridspec(1, 2, width_ratios=[3, 1])
        slider_ax, checks_ax = fig.add_subplot(ui[0, 0]), fig.add_subplot(ui[0, 1])
        if source.low < source.high:
            self.slider = Slider(slider_ax, "CT Z index", source.low, source.high, valinit=self.index, valstep=1, valfmt="%d")
            self.slider.on_changed(self.update)
        else:
            self.slider = None
            slider_ax.axis("off")
            slider_ax.text(0, .5, f"Single CT slice: {source.low}")
        self.checks = CheckButtons(checks_ax, list(self.visible), [True] * len(self.visible))
        self.checks.on_clicked(self.toggle)
        help_ax.axis("off")
        help_ax.text(0, .9, "Right panel: CT slice + nearby constraint points\n"
                     "Slider / Left, Right / mouse wheel: one slice; PageUp, PageDown: ten\n"
                     "CT indices refer to the supplied array; axes show original voxel coordinates.\n"
                     f"Point slab thickness: {source.thickness:g} original Z voxels. X/Y orientation matches the 3D plot.",
                     va="top", fontsize=9, transform=help_ax.transAxes)
        self.connection_ids = [fig.canvas.mpl_connect("key_press_event", self.on_key),
                               fig.canvas.mpl_connect("scroll_event", self.on_scroll)]
        ax.set_xlim(source.extent[:2])
        ax.set_ylim(source.extent[2:])
        ax.legend(loc="upper right")
        self.update(self.index)

    def selected(self, points):
        return np.abs(points[:, 2] - self.source.z(self.index)) <= self.source.thickness / 2 + 1e-9

    def update(self, index):
        self.index = int(np.clip(round(index), self.source.low, self.source.high))
        self.image.set_data(np.ma.masked_invalid(self.source.read(self.index)))
        selected = self.selected(self.points)
        self.scatter.set_offsets(self.points[selected, :2])
        self.scatter.set_facecolors(self.colors[selected])
        self.scatter.set_visible(self.visible["Points"])
        for annotation in self.annotations:
            annotation.remove()
        self.annotations = []
        pair_count = 0
        for line, (points, records, name) in zip(self.pair_lines, self.relatives):
            show = bool(self.selected(points).all())
            line.set_data(points[:, 0], points[:, 1])
            line.set_visible(show and self.visible["Pairs"])
            if show:
                pair_count += 1
                if not self.args.hide_labels and self.visible["Pairs"]:
                    for record in records:
                        self.annotations.append(self.ax.annotate(f"{name}: {record.get('wind_a', '?')}",
                            record["p"][:2], xytext=(5, 5), textcoords="offset points", fontsize=7, color="red"))
        z = self.source.z(self.index)
        on_axis = len(self.axis) > 0 and self.axis[0, 2] <= z <= self.axis[-1, 2]
        if on_axis:
            self.axis_marker.set_data([np.interp(z, self.axis[:, 2], self.axis[:, 0])],
                                      [np.interp(z, self.axis[:, 2], self.axis[:, 1])])
        self.axis_marker.set_visible(on_axis and self.visible["Umbilicus"])
        self.image.set_visible(self.visible["CT"])
        self.ax.set_title(f"CT slice {self.index} | original Z={z:g}\n{selected.sum()} same-winding points; {pair_count} relative pairs")
        self.fig.canvas.draw_idle()

    def toggle(self, label):
        self.visible[label] = not self.visible[label]
        self.update(self.index)

    def step(self, delta):
        index = int(np.clip(self.index + delta, self.source.low, self.source.high))
        if index != self.index:
            if self.slider is not None:
                self.slider.set_val(index)
            else:
                self.update(index)

    def on_key(self, event):
        steps = {"right": 1, "left": -1, "up": 1, "down": -1, "pageup": 10, "pagedown": -10}
        if event.key in steps:
            self.step(steps[event.key])

    def on_scroll(self, event):
        if event.inaxes == self.ax:
            self.step(1 if event.button == "up" else -1)


def build_figure(args):
    import matplotlib.pyplot as plt
    same = read_json(args.directory / "same_windings.json")["collections"]
    relative = read_json(args.directory / "relative_windings.json")["collections"]
    controls = read_json(args.directory / "umbilicus.json")["control_points"]
    axis = clip_axis(np.asarray([[p["x"], p["y"], p["z"]] for p in controls], dtype=float).reshape(-1, 3), args.z_range)
    groups, relatives = [], []
    for i, group in enumerate(same.values()):
        points = np.asarray([p["p"] for p in group["points"].values()], dtype=float).reshape(-1, 3)
        if args.z_range is not None:
            points = points[(points[:, 2] >= args.z_range[0]) & (points[:, 2] <= args.z_range[1])]
        if len(points):
            groups.append((points, plt.get_cmap("turbo")((i + .5) / max(len(same), 1))))
    for group in relative.values():
        records = sorted(group["points"].values(), key=lambda p: p.get("wind_a", 0))
        points = np.asarray([p["p"] for p in records], dtype=float).reshape(-1, 3)
        if len(points) and (args.z_range is None or np.all((points[:, 2] >= args.z_range[0]) & (points[:, 2] <= args.z_range[1]))):
            relatives.append((points, records, group.get("name", "")))
    source = CTSlices(args) if args.ct else None
    fig = plt.figure(figsize=(15, 9) if source else (14, 7), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, .16]) if source else fig.add_gridspec(1, 2)
    spatial = fig.add_subplot(grid[0, 0], projection="3d")
    planar = fig.add_subplot(grid[0, 1])
    all_points = []
    for points, color in groups:
        spatial.scatter(*points.T, s=5, color=color, alpha=.65)
        all_points.extend(points)
        if source is None:
            planar.scatter(*points[:, :2].T, s=5, color=color, alpha=.65)
    for points, records, name in relatives:
        spatial.plot(*points.T, "r.-", linewidth=2, markersize=8)
        all_points.extend(points)
        if source is None:
            planar.plot(*points[:, :2].T, "r.-", linewidth=2, markersize=8)
            if not args.hide_labels:
                for record in records:
                    planar.annotate(f"{name}: {record.get('wind_a', '?')}", record["p"][:2],
                                    xytext=(5, 5), textcoords="offset points", fontsize=7)
    if len(axis):
        all_points.extend(axis)
        spatial.plot(*axis.T, "k--", linewidth=2, label="Umbilicus")
        spatial.legend()
        if source is None:
            planar.plot(*axis[:, :2].T, "kx--", label="Umbilicus")
            planar.legend()
    spatial.set(xlabel="X [original voxels]", ylabel="Y [original voxels]", zlabel="Z [original voxels]", title="3D constraint points")
    if all_points:
        spatial.set_box_aspect(np.maximum(np.ptp(all_points, axis=0), 1) * [1, 1, args.z_exaggeration])
    planar.set(xlabel="X [original voxels]", ylabel="Y [original voxels]", title="XY projection (all Z slices)")
    planar.set_aspect("equal", adjustable="box")
    panel = SlicePanel(fig, planar, grid[1, 1], fig.add_subplot(grid[1, 0]), source, groups, relatives, axis, args) if source else None
    fig.suptitle(f"{len(same)} same-winding groups; {len(relative)} relative groups\n"
                 "Colors identify groups, not absolute windings. Red lines: relative constraints.")
    fig._constraint_slice_panel = panel
    return fig, panel


def main(argv=None):
    args = parse_args(argv)
    import matplotlib
    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        fig, _ = build_figure(args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise SystemExit(f"Viewer error: {exc}") from exc
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=180)
        print(f"Saved {args.out.resolve()}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
