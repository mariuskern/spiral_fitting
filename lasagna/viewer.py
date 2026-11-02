"""Interactive CT/Presence slice viewer for local or HTTP Lasagna manifests."""

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen

import numpy as np


ROOT = Path(__file__).resolve().parent


def open_array(location):
    import zarr

    if location.startswith(("http://", "https://")):
        from zarr.storage import FsspecStore

        return zarr.open_array(FsspecStore.from_url(location), mode="r")
    return zarr.open_array(location, mode="r")


def load_manifest(location):
    if location.startswith(("http://", "https://")):
        with urlopen(location, timeout=30) as response:
            manifest = json.load(response)
        presence_path = urljoin(location, manifest["groups"]["presence"]["zarr"])
    else:
        path = Path(location).expanduser().resolve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        presence_path = str(path.parent / manifest["groups"]["presence"]["zarr"])
    scale = float(manifest["source_to_base"]) * 2 ** int(
        manifest["groups"]["presence"]["scaledown"]
    )
    if scale <= 0:
        raise ValueError("Manifest prediction scale must be positive")
    return open_array(presence_path), scale, float(manifest["source_to_base"])


class Slices:
    def __init__(self, presence, ct, prediction_scale, ct_scale):
        if presence.ndim != 3 or (ct is not None and ct.ndim != 3):
            raise ValueError("Presence and CT must be 3D ZYX arrays")
        self.presence = presence
        self.ct = ct
        self.ratio = prediction_scale / ct_scale

    @lru_cache(maxsize=6)
    def read(self, axis, index):
        selection = [slice(None)] * 3
        selection[axis] = index
        presence = np.asarray(self.presence[tuple(selection)])
        if self.ct is None:
            return np.zeros(presence.shape), presence
        fixed = round(index * self.ratio)
        if not 0 <= fixed < self.ct.shape[axis]:
            raise ValueError("Slice is outside CT extent; check --ct-scale and input paths")
        coordinates = []
        for dim in range(3):
            if dim == axis:
                coordinates.append(fixed)
            else:
                coords = np.rint(np.arange(self.presence.shape[dim]) * self.ratio).astype(int)
                if coords[-1] >= self.ct.shape[dim]:
                    raise ValueError("Prediction exceeds CT extent; check --ct-scale")
                coordinates.append(coords)
        return np.asarray(self.ct.oindex[tuple(coordinates)]), presence


def show(slices, threshold=32):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons, Slider

    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharex=True, sharey=True)
    fig.subplots_adjust(bottom=0.34, top=0.88)
    axis = 0
    initial = slices.presence.shape[axis] // 2
    ct, presence = slices.read(axis, initial)
    low, high = np.percentile(ct, [1, 99])
    if high <= low:
        high = low + 1
    background = axes[0].imshow(ct, cmap="gray", vmin=low, vmax=high)
    display_max = max(1, int(presence.max()))
    heatmap = axes[1].imshow(presence, cmap="inferno", vmin=0, vmax=display_max)
    overlay_ct = axes[2].imshow(ct, cmap="gray", vmin=low, vmax=high)
    overlay = axes[2].imshow(np.ma.masked_less(presence, threshold),
                             cmap="autumn", vmin=0, vmax=255, alpha=0.65)
    for ax, title in zip(axes, ("CT", "Presence (display contrast)", "CT + Presence")):
        ax.set_title(title)
    fig.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)
    position = Slider(fig.add_axes([0.22, 0.24, 0.65, 0.03]), "Slice", 0,
                      max(1, slices.presence.shape[axis] - 1), valinit=initial, valstep=1)
    cutoff = Slider(fig.add_axes([0.22, 0.18, 0.65, 0.03]), "Threshold", 0, 255,
                    valinit=threshold, valstep=1)
    opacity = Slider(fig.add_axes([0.22, 0.12, 0.65, 0.03]), "Opacity", 0, 1, valinit=0.65)
    contrast = Slider(fig.add_axes([0.22, 0.06, 0.65, 0.03]), "Color max", 1, 255,
                      valinit=display_max, valstep=1)
    planes = RadioButtons(fig.add_axes([0.02, 0.05, 0.10, 0.16]), ("XY", "XZ", "YZ"))
    status = fig.suptitle("")

    def update(_=None):
        try:
            ct, presence = slices.read(axis, min(int(position.val), slices.presence.shape[axis] - 1))
            background.set_data(ct)
            overlay_ct.set_data(ct)
            heatmap.set_data(presence)
            heatmap.set_clim(0, contrast.val)
            overlay.set_data(np.ma.masked_less(presence, cutoff.val))
            overlay.set_alpha(opacity.val)
            height, width = presence.shape
            for artist in (background, overlay_ct, heatmap, overlay):
                artist.set_extent((-0.5, width - 0.5, height - 0.5, -0.5))
            status.set_text(f"{('Z', 'Y', 'X')[axis]}={int(position.val)} | "
                            f"Presence max={presence.max()} | "
                            f"Above threshold: {np.count_nonzero(presence >= cutoff.val):,} pixels")
        except Exception as exc:
            status.set_text(f"Could not load slice: {exc}")
        fig.canvas.draw_idle()

    def change_plane(label):
        nonlocal axis
        axis = ("XY", "XZ", "YZ").index(label)
        position.valmax = max(1, slices.presence.shape[axis] - 1)
        position.ax.set_xlim(0, position.valmax)
        position.set_val(slices.presence.shape[axis] // 2)
        height, width = slices.read(axis, int(position.val))[1].shape
        axes[0].set_xlim(-0.5, width - 0.5)
        axes[0].set_ylim(height - 0.5, -0.5)
        update()

    position.on_changed(update)
    cutoff.on_changed(update)
    opacity.on_changed(update)
    contrast.on_changed(update)
    planes.on_clicked(change_plane)
    update()
    plt.show()


def main(argv=None):
    from dotenv import load_dotenv

    load_dotenv(ROOT.parent / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", default="http://localhost:8000/fiber.lasagna.json")
    parser.add_argument("--ct", default=os.getenv("LASAGNA_INPUT"), help="Local CT Zarr array; defaults to LASAGNA_INPUT")
    parser.add_argument("--ct-scale", type=float, help="CT voxel size in base voxels; defaults to manifest source_to_base")
    parser.add_argument("--threshold", type=int, default=32)
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 255:
        parser.error("--threshold must be between 0 and 255")
    if args.ct_scale is not None and args.ct_scale <= 0:
        parser.error("--ct-scale must be positive")
    presence, scale, source_scale = load_manifest(args.manifest)
    ct_path = args.ct
    if ct_path and not ct_path.startswith(("http://", "https://")):
        ct_path = str(ROOT / Path(os.path.expandvars(ct_path)).expanduser())
    ct = open_array(ct_path) if ct_path else None
    show(Slices(presence, ct, scale, args.ct_scale or source_scale), args.threshold)


if __name__ == "__main__":
    main()
