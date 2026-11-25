"""Portable sheet pipeline: local volume I/O and atomic block checkpoints."""

import json
import re
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import tifffile
import zarr


def natural_key(path):
    return [int(p) if p.isdigit() else p.lower()
            for p in re.split(r"(\d+)", path.name)]


class Volume:
    """Read only requested TIFF slices or Zarr chunks; reject ambiguous axes."""

    def __init__(self, path, dataset=None):
        self.path = Path(path)
        self.files = None
        self.array = None
        if self.path.is_dir():
            files = sorted((p for p in self.path.iterdir()
                            if p.suffix.lower() in (".tif", ".tiff")), key=natural_key)
            if files:
                if dataset is not None:
                    raise ValueError("--*-dataset applies only to Zarr inputs")
                self.files = files
                with tifffile.TiffFile(files[0]) as tif:
                    shape = tif.series[0].shape
                if len(shape) != 2:
                    raise ValueError("A TIFF directory must contain one grayscale 2D slice per file")
                self.shape = (len(files), *shape)
                return
        if self.path.suffix.lower() in (".tif", ".tiff"):
            if dataset is not None:
                raise ValueError("--*-dataset applies only to Zarr inputs")
            with tifffile.TiffFile(self.path) as tif:
                series = tif.series[0]
                self.shape = series.shape
                if len(self.shape) != 3 or series.axes[-2:] != "YX" or series.axes[0] not in "ZQI":
                    raise ValueError(f"Expected grayscale ZYX TIFF stack, got {series.axes} {self.shape}")
            return
        node = zarr.open(str(self.path), mode="r")
        if dataset is not None:
            if isinstance(node, zarr.Array):
                raise ValueError(
                    f"Zarr path '{self.path}' already points to an array, not a group. "
                    "Remove --ct-dataset / --pred-dataset, or point to the parent Zarr group."
                )
            node = node[dataset]
        if not isinstance(node, zarr.Array):
            raise ValueError("Zarr input is a group. Select its array with --pred-dataset or --ct-dataset (e.g. 0)")
        self.array = node
        self.shape = node.shape
        if len(self.shape) != 3:
            raise ValueError(f"Expected 3D ZYX Zarr array, got {self.shape}")


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_npz(path, **arrays):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def read_region(source, origin, shape):
    """Read a bounded ZYX block, without reading a full Z slab into RAM."""
    slices = tuple(slice(o, o + n) for o, n in zip(origin, shape))
    if source.array is not None:
        return np.asarray(source.array[slices])
    if source.files is not None:
        images = []
        for path in source.files[slices[0]]:
            image = tifffile.imread(path)
            if image.shape != source.shape[1:]:
                raise ValueError(f"Inconsistent TIFF slice shape: {path}")
            images.append(image[slices[1:]].copy())
        return np.stack(images)
    with tifffile.TiffFile(source.path) as tif:
        store = tif.series[0].aszarr()
        try:
            return np.asarray(zarr.open(store, mode="r")[slices])
        finally:
            store.close()


def load_checkpoint(out, relative, origin, shape, index):
    """A block JSON is the commit record, written after labels and previews."""
    try:
        stats = json.loads((out / relative).with_suffix(".json").read_text(encoding="utf-8"))
        block = stats["checkpoint"]
        if (block["index"] != index or block["origin_zyx"] != list(origin)
                or block["shape_zyx"] != list(shape) or block["block"] != relative.as_posix()):
            return None
        with np.load(out / relative, allow_pickle=False) as data:
            labels = data["labels"]
            if not np.array_equal(data["origin"], origin):
                return None
        if labels.shape != shape or labels.dtype != np.int32 or np.any(labels < 0):
            return None
        ids = np.unique(labels)
        if ids[ids > 0].tolist() != block["ids"]:
            return None
        return labels, stats, block
    except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
        return None
