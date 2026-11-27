"""Shared array selection and chunked output for the Zarr conversion scripts."""

import itertools
import math
from pathlib import Path

import zarr

if __package__:
    from .inspect_zarr import MetadataReader, normalize_url
else:
    from inspect_zarr import MetadataReader, normalize_url


def open_array(source, array_path=None):
    node = zarr.open(normalize_url(source), mode="r")
    if isinstance(node, zarr.Group):
        if array_path is None:
            raise ValueError(
                "Input is a Zarr group. Choose an array with --array PATH "
                "(e.g. --array 0 or --array group/2). Use inspect_zarr.py to list paths."
            )
        parts = [part for part in array_path.strip("/").split("/") if part not in ("", ".")]
        if not parts or ".." in parts or "\\" in array_path:
            raise ValueError("--array must name an array inside the input group.")
        path = "/".join(parts)
        try:
            node = node[path]
        except KeyError as error:
            raise ValueError(f"Array {path!r} was not found in the input group.") from error
        if isinstance(node, zarr.Group):
            raise ValueError(f"{path!r} is a group. Select an array inside it, e.g. --array {path}/0.")
    elif array_path not in (None, "", ".", "/"):
        raise ValueError("Input is already an array; omit --array.")
    return node


def create_output(source, destination, array, shape, chunks, dtype, attrs, dimension_names, overwrite):
    destination = Path(destination).expanduser().resolve()
    reader = MetadataReader(source)
    protocols = reader.fs.protocol
    if isinstance(protocols, str):
        protocols = (protocols,)
    if "file" in protocols or "local" in protocols:
        input_path = Path(reader.base).resolve()
        if destination == input_path or destination in input_path.parents or input_path in destination.parents:
            raise ValueError("Input and output paths must be separate, non-overlapping directories.")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}. Use --overwrite to replace it.")

    version = array.metadata.zarr_format
    kwargs = {"dimension_names": dimension_names} if version == 3 and dimension_names is not None else {}
    output = zarr.open_array(
        str(destination), mode="w" if overwrite else "w-", zarr_format=version,
        shape=shape, chunks=chunks, dtype=dtype, **kwargs,
    )
    output.attrs.update(attrs)
    return output


def copy_blocks(destination, read_block):
    """Write one output chunk at a time, keeping memory independent of volume size."""
    counts = [(size + chunk - 1) // chunk for size, chunk in zip(destination.shape, destination.chunks)]
    total = math.prod(counts)
    last_percentage = -5
    for completed, indices in enumerate(itertools.product(*(range(count) for count in counts)), start=1):
        block = tuple(slice(index * chunk, min((index + 1) * chunk, size))
                      for index, chunk, size in zip(indices, destination.chunks, destination.shape))
        destination[block] = read_block(block)
        percentage = 100 * completed / total
        if percentage >= last_percentage + 5 or completed == total:
            print(f"Progress: {percentage:6.2f}% | chunks: {completed}/{total}", flush=True)
            last_percentage = percentage
