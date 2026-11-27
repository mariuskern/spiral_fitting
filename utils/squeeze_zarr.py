"""Select a channel from a C Z Y X prediction and write a standalone Z Y X array."""

import argparse
from copy import deepcopy
from pathlib import Path

if __package__:
    from ._zarr_transform import copy_blocks, create_output, open_array
else:
    from _zarr_transform import copy_blocks, create_output, open_array


def squeeze(source, destination, array_path=None, channel=0, overwrite=False):
    array = open_array(source, array_path)
    if array.ndim not in (3, 4):
        raise ValueError(f"Expected a 3D Z Y X or 4D C Z Y X array, got shape {array.shape}.")
    if channel < 0 or (array.ndim == 4 and channel >= array.shape[0]):
        raise ValueError("--channel is outside the input channel range.")
    if array.ndim == 3 and channel != 0:
        raise ValueError("A 3D array has no channel axis; omit --channel.")

    shape = array.shape[-3:]
    chunks = array.chunks[-3:]
    attrs = deepcopy(dict(array.attrs))
    names = getattr(array.metadata, "dimension_names", None)
    if array.ndim == 4:
        if names is not None:
            names = names[1:]
        for key in ("_ARRAY_DIMENSIONS", "axes"):
            if isinstance(attrs.get(key), list) and len(attrs[key]) == 4:
                attrs[key] = attrs[key][1:]
        for transform in attrs.get("coordinateTransformations", []):
            kind = transform.get("type")
            if kind in ("scale", "translation") and len(transform.get(kind, [])) == 4:
                transform[kind] = transform[kind][1:]

    print(f"Input: shape={array.shape}, dtype={array.dtype}, chunks={array.chunks}")
    output = create_output(source, destination, array, shape, chunks, array.dtype, attrs, names, overwrite)
    copy_blocks(output, lambda block: array[(channel,) + block] if array.ndim == 4 else array[block])
    print(f"Done: {destination} | shape={output.shape}, dtype={output.dtype}")


def main():
    parser = argparse.ArgumentParser(description="Select a channel from C Z Y X to produce a 3D Z Y X Zarr. Copies 3D input unchanged.")
    parser.add_argument("source", help="Local Zarr path or URL.")
    parser.add_argument("--out", required=True, type=Path, help="Output standalone 3D Zarr array.")
    parser.add_argument("--array", help="Explicit array path for group inputs, e.g. 0 or group/2.")
    parser.add_argument("--channel", type=int, default=0, help="Channel on the first axis of 4D input (default: 0).")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output.")
    args = parser.parse_args()
    try:
        squeeze(args.source, args.out, args.array, args.channel, args.overwrite)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
