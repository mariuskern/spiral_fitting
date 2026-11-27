"""Threshold a selected Zarr array to a standalone uint8 array containing 0 and 1."""

import argparse
import math
from pathlib import Path

import numpy as np

if __package__:
    from ._zarr_transform import copy_blocks, create_output, open_array
else:
    from _zarr_transform import copy_blocks, create_output, open_array


def threshold(source, destination, value, array_path=None, overwrite=False):
    if not math.isfinite(value):
        raise ValueError("--threshold must be a finite number.")
    array = open_array(source, array_path)
    names = getattr(array.metadata, "dimension_names", None)
    print(f"Input: shape={array.shape}, dtype={array.dtype}, chunks={array.chunks}")
    output = create_output(source, destination, array, array.shape, array.chunks, np.uint8,
                           dict(array.attrs), names, overwrite)
    copy_blocks(output, lambda block: (array[block] > value).astype(np.uint8))
    print(f"Done: {destination} | shape={output.shape}, dtype=uint8 | values: 0/1")


def main():
    parser = argparse.ArgumentParser(description="Write uint8 values 1 where input > threshold, otherwise 0. Preserves array shape.")
    parser.add_argument("source", help="Local Zarr path or URL.")
    parser.add_argument("--out", required=True, type=Path, help="Output standalone binary Zarr array.")
    parser.add_argument("--array", help="Explicit array path for group inputs, e.g. 0 or group/2.")
    parser.add_argument("--threshold", required=True, type=float, help="Threshold in the input's value range, e.g. 127 or 0.5.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output.")
    args = parser.parse_args()
    try:
        threshold(args.source, args.out, args.threshold, args.array, args.overwrite)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
