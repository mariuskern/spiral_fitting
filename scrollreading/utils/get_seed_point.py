import argparse
from pathlib import Path

import zarr

input_path = "volume.tif"
output_path = "volume.zarr"


def _get_arguments():
    parser = argparse.ArgumentParser(
        description="Convert all downloaded Zarr segments into TIFF stacks while preserving the directory structure."
    )

    parser.add_argument(
        "zarr",
        type=Path,
        help="Input zarr",
    )

    parser.add_argument(
        "x",
        type=int,
        help="x",
    )

    parser.add_argument(
        "y",
        type=int,
        help="y",
    )

    parser.add_argument(
        "z",
        type=int,
        help="z",
    )

    return parser.parse_args()


def main(args):
    z = zarr.open(
        args.zarr,
        mode="r",
        zarr_format=2,
        dimension_separator="/"
    )

    print("shape :", z.shape)
    print("chunks:", z.chunks)
    print("dtype  :", z.dtype)
    print("value  :", z[args.z, args.y, args.x])


if __name__ == "__main__":
    main(_get_arguments())
