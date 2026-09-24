import argparse
from pathlib import Path

import tifffile
import zarr
import numpy as np
from numcodecs import Blosc


def _get_arguments():
    parser = argparse.ArgumentParser(
        description="Convert all downloaded Zarr segments into TIFF stacks while preserving the directory structure."
    )

    parser.add_argument(
        "input_path",
        type=Path,
        help="Input tiff",
    )

    parser.add_argument(
        "output_path",
        type=Path,
        help="Output zarr",
    )

    parser.add_argument(
        "--mul",
        default=1,
        type=int,
        help="Multiply all voxel values by this factor",
    )

    return parser.parse_args()


def main(args):
    with tifffile.TiffFile(args.input_path) as tif:
        shape = tif.series[0].shape
        # dtype = tif.series[0].dtype

        chunks = (192, 192, 192)

        compressor = Blosc(
            cname="zstd",
            clevel=1,
            shuffle=Blosc.BITSHUFFLE,
        )

        z = zarr.open(
            args.output_path,
            mode="w",
            shape=shape,
            dtype=np.uint8,
            chunks=chunks,
            compressor=compressor,
            zarr_format=2,
            dimension_separator="/"
        )

        for i, page in enumerate(tif.pages):
            z[i] = page.asarray().astype(np.uint8) * args.mul

        print("dtype:", z.dtype)
        print("max:", z[:].max())
        print("min:", z[:].min())

    print("Done")


if __name__ == "__main__":
    main(_get_arguments())