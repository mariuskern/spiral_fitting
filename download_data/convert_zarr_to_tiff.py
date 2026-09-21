import argparse
import shutil
from pathlib import Path

import numpy as np
import tifffile
import zarr
from tqdm import tqdm


def _get_arguments():
    parser = argparse.ArgumentParser(
        description="Convert all downloaded Zarr segments into TIFF stacks while preserving the directory structure."
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Root directory containing the downloaded segments.",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Root directory where the converted data will be stored.",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for generated TIFF files (default: none).",
    )

    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Number of digits for zero-padding (default: 3).",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing TIFF files and copied files.",
    )

    parser.add_argument(
        "--array",
        type=str,
        default="0",
        help="Group to convert",
    )

    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="Disable ZSTD compression.",
    )

    return parser.parse_args()


def convert_zarr(input_dir, output_dir, array, prefix, digits, overwrite, compression):
    # --------------------------------------------------------
    # Create output segment directory
    # --------------------------------------------------------

    output_dir.mkdir(parents=True, exist_ok=True)


    # --------------------------------------------------------
    # Open Zarr
    # --------------------------------------------------------

    print("Opening Zarr...")
    z = zarr.open(input_dir, mode="r")
    print("Zarr structure:")
    print(z.tree())
    print("")


    # --------------------------------------------------------
    # Find volume
    # --------------------------------------------------------

    if array not in z:
        available = list(z.keys())

        raise ValueError(
            f"No {array} array found in:\n"
            f"{input_dir}\n\n"
            f"Available arrays/groups:\n"
            f"{available}"
        )

    volume = z[array]


    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3D volume with shape (Z, Y, X), but got:\n"
            f"{volume.shape}"
        )

    d, _, _ = volume.shape

    print(f"Volume shape:     {volume.shape}")
    print(f"Dtype:            {volume.dtype}")
    print(f"Chunks:           {volume.chunks}")
    print(f"Number of slices: {d}")
    print("")


    # --------------------------------------------------------
    # Create layers directory
    # --------------------------------------------------------

    layers_dir = output_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)
    print(f"Layers directory:\n" f"  {layers_dir}")
    print("")
    

    # --------------------------------------------------------
    # Convert slices
    # --------------------------------------------------------

    tif = tifffile.TiffWriter(output_dir / f"{output_dir.name}.tif", bigtiff=True)

    for i in tqdm(range(d), desc=f"Converting", unit="slice"):
        output_file = layers_dir / f"{prefix}{i:0{digits}d}.tif"

        if output_file.exists() and not overwrite:
            print(f"Skipping: {output_file}")
            continue

        # image = np.asarray(volume[i, :, :])

        try:
            image = np.asarray(volume[i, :, :])
        except Exception as e:
            print("")
            print("=" * 70)
            print("ZARR READ ERROR")
            print("=" * 70)
            print(f"Slice:   {i}")
            print(f"Shape:   {volume.shape}")
            print(f"Chunks:  {volume.chunks}")
            print(f"Dtype:   {volume.dtype}")
            print(f"Error:   {type(e).__name__}: {e}")
            print("=" * 70)
            raise

        tifffile.imwrite(output_file, image, compression=compression, bigtiff=True)
        tif.write(image)

    tif.close()

    print("")
    print(f"Finished")


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found:\n" f"{input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory:\n" f"{input_dir}")

    # Prevent output directory inside input directory
    input_dir_resolved = input_dir.resolve()
    output_dir_resolved = output_dir.resolve()
    if output_dir_resolved == input_dir_resolved or input_dir_resolved in output_dir_resolved.parents:
        raise ValueError(
            "The output directory must not be inside the input directory.\n\n"
            f"Input:  {input_dir_resolved}\n"
            f"Output: {output_dir_resolved}"
        )

    try:
        convert_zarr(
            input_dir=input_dir,
            output_dir=output_dir,
            array=args.array,
            prefix=args.prefix,
            digits=args.digits,
            overwrite=args.overwrite,
            compression="zstd" if not args.no_compression else None
        )
    except Exception as e:
        print("")
        print(f"ERROR")

        print(f"  {e}")
        raise

    # ========================================================
    # Summary
    # ========================================================

    print("Conversion completed")
    print("")


if __name__ == "__main__":
    main(_get_arguments())
