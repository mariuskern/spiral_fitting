#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import dask.array as da
import numpy as np
from scipy import ndimage
from dask_image import ndmeasure
import zarr


def choose_random_seeds(labels, rng):
    """
    Iterate over the Dask label array chunk-by-chunk and choose
    one random voxel for every connected component.

    Returns:
        dict:
            label -> (z, y, x)
    """

    # Number of labels is expected to be known at this point.
    # We process one chunk at a time, so the complete label array
    # is never loaded into RAM.
    seeds = {}

    # Iterate over delayed chunks.
    delayed_chunks = labels.to_delayed()

    # Flatten the chunk grid.
    delayed_chunks = delayed_chunks.ravel()

    # Keep track of global chunk coordinates.
    chunk_locations = list(np.ndindex(labels.numblocks))

    for i, delayed_chunk in enumerate(delayed_chunks):
        print(f"Processing label chunk {i + 1}/{len(delayed_chunks)}...")

        chunk = delayed_chunk.compute()

        if chunk.size == 0:
            continue

        # Position of this chunk in the global volume.
        chunk_index = chunk_locations[i]

        # Starting coordinate of this chunk.
        starts = [
            sum(labels.chunks[axis][j] for j in range(chunk_index[axis]))
            for axis in range(3)
        ]

        # Find labels contained in this chunk.
        unique_labels = np.unique(chunk)

        # Ignore background.
        unique_labels = unique_labels[unique_labels != 0]

        for label in unique_labels:

            label = int(label)

            mask = chunk == label

            count = int(mask.sum())

            if count == 0:
                continue

            # Random voxel within this component/chunk.
            flat_indices = np.flatnonzero(mask)

            random_index = rng.integers(0, len(flat_indices))
            flat_index = flat_indices[random_index]

            local_coord = np.unravel_index(flat_index, chunk.shape)

            global_coord = tuple(
                int(starts[axis] + local_coord[axis])
                for axis in range(3)
            )

            # If this component has already been seen in another chunk,
            # use reservoir sampling so that the final seed is uniformly
            # random among all voxels belonging to the component.
            if label not in seeds:
                seeds[label] = {
                    "count": count,
                    "seed": global_coord,
                }
            else:
                previous_count = seeds[label]["count"]

                # Reservoir sampling:
                # replace previous seed with probability
                # count / (previous_count + count)
                probability = count / (previous_count + count)

                if rng.random() < probability:
                    seeds[label]["seed"] = global_coord

                seeds[label]["count"] = previous_count + count

    return seeds


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find connected components in a large Zarr volume "
            "and choose one random seed point per component."
        )
    )

    parser.add_argument(
        "zarr_path",
        help="Path to the Zarr store",
    )

    parser.add_argument(
        "--array",
        default=None,
        help="Zarr array/component to process (default: 0)",
    )

    parser.add_argument(
        "--output",
        default="seeds.json",
        help="Output JSON file (default: seeds.json)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Threshold for foreground. "
            "Only voxels > threshold are considered foreground."
        ),
    )

    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help=(
            "3D connectivity: "
            "1 = 6-neighbour, "
            "2 = 18-neighbour, "
            "3 = 26-neighbour. "
            "Default: 1"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--chunks",
        type=int,
        nargs=3,
        default=None,
        metavar=("Z", "Y", "X"),
        help=(
            "Optional Dask chunk size, e.g. "
            "--chunks 128 512 512"
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load Zarr lazily
    # ------------------------------------------------------------------

    print("Opening Zarr...")

    z = zarr.open(
        args.zarr_path,
        mode="r",
    )

    if args.array is not None:
        z = z[args.array]

    print(f"Zarr shape:  {z.shape}")
    print(f"Zarr chunks: {z.chunks}")
    print(f"Zarr dtype:  {z.dtype}")

    volume = da.from_array(
        z,
        chunks=args.chunks or z.chunks,
    )

    print(f"Shape:  {volume.shape}")
    print(f"Dtype:  {volume.dtype}")
    print(f"Chunks: {volume.chunks}")

    # ------------------------------------------------------------------
    # Create binary foreground mask
    # ------------------------------------------------------------------

    if args.threshold is None:
        print("Foreground: all non-zero voxels")
        mask = volume != 0
    else:
        print(f"Foreground: values > {args.threshold}")
        mask = volume > args.threshold

    # ------------------------------------------------------------------
    # Connectivity
    # ------------------------------------------------------------------

    structure = ndimage.generate_binary_structure(
        rank=3,
        connectivity=args.connectivity,
    )

    print(
        f"Connectivity: {args.connectivity}"
    )

    # ------------------------------------------------------------------
    # Connected components
    # ------------------------------------------------------------------

    print("Finding connected components...")

    labels, num_components = ndmeasure.label(
        mask,
        structure=structure,
    )

    # Computing this only produces the number of components,
    # not the complete label volume.
    num_components = int(num_components.compute())

    print(f"Found {num_components} connected components.")

    if num_components == 0:
        print("No foreground components found.")

        output = {
            "zarr_path": str(args.zarr_path),
            "array": args.array,
            "shape": list(volume.shape),
            "chunks": [list(c) for c in volume.chunks],
            "connectivity": args.connectivity,
            "threshold": args.threshold,
            "num_components": 0,
            "seeds": [],
        }

        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)

        return

    # ------------------------------------------------------------------
    # Choose random seeds
    # ------------------------------------------------------------------

    print("Choosing random seed for every component...")

    rng = np.random.default_rng(args.seed)

    seeds = choose_random_seeds(
        labels,
        rng,
    )

    # ------------------------------------------------------------------
    # Create output
    # ------------------------------------------------------------------

    components = []

    for label in sorted(seeds):

        component = seeds[label]

        components.append(
            {
                "component": int(label),
                "size": int(component["count"]),
                "seed": [
                    int(component["seed"][0]),
                    int(component["seed"][1]),
                    int(component["seed"][2]),
                ],
            }
        )

    output = {
        "zarr_path": str(args.zarr_path),
        "array": args.array,
        "shape": list(volume.shape),
        "chunks": [list(c) for c in volume.chunks],
        "connectivity": args.connectivity,
        "threshold": args.threshold,
        "random_seed": args.seed,
        "num_components": len(components),
        "seeds": components,
    }

    output_path = Path(args.output)

    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Found components: {len(components)}")
    print(f"Seeds written to: {output_path}")


if __name__ == "__main__":
    main()