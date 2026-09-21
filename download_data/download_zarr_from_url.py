import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr


def parse_region(values):
    """
    Convert:
        x_start x_stop y_start y_stop z_start z_stop

    into:
        [(x_start, x_stop), (y_start, y_stop), (z_start, z_stop)]
    """

    if len(values) % 2 != 0:
        raise ValueError("Region must contain start/stop pairs.")

    return [(values[i], values[i + 1]) for i in range(0, len(values), 2)]


def validate_region(region, shape):
    if len(region) != len(shape):
        raise ValueError(f"Region has {len(region)} dimensions, but array has {len(shape)} dimensions.")

    for dim, ((start, stop), size) in enumerate(zip(region, shape)):
        if start < 0:
            raise ValueError(f"Dimension {dim}: start must be >= 0.")

        if stop <= start:
            raise ValueError(f"Dimension {dim}: stop must be greater than start.")

        if stop > size:
            raise ValueError(f"Dimension {dim}: {start}:{stop} exceeds dimension size {size}.")


def iterate_slices(shape, chunks):
    """
    Iterate over the output array chunk by chunk.

    Returns slices such as:

        (slice(0, 64), slice(0, 64), slice(0, 64))
    """

    import itertools

    chunk_counts = [(size + chunk - 1) // chunk for size, chunk in zip(shape, chunks)]

    for chunk_index in itertools.product(*[range(n) for n in chunk_counts]):
        slices = []

        for dim, index in enumerate(chunk_index):
            start = index * chunks[dim]
            stop = min(start + chunks[dim], shape[dim])

            slices.append(slice(start, stop))

        yield tuple(slices)


def slice_to_list(slices):
    """Convert tuple of slices into JSON serializable data."""

    return [[s.start, s.stop] for s in slices]


def main():
    parser = argparse.ArgumentParser(
        description="Download a region of a remote Zarr and store it as a new local Zarr starting at index 0."
    )

    parser.add_argument(
        "url",
        help="URL of the remote Zarr.",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Output Zarr directory.",
    )

    parser.add_argument(
        "--region",
        nargs="+",
        type=int,
        required=True,
        metavar="N",
        help="Region as start/stop pairs. For a 3D array: z_start z_stop y_start y_stop x_start x_stop",
    )

    parser.add_argument(
        "--array",
        type=str,
        default="0",
        help="Path of the array inside the Zarr, e.g. volume or group/volume.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Parse region
    # ------------------------------------------------------------

    region = parse_region(args.region)

    print("Remote Zarr:")
    print(args.url)

    print()
    print("Array:")
    print(args.array)

    # ------------------------------------------------------------
    # Open remote Zarr
    # ------------------------------------------------------------

    print()
    print("Opening remote Zarr...")

    root = zarr.open(
        args.url.rstrip("/") + "/",
        mode="r",
    )

    source = root[args.array]

    print()
    print("Source array:")
    print(f"  shape : {source.shape}")
    print(f"  chunks: {source.chunks}")
    print(f"  dtype : {source.dtype}")

    # ------------------------------------------------------------
    # Validate region
    # ------------------------------------------------------------

    validate_region(region, source.shape,)

    # ------------------------------------------------------------
    # Calculate local shape
    # ------------------------------------------------------------

    local_shape = tuple(stop - start for start, stop in region)

    # Use the source chunk size where possible.
    #
    # If the requested region is smaller than one source chunk,
    # the final chunk will naturally be smaller.
    local_chunks = tuple(min(chunk, size) for chunk, size in zip(source.chunks, local_shape))

    print()
    print("Requested region:")
    print(f"  {region}")

    print()
    print("Local array:")
    print(f"  shape : {local_shape}")
    print(f"  chunks: {local_chunks}")

    # ------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------

    args.output.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Create local Zarr
    # ------------------------------------------------------------

    output_zarr = args.output / "data.zarr"

    print()
    print(f"Creating: {output_zarr}")

    if output_zarr.exists():
        raise FileExistsError(
            f"Output already exists: {output_zarr}\n"
            "Delete it first or choose another output directory."
        )

    # destination = zarr.open(
    #     output_zarr,
    #     mode="w",
    #     shape=local_shape,
    #     chunks=local_chunks,
    #     dtype=source.dtype,
    # )

    z = zarr.open(
        output_zarr,
        mode="w"
    )

    z.create_array(
        args.array,
        shape=local_shape,
        chunks=local_chunks,
        dtype=source.dtype,
    )

    destination = z[args.array]

    # ------------------------------------------------------------
    # Copy attributes
    # ------------------------------------------------------------

    try:
        for key, value in source.attrs.items():
            destination.attrs[key] = value
    except Exception:
        print("Warning: Could not copy all source attributes.")

    # ------------------------------------------------------------
    # Download data chunk by chunk
    # ------------------------------------------------------------

    total_elements = int(np.prod(local_shape))

    processed_elements = 0

    print()
    print("Downloading region...")

    for local_slices in iterate_slices(local_shape, local_chunks):
        # Translate local coordinates to source coordinates.
        source_slices = tuple(slice(region[dim][0] + local_slices[dim].start, region[dim][0] + local_slices[dim].stop) for dim in range(len(region)))

        # Read only this part from the remote Zarr.
        data = source[source_slices]

        # Write it to the corresponding local position.
        destination[local_slices] = data

        chunk_elements = int(np.prod(data.shape))

        processed_elements += chunk_elements

        percentage = 100.0 * processed_elements / total_elements

        print(f"\rProgress: {percentage:6.2f}%", end="", flush=True)

    print()
    print("Download complete.")

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------

    metadata = {
        "source": {
            "url": args.url,
            "array": args.array,
        },

        "download": {
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        },

        "source_array": {
            "shape": list(source.shape),
            "chunks": list(source.chunks),
            "dtype": str(source.dtype),
            "ndim": len(source.shape),
        },

        "region": {
            "source": [
                {
                    "start": start,
                    "stop": stop,
                }
                for start, stop in region
            ],
        },

        "local_array": {
            "shape": list(local_shape),
            "chunks": list(local_chunks),
            "dtype": str(source.dtype),

            # The local array always starts at zero.
            "origin": [0] * len(local_shape),
        },

        "coordinate_mapping": {
            "source_origin": [
                start
                for start, _ in region
            ],

            "description": (
                "local_index + source_origin = "
                "source_index"
            ),
        },
    }

    # ------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------

    metadata_path = args.output / "meta.json"

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print()
    print("Created:")
    print(f"  Zarr: {output_zarr}")
    print(f"  Meta: {metadata_path}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()




# # import zarr


# # url = "https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll3.zarr/"

# # z = zarr.open(url, mode="r")

# # for group in z:
# #     print(group)
# #     print(z[group].shape)
# #     print(z[group].chunks)
# #     print()


# # import zarr

# # url = "https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll3.zarr/"

# # root = zarr.open(url, mode="r")

# # print(root.tree())


# import argparse
# import concurrent.futures
# from pathlib import Path

# import requests
# import zarr


# def download_file(url: str, output_path: Path, timeout: int = 60):
#     """Download one file."""
#     output_path.parent.mkdir(parents=True, exist_ok=True)

#     if output_path.exists():
#         return output_path, False

#     response = requests.get(url, timeout=timeout)
#     response.raise_for_status()

#     output_path.write_bytes(response.content)

#     return output_path, True


# def get_chunk_indices(chunks, region):
#     """
#     Calculate all chunk indices required for a given array region.

#     region:
#         [(start, stop), (start, stop), ...]
#     """

#     import itertools

#     chunk_ranges = []

#     for (start, stop), chunk_size in zip(region, chunks):
#         first_chunk = start // chunk_size
#         last_chunk = (stop - 1) // chunk_size

#         chunk_ranges.append(
#             range(first_chunk, last_chunk + 1)
#         )

#     return itertools.product(*chunk_ranges)


# def find_arrays(group, prefix=""):
#     """
#     Recursively find all arrays in a Zarr hierarchy.

#     Returns:
#         (path, array)
#     """

#     for name in group:
#         value = group[name]

#         path = f"{prefix}/{name}" if prefix else name

#         if isinstance(value, zarr.Array):
#             yield path, value

#         elif isinstance(value, zarr.Group):
#             yield from find_arrays(value, path)


# def main():
#     parser = argparse.ArgumentParser(
#         description=(
#             "Download selected chunks of a remote Zarr array "
#             "in parallel."
#         )
#     )

#     parser.add_argument(
#         "url",
#         help="URL of the Zarr root.",
#     )

#     parser.add_argument(
#         "output",
#         type=Path,
#         help="Local output directory.",
#     )

#     parser.add_argument(
#         "--array",
#         type=str,
#         required=True,
#         help=(
#             "Path of the Zarr array inside the hierarchy, "
#             "e.g. volume or data/volume."
#         ),
#     )

#     parser.add_argument(
#         "--region",
#         nargs="+",
#         type=int,
#         required=True,
#         metavar="N",
#         help=(
#             "Region as start/stop pairs. "
#             "For a 3D array: "
#             "x_start x_stop y_start y_stop z_start z_stop"
#         ),
#     )

#     parser.add_argument(
#         "--workers",
#         type=int,
#         default=8,
#         help="Number of parallel downloads (default: 8).",
#     )

#     args = parser.parse_args()

#     if len(args.region) % 2 != 0:
#         raise ValueError(
#             "--region must contain start/stop pairs."
#         )

#     # ------------------------------------------------------------
#     # Normalize URL
#     # ------------------------------------------------------------

#     base_url = args.url.rstrip("/") + "/"

#     # ------------------------------------------------------------
#     # Open remote Zarr
#     # ------------------------------------------------------------

#     print("Reading Zarr metadata...")

#     root = zarr.open(base_url, mode="r")

#     # ------------------------------------------------------------
#     # Print available arrays
#     # ------------------------------------------------------------

#     print()
#     print("Available arrays:")

#     arrays = dict(find_arrays(root))

#     for path, array in arrays.items():
#         print(
#             f"  {path}"
#             f"  shape={array.shape}"
#             f"  chunks={array.chunks}"
#             f"  dtype={array.dtype}"
#         )

#     print()

#     # ------------------------------------------------------------
#     # Select array
#     # ------------------------------------------------------------

#     if args.array not in arrays:
#         raise ValueError(
#             f"Array '{args.array}' not found.\n"
#             f"Available arrays:\n"
#             + "\n".join(f"  {path}" for path in arrays)
#         )

#     z = arrays[args.array]
#     array_url = base_url + args.array.rstrip("/") + "/"

#     print(f"Selected array: {args.array}")
#     print(f"Shape         : {z.shape}")
#     print(f"Chunks        : {z.chunks}")
#     print(f"Dtype         : {z.dtype}")

#     # ------------------------------------------------------------
#     # Build region
#     # ------------------------------------------------------------

#     ndim = len(z.shape)

#     if len(args.region) != ndim * 2:
#         raise ValueError(
#             f"Expected {ndim * 2} region values "
#             f"for a {ndim}D array, "
#             f"got {len(args.region)}."
#         )

#     region = [
#         (args.region[i], args.region[i + 1])
#         for i in range(0, len(args.region), 2)
#     ]

#     # ------------------------------------------------------------
#     # Validate region
#     # ------------------------------------------------------------

#     for dim, ((start, stop), size) in enumerate(
#         zip(region, z.shape)
#     ):
#         if start < 0 or stop > size or start >= stop:
#             raise ValueError(
#                 f"Invalid region for dimension {dim}: "
#                 f"{start}:{stop}, size={size}"
#             )

#     # ------------------------------------------------------------
#     # Determine required chunks
#     # ------------------------------------------------------------

#     chunks = list(
#         get_chunk_indices(
#             z.chunks,
#             region,
#         )
#     )

#     print()
#     print(f"Requested region: {region}")
#     print(f"Required chunks : {len(chunks)}")
#     print(f"Workers         : {args.workers}")
#     print()

#     # ------------------------------------------------------------
#     # IMPORTANT:
#     #
#     # We use the Zarr store to obtain the actual storage keys.
#     #
#     # This avoids assuming a layout such as:
#     #
#     #   0.0.0
#     #   0.0.1
#     #
#     # which is not necessarily correct for the Zarr.
#     # ------------------------------------------------------------

#     store = z.store

#     # ------------------------------------------------------------
#     # Download
#     # ------------------------------------------------------------

#     downloaded = 0
#     skipped = 0
#     failed = 0

#     def worker(chunk_index):
#         """
#         Download one Zarr chunk.

#         The actual storage key is obtained from Zarr's store
#         instead of constructing a Zarr-v2-style filename.
#         """

#         try:
#             # Ask Zarr for the actual chunk key.
#             chunk_key = z.metadata.encode_chunk_key(chunk_index)

#             # Convert the store key to a string.
#             chunk_key = str(chunk_key)

#             # The local path mirrors the Zarr store structure.
#             output_path = (
#                 args.output
#                 / args.array
#                 / chunk_key
#             )

#             # Construct remote URL.
#             url = array_url + chunk_key.lstrip("/")

#             path, was_downloaded = download_file(
#                 url,
#                 output_path,
#             )

#             return (
#                 chunk_index,
#                 path,
#                 was_downloaded,
#                 None,
#             )

#         except Exception as e:
#             return (
#                 chunk_index,
#                 None,
#                 False,
#                 e,
#             )

#     with concurrent.futures.ThreadPoolExecutor(
#         max_workers=args.workers
#     ) as executor:

#         futures = [
#             executor.submit(worker, chunk)
#             for chunk in chunks
#         ]

#         for future in concurrent.futures.as_completed(
#             futures
#         ):
#             (
#                 chunk_index,
#                 path,
#                 was_downloaded,
#                 error,
#             ) = future.result()

#             if error is not None:
#                 failed += 1

#                 print(
#                     f"[ERROR] {chunk_index}: {error}"
#                 )

#             elif was_downloaded:
#                 downloaded += 1

#                 print(
#                     f"[DOWNLOADED] {chunk_index}"
#                 )

#             else:
#                 skipped += 1

#                 print(
#                     f"[SKIP] {chunk_index}"
#                 )

#     print()
#     print("Finished.")
#     print(f"Downloaded: {downloaded}")
#     print(f"Skipped   : {skipped}")
#     print(f"Failed    : {failed}")


# if __name__ == "__main__":
#     main()