import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import zarr


def parse_region(values):
    """
    Convert:
        a_start a_stop b_start b_stop c_start c_stop

    into:
        [(a_start, a_stop), (b_start, b_stop), (c_start, c_stop)]
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


def iterate_slices(shape, block_shape):
    """
    Iterate over the output array block by block.

    block_shape must be a multiple of the local chunk shape, so that every
    block covers whole chunks. Then no two threads ever write to the same
    chunk (important: concurrent writes to one chunk would be a race).
    """

    block_counts = [(size + b - 1) // b for size, b in zip(shape, block_shape)]

    for block_index in itertools.product(*[range(n) for n in block_counts]):
        slices = []

        for dim, index in enumerate(block_index):
            start = index * block_shape[dim]
            stop = min(start + block_shape[dim], shape[dim])

            slices.append(slice(start, stop))

        yield tuple(slices)


def block_key(block):
    """Stable string identifying a block, e.g. '0:64,0:64,64:100'."""

    return ",".join(f"{s.start}:{s.stop}" for s in block)


def copy_block(source, destination, region, local_slices, retries):
    """Read one block from the remote array and write it to the local array."""

    source_slices = tuple(
        slice(region[dim][0] + s.start, region[dim][0] + s.stop)
        for dim, s in enumerate(local_slices)
    )

    for attempt in range(1, retries + 1):
        try:
            data = source[source_slices]
            break
        except Exception:
            if attempt == retries:
                raise
            # Exponential backoff: 1s, 2s, 4s, ...
            time.sleep(2 ** (attempt - 1))

    destination[local_slices] = data

    return int(np.prod(data.shape))


def main():
    parser = argparse.ArgumentParser(
        description="Download a remote Zarr array or a region and store it as a new local Zarr starting at index 0."
    )

    parser.add_argument(
        "url",
        help="URL of the remote Zarr.",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="Output directory; the local Zarr keeps the remote directory name.",
    )

    parser.add_argument(
        "--region",
        nargs="+",
        type=int,
        metavar="N",
        help="Region as start/stop pairs, one pair per dimension, in array axis order. "
             "If omitted, download the entire selected array. "
             "E.g. for a 3D array: axis0_start axis0_stop axis1_start axis1_stop axis2_start axis2_stop",
    )

    parser.add_argument(
        "--array",
        type=str,
        default="0",
        help="Path of the array inside the Zarr, e.g. volume or group/volume.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of parallel download threads (default: 32).",
    )

    parser.add_argument(
        "--block-factor",
        type=int,
        default=1,
        help="Each task copies block-factor chunks per dimension (default: 1 = one chunk per task). "
             "Larger values (e.g. 2 or 4) reduce task overhead but use more memory per task.",
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Number of attempts per block on errors (default: 5).",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted download. Region, --array and --block-factor "
             "must be identical to the interrupted run.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Parse region
    # ------------------------------------------------------------

    region = parse_region(args.region) if args.region is not None else None

    print(f"Remote Zarr: {args.url}")
    print(f"Array:       {args.array}")

    # ------------------------------------------------------------
    # Open remote Zarr
    # ------------------------------------------------------------

    print()
    print("Opening remote Zarr...", flush=True)

    root = zarr.open(
        args.url.rstrip("/") + "/",
        mode="r",
    )

    source = root[args.array]

    print()
    print(f"Source: shape={source.shape}, chunks={source.chunks}, dtype={source.dtype}")

    # ------------------------------------------------------------
    # Validate region
    # ------------------------------------------------------------

    if region is None:
        region = [(0, size) for size in source.shape]

    validate_region(region, source.shape)

    # ------------------------------------------------------------
    # Calculate local shape
    # ------------------------------------------------------------

    local_shape = tuple(stop - start for start, stop in region)

    # Use the source chunk size where possible.
    local_chunks = tuple(min(chunk, size) for chunk, size in zip(source.chunks, local_shape))

    # Block shape = multiple of the local chunk shape (see iterate_slices).
    block_shape = tuple(c * args.block_factor for c in local_chunks)

    print("Region: " + ", ".join(f"{start}:{stop}" for start, stop in region))
    print(f"Local:  shape={local_shape}, chunks={local_chunks}")

    # ------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------

    args.output.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Create local Zarr
    # ------------------------------------------------------------

    zarr_name = Path(urlsplit(args.url).path.rstrip("/")).name
    if not zarr_name or zarr_name == "..":
        raise ValueError("Remote Zarr URL must include a directory name.")
    output_zarr = args.output / zarr_name

    print()
    print(f"Output: {output_zarr}")

    progress_path = args.output / "progress.txt"

    if output_zarr.exists():
        if not args.resume:
            raise FileExistsError(
                f"Output already exists: {output_zarr}\n"
                "Use --resume to continue, or delete it first."
            )

        print("Resuming existing download...")

        z = zarr.open(
            output_zarr,
            mode="r+"
        )

        destination = z[args.array]

        if destination.shape != local_shape or destination.chunks != local_chunks:
            raise ValueError(
                "Existing local array does not match the requested region.\n"
                f"  existing: shape={destination.shape}, chunks={destination.chunks}\n"
                f"  requested: shape={local_shape}, chunks={local_chunks}"
            )

    else:
        if args.resume:
            print("Nothing to resume, starting a new download.")

        # A leftover progress file from an earlier run must not be reused.
        if progress_path.exists():
            progress_path.unlink()

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

        # --------------------------------------------------------
        # Copy attributes (only for a new array)
        # --------------------------------------------------------

        try:
            for key, value in source.attrs.items():
                destination.attrs[key] = value
        except Exception:
            print("Warning: Could not copy all source attributes.")

    # ------------------------------------------------------------
    # Download data in parallel
    # ------------------------------------------------------------

    total_elements = int(np.prod(local_shape))
    failed_blocks = []

    all_blocks = list(iterate_slices(local_shape, block_shape))

    # Blocks that were completed in an earlier run (only filled with --resume).
    done = set()

    if args.resume and progress_path.exists():
        done = set(progress_path.read_text().split())

    blocks = [b for b in all_blocks if block_key(b) not in done]

    # The progress display starts at the already completed part.
    processed_elements = sum(
        int(np.prod([s.stop - s.start for s in b]))
        for b in all_blocks
        if block_key(b) in done
    )

    print()
    print(
        f"Downloading region: {len(blocks)} of {len(all_blocks)} tasks remaining, "
        f"{args.workers} workers...", flush=True
    )

    interactive = sys.stdout.isatty()
    progress_step = 1 if interactive else 5
    last_percentage = -progress_step

    with progress_path.open("a") as progress_file, \
            ThreadPoolExecutor(max_workers=args.workers) as executor:

        futures = {
            executor.submit(copy_block, source, destination, region, block, args.retries): block
            for block in blocks
        }

        for completed, future in enumerate(as_completed(futures), start=1):
            block = futures[future]

            try:
                processed_elements += future.result()

                # Only written after the block is completely on disk.
                progress_file.write(block_key(block) + "\n")
                progress_file.flush()
            except Exception as e:
                failed_blocks.append(block)
                print(f"\n[ERROR] Block {block_key(block)}: {e}", flush=True)

            percentage = 100.0 * processed_elements / total_elements
            if percentage >= last_percentage + progress_step or completed == len(blocks):
                print(
                    "\r" if interactive else "",
                    f"Progress: {percentage:6.2f}%"
                    f" | tasks: {len(all_blocks) - len(blocks) + completed}/{len(all_blocks)}"
                    f" | failed: {len(failed_blocks)}",
                    sep="", end="" if interactive else "\n", flush=True,
                )
                last_percentage = percentage

    if interactive and blocks:
        print()

    if failed_blocks:
        raise RuntimeError(
            f"{len(failed_blocks)} block(s) failed after {args.retries} attempts. "
            "The local Zarr is incomplete. Run the same command again with --resume."
        )

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------

    metadata = {
        "source": {
            "url": args.url,
            "array": args.array,
        },

        "download": {
            "created_at": datetime.now(timezone.utc).isoformat(),
        },

        "source_array": {
            "shape": list(source.shape),
            "chunks": list(source.chunks),
            "dtype": str(source.dtype),
            "ndim": len(source.shape),
        },

        "region": {
            "source": [
                {"start": start, "stop": stop}
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
            "source_origin": [start for start, _ in region],
            "description": "local_index + source_origin = source_index",
        },
    }

    # ------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------

    metadata_path = args.output / "meta.json"

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # The download is complete, the progress file is no longer needed.
    # (Deleted last, so a failure above can still be resumed.)
    if progress_path.exists():
        progress_path.unlink()

    print()
    print("Download complete.")
    print(f"  Zarr: {output_zarr}")
    print(f"  Meta: {metadata_path}")


if __name__ == "__main__":
    main()
