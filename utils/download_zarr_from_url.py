import argparse
import itertools
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import zarr

if __package__:
    from .inspect_zarr import MetadataReader, normalize_url
else:
    from inspect_zarr import MetadataReader, normalize_url


def normalize_array_path(path):
    parts = [part for part in path.strip("/").split("/") if part not in ("", ".")]
    if ".." in parts or "\\" in path:
        raise ValueError(f"Invalid Zarr path: {path!r}")
    return "/".join(parts)


def multiscales(attrs):
    return attrs.get("multiscales", []) + attrs.get("ome", {}).get("multiscales", [])


def resolve_reference(reader, requested_path):
    """A group refers to its first OME level, or its sole contained array."""
    path = normalize_array_path(requested_path)
    data, attrs = reader.metadata(path)
    if "shape" in data:
        return path
    scales = multiscales(attrs)
    if len(scales) == 1 and scales[0].get("datasets"):
        return normalize_array_path("/".join((path, scales[0]["datasets"][0]["path"])))

    found = []

    def visit(group_path, group_attrs):
        for name in reader.children(group_path, group_attrs):
            child = normalize_array_path("/".join((group_path, name)))
            try:
                metadata, attributes = reader.metadata(child)
            except FileNotFoundError:
                continue
            if "shape" in metadata:
                found.append(child)
            else:
                visit(child, attributes)
            if len(found) > 1:
                return

    visit(path, attrs)
    if len(found) != 1:
        raise ValueError(
            f"Reference group {requested_path!r} does not identify a unique array. "
            "Use --reference-array with an explicit array path."
        )
    return found[0]


def array_geometry(reader, array_path):
    """Read an array's coordinate system without loading its chunks or codecs."""
    metadata, attrs = reader.metadata(array_path)
    if "shape" not in metadata:
        raise ValueError(f"Reference path {array_path!r} is not an array.")
    ndim = len(metadata["shape"])
    geometry = {
        "shape": metadata["shape"],
        "axes": metadata.get("dimension_names") or attrs.get("_ARRAY_DIMENSIONS"),
        "scale": None,
        "translation": None,
    }
    parts = array_path.split("/")
    for depth in range(len(parts) - 1, -1, -1):
        parent = "/".join(parts[:depth])
        try:
            _, parent_attrs = reader.metadata(parent)
        except FileNotFoundError:
            continue
        for scale_metadata in multiscales(parent_attrs):
            for dataset in scale_metadata.get("datasets", []):
                dataset_path = normalize_array_path("/".join((parent, dataset["path"])))
                if dataset_path != array_path:
                    continue
                axes = scale_metadata.get("axes")
                if axes:
                    geometry["axes"] = [axis["name"] if isinstance(axis, dict) else axis for axis in axes]
                transforms = dataset.get("coordinateTransformations", []) + scale_metadata.get("coordinateTransformations", [])
                if not transforms:
                    return geometry
                scale = [Fraction(1)] * ndim
                translation = [Fraction(0)] * ndim
                for transform in transforms:
                    kind = transform["type"]
                    if kind == "identity":
                        continue
                    if kind not in ("scale", "translation"):
                        raise ValueError(f"Array {array_path!r}: unsupported coordinate transform {kind!r}.")
                    values = [Fraction(str(value)) for value in transform[kind]]
                    if len(values) != ndim:
                        raise ValueError(f"Array {array_path!r}: coordinate transform has the wrong dimension count.")
                    if kind == "scale":
                        if any(value <= 0 for value in values):
                            raise ValueError(f"Array {array_path!r}: scales must be positive.")
                        scale = [current * value for current, value in zip(scale, values)]
                        translation = [current * value for current, value in zip(translation, values)]
                    else:
                        translation = [current + value for current, value in zip(translation, values)]
                geometry.update(scale=scale, translation=translation)
                return geometry
    return geometry


def map_reference_region(region, reference, target, reference_path):
    """Round outward to include the reference box in the target coordinate system."""
    if len(reference["shape"]) != len(target["shape"]):
        raise ValueError("Reference and target array must have the same number of dimensions.")
    ndim = len(region)
    order = list(range(ndim))
    if reference["axes"] is not None and target["axes"] is not None:
        ref_axes, target_axes = reference["axes"], target["axes"]
        if (len(ref_axes) != ndim or len(target_axes) != ndim
                or len(set(ref_axes)) != ndim or set(ref_axes) != set(target_axes)):
            raise ValueError("Reference and target array must have matching, unique axis names.")
        order = [ref_axes.index(axis) for axis in target_axes]

    if reference["scale"] is not None and target["scale"] is not None:
        method = "ome"
        scale = [reference["scale"][ref_dim] / target["scale"][dim] for dim, ref_dim in enumerate(order)]
        translation = [(reference["translation"][ref_dim] - target["translation"][dim]) / target["scale"][dim]
                       for dim, ref_dim in enumerate(order)]
    else:
        method = "shape_ratio"
        scale = [Fraction(target["shape"][dim], reference["shape"][ref_dim]) for dim, ref_dim in enumerate(order)]
        translation = [Fraction(0)] * ndim

    mapped = []
    for dim, ref_dim in enumerate(order):
        start, stop = region[ref_dim]
        start = math.floor(start * scale[dim] + translation[dim])
        stop = math.ceil(stop * scale[dim] + translation[dim])
        mapped.append((max(0, start), min(target["shape"][dim], stop)))
    validate_region(mapped, target["shape"])
    return mapped, {
        "array": reference_path,
        "method": method,
        "axis_order": order,
        "scale": [float(value) for value in scale],
        "translation": [float(value) for value in translation],
        "description": "source_index[i] = reference_index[axis_order[i]] * scale[i] + translation[i]; bounds rounded outward and clipped to source shape",
    }


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


def select_nodes(root, paths, metadata_reader=None):
    """Resolve arrays and recursive groups, preserving paths and removing duplicates."""
    arrays = {}
    groups = {"": root}
    visited = set()

    def visit(path, node):
        if path in visited:
            return
        visited.add(path)
        if isinstance(node, zarr.Array):
            arrays[path] = node
        else:
            groups[path] = node
            if metadata_reader:
                _, attrs = metadata_reader.metadata(path)
                names = metadata_reader.children(path, attrs)
            else:
                names = sorted(node.keys())
            for name in names:
                child_path = f"{path}/{name}" if path else name
                if metadata_reader:
                    try:
                        metadata_reader.metadata(child_path)
                    except FileNotFoundError:
                        continue
                visit(child_path, node[name])

    for path in paths:
        path = normalize_array_path(path)
        parts = path.split("/")
        visit(path, root[path] if path else root)
        # Copy attributes of ancestors of individually selected arrays as well.
        for depth in range(1, len(parts)):
            parent = "/".join(parts[:depth])
            groups.setdefault(parent, root[parent])

    return arrays, groups


def copy_attributes(source, destination):
    try:
        destination.attrs.update(dict(source.attrs))
    except Exception as error:
        print(f"Warning: Could not copy all source attributes: {error}")


def progress_key(array_path, block):
    """Namespace blocks by array so equal coordinates never share resume state."""
    return json.dumps([array_path, block_key(block)])


def download_array(args, array_path, source, local_root, region, progress_path, done):
    print()
    print(f"Array: {array_path}")
    print(f"Source: shape={source.shape}, chunks={source.chunks}, dtype={source.dtype}")

    local_shape = tuple(stop - start for start, stop in region)

    # Use the source chunk size where possible.
    local_chunks = tuple(min(chunk, size) for chunk, size in zip(source.chunks, local_shape))

    # Block shape = multiple of the local chunk shape (see iterate_slices).
    block_shape = tuple(c * args.block_factor for c in local_chunks)

    print("Region: " + ", ".join(f"{start}:{stop}" for start, stop in region))
    print(f"Local:  shape={local_shape}, chunks={local_chunks}")

    if array_path in local_root:
        destination = local_root[array_path]
        if (destination.shape != local_shape or destination.chunks != local_chunks
                or destination.dtype != source.dtype):
            raise ValueError(
                f"Existing local array {array_path!r} does not match the requested array/region.\n"
                f"  existing: shape={destination.shape}, chunks={destination.chunks}, dtype={destination.dtype}\n"
                f"  requested: shape={local_shape}, chunks={local_chunks}, dtype={source.dtype}"
            )
    else:
        destination = local_root.create_array(
            array_path,
            shape=local_shape,
            chunks=local_chunks,
            dtype=source.dtype,
        )
        copy_attributes(source, destination)
        # Resume markers cannot apply to a newly created local array.
        done = set()

    total_elements = int(np.prod(local_shape))
    failed_blocks = []

    all_blocks = list(iterate_slices(local_shape, block_shape))

    blocks = [b for b in all_blocks if progress_key(array_path, b) not in done]

    # The progress display starts at the already completed part.
    processed_elements = sum(
        int(np.prod([s.stop - s.start for s in b]))
        for b in all_blocks
        if progress_key(array_path, b) in done
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
                progress_file.write(progress_key(array_path, block) + "\n")
                progress_file.flush()
            except Exception as e:
                failed_blocks.append(block)
                print(f"\n[ERROR] Array {array_path}, block {block_key(block)}: {e}", flush=True)

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

    return {
        "source": {
            "url": args.url,
            "array": array_path,
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


def main():
    parser = argparse.ArgumentParser(
        description="Download remote Zarr arrays or groups, optionally cropped, into a new local Zarr."
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
             "With --reference-array, coordinates refer to that array and are converted for each download. "
             "Otherwise applied to each selected array in its own index coordinates. "
             "If omitted, download entire arrays. "
             "E.g. for a 3D array: axis0_start axis0_stop axis1_start axis1_stop axis2_start axis2_stop",
    )

    parser.add_argument(
        "--array", "--arrays", "--group", "--groups",
        nargs="+",
        action="extend",
        metavar="PATH",
        help="One or more array or group paths inside the Zarr (default: 0). "
             "Groups are downloaded recursively; use . for the entire group. "
             "May be repeated, e.g. --array 0 1 --array group/volume.",
    )

    parser.add_argument(
        "--reference-array", "--reference-group", "--reference",
        metavar="PATH",
        help="Interpret --region in this array's coordinates and map it to every selected array. "
             "The reference need not be downloaded. A group selects its first OME dataset or its sole array. "
             "Uses OME scales/translations when available, otherwise per-axis shape ratios.",
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
        help="Continue an interrupted download. Region, --array, --reference-array and --block-factor "
             "must be identical to the interrupted run.",
    )

    args = parser.parse_args()

    if args.workers < 1 or args.block_factor < 1 or args.retries < 1:
        parser.error("--workers, --block-factor and --retries must be positive.")
    if args.reference_array is not None and args.region is None:
        parser.error("--reference-array requires --region.")

    paths = args.array or ["0"]
    requested_region = parse_region(args.region) if args.region is not None else None
    print(f"Remote Zarr: {args.url}")
    print(f"Selected:    {', '.join(paths)}")
    print("Opening remote Zarr...", flush=True)
    url = normalize_url(args.url)
    reader = MetadataReader(url)
    root = zarr.open_group(url.rstrip("/") + "/", mode="r")
    arrays, groups = select_nodes(root, paths, reader)

    reference = None
    reference_metadata = None
    if args.reference_array is not None:
        reference_path = resolve_reference(reader, args.reference_array)
        reference = array_geometry(reader, reference_path)
        validate_region(requested_region, reference["shape"])
        reference_metadata = {
            "requested_path": args.reference_array,
            "array": reference_path,
            "shape": list(reference["shape"]),
            "region": [{"start": start, "stop": stop} for start, stop in requested_region],
        }
        print(f"Reference:   {reference_path}")
        print("Reference region: " + ", ".join(f"{start}:{stop}" for start, stop in requested_region))

    # Validate every selection before creating output or downloading any data.
    regions = {}
    mappings = {}
    for array_path, source in arrays.items():
        try:
            if reference is not None:
                target = array_geometry(reader, array_path)
                region, mappings[array_path] = map_reference_region(requested_region, reference, target, reference_path)
                print(f"Mapped {array_path} ({mappings[array_path]['method']}): "
                      + ", ".join(f"{start}:{stop}" for start, stop in region))
            else:
                region = requested_region if requested_region is not None else [(0, size) for size in source.shape]
            validate_region(region, source.shape)
        except ValueError as error:
            raise ValueError(f"Array {array_path!r}: {error}") from error
        regions[array_path] = region

    zarr_name = Path(urlsplit(url).path.rstrip("/")).name
    if not zarr_name or zarr_name == "..":
        raise ValueError("Remote Zarr URL must include a directory name.")
    output_zarr = args.output / zarr_name
    progress_path = args.output / "progress.txt"
    state_path = args.output / "progress.json"
    state = {
        "url": url,
        "reference": reference_metadata,
        "block_factor": args.block_factor,
        "arrays": {
            path: {"region": [[start, stop] for start, stop in regions[path]],
                   "shape": list(source.shape), "chunks": list(source.chunks),
                   "dtype": str(source.dtype), "mapping": mappings.get(path)}
            for path, source in arrays.items()
        },
    }
    if output_zarr.exists() and not args.resume:
        raise FileExistsError(
            f"Output already exists: {output_zarr}\n"
            "Use --resume to continue, or delete it first."
        )

    done = set()
    if output_zarr.exists():
        print("Resuming existing download...")
        if state_path.exists():
            if json.loads(state_path.read_text(encoding="utf-8")) != state:
                raise ValueError("Download settings differ from the interrupted run. Keep the URL, arrays, region, reference and block factor unchanged.")
        elif reference is not None and progress_path.exists() and progress_path.stat().st_size:
            raise ValueError("Cannot resume a reference-based download from legacy progress without matching coordinate metadata. Use a new output directory.")
        if progress_path.exists():
            done = set(progress_path.read_text().splitlines())
            legacy_done = {key for key in done if not key.startswith("[")}
            if legacy_done:
                if len(arrays) != 1:
                    raise ValueError("Legacy progress.txt requires resuming with the original single array first.")
                array_path = next(iter(arrays))
                done = (done - legacy_done) | {json.dumps([array_path, key]) for key in legacy_done}
        local_root = zarr.open_group(output_zarr, mode="r+")
    else:
        if args.resume:
            print("Nothing to resume, starting a new download.")
        args.output.mkdir(parents=True, exist_ok=True)
        if progress_path.exists():
            progress_path.unlink()
        local_root = zarr.open_group(output_zarr, mode="w")

    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"Output: {output_zarr}")
    for group_path, source_group in groups.items():
        destination_group = local_root.require_group(group_path) if group_path else local_root
        copy_attributes(source_group, destination_group)

    array_metadata = {}
    for array_path, source in arrays.items():
        array_metadata[array_path] = download_array(
            args, array_path, source, local_root, regions[array_path], progress_path, done,
        )
        if reference is not None:
            array_metadata[array_path]["coordinate_mapping"]["reference"] = mappings[array_path]

    # Keep the existing metadata layout for a single array.
    if len(array_metadata) == 1:
        metadata = next(iter(array_metadata.values()))
    else:
        metadata = {
            "source": {"url": args.url, "paths": paths},
            "download": {"created_at": datetime.now(timezone.utc).isoformat()},
            "arrays": array_metadata,
        }
    if reference_metadata is not None:
        metadata["reference"] = reference_metadata

    metadata_path = args.output / "meta.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Delete last so failures while saving metadata can still be resumed.
    if progress_path.exists():
        progress_path.unlink()
    if state_path.exists():
        state_path.unlink()
    print()
    print("Download complete.")
    print(f"  Zarr: {output_zarr}")
    print(f"  Meta: {metadata_path}")


if __name__ == "__main__":
    main()
