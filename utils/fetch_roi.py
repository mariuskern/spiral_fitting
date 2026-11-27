"""Download only the zarr chunks a given z-range of the fit actually reads.

The four lasagna fields together are >10 GB across ~460,000 files, but a fit
over z [4000, 5000) -- the tutorial's suggested first run -- touches under 10%
of that. The stores use `dimension_separator: "/"`, so a chunk's key is
literally `<group>/<cz>/<cy>/<cx>` and a z-slab is a directory subtree that can
be enumerated on its own.

Missing chunks are not an error for either field. Both stores declare
`fill_value: 0`; for normals/grad_mag that reads as zero, and for surf_sdt 0 is
the reserved no-data value the fitter's validity policy already handles. A
partial store is therefore safe as long as the fit stays inside the fetched
z-range.

Why the huggingface_hub API and not `hf buckets sync`: sync always lists its
source recursively, so even `--include '*.zarray'` enumerates all ~115,000
files of a store to fetch two of them -- which blows the API rate limit before
a single byte is downloaded. Listing per z-slab keeps each enumeration to a few
pages, and `download_bucket_files` takes an explicit path list with no
enumeration at all.

The two field families sit on different grids, and neither is the fit's own
coordinate system (working voxels, 9.6 um):

  normals, grad_mag  group 4, chunk 32, array_z = working_z / 4
                     (lasagna_scale in fit_spiral)
  surf_sdt           group 1, chunk 128, array_z = working_z / 2
                     (scale_vs_working in the store's .zattrs)

Both are read from the stores themselves rather than trusted from this
docstring, so a regenerated dataset cannot silently shift the mapping.

Usage:
    python fetch_roi.py 4000 5000 [--jobs 8] [--margin 320] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import HfApi

# The shared requests session behind HfApi keeps urllib3's default pool of 10
# connections; with --jobs 48 the other 38 threads queue on the pool, which is
# exactly the 0.7 MB/s plateau measured on 2026-07-29. Size the pool to the
# thread count instead (keep >= --jobs).
_POOL = 128


def _widen_http_pool() -> None:
    """Size the shared HTTP connection pool to the thread count.

    hub <1.0 (requests): configure_http_backend with a widened HTTPAdapter.
    hub >=1.0 (httpx): the default shared client caps at 100 connections;
    raise the limits before HfApi first builds it.
    """
    try:
        from huggingface_hub import configure_http_backend
        import requests

        def factory() -> requests.Session:
            s = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=_POOL, pool_maxsize=_POOL)
            s.mount("https://", adapter)
            s.mount("http://", adapter)
            return s

        configure_http_backend(backend_factory=factory)
        return
    except ImportError:
        pass
    try:
        import httpx
        from huggingface_hub import set_client_factory

        def hx_factory() -> httpx.Client:
            limits = httpx.Limits(
                max_connections=_POOL, max_keepalive_connections=_POOL)
            return httpx.Client(limits=limits, follow_redirects=True,
                                timeout=httpx.Timeout(60.0))

        set_client_factory(hx_factory)
    except ImportError:
        print("note: could not widen HTTP pool; using library default")


_widen_http_pool()

BUCKET = "scrollprize/datasets"
PREFIX = "spiral/PHercParis4/lasagna_inputs"
LOCAL = Path("/root/spiral-dataset/PHercParis4/lasagna_inputs")

# (store, group, working-voxels per array voxel)
FIELDS = [
    ("las_008_nx.ome.zarr", "4", 4),
    ("las_008_ny.ome.zarr", "4", 4),
    ("las_008_grad_mag.ome.zarr", "4", 4),
    ("las_008_surf_sdt.ome.zarr", "1", 2),
]

api = HfApi()
_print_lock = threading.Lock()


def say(*a) -> None:
    with _print_lock:
        print(*a, flush=True)


def fetch_metadata() -> None:
    """Pull each store's .zattrs and its group's .zarray/.zattrs by path.

    No listing at all: the paths are known, and get_bucket_paths_info confirms
    which of them exist before download_bucket_files is asked for them.
    """
    wanted: list[str] = []
    for store, group, _ in FIELDS:
        wanted += [f"{PREFIX}/{store}/.zattrs",
                   f"{PREFIX}/{store}/{group}/.zarray",
                   f"{PREFIX}/{store}/{group}/.zattrs"]

    present = {f.path for f in api.get_bucket_paths_info(BUCKET, wanted)}
    missing = [p for p in wanted if p not in present]
    if missing:
        say(f"note: not published (skipped): {[Path(p).name for p in missing]}")

    jobs = [(p, str(LOCAL / Path(p).relative_to(PREFIX))) for p in sorted(present)]
    api.download_bucket_files(BUCKET, jobs)
    say(f"metadata: {len(jobs)} files")


def chunk_range(store: str, group: str, scale: int,
                z_begin: int, z_end: int, margin: int) -> tuple[range, int]:
    zarray = LOCAL / store / group / ".zarray"
    if not zarray.is_file():
        sys.exit(f"{zarray} missing -- metadata fetch did not land")
    meta = json.loads(zarray.read_text())
    zsize, czsize = meta["shape"][0], meta["chunks"][0]
    n_chunks = -(-zsize // czsize)

    attrs_path = LOCAL / store / ".zattrs"
    if attrs_path.is_file():
        attrs = json.loads(attrs_path.read_text())
        declared = attrs.get("scale_vs_working")
        if declared is not None and int(declared) != scale:
            sys.exit(f"{store}: scale_vs_working={declared}, expected {scale}. "
                     f"The dataset grid changed; update FIELDS.")

    lo = max(0, (z_begin - margin) // scale // czsize)
    hi = min(n_chunks - 1, (z_end + margin) // scale // czsize)
    return range(lo, hi + 1), n_chunks


def list_slab(store: str, group: str, cz: int) -> tuple[str, list, str]:
    """Enumerate one z-slab; return the files not already local."""
    label = f"{store.split('.')[0]}/{group}/{cz}"
    slab_prefix = f"{PREFIX}/{store}/{group}/{cz}"
    try:
        files = [f for f in api.list_bucket_tree(BUCKET, slab_prefix, recursive=True)
                 if type(f).__name__ == "BucketFile"]
    except Exception as exc:                       # rate limit, transient 5xx
        return label, [], f"list failed: {exc}"[:160]
    todo = []
    for f in files:
        dst = LOCAL / Path(f.path).relative_to(PREFIX)
        if not (dst.is_file() and dst.stat().st_size == (f.size or 0)):
            todo.append(f)
    return label, todo, ""


def fetch_batch(batch: list) -> tuple[int, int, str]:
    """Download one batch of files, retrying through rate limits.

    The chunks are ~32 KB and a request round-trip through this link is
    ~0.5-1 s, so throughput is latency-bound: it scales with in-flight
    requests, not with per-connection bandwidth. Batches keep the Python/API
    overhead per file small; the thread pool supplies the concurrency.
    """
    import time
    jobs = [(f, str(LOCAL / Path(f.path).relative_to(PREFIX))) for f in batch]
    size = sum(f.size or 0 for f in batch)
    msg = "no attempt made"
    for attempt in range(6):
        try:
            api.download_bucket_files(BUCKET, jobs)
            return len(jobs), size, ""
        except Exception as exc:
            msg = str(exc)
            low = msg.lower()
            if "429" in msg or "rate limit" in low:
                time.sleep(min(120, 15 * (attempt + 1)))
                continue
            if "client has been closed" in low or "ssl" in low or "eof" in low:
                # A dropped connection closes huggingface_hub's shared client
                # but leaves it cached, so every later request in this process
                # fails until the cache is dropped.
                try:
                    from huggingface_hub.utils import close_session
                    close_session()
                except Exception:
                    pass
                time.sleep(2)
                continue
            # Anything else is transient too until proven otherwise: the Xet
            # CAS layer reports CDN drops as "Request middleware error", and
            # giving up on those turns a slow link into a failed download.
            time.sleep(min(30, 3 * (attempt + 1)))
    return 0, 0, f"gave up after 6 attempts: {msg[:120]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("z_begin", type=int)
    ap.add_argument("z_end", type=int)
    ap.add_argument("--jobs", type=int, default=64,
                    help="concurrent download batches. The link is "
                         "latency-bound for 32 KB chunks (~1 s per request "
                         "round-trip), so throughput scales with in-flight "
                         "requests until the HF rate limit pushes back")
    ap.add_argument("--margin", type=int, default=320,
                    help="working voxels of slack each side; the fit's flow "
                         "bounds extend past the fitted range "
                         "(flow_bounds_z_margin defaults to 160)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shard", default=None,
                    help="I:N -- download only files with crc32(path) %% N == I. "
                         "Lets several processes split the work with no overlap; "
                         "each still lists everything (listing is cheap).")
    args = ap.parse_args()

    fetch_metadata()

    slabs = []
    for store, group, scale in FIELDS:
        rng, total = chunk_range(store, group, scale,
                                 args.z_begin, args.z_end, args.margin)
        say(f"  {store:30s} group {group}: z-chunks {rng.start}-{rng.stop - 1} "
            f"of {total}  ({len(rng) / total * 100:.1f}%)")
        slabs += [(store, group, cz) for cz in rng]

    say(f"\n{len(slabs)} z-slabs to enumerate")
    if args.dry_run:
        return

    # stage 1: listing, slab-level parallelism (a listing is a few API pages)
    todo, list_failed = [], 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(list_slab, *s): s for s in slabs}
        for i, fut in enumerate(as_completed(futs), 1):
            label, files, err = fut.result()
            if err:
                list_failed += 1
                say(f"[list {i}/{len(slabs)}] FAIL {label}: {err}")
            else:
                todo += files
                say(f"[list {i}/{len(slabs)}] {label}: {len(files)} to fetch")
    if args.shard:
        import zlib
        i, n = (int(x) for x in args.shard.split(":"))
        todo = [f for f in todo if zlib.crc32(f.path.encode()) % n == i]
        say(f"shard {i}:{n} -> {len(todo)} files")
    total_mb = sum(f.size or 0 for f in todo) / 1e6
    say(f"\nstage 2: {len(todo)} files, {total_mb:.0f} MB, "
        f"{args.jobs} threads x batches of 16")

    # stage 2: downloading, file-level parallelism -- the actual speed lever
    import random
    import time
    random.shuffle(todo)          # spread load across stores/slabs
    batches = [todo[i:i + 16] for i in range(0, len(todo), 16)]
    got = failed = got_bytes = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = [pool.submit(fetch_batch, b) for b in batches]
        for i, fut in enumerate(as_completed(futs), 1):
            n, size, err = fut.result()
            got += n
            got_bytes += size
            if err:
                failed += 1
                say(f"  batch failed: {err}")
            if i % 50 == 0 or i == len(batches):
                el = time.time() - t0
                say(f"  {got}/{len(todo)} files  "
                    f"{got_bytes / 1e6:.0f}/{total_mb:.0f} MB  "
                    f"{got_bytes / 1e6 / el:.2f} MB/s  "
                    f"eta {((total_mb - got_bytes / 1e6) / max(0.01, got_bytes / 1e6 / el)) / 60:.0f} min")
    say(f"\n{got}/{len(todo)} files downloaded, {failed} failed batches, "
        f"{list_failed} failed listings")
    if failed or list_failed:
        say("rerun this command to retry what's missing (already-local files "
            "are skipped)")


if __name__ == "__main__":
    main()
