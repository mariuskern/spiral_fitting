"""Fetch arbitrary subtrees of the PHercParis4 spiral dataset.

Complements fetch_roi.py (which handles the z-sliced lasagna zarrs). This one
is for everything else: patch libraries, fibers, tracks, root JSONs.

Two modes, so several download processes can share one listing pass:

  --list      recursively list the given prefixes into --manifest (jsonl of
              {path, size}); listing 400k+ files costs minutes of API pages,
              so do it once, not once per shard
  --download  read --manifest, drop files already local with matching size,
              keep crc32(path) % N == I when --shard I:N is given, then batch
              download with the same 429 retry/backoff as fetch_roi

Usage:
    python fetch_tree.py --list --manifest m.jsonl verified_patches unverified_patches
    python fetch_tree.py --download --manifest m.jsonl --shard 0:4 --jobs 32
    python fetch_tree.py --direct outer_shell fibers tracks abs_winding.json ...
        (--direct = list+download inline, for small trees)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import HfApi

BUCKET = "scrollprize/datasets"
PREFIX = "spiral/PHercParis4"
LOCAL = Path("/root/spiral-dataset/PHercParis4")

api = HfApi()
_print_lock = threading.Lock()


def say(*a) -> None:
    with _print_lock:
        print(*a, flush=True)


def list_prefix(rel_prefix: str):
    """Yield (path, size) under one prefix; a bare file yields itself."""
    full = f"{PREFIX}/{rel_prefix}"
    infos = list(api.get_bucket_paths_info(BUCKET, [full]))
    if infos and type(infos[0]).__name__ == "BucketFile":
        yield full, infos[0].size or 0
        return
    for f in api.list_bucket_tree(BUCKET, full, recursive=True):
        if type(f).__name__ == "BucketFile":
            yield f.path, f.size or 0


def needs_fetch(path: str, size: int) -> bool:
    dst = LOCAL / Path(path).relative_to(PREFIX)
    return not (dst.is_file() and dst.stat().st_size == size)


def recover_http_client() -> None:
    """Rebuild huggingface_hub's shared HTTP client after it has been closed.

    An SSL EOF mid-transfer closes the shared httpx client but leaves the closed
    instance cached in utils._http._GLOBAL_CLIENT, so every later request in the
    process fails with "client has been closed" while the process stays alive --
    a download that looks like it is running and moves no bytes. close_session()
    drops the cached instance; the next get_session() builds a fresh one.
    """
    try:
        from huggingface_hub.utils import close_session
        close_session()
    except Exception as exc:
        say(f"  client reset failed: {exc}"[:120])


def fetch_batch(batch: list[tuple[str, int]]) -> tuple[int, int, str]:
    import time
    size = sum(s for _, s in batch)
    msg = "no attempt made"
    for attempt in range(6):
        # Re-filter every attempt. A batch can hold multi-GB files, and a
        # failure anywhere used to re-download the whole batch from zero --
        # on a flaky link that never converges, and the bytes are wasted
        # silently (network busy, disk flat).
        remaining = [(p, s) for p, s in batch if needs_fetch(p, s)]
        if not remaining:
            return len(batch), size, ""
        jobs = [(p, str(LOCAL / Path(p).relative_to(PREFIX))) for p, _ in remaining]
        try:
            api.download_bucket_files(BUCKET, jobs)
            return len(batch), size, ""
        except Exception as exc:
            msg = str(exc)
            low = msg.lower()
            if "429" in msg or "rate limit" in low:
                time.sleep(min(120, 15 * (attempt + 1)))
                continue
            if "client has been closed" in low or "ssl" in low or "eof" in low:
                recover_http_client()
                time.sleep(2)
                continue
            # Everything else is retried too. The bucket API's Xet layer fails
            # with "CAS Client Error: Request middleware error" whenever the
            # CDN connection drops, which on a flaky link is most batches;
            # treating unrecognised errors as permanent turned a slow download
            # into a failed one.
            time.sleep(min(30, 3 * (attempt + 1)))
    return 0, 0, f"gave up after 6 attempts: {msg[:120]}"


def download(todo: list[tuple[str, int]], jobs: int) -> int:
    import random
    import time
    random.shuffle(todo)
    batches = [todo[i:i + 16] for i in range(0, len(todo), 16)]
    total_mb = sum(s for _, s in todo) / 1e6
    say(f"downloading {len(todo)} files, {total_mb:.0f} MB, {jobs} threads")
    got = failed = got_bytes = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=jobs) as pool:
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
                say(f"  {got}/{len(todo)} files  {got_bytes / 1e6:.0f}/{total_mb:.0f} MB  "
                    f"{got_bytes / 1e6 / el:.2f} MB/s  "
                    f"eta {((total_mb - got_bytes / 1e6) / max(0.01, got_bytes / 1e6 / el)) / 60:.0f} min")
    say(f"done: {got}/{len(todo)} files, {failed} failed batches")
    return failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="*")
    ap.add_argument("--manifest")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--direct", action="store_true")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--exclude", action="append", default=[],
                    help="skip paths containing this substring; repeatable. "
                         "The tracks directory ships two interchangeable track "
                         "databases and a fit uses one, so excluding the other "
                         "saves tens of GB.")
    args = ap.parse_args()

    if args.direct:
        todo = []
        for prefix in args.prefixes:
            found = list(list_prefix(prefix))
            if args.exclude:
                kept = [(p, s) for p, s in found
                        if not any(x in p for x in args.exclude)]
                say(f"{prefix}: {len(found)} files, "
                    f"{sum(s for _, s in found) / 1e6:.1f} MB; "
                    f"excluded {len(found) - len(kept)} "
                    f"({sum(s for p, s in found if any(x in p for x in args.exclude)) / 1e9:.1f} GB)")
                found = kept
            else:
                say(f"{prefix}: {len(found)} files, "
                    f"{sum(s for _, s in found) / 1e6:.1f} MB")
            todo += [(p, s) for p, s in found if needs_fetch(p, s)]
        sys.exit(1 if download(todo, args.jobs) else 0)

    if args.list:
        n = 0
        with open(args.manifest, "w") as out:
            for prefix in args.prefixes:
                for p, s in list_prefix(prefix):
                    out.write(json.dumps({"p": p, "s": s}) + "\n")
                    n += 1
                    if n % 20000 == 0:
                        say(f"listed {n}...")
        say(f"manifest: {n} files")
        return

    if args.download:
        todo = []
        with open(args.manifest) as fh:
            for line in fh:
                rec = json.loads(line)
                todo.append((rec["p"], rec["s"]))
        if args.shard:
            i, n = (int(x) for x in args.shard.split(":"))
            todo = [t for t in todo if zlib.crc32(t[0].encode()) % n == i]
            say(f"shard {i}:{n} -> {len(todo)} files")
        todo = [t for t in todo if needs_fetch(*t)]
        say(f"after local-skip: {len(todo)} files")
        sys.exit(1 if download(todo, args.jobs) else 0)

    ap.error("pick --list, --download, or --direct")


if __name__ == "__main__":
    main()
