"""Portable sheet pipeline: bounded block processes and ordered result delivery.

Only small metadata crosses process queues. Finished labels are handed off as
temporary memory-mapped NPY files, avoiding pickling full CT/label arrays.
Workers save archives/previews, but only the coordinator commits checkpoints.
"""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import multiprocessing
from pathlib import Path
import time

import numpy as np

from vesuvius_sheet_tools.separate_sheets import distinct_colors, render
from sheet_compute import Compute
from sheet_io import Volume, load_checkpoint, read_region, save_npz
from sheet_processing import segment, split_labels
from sheet_progress import configure_worker_warnings, worker_warnings


def process_block(args, job, pred, ct, compute):
    """Compute one independent block; keep its commit record for the coordinator."""
    index, grid, origin, shape, total = job
    started = time.monotonic()
    z, y, x = origin
    relative = Path("blocks") / f"z{z}" / f"tile_y{y}_x{x}.npz"
    checkpoint = load_checkpoint(args.out, relative, origin, shape, index) if args.resume else None
    ct_block = None
    if checkpoint is not None:
        labels, stats, block = checkpoint
    else:
        tick = time.monotonic()
        prediction = read_region(pred, origin, shape)
        mask = np.isfinite(prediction) & (prediction > args.threshold)
        del prediction
        if mask.any() and (not args.no_ct_gating or not args.no_split):
            ct_block = read_region(ct, origin, shape)
            if not np.isfinite(ct_block).all():
                raise ValueError(f"CT block {origin} contains NaN or infinity")
            if not args.no_ct_gating:
                mask &= ct_block > 0
        timings = dict(read=round(time.monotonic() - tick, 3))
        stats = dict(fallback_seeds=0, instances_before_merge=0)
        report = []
        compute.fallbacks.clear()
        compute.gpu_wait_seconds = 0.0
        if mask.any():
            tick = time.monotonic()
            mask = compute.clean(mask, args.min_cc)
            timings["clean"] = round(time.monotonic() - tick, 3)
        if mask.any():
            tick = time.monotonic()
            labels, stats = segment(mask, args, compute)
            timings["segment_merge"] = round(time.monotonic() - tick, 3)
            if not args.no_split:
                tick = time.monotonic()
                labels, report = split_labels(labels, ct_block, args.mega_frac,
                                              args.min_center, compute=compute)
                timings["ct_split"] = round(time.monotonic() - tick, 3)
        else:
            labels = np.zeros(shape, dtype=np.int32)
        ids = np.unique(labels)
        stats.update(instances=int(np.count_nonzero(ids)), mask_voxels=int(mask.sum()),
                     split_report=report, cuda_oom_fallbacks=list(compute.fallbacks),
                     gpu_wait_seconds=round(compute.gpu_wait_seconds, 3), stage_seconds=timings)
        block = dict(index=index, block=relative.as_posix(), origin_zyx=list(origin),
                     shape_zyx=list(shape), ids=ids[ids > 0].tolist(),
                     source_metadata=dict(crop_origin_zyx=list(origin), shape_zyx=list(shape),
                                          source_shape_zyx=list(pred.shape)))
        (args.out / relative).parent.mkdir(parents=True, exist_ok=True)
        tick = time.monotonic()
        save_npz(args.out / relative, labels=labels, origin=np.array(origin), z0=z, y0=y, x0=x)
        timings["save"] = round(time.monotonic() - tick, 3)
    preview = args.out / "previews" / f"z{z}_y{y}_x{x}.png"
    if not args.no_previews and (checkpoint is None or not preview.exists()):
        tick = time.monotonic()
        mid = shape[0] // 2
        _, compact = np.unique(np.r_[0, labels[mid].ravel()], return_inverse=True)
        image = compact[1:].reshape(shape[1:])
        background = ct_block[mid] if ct_block is not None else np.zeros(shape[1:], dtype=np.uint8)
        preview.parent.mkdir(exist_ok=True)
        render(image, background, preview, distinct_colors(max(int(image.max()), 1)))
        if checkpoint is None:
            stats["stage_seconds"]["preview"] = round(time.monotonic() - tick, 3)
    return dict(labels=labels, stats=stats, block=block, grid=grid,
                reused=checkpoint is not None, elapsed=time.monotonic() - started)


def ordered_results(executor, function, jobs, limit):
    """Bound running AND completed results, yielding in input order (no eager map)."""
    jobs = iter(jobs)
    pending = deque()
    try:
        for _ in range(limit):
            job = next(jobs, None)
            if job is None:
                break
            pending.append(executor.submit(function, job))
        while pending:
            yield pending.popleft().result()
            job = next(jobs, None)
            if job is not None:
                pending.append(executor.submit(function, job))
    finally:
        for future in pending:
            future.cancel()


def _initialize(args, gpu_lock, log_queue):
    global _args, _gpu_lock, _state
    _args, _gpu_lock, _state = args, gpu_lock, None
    configure_worker_warnings(log_queue)


def _run_block(job):
    global _state
    if _state is None:
        # Initialize lazily so errors propagate through the task future.
        _state = (Volume(_args.pred, _args.pred_dataset),
                  Volume(_args.volume, _args.ct_dataset),
                  Compute(_args.device, _args.tensor_chunk_depth, _args.eigh_chunk,
                          _args.oom_policy, gpu_lock=_gpu_lock))
    result = process_block(_args, job, *_state)
    labels = result.pop("labels")
    path = _args.out / "portable_sheet_worker_cache" / f"block_{job[0]:06d}.npy"
    temporary = path.with_suffix(".npy.tmp")
    tick = time.monotonic()
    with temporary.open("wb") as stream:
        np.save(stream, labels, allow_pickle=False)
    temporary.replace(path)
    elapsed = time.monotonic() - tick
    if not result["reused"]:
        result["stats"]["stage_seconds"]["handoff"] = round(elapsed, 3)
    result["elapsed"] += elapsed
    result["labels_path"] = str(path)
    return result


@contextmanager
def block_results(args, jobs, pred, ct, compute):
    """Serial by default; optional spawned processes with one shared GPU gate."""
    if args.workers == 1:
        def serial_results():
            for job in jobs:
                try:
                    result = process_block(args, job, pred, ct, compute)
                finally:
                    compute.release()
                yield result
                del result
        try:
            yield serial_results()
        finally:
            compute.release()
        return
    context = multiprocessing.get_context("spawn")
    gpu_lock = context.Lock() if args.device == "cuda" else None
    cache = args.out / "portable_sheet_worker_cache"
    cache.mkdir(exist_ok=True)
    executor = None
    results = None
    try:
        with worker_warnings(context) as log_queue:
            try:
                executor = ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                                               initializer=_initialize,
                                               initargs=(args, gpu_lock, log_queue))
                results = ordered_results(executor, _run_block, jobs, args.workers)
                yield results
            finally:
                if results is not None:
                    results.close()
                # Drain active writers and their warnings before stopping the relay.
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
    finally:
        for pattern in ("block_*.npy", "block_*.npy.tmp"):
            for path in cache.glob(pattern):
                path.unlink()
        cache.rmdir()
        compute.release()


@contextmanager
def result_labels(result):
    """Close Windows mmap handles before deleting a consumed worker handoff."""
    if "labels" in result:
        yield result["labels"]
        return
    path = Path(result["labels_path"])
    labels = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        yield labels
    finally:
        labels._mmap.close()
        path.unlink()
