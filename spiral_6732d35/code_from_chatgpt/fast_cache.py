"""Drop-in replacement for BoundedSparseCudaCache.gather().

Two changes, both motivated by measurements in bench_gather.py on a 12 GB card:

1. CAPABILITY -- upstream raises RuntimeError when a *single* gather touches
   more distinct chunks than the LRU holds. Its LRU only helps across gathers;
   within one gather there is no degradation path, so a small card simply
   refuses to run. Here an oversized gather is split into spatially-coherent
   passes instead.

2. PERFORMANCE -- the stage breakdown of upstream gather() at 8M corner
   indices (fully resident) is 22.03 ms, of which the actual data lookup
   `pool[:, slots, linear]` is 0.90 ms (4%). The largest single item is
   torch.unique() over all 8M keys (5.23 ms), computed even though only the
   *missing* keys are ever needed. Here residency is tested first and unique()
   runs on the missing subset only -- empty in the steady state.

Replacing the host-side OrderedDict LRU is what makes (2) possible, so
recency is tracked in a device tensor of gather ids instead. Chunks touched by
the current gather carry the current id and are therefore never chosen as
eviction victims -- upstream gets the same guarantee from its `protected` set,
which it can only build by uniquing every key.

Output contract: bitwise-identical uint8 values to upstream for the same
access sequence (test_fast_cache.py). Which slot holds which chunk may differ
once eviction starts -- both policies evict least-recently-used and differ
only in how they break ties within one gather -- but the values read out do
not depend on slot assignment.
"""
from __future__ import annotations

import time

import torch

from sparse_cuda_cache_7769da8 import CHUNK_SIZE, BoundedSparseCudaCache

try:
    import fused_gather as _TRITON
except Exception:                                    # no triton, or no GPU
    _TRITON = None

_PROTECTED = torch.iinfo(torch.int64).max


class FastSparseCudaCache(BoundedSparseCudaCache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gz, gy, gx = self.chunk_grid_zyx
        dev = self.device
        self._grid_stride = torch.tensor([gy * gx, gx, 1], dtype=torch.int64, device=dev)
        self._origin_t = torch.tensor(self.origin_zyx, dtype=torch.int64, device=dev)
        self._shape_t = torch.tensor(self.shape_zyx, dtype=torch.int64, device=dev)
        self._table_flat = self.chunk_table.reshape(-1)
        # device-side LRU: which key each slot holds, and when it was last read
        self._slot_key = torch.full((self.capacity,), -1, dtype=torch.int64, device=dev)
        self._slot_stamp = torch.full((self.capacity,), -1, dtype=torch.int64, device=dev)
        self._gid = 0
        self._splits = 0
        self._fast_gathers = 0
        self.stamp_residents = True
        self.use_triton = _TRITON is not None
        self.single_pass = self.use_triton
        self._single_pass_hits = 0

    # ------------------------------------------------------------------
    def _decompose(self, indices_zyx: torch.Tensor):
        """(N,3) local indices -> (chunk_linear, voxel_linear).

        Same arithmetic as upstream, with the two bool(...any()) validation
        syncs folded into one.
        """
        flat = indices_zyx.detach().reshape(-1, 3).to(
            device=self.device, dtype=torch.int64)
        source = flat + self._origin_t
        if bool(((source < 0) | (source >= self._shape_t)).any()):
            raise IndexError(f"{self.label} gather received an out-of-bounds index")
        chunk = torch.div(source, CHUNK_SIZE, rounding_mode="floor")
        chunk_lin = (chunk * self._grid_stride).sum(dim=1)
        local = source - chunk * CHUNK_SIZE
        voxel_lin = (local[:, 0] * CHUNK_SIZE + local[:, 1]) * CHUNK_SIZE + local[:, 2]
        return chunk_lin, voxel_lin

    def _resolve(self, chunk_lin, slots=None, n_missing_idx=None) -> torch.Tensor:
        """Make every requested chunk resident; return each index's slot.

        `slots` / `n_missing_idx` let the fused kernel hand over what it
        already computed, so neither the table read nor the miss-count sync is
        repeated here.
        """
        self._gid += 1
        gid = self._gid
        if slots is None:
            slots = self._table_flat[chunk_lin].to(torch.int64)
        missing = slots < 0
        if n_missing_idx is None:
            n_missing_idx = int(missing.sum())       # the one unavoidable sync
        if n_missing_idx == 0:
            self._fast_gathers += 1
            self._hits += chunk_lin.numel()
            return slots

        missing_keys = torch.unique(chunk_lin[missing])
        n_new = int(missing_keys.numel())
        self._misses += n_new
        self._hits += chunk_lin.numel() - n_missing_idx
        if n_new > self.capacity:
            raise _Oversized(n_new)

        free = (self._slot_key < 0).nonzero(as_tuple=True)[0]
        n_free = int(free.numel())
        if n_free >= n_new:
            target = free[:n_new]
        else:
            need = n_new - n_free
            # Stamping every touched slot costs ~7.9 ms at 8M indices -- more
            # than the torch.unique() it replaced -- so it is deferred to the
            # only gathers that can act on it: the ones that evict. Reads on
            # non-evicting gathers therefore do not refresh recency, which
            # moves the policy from strict LRU toward least-recently-loaded.
            # Correctness is unaffected: what this stamp must guarantee is
            # that a slot needed by THIS gather is never a victim, and that
            # holds because it is written here, before the victims are picked.
            if self.stamp_residents:
                self._slot_stamp.index_fill_(0, slots.clamp(min=0), gid)
            # Victims may be neither (a) touched by this gather nor (b) one of
            # the free slots handed out above -- free slots carry the initial
            # stamp of -1, so without (b) topk would pick them a second time
            # and two keys would land on one slot.
            protect = (self._slot_stamp == gid) | (self._slot_key < 0)
            scores = torch.where(
                protect, torch.full_like(self._slot_stamp, _PROTECTED),
                self._slot_stamp)
            oldest, victims = torch.topk(scores, need, largest=False)
            if bool((oldest == _PROTECTED).any()):
                raise _Oversized(n_new)
            self._table_flat[self._slot_key[victims]] = -1
            self._evictions += need
            target = torch.cat([free, victims])

        self._table_flat[missing_keys] = target.to(torch.int32)
        self._slot_key[target] = missing_keys
        self._slot_stamp.index_fill_(0, target, gid)

        # _load_chunks is host-side I/O, so the (key, slot) pairs must come
        # back -- but only for the misses, which is the point.
        self._load_chunks(list(zip(
            [int(v) for v in missing_keys.cpu().tolist()],
            [int(v) for v in target.cpu().tolist()],
        )))
        return self._table_flat[chunk_lin].to(torch.int64)

    # ------------------------------------------------------------------
    def gather(self, indices_zyx: torch.Tensor) -> torch.Tensor:
        started = time.perf_counter()
        original_shape = tuple(indices_zyx.shape[:-1])
        if self.use_triton:
            idx = indices_zyx.detach().reshape(-1, 3)
            if idx.dtype != torch.int64 or idx.device != self.device:
                idx = idx.to(device=self.device, dtype=torch.int64)
            idx = idx.contiguous()
            if self.single_pass:
                # Optimistic: assume everything is resident and never spill the
                # addresses. Steady-state gathers take this path; a miss costs
                # one wasted pass and falls through to the two-kernel route.
                values, counters = _TRITON.resolve_gather(
                    idx, self._table_flat, self.pool,
                    self.origin_zyx, self.shape_zyx, self.chunk_grid_zyx,
                    CHUNK_SIZE)
                n_oob, n_miss = (int(v) for v in counters.cpu())
                if n_oob:
                    raise IndexError(
                        f"{self.label} gather received an out-of-bounds index")
                if n_miss == 0:
                    self._gid += 1
                    self._fast_gathers += 1
                    self._hits += idx.shape[0]
                    self._single_pass_hits += 1
                    elapsed = time.perf_counter() - started
                    self._gather_seconds += elapsed
                    self._gathers += 1
                    self.last_timings = {"gather_seconds": elapsed,
                                         "single_pass": True}
                    return values.reshape(*original_shape, self.channels)
            chunk_lin, voxel_lin, slots, counters = _TRITON.resolve(
                idx, self._table_flat,
                self.origin_zyx, self.shape_zyx, self.chunk_grid_zyx, CHUNK_SIZE)
            # one sync covers both validation and the miss count, where the
            # torch path needs a separate .any() for each
            n_oob, n_miss = (int(v) for v in counters.cpu())
            if n_oob:
                raise IndexError(
                    f"{self.label} gather received an out-of-bounds index")
            values = self._gather_resolved(
                chunk_lin, voxel_lin, slots=slots.to(torch.int64), n_miss=n_miss)
        else:
            chunk_lin, voxel_lin = self._decompose(indices_zyx)
            values = self._gather_resolved(chunk_lin, voxel_lin)
        elapsed = time.perf_counter() - started
        self._gather_seconds += elapsed
        self._gathers += 1
        self.last_timings = {
            "gather_seconds": elapsed,
            "resident_chunks": int((self._slot_key >= 0).sum()),
            "splits": self._splits,
            "fast_gathers": self._fast_gathers,
        }
        return values.reshape(*original_shape, self.channels)

    # Fraction of the pool a single pass aims to fill. Cutting at exactly
    # capacity leaves no headroom, so a pass that is even slightly less
    # coherent than average overflows and has to be split again; 0.6 costs a
    # few extra passes and removes almost all of that re-splitting.
    _FILL = 0.6

    def _gather_resolved(self, chunk_lin, voxel_lin, slots=None, n_miss=None,
                         depth: int = 0):
        """Resolve + read, splitting the request if it will not fit at once."""
        try:
            slots = self._resolve(chunk_lin, slots, n_miss)
        except _Oversized as exc:
            n = chunk_lin.numel()
            if depth > 12 or n < 2:
                raise RuntimeError(
                    f"{self.label}: {exc.count} distinct chunks in an "
                    f"indivisible request exceed the {self.capacity}-slot pool"
                ) from None
            # Callers hand us points ordered along the fitted surface, so an
            # index-space cut is also a spatial cut. Blind halving splits into
            # 2^k pieces and reloads the same chunks over and over; size the
            # pieces from the measured overflow instead, so a request that is
            # 4x too big becomes ~7 passes rather than ~64.
            pieces = max(2, -(-exc.count // max(1, int(self.capacity * self._FILL))))
            pieces = min(pieces, n)
            step = -(-n // pieces)
            self._splits += pieces - 1
            # sub-slices need their own table read: the parent's `slots` are
            # stale the moment the first pass evicts anything
            out = [
                self._gather_resolved(
                    chunk_lin[lo:lo + step], voxel_lin[lo:lo + step],
                    depth=depth + 1)
                for lo in range(0, n, step)
            ]
            return torch.cat(out, dim=0)
        if bool((slots < 0).any()):
            raise RuntimeError(f"{self.label} cache miss remained after loading")
        if self.use_triton:
            # re-reads the table itself, so it is correct after a load
            return _TRITON.gather(
                self.pool, self._table_flat, chunk_lin, voxel_lin)
        return self.pool[:, slots, voxel_lin].transpose(0, 1)

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        out = super().stats()
        out["resident_chunks"] = int((self._slot_key >= 0).sum())
        out["splits"] = self._splits
        out["fast_gathers"] = self._fast_gathers
        out["single_pass_hits"] = self._single_pass_hits
        return out


class _Oversized(Exception):
    def __init__(self, count: int):
        super().__init__(count)
        self.count = count
