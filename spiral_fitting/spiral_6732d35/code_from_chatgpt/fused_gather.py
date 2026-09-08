"""Fused Triton path for the sparse chunk gather.

The stage breakdown of upstream gather() at 8M corner indices, fully resident,
was 22.03 ms -- of which the actual pool read was 0.90 ms (4%). The rest is a
chain of ~10 elementwise int64 kernels, each materialising a 192 MB
intermediate, to compute an address.

Two kernels replace that chain:

  resolve  idx -> (chunk_linear, voxel_linear, slot), plus counters for
           out-of-bounds and missing slots. One pass, results in int32.
  gather   (chunk_linear, voxel_linear) -> values. Re-reads the chunk table
           itself, so it is correct after a load has filled in the misses --
           no second host round-trip to rebuild slots.

Both take the same (N, 3) int64 input the caller already has, so nothing is
transposed or copied to feed them.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _resolve_kernel(
    idx_ptr, table_ptr,
    chunk_out_ptr, voxel_out_ptr, slot_out_ptr,
    oob_ptr, miss_ptr,
    n_elem,
    oz, oy, ox,
    sz, sy, sx,
    gy, gx,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    base = offs * 3
    z = tl.load(idx_ptr + base + 0, mask=mask, other=0) + oz
    y = tl.load(idx_ptr + base + 1, mask=mask, other=0) + oy
    x = tl.load(idx_ptr + base + 2, mask=mask, other=0) + ox

    oob = (z < 0) | (z >= sz) | (y < 0) | (y >= sy) | (x < 0) | (x >= sx)
    # clamp so the address arithmetic below stays in range; the caller raises
    # on the counter, it does not rely on these values
    z = tl.maximum(tl.minimum(z, sz - 1), 0)
    y = tl.maximum(tl.minimum(y, sy - 1), 0)
    x = tl.maximum(tl.minimum(x, sx - 1), 0)

    cz = z // CHUNK
    cy = y // CHUNK
    cx = x // CHUNK
    chunk_lin = (cz * gy + cy) * gx + cx
    voxel_lin = ((z - cz * CHUNK) * CHUNK + (y - cy * CHUNK)) * CHUNK + (x - cx * CHUNK)
    slot = tl.load(table_ptr + chunk_lin, mask=mask, other=-1)

    # chunk_lin stays int64: it indexes torch tensors on the miss path, and
    # the extra 32 MB of writes at 8M indices is far cheaper than the casts
    # and copies that an int32 index would force there.
    tl.store(chunk_out_ptr + offs, chunk_lin, mask=mask)
    tl.store(voxel_out_ptr + offs, voxel_lin.to(tl.int32), mask=mask)
    tl.store(slot_out_ptr + offs, slot, mask=mask)

    n_oob = tl.sum((oob & mask).to(tl.int32))
    n_miss = tl.sum(((slot < 0) & mask).to(tl.int32))
    if n_oob > 0:
        tl.atomic_add(oob_ptr, n_oob)
    if n_miss > 0:
        tl.atomic_add(miss_ptr, n_miss)


@triton.jit
def _gather_kernel(
    pool_ptr, table_ptr, chunk_ptr, voxel_ptr, out_ptr,
    n_elem, plane, slot_stride,
    CHANNELS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # `plane` (= capacity * slot_stride) is computed on the host: as an in-kernel
    # product of two i32 scalars it wraps once the pool passes 2 GiB per channel,
    # which is exactly the size this cache runs at on the real card. A host int
    # above int32 range makes Triton type the argument i64.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    chunk_lin = tl.load(chunk_ptr + offs, mask=mask, other=0).to(tl.int64)
    voxel_lin = tl.load(voxel_ptr + offs, mask=mask, other=0).to(tl.int64)
    slot = tl.load(table_ptr + chunk_lin, mask=mask, other=0).to(tl.int64)
    inner = slot * slot_stride + voxel_lin
    for c in tl.static_range(CHANNELS):
        v = tl.load(pool_ptr + c * plane + inner, mask=mask, other=0)
        tl.store(out_ptr + offs * CHANNELS + c, v, mask=mask)


@triton.jit
def _resolve_gather_kernel(
    idx_ptr, table_ptr, pool_ptr, out_ptr,
    oob_ptr, miss_ptr,
    n_elem, plane, slot_stride,
    oz, oy, ox,
    sz, sy, sx,
    gy, gx,
    CHUNK: tl.constexpr,
    CHANNELS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Single-pass address+read for the fully-resident case.

    The two-kernel path has to spill chunk_lin (int64), voxel_lin and slot to
    memory between the passes -- 128 MB of writes at 8M indices, for values
    that are consumed once. When nothing is missing there is no host work to
    do in between, so the address never has to leave registers. Misses are
    still counted, and the caller falls back to the two-kernel path when the
    count is nonzero.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    base = offs * 3
    z = tl.load(idx_ptr + base + 0, mask=mask, other=0) + oz
    y = tl.load(idx_ptr + base + 1, mask=mask, other=0) + oy
    x = tl.load(idx_ptr + base + 2, mask=mask, other=0) + ox

    oob = (z < 0) | (z >= sz) | (y < 0) | (y >= sy) | (x < 0) | (x >= sx)
    z = tl.maximum(tl.minimum(z, sz - 1), 0)
    y = tl.maximum(tl.minimum(y, sy - 1), 0)
    x = tl.maximum(tl.minimum(x, sx - 1), 0)

    cz = z // CHUNK
    cy = y // CHUNK
    cx = x // CHUNK
    slot = tl.load(table_ptr + ((cz * gy + cy) * gx + cx), mask=mask, other=-1)
    voxel = ((z - cz * CHUNK) * CHUNK + (y - cy * CHUNK)) * CHUNK + (x - cx * CHUNK)

    n_oob = tl.sum((oob & mask).to(tl.int32))
    n_miss = tl.sum(((slot < 0) & mask).to(tl.int32))
    if n_oob > 0:
        tl.atomic_add(oob_ptr, n_oob)
    if n_miss > 0:
        tl.atomic_add(miss_ptr, n_miss)

    # a missing slot would index out of the pool; clamp and let the caller
    # discard the whole result on the counter
    inner = tl.maximum(slot, 0).to(tl.int64) * slot_stride + voxel
    ok = mask & (slot >= 0)
    for c in tl.static_range(CHANNELS):
        v = tl.load(pool_ptr + c * plane + inner, mask=ok, other=0)
        tl.store(out_ptr + offs * CHANNELS + c, v, mask=mask)


_BLOCK = 1024


def resolve(idx, table_flat, origin, shape, grid, chunk_size):
    """(N,3) int64 -> (chunk_lin int32, voxel_lin int32, slot int32, oob, miss)."""
    n = idx.shape[0]
    dev = idx.device
    chunk_out = torch.empty(n, dtype=torch.int64, device=dev)
    voxel_out = torch.empty(n, dtype=torch.int32, device=dev)
    slot_out = torch.empty(n, dtype=torch.int32, device=dev)
    counters = torch.zeros(2, dtype=torch.int32, device=dev)
    grid_launch = (triton.cdiv(n, _BLOCK),)
    _resolve_kernel[grid_launch](
        idx, table_flat, chunk_out, voxel_out, slot_out,
        # slices, not `counters + 1`: the latter is elementwise arithmetic
        # producing a throwaway tensor, so the miss count would be written to
        # memory nobody reads and every gather would look fully resident
        counters[0:1], counters[1:2],
        n,
        origin[0], origin[1], origin[2],
        shape[0], shape[1], shape[2],
        grid[1], grid[2],
        CHUNK=chunk_size, BLOCK=_BLOCK,
    )
    return chunk_out, voxel_out, slot_out, counters


def resolve_gather(idx, table_flat, pool, origin, shape, grid, chunk_size):
    """Single-pass attempt: returns (values, counters).

    values are only valid when counters[1] (the miss count) is zero; the
    caller checks and falls back to resolve()+load()+gather().
    """
    n = idx.shape[0]
    channels, capacity, slot_stride = pool.shape
    out = torch.empty((n, channels), dtype=torch.uint8, device=idx.device)
    counters = torch.zeros(2, dtype=torch.int32, device=idx.device)
    _resolve_gather_kernel[(triton.cdiv(n, _BLOCK),)](
        idx, table_flat, pool, out,
        counters[0:1], counters[1:2],
        n, capacity * slot_stride, slot_stride,
        origin[0], origin[1], origin[2],
        shape[0], shape[1], shape[2],
        grid[1], grid[2],
        CHUNK=chunk_size, CHANNELS=channels, BLOCK=_BLOCK,
    )
    return out, counters


def gather(pool, table_flat, chunk_lin, voxel_lin):
    """pool (C, capacity, slot_stride) uint8 -> (N, C) uint8."""
    channels, capacity, slot_stride = pool.shape
    n = chunk_lin.numel()
    out = torch.empty((n, channels), dtype=torch.uint8, device=pool.device)
    _gather_kernel[(triton.cdiv(n, _BLOCK),)](
        pool, table_flat, chunk_lin, voxel_lin, out,
        n, capacity * slot_stride, slot_stride,
        CHANNELS=channels, BLOCK=_BLOCK,
    )
    return out
