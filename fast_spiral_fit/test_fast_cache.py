"""Equivalence + capability tests for FastSparseCudaCache.

The contract being tested is exact: for the same access sequence the fast
cache must return uint8 values bitwise identical to upstream. Nothing here
touches scroll data -- a small synthetic zarr is built per run.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

SPIRAL_DIR = Path("/root/villa/volume-cartographer/scripts/spiral")
sys.path.insert(0, str(SPIRAL_DIR))
sys.path.insert(0, "/root")

import fast_cache  # noqa: E402
from sparse_cuda_cache import CHUNK_SIZE, BoundedSparseCudaCache  # noqa: E402
from fast_cache import FastSparseCudaCache  # noqa: E402

SHAPE = (256, 256, 256)          # 8^3 = 512 chunks
GIB = 1024 ** 3


def build(tmp: Path, name: str, channels: int, seed: int) -> list[str]:
    import zarr

    paths = []
    rng = np.random.default_rng(seed)
    for c in range(channels):
        p = tmp / name / f"ch{c}"
        z = zarr.create_array(
            store=str(p), shape=SHAPE, chunks=(CHUNK_SIZE,) * 3,
            dtype="uint8", zarr_format=2)
        # values must be distinctive per voxel, or a slot mix-up would not show
        z[:] = rng.integers(1, 256, SHAPE, dtype=np.uint8)
        paths.append(str(p))
    return paths


def spiral_indices(n: int, phase: float) -> torch.Tensor:
    """Points along a spiral sheet -- index-adjacent means space-adjacent,
    which is the ordering the split path relies on."""
    t = np.linspace(0, 2 * np.pi * 6, n) + phase
    r = np.linspace(10, 120, n)
    z = np.linspace(2, SHAPE[0] - 3, n)
    pts = np.stack([z, 128 + r * np.cos(t), 128 + r * np.sin(t)], -1)
    return torch.from_numpy(
        np.clip(pts, 0, np.array(SHAPE) - 1).astype(np.int64)).cuda()


def corners(pts: torch.Tensor) -> torch.Tensor:
    """8 trilinear corners per point, as sdt_losses.sample_sdt_trilinear does."""
    off = torch.tensor([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)],
                       dtype=torch.int64, device=pts.device)
    hi = torch.tensor(SHAPE, dtype=torch.int64, device=pts.device) - 1
    return torch.minimum((pts[:, None, :] + off[None]).clamp(min=0), hi).reshape(-1, 3)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class FastCacheEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.sdt = build(cls.tmp, "sdt", 1, seed=7)
        cls.nrm = build(cls.tmp, "normals", 2, seed=99)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _pair(self, paths, budget_gib):
        kw = dict(source_paths=paths, shape_zyx=SHAPE,
                  budget_bytes=int(budget_gib * GIB), device="cuda")
        return (BoundedSparseCudaCache(label="ref", **kw),
                FastSparseCudaCache(label="fast", **kw))

    def _run_sequence(self, paths, budget_gib, steps=8, n=20_000, tweak=None):
        ref, fast = self._pair(paths, budget_gib)
        if tweak is not None:
            tweak(fast)
        try:
            for s in range(steps):
                idx = corners(spiral_indices(n, s * 0.3))
                a = ref.gather(idx)
                b = fast.gather(idx)
                self.assertEqual(a.shape, b.shape)
                self.assertEqual(a.dtype, b.dtype)
                self.assertTrue(torch.equal(a, b),
                                f"step {s}: {int((a != b).sum())} voxels differ")
            return fast.stats()
        finally:
            ref.close()
            fast.close()

    # -- equivalence ---------------------------------------------------
    def test_identical_when_everything_fits(self):
        st = self._run_sequence(self.sdt, budget_gib=0.05)
        self.assertEqual(st["splits"], 0)

    def test_repeated_indices_take_the_no_miss_fast_path(self):
        """A re-read of an already-resident set must skip planning entirely."""
        ref, fast = self._pair(self.sdt, 0.05)
        try:
            idx = corners(spiral_indices(20_000, 0.0))
            for _ in range(3):
                self.assertTrue(torch.equal(ref.gather(idx), fast.gather(idx)))
            self.assertEqual(fast.stats()["fast_gathers"], 2,
                             "re-reads should not have planned")
        finally:
            ref.close()
            fast.close()

    def test_identical_multichannel(self):
        self._run_sequence(self.nrm, budget_gib=0.05)

    def test_identical_under_eviction(self):
        """Budget below the working set: both caches must still agree."""
        st = self._run_sequence(self.sdt, budget_gib=0.004)
        self.assertGreater(st["evictions"], 0, "test did not exercise eviction")

    # Every variant the benchmark prices must hold the same contract as the
    # default path -- a wrong ablation number is as misleading as a wrong
    # headline number.
    # 0.008 GiB on the 2-channel volume is 131 slots -- big enough for the
    # ~112-chunk step that upstream must also survive, small enough to evict.
    @unittest.skipUnless(fast_cache._TRITON, "triton unavailable")
    def test_identical_two_pass_variant(self):
        st = self._run_sequence(
            self.nrm, budget_gib=0.008,
            tweak=lambda f: setattr(f, "single_pass", False))
        self.assertGreater(st["evictions"], 0)
        self.assertEqual(st["single_pass_hits"], 0)

    def test_identical_torch_path_variant(self):
        self._run_sequence(
            self.nrm, budget_gib=0.008,
            tweak=lambda f: setattr(f, "use_triton", False))

    def test_random_access_identical(self):
        """Worst case for the split: random points have no spatial locality,
        so halving the index array barely shrinks the distinct-chunk count.
        Correctness must hold anyway; the cost shows up as split count."""
        g = torch.Generator(device="cuda").manual_seed(3)
        idx = torch.randint(0, SHAPE[0], (5_000, 3), dtype=torch.int64,
                            device="cuda", generator=g)

        big = BoundedSparseCudaCache(
            source_paths=self.sdt, shape_zyx=SHAPE,
            budget_bytes=int(0.05 * GIB), device="cuda", label="truth")
        truth = big.gather(idx).clone()
        big.close()

        fast = FastSparseCudaCache(
            source_paths=self.sdt, shape_zyx=SHAPE,
            budget_bytes=64 * CHUNK_SIZE ** 3, device="cuda", label="fast-rand")
        try:
            self.assertTrue(torch.equal(truth, fast.gather(idx)))
            print(f"\n      [random access] splits={fast.stats()['splits']} "
                  f"misses={fast.stats()['misses']}", end=" ")
        finally:
            fast.close()

    # -- capability ----------------------------------------------------
    def test_upstream_refuses_oversized_gather_and_fast_does_not(self):
        """The point of the change: a request bigger than the pool.

        Ground truth is upstream with a pool large enough to hold everything;
        the fast cache must reproduce it from a pool that cannot.
        """
        idx = corners(spiral_indices(60_000, 0.0))

        big = BoundedSparseCudaCache(
            source_paths=self.sdt, shape_zyx=SHAPE,
            budget_bytes=int(0.05 * GIB), device="cuda", label="truth")
        truth = big.gather(idx).clone()
        big.close()

        tiny = dict(source_paths=self.sdt, shape_zyx=SHAPE,
                    budget_bytes=64 * CHUNK_SIZE ** 3, device="cuda")
        ref = BoundedSparseCudaCache(label="ref-tiny", **tiny)
        with self.assertRaises(RuntimeError):
            ref.gather(idx)
        ref.close()

        fast = FastSparseCudaCache(label="fast-tiny", **tiny)
        try:
            got = fast.gather(idx)
            self.assertTrue(torch.equal(truth, got))
            self.assertGreater(fast.stats()["splits"], 0)
        finally:
            fast.close()

    def test_out_of_bounds_still_raises(self):
        fast = FastSparseCudaCache(
            source_paths=self.sdt, shape_zyx=SHAPE,
            budget_bytes=int(0.01 * GIB), device="cuda", label="oob")
        try:
            bad = torch.tensor([[0, 0, SHAPE[2]]], dtype=torch.int64, device="cuda")
            with self.assertRaises(IndexError):
                fast.gather(bad)
        finally:
            fast.close()

    @unittest.skipUnless(fast_cache._TRITON, "triton unavailable")
    def test_big_pool_channel1_addressing(self):
        """Pools past 2 GiB per channel: capacity * slot_stride overflows i32.

        The kernels take the per-channel plane as a host-computed integer for
        exactly this case; an in-kernel i32 product wraps at slot 65536 x 32^3
        and every channel-1 read lands outside the pool. capacity is clamped to
        the volume's total chunk count, so triggering it needs a volume of
        >= 65536 chunks -- built here as zarr metadata plus 4 written chunks,
        so it costs KB of disk, not GiB.
        """
        free, _ = torch.cuda.mem_get_info()
        if free < 5.5 * GIB:
            self.skipTest(f"needs 5.5 GiB free VRAM, have {free / GIB:.1f}")

        import zarr

        big_shape = (2048, 1024, 1024)               # 64*32*32 = 65536 chunks
        rng = np.random.default_rng(11)
        paths = []
        for c in range(2):
            p = self.tmp / "bigpool" / f"ch{c}"
            z = zarr.create_array(
                store=str(p), shape=big_shape, chunks=(CHUNK_SIZE,) * 3,
                dtype="uint8", zarr_format=2)
            z[:32, :64, :64] = rng.integers(1, 256, (32, 64, 64), dtype=np.uint8)
            paths.append(str(p))

        g = torch.Generator(device="cuda").manual_seed(5)
        idx = torch.stack([
            torch.randint(0, 32, (30_000,), device="cuda", generator=g),
            torch.randint(0, 64, (30_000,), device="cuda", generator=g),
            torch.randint(0, 64, (30_000,), device="cuda", generator=g),
        ], -1).to(torch.int64)

        ref = BoundedSparseCudaCache(
            source_paths=paths, shape_zyx=big_shape,
            budget_bytes=int(0.05 * GIB), device="cuda", label="big-ref")
        truth = ref.gather(idx).clone()
        ref.close()
        self.assertGreater(int(truth[:, 1].long().sum()), 0,
                           "channel 1 read all zeros; test data is broken")

        fast = FastSparseCudaCache(
            source_paths=paths, shape_zyx=big_shape,
            budget_bytes=65536 * 2 * CHUNK_SIZE ** 3,   # 65536 slots = 2^31 plane
            device="cuda", label="big-fast")
        try:
            self.assertEqual(fast.capacity, 65536)
            # miss path reads through _gather_kernel, resident re-read through
            # _resolve_gather_kernel -- both do channel-1 addressing
            self.assertTrue(torch.equal(truth, fast.gather(idx)))
            self.assertTrue(torch.equal(truth, fast.gather(idx)))
            self.assertGreater(fast.stats()["single_pass_hits"], 0)
        finally:
            fast.close()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main(verbosity=2)
