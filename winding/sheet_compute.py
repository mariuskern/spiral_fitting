"""Portable sheet pipeline: CPU/CUDA backend with bounded tensor workspaces.

The original structure_normals allocates a full-block tensor. This extension
uses the same tensor formula and original filter constants, with exact Gaussian
halos and configurable eigensolver batches on CPU and CUDA. It needs no cuCIM.
"""

import logging
import time

import numpy as np
from scipy import ndimage

from vesuvius_sheet_tools.split_stacked import SIGMA_GRAD, SIGMA_TENSOR


logger = logging.getLogger("portable_sheet")


def tensor_normals(eq, xp=np, ndi=ndimage, depth=16, eigh_chunk=500_000):
    """Evaluate the same structure tensor in slabs with exact Gaussian halos.

    Only the returned normals/planarity occupy a full host block. Gradients,
    tensors and eigensolver workspaces are bounded by depth/eigh_chunk.
    """
    halo = int(4 * SIGMA_GRAD + 0.5) + int(4 * SIGMA_TENSOR + 0.5)
    normals = np.empty(eq.shape + (3,), dtype=np.float32)
    planarity = np.empty(eq.shape, dtype=np.float32)
    to_host = np.asarray if xp is np else xp.asnumpy
    for z0 in range(0, eq.shape[0], depth):
        z1 = min(z0 + depth, eq.shape[0])
        lo, hi = max(0, z0 - halo), min(eq.shape[0], z1 + halo)
        vol = xp.asarray(eq[lo:hi], dtype=xp.float32)
        grads = [ndi.gaussian_filter(vol, SIGMA_GRAD,
                 order=tuple(int(i == axis) for i in range(3))) for axis in range(3)]
        del vol
        tensor = xp.empty((z1 - z0, *eq.shape[1:], 3, 3), dtype=xp.float32)
        for i in range(3):
            for j in range(i, 3):
                filtered = ndi.gaussian_filter(grads[i] * grads[j], SIGMA_TENSOR)
                tensor[..., i, j] = filtered[z0 - lo:z1 - lo]
                tensor[..., j, i] = tensor[..., i, j]
                del filtered
        del grads
        flat = tensor.reshape(-1, 3, 3)
        out_n = normals[z0:z1].reshape(-1, 3)
        out_p = planarity[z0:z1].reshape(-1)
        for start in range(0, len(flat), eigh_chunk):
            stop = min(start + eigh_chunk, len(flat))
            if xp is np:
                w, v = xp.linalg.eigh(flat[start:stop])
            else:
                # CuPy otherwise ignores some cuSOLVER convergence errors.
                from cupyx import errstate
                with errstate(linalg="raise"):
                    w, v = xp.linalg.eigh(flat[start:stop])
            out_n[start:stop] = to_host(v[..., :, 2])
            out_p[start:stop] = to_host((w[..., 2] - w[..., 1]) / (w[..., 2] + 1e-6))
            del w, v
        del flat, tensor
    return normals, planarity


class Compute:
    """Explicit CUDA selection; only GPU out-of-memory may fall back to CPU."""

    def __init__(self, device, tensor_depth=16, eigh_chunk=500_000, oom_policy="cpu",
                 gpu_lock=None):
        self.device = device
        self.tensor_depth, self.eigh_chunk = tensor_depth, eigh_chunk
        self.oom_policy = oom_policy
        self.fallbacks = []
        self.gpu_lock = gpu_lock
        self.gpu_wait_seconds = 0.0
        if device == "cuda":
            try:
                import cupy as cp
                import cupyx.scipy.ndimage as cndi
                # CuPy loads cuSOLVER lazily; check it before starting workers
                # rather than failing at the first structure-tensor eigh call.
                from cupy.cuda import cusolver
                from cupyx.scipy.ndimage import distance_transform_edt

                if cp.cuda.runtime.getDeviceCount() < 1:
                    raise RuntimeError("No CUDA device found")
                self.cp, self.cndi, self.gpu_edt = cp, cndi, distance_transform_edt
            except (ImportError, OSError, RuntimeError) as exc:
                raise ValueError(
                    f"CUDA initialization failed ({type(exc).__name__}: {exc}). "
                    "Install CuPy >= 14 with its CUDA libraries in this Python environment "
                    "(e.g. uv pip install 'cupy-cuda12x[ctk]>=14' for CUDA 12.x), "
                    "and run on a GPU node with a compatible NVIDIA driver, "
                    "or use --device cpu. cuCIM is not required."
                ) from exc

    def _execute(self, stage, cpu, gpu):
        if self.device == "cpu":
            return cpu()
        if self.gpu_lock is None:
            success, result = self._try_gpu(stage, gpu)
        else:
            started = time.monotonic()
            with self.gpu_lock:
                self.gpu_wait_seconds += time.monotonic() - started
                try:
                    success, result = self._try_gpu(stage, gpu)
                finally:
                    # All GPU work must finish before another process acquires
                    # the device. Do not retain a separate large cache per worker.
                    try:
                        self.cp.cuda.get_current_stream().synchronize()
                    finally:
                        self.release()
        if success:
            return result
        # CPU recovery must not hold the shared GPU lock.
        self.fallbacks.append(stage)
        logger.warning("CUDA out of memory during %s; executing this stage on CPU.", stage)
        return cpu()

    def _try_gpu(self, stage, gpu):
        try:
            return True, gpu()
        except self.cp.cuda.memory.OutOfMemoryError:
            self.release()
            if self.oom_policy == "error":
                raise ValueError(f"CUDA out of memory during {stage}; reduce --block-shape "
                                 "or use --oom-policy cpu") from None
        # Outside the except block: do not retain failed GPU arrays via traceback.
        self.release()
        return False, None

    @staticmethod
    def _clean(mask, min_cc, xp, ndi):
        components, _ = ndi.label(mask)
        keep = xp.bincount(components.ravel()) >= min_cc
        keep[0] = False
        return keep[components]

    def clean(self, mask, min_cc):
        return self._execute("component filtering",
            lambda: self._clean(mask, min_cc, np, ndimage),
            lambda: self.cp.asnumpy(self._clean(self.cp.asarray(mask), min_cc,
                                                self.cp, self.cndi)))

    def edt(self, mask):
        # SciPy's no-background convention differs from the GPU EDT's convention.
        if mask.all():
            return ndimage.distance_transform_edt(mask).astype(np.float32)
        return self._execute("distance transform",
            lambda: ndimage.distance_transform_edt(mask).astype(np.float32),
            lambda: self.cp.asnumpy(self.gpu_edt(self.cp.asarray(mask),
                                                float64_distances=False)))

    def seeds(self, mask, dist, core_dist, min_core):
        def prepare(xp, ndi):
            m, d = xp.asarray(mask), xp.asarray(dist)
            cores, _ = ndi.label((d >= core_dist) & m)
            keep = xp.bincount(cores.ravel()) >= min_core
            keep[0] = False
            markers, _ = ndi.label(keep[cores])
            del cores, keep, d
            components, n = ndi.label(m)
            if xp is not np:
                markers, components = xp.asnumpy(markers), xp.asnumpy(components)
            return markers, components, int(n)
        return self._execute("watershed seeds", lambda: prepare(np, ndimage),
                             lambda: prepare(self.cp, self.cndi))

    def structure_normals(self, eq):
        def gpu():
            depth = self.tensor_depth
            while True:
                try:
                    return tensor_normals(eq, self.cp, self.cndi, depth, self.eigh_chunk)
                except self.cp.cuda.memory.OutOfMemoryError:
                    if depth == 1:
                        raise
                self.release()
                depth = max(1, depth // 2)
                logger.warning("Reducing CUDA tensor slab depth to %s after out-of-memory.", depth)
        return self._execute("structure tensor",
            lambda: tensor_normals(eq, depth=self.tensor_depth, eigh_chunk=self.eigh_chunk), gpu)

    def release(self):
        if self.device == "cuda":
            self.cp.get_default_memory_pool().free_all_blocks()
