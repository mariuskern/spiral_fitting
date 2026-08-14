"""Run the official spiral fit with FastSparseCudaCache swapped in.

Two things this has to do without editing anything in the villa checkout, so
that `git diff` stays empty and the A/B is a one-flag change:

1. CACHE SWAP. Upstream's three pool construction sites (lasagna_data.py:
   normals, grad_mag, surf_sdt) each do a function-level
   `from sparse_cuda_cache import BoundedSparseCudaCache`, so rebinding the
   attribute on that module before the fit starts takes over every pool.

2. CONFIG. fit_spiral keeps dataset_path and the z-range as module-level
   literals and derives every other input path from them at import time --
   the tutorial tells users to edit the file. Importing it first and
   reassigning afterwards is therefore not enough: the derived paths are
   already built. So the source is read, two exact assignments are rewritten
   in memory, and the result is exec'd as __main__. Both substitutions assert
   their target line was found, so an upstream rename fails loudly here
   instead of silently fitting the wrong volume.

Usage:
    FIT_USE_FAST_CACHE=0 python run_fit.py    # upstream pools, for the baseline
    FIT_USE_FAST_CACHE=1 python run_fit.py    # ours (default)

    FIT_DATASET   dataset root (default /root/spiral-dataset/PHercParis4)
    FIT_Z_BEGIN   first z slice, full-res voxels (default 4000)
    FIT_Z_END     last z slice (default 5000 -- the tutorial's "small first
                  run"; the whole written region is 4000-17000 and needs about
                  60 GB on upstream, which is what this project attacks)

Upstream's own knobs still work untouched: FIT_SPIRAL_CONFIG_OVERRIDES,
FIT_SPIRAL_SPARSE_{NORMAL,GRAD,SDT}_CACHE_GB, FIT_SPIRAL_CACHE_DIR, ...
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


from dotenv import load_dotenv
load_dotenv()
for key in ["DATA_DIR", "CODE_DIR", "OUTPUT_DIR"]:
    if key in os.environ:
        os.environ[key] = os.path.expandvars(os.environ[key])


SPIRAL = Path("villa")
sys.path.insert(0, str(SPIRAL))
sys.path.insert(0, "/root")

DATASET = os.environ.get("FIT_DATASET", "/root/spiral-dataset/PHercParis4")
Z_BEGIN = int(os.environ.get("FIT_Z_BEGIN", "4000"))
Z_END = int(os.environ.get("FIT_Z_END", "5000"))
USE_FAST = os.environ.get("FIT_USE_FAST_CACHE", "1") != "0"
USE_FAST_LINK = os.environ.get("FIT_USE_FAST_LINK", "1") != "0"


def _substitute(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(
            f"run_fit.py expected exactly one occurrence of {old!r} in "
            f"fit_spiral.py, found {source.count(old)}. Upstream changed its "
            f"config block; update run_fit.py rather than guessing.")
    return source.replace(old, new)


def _install_fast_cache() -> None:
    import sparse_cuda_cache

    from fast_cache import FastSparseCudaCache

    if FastSparseCudaCache.__mro__[1] is not sparse_cuda_cache.BoundedSparseCudaCache:
        raise RuntimeError(
            "fast_cache subclasses a different BoundedSparseCudaCache than the "
            "one fit_spiral will import -- check sys.path ordering")
    sparse_cuda_cache.BoundedSparseCudaCache = FastSparseCudaCache


def _install_fast_link() -> None:
    """Prefilter the point->patch linking fallback (see fast_link.py).

    Only matters when the native vc.surface_index extension is absent, which is
    the normal state without a VC3D CMake build -- upstream then brute-forces
    every point against every patch. Left installed regardless: with the native
    index present this code is never reached, so it cannot change that path.
    """
    import fast_link
    fast_link.install()


def main() -> None:
    target = SPIRAL / "fit_spiral.py"
    source = target.read_text(encoding="utf-8")
    source = _substitute(
        source,
        "dataset_path = '/ephemeral/paul/spiral/dataset'",
        f"dataset_path = {DATASET!r}")
    source = _substitute(
        source,
        "z_begin, z_end = 4000, 17000",
        f"z_begin, z_end = {Z_BEGIN}, {Z_END}")

    # 3. PERIODIC CHECKPOINT. The batch path only saves at the very end
    #    (save_model('fitted') after the loop); periodic saving exists solely
    #    behind the interactive driver. A 30k-step run on a laptop needs to
    #    survive interruption, so hook the existing atomic save_model_to into
    #    the loop's logging site. Resume with
    #    FIT_SPIRAL_RESUME_PATH=<out>/checkpoint_periodic.ckpt -- the embedded
    #    completed_iterations restores the step, scheduler, and RNG states.
    source = _substitute(
        source,
        "        if iteration % 200 == 0:",
        "        if (is_main_process() and iteration > start_iteration\n"
        "                and (iteration + 1) % 1000 == 0):\n"
        "            save_model_to(f'{out_path}/checkpoint_periodic.ckpt',\n"
        "                          iteration + 1)\n"
        "        if iteration % 200 == 0:")

    if USE_FAST:
        _install_fast_cache()
    if USE_FAST_LINK:
        _install_fast_link()

    print(f"[run_fit] cache   : "
          f"{'FastSparseCudaCache' if USE_FAST else 'upstream BoundedSparseCudaCache'}")
    print(f"[run_fit] linking : "
          f"{'prefiltered' if USE_FAST_LINK else 'upstream brute force'}")
    print(f"[run_fit] dataset : {DATASET}")
    print(f"[run_fit] z-range : {Z_BEGIN}-{Z_END} ({Z_END - Z_BEGIN} slices)")
    for var in ("FIT_SPIRAL_SPARSE_NORMAL_CACHE_GB",
                "FIT_SPIRAL_SPARSE_GRAD_CACHE_GB",
                "FIT_SPIRAL_SPARSE_SDT_CACHE_GB",
                "FIT_SPIRAL_CONFIG_OVERRIDES",
                "FIT_SPIRAL_CACHE_DIR",
                "FIT_SPIRAL_OUT_DIR",
                "FIT_SPIRAL_RESUME_PATH"):
        if var in os.environ:
            print(f"[run_fit] {var}={os.environ[var]}")
    sys.stdout.flush()

    # cwd matters: fit_spiral's default cache dir is the relative '../cache'
    os.chdir(SPIRAL)
    code = compile(source, str(target), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(target)})


if __name__ == "__main__":
    main()
