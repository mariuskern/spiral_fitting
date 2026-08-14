# Consumer-GPU spiral fitting for PHerc. Paris 4: reproduction guide

Three self-contained additions to the official `villa` spiral-fitting pipeline,
none of which modifies a single upstream source line:

| # | Component | Claim | Evidence |
|---|-----------|-------|----------|
| 1 | `fast_cache.py` + `fused_gather.py` | Sparse GPU cache that runs on small pools and splits oversized gathers instead of refusing; 3.5x faster gather | 10/10 bitwise-equality tests (`test_fast_cache.py`, log in `results/`) |
| 2 | `fast_link.py` | Point->patch linking without the native `vc.surface_index` C++ extension: a measured 41-hour ETA -> minutes (44-67x on the test sample) | link-for-link identical output (`test_fast_link.py`, log in `results/`) |
| 3 | `run_fit.py` | Periodic checkpointing + resume for batch fits (upstream saves only at the very end) | resumed runs restore step, LR schedule, RNG states via upstream's own loader |

Headline numbers, all measured on one RTX 4080 Laptop 12 GB between 2026-07-29
and -31. The full ladder, the pool-budget sweep and the scope notes are in
`README.md`:

- z 4000-5000 (1,000 slices), 30,000 steps: **43 min 37 s, 5.2 GB VRAM.**
  Upstream cache, same everything: 47 min 17 s. At this scale upstream works,
  and the margin is 8.4% from one paired run.
- z 4000-8000 (4,000 slices), 30,000 steps, single configuration:
  **4 h 55 min 40 s, 8.2 GB VRAM, 21.5 GB peak RSS**, verified-patch
  satisfaction median 0.984.
- z 4000-12000 (8,000 slices): ours reaches the training loop; the upstream
  cache raises `RuntimeError` before step 0 (`results/upstream_refusal_8000slices.txt`).
- z 4000-17000 (13,000 slices): stopped by host RAM, which demanded >124 GB
  while only 6.9 GB of VRAM was in use. README section 3 has the detail.

## Environment

- Windows 11 + WSL2 Ubuntu 22.04, NVIDIA driver 592.82. Nothing below is
  WSL-specific except the memory cap note at the end; on native Linux the same
  steps apply without `.wslconfig`.
- GPU: RTX 4080 Laptop, 12,282 MiB VRAM
- Python 3.14.6, torch 2.12.1+cu130, triton 3.7.1, numpy 2.5.0, zarr 3.2.1,
  all resolved by `uv sync` from the spiral scripts' own lockfile
- villa @ `7769da8cf2233310570608feecc127066a7c0c7c`
  (github.com/ScrollPrize/villa)

## Setup

    git clone https://github.com/ScrollPrize/villa && cd villa
    git checkout 7769da8
    cd volume-cartographer/scripts/spiral && uv sync   # creates .venv with pinned deps

Note the directory: the fit's Python dependencies live in the *spiral
scripts'* project (`scripts/spiral/pyproject.toml`), not in
`volume-cartographer`'s (that one only builds the C++ bindings, which this
guide deliberately does not need).

Then put all eight files from this repository (`fast_cache.py`,
`fused_gather.py`, `fast_link.py`, `run_fit.py`, `test_fast_cache.py`,
`test_fast_link.py`, `fetch_roi.py`, `fetch_tree.py`) together in one
directory of your choice and run everything from there. Two constants to
check before first use:

- `run_fit.py` and `test_fast_link.py` locate the spiral scripts via a
  `SPIRAL = Path(...)` constant at the top. Point it at
  `<villa>/volume-cartographer/scripts/spiral`.
- `fetch_roi.py` / `fetch_tree.py` write the dataset to a `LOCAL = Path(...)`
  constant. Point it where you want the ~80 GB to live.

Verify the villa checkout stayed pristine (only lockfile marker churn from
`uv sync` should appear; line endings aside on Windows):

    git diff --ignore-cr-at-eol --stat    # expect: uv.lock only

## Data (public, no registration needed to download)

Everything comes from the public Hugging Face bucket
`hf://buckets/scrollprize/datasets`, prefix `spiral/PHercParis4`
(CC-BY-NC 4.0). The fetchers need `huggingface_hub`; the zero-install way is
`uvx --from huggingface_hub python fetch_roi.py ...`, or `uv pip install
huggingface_hub` into any environment. Anonymous access works but its rate
limit breaks the listing-heavy parts, so run `hf auth login` (free account)
first.

    # lasagna fields, only the z-slabs a fit actually reads (~7 GB for
    # z 4000-5000, ~28 GB for the full 4000-17000):
    python fetch_roi.py 4000 5000 --jobs 32
    # ...or shard it 4 ways for ~3.3 MB/s aggregate:
    #   python fetch_roi.py 4000 5000 --jobs 32 --shard 0:4   (etc.)

    # everything else: patch libraries, fiber tracks, winding JSONs, shell.
    # --exclude m7_ds2 skips a ~28 GB alternate track database the fit never
    # reads (fit_spiral.py names 2um_ds2_ps256_surf_v2.dbm).
    python fetch_tree.py --direct outer_shell fibers tracks umbilicus.json \
        abs_winding.json patch-overlap-pcls.json relative_windings.json \
        same_windings.json --exclude m7_ds2
    python fetch_tree.py --list --manifest patches.jsonl verified_patches unverified_patches
    python fetch_tree.py --download --manifest patches.jsonl --jobs 32   # 3.8 GB / 231k files

Both fetchers skip files that are already local and complete, so re-running
the same command is the verification pass: repeat it until it reports
`0 failed batches, 0 failed listings` with nothing left to fetch. Do not
trust a single pass on a flaky link, because a listing that failed mid-run
silently shrinks the file list.

Field notes baked into the fetchers: the account-level HF rate limit tolerates
4 parallel listing-heavy processes but not 6; an SSL drop used to permanently
break huggingface_hub's shared HTTP client (downloads keep "running" while
transferring nothing), so both fetchers now reset the client and retry. Batch
retries re-filter against local files first, so a failed 12 GB file resumes
instead of restarting.

## Verify the two equivalence suites (do this before believing anything)

    # cache: 10 bitwise-equality scenarios incl. a >2 GiB/channel pool that
    # exposes an int32 overflow trap, plus the oversized-gather capability
    # case upstream refuses outright. Runs in seconds on synthetic stores.
    .venv/bin/python test_fast_cache.py

    # linking: real patches + real point collections, upstream loop vs ours,
    # both hit policies, plus threaded-vs-serial ordering. Expect
    # "links IDENTICAL" everywhere and 44-67x; the run takes ~25 min because
    # the upstream baseline itself is the slow thing being measured.
    .venv/bin/python test_fast_link.py --patches 600 --collections 10

Reference outputs from this machine are in `results/test_fast_cache.log` and
`results/test_fast_link.log`.

## Run

    export FIT_DATASET=/path/to/PHercParis4
    export FIT_Z_BEGIN=4000 FIT_Z_END=8000      # the 4,000-slice row
    export FIT_SPIRAL_SPARSE_NORMAL_CACHE_GB=2.5
    export FIT_SPIRAL_SPARSE_GRAD_CACHE_GB=1
    export FIT_SPIRAL_SPARSE_SDT_CACHE_GB=5
    export FIT_SPIRAL_OUT_DIR=./fit_out
    export FIT_USE_FAST_CACHE=1     # 0 = upstream cache (the A/B baseline)
    export FIT_USE_FAST_LINK=1      # 0 = upstream brute-force linking
    .venv/bin/python run_fit.py

Pool budgets are the one knob that matters, and the right value *falls* as the
z-range grows because the fitted model grows with the range (41 M params at
4,000 slices, 263 M at 8,000). Measured at 8,000 slices: 8.5 GB of pools =
20.6 s/step, 5 GB = 6.4 s/step, 3.5 GB = 8.1 s/step. Details and telemetry in
README section 3 and `results/`.

Host RAM, not VRAM, sets the largest range that fits. On this 32 GB machine
the 4,000-slice row peaked at 21.5 GB RSS; the 8,000-slice range needed swap
(WSL: `memory=28GB`, `swap=96GB` in `.wslconfig`; native Linux: any
equivalent swap arrangement), and 13,000 slices exceeded 124 GB during
track-crossing indexing and could not finish here.

Resume after any interruption (config, z-range and data fingerprints are
checked by upstream's loader; step counter, LR schedule and RNG states are
restored from the checkpoint):

    export FIT_SPIRAL_RESUME_PATH=./fit_out/<run>/checkpoint_periodic.ckpt
    .venv/bin/python run_fit.py

`run_fit.py` asserts its two source substitutions still match upstream
verbatim, so a future villa change fails loudly instead of running subtly
wrong.

## Things that did not work

The 44-67x comes from the two-level tile index plus per-patch batching, run
serially. Neither half is optional, and these were the dead ends on the way
there:

- Threading the linking loop (24 workers) ran **7.6x slower** than serial:
  after prefiltering, the per-call work is too fine-grained for the GIL.
  `FIT_SPIRAL_LINK_THREADS` exists for experiments; the default is serial.
- A patch-level bounding-box filter alone yields only 2.5x: the verified
  patches are bands spanning the full fitted z-range, so their boxes admit
  nearly every point, and pruning must happen at tile level *inside* each
  patch (16x16-quad tiles).
