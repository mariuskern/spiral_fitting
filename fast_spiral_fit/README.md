# Spiral fitting on a 12 GB consumer GPU

Author: Shuhan Yang. July 2026 Progress Prize submission.

Run the official `villa` spiral fitter on a gaming laptop. Three drop-in
additions, no upstream source file edited.

Every number below was measured on one machine: Windows 11 + WSL2, RTX 4080
Laptop (12,282 MiB VRAM), 31.6 GB host RAM, villa @ `7769da8`, without the
native `vc` C++ extension built (`import vc` fails, which is what you get if
you do not build VC3D yourself).

## The problem

The spiral fitting scripts are written for a machine with the compiled VC
kernels and a datacenter GPU. Without the C++ build, two code paths that exist
only as fallbacks become the only paths you have: point-to-patch linking turns
into a 41-hour brute-force search, and the track-crossing rebuild wants more
host memory than most workstations carry. Separately, on a 12 GB card the
sparse GPU cache does not run at all; it raises an error before the first
training step.

This repository fixes the cache and the linking, with tests showing the fast
versions return exactly what the originals return. The crossing rebuild is
measured but not fixed; section 3 gives the number and points at the code
upstream already ships to fix it.

## 1. Sparse GPU cache that degrades instead of refusing

`sparse_cuda_cache.BoundedSparseCudaCache` raises when a single gather touches
more chunks than the pool holds:

```
if len(requested_keys) > self.capacity:
    raise RuntimeError(...)
```

There is no fallback. On current `main` as of 2026-07-30 this is still a hard
refusal, and the open PR #1281 moves in the other direction (fully
device-resident pools, ~33 GiB SDT + ~10 GiB normals). On a 12 GB card this
refusal is the first thing a user hits.

`fast_cache.py` + `fused_gather.py` replace it with a pool that splits an
oversized gather into sequential passes, and fuses the ten elementwise index
kernels into one Triton kernel.

| | result |
|---|---|
| Equivalence | 10/10 scenarios bitwise-equal (`torch.equal`) vs upstream, incl. a >2 GiB/channel pool and the oversized-gather case upstream refuses |
| Compute-bound speedup | **3.49x** (SDT) / **3.09x** (normals), paired alternating trials |
| Ablation | 2-pass variant 3.27x/2.81x; torch-only path 1.35x/1.17x, so the win is kernel fusion rather than the algorithm |
| IO-bound | ~1.0x, no effect |

**Where the two caches differ.** Fresh runs, identical configuration, cache as
the only variable:

| z range | ours | upstream |
|---|---|---|
| 1,000 slices, 30,000 steps | 43 min 37 s | 47 min 17 s (ours 8.4% faster, single paired run) |
| 4,000 slices, steps ~700 | 2.09 it/s, 8.18 GB VRAM | 2.03 it/s, 8.18 GB VRAM, i.e. parity |
| 8,000 slices | reaches training, 6.4 s/step at tuned pools | **`RuntimeError` before step 0** |

The middle row bounds the speed claim. When the working set fits in the pools,
this cache is not meaningfully faster end to end; the 3.5x is a gather-level
measurement and gather is only part of a training step. The bottom row is what
this cache is for.

```
RuntimeError: lasagna grad_mag gather touches 35355 chunks (1.08 GiB),
exceeding its 28105-chunk LRU capacity; increase the corresponding
FIT_SPIRAL_SPARSE_*_CACHE_GB
```

The suggested remedy is not available on this hardware, since the card is
already at 11.9 of 12.3 GB.

One bug, found by reading the kernel rather than by testing: `capacity *
slot_stride` multiplied as two int32 scalars overflows once a pool passes
2 GiB per channel, and channel 1 then reads garbage. The test suite and the
benchmark both used exactly 2 GiB pools, so neither could see it; it would
have appeared only on a real card with a larger pool. Fixed by computing the
value on the host, with a regression test.

## 2. Point→patch linking without the native surface index

`point_collection.link_points_to_patches` uses `vc.surface_index` when present.
Absent, it falls back to projecting every point onto every candidate patch,
and `Patch.project` tests each point against every triangle of that patch.

The patch count is not the problem (at z 4000–5000 the ROI filter leaves only
~2,800 patches); the triangle count is. The verified patches are bands
carrying millions of faces each, and every projection walks all of them.
Measured on the 1,000-slice fit: 10.1 s per point collection, giving an ETA of
**~41 hours** for the 14,658 collections before the first optimisation step.
That figure is the tqdm ETA of the stalled run, not an extrapolation from a
small sample. Current `main` has no KD-tree and no bounding-box prefilter,
nothing between the native path and brute force.

`fast_link.py` adds the pruning the native index would have provided, at two
levels, plus per-patch batching:

- tile boxes (16×16 quads) bound the *valid* faces of each patch;
- all tiles across all patches go into one z-bucketed index, so a point only
  reaches patches whose surface is actually near it;
- surviving candidates are projected once per patch for all their points.

The same lower-bound argument makes both levels sound. A triangle lies inside
its tile's vertex box, so box distance ≤ surface distance, and the linker
discards any patch beyond `tolerance` with a strict `>`. Anything the filter
drops, upstream drops too. Survivors then run upstream's own arithmetic in
upstream's own face order, so ties resolve identically.

| | result |
|---|---|
| Equivalence | links, `on_patch` annotations, ij coordinates, distances identical, for both hit policies, plus threaded-vs-serial ordering |
| Speedup | **44–67x** on the shared sample (upstream ~690 s → 10–15 s) |
| At full-scroll scale | 14,732 collections against 45,703 patches: ~7 minutes |

Two things that did not work, in case they save someone else the attempt:

- **Patch-level boxes alone: only 2.5x.** The verified patches are bands
  spanning the whole fitted range, so their boxes admit nearly every point.
  Pruning has to happen inside a patch.
- **24 worker threads: 7.6x slower than serial.** After prefiltering, the
  per-call work is too small for the GIL. Batching per patch is where the
  speed comes from. The default is serial; `FIT_SPIRAL_LINK_THREADS` remains
  for experiments.

## 3. Measured scaling ladder, and where it stops

All runs use stock configuration, 30,000 steps, fast cache + fast linking.
Pool budgets are noted per row.

| z range | slices | pools | wall clock | peak VRAM | host RSS | verified-patch satisfaction (median) |
|---|---|---|---|---|---|---|
| 4000–5000 | 1,000 | 2/1/3.5 | 43 min 37 s | 5.2 GB | 4.6 GB (spot) | 1.00 |
| 4000–8000 | 4,000 | 2.5/1/5 | **4 h 55 min 40 s** | 8.2 GB | 21.5 GB (peak) | 0.984 |
| 4000–12000 | 8,000 | see sweep | not completed: 20.6 s/step at 8.5 GB pools, **6.4 s/step tuned** | 11.9 → 11.0 GB | 25.7 GB peak + 18 GB swap | — |
| 4000–17000 | 13,000 | 2.5/1/5 | **did not fit** | 6.9 GB at death | >124 GB demanded | — |

The 8,000-slice row is a rate measurement rather than a finished fit: 30,000
steps at 20.6 s would take about seven days. It is here because of why it is
slow, and because the fix is a configuration change.

The fitted model scales with the range, from 41 M parameters at 1,000 and
4,000 slices to **263 M (1.05 GB) at 8,000**. A pool budget tuned for a small
range therefore leaves the card at 11.9 of 12.3 GB, with the model, optimiser
state and activations competing for what remains, and every step turns the
cache over. Sweeping the budget at 8,000 slices, everything else fixed:

| pool budget (normals/grad/SDT) | step time | VRAM |
|---|---|---|
| 8.5 GB (2.5/1/5) | 20.6 s | 11.9 GB |
| **5.0 GB (1.5/0.5/3)** | **6.4 s** | 11.0 GB |
| 3.5 GB (1/0.5/2) | 8.1 s | 10.5 GB |

A 3.2x speedup from shrinking the cache, with an optimum in the middle: too
large and the pools crowd out the model, too small and single gathers split
into too many passes. So the rule is to shrink the pool budget as the fitted
range grows, which is the opposite of what most people would try, and it means
a large range on a small card can look hopeless when it is only mistuned. Even
at the best setting, 30,000 steps at 8,000 slices needs about 53 h on this
laptop, so that row stays incomplete.

Two footnotes on the table. RSS for the 1,000-slice row is a spot reading
taken during training rather than a sampled peak, since per-minute telemetry
was only added from the 4,000-slice run onward; its pools were the earlier
2/1/3.5. The 4,000-slice row is a clean single-configuration run (2.5/1/5,
1.69 it/s average); an earlier completion of the same range was spliced across
two pool budgets after a mid-run retune and is not quoted, though both runs
land on the same quality (verified-patch satisfaction median 0.984 vs 0.986,
point-collection median 1.00 in both).

The 13,000-slice attempt is worth reading closely. It failed with **6.9 GB of
VRAM in use**, so the GPU was never the limit. What ran out was host memory:
28 GB RAM plus 96 GB of swap, exhausted while building the track-crossing
index in memory over 18.3 M tracks / 675 M points.

**Scope of that claim.** Upstream does ship a memory-frugal offline builder
(`build_track_crossings.py`, disk-backed) and the dataset ships its prebuilt
`.crossings.npz` sidecar. The fit loads that sidecar and then discards it,
because the default `track_exclusion_radius=16` clips track points near
patches while the cached crossing records index unclipped points:

```
WARNING: track crossing cache cannot be used after point-level
track exclusion; rebuilding crossings from the clipped tracks
```

The rebuild uses the in-memory builder, not the disk-backed one. So the claim
is narrow and testable: without the native kernel, a stock full-region fit
rebuilds crossings in memory and needs >124 GB. It is not that upstream cannot
do this; with the C++ kernels and their sort-array release strategy, memory is
far lower.

The obvious fix is upstream's own code: have the in-fit rebuild use the
disk-backed path it already has. A second option is to remap the cached
crossings through the clipping mask instead of rebuilding, which is mechanical
because clipping only deletes points and preserves order. Whether that is
bit-identical to a rebuild is an open question, since crossing acceptance uses
a 30° angle test on neighbouring points and a clipped neighbour perturbs the
direction. It is offered as a proposal, not a result.

## 4. Batch-mode checkpointing

Upstream's periodic saving lives behind the interactive driver, so a batch fit
saves once, at the end. `run_fit.py` hooks the existing atomic `save_model_to`
into the training loop every 1,000 steps, so a 30,000-step laptop run survives
interruption. Resume uses upstream's own loader, which restores step counter,
LR schedule and RNG states and validates config and data fingerprints. This is
five lines of wiring around upstream machinery, listed for completeness rather
than as a contribution.

## Reproduction

See `REPRO.md` for the pinned commit, the public CC-BY-NC data, both
equivalence suites, and the exact environment variables for every run in the
table above. `git diff --ignore-cr-at-eol` on the villa checkout is empty apart
from lockfile marker churn, since every behaviour change is injected from
outside.

## Files

| file | role |
|---|---|
| `fast_cache.py`, `fused_gather.py` | sparse pool with gather splitting + fused Triton gather |
| `fast_link.py` | tile-level spatial index and batched projection for linking |
| `run_fit.py` | entry point: installs both, adds periodic checkpointing, rewrites nothing upstream |
| `test_fast_cache.py`, `test_fast_link.py` | the equivalence suites the claims above rest on |
| `fetch_roi.py`, `fetch_tree.py` | z-sliced / manifest-driven dataset fetchers (rate-limit and CDN-failure hardened) |
| `results/` | logs and telemetry behind every number, indexed in `results/NOTES.md` |

All files here are MIT licensed. The villa checkout they drive is unmodified
upstream, with its own licenses, and the dataset is the public CC-BY-NC 4.0
release.
