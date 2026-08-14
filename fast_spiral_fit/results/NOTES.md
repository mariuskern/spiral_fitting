# Evidence index

Every number in the README maps to a file here. All produced on the machine
described in README (RTX 4080 Laptop 12 GB, WSL2), 2026-07-29 to -31.

| file | backs which claim |
|---|---|
| `test_fast_cache.log` | cache equivalence: 10/10 bitwise, incl. the oversized-gather case upstream refuses (README §1) |
| `test_fast_link.log` | linking equivalence: links IDENTICAL both hit policies + threaded ordering, 58x/67x on the shared sample (README §2) |
| `upstream_refusal_8000slices.txt` | the verbatim `RuntimeError` from the upstream cache at 8,000 slices (README §1 table) |
| `crossing_cache_discard_warning.txt` | the fit discarding the shipped crossings sidecar under default config (README §3 scope note) |
| `fit_4000slices_keylines.txt` | clean 4,000-slice run: inputs loaded, 30000/30000 in 4:55:40 (README §3 ladder) |
| `satisfied_4000slices.json` | per-patch satisfaction for that run (median 0.984 over 1,259 verified patches) |
| `telemetry_4000slices_clean.log` | per-minute RSS/VRAM of that run (peak 21.5 GB RSS / 8.2 GB VRAM) |
| `telemetry_8000slices_{8.5,5,3.5}GBpools.log` | the pool-budget sweep: 20.6 / 6.4 / 8.1 s per step (README §3 sweep) |
| `telemetry_13000slices_oom.log` | the full-range attempt: swap exhaustion during crossing indexing, GPU at 6.9 GB (README §3) |
| `telemetry_4000slices_16GB_oomkill.log` | the earlier 16 GB-limit OOM kill that motivated raising host memory |

Log format for telemetry: `HH:MM:SS rss_kb=<resident set> gpu=<MiB used>,<util %> [swap_mb=<swap used>]`, one line per minute.

Not included: the raw fit logs (hundreds of MB of tqdm output) and the 497 MB
to 1.6 GB checkpoints. Both are regenerable via REPRO.md; checkpoints
available on request.
