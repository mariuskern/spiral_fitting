# Spiral Fitting

<a id="generating-the-spiral-fit-input"></a>

## Guides

| Guide | Purpose |
| --- | --- |
| [Getting started](#getting-started) | Download published inputs and fit a spiral. |
| [Create a spiral dataset](docs/create_spiral_dataset.md) | Generate tracks, patches, fiber directions and a scroll specification. |
| [VC3D](docs/vc3d.md) | Install VC tools, flatten fitted meshes and render CT layers for ink detection. |

## Getting started

Run commands from `spiral_fitting/` with the fitting environment active.

Relative paths use this working directory; paths in `spiral-scroll.json` use the dataset directory.

### 1. Create the environment

```bash
conda create -n spiral-fitting python=3.14
conda activate spiral-fitting
pip install uv
uv pip install -e villa_8aa507c
uv pip install python-dotenv
```


### 2. Download a dataset

| Dataset | Fitting inputs | Corresponding CT volume |
| --- | --- | --- |
| PHercParis4 | [Fitting inputs (dl.ash2txt.org)](https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/) · [Fitting inputs (Hugging Face)](https://huggingface.co/buckets/scrollprize/datasets/tree/spiral/PHercParis4) | [CT scan: 2026-04-11, 2.400 µm, 78 keV (Zarr)](https://vesuvius-challenge-open-data.s3.amazonaws.com/index.html#PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/) |
| PHerc0332 | [Fitting inputs (dl.ash2txt.org)](https://dl.ash2txt.org/datasets/spiral_datasets/PHerc0332/20251211183505/) | [CT scan: 2025-12-11, 2.399 µm, 78 keV (Zarr)](https://vesuvius-challenge-open-data.s3.amazonaws.com/index.html#PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked-fast-q1-c64-v0.3.2.vcd3d.zarr/) |

More fitting datasets:

- [Spiral fitting dataset collection (dl.ash2txt.org)](https://dl.ash2txt.org/datasets/spiral_datasets/)
- [Spiral fitting dataset collection (Hugging Face)](https://huggingface.co/buckets/scrollprize/datasets/tree/spiral)

PHercParis4 annotations use **9.6 µm voxels**, level `2` of the **2.4 µm** CT:
multiply coordinates by `4` for level `0`, and set `voxel_size_um` to `9.6`
in the fit. The CT is downloaded separately. See the
[dataset reference](https://scrollprize.org/data/datasets/spiral-input-PHercParis4-README.md)
for contents and coordinates; its fitting commands target a newer upstream
version. This bundled fitter uses `spiral-scroll.json` as shown below.

With [rclone](https://rclone.org/downloads/) installed:

```bash
rclone copy :http: ./spiral_datasets/phercparis4 \
    --http-url https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/ \
    --transfers 25 \
    --checkers 4 \
    --retries 20 \
    -P
```

| Argument | Purpose |
| --- | --- |
| `copy :http: <destination>` | Download to the local dataset directory. Rerun to update. |
| `--http-url` | Dataset URL from the table above. |
| `--transfers` | Parallel file transfers (`25` here). |
| `--checkers` | Parallel file checks (`4` here). |
| `--retries` | Retry count (`20` here). |
| `-P` | Flag: show progress. |

Use `sync` instead of `copy` only to mirror the server, including deleting
local files absent from it. On the HPC cluster, `mktemp -d` provides a
temporary directory without the usual file limit.

#### Hugging Face alternative (PHercParis4)

Set `LOCAL` in `../download_data/fetch_tree.py` to your dataset directory,
and `LOCAL` in `../download_data/fetch_roi.py` to its `lasagna_inputs/` subdirectory.
The ROI download must cover the fit's Z range; this example fits `[4000, 5000)`.

```bash
uv pip install huggingface_hub requests
python ../download_data/fetch_roi.py 4000 5000
python ../download_data/fetch_tree.py --list --manifest m.jsonl verified_patches unverified_patches
python ../download_data/fetch_tree.py --download --manifest m.jsonl --jobs 32
python ../download_data/fetch_tree.py --direct outer_shell fibers tracks abs_winding.json patch-overlap-pcls.json relative_windings.json same_windings.json umbilicus.json
```

| Script / argument | Purpose |
| --- | --- |
| `fetch_roi.py <z_begin> <z_end>` | Download dense inputs for this half-open Z range in fit voxels. |
| `--jobs` | Parallel downloads (ROI default: `64`; tree default: `32`). |
| `--margin` | ROI only: extra fit voxels on each side (default: `320`). |
| `--dry-run` | ROI only: fetch metadata and report slabs without downloading chunk data. |
| `fetch_tree.py <prefixes...>` | Dataset files or directories to fetch. |
| `--list --manifest <file>` | List selected prefixes into a JSONL manifest. |
| `--download --manifest <file>` | Download all files in the manifest. |
| `--direct` | List and download selected prefixes in one step. |
| `--exclude <text>` | Tree direct mode: skip paths containing this text; repeatable. |
| `--shard <i>:<n>` | Either script: download shard `i` of `n` (`i` starts at `0`). Run every shard for a complete download. |
| `-h` / `--help` | Flag: show script help. |

### 3. Configure the fit

Add settings to `../.env`; `fit_spiral.py` loads it automatically.
Merge changes into a single `FIT_SPIRAL_CONFIG_OVERRIDES` JSON object.

```dotenv
FIT_SPIRAL_CONFIG_OVERRIDES='{"z_begin": 4000, "z_end": 5000, "optimizer_num_training_steps": 5000, "dense_spacing_mode": "grad_mag"}'
FIT_SPIRAL_OUT_DIR=./out
WANDB_MODE=disabled
```

| Variable | Purpose |
| --- | --- |
| `FIT_SPIRAL_CONFIG_OVERRIDES` | Override keys from [`config.py`](villa_8aa507c/config.py). `z_begin` is inclusive; `z_end` is exclusive. Add `"input_use_tracks": true` to enable tracks (disabled by default). |
| `FIT_SPIRAL_OUT_DIR` | Parent directory for generated runs (default: `./out`). |
| `FIT_SPIRAL_RUN_DIR` | Exact run directory instead of a generated name. |
| `FIT_SPIRAL_RUN_TAG` | Suffix for the run directory and mesh names. |
| `FIT_SPIRAL_RESUME_PATH` | Checkpoint to resume; the saved iteration is restored automatically. |
| `FIT_SPIRAL_CACHE_DIR` | Shared preprocessing cache (default: `~/.cache/vc3d/spiral`, respecting `XDG_CACHE_HOME`). Reuse between runs. |
| `WANDB_MODE` | Logging: `disabled` (default), `online` or `offline`. |

Use literal paths in `.env` and JSON rather than embedding environment variables.

Save the scroll metadata as `<dataset>/spiral-scroll.json`. For PHercParis4:

```json
{
    "schema_version": 1,
    "name": "PHercParis4",
    "voxel_size_um": 9.6,
    "spiral_outward_sense": "CW"
}
```

| Field | Purpose |
| --- | --- |
| `schema_version` | **Required.** `1`. |
| `name` | **Required.** Scroll name used in output names. |
| `voxel_size_um` | **Required.** Voxel size of the fitting inputs in µm. |
| `spiral_outward_sense` | `CW` or `ACW`; match your scroll. See [orientation metadata](villa_8aa507c/README.md#scroll-specification-spiral-scrolljson) for catalog-based derivation. |
| `paths` | Optional input path overrides, relative to the dataset. For custom tracks: `"tracks_dbm": "tracks/tracks.dbm"`; default: `tracks/2um_ds2_ps256_surf_v2.dbm`. |

### 4. Run the pipeline

```bash
python fit_spiral.py --dataset "<dataset>"
```

| Argument | Purpose |
| --- | --- |
| `--dataset` | **Required.** Dataset directory. |
| `--scroll-spec` | Scroll JSON (default: `<dataset>/spiral-scroll.json`). |
| `--cache` | Preprocessing cache; overrides `FIT_SPIRAL_CACHE_DIR`. |
| `-h` / `--help` | Flag: show CLI help. |

### 5. Visualize fitted meshes on CT slices

```bash
python visualize_meshes.py "<fit_run>" \
    --ct "<ct_volume>" \
    --ct-dataset 0 \
    --num-slices 6 \
    --mesh-range 10 36 \
    --out "<output_dir>/colored_sections.png"
```

| Argument | Purpose |
| --- | --- |
| `<fit_run>` (`meshes`) | **Required.** Fit run, `meshes/` or `meshes/fitted/`. |
| `--ct` | **Required.** CT Zarr with the same voxel resolution and origin as the meshes. |
| `--ct-dataset` | Array inside the CT group (default: `0`); ignored for a standalone array. PHercParis4 fits at 9.6 µm need level `2` of the 2.4 µm CT. |
| `--num-slices` | Evenly spaced sections (default: `6`); mutually exclusive with `--slices`. |
| `--slices <z> [<z> ...]` | Explicit CT Z indices. |
| `--z-range <min> <max>` | Inclusive Z range (default: mesh/CT overlap). |
| `--meshes <n> [<n> ...]` | Winding numbers; mutually exclusive with `--mesh-range`. |
| `--mesh-range <min> <max>` | Inclusive winding range (default: all). |
| `--mesh-step` | Show every Nth selected mesh (default: `1`). |
| `--spliced` | Flag: use `wNNN_spliced` meshes. |
| `--columns` | Maximum subplot columns (default: `3`). |
| `--cmap` | Mesh colormap (default: `turbo`). |
| `--colorbar-step` | Label every Nth selected mesh (default: at most 16 labels). |
| `--line-width` | Mesh line width (default: `0.8`). |
| `--alpha` | Mesh opacity, `0`–`1` (default: `0.9`). |
| `--ct-window <min> <max>` | Grayscale limits (default: first slice's nonzero 1st/99.5th percentiles). |
| `--xy-range <xmin> <xmax> <ymin> <ymax>` | Shared zoom rectangle in CT voxels. |
| `--out` | Save an image; omit to open a window. |
| `--dpi` | Export resolution (default: `160`). |
| `-h` / `--help` | Flag: show CLI help. |

### 6. Rank meshes for ink detection

Print estimated papyrus coverage and area per winding, sorted by coverage.

```bash
python rank_meshes.py "<fit_run>" \
    --ct "<masked_ct_volume>" \
    --ct-dataset 0 \
    --spliced \
    --top 20
```

| Argument | Purpose |
| --- | --- |
| `<fit_run>` (`meshes`) | **Required.** Fit run, `meshes/` or `meshes/fitted/`. |
| `--ct` | **Required.** Papyrus mask or CT Zarr with zero background, aligned with the meshes. |
| `--ct-dataset` | Array inside the Zarr group (default: `0`; Scroll3: `0`, PHercParis4 at 9.6 µm: `2`); ignored for a standalone array. |
| `--threshold` | Count voxel values above this threshold as material (default: `0`). |
| `--samples` | Area-uniform surface samples per mesh (default: `100000`). |
| `--seed` | Nonnegative sampling seed for reproducible results (default: `0`). |
| `--sort-by` | `coverage`: papyrus percentage; `area`: absolute papyrus area (default: `coverage`). |
| `--top` | Print the first N results (default: all). |
| `--meshes <n> [<n> ...]` | Winding numbers; mutually exclusive with `--mesh-range`. |
| `--mesh-range <min> <max>` | Inclusive winding range (default: all). |
| `--spliced` | Flag: use `wNNN_spliced` instead of regular `wNNN` meshes. |
| `--voxel-size-um` | Mesh voxel size in µm for areas in cm² (default: infer from `meta.json`, otherwise use voxel²). |
| `--coordinate-scale` | Volume voxels per mesh voxel (default: `1`; use `4` for 9.6 µm meshes in a 2.4 µm CT). |
| `--volume-origin <x> <y> <z>` | First volume voxel in mesh coordinates for a crop (default: `0 0 0`). Mapping: `(mesh_xyz - origin) * coordinate_scale`. |
| `-h` / `--help` | Flag: show CLI help. |

<a id="getting-started-with-render_inkpy"></a>

For flattening and CT layer export, continue with the
[VC3D guide](docs/vc3d.md#export-segments-for-2d-ink-detection).
The bundled [`render_ink.py`](villa_8aa507c/render_ink.py) instead projects an
existing **3D ink prediction** through fitted surfaces into 2D images.

## Sources

### Current fitter: `villa_8aa507c/`

The `villa_8aa507c/` folder contains the spiral-fitting code from the following GitHub repository:

- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/tree/8aa507c1215f1fa8758f13d0538f7af9710e02d7/spiral-fitting) (Commit: `8aa507c`)

The original, unmodified code can be found in
[`villa_8aa507c/.original_code/spiral-fitting/`](villa_8aa507c/.original_code/spiral-fitting/).

### Earlier fitter: `villa_8aa507c/.old/`

This folder contains an earlier version of the fitting code from the following GitHub repository:

- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/tree/d8c5f488a105286c548c99c5f7c7ad9f29e3ed14) (Commit: `d8c5f48`)

The original, unmodified code can be found in
[`villa_8aa507c/.old/original_code/`](villa_8aa507c/.old/original_code/).

The following subfolders combine code from additional sources.

#### `villa_8aa507c/.old/spiral_7769da8/`

This folder combines the spiral-fitting code from **ScrollPrize/villa** with
a faster implementation from **7jycwjmbfn-eng/spiral-fit-consumer-gpu**.
This version of the spiral fitting is working.

- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/tree/7769da8cf2233310570608feecc127066a7c0c7c/volume-cartographer/scripts/spiral) (Commit: `7769da8`) - base spiral-fitting code.
- [**7jycwjmbfn-eng/spiral-fit-consumer-gpu**](https://github.com/7jycwjmbfn-eng/spiral-fit-consumer-gpu/commit/f189740211f462193973055a32e3269c03301587) (Commit: `f189740`) - faster fitting implementation.
- [**ScrollPrize/villa - spiral_progress.py**](https://github.com/ScrollPrize/villa/blob/9761f14773a3ed41f2459bbf689a8f1998a656ed/volume-cartographer/scripts/spiral/spiral_progress.py) (Commit: `9761f14`) - source of `spiral_progress.py` only.

#### `villa_8aa507c/.old/spiral_6732d35/`

This folder combines the spiral-fitting code from **ScrollPrize/villa** with
a faster implementation from **7jycwjmbfn-eng/spiral-fit-consumer-gpu**.
This version is **not working**.

- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/tree/6732d3587bc3224a8a69a15886fb68c6e57a9342/volume-cartographer/scripts/spiral) (Commit: `6732d35`) - base spiral-fitting code.
- [**7jycwjmbfn-eng/spiral-fit-consumer-gpu**](https://github.com/7jycwjmbfn-eng/spiral-fit-consumer-gpu/commit/f189740211f462193973055a32e3269c03301587) (Commit: `f189740`) - faster fitting implementation.
- [**ScrollPrize/villa - spiral_progress.py**](https://github.com/ScrollPrize/villa/blob/9761f14773a3ed41f2459bbf689a8f1998a656ed/volume-cartographer/scripts/spiral/spiral_progress.py) (Commit: `9761f14`) - source of `spiral_progress.py` only.

> **Note:** The `.old/` folder and both variants above are deprecated. Use the
> current fitter in [`villa_8aa507c/`](villa_8aa507c/) for new runs.

