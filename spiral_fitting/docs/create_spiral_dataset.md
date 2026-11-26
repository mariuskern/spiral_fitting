# Create a spiral dataset

Use the [fitting environment](../README.md#1-create-the-environment) and run
commands from `spiral_fitting/`. Replace `<...>` with your values.
All inputs must share the fit's voxel resolution and origin. Z ranges are
half-open: the start is included, the end excluded. Flags take no value.

Find CT volumes in the [Data Browser](https://scrollprize.org/data_browser)
or on the [scroll data server](https://dl.ash2txt.org/).

## 1. Generate a surface prediction

Follow the [surface prediction guide](../../surface_prediction/README.md).
Track extraction requires a **binary 3D Zarr array in Z Y X order**.

## 2. Inspect and select a prediction channel

```bash
python squeeze_preciction.py "<prediction_run>/final_output.zarr"
```

For a 4D `(C, Z, Y, X)` prediction, export one channel:

```bash
python squeeze_preciction.py "<prediction_run>/final_output.zarr" \
    --out "<prediction_run>/surface_3d.zarr" \
    --channel 0 \
    --threshold 127
```

| Argument | Purpose |
| --- | --- |
| `<prediction>` | **Required.** Zarr array or group containing array `0`. Without `--out`, print shape, dtype and sample values. |
| `--out` | Output 3D Zarr; existing contents are overwritten. Only copies from 4D input. |
| `--channel` | Channel to export (default: `0`). |
| `--threshold` | Write `1` for values above this threshold, otherwise `0`. Omit for an already binary channel; match the threshold to your prediction's value range. |
| `-h` / `--help` | Flag: show CLI help. |

Use an already binary 3D input directly. This helper neither copies nor
thresholds 3D input and does not crop Z. Cropped predictions must be aligned
with the dataset's voxel frame before track extraction.

## 3. Extract tracks

```bash
python generate_tracks.py \
    --predictions "<prediction_run>/surface_3d.zarr" \
    --out "<dataset>/tracks/tracks.dbm" \
    --z-min 0 --z-max 800
```

| Argument | Purpose |
| --- | --- |
| `--predictions` | **Required.** Binary 3D surface prediction, local Zarr path or URL. |
| `--out` | **Required.** Output DBM path. |
| `--z-min` / `--z-max` | **Required.** Extraction range in prediction voxels; cover at least the fit range. |
| `--backend` | `auto` (default), `gpu` (Brook) or `cpu` (Kimimaro). |
| `--gpus <ids>` | Comma-separated NVIDIA device indices, e.g. `0,1`. Requires the GPU backend, including in `auto` mode. |
| `--skip-crossings` | Flag: skip the crossing cache. The fit can compute crossings in memory; graph patch generation requires the cache. |
| `--package-dir` | Package directory (default: `spiral_fitting/villa_8aa507c`). |
| `-h` / `--help` | Flag: show CLI help. |

Creates `tracks.dbm`, packed `tracks.dbm.vctracks/` and
`tracks.dbm.crossings.npz`. Rerunning resumes completed slabs.
The helper starts `extract_surface_tracks` and `build_track_crossings` as
package modules to support their relative imports.

The GPU backend needs Linux x86-64, an NVIDIA GPU with compute capability
≥ 8.0, and the CUDA dependencies. For setup, see the
[bundled environment guide](../villa_8aa507c/README.md#creating-the-spiral-python-environment).
For CPU extraction on Linux x86-64, install Kimimaro into the active environment:

```bash
uv pip install -e 'villa_8aa507c[cpu]'
```

Other platforms install Kimimaro by default. CPU extraction is slower;
start with a small Z range.

## 4. Grow graph patches

Requires both track sidecars from step 3; generate them without `--skip-crossings`.

```bash
python generate_graph_patches.py \
    --tracks "<dataset>/tracks/tracks.dbm" \
    --out "<dataset>/graph_patches" \
    --random-count 1000 \
    --workers 16
```

| Argument | Purpose |
| --- | --- |
| `--tracks` | **Required.** DBM path; `.vctracks` and `.crossings.npz` paths are derived from it. |
| `--out` | **Required.** Output directory, one tifxyz folder per patch. |
| `--random-count` | Number of random seed tracks; provide this or `--seeds`. |
| `--seeds` | Explicit seed track rows; see `python -m villa_8aa507c.grow_track_graph --help` for the format. |
| `--workers` | Parallel workers (default: `min(16, CPU count)`). |
| `--package-dir` | Package directory (default: `spiral_fitting/villa_8aa507c`). |
| `-- <options...>` | Forward additional options to `grow_track_graph`. |
| `-h` / `--help` | Flag: show CLI help. |

## 5. Export fiber directions (optional)

Use a fiber prediction from `<scroll>/representations/predictions/fibers/`
in the open-data bucket or create one with [Lasagna](../../lasagna/README.md).
Its `.lasagna.json` manifest references the `presence`, `nx` and `ny` Zarr channels.

```bash
uv pip install requests
python generate_fiber_directions.py "<fiber.lasagna.json>" "<dataset>/fiber_directions.npz" \
    --z-roi 0,800 \
    --output-scale 1 \
    --presence-threshold 160 \
    --cell-size 2
```

| Argument | Purpose |
| --- | --- |
| `<manifest>` | **Required.** Local path or HTTP(S) URL; relative channel paths resolve against the manifest. |
| `<output>` | **Required.** Output `.npz` path. |
| `--z-roi <begin>,<end>` | **Required.** Z range in fit voxels. |
| `--output-scale` | Scan level-0 voxels per fit voxel: fit level `0` → `1`, level `1` → `2`, level `2` → `4` (default: `1`). Independent of prediction resolution. |
| `--presence-threshold` | Minimum presence on the `0`–`255` scale (default: `160`). |
| `--cell-size` | Keep the highest-presence sample per cell, in prediction voxels (default: `2`); must divide the source chunk dimensions. |
| `--workers` | Parallel chunk downloads (default: `1`). |
| `--overwrite` | Flag: replace an existing export. |
| `-h` / `--help` | Flag: show CLI help. |

To use the export, add `paths.fiber_directions: "fiber_directions.npz"` to
the scroll JSON and `"input_use_fiber_directions": true,
"loss_weight_fiber_directions": 1.0` to the fit overrides. The positive weight
controls an orientation loss; it does not assign winding numbers.

## 6. Create the scroll axis

Save `<dataset>/umbilicus.json` with at least two centerline points at
different Z positions. Replace the example coordinates with your scroll's
winding-core centerline in fit voxels, covering the fitted Z range.

```json
{
    "control_points": [
        {"x": 400, "y": 400, "z": 0},
        {"x": 410, "y": 405, "z": 799}
    ]
}
```

## 7. Assemble the dataset

By default, the fitter expects this layout for the inputs generated here:

```text
<dataset>/
├── verified_patches/                      # one tifxyz directory per graph patch
├── tracks/
│   ├── 2um_ds2_ps256_surf_v2.dbm
│   ├── 2um_ds2_ps256_surf_v2.dbm.vctracks/
│   └── 2um_ds2_ps256_surf_v2.dbm.crossings.npz
├── umbilicus.json
├── fiber_directions.npz                   # optional
└── spiral-scroll.json
```

Steps 3 and 4 instead produce `tracks/tracks.dbm` and `graph_patches/`.
Choose one way to connect these outputs to the fit:

- **Keep the generated names:** save the `spiral-scroll.json` below. Its `paths`
  entries override the default track filename and patch directory shown above.
- **Use the default layout:** rename `graph_patches/` to `verified_patches/`
  and the track database to `tracks/2um_ds2_ps256_surf_v2.dbm`. Rename all
  associated DBM files and the `.vctracks/` and `.crossings.npz` sidecars to
  the same new basename, preserving their suffixes. Then omit `paths` from
  the JSON below, or update its entries to the renamed paths.

In both cases, replace the example name, numeric voxel size and winding sense
with your scroll's values. The filename `2um_ds2_ps256_surf_v2.dbm` is only a
default name; it does not determine the fit's voxel size.

```json
{
    "schema_version": 1,
    "name": "my-scroll",
    "voxel_size_um": 7.91,
    "spiral_outward_sense": "CW",
    "paths": {
        "umbilicus": "umbilicus.json",
        "tracks_dbm": "tracks/tracks.dbm",
        "verified_patches": "graph_patches"
    }
}
```

| Field | Purpose |
| --- | --- |
| `schema_version` | **Required.** `1`. |
| `name` | **Required.** Scroll name. |
| `voxel_size_um` | **Required.** Numeric voxel size in µm, matching all fitting inputs. |
| `spiral_outward_sense` | `CW` or `ACW`; see [orientation metadata](../villa_8aa507c/README.md#scroll-specification-spiral-scrolljson). |
| `paths.umbilicus` | Scroll axis (default: `umbilicus.json`). |
| `paths.tracks_dbm` | Track database from step 3; default without an override: `tracks/2um_ds2_ps256_surf_v2.dbm`. |
| `paths.verified_patches` | Graph patch directory from step 4; default: `verified_patches/`. Keep `input_use_verified_patches` enabled. |
| `paths.fiber_directions` | Optional `.npz` export from step 5. |

All `paths` values are relative to the dataset directory.

## 8. Run the fit

For the tracks-and-patches dataset above, set this in `../.env`:

```dotenv
FIT_SPIRAL_CONFIG_OVERRIDES='{"z_begin": 0, "z_end": 800, "optimizer_num_training_steps": 5000, "dense_spacing_mode": "grad_mag", "input_use_tracks": true, "input_use_outer_shell": false, "input_use_normals": false, "input_use_gradient_magnitude": false}'
```

| Override | Purpose |
| --- | --- |
| `z_begin` / `z_end` | Fit range; match the generated inputs. |
| `optimizer_num_training_steps` | Optimization steps (`5000` here). |
| `dense_spacing_mode` | Dense spacing loss mode (`grad_mag` here). |
| `input_use_tracks` | Enable the generated tracks. |
| `input_use_outer_shell` / `input_use_normals` / `input_use_gradient_magnitude` | Disable the shell and dense inputs absent from this dataset. Enable them when supplied. |

```bash
python fit_spiral.py --dataset "<dataset>"
```

| Argument | Purpose |
| --- | --- |
| `--dataset` | **Required.** Assembled dataset directory. |

See the [main guide](../README.md#4-run-the-pipeline) for other fit arguments,
then [visualize the meshes](../README.md#5-visualize-fitted-meshes-on-ct-slices)
or [export flattened CT segments](vc3d.md#export-segments-for-2d-ink-detection).
