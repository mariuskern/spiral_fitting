# Utils

Utility scripts for downloading CT volumes and published spiral-fitting inputs,
and converting local volumes between Zarr and TIFF.

Run commands from `utils/` with your Python environment active.
Replace `<...>` with your values. Relative paths use this working directory.
Tables list all arguments; commands show typical selections. Flags take no value.

## Download

You can use a temporary directory created with `mktemp -d` if downloads would
otherwise exceed your quota.

### `download_zarr_from_url.py`

Download a remote Zarr array, optionally restricted to a region. This example
downloads 500 Z slices and 600 by 600 pixels per slice from Scroll 332:

```bash
python download_zarr_from_url.py \
    https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll3.zarr/ \
    "<output_dir>" \
    --region 5000 5500 1600 2200 1300 1900 \
    --array 0 \
    --workers 32
```

| Argument | Purpose |
| --- | --- |
| `<url>` | **Required.** Remote Zarr group URL. |
| `<output_dir>` | **Required.** Output directory; the downloaded Zarr keeps its remote directory name. |
| `--region <start> <stop> ...` | One start/stop pair per array dimension, in array axis order (usually **Z Y X**). Stops are exclusive; omit for the whole array. |
| `--array` | Array path inside the Zarr group (default: `0`). |
| `--workers` | Parallel download threads (default: `32`). |
| `--block-factor` | Chunks per dimension per task; larger values use more memory (default: `1`). |
| `--retries` | Maximum read attempts per block (default: `5`). |
| `--resume` | Flag: reuse completed blocks from an interrupted download. |
| `-h` / `--help` | Flag: show CLI help. |

Writes `<output_dir>/<remote_name>.zarr` and `meta.json` with source and
coordinate metadata. To resume, repeat the command with `--resume`, keeping the
URL, output directory, region, `--array` and `--block-factor` unchanged.
An existing output Zarr requires `--resume`. Completed blocks are recorded in
`progress.txt`, which is removed after a successful download.

> **Warning:** Cropped downloads start at local index zero. Convert mesh
> coordinates to the selected array's resolution, then subtract the crop origin
> recorded in `meta.json`; scaling alone is insufficient.

### `fetch_roi.py` and `fetch_tree.py`

Download fitting inputs from the
[PHercParis4 dataset on Hugging Face](https://huggingface.co/buckets/scrollprize/datasets/tree/spiral/PHercParis4).
`fetch_roi.py` downloads dense `lasagna_inputs/` for a selected Z range;
`fetch_tree.py` downloads patches, tracks, fibers and annotations.
The corresponding CT volume is downloaded separately; see
[Spiral Fitting, step 2](../spiral_fitting/README.md#2-download-a-dataset).

Both Hugging Face helpers require `huggingface_hub` and `requests`
(`uv pip install huggingface_hub requests`). Before running, set the `LOCAL`
constant in each script: use your dataset directory in `fetch_tree.py` and its
`lasagna_inputs/` subdirectory in `fetch_roi.py`.

For large patch collections, list once into a manifest, then download from it.
Use `--direct` for smaller selections:

```bash
python fetch_roi.py 4000 5000 --jobs 64 --margin 320
python fetch_tree.py --list --manifest m.jsonl verified_patches unverified_patches
python fetch_tree.py --download --manifest m.jsonl --jobs 32
python fetch_tree.py --direct outer_shell fibers tracks abs_winding.json patch-overlap-pcls.json relative_windings.json same_windings.json umbilicus.json
```

| Script / argument | Purpose |
| --- | --- |
| `fetch_roi.py <z_begin> <z_end>` | **Required.** Half-open Z range in fit voxels; PHercParis4 uses 9.6 micrometers per fit voxel. |
| `fetch_tree.py <prefixes...>` | Dataset-relative files or directories; used with `--list` or `--direct`. |
| `--jobs` | Concurrent download batches (ROI default: `64`; tree default: `32`). |
| `--margin` | ROI only: extra fit voxels on each side of the range (default: `320`). |
| `--dry-run` | ROI only: flag to fetch metadata and report slabs without downloading chunk data. |
| `--list` | Tree only: flag to write selected file paths and sizes to a manifest. |
| `--download` | Tree only: flag to download the files recorded in a manifest. |
| `--direct` | Tree only: flag to list and download selected prefixes in one step. |
| `--manifest <file>` | Tree only: JSONL manifest path; required with `--list` and `--download`. |
| `--exclude <text>` | Tree only: skip paths containing this text in `--direct` mode; repeatable. |
| `--shard <i>:<n>` | Download part `i` of `n`; run all parts, starting at `0`. Tree: use with `--download`. |
| `-h` / `--help` | Flag: show either script's CLI help. |

The ROI example covers `[4000, 5000)` plus the margin. Its stores retain their
original coordinates; keep the fit inside the downloaded range.

For `fetch_tree.py`, choose one mode per command. Both scripts skip existing files with
matching sizes. A manifest can be reused for retries and separate shard downloads.

## Conversion

### `zarr_to_tiff.py`

Convert a local Zarr volume into a TIFF stack and one TIFF file per Z slice:

```bash
python zarr_to_tiff.py "<input.zarr>" "<output_dir>" --array 0
```

| Argument | Purpose |
| --- | --- |
| `<input_dir>` | **Required.** Local Zarr group or standalone array. |
| `<output_dir>` | **Required.** Output directory; must be outside the input directory. |
| `--array` | Array path inside a group (default: `0`); ignored for a standalone array. |
| `--prefix` | Filename prefix for individual slices (default: empty). |
| `--digits` | Minimum digits for zero-padded slice numbers (default: `2`). |
| `--overwrite` | Flag: replace existing slice files. |
| `--no-compression` | Flag: disable ZSTD compression for individual slice files. |
| `-h` / `--help` | Flag: show CLI help. |

Writes `<output_dir>/<output_dir_name>.tif` and `layers/<prefix><slice>.tif`.
Supported shapes are `(Z, Y, X)` and `(1, Z, Y, X)`; a leading singleton channel
is removed and slices are read individually. For repeated exports, use
`--overwrite` so every slice is included in the rebuilt stack.

### `tiff_to_zarr.py`

Convert a 3D TIFF stack into a Zarr array. For a binary mask with values `0` and
`1`, use `--mul 255` to produce values `0` and `255`:

```bash
python tiff_to_zarr.py "<input.tiff>" "<output.zarr>" --mul 255
```

| Argument | Purpose |
| --- | --- |
| `<input_path>` | **Required.** Input TIFF stack in **Z Y X** order. |
| `<output_path>` | **Required.** Output Zarr array path; an existing array is overwritten. |
| `--mul` | Integer multiplier applied after conversion to `uint8` (default: `1`). |
| `-h` / `--help` | Flag: show CLI help. |

Writes a `uint8` Zarr v2 array with `192 x 192 x 192` chunks, Blosc/ZSTD
compression and `/`-separated chunk keys. Omit `--mul` to keep existing `uint8`
intensities. Fractional input values are truncated before multiplication;
this option does not rescale a continuous floating-point volume from 0 to 1.

## Sources

The scripts `fetch_roi.py` and `fetch_tree.py` come from the following GitHub repository:

- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/tree/d8c5f488a105286c548c99c5f7c7ad9f29e3ed14) (Commit: `d8c5f48`)
