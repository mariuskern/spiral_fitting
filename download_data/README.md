# Download Data


## Sources

The `fetch_roi.py` and `fetch_tree.py` are from the following GitHub repository:

- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/tree/d8c5f488a105286c548c99c5f7c7ad9f29e3ed14) (Commit: d8c5f48)


## Scripts

### `download_zarr_from_url.py`

Downloads a remote Zarr array in parallel. Optionally select a region with one start/stop pair per dimension in array axis order (e.g. Z, Y, X); stop indices are exclusive.

**Usage:**

```bash
python download_zarr_from_url.py <url_to_zarr> <output_path> \
    [--region <z_start> <z_stop> <y_start> <y_stop> <x_start> <x_stop>] \
    [--array <array>] [--workers 32] [--block-factor 1] [--retries 5] [--resume]
```

- `--region`: Optional region; omit to download the entire selected array.
- `--array`: Array path inside the Zarr group (default: `0`).
- `--workers`: Parallel download threads (default: `32`).
- `--block-factor`: Chunks per dimension per task; larger values use more memory (default: `1`).
- `--retries`: Maximum read attempts per block (default: `5`).
- `--resume`: Continue an interrupted download, skipping completed blocks. Repeat the same command with this flag; keep URL, output path, region, `--array` and `--block-factor` unchanged. Starts a new download if none exists.

Creates `<output_path>/<remote_name>.zarr` using the remote Zarr's directory name, plus `meta.json` with source and coordinate metadata. Local indices start at zero; add the source region's start coordinates to recover source indices. An existing output Zarr requires `--resume`. Completed blocks are tracked in `progress.txt`, which is removed after a successful download.

> **Warning:** Cropped downloads start at local index zero. Convert mesh
> coordinates to the selected array's resolution, then subtract the crop origin
> recorded in `meta.json`; scaling alone is insufficient.

**Example: Download a region of Scroll 332**

```bash
python download_zarr_from_url.py \
    https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll3.zarr/ \
    <output_path> \
    --region 5000 5500 1600 2200 1300 1900 \
    --array 0
```

This example downloads a region with 500 slices along Z and 600 × 600 pixels per slice.

### `zarr_to_tiff.py`

Converts a Zarr volume into a TIFF image stack and a folder with one tiff per layer. Use this script after downloading a volume if you need the data in TIFF format.

**Usage:**

```bash
python zarr_to_tiff.py <path_to_zarr> <output_path> --array <array>
```

- `<path_to_zarr>`: Path to the input Zarr volume.
- `<output_path>`: Output path for the TIFF image stack.
- `--array`: Specifies the array path inside a Zarr group (default: `0`). If the Zarr volume is itself an array, it is used directly and `--array` is not needed.

Supported volume shapes are `(Z, Y, X)` and `(1, Z, Y, X)`. A leading singleton
channel dimension is removed during export; slices are read individually.

### `tiff_to_zarr.py`

Converts a TIFF image stack into a Zarr volume.

**Usage:**

```bash
python utils/tiff_to_zarr.py <input.tiff> <output.zarr> --mul 255
```

The `--mul 255` option multiplies the input values by 255. This is useful when converting normalized intensity values in the range 0-1 to the range 0-255.
