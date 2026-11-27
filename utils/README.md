# Utils

Run commands from `utils/` with a suitable environment activated (e.g. `spiral-fitting`).
Use `python <script> --help` for all options.

## Download

### `download_zarr_from_url.py`

Download arrays or recursive groups; `--array .` selects everything.
Requires `zarr`, NumPy and `fsspec[http]`.

```bash
python download_zarr_from_url.py "<url>" "<output_dir>" \
    --array 1 2 3 \
    --reference-array 0 \
    --region 5000 5500 1600 2200 1300 1900
```

| Argument | Purpose |
| --- | --- |
| `<url>` | **Required.** Remote Zarr group URL. |
| `<output_dir>` | **Required.** Output directory; the Zarr keeps its remote directory name. |
| `--region <start> <stop> ...` | Start/stop pairs in array axis order (usually Z Y X), stops exclusive. Omit for whole arrays. |
| `--array <path> ...` | Arrays or recursive groups (default: `0`); `.` selects everything. Repeatable; aliases: `--arrays`, `--group`, `--groups`. |
| `--reference-array <path>` | Interpret `--region` in this array's coordinates and map to each selected array. Reference is not automatically downloaded. Aliases: `--reference-group`, `--reference`. |
| `--workers` | Parallel download threads (default: `32`). |
| `--block-factor` | Chunks per dimension per task (default: `1`); larger values use more memory. |
| `--retries` | Maximum read attempts per block (default: `5`). |
| `--resume` | Continue an interrupted download with the same settings. |
| `-h` / `--help` | Show CLI help. |

Reference mapping uses OME transforms or shape ratios assuming the same origin
and field of view. Without a reference, the same index ranges apply to each array.

Writes `<output_dir>/<remote_name>.zarr` and `meta.json`. Crops start at local
index zero; use the recorded crop origin when converting mesh coordinates.
To continue an interrupted download, repeat the same command with `--resume`.

### `fetch_roi.py` and `fetch_tree.py`

Download fitting inputs from the
[PHercParis4 dataset on Hugging Face](https://huggingface.co/buckets/scrollprize/datasets/tree/spiral/PHercParis4).
Requires `huggingface_hub` and `requests`. Set `LOCAL` in each script first:
the dataset directory for `fetch_tree.py`, its `lasagna_inputs/` subdirectory for
`fetch_roi.py`. Download the CT volume separately.

```bash
python fetch_roi.py 4000 5000 --jobs 64 --margin 320
python fetch_tree.py --list --manifest m.jsonl verified_patches unverified_patches
python fetch_tree.py --download --manifest m.jsonl --jobs 32
python fetch_tree.py --direct outer_shell tracks abs_winding.json
```

| Script / argument | Purpose |
| --- | --- |
| `fetch_roi.py <z_begin> <z_end>` | **Required.** Half-open Z range in fit voxels (9.6 micrometers). |
| `fetch_tree.py <prefixes...>` | Dataset-relative files or directories for `--list` or `--direct`. |
| `--jobs` | Concurrent download batches (ROI default: `64`; tree default: `32`). |
| `--margin` | ROI only: extra fit voxels on each side (default: `320`). |
| `--dry-run` | ROI only: report slabs without downloading chunk data. |
| `--list` | Tree only: write selected paths and sizes to a manifest. |
| `--download` | Tree only: download files from a manifest. |
| `--direct` | Tree only: list and download selected prefixes in one step. |
| `--manifest <file>` | Tree only: JSONL manifest, required for `--list` and `--download`. |
| `--exclude <text>` | Tree only: exclude matching paths in `--direct` mode; repeatable. |
| `--shard <i>:<n>` | Tree only: download shard `i` of `n`, starting at `0`. |
| `-h` / `--help` | Show CLI help. |

Use manifests for large collections and `--direct` for small selections.

## Conversion

### `squeeze_zarr.py` and `threshold_zarr.py`

Select a channel from `(C, Z, Y, X)` to produce 3D data, or threshold an array to
binary `uint8` (0/1). Both write standalone arrays chunk by chunk.

```bash
python squeeze_zarr.py "prediction.zarr" --array 0 --channel 0 --out "surface_3d.zarr"
python threshold_zarr.py "surface_3d.zarr" --threshold 127 --out "surface_binary.zarr"
```

| Argument | Purpose |
| --- | --- |
| `<source>` | **Required.** Local Zarr path or URL. |
| `--array <path>` | Required for group inputs: select one array, e.g. `2` or `group/0`. |
| `--out <path>` | **Required.** Output standalone Zarr array. |
| `--channel` | Squeeze only: first-axis channel (default: `0`); 3D input is copied unchanged. |
| `--threshold` | **Required** for threshold: values strictly above it become `1`, all others `0`. Match the input range, e.g. `127` for 0–255 or `0.5` for probabilities. |
| `--overwrite` | Replace existing output. Input and output must not overlap. |
| `-h` / `--help` | Show CLI help. |

Use `inspect_zarr.py` to find array paths first. Groups may contain different
resolutions; each conversion processes one explicit `--array`, leaving the source
and other arrays unchanged. No array is selected automatically from a group.

Skip squeeze for 3D input and threshold for existing 0/1 data. Threshold preserves
the shape and works on 3D or 4D input. Neither tool crops Z or changes the spatial
origin; inputs must already align with the dataset's voxel frame.

### `zarr_to_tiff.py`

Export a Zarr volume as a TIFF stack and individual slices in `layers/`.

```bash
python zarr_to_tiff.py "<input.zarr>" "<output_dir>" --array 0
```

| Argument | Purpose |
| --- | --- |
| `<input_dir>` | **Required.** Local Zarr group or standalone array. |
| `<output_dir>` | **Required.** Output directory, outside the input directory. |
| `--array` | Array path inside a group (default: `0`); ignored for standalone arrays. |
| `--prefix` | Slice filename prefix (default: empty). |
| `--digits` | Minimum digits in slice numbers (default: `2`). |
| `--overwrite` | Replace existing slice files. |
| `--no-compression` | Disable ZSTD compression for slice files. |
| `-h` / `--help` | Show CLI help. |

Supports `(Z, Y, X)` and `(1, Z, Y, X)`. Use `--overwrite` for repeated exports.

### `tiff_to_zarr.py`

Convert a TIFF stack in Z Y X order to a `uint8` Zarr v2 array.

```bash
python tiff_to_zarr.py "<input.tiff>" "<output.zarr>" --mul 255
```

| Argument | Purpose |
| --- | --- |
| `<input_path>` | **Required.** Input TIFF stack in Z Y X order. |
| `<output_path>` | **Required.** Output Zarr path; existing output is overwritten. |
| `--mul` | Integer multiplier after conversion to `uint8` (default: `1`). |
| `-h` / `--help` | Show CLI help. |

`--mul 255` converts binary masks from 0/1 to 0/255; omit for normal intensities.

## Other

### `inspect_zarr.py`

Show groups, array shapes, chunks and dtypes for a local path or remote URL.
By default, reads metadata only; supports Zarr v2/v3 without custom codecs.
Requires NumPy and fsspec (`fsspec[http]` for HTTP URLs).

```bash
python inspect_zarr.py "../data/scan.zarr"
python inspect_zarr.py "<url>" --path labels --attrs
python inspect_zarr.py "prediction.zarr" --path 0 --sample --check-binary
```

| Argument | Purpose |
| --- | --- |
| `<path_or_url>` | **Required.** Local directory path or remote Zarr URL. |
| `--path <path>` | Inspect only this subgroup or array. |
| `--attrs` | Include group and array attributes. |
| `--json` | Print JSON instead of a tree. |
| `--sample` | Read up to 65536 values from the first chunk; show sample min/max and up to 16 unique values. |
| `--check-binary` | Check all chunks for values 0/1; stop at the first other value. Can read the entire selected array. |
| `-h` / `--help` | Show CLI help. |

S3 browser links (`index.html#...`) are accepted.
Value sampling/checks require `zarr` and the input codecs. A 0/255 mask is not a
0/1 mask; `--sample` alone can miss non-binary values elsewhere.

## Sources

The scripts `fetch_roi.py` and `fetch_tree.py` come from the following GitHub repository:

- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/tree/d8c5f488a105286c548c99c5f7c7ad9f29e3ed14) (Commit: `d8c5f48`)
