# Download Data


## Sources

The `fetch_roi.py` and `fetch_tree.py` are from the following GitHub repository:

- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/tree/d8c5f488a105286c548c99c5f7c7ad9f29e3ed14) (Commit: d8c5f48)


## Scripts

### `download_zarr_from_url.py`

Downloads a 3D region from a remote Zarr volume using a URL. The region can be specified by its Z, Y, and X coordinate ranges, allowing you to download only the required part of a volume.

**Usage:**

```bash
python download_zarr_from_url.py <url_to_zarr> <output_path> \
    --region <z_start> <z_stop> <y_start> <y_stop> <x_start> <x_stop> \
    --array <array>
```

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
- `--array`: Specifies the array to convert.

### `tiff_to_zarr.py`

Converts a TIFF image stack into a Zarr volume.

**Usage:**

```bash
python utils/tiff_to_zarr.py <input.tiff> <output.zarr> --mul 255
```

The `--mul 255` option multiplies the input values by 255. This is useful when converting normalized intensity values in the range 0-1 to the range 0-255.