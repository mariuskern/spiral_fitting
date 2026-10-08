# Download Data

## Download a 3D zarr from a url and convert it to tif

```bash
python download_zarr_from_url.py <url_to_zarr> <output_path> --region <z_start> <z_stop> <y_start> <y_stop> <x_start> <x_stop> --array <array>
python convert_zarr_to_tiff.py path/to/zarr <output_path> --array <array>

# Download a part of scroll 332
python download_zarr_from_url.py https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll3.zarr/ /<output_path> --region 5000 5500 1600 2200 1300 1900 --array 0
```

## Convert a tiff image stack to a zarr volume

Use `utils/tiff_to_zarr.py` to convert a tiff image stack to a zarr volume

```bash
python utils/tiff_to_zarr.py <input.tiff> <output.zarr> --mul 255
```