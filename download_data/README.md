# Download Data

## Download a 3D zarr from a url and convert it to tif

```bash
python download_zarr_from_url.py <url_to_zarr> <output_path> --region <z_start> <z_stop> <y_start> <y_stop> <x_start> <x_stop> --array <array>
python convert_zarr_to_tiff.py path/to/zarr <output_path> --array <array>
```