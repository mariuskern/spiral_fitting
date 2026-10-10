# Running the whole Pipeline


## Download a Scroll Volume (Example: Scroll 332)

All of the following commands need to be run inside the `download_data` folder.

Use the following command to download a region of Scroll 332 from the Scrolls dataset and save it to the specified output path:

```bash
python download_zarr_from_url.py \
    https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll3.zarr/ \
    <output_path> \
    --region 5000 6000 1200 2600 900 2300 \
    --array 0
```

Replace `<output_path>` with the directory where you want to save the downloaded volume.

The `--region` argument specifies the region to download using the following coordinate order:

- **Z:** 5000–6000
- **Y:** 1200–2600
- **X:** 900–2300

The `--array 0` argument selects array `0` from the Zarr dataset.

### Working with Temporary Storage on a Cluster

If you encounter file quota issues on the cluster, you can create a temporary directory using:

```bash
mktemp -d
```

The command prints the path of the newly created directory.

**Important:** Save the directory path so you can access your downloaded files in case you clear the terminal. Before moving the files to a permanent location, archive them if appropriate.

Archive the downloaded scroll by using the following command:

```bash
tar -cf 332/data.tar 332/data.zarr
```


### Preprocess data

Extract the scroll tar

```bash
tar -xvf 332.tar
```

Convert the zarr volume to a 3d tif image stack

```bash
python zarr_to_tiff.py <path_to_zarr> <output_path> --array <array> --digits <digits>
```


## Run surface detection

All of the following commands need to be run inside the `surface_detection` folder.
