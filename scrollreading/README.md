# scrollreading


## Sources

This folder contains code from the following GitHub repository:

- [**WillStevens/scrollreading**](https://github.com/WillStevens/scrollreading/tree/62cbc21bdafbe0313fee11d781d729256a14262e) (Commit: 62cbc21)

The original, unmodified code can be found in `original_code/`.


## Getting Started

1. Create a python environment

    Create the python environment described in the `../spiral_fitting/README.md`.

    After that install the following packages:

    ```bash
    pip install "dask[array]" dask-image zarr scipy
    uv pip install python-dotenv
    ```

1. Download a surface prediction OME zarr and the corresponding surface volume OME zarr

    <!-- Use `utils/tiff_to_zarr.py` to convert a tiff image stack to a zarr volume

    ```bash
    python utils/tiff_to_zarr.py <input.tiff> <output.zarr> --mul 255
    ``` -->

3. Get the seed point

    ```bash
    python utils/get_seed_point.py <input_tiff>
    ```

3. Create a `.env` file with the following values

    ```bash
    OUTPUT_DIR=
    VOLUME_ZARR=
    SURFACE_ZARR=
    SEED_X=
    SEED_Y=
    SEED_Z=
    SEED_AXIS1_X=
    SEED_AXIS1_Y=
    SEED_AXIS1_Z=
    SEED_AXIS2_X=
    SEED_AXIS2_Y=
    SEED_AXIS2_Z=
    ```


./bin2csv output/patch_0.bin > output/patch_0.csv
sudo apt install libtiff-dev
g++ csv2tifxyz.cpp -o csv2tifxyz -ltiff