# scrollreading


## Sources

This folder contains code from the following GitHub repository:

- [**WillStevens/scrollreading**](https://github.com/WillStevens/scrollreading/tree/62cbc21bdafbe0313fee11d781d729256a14262e) (Commit: 62cbc21)

The original, unmodified code can be found in `original_code/`.


## Getting Started

1. Create a python environment

    <!-- Create the python environment described in the `../spiral_fitting/README.md`.

    After that install the following packages:

    ```bash
    pip install "dask[array]" dask-image zarr scipy
    uv pip install python-dotenv
    ``` -->

    ```bash
    pip install uv
    uv venv
    source .venv/bin/activate
    uv pip install numpy scipy zarr "dask[array]" dask-image
    uv pip install python-dotenv
    ```

    If it doesnt work, also install the blosc2 and libtiff-dev C libraries.

2. Download a surface prediction OME zarr and the corresponding surface volume OME zarr

    <!-- Use `utils/tiff_to_zarr.py` to convert a tiff image stack to a zarr volume

    ```bash
    python utils/tiff_to_zarr.py <input.tiff> <output.zarr> --mul 255
    ``` -->

    Use `../download_data/download_zarr_from_url.py` for downloading a zarr volume and `../surface_detection/` for creating the corresponding surface prediction.

3. Get seed points

    ```bash
    python utils/get_seed_points.py <input_tiff>
    ```

    You can check the value of a seed point using:

    ```bash
    python utils/get_seed_points.py <input_tiff> <x> <y> <z>
    ```

4. Get axis

    ```bash
    python utils/get_axis.py --surface <path_to_surface_prediction> --seed <x> <y> <z>
    ```

4. Create a `.env` file with the following values (from the previous steps)

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

5. Create the output directory, and within it make the folders `surface.bp/surface` and `boundary.bp/surface` and `patches`

6. Run make

    ```bash
    make
    ```

7. Run the pipeline

    ```bash
    ./simpaper10 g 100
    ```

    If you get an error saying `Aborting, too few points added` use a larger component.

    Additional commands:

    ```bash
    ./simpaper10 c    # Check patches and create badpatches.csv
    ./simpaper10 l    # Output 3D coordinates of patch centers
    ./simpaper10 v    # Generate visit order and initial 2D placement
    ./simpaper10 h      # Refine positions using a ball-and-spring model
    ./simpaper10 f 30   # Flatten surface -> OUTPUT_DIR/patch_0.bin
    ```

    `patch_0.bin` contains binary ux, uy, vx, vy, vz data. Convert it to CSV with `bin2csv`, then to tifxyz with `csv2tifxyz`.

    ```bash
    make bin2csv
    ./bin2csv out/patch_0.bin > out/patch_0.csv

    g++ csv2tifxyz.cpp -o csv2tifxyz -ltiff
    ./csv2tifxyz out/patch_0.csv out/patch_0
    ```