# Spiral Fitting

This folder contains the code and configuration required to run the spiral fitting pipeline.


## Sources

This folder contains code from the following GitHub repository:

- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/tree/d8c5f488a105286c548c99c5f7c7ad9f29e3ed14) (Commit: d8c5f48)

The original, unmodified code can be found in `original_code/`.

The following subfolders contain code from additional sources:

`spiral_7769da8/`:

This folder combines the spiral-fitting code from **ScrollPrize/villa** with a faster implementation from **7jycwjmbfn-eng/spiral-fit-consumer-gpu**. This version of the spiral fitting is working.

- [**7jycwjmbfn-eng/spiral-fit-consumer-gpu**](https://github.com/7jycwjmbfn-eng/spiral-fit-consumer-gpu/commit/f189740211f462193973055a32e3269c03301587) (Commit: f189740)
- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/tree/7769da8cf2233310570608feecc127066a7c0c7c/volume-cartographer/scripts/spiral) (Commit: 7769da8)
- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/blob/9761f14773a3ed41f2459bbf689a8f1998a656ed/volume-cartographer/scripts/spiral/spiral_progress.py) (Commit: 9761f14) (only `villa/spiral_progress.py`)

`spiral_6732d35/`:

This folder attempts to combine a newer version of the spiral-fitting code from **ScrollPrize/villa** with the faster implementation from **7jycwjmbfn-eng/spiral-fit-consumer-gpu**. This version is **not working**.

- [**7jycwjmbfn-eng/spiral-fit-consumer-gpu**](https://github.com/7jycwjmbfn-eng/spiral-fit-consumer-gpu/commit/f189740211f462193973055a32e3269c03301587) (Commit: f189740)
- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/tree/6732d3587bc3224a8a69a15886fb68c6e57a9342/volume-cartographer/scripts/spiral) (Commit: 6732d35) (doesn't work yet)
- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/blob/9761f14773a3ed41f2459bbf689a8f1998a656ed/volume-cartographer/scripts/spiral/spiral_progress.py) (Commit: 9761f14) (only `villa/spiral_progress.py`)

> **Note:** The `spiral_7769da8/` and `spiral_6732d35/` folders are deprecated and should no longer be used. A newer working version of the spiral-fitting code is available in the current folder (`./`).


## Getting Started

1. Create the python environment:

    ```bash
    conda create -n spiral-fitting python=3.14
    conda activate spiral-fitting

    pip install uv

    uv pip install -e .
    uv pip install torch torchvision
    uv pip install -e volume-cartographer
    uv pip install python-dotenv
    ```

    If `uv pip install -e volume-cartographer` fails, you can try the following commands on the cluster. This setup worked for me, although I cannot guarantee that the list of commands is complete or that all of them are necessary in every environment.


    ```bash
    module load gcc

    conda install -c conda-forge ceres-solver
    conda install -c conda-forge metis
    conda install -c conda-forge opencv
    conda install -c conda-forge nlohmann_json
    conda install -c conda-forge curl
    conda install -c conda-forge libtiff
    conda install -c conda-forge blosc
    conda install -c conda-forge lz4
    export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"

    mkdir -p ~/boost-build
    cp -a /apps/openfoam/v2506/ThirdParty-v2506/sources/boost/boost_1_74_0 ~/boost-build/
    cd ~/boost-build/boost_1_74_0
    # ./bootstrap.sh
    # export BOOST_LOCAL="$HOME/boost-build/boost-1.74-local"
    ./bootstrap.sh --with-libraries=program_options
    ./b2 -j8 --with-program_options --prefix="$HOME/boost-build/install" install

    cd ~/spiral_fitting/spiral_fitting

    BOOST_ROOT="$HOME/boost-build/install"

    uv pip install -e volume-cartographer \
        -Ccmake.define.CMAKE_CXX_FLAGS="-I$BOOST_ROOT/include" \
        -Ccmake.define.CMAKE_EXE_LINKER_FLAGS="-L$BOOST_ROOT/lib -Wl,-rpath,$BOOST_ROOT/lib" \
        -Ccmake.define.CMAKE_SHARED_LINKER_FLAGS="-L$BOOST_ROOT/lib -Wl,-rpath,$BOOST_ROOT/lib" \
        -Ccmake.define.METIS_INCLUDE_DIR="$CONDA_PREFIX/include" \
        -Ccmake.define.METIS_LIBRARY="$CONDA_PREFIX/lib/libmetis.so" \
        -Ccmake.define.OpenCV_DIR="$CONDA_PREFIX/lib/cmake/opencv4" \
        -Ccmake.define.CURL_INCLUDE_DIR="$CONDA_PREFIX/include" \
        -Ccmake.define.CURL_LIBRARY="$CONDA_PREFIX/lib/libcurl.so" \
        -Ccmake.define.TIFF_INCLUDE_DIR="$CONDA_PREFIX/include" \
        -Ccmake.define.TIFF_LIBRARY="$CONDA_PREFIX/lib/libtiff.so" \
        -Ccmake.define.BLOSC_INCLUDE_DIR="$CONDA_PREFIX/include" \
        -Ccmake.define.BLOSC_LIBRARY="$CONDA_PREFIX/lib/libblosc.so"
    ```

    <!-- ```bash
    pip install uv
    uv sync
    uv pip install torch torchvision
    uv pip install -e volume-cartographer
    uv pip install python-dotenv

    source .venv/bin/activate
    ``` -->

2. Download the dataset:

    ```bash
    rclone copy :http: ./spiral_datasets/phercparis4 \
    --http-url https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/ \
    --transfers 25 \
    --checkers 2 \
    --retries 20 \
    -P
    ```

    Alternatively, the dataset can be downloaded from Hugging Face:

    https://huggingface.co/buckets/scrollprize/datasets/tree/spiral/PHercParis4

    The helper scripts `download_data/fetch_roi.py` and `download_data/fetch_tree.py` can be used to download the dataset. Set the `LOCAL` constant at the beginning of each script to the desired dataset location before running it.

    ```bash
    python ../download_data/fetch_roi.py 4000 17000

    python ../download_data/fetch_tree.py --list --manifest m.jsonl verified_patches unverified_patches
    python ../download_data/fetch_tree.py --download --manifest m.jsonl --shard 0:4 --jobs 32
    python ../download_data/fetch_tree.py --direct outer_shell fibers tracks abs_winding.json patch-overlap-pcls.json relative_windings.json same_windings.json umbilicus.json
    ```

3. Adjust the fitting configuration

    Before running the pipeline, you can customize the fitting configuration by setting the `FIT_SPIRAL_CONFIG_OVERRIDES` environment variable to a JSON object.

    For example:

    ```json
    {
        "z_begin": 4000,
        "z_end": 5000,
        "optimizer_num_training_steps": 5000,
        "dense_spacing_mode": "phase"
    }
    ```

    All available configuration options are defined in `config.py`.

    Add the `FIT_SPIRAL_CONFIG_OVERRIDES` variable to your `.env` file:

    ```bash
    FIT_SPIRAL_CONFIG_OVERRIDES='{"z_begin": 4000, "z_end": 5000, "optimizer_num_training_steps": 5000, "dense_spacing_mode": "phase"}'
    ```

    The values specified in `FIT_SPIRAL_CONFIG_OVERRIDES` override the corresponding default values from `config.py`.

    The following environment variables can be used to control the pipeline run and override default settings:

    | Variable | Description |
    | --- | --- |
    | `FIT_SPIRAL_CONFIG_OVERRIDES` | JSON dictionary for overriding values from the fitting configuration. For example: `'{"optimizer_num_training_steps": 10000}'` |
    | `FIT_SPIRAL_OUT_DIR` | Parent directory in which the run directory is created. Defaults to `./out`. |
    | `FIT_SPIRAL_RUN_DIR` | Specifies the exact directory to use for the run instead of generating a run directory automatically. |
    | `FIT_SPIRAL_RUN_TAG` | Adds a custom tag to the run directory and generated mesh names. |
    | `FIT_SPIRAL_RESUME_PATH` / `FIT_SPIRAL_RESUME_STEP` | Resume a previous run from a checkpoint. |
    | `FIT_SPIRAL_CACHE_DIR` | Directory used to store the cache of preprocessed inputs. Defaults to `~/.cache/vc3d/spiral`. |
    | `WANDB_MODE` | Controls Weights & Biases logging. Set to `online` to enable logging of losses and visualizations. Defaults to `disabled`. |

    The preprocessing cache is content-addressed, so it can be shared across different datasets. Keeping the cache between runs can significantly speed up subsequent runs.

    The cache location can be changed using `FIT_SPIRAL_CACHE_DIR` or the `--cache DIR` command-line option.

    > **Note:** Avoid using environment variables in paths, as they may not work correctly in all cases.

    <!-- Before running the pipeline, edit the configuration in `config.py`:

    - Around **line 304/305**: set z_start and z_end.
    - Around **line 311**: set the number of training steps. -->

4. Run the pipeline

    ```bash
    python fit_spiral.py --dataset <path_to_dataset> --scroll-spec <path_to_scroll_spec>
    ```

    The scroll spec should be provided as a JSON file. The `--scroll-spec` argument is optional if a `spiral-scroll.json` file is present in the dataset folder.

    The file should contain the following line. The `paths` key is optional.

    ```json
    {
        "schema_version": 1,
        "name": "PHercParis4",
        "voxel_size_um": 0.08,
        "spiral_outward_sense": "CW",
        "paths": {
            "tracks_dbm": "<path>" # optional
        }
    }
    ```


## Installing the VC binaries

This guide installs the VC binaries in your home directory using an Apptainer container. You can choose a different location if you prefer.

1. Download the Apptainer container

    Pull the Volume Cartographer container from GitHub Container Registry:

    ```bash
    apptainer pull ~/volume-cartographer.sif docker://ghcr.io/scrollprize/villa/volume-cartographer:edge
    ```

    You can test the container by starting a shell:

    ```bash
    apptainer shell ~/volume-cartographer.sif
    ```

    The binaries can be found in `/usr/local/bin/`.

    Alternatively, you can run a binary directly:

    ```bash
    apptainer exec ~/volume-cartographer.sif <name_of_binary>
    ```

2. Create a wrapper script

    Create a `bin` directory in your home directory and a wrapper script called `vc`:

    ```bash
    cd ~
    mkdir -p bin
    cd bin
    touch vc
    nano vc
    ```

    Add the following content to `vc`:

    ```bash
    #!/bin/bash

    IMAGE="~/volume-cartographer.sif"

    exec apptainer exec "$IMAGE" "$(basename "$0")" "$@"
    ```

    The script uses the name with which it was called (`$0`) to determine which binary to execute inside the container. This allows the same wrapper to be used for all VC binaries.

3. Make the wrapper executable

    ```bash
    cd ~/bin
    chmod +x vc
    ```

4. Create symlinks for the VC binaries

    Create a symlink for each binary you want to use:

    <!-- ```bash
    cd ~/bin

    ln -s vc vc_render_tifxyz
    ln -s vc vc_tifxyz_trim
    ln -s vc vc_tifxyz2obj
    ln -s vc flatboi
    ln -s vc vc_obj2tifxyz
    ln -s vc vc_obj_uv_lift
    ``` -->

    <!-- ```bash
    ln -s ~/bin/vc ~/bin/vc_render_tifxyz
    ln -s ~/bin/vc ~/bin/vc_tifxyz_trim
    ln -s ~/bin/vc ~/bin/vc_tifxyz2obj
    ln -s ~/bin/vc ~/bin/flatboi
    ln -s ~/bin/vc ~/bin/vc_obj2tifxyz
    ln -s ~/bin/vc ~/bin/vc_obj_uv_lift
    ``` -->

    ```bash
    BIN_DIR=~/bin
    VC_BIN="$BIN_DIR/vc"

    ln -s "$VC_BIN" "$BIN_DIR/flatboi"
    ln -s "$VC_BIN" "$BIN_DIR/vc_add_ignore_label"
    ln -s "$VC_BIN" "$BIN_DIR/vc_calc_surface_metrics"
    ln -s "$VC_BIN" "$BIN_DIR/vc_create_segment_mask"
    ln -s "$VC_BIN" "$BIN_DIR/vc_cut_windings"
    ln -s "$VC_BIN" "$BIN_DIR/vc_diffuse_winding"
    ln -s "$VC_BIN" "$BIN_DIR/vc_flatten"
    ln -s "$VC_BIN" "$BIN_DIR/vc_gen_normalgrids"
    ln -s "$VC_BIN" "$BIN_DIR/vc_grow_seg_from_seed"
    ln -s "$VC_BIN" "$BIN_DIR/vc_grow_seg_from_segments"
    ln -s "$VC_BIN" "$BIN_DIR/vc_merge_tifxyz"
    ln -s "$VC_BIN" "$BIN_DIR/vc_ngrids"
    ln -s "$VC_BIN" "$BIN_DIR/vc_obj2tifxyz"
    ln -s "$VC_BIN" "$BIN_DIR/vc_obj2tifxyz_legacy"
    ln -s "$VC_BIN" "$BIN_DIR/vc_objrefine"
    ln -s "$VC_BIN" "$BIN_DIR/vc_project_tifxyz"
    ln -s "$VC_BIN" "$BIN_DIR/vc_render_tifxyz"
    ln -s "$VC_BIN" "$BIN_DIR/vc_render_video"
    ln -s "$VC_BIN" "$BIN_DIR/vc_rgb2tifxyz"
    ln -s "$VC_BIN" "$BIN_DIR/vc_seg_add_overlap"
    ln -s "$VC_BIN" "$BIN_DIR/vc_thinning"
    ln -s "$VC_BIN" "$BIN_DIR/vc_tifxyz"
    ln -s "$VC_BIN" "$BIN_DIR/vc_tifxyz2obj"
    ln -s "$VC_BIN" "$BIN_DIR/vc_tifxyz2rgb"
    ln -s "$VC_BIN" "$BIN_DIR/vc_tifxyz2zarr_sparse"
    ln -s "$VC_BIN" "$BIN_DIR/vc_tifxyz_gengt"
    ln -s "$VC_BIN" "$BIN_DIR/vc_tifxyz_inp_mask"
    ln -s "$VC_BIN" "$BIN_DIR/vc_tifxyz_mmap_prepare"
    ln -s "$VC_BIN" "$BIN_DIR/vc_tifxyz_trim"
    ln -s "$VC_BIN" "$BIN_DIR/vc_tifxyz_winding"
    ln -s "$VC_BIN" "$BIN_DIR/vc_transform_geom"
    ln -s "$VC_BIN" "$BIN_DIR/vc_visualize"
    ln -s "$VC_BIN" "$BIN_DIR/vc_volpkg_convert"
    ln -s "$VC_BIN" "$BIN_DIR/vc_zarr_recompress"
    ln -s "$VC_BIN" "$BIN_DIR/vc_zarr_to_tiff"
    ```

    For example, when you run:

    ```bash
    vc_tifxyz_trim
    ```

    the wrapper effectively executes:

    ```bash
    apptainer exec ~/volume-cartographer.sif vc_tifxyz_trim
    ```

    Similarly:

    ```bash
    flatboi
    ```

    will execute the `flatboi` binary inside the container.

5. Add `~/bin` to your `PATH`

    For the commands to be available from anywhere, add `~/bin` to your `PATH`:

    ```bash
    export PATH="~/bin:$PATH"
    ```

    This change only applies to the current shell session. To make it permanent, add the same line to your `~/.bashrc`:

    ```bash
    echo 'export PATH="~/bin:$PATH"' >> ~/.bashrc
    ```

    Then reload your shell configuration:

    ```bash
    source ~/.bashrc
    ```

6. Test the installation

    You should now be able to run the VC binaries directly:

    ```bash
    vc_render_tifxyz --help
    vc_tifxyz_trim --help
    ```

    If these commands work, the VC binaries are successfully accessible from your shell without having to manually invoke Apptainer.


## Getting started with `render_ink.py`

The python environment should have already been created as described in `Getting Started` and the **VC binaries** should have been installed as described in `Installing the VC binaries`.

1. Download the dataset:

    ```bash
    wget https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1B.7z
    ```

2. Unpack the dataset

    ```bash
    conda install -c conda-forge 7zip
    
    7z x 54keV_7.91um_Scroll1B.7z

    mkdir extracted
    tar -xf 54keV_7.91um_Scroll1B -C extracted
    ```

    Optional: Use `tar.zst` for faster unpacking:

    ```bash
    tar -I 'zstd -T0' -cf <dataset>.tar.zst <dataset>/
    tar -I 'zstd -T0' -xf <dataset>.tar.zst
    ```

3. Run `render_ink.py`

    ```bash
    python render_ink.py --volume <path_to_dataset> <path_to_fit_spiral_output>/meshes/fitted/ --lasagna-dir lasagna
    ```

    Add `--lasagna-device cpu` to run it on a cpu
