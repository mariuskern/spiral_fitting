# Spiral Fitting

This folder contains the code and configuration required to run the spiral fitting pipeline.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`fit_spiral.py`](fit_spiral.py) | Entry point that loads `.env` and starts the bundled fitter. |
| [`generate_tracks.py`](generate_tracks.py) | Extract tracks and build their crossing cache. |
| [`generate_graph_patches.py`](generate_graph_patches.py) | Grow graph patches (tifxyz) from tracks using `grow_track_graph`. |
| [`squeeze_preciction.py`](squeeze_preciction.py) | Inspect a prediction and copy one channel of a 4D array to a 3D Zarr. |

## Getting Started

Run the following commands from the `spiral_fitting/` directory. From the
repository root, enter it first:

```bash
cd spiral_fitting
```

Relative paths in `.env` and command-line arguments are resolved against
`spiral_fitting/` when starting the pipeline from this directory.

1. Create the python environment:

    ```bash
    conda create -n spiral-fitting python=3.14
    conda activate spiral-fitting

    pip install uv

    uv pip install -e villa_8aa507c
    uv pip install python-dotenv
    ```

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

    All available configuration options are defined in
    [`villa_8aa507c/config.py`](villa_8aa507c/config.py).

    Add the `FIT_SPIRAL_CONFIG_OVERRIDES` variable to `.env` in the repository
    root (`../.env` from this folder), which `fit_spiral.py` loads automatically:

    ```bash
    FIT_SPIRAL_CONFIG_OVERRIDES='{"z_begin": 4000, "z_end": 5000, "optimizer_num_training_steps": 5000, "dense_spacing_mode": "grad_mag"}'
    ```

    The values specified in `FIT_SPIRAL_CONFIG_OVERRIDES` override the corresponding default values from `villa_8aa507c/config.py`.

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


4. Run the pipeline

    ```bash
    python fit_spiral.py --dataset <path_to_dataset> --scroll-spec <path_to_scroll_spec>
    ```

    The scroll spec should be provided as a JSON file. The `--scroll-spec` argument is optional if a `spiral-scroll.json` file is present in the dataset folder.

    Use the metadata for your scroll, including its actual voxel size in
    micrometers and outward winding sense. The following is a template;
    replace the placeholder before saving it as JSON. The `paths` key is optional.

    ```json
    {
        "schema_version": 1,
        "name": "PHercParis4",
        "voxel_size_um": 0.08,
        "spiral_outward_sense": "CW",
        "paths": {
            "tracks_dbm": "tracks/tracks.dbm"
        }
    }
    ```


## Generating the spiral fit input

The spiral fit can use *tracks* as input. They are skeletonized 2D curves through a binarized surface prediction. This section describes how to generate them from a volume. The steps assume that the python environment from `Getting Started` is active.

All paths below are examples. Replace them with your own.

1. Run the surface prediction as described in the [Surface Prediction README](../surface_prediction/README.md).

2. Check the prediction and convert it to 3D

    The track extraction expects a 3D array in `(Z, Y, X)` order. The finalized prediction may contain an additional channel axis `(C, Z, Y, X)`, which makes the extraction fail with `DimensionError: Only 1D, 2D, and 3D arrays supported. Got: 4`.

    First inspect the prediction. This prints the shape, the data type and the values of the middle slice for each channel:

    ```bash
    python squeeze_preciction.py <path_to_prediction_run>/final_output.zarr
    ```

    Check that the values are binary (0/1) and that the fraction of values `> 0` is plausible. Then write a 3D copy:

    ```bash
    python squeeze_preciction.py <path_to_prediction_run>/final_output.zarr \
        --out <path_to_prediction_run>/surface_3d.zarr
    ```

    Useful options:

    | Option | Description |
    | --- | --- |
    | `--channel N` | Channel that is copied (default `0`). |
    | `--threshold V` | Binarizes the output (`> V` becomes 1, otherwise 0). Only needed if the prediction was not thresholded in step 1. |

    If the prediction is already a binary 3D array, use it directly in step 3;
    the helper does not write a copy of an already 3D array. Its `--threshold`
    option applies only when copying a channel from a 4D array.

    The helper does not crop the z axis. If the input prediction covers the full
    volume, `--z-min` / `--z-max` below refer to the original z coordinates.
    A cropped prediction must first be aligned with the dataset's voxel frame.

3. Extract the tracks

    ```bash
    python generate_tracks.py \
        --predictions <path_to_prediction_run>/surface_3d.zarr \
        --out <path_to_dataset>/tracks/tracks.dbm \
        --z-min 4000 --z-max 5000
    ```

    The script runs `extract_surface_tracks` followed by `build_track_crossings` (the latter can be skipped with `--skip-crossings`). Both are started as modules of the `villa_8aa507c` package (`python -m villa_8aa507c.<module>`), because they use relative imports. If the package folder has a different name or location, pass it with `--package-dir`.

    Further options:

    | Option | Description |
    | --- | --- |
    | `--backend` | Backend of the track extraction, see `python -m villa_8aa507c.extract_surface_tracks --help`. |
    | `--gpus` | GPUs used by the track extraction. |
    | `--skip-crossings` | Skip the crossing cache. The fit then computes the crossings in memory. |

    The following files are created next to `tracks.dbm`:

    ```text
    tracks/
    ├── tracks.dbm
    ├── tracks.dbm.vctracks/        # packed store, loaded fastest by the fit
    └── tracks.dbm.crossings.npz    # crossing cache (step 3 of the script)
    ```

    > **Notes:**
    > - The fast GPU backend is only available on Linux x86-64 with an NVIDIA GPU of compute capability >= 8.0. Otherwise (e.g. on Windows) the much slower CPU skeletonizer is used. Test with a small z range first.
    > - The extraction resumes: slabs that are already in the database are skipped, so an interrupted run can be restarted with the same arguments.
    > - The tracks must be in the same voxel frame as the other inputs of the fit (umbilicus, patches, `z_begin` / `z_end`) and should cover at least the z range that is fitted. Tracks outside the fit window are dropped when loading.

4. Use the tracks in the fit

    Tracks are disabled by default. Enable them in the `.env` file:

    ```bash
    FIT_SPIRAL_CONFIG_OVERRIDES='{"z_begin": 4000, "z_end": 5000, "input_use_tracks": true}'
    ```

    The fit looks for `tracks/2um_ds2_ps256_surf_v2.dbm` in the dataset. If the database has a different name, or if there are several databases in `tracks/`, set the path in `spiral-scroll.json`:

    ```json
    {
        "schema_version": 1,
        "name": "PHercParis4",
        "voxel_size_um": 0.08,
        "spiral_outward_sense": "CW",
        "paths": {
            "tracks_dbm": "tracks/tracks.dbm"
        }
    }
    ```

    Then run the fit as described in `Getting Started`.


5. Grow graph patches from tracks

    Run this command from `spiral_fitting/` after extracting the tracks and building
    their crossing cache:

    ```bash
    python generate_graph_patches.py \
        --tracks <path_to_dataset>/tracks/tracks.dbm \
        --out <path_to_dataset>/graph_patches \
        --random-count <number_of_patches> \
        --workers <number_of_workers>
    ```

    Replace all placeholders with your paths and positive integer counts. The helper
    runs `villa_8aa507c.grow_track_graph` and derives the required
    `tracks.dbm.vctracks` and `tracks.dbm.crossings.npz` paths from `--tracks`.
    Generate these sidecars with `generate_tracks.py` without `--skip-crossings`.

    Use `--seeds "<seed_track_rows>"` instead of `--random-count` to select explicit
    seed tracks. Additional `grow_track_graph` options can be passed after `--`;
    see `python -m villa_8aa507c.grow_track_graph --help` for their format and defaults.

    To use the resulting patches in the fit, set `paths.verified_patches` to
    `"graph_patches"` in the dataset's `spiral-scroll.json` and keep
    `input_use_verified_patches` enabled.


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

    IMAGE="$HOME/volume-cartographer.sif"

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
    export PATH="$HOME/bin:$PATH"
    ```

    This change only applies to the current shell session. To make it permanent, add the same line to your `~/.bashrc`:

    ```bash
    echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
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
    python villa_8aa507c/render_ink.py --volume <path_to_dataset> <path_to_fit_spiral_output>/meshes/fitted/ --lasagna-dir lasagna
    ```

    Add `--lasagna-device cpu` to run it on a cpu


## Sources

This folder contains code from the following GitHub repository:

- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/tree/8aa507c1215f1fa8758f13d0538f7af9710e02d7) (Commit: 8aa507c)

The original, unmodified code is in `.original_code/`.

The following subfolders contain code from additional sources:

`.out/`

This folder contains code from the following GitHub repository:

- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/tree/d8c5f488a105286c548c99c5f7c7ad9f29e3ed14) (Commit: d8c5f48)

The original, unmodified code can be found in `original_code/`.

`.out/spiral_7769da8/`:

This folder combines the spiral-fitting code from **ScrollPrize/villa** with a faster implementation from **7jycwjmbfn-eng/spiral-fit-consumer-gpu**. This version of the spiral fitting is working.

- [**7jycwjmbfn-eng/spiral-fit-consumer-gpu**](https://github.com/7jycwjmbfn-eng/spiral-fit-consumer-gpu/commit/f189740211f462193973055a32e3269c03301587) (Commit: f189740)
- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/tree/7769da8cf2233310570608feecc127066a7c0c7c/volume-cartographer/scripts/spiral) (Commit: 7769da8)
- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/blob/9761f14773a3ed41f2459bbf689a8f1998a656ed/volume-cartographer/scripts/spiral/spiral_progress.py) (Commit: 9761f14) (only `villa/spiral_progress.py`)

`.out/spiral_6732d35/`:

This folder attempts to combine a newer version of the spiral-fitting code from **ScrollPrize/villa** with the faster implementation from **7jycwjmbfn-eng/spiral-fit-consumer-gpu**. This version is **not working**.

- [**7jycwjmbfn-eng/spiral-fit-consumer-gpu**](https://github.com/7jycwjmbfn-eng/spiral-fit-consumer-gpu/commit/f189740211f462193973055a32e3269c03301587) (Commit: f189740)
- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/tree/6732d3587bc3224a8a69a15886fb68c6e57a9342/volume-cartographer/scripts/spiral) (Commit: 6732d35) (doesn't work yet)
- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/blob/9761f14773a3ed41f2459bbf689a8f1998a656ed/volume-cartographer/scripts/spiral/spiral_progress.py) (Commit: 9761f14) (only `villa/spiral_progress.py`)

> **Note:** The `.out/`, `.out/spiral_7769da8/` and `.out/spiral_6732d35/` folders are deprecated and should no longer be used. A newer working version of the spiral-fitting code is available in the current folder (`./`).
