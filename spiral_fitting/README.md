# Spiral Fitting

This folder contains the code and configuration required to run the spiral fitting pipeline.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`fit_spiral.py`](fit_spiral.py) | Entry point that loads `.env` and starts the bundled fitter. |
| [`generate_tracks.py`](generate_tracks.py) | Extract tracks and build their crossing cache. |
| [`generate_fiber_directions.py`](generate_fiber_directions.py) | Export fiber direction samples from an existing prediction. |
| [`generate_graph_patches.py`](generate_graph_patches.py) | Grow graph patches (tifxyz) from tracks using `grow_track_graph`. |
| [`squeeze_preciction.py`](squeeze_preciction.py) | Inspect a prediction and copy one channel of a 4D array to a 3D Zarr. |
| [`visualize_meshes.py`](visualize_meshes.py) | Overlay winding meshes on CT cross-sections with a shared colorbar. |

## Getting Started

Run the following commands from the `spiral_fitting/` directory.

Relative paths in `.env` and command-line arguments are resolved against `spiral_fitting/` when starting the pipeline from this directory.

1. Create the python environment:

    ```bash
    conda create -n spiral-fitting python=3.14
    conda activate spiral-fitting

    pip install uv

    uv pip install -e villa_8aa507c
    uv pip install python-dotenv
    ```

2. Download the dataset:

    Datasets can be found here:
    - [https://dl.ash2txt.org/datasets/spiral_datasets/](https://dl.ash2txt.org/datasets/spiral_datasets/)
    - [https://huggingface.co/buckets/scrollprize/datasets/tree/spiral](https://huggingface.co/buckets/scrollprize/datasets/tree/spiral)


    Download a dataset from dl.ash2txt.org using:

    ```bash
    rclone copy :http: ./spiral_datasets/phercparis4 \
    --http-url https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/ \
    --transfers 25 \
    --checkers 4 \
    --retries 20 \
    -P
    ```

    To update an already downloaded dataset, rerun the command with `sync` instead of `copy` (this also removes local files that no longer exist on the server).

    Use `mktemp -d` to create a temporary folder without a file limit when working on the HPC Cluster.

    Alternatively, the dataset can be downloaded from Hugging Face:

    https://huggingface.co/buckets/scrollprize/datasets/tree/spiral/PHercParis4

    The helper scripts `download_data/fetch_roi.py` and `download_data/fetch_tree.py` can be used to download the dataset. Set the `LOCAL` constant at the beginning of each script to the desired dataset location before running it.

    ```bash
    python ../download_data/fetch_roi.py 4000 17000

    python ../download_data/fetch_tree.py --list --manifest m.jsonl verified_patches unverified_patches
    python ../download_data/fetch_tree.py --download --manifest m.jsonl --shard 0:4 --jobs 32
    python ../download_data/fetch_tree.py --direct outer_shell fibers tracks abs_winding.json patch-overlap-pcls.json relative_windings.json same_windings.json umbilicus.json
    ```

    The corresponding CT volume for the PHercParis4 spiral-fitting dataset is `20260411134726-2.400um-0.2m-78keV-masked.zarr`. The annotations use **9.6 µm voxels**, corresponding to level `2` of this **2.4 µm** scan; multiply their coordinates by `4` to obtain level-0 coordinates. Use `voxel_size_um: 9.6` for a fit using these annotations. The CT scan is available separately from the fitting inputs:
    - [CT volume](https://vesuvius-challenge-open-data.s3.amazonaws.com/index.html#PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/) (CT scan corresponding to the spiral-fitting dataset)
    - [Spiral dataset](https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/) (PHercParis4 spiral-fitting inputs)
    - [Spiral dataset README](https://scrollprize.org/data/datasets/spiral-input-PHercParis4-README.md) (contents and coordinate conventions of the spiral-fitting dataset)
    - [Spiral dataset overview](https://scrollprize.org/data_datasets#spiral-input-2026-07) (page linking to the dataset and its README)

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

    Tracks are disabled by default. To use them in the fit, add `"input_use_tracks": true` to your existing `FIT_SPIRAL_CONFIG_OVERRIDES` object in `.env`. For example:

    ```bash
    FIT_SPIRAL_CONFIG_OVERRIDES='{"z_begin": 4000, "z_end": 5000, "input_use_tracks": true}'
    ```

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
        "voxel_size_um": 9.6,
        "spiral_outward_sense": "CW",
        "paths": {
            "tracks_dbm": "tracks/tracks.dbm"
        }
    }
    ```

    When using tracks, set `paths.tracks_dbm` to the database path relative to
    the dataset folder, as shown above. The default path is
    `tracks/2um_ds2_ps256_surf_v2.dbm`; set the path explicitly if your database
    has a different name or there are several databases in `tracks/`.


5. Visualize fitted meshes on CT slices

    ```bash
    python visualize_meshes.py \
      ../out/53keV_7.91um_Scroll3/spiral_fitting/winding_model/2026-11-26_53keV_7.91um_Scroll3_slice-0-1000_410-patch \
      --ct ../data/scrolls/53keV_7.91um_Scroll3/53keV_7.91um_Scroll3.zarr \
      --num-slices 6 \
      --mesh-range 10 36 \
      --ct-window 0 255 \
      --out ../out/53keV_7.91um_Scroll3/mesh_diagnostics/colored_sections.png
    ```

    | Argument | Description |
    | --- | --- |
    | `meshes` | Required path to a fit run, `meshes/`, or `meshes/fitted/`. |
    | `--ct PATH` | Required CT Zarr array or group. Must use the same origin and voxel resolution as the meshes. |
    | `--ct-dataset NAME` | Array inside the CT group (default: `0`); ignored for a direct array. |
    | `--num-slices N` | Number of evenly spaced sections (default: `6`). Mutually exclusive with `--slices`. |
    | `--slices Z [Z ...]` | Explicit CT Z indices. |
    | `--z-range MIN MAX` | Inclusive Z range (default: overlap of meshes and CT). |
    | `--meshes N [N ...]` | Select winding numbers. Mutually exclusive with `--mesh-range`. |
    | `--mesh-range MIN MAX` | Inclusive winding-number range (default: all meshes). |
    | `--mesh-step N` | Show every Nth mesh in the selected, sorted list (default: `1`). |
    | `--spliced` | Use `wNNN_spliced` instead of `wNNN`. |
    | `--columns N` | Maximum subplot columns (default: `3`). |
    | `--cmap NAME` | Mesh color gradient (default: `turbo`). |
    | `--colorbar-step N` | Label every Nth selected mesh (default: at most 16 labels). |
    | `--line-width VALUE` | Mesh line width (default: `0.8`). |
    | `--alpha VALUE` | Mesh opacity from `0` to `1` (default: `0.9`). |
    | `--ct-window MIN MAX` | Grayscale limits (default: first displayed slice's nonzero 1st/99.5th percentiles). |
    | `--xy-range XMIN XMAX YMIN YMAX` | Zoom all panels to the same XY rectangle in CT voxels. |
    | `--out PATH` | Save an image instead of opening a Matplotlib window. |
    | `--dpi N` | Export resolution (default: `160`). |
    | `-h`, `--help` | Show usage and argument help. |


## Generating the spiral fit input

The spiral fit can use *tracks* as input. They are skeletonized 2D curves through a binarized surface prediction. This section describes how to generate them from a volume. The steps assume that the python environment from `Getting Started` is active.

Scroll volumes for generating your own inputs can be found here:

- [Data Browser](https://scrollprize.org/data_browser) — browse available scrolls and their downloads, including [PHercParis4](https://scrollprize.org/data_browser/PHercParis4).
- [Scroll data server](https://dl.ash2txt.org/) — another source for scroll volumes and related data.

All paths below are examples. Replace them with your own.

1. Generate the surface prediction

    Follow the [surface prediction README](../surface_prediction/README.md).

2. Inspect the prediction and convert it to 3D if needed

    The track extraction expects a 3D array in `(Z, Y, X)` order. The finalized prediction may contain an additional channel axis `(C, Z, Y, X)`.

    Use the following command to inspect the array shape, data type, and a sample of values from the middle slice of each channel:

    ```bash
    python squeeze_preciction.py <path_to_prediction_run>/final_output.zarr
    ```

    If the prediction has a channel axis, use the following command to extract one channel and save it as a 3D array. By default, channel 0 is selected:

    ```bash
    python squeeze_preciction.py <path_to_prediction_run>/final_output.zarr --out <path_to_prediction_run>/surface_3d.zarr
    ```

    To binarize the selected channel in the same step, add `--threshold 127` (or another threshold appropriate for your prediction).

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

    To use the generated tracks in the fit, configure their path and enable
    track input as described in [Getting Started](#getting-started).


4. Grow graph patches from tracks

    Run this command from `spiral_fitting/` after extracting the tracks and building
    their crossing cache:

    ```bash
    python generate_graph_patches.py \
        --tracks <path_to_dataset>/tracks/tracks.dbm \
        --out <path_to_dataset>/graph_patches \
        --random-count 1000 \
        --workers 16
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


5. Generate fiber direction samples

    Requires an existing fiber prediction: download one from
    `<scroll>/representations/predictions/fibers/` in the open-data bucket or
    create one using the [Lasagna guide](../lasagna/README.md).
    It writes `presence`, `nx` and `ny` channels plus a `.lasagna.json` manifest. Use a local manifest path or an HTTP(S) URL; relative Zarr paths are resolved against the manifest.

    From `spiral_fitting/`, in the fitting environment:

    ```bash
    pip install requests
    python generate_fiber_directions.py /path/to/fiber.lasagna.json ../data/fiber_directions.npz --z-roi 0,800 --output-scale 1 --presence-threshold 160 --cell-size 2
    ```

    `--z-roi` is half-open in fit coordinates; `--output-scale` is scan `/0`
    voxels per fit voxel: fit level `/0` = `1`, `/1` = `2`, `/2` = `4`.
    This refers to the fit resolution, not the fiber inference output. Match the fit's coordinate
    origin and scale. Threshold `160` filters Presence values on the 0-255 scale;
    `--cell-size 2` keeps one sample per 2-voxel cell in prediction space.
    Use `--overwrite` to replace an existing export.

    Set `paths.fiber_directions` in the dataset's `spiral-scroll.json` to the
    exported file (relative to the dataset), then enable
    `input_use_fiber_directions=true` and `loss_weight_fiber_directions>0`.
    The loss constrains sheet orientation, not winding assignment.


6. Create umbilicus.json

    The `umbilicus.json` contains points along the centerline of the wooden winding core, in dataset voxel coordinates. It should contain at least two points with different z values. It should look like this (replace the example coordinates with your own):

    ```json
    {
        "control_points": [
            {"x": 400, "y": 400, "z": 0},
            {"x": 410, "y": 405, "z": 799}
        ]
    }
    ```


7. Assemble the spiral fitting dataset

    Place your outputs in the following layout for spiral fitting:

    ```text
    spiral_fitting_dataset/
    ├── verified_patches/        <-- Place your graph patches here, one tifxyz directory per patch
    ├── tracks/                  <-- Place your extracted tracks here
    │   ├── tracks.dbm
    │   ├── tracks.dbm.vctracks/
    │   └── tracks.dbm.crossings.npz
    ├── umbilicus.json           <-- Save your scroll centerline here
    ├── fiber_directions.npz     <-- Place your extracted fiber directions here, if available
    └── spiral-scroll.json       <-- Save your scroll specification here
    ```

    The `spiral-scroll.json` should contain the following:

    ```json
    {
        "schema_version": 1,
        "name": "<name>",
        "voxel_size_um": "<size>",
        "spiral_outward_sense": "CW",
        "paths": {
            "tracks_dbm": "tracks/tracks.dbm"
        }
    }
    ```


8. Run spiral fitting

    Use the following overrides in the `.env`

    ```
    FIT_SPIRAL_CONFIG_OVERRIDES='{"z_begin": 0, "z_end": 800, "optimizer_num_training_steps": 5000, "dense_spacing_mode": "grad_mag", "input_use_tracks": true, "input_use_outer_shell": false, "input_use_normals": false, "input_use_gradient_magnitude": false}'
    ```


9. Generate winding 


## Installing VC3D

For the VC3D desktop app, see the [official installation and unwrapping guide](https://scrollprize.org/tutorial_VC3D).
The [archived segmentation tutorial](https://scrollprize.org/segmentation) covers an earlier version of the workflow.

### Running VC3D with Docker in WSL

On Windows, VC3D can also run in Docker inside WSL. This setup requires a working
X11 display in WSL and compatible NVIDIA GPU support for `--gpus all`.

Pull the Docker image and allow local Docker containers to access the X11 display.

```bash
docker pull ghcr.io/scrollprize/villa/volume-cartographer:edge
xhost +local:docker
```

Start the container.

```bash
docker run \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -it \
  -e DISPLAY=$DISPLAY \
  --gpus all \
  -e QT_QPA_PLATFORM=xcb \
  -e QT_X11_NO_MITSHM=1 \
  --rm \
  ghcr.io/scrollprize/villa/volume-cartographer:edge
```

Launch the application from inside the container.

```bash
cd /usr/local/bin
VC3D
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


## Creating segments for 2D ink detection with VC3D

Use the VC binaries installed above to flatten the fit's `_spliced` winding
meshes and render CT layers for ink detection. These commands work with the
old `edge` container (2026-05-13).

The defaults below are for **PHercParis4**: the fit uses **9.6 µm coordinates**
and the CT scan has **2.4 µm voxels** at level `0`. The function derives the
segmentation scale from these voxel sizes and defaults to a **9.6 µm** export.
Choose the resolution and layer count to match your ink model and
keep the layers as a stack. Lower-resolution CT levels also lose depth detail.
For cropped CT downloads, see the [coordinate warning](../download_data/README.md#download_zarr_from_urlpy).

> **Container bug:** The old `flatboi` W&B logger hangs if no parent folder ends
> in `.volpkg`. Using `work.volpkg` below avoids this; disabling W&B also works.
> The renderer also shares temporary buffers between threads, causing crashes
> or incorrect data. The function below uses one CPU thread to avoid this race.

### Set the variables

Run the blocks in the same Bash session. Use absolute paths and a new output
directory. `CT_VOLUME` can be an HTTPS URL or the path to a complete local Zarr.

```bash
export CT_VOLUME=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
export MESHES_DIR=/absolute/path/to/run/meshes/fitted
export OUTPUT_DIR=/absolute/path/to/segment-export

# PHercParis4 defaults. Match these to your fit and CT volume.
export FIT_VOXEL_SIZE_UM=9.6  # voxel_size_um in the fit's spiral-scroll.json
export CT_VOXEL_SIZE_UM=2.4   # CT voxel size at input level 0
export CT_LEVEL=2            # input CT level; it must exist in CT_VOLUME
export NUM_SLICES=16         # number of layers to render
```

The segmentation scale is `FIT_VOXEL_SIZE_UM / CT_VOXEL_SIZE_UM`; it stays
the same for every input CT level. Levels `0`, `1`, and `2` are assumed to
have voxel sizes of `CT_VOXEL_SIZE_UM` times `1`, `2`, and `4`, respectively.
The function derives the layer spacing, output voxel size, and folder name.

For **53keV_7.91um_Scroll3** with a fit and local CT both at 7.91 µm, replace
the four resolution/layer settings above with these values, and set
`CT_VOLUME`, `MESHES_DIR`, and `OUTPUT_DIR` to the corresponding paths:

```bash
export FIT_VOXEL_SIZE_UM=7.91
export CT_VOXEL_SIZE_UM=7.91
export CT_LEVEL=0
export NUM_SLICES=16
```

This gives a segmentation scale of `1` and a 7.91 µm output. A surface outside
the CT volume still renders black; choose a winding inside the available scan.

### Choose logging with or without W&B

Run **one** of these alternatives.

**Without logging:**

```bash
export WANDB_DISABLED=true APPTAINERENV_WANDB_DISABLED=true
```

**With W&B logging:**

Use the SIF from your `vc` wrapper. Replace the W&B user/team and enter your
API key when prompted. The container already includes `wandb`.

```bash
VC_IMAGE=/absolute/path/to/volume-cartographer.sif
unset WANDB_DISABLED APPTAINERENV_WANDB_DISABLED
export APPTAINERENV_WANDB_PROJECT=spiral-flattening
export APPTAINERENV_WANDB_ENTITY=YOUR_USER_OR_TEAM
export APPTAINERENV_WANDB_MODE=online
apptainer exec "$VC_IMAGE" wandb login
```

`APPTAINERENV_` passes settings into the container. Use mode `offline` instead
of `online` to save W&B runs locally without uploading them.

### Define the export function

Arguments: `export_segment MESH [CT_LEVEL] [LAYERS]`. Omitted arguments use
the `CT_LEVEL` and `NUM_SLICES` environment variables defined above.
Each resolution gets its own output folder.

```bash
export_segment() (
    set -euo pipefail
    mesh="$1"
    name="$(basename "$mesh")"
    group="${2:-$CT_LEVEL}"
    num_slices="${3:-$NUM_SLICES}"
    case "$group" in
        0) slice_step=1 ;;
        1) slice_step=2 ;;
        2) slice_step=4 ;;
        *) echo "CT_LEVEL must be 0, 1 or 2" >&2; exit 2 ;;
    esac
    scale_segmentation="$(awk -v fit="$FIT_VOXEL_SIZE_UM" -v ct="$CT_VOXEL_SIZE_UM" \
        'BEGIN { printf "%.12g", fit / ct }')"
    voxel_size="$(awk -v ct="$CT_VOXEL_SIZE_UM" -v step="$slice_step" \
        'BEGIN { printf "%.12g", ct * step }')"
    work="$OUTPUT_DIR/work.volpkg/$name"
    segment_name="${name}_${voxel_size}um"
    segment="$OUTPUT_DIR/segments/$segment_name"
    mkdir -p "$work" "$OUTPUT_DIR/segments"

    # Flatten the mesh using VC3D's SLIM workflow.
    vc_tifxyz2obj "$mesh" "$work/surface.obj"
    flatboi "$work/surface.obj" 50 symmetric_dirichlet
    vc_obj2tifxyz "$work/surface_flatboi.obj" "$segment" --uv-metric

    volume_args=(--volume "$CT_VOLUME")
    if [[ "$CT_VOLUME" == http://* || "$CT_VOLUME" == https://* ]]; then
        volume_args=(--volume "$work/ct-reference.zarr" --remote-url "$CT_VOLUME")
    fi

    # Sample the original CT around the flattened surface.
    OMP_NUM_THREADS=1 APPTAINERENV_OMP_NUM_THREADS=1 \
    vc_render_tifxyz \
        "${volume_args[@]}" \
        --segmentation "$segment" \
        --scale-segmentation "$scale_segmentation" \
        --group-idx "$group" --scale 1 \
        --num-slices "$num_slices" --slice-step "$slice_step" \
        --voxel-size "$voxel_size" --voxel-unit micrometer \
        --pyramid false \
        --zarr-output "$segment/$segment_name.zarr"
)
```

For URLs, the function supplies the old renderer's required local reference
path internally. The CT stays remote; it is not copied into `OUTPUT_DIR`.

The renderer options mean:

| Option | Purpose |
| --- | --- |
| `--volume`, `--remote-url` | Select the local CT or streamed CT source; derived from `CT_VOLUME`. |
| `--segmentation` | The flattened surface to sample. |
| `--scale-segmentation` | Map fit coordinates to CT level-0 coordinates using `FIT_VOXEL_SIZE_UM / CT_VOXEL_SIZE_UM`: `4` for PHercParis4, `1` for the Scroll3 example. |
| `--group-idx` | Input CT level from `CT_LEVEL` or the second function argument. For PHercParis4: `0` = 2.4 µm, `1` = 4.8 µm, `2` = 9.6 µm. |
| `--scale 1` | One output pixel per voxel of the selected CT level across the sheet. |
| `--num-slices` | Number of layers around the surface from `NUM_SLICES` or the third function argument. |
| `--slice-step` | Layer spacing in original scan voxels in this old container: `1`, `2`, `4` for input CT levels `0`, `1`, `2`. Multiply by `CT_VOXEL_SIZE_UM` for the spacing in µm. |
| `--voxel-size`, `--voxel-unit` | Record the output spacing in the Zarr metadata. These values do not rescale the mesh. |
| `--pyramid false` | Write only output array `0`, without additional downsampled levels. |
| `--zarr-output` | Destination for the rendered CT stack. |

The paired `slice-step` and `voxel-size` settings above are specific to this
old container and produce matching spacing in all three output axes.

### Export a single segment at different resolutions

To use the configured resolution and layer count:

```bash
export_segment "$MESHES_DIR/w020_spliced"
```

With the **PHercParis4** settings, you can also override the input level and
layer count for an individual export:

```bash
export_segment "$MESHES_DIR/w020_spliced" 0 64  # 2.4 µm
export_segment "$MESHES_DIR/w020_spliced" 1 32  # 4.8 µm
export_segment "$MESHES_DIR/w020_spliced" 2 16  # 9.6 µm
```

Replace `w020_spliced` with an actual mesh folder from your fit, including any
run tag. These examples cover approximately the same physical depth range
(154 µm); fewer pixels and layers make coarser exports cheaper. Actual runtime
also depends on CT downloads. Change the third argument if your model needs
a different layer count, for example `export_segment "$MESHES_DIR/w020_spliced" 2 64`.

With the PHercParis4 defaults, the result is `$OUTPUT_DIR/segments/w020_spliced_9.6um/`, containing the
flattened mesh and `w020_spliced_9.6um.zarr`. Its **array `0` is already at
9.6 µm**; the input CT level does not determine the output array name.

### Export all segments

Use this for all windings at the configured `CT_LEVEL` and `NUM_SLICES`
(9.6 µm and 16 layers with the PHercParis4 defaults). Use a fresh `OUTPUT_DIR`
when repeating an export at the same resolution.

```bash
set -e
shopt -s nullglob
for mesh in "$MESHES_DIR"/w[0-9]*_spliced*; do
    [[ -d "$mesh" && -f "$mesh/meta.json" ]] || continue
    export_segment "$mesh"
done
```

Each winding becomes a separate segment. Use its CT stack for
[ink detection](https://scrollprize.org/tutorial5).

> **Note:** zarr to tiff conversion is described in the
> [Download Data README](../download_data/README.md#zarr_to_tiffpy).
> Use `--array 0` for these exports; the resolution was selected during rendering.

## `render_ink.py`

`render_ink.py` takes an existing 3D ink-detection prediction volume as input
and renders it through the spiral-fitting surfaces into 2D images.


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
