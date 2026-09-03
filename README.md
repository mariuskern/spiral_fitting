# Spiral Fitting

This repository contains the code and configuration required to run the Spiral Fitting pipeline. It combines components from the the ScrollPrize **villa** repository and **spiral-fit-consumer-gpu** repository.

## Sources

The repository combines code from multiple sources:

- [**7jycwjmbfn-eng/spiral-fit-consumer-gpu**](https://github.com/7jycwjmbfn-eng/spiral-fit-consumer-gpu/commit/f189740211f462193973055a32e3269c03301587) (Commit: f189740)
- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/tree/7769da8cf2233310570608feecc127066a7c0c7c/volume-cartographer/scripts/spiral) (Commit: 7769da8) (spiral)
- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/blob/9761f14773a3ed41f2459bbf689a8f1998a656ed/volume-cartographer/scripts/spiral/spiral_progress.py) (Commit: 9761f14) (only `villa/spiral_progress.py`)
- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/tree/6732d3587bc3224a8a69a15886fb68c6e57a9342/volume-cartographer/scripts/spiral) (Commit: 6732d35) (spiral_6732d35, doesn't work yet)
- [**Vesuvius Challenge - Surface Detection (2nd Place)**](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/2nd-place-solution-vesuvius-challenge-a-postproc) ([challenge](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/overview), [notebook](https://www.kaggle.com/code/mariusheuser/local-interpolation-interference))

## Getting Started

1. Create the python environment:

    ```bash
    conda create -n villa-spiral python
    conda activate villa-spiral

    pip install uv
    uv pip install torch torchvision
    uv pip install -e spiral
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

    The helper scripts `fast_spiral_fit/fetch_roi.py` and `fast_spiral_fit/fetch_tree.py` can be used to download the dataset. Set the `LOCAL` constant at the beginning of each script to the desired dataset location before running it.

    ```bash
    python ./download_data/fetch_roi.py 4000 17000

    python ./download_data/fetch_tree.py --list --manifest m.jsonl verified_patches unverified_patches
    python ./download_data/fetch_tree.py --download --manifest m.jsonl --shard 0:4 --jobs 32
    python ./download_data/fetch_tree.py --direct outer_shell fibers tracks abs_winding.json patch-overlap-pcls.json relative_windings.json same_windings.json umbilicus.json
    ```

<!-- 3. Create a `.env` file in the project root and define the following environment variables:

    ```bash
    FIT_DATASET=/path/to/PHercParis4
    FIT_Z_BEGIN=4000
    FIT_Z_END=8000
    FIT_SPIRAL_SPARSE_NORMAL_CACHE_GB=2.5
    FIT_SPIRAL_SPARSE_GRAD_CACHE_GB=1
    FIT_SPIRAL_SPARSE_SDT_CACHE_GB=5
    FIT_SPIRAL_OUT_DIR=./fit_out
    FIT_USE_FAST_CACHE=1 # 0 = upstream cache (the A/B baseline)
    FIT_USE_FAST_LINK=1 # 0 = upstream brute-force linking
    FIT_SPIRAL_RESUME_PATH=./fit_out/<run>/checkpoint_periodic.ckpt # if you want to resume from checkpoint
    ``` -->

3. Create a `.env` file in the project root and define the following environment variables:

    ```bash
    DATASET=/path/to/PHercParis4
    NUM_TRAINING_STEPS=<num_training_steps>
    ```

4. Adjust the fitting configuration

    Before running the pipeline, edit the configuration at the beginning of `villa/fit_spiral.py`:

    - Around **line 150**: set the required variables.
    - Around **line 160**: adjust the `default_config` as needed for your experiment.

5. Run the pipeline

    ```bash
    python spiral/fit_spiral.py
    ```

## Installing the VC binaries

1. Download the docker container

    ```bash
    # Download
    apptainer pull ~/volume-cartographer.sif docker://ghcr.io/scrollprize/volume-cartographer:edge

    # Run shell
    apptainer shell ~/volume-cartographer.sif

    # Run command
    apptainer exec ~/volume-cartographer.sif <name_of_bin>
    ```

2. Create a file that calls the scripts that are located inside the docker container (e.g., called vc)

    ```
    #!/bin/bash

    IMAGE=$HOME/volume-cartographer.sif
    exec apptainer exec "$IMAGE" "$(basename "$0")" "$@"b
    ```

3. Make that file executable

    ```bash
    chmod +x ~/bin/vc
    ```

4. Create links connecting the desired command with `~/bin/vc`

    ```bash
    ln -s ~/bin/vc ~/bin/vc_render_tifxyz
    ln -s ~/bin/vc ~/bin/vc_tifxyz_trim
    ln -s ~/bin/vc ~/bin/vc_tifxyz2obj
    ln -s ~/bin/vc ~/bin/flatboi
    ln -s ~/bin/vc ~/bin/vc_obj2tifxyz
    ln -s ~/bin/vc ~/bin/vc_obj_uv_lift
    ```

5. `export PATH=$HOME/bin:$PATH`


## Getting started with `render_ink.py`

The python environment should already have been created.

1. Download the dataset:

    ```bash
    wget https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1B.7z
    ```


## Getting started with surface detection

1. Create python environment as described in `Getting Started`

2. Install surface detection dependencies

    ```bash
    uv pip install -r ./surface_detection/requirements.txt
    uv pip install "numpy<=2.4"
    conda install numba
    ```

3. Download data from kaggle [here](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/data)

4. Download both checkpoints from kaggel [here](https://www.kaggle.com/code/mariusheuser/local-interpolation-interference/input)