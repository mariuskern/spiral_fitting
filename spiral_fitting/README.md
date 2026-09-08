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
    pip install uv
    uv sync
    uv pip install torch torchvision
    uv pip install -e volume-cartographer
    uv pip install python-dotenv

    source .venv/bin/activate
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

    Before running the pipeline, edit the configuration in `config.py`:

    - Around **line 304/305**: set z_start and z_end.
    - Around **line 311**: set the number of training steps.

4. Run the pipeline

    ```bash
    python fit_spiral.py --dataset <path_to_dataset> --scroll-spec <path_to_scroll_spec> # For the 6732d35 version
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


## Getting started with `render_ink.py`

The python environment should have already been as described in `Getting Started`.

1. Download the dataset:

    ```bash
    wget https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1B.7z
    ```

2. Unpack the dataset

    ```bash
    7z x 54keV_7.91um_Scroll1B.7z 
    ```

    Optional: Use `tar.zst` for faster unpacking:

    ```bash
    tar -I 'zstd -T0' -cf <dataset>.tar.zst <dataset>/
    tar -I 'zstd -T0' -xf <dataset>.tar.zst
    ```

3. Run `render_ink.py`

    ```bash
    python spiral/render_ink.py --volume <path_to_dataset> <path_to_fit_spiral_output>/meshes/fitted/ --lasagna-dir spiral/lasagna
    ```
