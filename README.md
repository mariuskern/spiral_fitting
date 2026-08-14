# Spiral Fitting

This repository contains the code and configuration required to run the Spiral Fitting pipeline. It combines components from the the ScrollPrize **villa** repository and **spiral-fit-consumer-gpu** repository.

## Sources

The repository combines code from multiple sources:

- [**spiral-fit-consumer-gpu**](https://github.com/7jycwjmbfn-eng/spiral-fit-consumer-gpu/commit/f189740211f462193973055a32e3269c03301587) (Commit: f189740)
- [**villa**](https://github.com/ScrollPrize/villa/tree/7769da8cf2233310570608feecc127066a7c0c7c/volume-cartographer/scripts/spiral) (Commit: 7769da8)
- [**villa**](https://github.com/ScrollPrize/villa/blob/9761f14773a3ed41f2459bbf689a8f1998a656ed/volume-cartographer/scripts/spiral/spiral_progress.py) (Commit: 9761f14) (only `villa/spiral_progress.py`)

## Getting Started

1. Create the python environment:

    ```bash
    conda create -n villa-spiral python
    conda activate villa-spiral

    pip install uv
    uv pip install torch torchvision
    uv pip install -e villa
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
    ```

4. Adjust the fitting configuration

    Before running the pipeline, edit the configuration at the beginning of `villa/fit_spiral.py`:

    - Around **line 150**: set the required variables.
    - Around **line 160**: adjust the `default_config` as needed for your experiment.

5. Run

    ```bash
    python spiral/fit_spiral.py
    ```