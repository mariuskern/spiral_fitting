# Spiral Fitting

## Folders

- `spiral-fit-consumer-gpu`: Contains the code from [https://github.com/7jycwjmbfn-eng/spiral-fit-consumer-gpu](https://github.com/7jycwjmbfn-eng/spiral-fit-consumer-gpu/commit/f189740211f462193973055a32e3269c03301587) (Commit: f189740)
- `villa`: Contains the spiral code from [https://github.com/ScrollPrize/villa](https://github.com/ScrollPrize/villa/tree/7769da8cf2233310570608feecc127066a7c0c7c/volume-cartographer/scripts/spiral) (Commit: 7769da8)
- `villa/spiral_progress.py`: Code comes from volume-cartographer/scripts/spiral/spiral_progress.py(https://github.com/ScrollPrize/villa/blob/9761f14773a3ed41f2459bbf689a8f1998a656ed/volume-cartographer/scripts/spiral/spiral_progress.py) (Commit: 9761f14)

## How to run

1. Create conda environment:

    ```bash
    conda create -n villa-spiral python
    conda activate villa-spiral
    pip install uv
    uv pip install torch torchvision
    uv pip install -e villa
    uv pip install python-dotenv
    ```

2. Download data:

    ```bash
    rclone copy :http: ./spiral_datasets/phercparis4 \
    --http-url https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/ \
    --transfers 25 \
    --checkers 2 \
    --retries 20 \
    -P
    ```

    Alternativel the data can be downloaded the data from huggingface(https://huggingface.co/buckets/scrollprize/datasets/tree/spiral/PHercParis4). The `fetch_roi.py` and `fetch_tree.py` can be used for that. Refer to `fast_spiral_fit/README.md` and `fast_spiral_fit/REPRO.md`.

3. Create a `.env` and set the following environment variables:

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
    ```

    4. Set variables at the beginning of `villa/fit_spiral.py` (Starting around line: 150) and `default_config` (Starting around line: 160)

5. Run

    ```bash
    python fast_spiral_fit/run_fit.py 
    ```