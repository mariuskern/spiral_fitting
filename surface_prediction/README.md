# Surface Prediction

Predict surfaces in a Zarr volume using an nnUNet model. `main.py` runs inference,
blending, and finalization in sequence and stops if a step fails.

## Sources

This folder contains code from [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/tree/8aa507c1215f1fa8758f13d0538f7af9710e02d7)
(Commit: 8aa507c). The original, unmodified code is in
`villa_8aa507c/original_code/`.

## Getting Started

Run the following commands from the `surface_prediction/` directory. From the
repository root, enter it first:

```bash
cd surface_prediction
```

Relative paths in `.env` and command-line arguments are resolved against
`surface_prediction/` when starting the pipeline from this directory.

1. Create and activate the Python environment

    ```bash
    conda create -n surface-prediction python=3.14
    conda activate surface-prediction
    pip install uv
    ```

    Install PyTorch and torchvision with CUDA support using the command for your
    system from the [PyTorch installation guide](https://pytorch.org/get-started/locally/).
    For CPU execution, install them with `uv pip install torch torchvision` and
    use `DEVICE=cpu` in step 4.

    Install the remaining dependencies:

    ```bash
    uv pip install python-dotenv zarr tqdm requests nnunetv2 pynrrd
    ```

    Check that your PyTorch installation can access the GPU:

    ```bash
    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
    ```

    For GPU execution, the second value must be `True`.

    If you want to use model compilation with PyTorch 2.14, also install Triton 3.8:

    | Platform | Installation command |
    | --- | --- |
    | Windows | `uv pip install "triton-windows>=3.8,<3.9"` |
    | Linux | `uv pip install "triton>=3.8,<3.9"` |

    For other PyTorch versions, check the
    [PyTorch/Triton compatibility table](https://github.com/triton-lang/triton-windows/blob/readme/README.md#3-pytorch).
    Set `nnUNet_compile=true` in `.env` to enable compilation on Windows.
    Windows uses eager execution by default; CUDA inference works without Triton.
    On other platforms, CUDA compilation is enabled by default. Set
    `nnUNet_compile=false` to disable it. This setting applies to nnUNet and
    train.py models.

2. Prepare the input volume

    Download or prepare the Zarr volume you want to process. See the
    [download instructions](../download_data/README.md) for downloading a region
    of a scroll. Use the local volume path as `INPUT_DIR` in step 4.

3. Download the model checkpoint

    Download the model from
    [scrollprize/surface_m7_nnunet on Hugging Face](https://huggingface.co/scrollprize/surface_m7_nnunet).
    Include `plans.json` and `dataset.json` alongside the fold directory:

    ```text
    ../checkpoints/surface_prediction/
      plans.json
      dataset.json
      fold_0/
        checkpoint_final.pth
    ```

    Set `MODEL_PATH` to the checkpoint file or the `fold_0` directory.
    The auto loader recognizes nnUNet checkpoints by their metadata, including
    when a `.pth` file is passed directly. The accompanying JSON files supply
    the architecture configuration; checkpoint weights alone are not sufficient.

4. Configure the run in `.env`

    Create or update `.env` in the repository root (`../.env` from this folder).
    `main.py` loads it automatically:

    ```dotenv
    MODEL_PATH=../checkpoints/surface_prediction/fold_0/checkpoint_final.pth
    INPUT_DIR=../data/input.zarr
    OUTPUT_DIR=out/run_01
    DEVICE=cuda:0
    BATCH_SIZE=1
    MODE=binary
    THRESHOLD=false
    nnUNet_compile=false
    ```

    Replace the model and input paths with your local paths. Use a fresh output
    directory for each run. The three paths are required, either through `.env`,
    shell environment variables, or command-line arguments.

    To reduce tile memory usage, optionally add:

    ```dotenv
    PATCH_SIZE=128,128,128
    ```

    The order is `Z,Y,X`. Each dimension must be a positive integer compatible
    with the model architecture. Without `PATCH_SIZE`, the model patch size is
    used. Batch size already defaults to 1; setting it to 1 does not reduce it
    further.

    For a binary mask instead of probabilities, set `THRESHOLD=true` and
    optionally `THRESHOLD_VALUE=0.3`. The appropriate threshold depends on the
    model; `0.3` is only an example. With thresholding enabled, the default is 0.5.

5. Check the configuration and run the pipeline

    Preview the commands without running inference:

    ```bash
    python main.py --dry_run
    ```

    Then start the pipeline:

    ```bash
    python main.py
    ```

    Command-line arguments override environment values. For example:

    ```bash
    python main.py --batch_size 1 --patch_size 128,128,128
    ```

    Additional inference options go after `--`:

    ```bash
    python main.py -- --disable_tta
    ```

6. Inspect or export the results

    The output directory contains:

    | Path | Contents |
    | --- | --- |
    | `logits/` | Predictions for the individual parts |
    | `merged_logits.zarr` | Blended logits |
    | `final_output.zarr` | Final probabilities or mask, depending on the threshold settings |

    To convert the final output to TIFF, use
    [zarr_to_tiff.py](../download_data/README.md#zarr_to_tiffpy):

    ```bash
    python ../download_data/zarr_to_tiff.py out/run_01/final_output.zarr out/run_01_tiff
    ```

    The converter supports `(Z,Y,X)` and `(1,Z,Y,X)` volumes.
    `--delete_intermediates` removes only `merged_logits.zarr` after successful
    finalization; `logits/` is retained.

## Configuration reference

Every `main.py` option can also be set using its uppercase name without `--`.
Command-line arguments take precedence over shell environment variables, which
in turn take precedence over `.env`. Unset options use their defaults; empty
main.py environment values are ignored.

| Argument | Environment variable | Description / default |
| --- | --- | --- |
| `--model_path` | `MODEL_PATH` | Model checkpoint or directory; required from CLI or environment |
| `--input_dir` | `INPUT_DIR` | Input volume; required from CLI or environment |
| `--output_dir` | `OUTPUT_DIR` | Run output directory; required from CLI or environment |
| `--device` | `DEVICE` | Default: `cuda:0`; use `cpu` for CPU execution |
| `--num_parts N` | `NUM_PARTS` | Process N parts sequentially; default: 1 |
| `--batch_size N` | `BATCH_SIZE` | Inference batch size; default: 1 |
| `--patch_size Z,Y,X` | `PATCH_SIZE` | Inference tile dimensions; default: model patch size |
| `--mode` | `MODE` | Finalization mode: `binary` (default) or `multiclass` |
| `--threshold` | `THRESHOLD` | Output a mask instead of probabilities; default: false |
| `--threshold_value T` | `THRESHOLD_VALUE` | Binary cutoff in (0,1); requires thresholding; default: 0.5 |
| `--num_workers N` | `NUM_WORKERS` | Workers for blending and finalization |
| `--chunk_size Z,Y,X` | `CHUNK_SIZE` | Output chunks for blending and finalization |
| `--sigma_scale` | `SIGMA_SCALE` | Blending Gaussian sigma scale; default: 8.0 |
| `--delete_intermediates` | `DELETE_INTERMEDIATES` | Delete merged logits after finalization; default: false |
| `--dry_run` | `DRY_RUN` | Print commands without executing; default: false |

Boolean variables accept `1/0`, `true/false`, `yes/no`, or `on/off`
(case-insensitive). Use `--no-threshold`, `--no-delete_intermediates`, or
`--no-dry_run` to override an enabled environment flag.
