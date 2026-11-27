# Surface Prediction

Predict papyrus surfaces in a CT Zarr volume with an nnU-Net model.
`main.py` runs tiled inference, blends overlapping logits and writes the final
surface prediction. The pipeline stops if a step fails.

## Getting started

Run commands from `surface_prediction/` with the prediction environment active.
From the repository root:

```bash
cd surface_prediction
```

Relative paths use this working directory, including paths in `../.env`.
Replace `<...>` with your values. Flags take no value.

### 1. Create the environment

```bash
conda create -n surface-prediction python=3.14
conda activate surface-prediction
pip install uv
```

Install PyTorch and torchvision with CUDA support using the command for your
system from the [PyTorch installation guide](https://pytorch.org/get-started/locally/).
For CPU execution, install them with `uv pip install torch torchvision` and
set `DEVICE=cpu` in step 4.

Install the remaining dependencies:

```bash
uv pip install python-dotenv zarr tqdm requests nnunetv2 pynrrd
```

Check that PyTorch can access the GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

For CUDA execution, the second value must be `True`. The example configuration
below disables model compilation. See [model compilation](#model-compilation)
to enable it.

### 2. Prepare the input volume

Download or prepare the CT volume in Zarr format. Use its local array path as
`INPUT_DIR` in step 4. For a multiscale group, select the array at the desired
resolution, e.g. `../data/input.zarr/0` for level `0`.

Inspect the available arrays before selecting a level:

```bash
python ../utils/inspect_zarr.py "<ct_volume.zarr>"
```

| Argument | Purpose |
| --- | --- |
| `<ct_volume.zarr>` | **Required.** Zarr volume to inspect. Lists arrays, dimensions and metadata. |

For downstream spiral fitting, use the same voxel resolution and origin for
the prediction, scroll axis and annotations. See the
[dataset guide](../spiral_fitting/docs/create_spiral_dataset.md).

### 3. Download the model

Download the checkpoint, `plans.json` and `dataset.json` from
[scrollprize/surface_m7_nnunet on Hugging Face](https://huggingface.co/scrollprize/surface_m7_nnunet).
Keep this layout:

```text
../checkpoints/surface_prediction/
├── plans.json
├── dataset.json
└── fold_0/
    └── checkpoint_final.pth
```

Set `MODEL_PATH` to the checkpoint file or the `fold_0` directory. The loader
detects nnU-Net checkpoints from their metadata, including when a `.pth` file
is passed directly. The JSON files provide the architecture configuration;
weights alone are insufficient.

### 4. Configure the prediction

Add settings to `../.env`; `main.py` loads it automatically. Replace the model
and input paths with your local paths and use a fresh output directory for
each run.

```dotenv
MODEL_PATH=../checkpoints/surface_prediction/fold_0/checkpoint_final.pth
INPUT_DIR=../data/input.zarr/0
OUTPUT_DIR=out/run_01
DEVICE=cuda:0
BATCH_SIZE=1
MODE=binary
THRESHOLD=false
nnUNet_compile=false
```

| Variable | Purpose |
| --- | --- |
| `MODEL_PATH` | **Required.** Checkpoint file or model directory from step 3. |
| `INPUT_DIR` | **Required.** CT Zarr array from step 2. |
| `OUTPUT_DIR` | **Required.** Run directory for intermediate and final outputs. |
| `DEVICE` | Inference device (default: `cuda:0`); use `cpu` for CPU execution. |
| `BATCH_SIZE` | Inference batch size (default: `1`). |
| `MODE` | Finalization mode: `binary` (default) for surfaces, or `multiclass`. |
| `THRESHOLD` | Write probabilities (`false`, the default and example above) or a mask (`true`). Binary masks contain `0/255`. |
| `THRESHOLD_VALUE` | Optional probability cutoff in `(0, 1)` for binary masks (default: `0.5`). Requires `THRESHOLD=true` and `MODE=binary`. |
| `PATCH_SIZE` | Optional tile dimensions in `Z,Y,X` order (default: model patch size). |
| `nnUNet_compile` | Enable or disable model compilation; the example disables it. Preserve this variable's case. |

For a mask, set `THRESHOLD=true`. The appropriate cutoff depends on the model.
For example, add `THRESHOLD_VALUE=0.3` to use a lower cutoff;
`0.3` is only an example.

To reduce GPU memory usage, optionally add `PATCH_SIZE=128,128,128`.
Each dimension must be a positive integer compatible with the model
architecture. Batch size already defaults to `1`.

Command-line options override shell environment variables, which override
`.env`. Empty environment values for `main.py` options are ignored.
The launcher expands `$VAR`, `${VAR}` and `~` in the model, input and output
paths. Export variables used in those paths in the shell running the pipeline.

### 5. Run the pipeline

Preview the commands, then start the prediction:

```bash
python main.py --dry_run
python main.py
```

To override individual settings or pass additional inference options:

```bash
python main.py --batch_size 1 --patch_size 128,128,128
python main.py -- --disable_tta
```

| Argument | Purpose |
| --- | --- |
| `--dry_run` | Flag: print commands without executing them. Validates launcher arguments; does not load the model or input. |
| `--batch_size` | Override inference batch size. |
| `--patch_size Z,Y,X` | Override inference tile dimensions. |
| `-- <options...>` | Forward additional options to inference, e.g. `--disable_tta`. |
| `-h` / `--help` | Flag: show launcher help. |

All prediction parts run sequentially before blending and finalization.
See the [configuration reference](#configuration-reference) for all launcher
options.

### 6. Inspect and use the output

| Path inside `OUTPUT_DIR` | Contents |
| --- | --- |
| `logits/` | Per-part patch predictions and coordinates. |
| `merged_logits.zarr` | Blended logits. |
| `final_output.zarr` | Final `uint8` prediction with axes `(C, Z, Y, X)`. For the binary surface model, `C=1`. |

With `MODE=binary`, `THRESHOLD=true` writes mask values `0/255`.
`THRESHOLD=false` writes foreground probabilities scaled from `[0, 1]` to
`[0, 255]`.

Inspect the result and visualize it in
[ScrollSlabViewer](https://paul-g2.github.io/ScrollSlabViewer/):

```bash
python ../utils/inspect_zarr.py "<prediction_run>/final_output.zarr"
```

For track extraction, convert the prediction to a **3D `(Z, Y, X)` array with
values `0/1`**. Follow the
[inspection, channel selection and thresholding steps](../spiral_fitting/docs/create_spiral_dataset.md#2-inspect-squeeze-and-threshold-a-prediction),
then continue with track extraction and graph patches in the same guide.
The [utils README](../utils/README.md) documents the conversion arguments.

## Configuration reference

Every launcher option below also accepts the environment variable shown.
Values supplied before `--` configure `main.py`; options after `--` configure
inference only.

| Argument | Environment variable | Purpose / default |
| --- | --- | --- |
| `--model_path` | `MODEL_PATH` | **Required.** Model checkpoint or directory. |
| `--input_dir` | `INPUT_DIR` | **Required.** Input Zarr volume array. |
| `--output_dir` | `OUTPUT_DIR` | **Required.** Run output directory. |
| `--device` | `DEVICE` | Inference device (default: `cuda:0`). |
| `--num_parts N` | `NUM_PARTS` | Split inference into N sequential parts (default: `1`). |
| `--batch_size N` | `BATCH_SIZE` | Inference batch size (default: `1`). |
| `--patch_size Z,Y,X` | `PATCH_SIZE` | Inference tile dimensions (default: model patch size). |
| `--mode` | `MODE` | Finalization mode: `binary` (default) or `multiclass`. |
| `--threshold` | `THRESHOLD` | Flag: write a binary mask, or class indices in multiclass mode (default: `false`). |
| `--threshold_value T` | `THRESHOLD_VALUE` | Binary probability cutoff in `(0, 1)`; requires thresholding and binary mode (default: `0.5`). |
| `--num_workers N` | `NUM_WORKERS` | Worker processes for blending and finalization (default: half the CPU count, at least `1`). |
| `--chunk_size Z,Y,X` | `CHUNK_SIZE` | Output chunks for blending and finalization (default: derived from input chunks). |
| `--sigma_scale` | `SIGMA_SCALE` | Blending Gaussian sigma scale (default: `8.0`). |
| `--delete_intermediates` | `DELETE_INTERMEDIATES` | Flag: delete merged logits after successful finalization; retain patch logits (default: `false`). |
| `--dry_run` | `DRY_RUN` | Flag: print commands without executing (default: `false`). |
| `-h` / `--help` | — | Flag: show launcher help. |

Boolean launcher variables accept `1/0`, `true/false`, `yes/no` or `on/off`
(case-insensitive). Use `--no-threshold`, `--no-delete_intermediates` or
`--no-dry_run` to override an enabled environment flag.

`--num_workers` before `--` controls blending and finalization. To change
inference data-loader workers, pass it after `--`, e.g.
`python main.py -- --num_workers 2`. Inspect all inference options with:

```bash
python -m villa_8aa507c.models.run.inference --help
```

## Model compilation

Set `nnUNet_compile=true` in `../.env` to enable compilation, or `false` to
disable it. Without this setting, CUDA compilation is enabled on non-Windows
platforms; Windows and CPU execution use eager mode. This setting applies to
nnU-Net and `train.py` models.

For model compilation with PyTorch 2.14, install Triton 3.8:

| Platform | Installation command |
| --- | --- |
| Windows | `uv pip install "triton-windows>=3.8,<3.9"` |
| Linux | `uv pip install "triton>=3.8,<3.9"` |

For other PyTorch versions, check the
[PyTorch/Triton compatibility table](https://github.com/triton-lang/triton-windows/blob/readme/README.md#3-pytorch).
CUDA inference works without enabling compilation.

## Sources

The inference, blending and finalization code in `villa_8aa507c/` comes from
the following GitHub repository:

- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/tree/8aa507c1215f1fa8758f13d0538f7af9710e02d7) (Commit: `8aa507c`)

The original, unmodified code can be found in
[`villa_8aa507c/original_code/`](villa_8aa507c/original_code/).
