# Lasagna

Predicts fiber presence and directions (`presence`, `nx`, `ny`), not traced
fiber curves or sheet normals. Run commands from `lasagna/`.

## Getting Started

```bash
cd lasagna
```

1. Set up the environment:

    ```bash
    python villa_8aa507c/scripts/bootstrap_venv.py --venv .venv --backend cu128
    ```

    Activate with `.venv\Scripts\Activate.ps1` (PowerShell) or
    `source .venv/bin/activate` (Linux), then install the additional dependencies:

    ```bash
    uv pip install "python-dotenv>=1.0" acvl-utils matplotlib SimpleITK
    ```

2. Choose the model and input level:

    Choose the checkpoint by the **physical voxel size of the selected input level**.
    SD1 and SD2 are separate models, not interchangeable weights.

    | Model | Scan voxel size at `/0` | Input level | Effective input voxel size |
    |---|---|---|---|
    | SD1 | 2.4 um | `/1` | 4.8 um |
    | SD2 | 2.4 um | `/2` | 9.6 um |
    | SD2 (transfer use) | approximately 8-9 um | `/0` | approximately 8-9 um |

    For Scroll3/PHerc332 at **7.91 um**, use **SD2 on `/0`** if the downloaded
    crop has not been rescaled. SD2 with larger tiles worked for our local crop;
    other scans/resolutions still require validation.

3. Download the checkpoint:

    Download the chosen checkpoint (both shown here):

    ```bash
    hf download scrollprize/lasagna-fiber s1a_128_1_single_8x8_20260801_084232_best91_5k.pt --local-dir ../checkpoints/models/lasagna-fiber
    hf download scrollprize/lasagna-fiber s1a_128_2_single_8x8_20260728_094259_best_25_9k.pt --local-dir ../checkpoints/models/lasagna-fiber
    ```

4. Configure inference:

    Use **tile size 512, overlap 96, border 32** as the starting point. Tiles of
    64 produced weak/sparse predictions in our test. Reduce tile size only if
    GPU memory requires it; model choice and tile size both affected the result.

    `--inference-scaledown-power 2` reduces output resolution by **4 relative to
    the input**. It does not select the input pyramid level. Use `0` for output
    at input resolution (higher memory/storage cost).

    Example for the 7.91 um crop, in the repository root `.env`:

    ```dotenv
    LASAGNA_INPUT=../data/332_large/volume/data.zarr/0
    LASAGNA_OUTPUT=out/sd2/fiber.lasagna.json
    LASAGNA_CHECKPOINT=../checkpoints/models/lasagna-fiber/s1a_128_2_single_8x8_20260728_094259_best_25_9k.pt
    LASAGNA_DEVICES=all
    LASAGNA_TILE_SIZE=512
    LASAGNA_OVERLAP=96
    LASAGNA_BORDER=32
    LASAGNA_INFERENCE_SCALEDOWN_POWER=2
    LASAGNA_INPUT_READER=python-zarr
    LASAGNA_NO_DOWNLOAD=true
    ```

5. Run inference:

    ```bash
    python main.py
    ```

    CLI arguments override environment values; relative paths resolve against
    `lasagna/`. `main.py` defaults to `python-zarr` for Zarr v3/v2 inputs;
    the TensorStore reader in this code supports v2 only. Use `--help` for options.
    When changing checkpoint or inference settings, use a **new output directory**
    to avoid resuming old predictions. Keep `inference.json` with the manifest.

6. Serve the predictions:

    Serve the output in a separate terminal (from `lasagna/`):

    ```bash
    python -m http.server 8000 --directory out/sd2
    ```

7. Inspect predictions in the viewer:

    ```bash
    python viewer.py http://localhost:8000/fiber.lasagna.json
    python viewer.py http://localhost:8000/fiber.lasagna.json --ct=
    ```

    The viewer takes CT from `LASAGNA_INPUT`; `--ct=` displays predictions only.
    Use XY/XZ/YZ and Slice to navigate, Color max for Presence contrast, and
    Threshold/Opacity for the overlay. Lowering the threshold does not improve
    prediction quality.

For sample export and fitting, follow [Spiral Fitting, step 6](../spiral_fitting/README.md).

## Sources

- [Model card and inference guide](https://huggingface.co/scrollprize/lasagna-fiber)
- [ScrollPrize/villa, commit 8aa507c](https://github.com/ScrollPrize/villa/tree/8aa507c1215f1fa8758f13d0538f7af9710e02d7)

The original, unmodified code is in `villa_8aa507c/.original_code/`.
