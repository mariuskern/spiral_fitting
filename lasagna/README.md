# Lasagna

Predicts fiber presence and directions (`presence`, `nx`, `ny`), not traced fiber curves or sheet normals. Run commands from `lasagna/`.

## Getting Started

1. Set up the environment:

    ```bash
    python villa_8aa507c/scripts/bootstrap_venv.py --venv .venv --backend cu128
    ```

    Activate with `.venv\Scripts\Activate.ps1` (PowerShell) or `source .venv/bin/activate` (Linux), then install the additional dependencies:

    ```bash
    uv pip install "python-dotenv>=1.0" acvl-utils matplotlib SimpleITK
    ```

2. Choose the model and input level:

    Choose the checkpoint by the **physical voxel size of the selected input level**. SD1 and SD2 are separate models, not interchangeable weights.

    | Model | Scan voxel size at `/0` | Input level | Effective input voxel size |
    |---|---|---|---|
    | SD1 | 2.4 um | `/1` | 4.8 um |
    | SD2 | 2.4 um | `/2` | 9.6 um |
    | SD2 (transfer use) | approximately 8-9 um | `/0` | approximately 8-9 um |

    For Scroll3/PHerc332 at **7.91 um**, use **SD2 on `/0`** if the downloaded crop has not been rescaled. SD2 with larger tiles worked for our local crop; other scans/resolutions still require validation.

3. Download the checkpoint:

    Download the chosen checkpoint (both shown here):

    ```bash
    hf download scrollprize/lasagna-fiber s1a_128_1_single_8x8_20260801_084232_best91_5k.pt --local-dir ../checkpoints/models/lasagna-fiber # SD1
    hf download scrollprize/lasagna-fiber s1a_128_2_single_8x8_20260728_094259_best_25_9k.pt --local-dir ../checkpoints/models/lasagna-fiber # SD2
    ```

4. Configure inference:

    Use **tile size 512, overlap 96, border 32** as the starting point. Tiles of 64 produced weak/sparse predictions in our test. Reduce tile size only if GPU memory requires it; model choice and tile size both affected the result.

    `LASAGNA_INFERENCE_SCALEDOWN_POWER=2` reduces output resolution by **4 relative to  the input**. It does not select the input pyramid level. Use `0` for output at input resolution (higher memory/storage cost).

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

    CLI arguments override environment values; relative paths resolve against `lasagna/`. Use `--help` for options. When changing checkpoint or inference settings, use a **new output directory** to avoid resuming old predictions. Keep `inference.json` with the manifest.

6. Serve the predictions:

    Serve the output in a separate terminal (from `lasagna/`):

    ```bash
    python -m http.server 8000 --directory out/sd2
    ```

7. Inspect predictions in the viewer:

    ```bash
    python viewer.py <path/to/fiber.lasagna.json>
    ```

    Export a slice without opening a viewer window:

    ```bash
    python viewer.py <path/to/fiber.lasagna.json> --save-dir <path/to/out> --plane XY --slice 100 --threshold 32 --opacity 0.65
    ```

    | Argument | Purpose |
    | --- | --- |
    | `<manifest>` | Local Lasagna manifest path or HTTP URL (default: `http://localhost:8000/fiber.lasagna.json`). |
    | `--ct <path>` | CT Zarr array path or HTTP URL (default: `LASAGNA_INPUT`). Relative local paths resolve against `lasagna/`; use `--ct=` for predictions only. |
    | `--ct-scale <scale>` | Positive CT voxel size in base voxels (default: manifest `source_to_base`). |
    | `--save-dir <directory>` | Save CT, Presence, and overlay PNGs, then exit without opening a window. Relative paths resolve against the working directory; existing files for the same slice are overwritten. Omit to open the interactive viewer. |
    | `--plane <XY/XZ/YZ>` | Slice plane (default: `XY`): fixes Z, Y, or X, respectively. |
    | `--slice <index>` | Zero-based slice index in the Presence array along the selected plane's fixed axis (default: middle slice). |
    | `--threshold <value>` | Minimum Presence value shown in the overlay, from `0` to `255` (default: `32`). |
    | `--opacity <value>` | Overlay opacity from `0` to `1` (default: `0.65`). |
    | `--color-max <value>` | Presence display maximum, from `1` to `255` (default: slice maximum, at least `1`). |
    | `-h` / `--help` | Flag: show CLI help and exit. |

    This writes `XY_Z00100_ct.png`, `XY_Z00100_presence.png`, and `XY_Z00100_overlay.png` at native slice resolution, without axes or controls. The plane, slice, threshold, opacity, and contrast options also set the initial interactive view when `--save-dir` is omitted.

For sample export and fitting, follow [Spiral Fitting, step 6](../spiral_fitting/README.md).


## Sources

The fiber prediction tools in `villa_8aa507c/` come from the following GitHub repository:

- [**ScrollPrize/villa**](https://github.com/ScrollPrize/villa/tree/8aa507c1215f1fa8758f13d0538f7af9710e02d7) (Commit: `8aa507c`)

The original, unmodified code can be found in [`villa_8aa507c/.original_code/lasagna/`](villa_8aa507c/.original_code/lasagna/).

