# Surface detection


## Sources

This folder contains code from the following sources:

- [**Vesuvius Challenge - Surface Detection**](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/overview)
- [**2nd Place Solution Vesuvius Challenge – A postprocessing win**](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/2nd-place-solution-vesuvius-challenge-a-postproc)
- [**local interpolation interference**](https://www.kaggle.com/code/mariusheuser/local-interpolation-interference) (notebook)

The original, unmodified code can be found in `original_code/`.


## Getting started with surface detection

1. Create python environment

    <!-- ```bash
    conda create --prefix <path_to_env>/surface-detection python
    conda config --append envs_dirs <path_to_env>
    conda config --set env_prompt '({name}) ' # Optional (reverse it using conda config --remove-key env_prompt)
    conda activate surface-detection

    pip install uv

    uv pip install -r requirements.txt
    uv pip install torch torchvision
    uv pip install "numpy<=2.4"
    uv pip install numba
    ``` -->

    ```bash
    conda create -n surface-detection python
    conda activate surface-detection

    pip install uv

    uv pip install -r requirements.txt
    uv pip install torch torchvision
    uv pip install "numpy<=2.4"
    uv pip install numba
    uv pip install python-dotenv
    ```

2. Download data from kaggle [here](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/data)

3. Download both checkpoints from kaggel [here](https://www.kaggle.com/code/mariusheuser/local-interpolation-interference/input)

4. Configuration using a `.env` File

    Create a `.env` file in the project directory to override the default configuration:

    ```bash
    MODEL1_FOLDER=../checkpoints/surface_detection/v3
    MODEL1_CHECKPOINT=checkpoint_final.pth
    MODEL1_DEVICE=cuda
    MODEL2_FOLDER=../checkpoints/surface_detection/v2
    MODEL2_CHECKPOINT=checkpoint_final.pth
    MODEL2_DEVICE=cuda
    INPUT_DIR=<input_dir>
    OUTPUT_ZIP=out/submission.zip
    ```

    Make sure the output folder exists.

Environment variables not specified in `.env` retain their default values.

5. Run surface detection

    ```bash
    python main.py
    ```

6. Visualize the results using the [ScrollSlabViewer](https://paul-g2.github.io/ScrollSlabViewer/).