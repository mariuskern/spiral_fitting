# Surface detection


## Sources

This folder contains code from the following sources:

- [**Vesuvius Challenge - Surface Detection**](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/overview)
- [**2nd Place Solution Vesuvius Challenge – A postprocessing win**](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/2nd-place-solution-vesuvius-challenge-a-postproc)
- [**local interpolation interference**](https://www.kaggle.com/code/mariusheuser/local-interpolation-interference) (notebook)

The original, unmodified code can be found in `original_code/`.


## Getting started with surface detection

1. Create python environment

    ```bash
    pip install uv
    uv pip install -r requirements.txt
    uv pip install torch torchvision
    uv pip install "numpy<=2.4"
    uv pip install numba
    ```

2. Download data from kaggle [here](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/data)

3. Download both checkpoints from kaggel [here](https://www.kaggle.com/code/mariusheuser/local-interpolation-interference/input)

4. Run surface detection

    ```bash
    python main.py
    ```

5. Visualize the results using the [ScrollSlabViewer](https://paul-g2.github.io/ScrollSlabViewer/).