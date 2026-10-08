# Spiral Fitting

This repository contains code for unwrapping Herculaneum scrolls. It includes tools for **surface detection**, **evidence extraction**, and **spiral fitting**.

## Repository Structure

The repository is organized into the following subfolders:

- `download_data`: Utilities for downloading and preparing data.
- `scrollreading`: Tools for finding surface patches using surface detection and volume data stored as Zarr.
- `spiral_fitting`: Code for running the spiral fitting.
- `surface_detection`: Tools for detecting papyrus surfaces in TIFF image stacks.

## Pipeline

This repository is a work in progress. The current idea is to use **surface detection** as the first step to preprocess the data into a more suitable format.

The next step is to collect different kinds of evidence about the scroll, for example:

- Surface patches
- Strips and lines of points
- Relative winding annotations
- Absolute winding annotations
- Other geometric or structural information

Some of this evidence could potentially be generated using tools such as `scrollreading`.

The final step is to run **spiral fitting**, which combines the available evidence into one consistent global solution for the scroll. Once a global solution has been obtained, the scroll can be unwrapped.

The overall pipeline would look approximately like this:

1. Run surface detection
2. Collect evidence (e.g., use `scrollreading` to find surface patches)
3. Run spiral fitting
4. Unwrap the scroll

## Useful Links

- [Scroll Prize](https://scrollprize.org/)
- [Spiral Fitting Tutorial](https://scrollprize.org/tutorial_spiral)