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

- [Vesuvius Challenge (scrollprize.org)](https://scrollprize.org/)
- [Tutorial: Spiral Fitting](https://scrollprize.org/tutorial_spiral)
- [Virtual Unwrapping with VC3D](https://scrollprize.org/tutorial_VC3D)
- [Tutorial: Segmentation (VC3D, Archive)](https://scrollprize.org/segmentation)


## Installing VC3D

Download and install VC3D by following the [official installation instructions](https://scrollprize.org/tutorial_VC3D).

Alternatively, you can run VC3D on Windows using Docker inside WSL. This approach worked for me.

### Running VC3D with Docker in WSL

Pull the Docker image:

```bash
docker pull ghcr.io/scrollprize/villa/volume-cartographer:edge
```

Allow local Docker containers to access the X11 display:

```bash
xhost +local:docker
```

Start the container:

```bash
docker run \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -it \
  -e DISPLAY=$DISPLAY \
  --gpus all \
  -e QT_QPA_PLATFORM=xcb \
  -e QT_X11_NO_MITSHM=1 \
  --rm \
  ghcr.io/scrollprize/villa/volume-cartographer:edge
```

Once inside the container, navigate to the directory containing the VC3D executable and launch the application:

```bash
cd /usr/local/bin
VC3D
```

**Note:** Running VC3D this way requires a working X11 display configuration in WSL and, if GPU acceleration is needed, compatible NVIDIA GPU support. The commands above worked in my setup, but additional configuration may be required on other systems.