# VC3D

[Back to Spiral Fitting](../README.md)

Commands use Bash on Linux/WSL. Replace `<...>` with your values.
For the desktop installer, see the [official VC3D guide](https://scrollprize.org/tutorial_VC3D)
or the [archived segmentation tutorial](https://scrollprize.org/segmentation).

## Desktop app with Docker in WSL

Requires Docker, an X11 display and compatible NVIDIA GPU support.

```bash
docker pull ghcr.io/scrollprize/villa/volume-cartographer:edge
xhost +local:docker
docker run --rm -it \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY="$DISPLAY" \
    --gpus all \
    -e QT_QPA_PLATFORM=xcb \
    -e QT_X11_NO_MITSHM=1 \
    ghcr.io/scrollprize/villa/volume-cartographer:edge
```

| Argument | Purpose |
| --- | --- |
| `xhost +local:docker` | Allow local Docker containers to access X11. |
| `--rm` | Flag: remove the container when it exits. |
| `-it` | Flags: keep an interactive terminal open. |
| `-v <host>:<container>` | Mount the X11 socket; add mounts for local data as needed. |
| `-e DISPLAY` | Pass the host's display address. |
| `--gpus all` | Expose NVIDIA GPUs. |
| `-e QT_QPA_PLATFORM=xcb` | Use Qt's X11 backend. |
| `-e QT_X11_NO_MITSHM=1` | Disable shared-memory X11 access. |

Inside the container:

```bash
cd /usr/local/bin
VC3D
```

## Install the command-line tools

Requires Apptainer. Run the blocks in the same Bash session.

### 1. Download the container

```bash
apptainer pull "$HOME/volume-cartographer.sif" \
    docker://ghcr.io/scrollprize/villa/volume-cartographer:edge
apptainer exec "$HOME/volume-cartographer.sif" vc_render_tifxyz --help
```

| Argument | Purpose |
| --- | --- |
| `pull <image.sif> <source>` | Download the container to a local SIF file. |
| `exec <image.sif> <binary> [<args> ...]` | Run a binary from the container's `/usr/local/bin/`. |
| `shell <image.sif>` | Optional alternative: open an interactive container shell. |

The export recipe below targets the **old `edge` build from 2026-05-13**.
Use that SIF if available; the floating `edge` tag may now supply different
CLI behavior. Check `vc_render_tifxyz --help` before applying the recipe to another build.

### 2. Create the wrapper script

Create `~/bin/vc`. The wrapper runs the matching binary inside the container
and forwards all arguments.

```bash
mkdir -p "$HOME/bin"
cat > "$HOME/bin/vc" <<'EOF'
#!/bin/bash
IMAGE="$HOME/volume-cartographer.sif"
exec apptainer exec "$IMAGE" "$(basename "$0")" "$@"
EOF
```

| Setting | Purpose |
| --- | --- |
| `IMAGE` | SIF path; change it if the container is installed elsewhere. |
| `$(basename "$0")` | Select the binary by the name used to call the wrapper. |
| `"$@"` | Forward all command-line arguments. |

### 3. Make the wrapper executable

```bash
chmod +x "$HOME/bin/vc"
```

### 4. Create symlinks for the binaries

Each symlink calls the wrapper under a binary's name, for example
`vc_render_tifxyz`. Skip names that already exist in `~/bin`.

```bash
for binary in \
    flatboi vc_add_ignore_label vc_calc_surface_metrics vc_create_segment_mask \
    vc_cut_windings vc_diffuse_winding vc_flatten vc_gen_normalgrids \
    vc_grow_seg_from_seed vc_grow_seg_from_segments vc_merge_tifxyz vc_ngrids \
    vc_obj2tifxyz vc_obj2tifxyz_legacy vc_objrefine vc_project_tifxyz \
    vc_render_tifxyz vc_render_video vc_rgb2tifxyz vc_seg_add_overlap \
    vc_thinning vc_tifxyz vc_tifxyz2obj vc_tifxyz2rgb vc_tifxyz2zarr_sparse \
    vc_tifxyz_gengt vc_tifxyz_inp_mask vc_tifxyz_mmap_prepare vc_tifxyz_trim \
    vc_tifxyz_winding vc_transform_geom vc_visualize vc_volpkg_convert \
    vc_zarr_recompress vc_zarr_to_tiff; do
    ln -s "$HOME/bin/vc" "$HOME/bin/$binary"
done
```

Calling `vc_render_tifxyz --help` then runs
`apptainer exec "$HOME/volume-cartographer.sif" vc_render_tifxyz --help`.

### 5. Add the wrappers to PATH

For the current Bash session:

```bash
export PATH="$HOME/bin:$PATH"
```

For future sessions, add the same line to `~/.bashrc` once and reload it:

```bash
echo 'export PATH="$HOME/bin:$PATH"' >> "$HOME/.bashrc"
source "$HOME/.bashrc"
```

### 6. Test the installation

```bash
vc_render_tifxyz --help
vc_tifxyz_trim --help
```

If both commands show their help, the VC binaries are available from your shell.

## Export segments for 2D ink detection

Flatten fitted `_spliced` meshes and render CT layers with the old container
above. Keep the layers as a stack; match voxel size and layer count to the ink model.
All blocks below run in the **same Bash session**.

### 1. Set paths and resolution

Use absolute paths and a fresh output directory. `CT_VOLUME` accepts an HTTPS
URL or a complete local CT Zarr. For cropped CT, account for the
[crop origin](../../download_data/README.md#download_zarr_from_urlpy) before export.

```bash
export CT_VOLUME=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr
export MESHES_DIR=/absolute/path/to/run/meshes/fitted
export OUTPUT_DIR=/absolute/path/to/segment-export
export FIT_VOXEL_SIZE_UM=9.6
export CT_VOXEL_SIZE_UM=2.4
export CT_LEVEL=2
export NUM_SLICES=16
```

| Variable | Purpose |
| --- | --- |
| `CT_VOLUME` | CT source; HTTP(S) is streamed rather than copied to the output. |
| `MESHES_DIR` | Fitted mesh directory containing `wNNN_spliced` folders, possibly with a run tag. |
| `OUTPUT_DIR` | Export destination; use a fresh directory when repeating the same resolution. |
| `FIT_VOXEL_SIZE_UM` | Fit coordinate voxel size from `spiral-scroll.json`. |
| `CT_VOXEL_SIZE_UM` | CT level-0 voxel size. |
| `CT_LEVEL` | Existing input level: `0`, `1` or `2`, assumed to have downsample factors `1`, `2`, `4`. |
| `NUM_SLICES` | Layers around the surface. |

| Example | Fit voxel size | CT level-0 voxel size | CT level | Layers | Output voxel size |
| --- | --- | --- | --- | --- | --- |
| PHercParis4 (above) | 9.6 µm | 2.4 µm | `2` | `16` | 9.6 µm |
| 53keV_7.91um_Scroll3 | 7.91 µm | 7.91 µm | `0` | `16` | 7.91 µm |

For Scroll3, change the variables and paths accordingly. A mesh outside the
available CT renders black. Coarser CT levels also lose depth detail.

### 2. Choose logging

Without W&B:

```bash
export WANDB_DISABLED=true APPTAINERENV_WANDB_DISABLED=true
```

Alternatively, with W&B (use the same SIF as the wrapper):

```bash
VC_IMAGE="$HOME/volume-cartographer.sif"
unset WANDB_DISABLED APPTAINERENV_WANDB_DISABLED
export APPTAINERENV_WANDB_PROJECT=spiral-flattening
export APPTAINERENV_WANDB_ENTITY="<wandb_user_or_team>"
export APPTAINERENV_WANDB_MODE=online
apptainer exec "$VC_IMAGE" wandb login
```

| Variable | Purpose |
| --- | --- |
| `WANDB_DISABLED` / `APPTAINERENV_WANDB_DISABLED` | Disable logging outside/inside the container. |
| `VC_IMAGE` | SIF containing `wandb`; login prompts for your API key. |
| `APPTAINERENV_WANDB_PROJECT` | W&B project. |
| `APPTAINERENV_WANDB_ENTITY` | W&B user or team. |
| `APPTAINERENV_WANDB_MODE` | `online` uploads; `offline` keeps logs locally. `APPTAINERENV_` passes variables into the container. |

### 3. Define the export function

The `.volpkg` working directory avoids an old `flatboi` W&B hang.
One renderer thread avoids a temporary-buffer race in the old container.

```bash
export_segment() (
    set -euo pipefail
    mesh="$1"
    name="$(basename "$mesh")"
    group="${2:-$CT_LEVEL}"
    num_slices="${3:-$NUM_SLICES}"
    case "$group" in
        0) slice_step=1 ;;
        1) slice_step=2 ;;
        2) slice_step=4 ;;
        *) echo "CT_LEVEL must be 0, 1 or 2" >&2; exit 2 ;;
    esac
    scale_segmentation="$(awk -v fit="$FIT_VOXEL_SIZE_UM" -v ct="$CT_VOXEL_SIZE_UM" \
        'BEGIN { printf "%.12g", fit / ct }')"
    voxel_size="$(awk -v ct="$CT_VOXEL_SIZE_UM" -v step="$slice_step" \
        'BEGIN { printf "%.12g", ct * step }')"
    work="$OUTPUT_DIR/work.volpkg/$name"
    segment_name="${name}_${voxel_size}um"
    segment="$OUTPUT_DIR/segments/$segment_name"
    mkdir -p "$work" "$OUTPUT_DIR/segments"

    # Flatten the mesh using VC3D's SLIM workflow.
    vc_tifxyz2obj "$mesh" "$work/surface.obj"
    flatboi "$work/surface.obj" 50 symmetric_dirichlet
    vc_obj2tifxyz "$work/surface_flatboi.obj" "$segment" --uv-metric

    volume_args=(--volume "$CT_VOLUME")
    if [[ "$CT_VOLUME" == http://* || "$CT_VOLUME" == https://* ]]; then
        volume_args=(--volume "$work/ct-reference.zarr" --remote-url "$CT_VOLUME")
    fi

    # Sample the original CT around the flattened surface.
    OMP_NUM_THREADS=1 APPTAINERENV_OMP_NUM_THREADS=1 \
    vc_render_tifxyz \
        "${volume_args[@]}" \
        --segmentation "$segment" \
        --scale-segmentation "$scale_segmentation" \
        --group-idx "$group" --scale 1 \
        --num-slices "$num_slices" --slice-step "$slice_step" \
        --voxel-size "$voxel_size" --voxel-unit micrometer \
        --pyramid false \
        --zarr-output "$segment/$segment_name.zarr"
)
```

| Argument / option | Purpose |
| --- | --- |
| `export_segment <mesh> [<ct_level>] [<layers>]` | Export one mesh; omitted values use `CT_LEVEL` and `NUM_SLICES`. |
| `flatboi ... 50 symmetric_dirichlet` | Flatten with 50 SLIM iterations and symmetric Dirichlet energy. |
| `--uv-metric` | Convert the flattened OBJ back to a tifxyz segment using UV metric coordinates. |
| `--volume` / `--remote-url` | Local CT or streamed source; remote CT gets a local reference path automatically. |
| `--segmentation` | Flattened surface to sample. |
| `--scale-segmentation` | Fit-to-CT-level-0 scale: `FIT_VOXEL_SIZE_UM / CT_VOXEL_SIZE_UM` (`4` for PHercParis4; `1` for Scroll3). |
| `--group-idx` | Input CT level. |
| `--scale 1` | One output pixel per selected-level CT voxel across the sheet. |
| `--num-slices` | Number of layers around the surface. |
| `--slice-step` | Layer spacing in CT level-0 voxels: `1`, `2`, `4` for levels `0`, `1`, `2`. |
| `--voxel-size` / `--voxel-unit` | Output spacing metadata in µm; does not rescale meshes. |
| `--pyramid false` | Write only output array `0`. |
| `--zarr-output` | Rendered CT stack destination. |

The scale mapping stays the same at every input CT level. The paired
`slice-step` and `voxel-size` settings give equal spacing in all three output
axes for this old container.

### 4. Export one segment

Replace `w020_spliced` with an existing mesh directory, including any run tag.

```bash
export_segment "$MESHES_DIR/w020_spliced"
```

With the PHercParis4 settings, override resolution and layer count as follows:

```bash
export_segment "$MESHES_DIR/w020_spliced" 0 64
export_segment "$MESHES_DIR/w020_spliced" 1 32
export_segment "$MESHES_DIR/w020_spliced" 2 16
```

| CT level | Output voxel size | Layers | Approximate depth |
| --- | --- | --- | --- |
| `0` | 2.4 µm | `64` | 154 µm |
| `1` | 4.8 µm | `32` | 154 µm |
| `2` | 9.6 µm | `16` | 154 µm |

Each resolution gets a separate folder. With the defaults, the segment is
`$OUTPUT_DIR/segments/w020_spliced_9.6um/`, containing the flattened mesh
and `w020_spliced_9.6um.zarr`. **Array `0` already has 9.6 µm voxels**;
the input CT level does not determine the output array name.

### 5. Export all segments

Uses the configured `CT_LEVEL` and `NUM_SLICES` for every spliced winding.

```bash
set -e
shopt -s nullglob
for mesh in "$MESHES_DIR"/w[0-9]*_spliced*; do
    [[ -d "$mesh" && -f "$mesh/meta.json" ]] || continue
    export_segment "$mesh"
done
```

Use each segment's CT stack for [ink detection](https://scrollprize.org/tutorial5).
For [Zarr-to-TIFF conversion](../../download_data/README.md#zarr_to_tiffpy),
select `--array 0`.
