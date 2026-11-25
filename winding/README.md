# Winding

## Getting started

Replace `<...>` with your values. Triplets use **Z Y X** order.
Tables list all arguments; commands show a typical selection. Flags take no value.

1. Create the env

    ```bash
    uv venv
    source .venv/bin/activate

    uv pip install -r requirements.txt
    ```

2. Run the pipeline

    ```bash
    python sheet_pipeline.py \
        --volume "<ct_volume>" \
        --pred "<surface_prediction>" \
        --device "<device>" \
        --workers "<worker_count>" \
        --min-cc "<min_component_voxels>" \
        --out "<pipeline_out>" \
        --ct-dataset "<ct_dataset>"
    ```

    | Argument | Purpose |
    | --- | --- |
    | `--volume` / `--ct` | **Required.** CT volume path; aligned with the prediction. |
    | `--pred` | **Required.** Surface prediction path. |
    | `--ct-dataset` | Array path inside the CT Zarr group; omit for a standalone array. |
    | `--pred-dataset` | Array path inside the prediction Zarr group. |
    | `--out` | Pipeline output directory. |
    | `--block-shape <z> <y> <x>` | Processing block dimensions in voxels. |
    | `--overlap <z> <y> <x>` | Block overlap in voxels. |
    | `--z0` | First input Z slice, inclusive. |
    | `--z1` | Last input Z boundary, exclusive. |
    | `--threshold` | Keep prediction values above this threshold. |
    | `--min-cc` | Minimum connected component size in voxels. |
    | `--core-dist` | Minimum distance from the mask boundary for watershed seeds, in voxels. |
    | `--min-core` | Minimum watershed seed component size in voxels. |
    | `--no-merge` | Flag: skip instance merging. |
    | `--no-ct-gating` | Flag: skip filtering the prediction by positive CT values. |
    | `--no-split` | Flag: skip CT-based splitting of fused stacks. |
    | `--mega-frac` | Minimum share of block foreground for an instance to become a split candidate. |
    | `--min-center` | Minimum centerline component size in voxels for splitting. |
    | `--no-previews` | Flag: skip preview images. |
    | `--device` | Compute backend: `cpu` or `cuda`. |
    | `--tensor-chunk-depth` | Z slices per structure tensor workspace; controls memory use. |
    | `--eigh-chunk` | Maximum matrices per eigenvalue batch; controls memory use. |
    | `--oom-policy` | CUDA memory failure: `cpu` retries on CPU; `error` stops. |
    | `--workers` | Number of parallel block processes. |
    | `--no-progress` | Flag: hide the progress bar. |
    | `--resume` | Flag: reuse completed blocks with matching configuration; rebuild stitching. |
    | `-h` / `--help` | Flag: show CLI help and exit. |

3. Create the constraints

    ```bash
    python sheet_constraints.py "<pipeline_out>" \
        --out "<constraints_out>" \
        --scale "<scale_z>" "<scale_y>" "<scale_x>" \
        --full-shape "<size_z>" "<size_y>" "<size_x>" \
        --pitch-vox "<winding_spacing>"
    ```

    | Argument | Purpose |
    | --- | --- |
    | `<pipeline_out>` (`run_dir`) | **Required.** Pipeline output directory containing `local_run.json`. |
    | `--out` | Constraint output directory. |
    | `--scale <z> <y> <x>` | **Required.** Factors mapping input to original voxel coordinates. |
    | `--full-shape <z> <y> <x>` | **Required.** Original volume dimensions in voxels. |
    | `--umbilicus` | Existing scroll axis JSON; mutually exclusive with `--axis`. |
    | `--axis <x> <y>` | Fixed scroll axis in input XY voxels; otherwise estimated from the mask. |
    | `--only-umbilicus` | Flag: create only axis JSON, report and preview. |
    | `--pitch-vox` | Expected winding spacing in input XY voxels; required unless `--only-umbilicus`. |
    | `--z-step` | Input Z sampling interval for axis estimation and relative windings. |
    | `--smooth-window` | Odd median filter window for the estimated axis. |
    | `--min-pixels` | Minimum mask pixels per slice for axis estimation. |
    | `--max-jump` | Axis jump limit between control points, in original XY voxels. |
    | `--batch-slices` | Slices composited per batch; controls memory use. |
    | `--min-volume` | Minimum instance size in input voxels for same-winding constraints. |
    | `--points-per-instance` | Maximum sampled constraint points per instance. |
    | `--candidates-per-instance` | Candidate point reservoir size per instance and sampling pool. |
    | `--point-spacing` | Minimum sampled point spacing in input voxels. |
    | `--max-instance-fraction` | Reject instances reaching this share of total foreground; `1` disables the filter. |
    | `--max-deep-frac` | Maximum fraction of deep interior voxels per instance. |
    | `--rays` | Radial sampling rays per slice for relative windings. |
    | `--ray-step` | Radial sampling interval in input XY voxels. |
    | `--min-support` | Minimum slice/ray observations supporting a relative-winding pair. |
    | `--no-progress` | Flag: hide progress bars. |
    | `-h` / `--help` | Flag: show CLI help and exit. |

4. Visualize the constraints

    ```bash
    python view_sheet_constraints.py "<constraints_out>" \
        --ct "<ct_volume>" \
        --ct-dataset "<ct_dataset>" \
        --ct-scale "<scale_z>" "<scale_y>" "<scale_x>" \
        --slice-thickness "<overlay_thickness>"
    ```

    | Argument | Purpose |
    | --- | --- |
    | `<constraints_out>` (`directory`) | **Required.** Constraint output directory to visualize. |
    | `--out` | Save the initial view as an image instead of opening a window. |
    | `--z-range <min> <max>` | Inclusive display range in original Z voxels. |
    | `--z-exaggeration` | Visual stretch factor along Z. |
    | `--hide-labels` | Flag: hide constraint labels. |
    | `--ct [<path>]` | CT slice view; omit the path to infer CT from the pipeline configuration. |
    | `--ct-dataset` | Array path inside the CT Zarr group; omit for a standalone array. |
    | `--ct-scale <z> <y> <x>` | Factors mapping CT to original voxel coordinates. |
    | `--ct-origin <z> <y> <x>` | Original voxel coordinate of the first CT voxel; for cropped CT. |
    | `--slice` | Initial Z slice index in the supplied CT array. |
    | `--slice-thickness` | Point overlay thickness along Z, in original voxels. |
    | `--ct-window <min> <max>` | Fixed CT grayscale intensity window. |
    | `--ct-alpha` | CT image opacity from `0` to `1`. |
    | `-h` / `--help` | Flag: show CLI help and exit. |

## Sources

The `vesuvius_sheet_tools` folder contains code from the following GitHub repository:

- [**vesuvius-sheet-tools**](https://github.com/IyanDopico/vesuvius-sheet-tools/tree/cc1fb1429d62670f7dc00943932e90d0e6d31c8a) (Commit: cc1fb14)

The original, unmodified code is in `vesuvius_sheet_tools/.original/`.
