# Single cell z-drift QC

This folder contains a workflow for single cell z-drift QC.
It uses local z-stack - Register the z-stack to FOV, match each ROI in FOV to z-stack, and calculate intensity change across planar z-drift.

## Modules

- `register_fov_local_zstack.py`
  - Main pipeline for one plane path (no DataFrame input, no parallel processing)
- `register_fov_local_zstack_methods.py`
  - Method-level registration math/utilities shared by the pipeline
- `register_fov_local_zstack_viz.py`
  - Result loading and method-comparison visualization
- `register_fov_local_zstack_qc.py`
  - QC metrics, ROI/neuropil overlays, and GIF generation

## What This Workflow Does

For a single `plane_path`:

1. Finds local z-stack data:
   - `local_zstack_reg = cdu.get_local_zstack_reg(plane_path)`
2. Registers local z-stack between planes:
   - `reg_stack_imgs, shift_all = zstack.reg_between_planes(local_zstack_reg, ref_ind=0)`
3. Saves registered z-stack TIFF (intermediate result)
4. Registers local z-stack to FOV mean image (translation/affine/nonrigid cascade)
5. Saves flat result files:
   - `*_results.h5`
   - `*_metadata.json`
6. Generates QC outputs:
   - Using the ROI table
   - `registration_method_comparison.png`
   - ROI overlay PNG
   - final QC GIF

## Quick Start

```python
from pathlib import Path
import sys

if '/root/capsule/code' not in sys.path:
    sys.path.append('/root/capsule/code')

import register_fov_local_zstack as reg_mod
import register_fov_local_zstack_qc as qc_mod
import register_fov_local_zstack_viz as viz_mod

plane_path = Path('/root/capsule/data/<processed_session>/<plane_id>')
out_dir = Path('/root/capsule/results')
qc_dir = out_dir / 'qc'

# End-to-end registration + save
run_out = reg_mod.run_single_plane(
    plane_path=plane_path,
    output_dir=out_dir,
    reg_ref_ind=0,
    save_tiff=True,
    file_suffix='local_zstack_to_fov',
    pad=3,
)

result = run_out['result']
saved_paths = run_out['saved_paths']

# Method comparison figure
viz_mod.save_method_comparison_figure(
    result=result,
    row_i=0,
    out_path=qc_dir / 'registration_method_comparison.png',
)

# QC (ROI overlay + GIF)
qc_out = qc_mod.run_qc(result, qc_dir)
print(saved_paths)
print(qc_out['overlay_image_path'])
print(qc_out['gif_path'])
```

## Parallel processing
- Not much of a gain for now (8 planes, 8 workers, with 16 cores)
- Serial processing takes about 8 minutes
```python
from pathlib import Path
import register_fov_local_zstack_parallel as par_mod

plane_paths = [
    Path("/root/capsule/data/session/plane_0"),
    Path("/root/capsule/data/session/plane_1"),
    # ... up to plane_7
]

results = par_mod.run_planes_parallel(
    plane_paths,
    output_dir=Path("/root/capsule/results"),
    n_workers=8,
)

par_mod.print_parallel_results(results)
```

## Core APIs

### Main Pipeline (`register_fov_local_zstack_.py`)

- `prepare_local_zstack(plane_path, reg_ref_ind=0, save_tiff=True, tiff_save_dir=None)`
- `prepare_plane_data(plane_path, zstack_data)`
- `register_local_zstack_to_fov(plane_path, ..., pad=3)`
- `save_result(result, output_dir, file_suffix='local_zstack_to_fov')`
- `run_(plane_path, output_dir, ...)`
- `list_saved_result_files(output_dir)`
- `get_registration_benchmark_summary(result)`

### Visualization (`register_fov_local_zstack__viz.py`)

- `load_result_from_bundle(result_path, h5_path=None)`
- `save_method_comparison_figure(result, row_i, out_path)`
- `save_method_comparison_figure_from_result_file(result_file, out_path, ...)`

### QC (`register_fov_local_zstack__qc.py`)

- `get_valid_roi_table(result)`
- `make_roi_overlay_image(result, valid_roi_table, ...)`
- `save_roi_overlay_image(result, valid_roi_table, save_path)`
- `save_qc_gif(result, valid_roi_table, save_path, fps=3)`
- `run_qc(result, save_dir, gif_fps=3)`

## Output Files

### Registration Result

- `<session>_<plane>_<suffix>_results.h5`
- `<session>_<plane>_<suffix>_metadata.json`

The H5 file includes:
- `registered_zstack`
- `matched_plane_indices`
- `padded_plane_indices`
- `crop_y_inds`, `crop_x_inds`
- `fov_mean`, `fov_mean_cropped`
- transform groups per method

### Visualization + QC

- `registration_method_comparison.png`
- `<session>_<plane>_registered_mean_zstack_roi_overlay.png`
- `<session>_<plane>_registered_zstack_with_roi_outlines.gif`

Color convention in ROI overlay image:
- ROI contours: red
- Neuropil contours: yellow

## Notebook Example

A runnable test notebook is provided at:

- `test_notebook.ipynb`

It covers:
- preprocessing
- registration
- benchmark summary
- save/load roundtrip
- visualization
- QC output generation

## Troubleshooting

- **Read-only filesystem error when saving TIFF**
  - Set `tiff_save_dir` to a writable path (for example under `/root/capsule/scratch/...`).
- **Missing local z-stack file**
  - Ensure `plane_path` contains `*_z_stack_local_reg.h5`.
- **No result files found**
  - Check `output_dir` and filename suffix.

## Suggested Folder Layout for Runs

```text
/root/capsule/scratch/_run/
  results/
    zstack_tiff/
    *_results.h5
    *_metadata.json
  qc/
    registration_method_comparison.png
    *_registered_mean_zstack_roi_overlay.png
    *_registered_zstack_with_roi_outlines.gif
```
