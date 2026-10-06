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
  - Ensure `plane_path` contains `*_z_stack_local_reg.h5`. Sessions processed
    before the `movie_qc` pipeline step existed never have this file -- see
    **Local z-stack processing for missing `movie_qc`** below.
- **No result files found**
  - Check `output_dir` and filename suffix.

## Local z-stack processing for missing `movie_qc`

Some processed data assets predate the pipeline's `movie_qc` step and are
missing both `*_z_stack_local_reg.h5` and `movie_qc/*_z_drift_evaluation.json`,
which `register_fov_local_zstack.prepare_local_zstack`/`prepare_plane_data`
require. On these assets `run_capsule.py` fails with either:

```
FileNotFoundError: No '*_z_stack_local_reg.h5' found under .../VISp_0
```

or, if a `movie_qc/*_z_drift_evaluation.json` exists but its
`matched_plane_indices` is empty for some other reason:

```
ValueError: zero-size array to reduction operation minimum which has no identity
```

`local_zstack_processing.py` rebuilds both missing files from inputs that
*are* present, using the exact `lamf_analysis` building blocks the production
pipeline itself uses for this:

1. **`*_z_stack_local_reg.h5`** -- found via
   `zstack_utils.get_local_zstack_filepath`, which already falls back from the
   processed asset to the corresponding **raw** asset's `*_z_stack_local.h5`
   (a raw, interleaved `(n_slices * n_repeats, H, W)` stack), then rebuilt with
   `zstack.register_local_z_stack`: split into one sub-stack per z-slice
   (de-interleave), register + average each slice's repeat frames
   (within-plane), then register the resulting per-slice means to each other
   (between-plane).
2. **`movie_qc/*_z_drift_evaluation.json`** (`matched_plane_indices`) -- the
   same one-minute-mean-FOV-vs-z-stack estimation
   `capsule_data_utils.get_zdrift_matched_plane_indices(...,
   run_z_drift_estimation_if_not_found=True)` falls back to, inlined here to
   reuse the z-stack from step 1 and to stream the (often 50+ GB)
   decrosstalked movie from disk in ~one-minute chunks rather than loading it
   whole.

Rather than editing `register_fov_local_zstack.py`, this builds a writable
**shadow** copy of the affected plane's session directory under a scratch
root -- symlinks to every existing file/folder (including the sibling raw
asset, so `get_raw_path_from_plane_path` and friends keep working), plus a
*real* `movie_qc/` holding the two rebuilt files -- so the rest of the
pipeline runs against it completely unmodified.

**Requires the plane's raw data asset to be attached next to its processed
asset under the same `/data` directory.** `run_capsule.py`'s own input-folder
check now looks specifically for `_processed_` in the name, so the raw
sibling being attached alongside it is expected and does not confuse it.

```python
from pathlib import Path
import local_zstack_processing as lzp

plane_path = Path('/root/capsule/data/<processed_session>/<plane_id>')
shadow_root = Path('/root/capsule/scratch/zstack_processing_shadow')

# No-op if plane_path already has *_z_stack_local_reg.h5.
plane_path = lzp.ensure_plane_path(plane_path, shadow_root)

# Then run the existing pipeline exactly as before, against plane_path.
```

`run_capsule.py --zstack_processing_root /root/capsule/scratch/zstack_processing_shadow`
(the default) applies this automatically for every plane in the sequential
path; pass `--zstack_processing_root ''` to disable it and keep the original
error behavior. The parallel path (`register_fov_local_zstack_parallel.py` /
`--parallel 1`) does not go through this yet.

**This rebuilds inputs, not ground truth.** The rebuilt z-drift evaluation is
marked `"local_zstack_processing_generated": true` in the json so it's
distinguishable from a real on-rig `movie_qc` run.

### Testing local z-stack processing

Use the `codeocean-data-assets` skill to attach a processed asset missing the
file, plus its raw counterpart, to this capsule:

```bash
S=.claude/skills/codeocean-data-assets/scripts/co_data_assets.py
python $S search --subject <id> --type result --name multiplane-ophys  # find the processed asset id
python $S attach --asset <processed_asset_id> --asset <raw_asset_id>
```

Then run `local_zstack_processing.ensure_plane_path` (or `run_capsule.py` with
the default `--zstack_processing_root`) against the attached plane.

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
