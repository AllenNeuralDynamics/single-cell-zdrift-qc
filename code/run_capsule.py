import argparse
import json
import shutil
from pathlib import Path
import datetime

from lamf_analysis.code_ocean import capsule_data_utils as cdu
from lamf_analysis.code_ocean import json_utils

import register_fov_local_zstack as reg_mod
import register_fov_local_zstack_qc as qc_mod
import register_fov_local_zstack_viz as viz_mod
import register_fov_local_zstack_parallel as par_mod
import local_zstack_processing as lzp_mod

SUFFIX = 'single-cell-zdrift-qc'
PROCESS_LEVEL = 'session' # 'subject' or 'session'
INPUT_PROCESSING_DICT = {"name": "Other", 
                         "software_version": "0.1.0",
                         "code_url": "https://codeocean.allenneuraldynamics.org/capsule/5658860/tree",
                         "notes": 'Single cell z-drift QC for multiplane-ophys data'}
''' "name" should be 'Analysis', 'Compression', 'Denoising', 'dF/F estimation', 'Ephys curation', 'Ephys postprocessing', 'Ephys preprocessing', 'Ephys visualization', 'Fiducial segmentation', 'File format conversion', 'Fluorescence event detection', 'Image atlas alignment', 'Image background subtraction', 'Image cell classification', 'Image cell quantification', 'Image cell segmentation', 'Image cross-image alignment', 'Image destriping', 'Image flat-field correction', 'Image importing', 'Image mip visualization', 'Image thresholding', 'Image tile alignment', 'Image tile fusing', 'Image tile projection', 'Image spot detection', 'Image spot spectral unmixing', 'Model evaluation', 'Model training', 'Neuropil subtraction', 'Other', 'Simulation', 'Skull stripping', 'Spatial timeseries demixing', 'Spike sorting', 'Video motion correction', 'Video plane decrosstalk', 'Video ROI classification', 'Video ROI cross session matching', 'Video ROI segmentation' or 'Video ROI timeseries extraction'
'''

def run_plane(plane_path, out_dir, intensity_threshold=0.5, zdrift_calc_bin=5, zstack_processing_root=None):
    plane_out_dir = out_dir / plane_path.name
    qc_dir = plane_out_dir / 'qc'
    qc_dir.mkdir(parents=True, exist_ok=True)

    # If this plane has no '*_z_stack_local_reg.h5' (sessions processed before
    # the movie_qc pipeline step existed), rebuild it -- and the
    # movie_qc/*_z_drift_evaluation.json it depends on -- from the raw local
    # z-stack and decrosstalked movie, and swap in the resulting shadow plane
    # path. Requires the raw asset to be attached next to the processed one
    # (see README "Local z-stack processing for missing movie_qc"). No-op
    # (returns plane_path unchanged) when the file is already present.
    if zstack_processing_root is not None:
        plane_path = lzp_mod.ensure_plane_path(plane_path, zstack_processing_root)

    # End-to-end registration + save
    run_out = reg_mod.run_single_plane(
        plane_path=plane_path,
        output_dir=plane_out_dir,
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
    qc_out = qc_mod.run_qc(result, qc_dir,
                           intensity_threshold=intensity_threshold,
                           zdrift_calc_bin=zdrift_calc_bin)
    print(saved_paths)
    print(qc_out['overlay_image_path'])
    print(qc_out['gif_path'])

    # register_fov_local_zstack.prepare_plane_data only ever reads
    # matched_plane_indices out of movie_qc/*_z_drift_evaluation.json, so a
    # generated-vs-genuine flag left only in that scratch-shadow file would
    # otherwise never reach the published result. Fold it into both: the
    # metadata.json save_result already wrote (machine-readable without
    # knowing to look for a second file), and a copy of the evaluation json
    # itself (full detail, including the note).
    provenance = lzp_mod.read_provenance(plane_path)
    if provenance is not None:
        source_evaluation_json = Path(provenance.pop('source_evaluation_json'))

        metadata_path = saved_paths['metadata_path']
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata.update(provenance)  # local_zstack_processing_generated, processing_note
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        shutil.copy2(source_evaluation_json, qc_dir / source_evaluation_json.name)
        print(f"local z-stack processing generated this plane's movie_qc -- flagged in {metadata_path}")

    return provenance is not None

if __name__ == '__main__':
    start_date_time = datetime.datetime.now()

    parser = argparse.ArgumentParser(description="Run registration for all planes in a session in parallel.")
    parser.add_argument('--input_dir', type=str, default='/root/capsule/data', help='Directory containing session data with multiplane-ophys folders')
    parser.add_argument('--output_dir', type=str, default='/root/capsule/results', help='Directory to save registration results')
    parser.add_argument('--num_planes', type=int, default=8, help='Number of planes expected in the session')
    parser.add_argument('--intensity_threshold', type=float, default=0.5, help='Intensity threshold for pass/fail single cell drift')
    parser.add_argument('--zdrift_calc_bin', type=int, default=5, help='Bin size (in minutes) for calculating z-drift min/max')
    parser.add_argument('--parallel', type=int, default=0, help='Whether to run planes in parallel (1) or sequentially (0)')
    parser.add_argument('--n_workers', type=int, default=8, help='Number of parallel workers to use. Only used when parallel=1.')
    parser.add_argument('--zstack_processing_root', type=str, default='/root/capsule/scratch/zstack_processing_shadow',
                         help="Writable scratch dir for local z-stack processing's shadow plane "
                              "paths (see local_zstack_processing.py). Set to '' to disable it "
                              "entirely and keep the original FileNotFoundError behavior.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    num_planes = args.num_planes
    output_dir = Path(args.output_dir)
    intensity_threshold = args.intensity_threshold
    zdrift_calc_bin = args.zdrift_calc_bin
    zstack_processing_root = Path(args.zstack_processing_root) if args.zstack_processing_root else None
    # Processed asset only -- its raw counterpart may be attached alongside it
    # for local z-stack processing (see README "Local z-stack processing for
    # missing movie_qc") and must not be mistaken for a second input here.
    input_data = [p for p in input_dir.glob('multiplane-ophys*') if '_processed_' in p.name]
    assert len(input_data) == 1, (
        f"Expected exactly one PROCESSED input asset under {input_dir}, found {len(input_data)}: "
        f"{[p.name for p in input_data]}"
    )
    input_folder = input_data[0]
    plane_ids = cdu.get_plane_ids_from_processed_path(input_folder)
    print(f"Found plane IDs: {plane_ids}")
    assert len(plane_ids) == num_planes, f"Expected {num_planes} plane IDs, found {len(plane_ids)}"

    plane_paths = [input_folder / plane_id for plane_id in plane_ids]

    # if using parallel (but it is not much faster, likely due to IO bottleneck)
    locally_processed_planes = []
    if args.parallel:
        # local z-stack processing is not wired into the parallel path yet.
        results = par_mod.run_planes_parallel(
            plane_paths,
            output_dir=output_dir,
            n_workers=args.n_workers,
        )
        par_mod.print_parallel_results(results)
    else:
        for plane_path in plane_paths:
            was_generated = run_plane(plane_path,
                      output_dir,
                      intensity_threshold=intensity_threshold,
                      zdrift_calc_bin=zdrift_calc_bin,
                      zstack_processing_root=zstack_processing_root)
            if was_generated:
                locally_processed_planes.append(plane_path.name)

    # Session-level record of which planes (if any) got their movie_qc from
    # local z-stack processing rather than genuine on-rig output -- same
    # provenance gap as the per-plane metadata.json (see run_plane): without
    # this, the published processing.json for the whole session can't tell
    # either.
    run_parameters = (
        {'local_zstack_processing_generated_planes': locally_processed_planes}
        if locally_processed_planes else {}
    )
    source_asset_name = input_folder.name.split('_processed_')[0]
    capture_name = source_asset_name

    json_utils.process_json_files(source_asset_name,
                                    capture_name,
                                    start_date_time,
                                    run_parameters,
                                    INPUT_PROCESSING_DICT,
                                    SUFFIX,
                                    PROCESS_LEVEL
                                    )
