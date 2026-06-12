import argparse
from pathlib import Path
import datetime

from lamf_analysis.code_ocean import capsule_data_utils as cdu
from lamf_analysis.code_ocean import json_utils

import register_fov_local_zstack as reg_mod
import register_fov_local_zstack_qc as qc_mod
import register_fov_local_zstack_viz as viz_mod
import register_fov_local_zstack_parallel as par_mod

SUFFIX = 'single-cell-zdrift-qc'
PROCESS_LEVEL = 'session' # 'subject' or 'session'
INPUT_PROCESSING_DICT = {"name": "Other", 
                         "software_version": "0.1.0",
                         "code_url": "https://codeocean.allenneuraldynamics.org/capsule/5658860/tree",
                         "notes": 'Single cell z-drift QC for multiplane-ophys data'}
''' "name" should be 'Analysis', 'Compression', 'Denoising', 'dF/F estimation', 'Ephys curation', 'Ephys postprocessing', 'Ephys preprocessing', 'Ephys visualization', 'Fiducial segmentation', 'File format conversion', 'Fluorescence event detection', 'Image atlas alignment', 'Image background subtraction', 'Image cell classification', 'Image cell quantification', 'Image cell segmentation', 'Image cross-image alignment', 'Image destriping', 'Image flat-field correction', 'Image importing', 'Image mip visualization', 'Image thresholding', 'Image tile alignment', 'Image tile fusing', 'Image tile projection', 'Image spot detection', 'Image spot spectral unmixing', 'Model evaluation', 'Model training', 'Neuropil subtraction', 'Other', 'Simulation', 'Skull stripping', 'Spatial timeseries demixing', 'Spike sorting', 'Video motion correction', 'Video plane decrosstalk', 'Video ROI classification', 'Video ROI cross session matching', 'Video ROI segmentation' or 'Video ROI timeseries extraction'
'''

def run_plane(plane_path, out_dir, intensity_threshold=0.5, zdrift_calc_bin=5):
    plane_out_dir = out_dir / plane_path.name
    qc_dir = plane_out_dir / 'qc'
    qc_dir.mkdir(parents=True, exist_ok=True)

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
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    num_planes = args.num_planes
    output_dir = Path(args.output_dir)
    intensity_threshold = args.intensity_threshold
    zdrift_calc_bin = args.zdrift_calc_bin
    input_data = list(input_dir.glob('multiplane-ophys*'))
    assert len(input_data) == 1, f"Expected exactly one input file (processed asset), found {len(input_data)}"
    input_folder = input_data[0]
    plane_ids = cdu.get_plane_ids_from_processed_path(input_folder)
    print(f"Found plane IDs: {plane_ids}")
    assert len(plane_ids) == num_planes, f"Expected {num_planes} plane IDs, found {len(plane_ids)}"

    plane_paths = [input_folder / plane_id for plane_id in plane_ids]

    # if using parallel (but it is not much faster, likely due to IO bottleneck)
    if args.parallel:
        results = par_mod.run_planes_parallel(
            plane_paths,
            output_dir=output_dir,
            n_workers=args.n_workers,
        )

        par_mod.print_parallel_results(results)
    else:
        for plane_path in plane_paths:
            run_plane(plane_path,
                      output_dir,
                      intensity_threshold=intensity_threshold,
                      zdrift_calc_bin=zdrift_calc_bin)

    run_parameters = {}
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
