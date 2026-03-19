"""Parallel wrapper for single-plane z-stack registration.

Runs register_fov_local_zstack.run_single_plane() across multiple planes
using multiprocessing.Pool.
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any

import register_fov_local_zstack as reg_mod


def _run_single_plane_worker(args: tuple) -> dict[str, Any]:
    """Worker function for multiprocessing.
    
    Unpacks arguments and calls run_single_plane.
    
    Parameters
    ----------
    args : tuple
        (plane_path, output_dir, run_kwargs, include_result)
    
    Returns
    -------
    dict with keys:
        - plane_path: the input plane path
        - result: full registration result dict (optional, only when include_result=True)
        - summary: lightweight summary payload
        - saved_paths: dict with h5_path and metadata_path
        - error: None if successful, error message string if failed
    """
    plane_path, root_output_dir, run_kwargs, include_result = args
    plane_output_dir = Path(root_output_dir) / Path(plane_path).name
    plane_output_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_out = reg_mod.run_single_plane(
            plane_path=plane_path,
            output_dir=plane_output_dir,
            **run_kwargs,
        )
        result = run_out["result"]
        summary = {
            "selected_method": result.get("summary", {}).get("selected_method"),
            "shared_eval_valid_frac": result.get("summary", {}).get("shared_eval_valid_frac"),
        }
        return {
            "plane_path": str(plane_path),
            "result": result if include_result else None,
            "summary": summary,
            "saved_paths": run_out["saved_paths"],
            "error": None,
        }
    except Exception as e:
        return {
            "plane_path": str(plane_path),
            "result": None,
            "summary": None,
            "saved_paths": None,
            "error": str(e),
        }


def run_planes_parallel(
    plane_paths: list[Path | str],
    output_dir: Path | str,
    *,
    n_workers: int | None = None,
    reg_ref_ind: int = 0,
    save_tiff: bool = True,
    file_suffix: str = "local_zstack_to_fov",
    nonrigid_block_sizes: tuple[tuple[int, int], ...] = ((32, 32), (64, 64), (128, 128)),
    nonrigid_maxregshift_values: tuple[int, ...] = (3, 5),
    pad: int = 3,
    include_result: bool = False,
) -> list[dict[str, Any]]:
    """Run registration pipeline for multiple planes in parallel.
    
    Parameters
    ----------
    plane_paths : list of Path or str
        List of plane directory paths.
    output_dir : Path or str
        Root output directory; each plane writes to ``output_dir / plane_id``.
    n_workers : int, optional
        Number of parallel workers. Defaults to number of CPU cores.
    reg_ref_ind : int, default 0
        Reference index for z-stack registration.
    save_tiff : bool, default True
        Whether to save registered z-stack as TIFF intermediate.
    file_suffix : str, default "local_zstack_to_fov"
        Suffix for output result files.
    nonrigid_block_sizes : tuple of tuples, default ((32, 32), (64, 64), (128, 128))
        Block sizes to test for nonrigid registration.
    nonrigid_maxregshift_values : tuple of ints, default (3, 5)
        Max registration shift values to test for nonrigid.
    pad : int, default 3
        Padding for crop indices.
    include_result : bool, default False
        If True, includes full registration result payload in each returned item.
        Keep False for parallel batch runs to avoid large IPC payloads.
    
    Returns
    -------
    list of dict
        Each dict contains:
        - plane_path: input plane path
        - result: registration result dict (or None if error)
        - saved_paths: dict with h5_path and metadata_path (or None if error)
        - error: error message (or None if successful)
    """
    output_dir = Path(output_dir)
    plane_paths = [Path(p) for p in plane_paths]
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    kwargs = {
        "reg_ref_ind": reg_ref_ind,
        "save_tiff": save_tiff,
        "file_suffix": file_suffix,
        "nonrigid_block_sizes": nonrigid_block_sizes,
        "nonrigid_maxregshift_values": nonrigid_maxregshift_values,
        "pad": pad,
    }
    
    # Prepare worker arguments: (plane_path, output_dir, kwargs)
    worker_args = [
        (plane_path, output_dir, kwargs, include_result)
        for plane_path in plane_paths
    ]
    
    # Run in parallel
    # Use spawn for notebook safety; fork can hang in interactive kernels.
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        results = list(pool.imap_unordered(_run_single_plane_worker, worker_args, chunksize=1))
    
    return results


def run_planes_parallel_from_dataframe(
    df: Any,
    plane_path_column: str = "plane_path",
    output_dir: Path | str | None = None,
    *,
    n_workers: int | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run registration pipeline for planes listed in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with plane information.
    plane_path_column : str, default "plane_path"
        Column name containing plane directory paths.
    output_dir : Path or str, optional
        Output directory for results. If None, uses current working directory.
    n_workers : int, optional
        Number of parallel workers. Defaults to number of CPU cores.
    **kwargs
        Additional keyword arguments passed to run_planes_parallel.
    
    Returns
    -------
    list of dict
        Results for each plane (see run_planes_parallel).
    """
    if output_dir is None:
        output_dir = Path.cwd()
    
    plane_paths = df[plane_path_column].tolist()
    
    return run_planes_parallel(
        plane_paths=plane_paths,
        output_dir=output_dir,
        n_workers=n_workers,
        **kwargs,
    )


def print_parallel_results(results: list[dict[str, Any]]) -> None:
    """Print summary of parallel results.
    
    Parameters
    ----------
    results : list of dict
        Results from run_planes_parallel or run_planes_parallel_from_dataframe.
    """
    n_success = sum(1 for r in results if r["error"] is None)
    n_failed = len(results) - n_success
    
    print(f"\n{'='*70}")
    print(f"Parallel Registration Summary")
    print(f"{'='*70}")
    print(f"Total planes: {len(results)}")
    print(f"Successful: {n_success}")
    print(f"Failed: {n_failed}")
    
    for result in results:
        status = "✓" if result["error"] is None else "✗"
        plane_path = result["plane_path"]
        print(f"\n{status} {plane_path}")
        if result["error"]:
            print(f"  Error: {result['error']}")
        else:
            h5_path = result["saved_paths"].get("h5_path")
            print(f"  Results: {h5_path}")
            summary = result.get("summary")
            if summary is not None:
                print(f"  Selected: {summary.get('selected_method')}")


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    # Define plane paths (adjust for your dataset)
    plane_paths = [
        Path("/root/capsule/data/session/plane_0"),
        Path("/root/capsule/data/session/plane_1"),
        # ... up to plane_7
    ]
    
    output_dir = Path("/root/capsule/scratch/_run/results")
    
    # Run in parallel
    results = run_planes_parallel(
        plane_paths,
        output_dir,
        n_workers=8,
    )
    
    # Print summary
    print_parallel_results(results)
