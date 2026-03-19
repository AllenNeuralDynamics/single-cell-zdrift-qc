"""Single-plane local z-stack to FOV registration pipeline.

This module replaces the parallel/batch version (register_fov_local_zstack_parallel.py)
with a single-plane API that accepts a plain ``plane_path`` instead of a DataFrame row.

Key differences from the parallel variant:
- No multiprocessing / parallel execution
- Input: ``plane_path`` (Path/str) pointing to a processed ophys plane directory
- z-stack is loaded from the plane's ``*_z_stack_local_reg.h5`` file via
  ``cdu.get_local_zstack_reg``, then registered between planes via
  ``zstack.reg_between_planes`` (plane-to-plane phase-correlation).
- The registered z-stack is optionally saved as a TIFF intermediate result.
- Output file format: flat ``{session_key}_{plane_id}_{suffix}_results.h5`` +
  ``{session_key}_{plane_id}_{suffix}_metadata.json``.
"""
from __future__ import annotations

import json
import importlib
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import tifffile

from lamf_analysis.code_ocean import capsule_data_utils as cdu
from lamf_analysis.ophys import zstack as zstack_mod
import lamf_analysis.utils as lamf_utils

# Method-level registration logic in sibling methods module.
_methods_mod_name = "register_fov_local_zstack_methods"
try:
    _methods_mod = importlib.import_module(_methods_mod_name)
except ModuleNotFoundError:
    this_dir = str(Path(__file__).resolve().parent)
    if this_dir not in sys.path:
        sys.path.append(this_dir)
    _methods_mod = importlib.import_module(_methods_mod_name)

_compute_registration_metrics = _methods_mod._compute_registration_metrics
_rescore_methods_with_shared_mask = _methods_mod._rescore_methods_with_shared_mask
_method_rank_key_from_metrics = _methods_mod._method_rank_key_from_metrics
_register_translation = _methods_mod._register_translation
_register_affine = _methods_mod._register_affine
_compute_blank_safe_crop_inds = _methods_mod._compute_blank_safe_crop_inds
_nonrigid_not_worse_than_translation = _methods_mod._nonrigid_not_worse_than_translation
_search_best_nonrigid = _methods_mod._search_best_nonrigid
_search_best_affine = _methods_mod._search_best_affine
_apply_winner_transforms_to_stack = _methods_mod._apply_winner_transforms_to_stack
reproduce_transformations_from_saved = _methods_mod.reproduce_transformations_from_saved


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _to_json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    return value


def _safe_filename_token(value: Any, default: str) -> str:
    if value is None:
        token = default
    else:
        token = str(value).strip()
        token = token if token else default
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", token)
    return token.strip("_") or default


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# z-stack preparation: load from plane dir and register between planes
# ---------------------------------------------------------------------------

def prepare_local_zstack(
    plane_path: Path | str,
    *,
    reg_ref_ind: int = 0,
    save_tiff: bool = True,
    output_dir: Path | str | None = None,
) -> np.ndarray:
    """Load and inter-plane-register the local z-stack for a single plane.

    Loads ``*_z_stack_local_reg.h5`` from ``plane_path`` via
    :func:`cdu.get_local_zstack_reg`, runs plane-to-plane registration via
    :func:`zstack_mod.reg_between_planes`, and optionally saves the result as a
    TIFF file.

    Parameters
    ----------
    plane_path : Path or str
        Processed ophys plane directory.
    reg_ref_ind : int
        Reference plane index for inter-plane registration. Default 0.
    save_tiff : bool
        Write the registered stack as a TIFF. Default True.
    output_dir : Path, str, or None
        Where to save the TIFF. Must be writable. If None, TIFF is not saved.

    Returns
    -------
    np.ndarray  (Z, H, W)  float32
        Plane-to-plane registered z-stack.

    Raises
    ------
    FileNotFoundError
        If no local z-stack registration file is found.
    """
    plane_path = Path(plane_path)
    raw_stack = cdu.get_local_zstack_reg(plane_path)
    if raw_stack is None:
        raise FileNotFoundError(
            f"No '*_z_stack_local_reg.h5' found under {plane_path}"
        )

    reg_stack, _shifts = zstack_mod.reg_between_planes(raw_stack, ref_ind=reg_ref_ind)
    reg_stack = reg_stack.astype(np.float32)

    if save_tiff and output_dir is not None:
        tiff_dir = Path(output_dir)
        tiff_dir.mkdir(parents=True, exist_ok=True)
        tiff_path = tiff_dir / f"{plane_path.name}_local_zstack_registered.tif"
        tifffile.imwrite(str(tiff_path), reg_stack)

    return reg_stack


# ---------------------------------------------------------------------------
# Data preparation: FOV mean, crop indices, matched z-indices
# ---------------------------------------------------------------------------

def prepare_plane_data(
    plane_path: Path | str,
    zstack_data: np.ndarray,
) -> dict[str, Any]:
    """Load FOV mean image, crop indices, and z-drift matched indices.

    Parameters
    ----------
    plane_path : Path or str
        Processed ophys plane directory.
    zstack_data : np.ndarray  (Z, H, W)
        Pre-registered local z-stack (output of :func:`prepare_local_zstack`).

    Returns
    -------
    dict with keys:
        session_key, plane_path, plane_id, matched_plane_indices, zstack_data,
        matched_zstack, matched_zstack_cropped, mean_img, mean_img_cropped,
        mean_zstack_cropped, crop_y_inds, crop_x_inds, valid_mask
    """
    plane_path = Path(plane_path)
    plane_id = plane_path.name
    # Derive session_key from the processed directory name:
    # pattern: multiplane-ophys_{subject_id}_{date}_processed_.../{plane_id}
    # session_key = "{subject_id}_{date}"
    processed_dir = plane_path.parent
    parts = processed_dir.name.split("_")
    # "multiplane-ophys_<subject>_<date>_processed_..." -> parts[1]_parts[2]
    if len(parts) >= 3 and parts[0] == "multiplane-ophys":
        session_key = f"{parts[1]}_{parts[2]}"
    else:
        session_key = processed_dir.name

    evaluation_path = next((plane_path / "movie_qc").glob("*_z_drift_evaluation.json"))
    with open(evaluation_path) as f:
        evaluation_data = json.load(f)

    matched_plane_indices = np.array(
        lamf_utils.find_keys(evaluation_data, "matched_plane_indices", exact_match=True)
    ).squeeze()
    matched_zstack = zstack_data[matched_plane_indices.min() : matched_plane_indices.max() + 1]

    mean_img = cdu.load_projection_image(plane_path, projection_type="mean")
    mean_matched_zstack = np.mean(matched_zstack, axis=0)

    range_y, range_x = lamf_utils.get_motion_correction_crop_xy_range(plane_path)
    range_y_all_inds = np.arange(mean_img.shape[0])[range_y[0] : range_y[1] + 1]
    range_x_all_inds = np.arange(mean_img.shape[1])[range_x[0] : range_x[1] + 1]

    blank_row_inds = np.where(
        np.sum(matched_zstack.min(axis=0) == 0, axis=1) == matched_zstack.shape[1]
    )[0]
    blank_col_inds = np.where(
        np.sum(matched_zstack.min(axis=0) == 0, axis=0) == matched_zstack.shape[2]
    )[0]
    zstack_range_y = np.setdiff1d(np.arange(matched_zstack.shape[1]), blank_row_inds)
    zstack_range_x = np.setdiff1d(np.arange(matched_zstack.shape[2]), blank_col_inds)

    both_range_y = np.intersect1d(range_y_all_inds, zstack_range_y)
    both_range_x = np.intersect1d(range_x_all_inds, zstack_range_x)

    if len(both_range_y) <= mean_img.shape[0] * 0.75:
        raise ValueError("y crop intersection too small")
    if len(both_range_x) <= mean_img.shape[1] * 0.75:
        raise ValueError("x crop intersection too small")

    mean_img_cropped = mean_img[both_range_y[:, None], both_range_x]
    mean_zstack_cropped = mean_matched_zstack[both_range_y[:, None], both_range_x]
    matched_zstack_cropped = matched_zstack[:, both_range_y[:, None], both_range_x]
    valid_mask = (mean_img_cropped > 0) & (mean_zstack_cropped > 0)

    return {
        "session_key": session_key,
        "plane_path": str(plane_path),
        "plane_id": plane_id,
        "matched_plane_indices": matched_plane_indices,
        "zstack_data": zstack_data,
        "matched_zstack": matched_zstack,
        "matched_zstack_cropped": matched_zstack_cropped,
        "mean_img": mean_img,
        "mean_img_cropped": mean_img_cropped,
        "mean_zstack_cropped": mean_zstack_cropped,
        "crop_y_inds": both_range_y,
        "crop_x_inds": both_range_x,
        "valid_mask": valid_mask,
    }


# ---------------------------------------------------------------------------
# Core registration: translation → affine → nonrigid cascade
# ---------------------------------------------------------------------------

def register_local_zstack_to_fov(
    plane_path: Path | str,
    *,
    zstack_data: np.ndarray | None = None,
    reg_ref_ind: int = 0,
    save_tiff: bool = True,
    output_dir: Path | str | None = None,
    nonrigid_block_sizes: tuple[tuple[int, int], ...] = ((32, 32), (64, 64), (128, 128)),
    nonrigid_maxregshift_values: tuple[int, ...] = (3, 5),
    pad: int = 3,
    gate: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Register a single plane's local z-stack to its FOV mean image.

    Parameters
    ----------
    plane_path : Path or str
        Processed ophys plane directory.
    zstack_data : ndarray or None
        Pre-registered z-stack. If None, calls :func:`prepare_local_zstack`.
    reg_ref_ind : int
        Reference plane for inter-plane z-stack registration. Default 0.
    save_tiff : bool
        Save intermediate registered z-stack TIFF. Default True.
    output_dir : Path, str, or None
        Output directory for TIFF files. Required if save_tiff=True.
    nonrigid_block_sizes : tuple of (H, W)
        Block sizes for nonrigid candidate search.
    nonrigid_maxregshift_values : tuple of int
        Max shift values for nonrigid candidate search.
    pad : int
        Number of z-planes to pad around matched range. Default 3.
    gate : dict or None
        Quality-gate thresholds (used for metadata only).

    Returns
    -------
    dict
        registered_zstack, matched_plane_indices, padded_plane_indices,
        padded_z_start, padded_z_end, fov_mean, fov_mean_cropped, summary,
        all_methods, affine_candidates, nonrigid_candidates, crop_y_inds,
        crop_x_inds, plane_path, session_key, plane_id.
    """
    if gate is None:
        gate = {
            "max_residual_l2_px": 0.60,
            "max_residual_abs_px": 0.50,
            "min_ncc": 0.60,
            "min_ssim": 0.30,
            "min_valid_frac": 0.80,
        }

    if zstack_data is None:
        zstack_data = prepare_local_zstack(
            plane_path,
            reg_ref_ind=reg_ref_ind,
            save_tiff=save_tiff,
            output_dir=output_dir,
        )

    data = prepare_plane_data(plane_path, zstack_data)
    matched_zstack = data["matched_zstack"]
    mean_img_cropped = data["mean_img_cropped"]
    mean_zstack_cropped = data["mean_zstack_cropped"]
    valid_mask = data["valid_mask"]
    yind = data["crop_y_inds"]
    xind = data["crop_x_inds"]

    methods: dict[str, Any] = {}
    affine_candidates: list[dict[str, Any]] = []

    full_binary_mask = np.asarray(matched_zstack.min(axis=0) > 0, dtype=bool)

    # Stage A: translation
    reg_t, t_info = _register_translation(
        matched_zstack,
        mean_img_cropped,
        mean_zstack_cropped,
        valid_mask=valid_mask,
        full_binary_mask=full_binary_mask,
        crop_y_inds=yind,
        crop_x_inds=xind,
        cropped_overlap_mask=valid_mask,
    )
    valid_mask_t = np.asarray(t_info.pop("updated_valid_mask", valid_mask), dtype=bool)
    reg_t_crop = reg_t[:, yind[:, None], xind]
    mean_t = reg_t_crop.mean(axis=0)
    metrics_t = _compute_registration_metrics(mean_img_cropped, mean_t, valid_mask=valid_mask_t)
    methods["translation"] = {
        "registered_zstack": reg_t,
        "registered_mean_cropped": mean_t,
        "valid_mask": valid_mask_t,
        "metrics": metrics_t,
        "transform": t_info,
    }

    # Stage B: affine search (test CLAHE=False/True and select best affine)
    row_keep, col_keep = _compute_blank_safe_crop_inds(reg_t_crop)
    affine_results: dict[str, dict[str, Any]] = {}
    
    if len(row_keep) > 30 and len(col_keep) > 30:
        fov_phase2 = mean_img_cropped[row_keep[:, None], col_keep]
        mov_phase2 = mean_t[row_keep[:, None], col_keep]
        reg_input = reg_t[:, yind[row_keep][:, None], xind[col_keep]]
        valid_mask_aff_local = valid_mask_t[row_keep[:, None], col_keep]
        
        aff_search = _search_best_affine(
            reg_input=reg_input,
            fov_ref=fov_phase2,
            mov_img=mov_phase2,
            valid_mask=valid_mask_aff_local,
        )
        if aff_search is None:
            # Fallback path: explicit two-variant testing if search fails
            for use_clahe in (False, True):
                reg_a_crop_local, a_info = _register_affine(
                    reg_input, fov_phase2, mov_phase2, use_clahe=use_clahe
                )
                a_info = dict(a_info)
                a_info["affine_use_clahe"] = bool(use_clahe)
                valid_mask_a_local = valid_mask_aff_local

                reg_a = reg_t.copy().astype(np.float32)
                reg_a[:, yind[row_keep][:, None], xind[col_keep]] = reg_a_crop_local
                reg_a_crop = reg_a[:, yind[:, None], xind]
                valid_mask_a = valid_mask_t.copy()
                valid_mask_a[row_keep[:, None], col_keep] = valid_mask_a_local

                mean_a = reg_a_crop.mean(axis=0)
                metrics_a = _compute_registration_metrics(mean_img_cropped, mean_a, valid_mask=valid_mask_a)
                key = "affine_clahe_true" if use_clahe else "affine_clahe_false"
                affine_results[key] = {
                    "registered_zstack": reg_a,
                    "registered_mean_cropped": mean_a,
                    "valid_mask": valid_mask_a,
                    "metrics": metrics_a,
                    "transform": a_info,
                }
        else:
            _best_aff, all_affine = aff_search
            for cand in all_affine:
                tr = dict(cand["transform"])
                tr["candidate_protocol"] = "affine_after_translation"
                affine_candidates.append({"metrics": cand["metrics"], "transform": tr})

                use_clahe = bool(tr.get("candidate_use_clahe", tr.get("affine_use_clahe", False)))
                key = "affine_clahe_true" if use_clahe else "affine_clahe_false"

                reg_a = reg_t.copy().astype(np.float32)
                reg_a[:, yind[row_keep][:, None], xind[col_keep]] = cand["registered_stack"]
                reg_a_crop = reg_a[:, yind[:, None], xind]

                valid_mask_a_local = np.asarray(cand.get("valid_mask", valid_mask_aff_local), dtype=bool)
                valid_mask_a = valid_mask_t.copy()
                valid_mask_a[row_keep[:, None], col_keep] = valid_mask_a_local

                mean_a = reg_a_crop.mean(axis=0)
                metrics_a = _compute_registration_metrics(mean_img_cropped, mean_a, valid_mask=valid_mask_a)
                affine_results[key] = {
                    "registered_zstack": reg_a,
                    "registered_mean_cropped": mean_a,
                    "valid_mask": valid_mask_a,
                    "metrics": metrics_a,
                    "transform": tr,
                }
    else:
        aff_search = _search_best_affine(
            reg_input=reg_t,
            fov_ref=mean_img_cropped,
            mov_img=mean_t,
            valid_mask=valid_mask_t,
        )
        if aff_search is None:
            for use_clahe in (False, True):
                reg_a, a_info = _register_affine(
                    reg_t, mean_img_cropped, mean_t, use_clahe=use_clahe
                )
                a_info = dict(a_info)
                a_info["affine_use_clahe"] = bool(use_clahe)
                valid_mask_a = valid_mask_t
                reg_a_crop = reg_a[:, yind[:, None], xind]

                mean_a = reg_a_crop.mean(axis=0)
                metrics_a = _compute_registration_metrics(mean_img_cropped, mean_a, valid_mask=valid_mask_a)
                key = "affine_clahe_true" if use_clahe else "affine_clahe_false"
                affine_results[key] = {
                    "registered_zstack": reg_a,
                    "registered_mean_cropped": mean_a,
                    "valid_mask": valid_mask_a,
                    "metrics": metrics_a,
                    "transform": a_info,
                }
        else:
            _best_aff, all_affine = aff_search
            for cand in all_affine:
                tr = dict(cand["transform"])
                tr["candidate_protocol"] = "affine_after_translation"
                affine_candidates.append({"metrics": cand["metrics"], "transform": tr})

                use_clahe = bool(tr.get("candidate_use_clahe", tr.get("affine_use_clahe", False)))
                key = "affine_clahe_true" if use_clahe else "affine_clahe_false"
                reg_a = cand["registered_stack"]
                valid_mask_a = np.asarray(cand.get("valid_mask", valid_mask_t), dtype=bool)
                reg_a_crop = reg_a[:, yind[:, None], xind]

                mean_a = reg_a_crop.mean(axis=0)
                metrics_a = _compute_registration_metrics(mean_img_cropped, mean_a, valid_mask=valid_mask_a)
                affine_results[key] = {
                    "registered_zstack": reg_a,
                    "registered_mean_cropped": mean_a,
                    "valid_mask": valid_mask_a,
                    "metrics": metrics_a,
                    "transform": tr,
                }
    
    # Select best affine variant (and add both to candidates)
    best_affine_name, best_affine_data = min(
        affine_results.items(), key=lambda kv: _method_rank_key_from_metrics(kv[1]["metrics"])
    )
    
    # Add both variants as candidates for potential reporting
    for affine_name_cand, affine_cand_data in affine_results.items():
        if affine_name_cand != best_affine_name:
            affine_candidates.append({
                "metrics": affine_cand_data["metrics"],
                "transform": {
                    **affine_cand_data["transform"],
                    "candidate_protocol": "affine_after_translation",
                    "affine_use_clahe": affine_name_cand == "affine_clahe_true",
                }
            })
    
    methods["affine_after_translation"] = best_affine_data

    # Stage C: nonrigid search - use best of (translation, affine_clahe_false, affine_clahe_true)
    nonrigid_candidates: list[dict[str, Any]] = []
    
    # Determine best base for nonrigid: compare translation vs both affine variants
    candidates_for_nonrigid = {
        "translation": methods["translation"],
    }
    candidates_for_nonrigid.update(affine_results)
    
    best_base = min(
        candidates_for_nonrigid.items(),
        key=lambda kv: _method_rank_key_from_metrics(kv[1]["metrics"]),
    )
    best_base_name_raw, best_base_data = best_base
    best_base_name = "affine" if str(best_base_name_raw).startswith("affine_") else "translation"
    
    init_stack_full = best_base_data["registered_zstack"]
    init_stack_crop = init_stack_full[:, yind[:, None], xind]
    init_mean_crop = init_stack_crop.mean(axis=0)

    nr2_search = _search_best_nonrigid(
        init_stack_crop=init_stack_crop,
        init_mean_crop=init_mean_crop,
        mean_img_cropped=mean_img_cropped,
        valid_mask=best_base_data["valid_mask"],
        block_sizes=nonrigid_block_sizes,
        maxregshift_values=nonrigid_maxregshift_values,
    )

    if nr2_search is None:
        methods["nonrigid_after_affine"] = {
            "registered_zstack": init_stack_full,
            "registered_mean_cropped": init_mean_crop,
            "valid_mask": best_base_data["valid_mask"],
            "metrics": best_base_data["metrics"],
            "transform": {
                "failed": True,
                "fallback_to_affine": True,
                "reason": "No nonrigid-after-affine candidate succeeded",
                "candidate_block_sizes": [tuple(x) for x in nonrigid_block_sizes],
                "candidate_maxregshift_values": list(nonrigid_maxregshift_values),
                "nonrigid_protocol": "after_affine",
                "nonrigid_init_from": best_base_name,
            },
        }
    else:
        best_nr2, all_candidates2 = nr2_search
        for cand in all_candidates2:
            tr = dict(cand["transform"])
            tr["candidate_protocol"] = "nonrigid_after_affine"
            nonrigid_candidates.append({"metrics": cand["metrics"], "transform": tr})

        reg_nr2 = init_stack_full.copy().astype(np.float32)
        reg_nr2[:, yind[:, None], xind] = best_nr2["registered_crop"]
        metrics_nr2 = best_nr2["metrics"]
        valid_mask_nr2 = np.asarray(
            best_nr2.get("valid_mask", best_base_data["valid_mask"]), dtype=bool
        )

        ok_vs_affine = _nonrigid_not_worse_than_translation(
            metrics_nr=metrics_nr2,
            metrics_t=best_base_data["metrics"],
        )
        if not ok_vs_affine:
            methods["nonrigid_after_affine"] = {
                "registered_zstack": init_stack_full,
                "registered_mean_cropped": init_mean_crop,
                "valid_mask": best_base_data["valid_mask"],
                "metrics": best_base_data["metrics"],
                "transform": {
                    **best_nr2["transform"],
                    "fallback_to_affine": True,
                    "reason": "Best nonrigid-after-affine candidate worse than base",
                    "nonrigid_protocol": "after_affine",
                    "nonrigid_init_from": best_base_name,
                },
            }
        else:
            methods["nonrigid_after_affine"] = {
                "registered_zstack": reg_nr2,
                "registered_mean_cropped": best_nr2["registered_mean_cropped"],
                "valid_mask": valid_mask_nr2,
                "metrics": metrics_nr2,
                "transform": {
                    **best_nr2["transform"],
                    "fallback_to_affine": False,
                    "nonrigid_protocol": "after_affine",
                    "nonrigid_init_from": best_base_name,
                },
            }

    # Shared-mask rescoring + winner selection
    methods, shared_eval_valid_frac = _rescore_methods_with_shared_mask(
        methods=methods,
        mean_img_cropped=mean_img_cropped,
        base_mask=valid_mask,
    )
    method_rank = sorted(
        methods.items(), key=lambda kv: _method_rank_key_from_metrics(kv[1]["metrics"])
    )
    selected = method_rank[0][0]

    summary = {
        "selected_method": selected,
        "metrics_by_method": {k: v["metrics"] for k, v in methods.items()},
        "transforms_by_method": {k: v["transform"] for k, v in methods.items()},
        "shared_eval_valid_frac": shared_eval_valid_frac,
        "gate": gate,
    }

    # Build padded registered z-stack
    matched_plane_indices = data["matched_plane_indices"]
    if pad > 0:
        z_min = int(matched_plane_indices.min())
        z_max = int(matched_plane_indices.max())
        z_start = max(0, z_min - pad)
        z_end = min(zstack_data.shape[0], z_max + pad + 1)
        padded_raw = zstack_data[z_start:z_end].astype(np.float32)
        registered_zstack = _apply_winner_transforms_to_stack(
            raw_stack=padded_raw,
            selected=selected,
            methods=methods,
            crop_y_inds=yind,
            crop_x_inds=xind,
        )
        padded_plane_indices = np.arange(z_start, z_end)
    else:
        registered_zstack = methods[selected]["registered_zstack"]
        z_min = int(matched_plane_indices.min())
        z_max = int(matched_plane_indices.max())
        z_start = z_min
        z_end = z_max + 1
        padded_plane_indices = np.arange(z_start, z_end)

    return {
        "registered_zstack": registered_zstack,
        "matched_plane_indices": matched_plane_indices,
        "padded_plane_indices": padded_plane_indices,
        "padded_z_start": int(z_start),
        "padded_z_end": int(z_end),
        "fov_mean": data["mean_img"],
        "fov_mean_cropped": mean_img_cropped,
        "summary": summary,
        "all_methods": methods,
        "affine_candidates": affine_candidates,
        "nonrigid_candidates": nonrigid_candidates,
        "crop_y_inds": yind,
        "crop_x_inds": xind,
        "plane_path": data["plane_path"],
        "session_key": data["session_key"],
        "plane_id": data["plane_id"],
    }


# ---------------------------------------------------------------------------
# Result persistence (flat H5 + JSON)
# ---------------------------------------------------------------------------

def save_result(
    result: dict[str, Any],
    output_dir: Path | str,
    *,
    file_suffix: str = "local_zstack_to_fov",
) -> dict[str, Path]:
    """Save a single-plane registration result as flat H5 + metadata JSON.

    Output format is identical to the parallel variant's ``save_flat_result_h5_bundle``.

    Returns
    -------
    dict with keys ``h5_path`` and ``metadata_path``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = result.get("summary", {})
    all_methods = result.get("all_methods", {})
    winner = summary.get("selected_method", None)

    matched_plane_indices = np.asarray(result.get("matched_plane_indices", np.array([])))
    padded_plane_indices = np.asarray(result.get("padded_plane_indices", np.array([])))
    padded_z_start = int(result.get("padded_z_start", -1))
    padded_z_end = int(result.get("padded_z_end", -1))

    if matched_plane_indices.size > 0:
        z_drift_min = int(np.min(matched_plane_indices))
        z_drift_max = int(np.max(matched_plane_indices))
    else:
        z_drift_min = -1
        z_drift_max = -1

    if padded_z_start < 0 and padded_plane_indices.size > 0:
        padded_z_start = int(np.min(padded_plane_indices))
    if padded_z_end < 0 and padded_plane_indices.size > 0:
        padded_z_end = int(np.max(padded_plane_indices) + 1)

    if z_drift_min >= 0 and z_drift_max >= 0 and padded_z_start >= 0 and padded_z_end >= 0:
        pad_lower = max(0, z_drift_min - padded_z_start)
        pad_upper = max(0, padded_z_end - (z_drift_max + 1))
        pad_value = int(min(pad_lower, pad_upper))
    else:
        pad_value = -1

    session_tok = _safe_filename_token(result.get("session_key"), default="NA")
    plane_tok = _safe_filename_token(result.get("plane_id"), default="NA")
    stem = f"{session_tok}_{plane_tok}_{file_suffix}"
    h5_path = output_dir / f"{stem}_results.h5"
    metadata_path = output_dir / f"{stem}_metadata.json"

    array_keys = ("tmat", "nr_ymax1", "nr_xmax1", "nr_yblock", "nr_xblock")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset(
            "registered_zstack",
            data=result.get("registered_zstack", np.array([])),
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset("matched_plane_indices", data=matched_plane_indices)
        f.create_dataset("padded_plane_indices", data=padded_plane_indices)
        f.create_dataset("crop_y_inds", data=result.get("crop_y_inds", np.array([])))
        f.create_dataset("crop_x_inds", data=result.get("crop_x_inds", np.array([])))
        f.create_dataset(
            "fov_mean",
            data=result.get("fov_mean", np.array([])),
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "fov_mean_cropped",
            data=result.get("fov_mean_cropped", np.array([])),
            compression="gzip",
            compression_opts=4,
        )

        f.attrs["session_key"] = str(result.get("session_key") or "")
        f.attrs["plane_id"] = str(result.get("plane_id") or "")
        f.attrs["selected_method"] = "" if winner is None else str(winner)
        f.attrs["padded_z_start"] = int(padded_z_start)
        f.attrs["padded_z_end"] = int(padded_z_end)
        f.attrs["z_drift_min"] = int(z_drift_min)
        f.attrs["z_drift_max"] = int(z_drift_max)
        f.attrs["desired_z_start_with_padding"] = int(padded_z_start)
        f.attrs["desired_z_end_with_padding"] = int(padded_z_end)
        f.attrs["pad"] = int(pad_value)

        tg = f.create_group("transforms")
        for method_name, method_data in all_methods.items():
            mg = tg.create_group(str(method_name))
            transform = dict(method_data.get("transform", {}))
            for key in array_keys:
                if key in transform:
                    mg.create_dataset(key, data=np.asarray(transform[key]))
            scalar_tr = {k: v for k, v in transform.items() if k not in array_keys}
            mg.attrs["transform_json"] = json.dumps(_to_json_safe(scalar_tr))

    metadata = {
        "session_key": result.get("session_key"),
        "plane_id": result.get("plane_id"),
        "plane_path": result.get("plane_path"),
        "selected_method": winner,
        "metrics_by_method": summary.get("metrics_by_method", {}),
        "transforms_by_method": {k: v.get("transform", {}) for k, v in all_methods.items()},
        "matched_plane_indices": matched_plane_indices,
        "padded_plane_indices": padded_plane_indices,
        "padded_z_start": padded_z_start,
        "padded_z_end": padded_z_end,
        "z_stack_padding_info": {
            "z_drift_range": [int(z_drift_min), int(z_drift_max)],
            "desired_padded_range": [int(padded_z_start), int(padded_z_end)],
            "pad": int(pad_value),
            "note": "Saved z-stack contains the padded range with winner transforms applied.",
        },
        "h5_file": str(h5_path),
    }
    with open(metadata_path, "w") as f:
        json.dump(_to_json_safe(metadata), f, indent=2)

    return {"h5_path": h5_path, "metadata_path": metadata_path}


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def list_saved_result_files(output_dir: Path | str) -> list[Path]:
    """Return all ``*_metadata.json`` flat result files in ``output_dir``."""
    return sorted(Path(output_dir).glob("*_metadata.json"))


def get_registration_benchmark_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Build a tidy per-plane benchmark summary dict from a result.

    Suitable for converting to a DataFrame row or logging.
    """
    summary = result.get("summary", {})
    metrics_by_method = summary.get("metrics_by_method", {})

    def _metric(method: str, key: str, default: float = float("nan")) -> float:
        return metrics_by_method.get(method, {}).get(key, default)

    return {
        "session_key": result.get("session_key"),
        "plane_id": result.get("plane_id"),
        "selected_method": summary.get("selected_method"),
        "translation_residual_l2": _metric("translation", "residual_shift_l2_px", float("inf")),
        "affine_after_translation_residual_l2": _metric("affine_after_translation", "residual_shift_l2_px", float("inf")),
        "nonrigid_after_affine_residual_l2": _metric("nonrigid_after_affine", "residual_shift_l2_px", float("inf")),
        "translation_ncc": _metric("translation", "ncc"),
        "affine_after_translation_ncc": _metric("affine_after_translation", "ncc"),
        "nonrigid_after_affine_ncc": _metric("nonrigid_after_affine", "ncc"),
        "shared_eval_valid_frac": summary.get("shared_eval_valid_frac"),
    }


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------

def run_single_plane(
    plane_path: Path | str,
    output_dir: Path | str,
    *,
    reg_ref_ind: int = 0,
    save_tiff: bool = True,
    file_suffix: str = "local_zstack_to_fov",
    nonrigid_block_sizes: tuple[tuple[int, int], ...] = ((32, 32), (64, 64), (128, 128)),
    nonrigid_maxregshift_values: tuple[int, ...] = (3, 5),
    pad: int = 3,
) -> dict[str, Any]:
    """End-to-end single-plane pipeline: z-stack prep → register → save.

    Returns
    -------
    dict with ``result`` (registration result dict) and ``saved_paths``
    (dict with ``h5_path`` and ``metadata_path``).
    """
    result = register_local_zstack_to_fov(
        plane_path=plane_path,
        reg_ref_ind=reg_ref_ind,
        save_tiff=save_tiff,
        output_dir=output_dir,
        nonrigid_block_sizes=nonrigid_block_sizes,
        nonrigid_maxregshift_values=nonrigid_maxregshift_values,
        pad=pad,
    )
    saved_paths = save_result(result, output_dir, file_suffix=file_suffix)
    return {"result": result, "saved_paths": saved_paths}

