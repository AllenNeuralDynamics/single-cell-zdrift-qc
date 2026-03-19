from __future__ import annotations

from functools import cmp_to_key
from typing import Any
import json

import h5py
import numpy as np
import torch
from pystackreg import StackReg
from skimage import exposure
from skimage.metrics import structural_similarity as ssim
from skimage.registration import phase_cross_correlation
from suite2p.registration import nonrigid
from scipy.ndimage import binary_closing
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label as ndi_label
from scipy.ndimage import shift as ndi_shift


def _normalize(img):
    arr = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)

    pos = arr[finite & (arr > 0)]
    if pos.size:
        lo = float(np.min(pos))
    else:
        lo = float(np.min(arr[finite]))
    hi = float(np.max(arr[finite]))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)

    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0, 1).astype(np.float32)


def _gradient_magnitude(img):
    gy, gx = np.gradient(img.astype(np.float32))
    return np.sqrt(gx * gx + gy * gy)


def _safe_corrcoef(a, b):
    a = a.ravel()
    b = b.ravel()
    if a.size < 10:
        return np.nan
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _localized_mean_corr(a, b, valid_mask=None, grid=(4, 4), min_pixels=64):
    if a.shape != b.shape:
        return np.nan

    gy, gx = grid
    h, w = a.shape
    y_edges = np.linspace(0, h, gy + 1, dtype=int)
    x_edges = np.linspace(0, w, gx + 1, dtype=int)

    vals = []
    for yi in range(gy):
        ys, ye = y_edges[yi], y_edges[yi + 1]
        for xi in range(gx):
            xs, xe = x_edges[xi], x_edges[xi + 1]
            a_tile = a[ys:ye, xs:xe]
            b_tile = b[ys:ye, xs:xe]

            if valid_mask is None:
                vm = np.isfinite(a_tile) & np.isfinite(b_tile)
            else:
                vm = valid_mask[ys:ye, xs:xe] & np.isfinite(a_tile) & np.isfinite(b_tile)

            if int(vm.sum()) < int(min_pixels):
                continue
            c = _safe_corrcoef(a_tile[vm], b_tile[vm])
            if np.isfinite(c):
                vals.append(float(c))

    if not vals:
        return np.nan
    return float(np.mean(vals))


def _compute_registration_metrics(fov_img, mov_img, valid_mask=None, enforce_exact_mask=False):
    if valid_mask is None:
        valid_mask = np.isfinite(fov_img) & np.isfinite(mov_img)
    else:
        valid_mask = valid_mask & np.isfinite(fov_img) & np.isfinite(mov_img)

    if not enforce_exact_mask:
        valid_mask = valid_mask & (fov_img > 0) & (mov_img > 0)
    valid_frac = float(valid_mask.mean())

    if valid_mask.sum() < 100:
        valid_mask = np.isfinite(fov_img) & np.isfinite(mov_img)
        valid_frac = float(valid_mask.mean())

    ncc = _safe_corrcoef(fov_img[valid_mask], mov_img[valid_mask])
    localized_ncc = _localized_mean_corr(fov_img, mov_img, valid_mask=valid_mask)

    try:
        ssim_val = float(ssim(fov_img, mov_img, data_range=1.0))
    except Exception:
        ssim_val = np.nan

    diff = fov_img - mov_img
    nrmse = float(np.sqrt(np.mean(diff[valid_mask] ** 2))) if valid_mask.any() else np.nan

    residual_shift, _, _ = phase_cross_correlation(fov_img, mov_img, upsample_factor=20, normalization=None)
    residual_shift = np.asarray(residual_shift, dtype=np.float32)

    return {
        "ncc": ncc,
        "localized_ncc": localized_ncc,
        "ssim": ssim_val,
        "nrmse": nrmse,
        "valid_frac": valid_frac,
        "residual_shift_y_px": float(residual_shift[0]),
        "residual_shift_x_px": float(residual_shift[1]),
        "residual_shift_l2_px": float(np.linalg.norm(residual_shift)),
        "residual_shift_max_abs_px": float(np.max(np.abs(residual_shift))),
    }


def _build_shared_eval_mask(base_mask, fov_img, images, method_masks=None, min_pixels=100):
    shared = np.asarray(base_mask, dtype=bool).copy()
    shared &= np.isfinite(fov_img)
    shared &= (fov_img > 0)
    if method_masks is not None:
        for m in method_masks:
            if m is None:
                continue
            shared &= np.asarray(m, dtype=bool)
    for img in images:
        shared &= np.isfinite(img)
        shared &= (img > 0)

    if int(shared.sum()) < int(min_pixels):
        # If strict positivity intersection is too small, keep fair finite-pixel overlap.
        shared = np.asarray(base_mask, dtype=bool).copy()
        shared &= np.isfinite(fov_img)
        for img in images:
            shared &= np.isfinite(img)

    return shared


def _rescore_methods_with_shared_mask(methods, mean_img_cropped, base_mask):
    method_names = [m for m in ("translation", "affine_after_translation", "nonrigid", "nonrigid_after_affine") if m in methods]
    if not method_names:
        return methods, float(np.asarray(base_mask, dtype=bool).mean())

    shared_mask = _build_shared_eval_mask(
        base_mask=base_mask,
        fov_img=mean_img_cropped,
        images=[methods[m]["registered_mean_cropped"] for m in method_names],
        method_masks=[methods[m].get("valid_mask", None) for m in method_names],
    )

    for m in method_names:
        old_metrics = methods[m]["metrics"]
        new_metrics = _compute_registration_metrics(
            mean_img_cropped,
            methods[m]["registered_mean_cropped"],
            valid_mask=shared_mask,
            enforce_exact_mask=True,
        )
        for key, val in old_metrics.items():
            if key not in new_metrics:
                new_metrics[key] = val
        methods[m]["metrics"] = new_metrics

    return methods, float(shared_mask.mean())


def _safe_metric_for_max(v):
    try:
        fv = float(v)
    except Exception:
        return -np.inf
    return fv if np.isfinite(fv) else -np.inf


def _safe_metric_for_min(v):
    try:
        fv = float(v)
    except Exception:
        return np.inf
    return fv if np.isfinite(fv) else np.inf


def _mean_two(a, b, prefer_max=True):
    vals = []
    for x in (a, b):
        try:
            fx = float(x)
            if np.isfinite(fx):
                vals.append(fx)
        except Exception:
            continue
    if not vals:
        return -np.inf if prefer_max else np.inf
    return float(np.mean(vals))


def _method_rank_key_from_metrics(metrics: dict[str, Any]):
    ncc_val = _safe_metric_for_max(metrics.get("ncc", np.nan))
    localized_ncc_val = _safe_metric_for_max(metrics.get("localized_ncc", np.nan))
    ssim_val = _safe_metric_for_max(metrics.get("ssim", np.nan))
    return (-ncc_val, -localized_ncc_val, -ssim_val)


def _compare_metrics_by_priority(
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
    *,
    ncc_tie_tol: float = 0.005,
    localized_ncc_tie_tol: float = 0.002,
) -> int:
    """Compare two metric dicts using lexicographic tolerances for method selection.

    Higher NCC wins unless the values are within ``ncc_tie_tol``.
    If NCC is effectively tied, higher localized NCC wins unless the values are within
    ``localized_ncc_tie_tol``. SSIM is the final tiebreaker.
    Returns ``-1`` when ``metrics_a`` should rank ahead of ``metrics_b``.
    """

    def _cmp_max(key: str, tie_tol: float) -> int:
        a_val = _safe_metric_for_max(metrics_a.get(key, np.nan))
        b_val = _safe_metric_for_max(metrics_b.get(key, np.nan))
        if a_val > b_val + tie_tol:
            return -1
        if b_val > a_val + tie_tol:
            return 1
        return 0

    for key, tie_tol in (("ncc", ncc_tie_tol), ("localized_ncc", localized_ncc_tie_tol), ("ssim", 0.0)):
        cmp_result = _cmp_max(key, tie_tol)
        if cmp_result != 0:
            return cmp_result

    for key in ("ncc", "localized_ncc", "ssim"):
        cmp_result = _cmp_max(key, 0.0)
        if cmp_result != 0:
            return cmp_result

    return 0


def _candidate_rank_key_from_metrics(metrics: dict[str, Any]):
    return _method_rank_key_from_metrics(metrics)


def _passes_gate(metrics, gate):
    return (
        metrics["residual_shift_l2_px"] <= gate["max_residual_l2_px"]
        and metrics["residual_shift_max_abs_px"] <= gate["max_residual_abs_px"]
        and (np.isnan(metrics["ncc"]) or metrics["ncc"] >= gate["min_ncc"])
        and (np.isnan(metrics["ssim"]) or metrics["ssim"] >= gate["min_ssim"])
        and metrics["valid_frac"] >= gate["min_valid_frac"]
    )


def _translation_mask_from_full(
    full_binary_mask,
    shift_yx,
    crop_y_inds,
    crop_x_inds,
    cropped_overlap_mask=None,
):
    full_mask = np.asarray(full_binary_mask, dtype=bool)
    moved_full = _transform_binary_valid_mask(
        full_mask,
        method="translation",
        shift_yx=np.asarray(shift_yx, dtype=np.float32),
    )
    moved_crop = np.asarray(moved_full[crop_y_inds[:, None], crop_x_inds], dtype=bool)
    if cropped_overlap_mask is not None:
        moved_crop &= np.asarray(cropped_overlap_mask, dtype=bool)
    return moved_crop


def _transform_binary_valid_mask(
    binary_mask,
    method,
    shift_yx=None,
    tmat=None,
    nonrigid_blocks=None,
    nonrigid_shifts=None,
):
    mask = np.asarray(binary_mask, dtype=np.float32)
    mask = np.where(np.isfinite(mask) & (mask > 0), 1.0, 0.0).astype(np.float32)

    def _postprocess_mask(mask_bool):
        out = np.asarray(mask_bool, dtype=bool)
        if not out.any():
            return out

        # 1) fill enclosed holes, 2) keep the main connected support,
        # 3) close thin edge cuts introduced by interpolation.
        out = np.asarray(binary_fill_holes(out), dtype=bool)
        labeled, nlab = ndi_label(out)
        if nlab > 1:
            counts = np.bincount(labeled.ravel())
            if counts.size > 1:
                counts[0] = 0
                out = labeled == int(np.argmax(counts))
        out = np.asarray(binary_closing(out, structure=np.ones((3, 3), dtype=bool), iterations=1), dtype=bool)
        out = np.asarray(binary_fill_holes(out), dtype=bool)
        return out

    if method == "translation":
        if shift_yx is None:
            raise ValueError("shift_yx is required for translation mask transform")
        reg = ndi_shift(mask, shift=tuple(np.asarray(shift_yx, dtype=np.float32)), mode="constant", cval=0.0, order=0)
        out = np.asarray(reg > 0.5, dtype=bool)
        return _postprocess_mask(out)

    if method == "affine":
        if tmat is None:
            raise ValueError("tmat is required for affine mask transform")
        sr = StackReg(StackReg.AFFINE)
        reg = sr.transform(mask, tmat=np.asarray(tmat, dtype=np.float32))
        # Use a permissive threshold because interpolation can push interior pixels below 0.5.
        out = np.asarray(reg > 1e-6, dtype=bool)
        return _postprocess_mask(out)

    if method == "nonrigid":
        if nonrigid_blocks is None or nonrigid_shifts is None:
            raise ValueError("nonrigid_blocks and nonrigid_shifts are required for nonrigid mask transform")
        nblocks, xblock, yblock = nonrigid_blocks
        ymax1, xmax1 = nonrigid_shifts
        mask_t = torch.from_numpy(mask)[None, :, :]
        reg = nonrigid.transform_data(
            data=mask_t,
            nblocks=nblocks,
            xblock=xblock,
            yblock=yblock,
            ymax1=ymax1,
            xmax1=xmax1,
        )
        reg_arr = np.asarray(reg, dtype=np.float32)
        if reg_arr.ndim == 3:
            reg_arr = reg_arr[0]
        # Suite2p nonrigid uses interpolation; avoid creating speckled false negatives.
        out = np.asarray(reg_arr > 1e-6, dtype=bool)
        return _postprocess_mask(out)

    raise ValueError(f"Unknown method for mask transform: {method}")


def _register_translation(
    matched_zstack,
    mean_img_cropped,
    mean_zstack_cropped,
    valid_mask=None,
    full_binary_mask=None,
    crop_y_inds=None,
    crop_x_inds=None,
    cropped_overlap_mask=None,
):
    shift, _, _ = phase_cross_correlation(mean_img_cropped, mean_zstack_cropped, upsample_factor=20, normalization=None)
    registered_zstack = ndi_shift(matched_zstack, shift=(0, *shift), mode="constant", cval=0)
    out = {"shift_yx": np.asarray(shift, dtype=np.float32)}
    if full_binary_mask is not None and crop_y_inds is not None and crop_x_inds is not None:
        out["updated_valid_mask"] = _translation_mask_from_full(
            full_binary_mask=full_binary_mask,
            shift_yx=np.asarray(shift, dtype=np.float32),
            crop_y_inds=np.asarray(crop_y_inds, dtype=int),
            crop_x_inds=np.asarray(crop_x_inds, dtype=int),
            cropped_overlap_mask=cropped_overlap_mask,
        )
    elif valid_mask is not None:
        out["updated_valid_mask"] = _transform_binary_valid_mask(
            valid_mask,
            method="translation",
            shift_yx=np.asarray(shift, dtype=np.float32),
        )
    return registered_zstack, out


def _register_affine(matched_zstack, mean_img_cropped, mean_zstack_cropped, use_clahe=False):
    sr = StackReg(StackReg.AFFINE)

    ref = _normalize(mean_img_cropped)
    mov = _normalize(mean_zstack_cropped)
    if use_clahe:
        ref = exposure.equalize_adapthist(ref)
        mov = exposure.equalize_adapthist(mov)

    tmat = sr.register(ref, mov)

    reg_stack = np.zeros_like(matched_zstack, dtype=np.float32)
    for zi in range(matched_zstack.shape[0]):
        reg_stack[zi] = sr.transform(matched_zstack[zi], tmat=tmat)

    return reg_stack, {
        "tmat": tmat,
        "affine_det2x2": float(np.linalg.det(tmat[:2, :2])),
        "affine_tx": float(tmat[0, 2]),
        "affine_ty": float(tmat[1, 2]),
    }


def _prep_affine_image(img, normalize=True, use_clahe=False):
    out = img.astype(np.float32)
    if normalize:
        out = _normalize(out)
    if use_clahe:
        # CLAHE expects bounded input. If normalization is disabled for this mode,
        # still scale to [0, 1] with global min/max to keep behavior deterministic.
        if not normalize:
            finite = np.isfinite(out)
            if finite.any():
                lo = float(np.min(out[finite]))
                hi = float(np.max(out[finite]))
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    out = (out - lo) / (hi - lo)
                else:
                    out = np.zeros_like(out, dtype=np.float32)
            else:
                out = np.zeros_like(out, dtype=np.float32)
        out = exposure.equalize_adapthist(out)
    return out.astype(np.float32)


def _register_affine_with_preprocess(
    matched_zstack,
    mean_img_cropped,
    mean_zstack_cropped,
    normalize=True,
    use_clahe=False,
    valid_mask=None,
):
    sr = StackReg(StackReg.AFFINE)

    ref = _prep_affine_image(mean_img_cropped, normalize=normalize, use_clahe=use_clahe)
    mov = _prep_affine_image(mean_zstack_cropped, normalize=normalize, use_clahe=use_clahe)

    tmat = sr.register(ref, mov)

    reg_stack = np.zeros_like(matched_zstack, dtype=np.float32)
    for zi in range(matched_zstack.shape[0]):
        reg_stack[zi] = sr.transform(matched_zstack[zi], tmat=tmat)

    updated_valid_mask = None
    if valid_mask is not None:
        updated_valid_mask = _transform_binary_valid_mask(valid_mask, method="affine", tmat=tmat)

    return reg_stack, {
        "tmat": tmat,
        "affine_det2x2": float(np.linalg.det(tmat[:2, :2])),
        "affine_tx": float(tmat[0, 2]),
        "affine_ty": float(tmat[1, 2]),
        "affine_use_clahe": bool(use_clahe),
        "updated_valid_mask": updated_valid_mask,
    }


def _prep_nonrigid_image(img, normalize=True, use_clahe=False):
    out = img.astype(np.float32)
    if normalize:
        out = _normalize(out)
    if use_clahe:
        if not normalize:
            finite = np.isfinite(out)
            if finite.any():
                lo = float(np.min(out[finite]))
                hi = float(np.max(out[finite]))
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    out = (out - lo) / (hi - lo)
                else:
                    out = np.zeros_like(out, dtype=np.float32)
            else:
                out = np.zeros_like(out, dtype=np.float32)
        out = exposure.equalize_adapthist(out)
    return out.astype(np.float32)


def _register_nonrigid_suite2p(
    matched_zstack_cropped,
    moving_mean_img_cropped,
    ref_img_cropped,
    maxregshift_nr=3,
    block_size=(128, 128),
    normalize=True,
    use_clahe=False,
    valid_mask=None,
):
    Ly, Lx = ref_img_cropped.shape
    yblock, xblock, nblocks, block_size_eff, NRsm, Kmat, nup = nonrigid.make_blocks(
        Ly, Lx, block_size=block_size, lpad=3, subpixel=10
    )

    ref_t = torch.from_numpy(_prep_nonrigid_image(ref_img_cropped, normalize=normalize, use_clahe=use_clahe))
    mov_t = torch.from_numpy(_prep_nonrigid_image(moving_mean_img_cropped, normalize=normalize, use_clahe=use_clahe))[None, :, :]

    maskMul, maskOffset, cfRefImg = nonrigid.compute_masks_ref_smooth_fft(
        refImg0=ref_t,
        maskSlope=3.0,
        smooth_sigma=1.15,
        yblock=yblock,
        xblock=xblock,
    )

    blocks = (yblock, xblock, nblocks, block_size_eff, NRsm, Kmat, nup)
    out = nonrigid.phasecorr(
        data=mov_t,
        blocks=blocks,
        maskMul=maskMul,
        maskOffset=maskOffset,
        cfRefImg=cfRefImg,
        snr_thresh=1.2,
        maxregshiftNR=maxregshift_nr,
        subpixel=10,
        lpad=3,
    )

    ymax1, xmax1 = out[0], out[1]
    ymax1 = torch.clamp(ymax1, -maxregshift_nr, maxregshift_nr)
    xmax1 = torch.clamp(xmax1, -maxregshift_nr, maxregshift_nr)

    reg_stack_crop = np.zeros_like(matched_zstack_cropped, dtype=np.float32)
    for zi in range(matched_zstack_cropped.shape[0]):
        plane_t = torch.from_numpy(matched_zstack_cropped[zi].astype(np.float32))[None, :, :]
        reg_plane = nonrigid.transform_data(
            data=plane_t,
            nblocks=nblocks,
            xblock=xblock,
            yblock=yblock,
            ymax1=ymax1,
            xmax1=xmax1,
        )
        reg_stack_crop[zi] = np.asarray(reg_plane, dtype=np.float32)

    updated_valid_mask = None
    if valid_mask is not None:
        updated_valid_mask = _transform_binary_valid_mask(
            valid_mask,
            method="nonrigid",
            nonrigid_blocks=(nblocks, xblock, yblock),
            nonrigid_shifts=(ymax1, xmax1),
        )

    return reg_stack_crop, {
        "maxregshift_nr": float(maxregshift_nr),
        "nr_block_size": tuple(block_size_eff),
        "nr_shift_max_abs": float(max(torch.max(torch.abs(ymax1)), torch.max(torch.abs(xmax1))).item()),
        "nr_use_clahe": bool(use_clahe),
        "nr_ymax1": ymax1.cpu().numpy(),
        "nr_xmax1": xmax1.cpu().numpy(),
        "nr_yblock": np.array(yblock),
        "nr_xblock": np.array(xblock),
        "nr_nblocks": list(nblocks),
        "updated_valid_mask": updated_valid_mask,
    }


def _compute_blank_safe_crop_inds(stack_3d, min_valid_frac=0.9):
    plane_min = stack_3d.min(axis=0)
    valid = plane_min > 0
    row_keep = np.where(valid.sum(axis=1) > valid.shape[1] * (1 - (1 - min_valid_frac)))[0]
    col_keep = np.where(valid.sum(axis=0) > valid.shape[0] * (1 - (1 - min_valid_frac)))[0]
    if len(row_keep) == 0:
        row_keep = np.where(valid.sum(axis=1) > 0)[0]
    if len(col_keep) == 0:
        col_keep = np.where(valid.sum(axis=0) > 0)[0]
    return row_keep, col_keep


def _nonrigid_not_worse_than_translation(
    metrics_nr,
    metrics_t,
    max_extra_l2=0.03,
    max_extra_abs=0.03,
    max_ncc_drop=0.01,
    max_ssim_drop=0.01,
):
    if metrics_nr is None:
        return False
    if not np.isfinite(metrics_nr.get("residual_shift_l2_px", np.nan)):
        return False
    if not np.isfinite(metrics_nr.get("residual_shift_max_abs_px", np.nan)):
        return False

    l2_ok = metrics_nr["residual_shift_l2_px"] <= metrics_t["residual_shift_l2_px"] + max_extra_l2
    abs_ok = metrics_nr["residual_shift_max_abs_px"] <= metrics_t["residual_shift_max_abs_px"] + max_extra_abs

    ncc_t = metrics_t.get("ncc", np.nan)
    ncc_nr = metrics_nr.get("ncc", np.nan)
    ncc_ok = True if (np.isnan(ncc_t) or np.isnan(ncc_nr)) else (ncc_nr >= ncc_t - max_ncc_drop)

    ssim_t = metrics_t.get("ssim", np.nan)
    ssim_nr = metrics_nr.get("ssim", np.nan)
    ssim_ok = True if (np.isnan(ssim_t) or np.isnan(ssim_nr)) else (ssim_nr >= ssim_t - max_ssim_drop)

    return l2_ok and abs_ok and ncc_ok and ssim_ok


def _candidate_block_sizes(shape_hw):
    Ly, Lx = shape_hw
    max_b = min(Ly, Lx)
    base = [64, 96, 128, 160]
    out = [(b, b) for b in base if b <= max_b]
    if not out:
        b = max(32, max_b // 2)
        out = [(int(b), int(b))]
    return out


def _candidate_maxregshift_values(tmag):
    _ = tmag
    return [3, 5, 10]


def _search_best_nonrigid(
    init_stack_crop,
    init_mean_crop,
    mean_img_cropped,
    valid_mask,
    block_sizes=None,
    maxregshift_values=None,
):
    if block_sizes is None:
        block_sizes = _candidate_block_sizes(mean_img_cropped.shape)

    if maxregshift_values is None:
        maxregshift_values = [3]

    preprocess_opts = [
        {"normalize": True, "use_clahe": False},
        {"normalize": True, "use_clahe": True},
    ]

    candidates = []
    for maxreg in maxregshift_values:
        for bs in block_sizes:
            for pp in preprocess_opts:
                try:
                    reg_crop, nr_info = _register_nonrigid_suite2p(
                        matched_zstack_cropped=init_stack_crop,
                        moving_mean_img_cropped=init_mean_crop,
                        ref_img_cropped=mean_img_cropped,
                        maxregshift_nr=int(maxreg),
                        block_size=bs,
                        normalize=pp["normalize"],
                        use_clahe=pp["use_clahe"],
                        valid_mask=valid_mask,
                    )
                    mean_nr = reg_crop.mean(axis=0)
                    candidate_valid_mask = nr_info.pop("updated_valid_mask", None)
                    if candidate_valid_mask is None:
                        candidate_valid_mask = valid_mask
                    metrics_nr = _compute_registration_metrics(mean_img_cropped, mean_nr, valid_mask=candidate_valid_mask)
                    nr_info = dict(nr_info)
                    nr_info.update(
                        {
                            "candidate_block_size": tuple(bs),
                            "candidate_use_clahe": bool(pp["use_clahe"]),
                            "candidate_preprocess_mode": "clahe" if pp["use_clahe"] else "none",
                            "candidate_maxregshift_nr": int(maxreg),
                        }
                    )
                    candidates.append(
                        {
                            "registered_crop": reg_crop,
                            "registered_mean_cropped": mean_nr,
                            "valid_mask": np.asarray(candidate_valid_mask, dtype=bool),
                            "metrics": metrics_nr,
                            "transform": nr_info,
                        }
                    )
                except Exception:
                    continue

    if not candidates:
        return None

    shared_mask = _build_shared_eval_mask(
        base_mask=valid_mask,
        fov_img=mean_img_cropped,
        images=[c["registered_mean_cropped"] for c in candidates],
        method_masks=[c.get("valid_mask", None) for c in candidates],
    )
    for c in candidates:
        c["metrics"] = _compute_registration_metrics(
            mean_img_cropped,
            c["registered_mean_cropped"],
            valid_mask=shared_mask,
            enforce_exact_mask=True,
        )

    candidates = sorted(
        candidates,
        key=cmp_to_key(lambda a, b: _compare_metrics_by_priority(a["metrics"], b["metrics"])),
    )
    best = candidates[0]
    best["transform"] = dict(best["transform"])
    best["transform"]["num_nonrigid_candidates"] = len(candidates)
    best["transform"]["shared_eval_valid_frac"] = float(shared_mask.mean())
    return best, candidates


def _search_best_affine(
    reg_input,
    fov_ref,
    mov_img,
    valid_mask,
):
    preprocess_opts = [
        {"normalize": True, "use_clahe": False},
        {"normalize": True, "use_clahe": True},
    ]

    candidates = []
    for pp in preprocess_opts:
        try:
            reg_a, a_info = _register_affine_with_preprocess(
                reg_input,
                fov_ref,
                mov_img,
                normalize=pp["normalize"],
                use_clahe=pp["use_clahe"],
                valid_mask=valid_mask,
            )
            mean_a = reg_a.mean(axis=0)
            candidate_valid_mask = a_info.pop("updated_valid_mask", None)
            if candidate_valid_mask is None:
                candidate_valid_mask = valid_mask
            metrics_a = _compute_registration_metrics(fov_ref, mean_a, valid_mask=candidate_valid_mask)
            a_info = dict(a_info)
            a_info.update(
                {
                    "candidate_use_clahe": bool(pp["use_clahe"]),
                    "candidate_preprocess_mode": "clahe" if pp["use_clahe"] else "none",
                }
            )
            candidates.append(
                {
                    "registered_stack": reg_a,
                    "registered_mean": mean_a,
                    "valid_mask": np.asarray(candidate_valid_mask, dtype=bool),
                    "metrics": metrics_a,
                    "transform": a_info,
                }
            )
        except Exception:
            continue

    if not candidates:
        return None

    shared_mask = _build_shared_eval_mask(
        base_mask=valid_mask,
        fov_img=fov_ref,
        images=[c["registered_mean"] for c in candidates],
        method_masks=[c.get("valid_mask", None) for c in candidates],
    )
    for c in candidates:
        c["metrics"] = _compute_registration_metrics(
            fov_ref,
            c["registered_mean"],
            valid_mask=shared_mask,
            enforce_exact_mask=True,
        )

    candidates = sorted(
        candidates,
        key=cmp_to_key(lambda a, b: _compare_metrics_by_priority(a["metrics"], b["metrics"])),
    )
    best = candidates[0]
    best["transform"] = dict(best["transform"])
    best["transform"]["num_affine_candidates"] = len(candidates)
    best["transform"]["shared_eval_valid_frac"] = float(shared_mask.mean())
    return best, candidates


def load_transformations_from_json(json_path: str) -> dict[str, Any]:
    """Load transformations from a JSON file.
    
    Args:
        json_path: Path to the JSON file containing transformations
        
    Returns:
        Dictionary with transformation data
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def reproduce_transformations_from_saved(
    matched_zstack: np.ndarray,
    transforms_by_method: dict[str, Any],
    crop_y_inds: np.ndarray,
    crop_x_inds: np.ndarray,
) -> dict[str, np.ndarray]:
    """Re-apply saved transformations to a z-stack in sequence and return per-method stacks.

    The pipeline stages are applied in order:
      1. translation  - shift applied to original matched_zstack
      2. affine_after_translation - affine applied to translation result
      3. nonrigid_after_affine  - nonrigid applied to affine result

    Args:
        matched_zstack: Raw matched z-stack array, shape (Z, H, W).
        transforms_by_method: Dict keyed by method name, each value containing the
            transform parameters as saved (e.g. from metadata.json with numpy arrays
            loaded back from the h5 transforms group).
        crop_y_inds: 1-D index array for row crop.
        crop_x_inds: 1-D index array for col crop.

    Returns:
        Dict mapping method name to reproduced registered z-stack (float32).
    """
    reproduced: dict[str, np.ndarray] = {}
    current = matched_zstack.astype(np.float32)

    # --- Stage 1: translation ---
    t_tr = transforms_by_method.get("translation", {})
    if t_tr and "shift_yx" in t_tr:
        shift = np.asarray(t_tr["shift_yx"], dtype=np.float32)
        current = ndi_shift(current, shift=(0.0, float(shift[0]), float(shift[1])), mode="constant", cval=0)
    reproduced["translation"] = current.copy()

    # --- Stage 2: affine_after_translation ---
    a_tr = transforms_by_method.get("affine_after_translation", {})
    if a_tr and "tmat" in a_tr:
        sr = StackReg(StackReg.AFFINE)
        tmat = np.asarray(a_tr["tmat"], dtype=np.float32)
        current_a = np.zeros_like(current, dtype=np.float32)
        for zi in range(current.shape[0]):
            current_a[zi] = sr.transform(current[zi], tmat=tmat)
        current = current_a
    reproduced["affine_after_translation"] = current.copy()

    # --- Stage 3: nonrigid_after_affine ---
    nr_tr = transforms_by_method.get("nonrigid_after_affine", {})
    if nr_tr and "nr_ymax1" in nr_tr and not nr_tr.get("failed", False):
        ymax1 = torch.from_numpy(np.asarray(nr_tr["nr_ymax1"], dtype=np.float32))
        xmax1 = torch.from_numpy(np.asarray(nr_tr["nr_xmax1"], dtype=np.float32))
        nblocks = tuple(int(x) for x in nr_tr["nr_nblocks"])
        yblock = np.asarray(nr_tr["nr_yblock"])
        xblock = np.asarray(nr_tr["nr_xblock"])

        stack_crop = current[:, crop_y_inds[:, None], crop_x_inds]
        reg_crop = np.zeros_like(stack_crop, dtype=np.float32)
        for zi in range(stack_crop.shape[0]):
            plane_t = torch.from_numpy(stack_crop[zi].astype(np.float32))[None, :, :]
            reg_plane = nonrigid.transform_data(
                data=plane_t,
                nblocks=nblocks,
                xblock=xblock,
                yblock=yblock,
                ymax1=ymax1,
                xmax1=xmax1,
            )
            reg_crop[zi] = np.asarray(reg_plane, dtype=np.float32)
        current_nr = current.copy()
        current_nr[:, crop_y_inds[:, None], crop_x_inds] = reg_crop
        current = current_nr
    reproduced["nonrigid_after_affine"] = current.copy()

    return reproduced



def _apply_winner_transforms_to_stack(
    raw_stack,
    selected,
    methods,
    crop_y_inds,
    crop_x_inds,
):
    """Apply the winner transform pipeline sequentially to a raw z-stack of arbitrary length.

    Stages applied in order (stopping after selected):
      1. translation
      2. affine_after_translation
      3. nonrigid_after_affine

    Used to extend the registration to padding planes outside the matched range.
    """
    current = raw_stack.astype(np.float32)

    # Stage 1: translation
    t_tr = methods.get("translation", {}).get("transform", {})
    if "shift_yx" in t_tr:
        shift = np.asarray(t_tr["shift_yx"], dtype=np.float32)
        current = ndi_shift(current, shift=(0.0, float(shift[0]), float(shift[1])), mode="constant", cval=0)
    if selected == "translation":
        return current

    # Stage 2: affine_after_translation
    a_tr = methods.get("affine_after_translation", {}).get("transform", {})
    if "tmat" in a_tr:
        sr = StackReg(StackReg.AFFINE)
        tmat = np.asarray(a_tr["tmat"], dtype=np.float32)
        current_a = np.zeros_like(current, dtype=np.float32)
        for zi in range(current.shape[0]):
            current_a[zi] = sr.transform(current[zi], tmat=tmat)
        current = current_a
    if selected == "affine_after_translation":
        return current

    # Stage 3: nonrigid_after_affine
    nr_tr = methods.get("nonrigid_after_affine", {}).get("transform", {})
    if not nr_tr.get("failed", False) and "nr_ymax1" in nr_tr:
        ymax1 = torch.from_numpy(np.asarray(nr_tr["nr_ymax1"], dtype=np.float32))
        xmax1 = torch.from_numpy(np.asarray(nr_tr["nr_xmax1"], dtype=np.float32))
        nblocks = tuple(int(x) for x in nr_tr["nr_nblocks"])
        yblock = np.asarray(nr_tr["nr_yblock"])
        xblock = np.asarray(nr_tr["nr_xblock"])

        stack_crop = current[:, crop_y_inds[:, None], crop_x_inds]
        reg_crop = np.zeros_like(stack_crop, dtype=np.float32)
        for zi in range(stack_crop.shape[0]):
            plane_t = torch.from_numpy(stack_crop[zi].astype(np.float32))[None, :, :]
            reg_plane = nonrigid.transform_data(
                data=plane_t,
                nblocks=nblocks,
                xblock=xblock,
                yblock=yblock,
                ymax1=ymax1,
                xmax1=xmax1,
            )
            reg_crop[zi] = np.asarray(reg_plane, dtype=np.float32)
        current[:, crop_y_inds[:, None], crop_x_inds] = reg_crop

    return current


def apply_transformations_to_hdf5(
    input_hdf5_path: str,
    output_hdf5_path: str,
    transform_data: dict[str, Any],
    dataset_name: str = 'data',
) -> None:
    """Apply transformations to data in an HDF5 file.
    
    Args:
        input_hdf5_path: Path to input HDF5 file
        output_hdf5_path: Path to output HDF5 file
        transform_data: Dictionary containing transformation parameters
        dataset_name: Name of the dataset within the HDF5 file to transform
    """
    with h5py.File(input_hdf5_path, 'r') as f_in:
        data = f_in[dataset_name][()]
    
    # Apply the transformation based on the type
    transform_type = transform_data.get('type', 'affine')
    
    if transform_type == 'affine':
        # Apply affine transformation
        from scipy.ndimage import affine_transform
        matrix = np.array(transform_data['matrix'])
        offset = np.array(transform_data.get('offset', [0, 0]))
        transformed_data = affine_transform(data, matrix, offset=offset, order=1, cval=0)
    elif transform_type == 'shift':
        # Apply shift transformation
        shift_vector = transform_data['shift']
        transformed_data = ndi_shift(data, shift_vector, order=1, cval=0)
    else:
        raise ValueError(f"Unknown transformation type: {transform_type}")
    
    # Save transformed data to output HDF5
    with h5py.File(output_hdf5_path, 'w') as f_out:
        f_out.create_dataset(dataset_name, data=transformed_data)
        # Copy metadata if present
        if 'metadata' in transform_data:
            for key, value in transform_data['metadata'].items():
                f_out.attrs[key] = value
