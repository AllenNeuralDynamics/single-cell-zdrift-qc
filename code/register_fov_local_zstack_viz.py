from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    return float(metrics.get(key, np.nan))


def _subparam_summary(transform: dict[str, Any]) -> str:
    clahe = transform.get(
        "candidate_use_clahe",
        transform.get("nr_use_clahe", transform.get("affine_use_clahe", "n/a")),
    )
    block = transform.get("candidate_block_size", transform.get("nr_block_size", "n/a"))
    maxreg = transform.get("candidate_maxregshift_nr", transform.get("maxregshift_nr", "n/a"))
    lines = [
        "params:",
        f"  clahe: {clahe}",
        f"  block_size: {block}",
        f"  maxregshift_nr: {maxreg}",
    ]
    if "nonrigid_init_from" in transform:
        lines.append(f"  nonrigid_base: {transform.get('nonrigid_init_from')}")
    return "\n".join(lines)


def _shared_display_bounds(images: list[np.ndarray]) -> tuple[float, float]:
    finite_chunks = []
    pos_chunks = []
    for img in images:
        arr = np.asarray(img, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            finite_chunks.append(finite)
            pos = finite[finite > 0]
            if pos.size:
                pos_chunks.append(pos)

    if pos_chunks:
        lo = float(np.min(np.concatenate(pos_chunks)))
    elif finite_chunks:
        lo = float(np.min(np.concatenate(finite_chunks)))
    else:
        return 0.0, 1.0

    hi = float(np.max(np.concatenate(finite_chunks)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0, 1.0
    return lo, hi


def _bounds_from_masked_values(
    a: np.ndarray,
    b: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[float, float]:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    vm = np.asarray(valid_mask, dtype=bool)
    vm &= np.isfinite(aa)
    vm &= np.isfinite(bb)

    if int(vm.sum()) < 10:
        return _shared_display_bounds([aa, bb])

    vals = np.concatenate([aa[vm], bb[vm]])
    lo, hi = np.percentile(vals, [1.0, 99.0])
    lo = float(lo)
    hi = float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return _shared_display_bounds([aa, bb])
    return lo, hi


def _scale_for_display(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0, 1).astype(np.float32)


def _valid_overlap_mask(images: list[np.ndarray]) -> np.ndarray:
    if not images:
        return np.zeros((0, 0), dtype=bool)

    mask = np.ones_like(np.asarray(images[0]), dtype=bool)
    for img in images:
        arr = np.asarray(img, dtype=np.float32)
        mask &= np.isfinite(arr)
        mask &= (arr > 0)
    return mask


def _method_valid_mask(
    fov_img: np.ndarray,
    mov_img: np.ndarray,
    method_payload: dict[str, Any],
) -> np.ndarray:
    method_mask = method_payload.get("valid_mask", None)
    if method_mask is None:
        return _valid_overlap_mask([fov_img, mov_img])

    vm = np.asarray(method_mask, dtype=bool)
    vm &= np.isfinite(np.asarray(fov_img, dtype=np.float32))
    vm &= np.isfinite(np.asarray(mov_img, dtype=np.float32))
    vm &= (np.asarray(fov_img, dtype=np.float32) > 0)
    vm &= (np.asarray(mov_img, dtype=np.float32) > 0)

    if int(vm.sum()) < 10:
        return _valid_overlap_mask([fov_img, mov_img])
    return vm


def _build_method_comparison_figure(
    fov_raw: np.ndarray,
    methods: dict,
    row_i: int,
    session_key: str,
    plane_id: str,
    selected_method: str,
    method_order: tuple[str, ...] = ("translation", "affine_after_translation", "nonrigid", "nonrigid_after_affine"),
) -> Any:
    """Build a method comparison figure with n_methods 3-panel rows + metric bar chart.
    
    Internal helper used by save_method_comparison_figure and visualize_method_comparison.
    Handles all panel layout, metrics text, and styling logic.
    """
    available = [m for m in method_order if m in methods]
    if not available:
        return None

    n_methods = len(available)
    fig = plt.figure(figsize=(13, 4 * n_methods + 3.5))
    gs = fig.add_gridspec(n_methods + 1, 3, height_ratios=[1] * n_methods + [0.9])

    for i, m in enumerate(available):
        mov_raw = methods[m]["registered_mean_cropped"]
        valid_mask = _method_valid_mask(fov_raw, mov_raw, methods[m])
        lo, hi = _bounds_from_masked_values(fov_raw, mov_raw, valid_mask)
        fov = _scale_for_display(fov_raw, lo, hi)
        mov = _scale_for_display(mov_raw, lo, hi)
        diff = fov - mov
        blend = np.stack([fov, mov, np.zeros_like(fov)], axis=-1)

        ax0 = fig.add_subplot(gs[i, 0])
        ax0.imshow(blend)
        ax0.set_title(f"{m} overlay (R=FOV, G=registered)")
        ax0.axis("off")

        ax1 = fig.add_subplot(gs[i, 1])
        ax1.imshow(diff, cmap="bwr", vmin=-0.5, vmax=0.5)
        ax1.set_title(f"{m} residual")
        ax1.axis("off")

        txt = methods[m]["metrics"]
        tr = methods[m].get("transform", {})
        ax2 = fig.add_subplot(gs[i, 2])
        ax2.axis("off")
        ax2.text(
            0.0,
            1.0,
            "\n".join(
                [
                    f"method: {m}",
                    f"ncc: {txt.get('ncc', np.nan):.4f}",
                    f"localized_ncc: {txt.get('localized_ncc', np.nan):.4f}",
                    f"ssim: {txt.get('ssim', np.nan):.4f}",
                    f"shift_residual_l2: {txt.get('residual_shift_l2_px', np.nan):.4f}",
                    f"shift_residual_abs: {txt.get('residual_shift_max_abs_px', np.nan):.4f}",
                    "",
                    _subparam_summary(tr),
                ]
            ),
            va="top",
            family="monospace",
            fontsize=9,
        )

    metric_names = ["ncc", "localized_ncc", "ssim"]
    x = np.arange(len(metric_names), dtype=float)
    width = 0.8 / max(1, n_methods)
    ax_bar = fig.add_subplot(gs[n_methods, :])
    for k, m in enumerate(available):
        met = methods[m].get("metrics", {})
        vals = [_metric_value(met, mn) for mn in metric_names]
        xpos = x + (k - (n_methods - 1) / 2.0) * width
        ax_bar.bar(xpos, vals, width=width, label=m)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_names, rotation=15)
    ax_bar.set_ylim(-0.1, 1.05)
    ax_bar.set_ylabel("metric value")
    ax_bar.set_title("Metric comparison across methods")
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.legend(loc="upper right", ncol=max(1, min(3, n_methods)))

    fig.suptitle(f"session_key={session_key} | plane_id={plane_id} | winner={selected_method}", y=1.02, fontsize=10)
    fig.tight_layout()
    return fig


def save_method_comparison_figure(result: dict[str, Any], row_i: int, out_path: Path) -> None:
    """Save a three-panel-per-method comparison figure for one plane result.

    Parameters:
    -----------
    result : dict
        Registration result dict with 'fov_mean_cropped' and 'all_methods'
    row_i : int
        Row index for figure title
    out_path : Path
        Output path for saved PNG figure
    """
    fov_raw = np.asarray(result["fov_mean_cropped"], dtype=np.float32)
    methods = result["all_methods"]
    session_key = result.get("session_key", "NA")
    plane_id = result.get("plane_id", "NA")
    selected_method = result.get("summary", {}).get("selected_method", "NA")

    matplotlib.use("Agg")
    fig = _build_method_comparison_figure(fov_raw, methods, row_i, session_key, plane_id, selected_method)
    if fig is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _has_visualization_payload(result: dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    if "fov_mean_cropped" not in result or "all_methods" not in result:
        return False
    methods = result.get("all_methods")
    return isinstance(methods, dict) and len(methods) > 0


def _resolve_result_pair(result_path: Path | str) -> tuple[Path, Path]:
    """Resolve metadata and h5 paths from either file in a result pair.

    Supports both legacy bundle names (metadata.json + result.h5) and flat names
    (*_metadata.json + *_results.h5).
    """
    p = Path(result_path)

    if p.is_dir():
        meta = p / "metadata.json"
        h5 = p / "result.h5"
        if meta.exists() and h5.exists():
            return meta, h5

    name = p.name
    suffix = p.suffix.lower()

    if name == "metadata.json":
        return p, p.with_name("result.h5")
    if name == "result.h5":
        return p.with_name("metadata.json"), p

    if name.endswith("_metadata.json"):
        stem = name[: -len("_metadata.json")]
        return p, p.with_name(f"{stem}_results.h5")
    if name.endswith("_results.h5"):
        stem = name[: -len("_results.h5")]
        return p.with_name(f"{stem}_metadata.json"), p

    if suffix == ".json":
        # If this is not a known metadata filename pattern, assume json->h5 stem swap.
        return p, p.with_suffix(".h5")
    if suffix == ".h5":
        # If this is not a known h5 filename pattern, assume h5->json stem swap.
        return p.with_suffix(".json"), p

    raise ValueError(f"Cannot infer metadata/h5 pair from path: {p}")


def load_result_from_bundle(
    result_path: Path | str,
    h5_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load a full result dict from metadata + h5 result files."""
    import h5py

    if h5_path is None:
        metadata_path, h5_file_path = _resolve_result_pair(result_path)
    else:
        metadata_path = Path(result_path)
        h5_file_path = Path(h5_path)

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not h5_file_path.exists():
        raise FileNotFoundError(h5_file_path)
    
    with open(metadata_path, "r") as f:
        manifest = json.load(f)

    result = {}
    with h5py.File(h5_file_path, "r") as f:
        if "registered_zstack" in f:
            result["registered_zstack"] = np.array(f["registered_zstack"])
        if "matched_plane_indices" in f:
            result["matched_plane_indices"] = np.array(f["matched_plane_indices"])
        if "padded_plane_indices" in f:
            result["padded_plane_indices"] = np.array(f["padded_plane_indices"])
        if "crop_y_inds" in f:
            result["crop_y_inds"] = np.array(f["crop_y_inds"])
        if "crop_x_inds" in f:
            result["crop_x_inds"] = np.array(f["crop_x_inds"])
        if "fov_mean" in f:
            result["fov_mean"] = np.array(f["fov_mean"])
        if "fov_mean_cropped" in f:
            result["fov_mean_cropped"] = np.array(f["fov_mean_cropped"])

        padded_z_start = int(f.attrs.get("padded_z_start", -1))
        padded_z_end = int(f.attrs.get("padded_z_end", -1))
        desired_z_start = int(f.attrs.get("desired_z_start_with_padding", padded_z_start))
        desired_z_end = int(f.attrs.get("desired_z_end_with_padding", padded_z_end))

        h5_pad = f.attrs.get("pad", None)
        if h5_pad is None and ("matched_plane_indices" in result) and desired_z_start >= 0 and desired_z_end >= 0:
            matched = np.asarray(result["matched_plane_indices"]).astype(int)
            if matched.size > 0:
                z_min = int(np.min(matched))
                z_max = int(np.max(matched))
                lower = max(0, z_min - desired_z_start)
                upper = max(0, desired_z_end - (z_max + 1))
                h5_pad = int(min(lower, upper))

        result["z_stack_padding"] = {
            "z_drift_min": int(f.attrs.get("z_drift_min", -1)),
            "z_drift_max": int(f.attrs.get("z_drift_max", -1)),
            "desired_z_start_with_padding": int(desired_z_start),
            "desired_z_end_with_padding": int(desired_z_end),
            "pad": int(h5_pad) if h5_pad is not None else -1,
        }

        if result["z_stack_padding"]["z_drift_min"] < 0 and ("matched_plane_indices" in result):
            matched = np.asarray(result["matched_plane_indices"]).astype(int)
            if matched.size > 0:
                result["z_stack_padding"]["z_drift_min"] = int(np.min(matched))
                result["z_stack_padding"]["z_drift_max"] = int(np.max(matched))

    if "z_stack_padding_info" in manifest:
        padding_info = manifest["z_stack_padding_info"]
        result["z_stack_padding"] = {
            "z_drift_min": padding_info.get("z_drift_range", [-1, -1])[0],
            "z_drift_max": padding_info.get("z_drift_range", [-1, -1])[1],
            "desired_z_start_with_padding": padding_info.get("desired_padded_range", [-1, -1])[0],
            "desired_z_end_with_padding": padding_info.get("desired_padded_range", [-1, -1])[1],
            "pad": padding_info.get("pad", -1),
        }
    elif ("padded_z_start" in manifest) or ("padded_z_end" in manifest):
        current = result.get("z_stack_padding", {})
        result["z_stack_padding"] = {
            "z_drift_min": int(current.get("z_drift_min", -1)),
            "z_drift_max": int(current.get("z_drift_max", -1)),
            "desired_z_start_with_padding": int(manifest.get("padded_z_start", current.get("desired_z_start_with_padding", -1))),
            "desired_z_end_with_padding": int(manifest.get("padded_z_end", current.get("desired_z_end_with_padding", -1))),
            "pad": int(current.get("pad", -1)),
        }

    result["summary"] = {
        "selected_method": manifest.get("selected_method"),
        "metrics_by_method": manifest.get("metrics_by_method", {}),
        "transforms_by_method": manifest.get("transforms_by_method", {}),
        "shared_eval_valid_frac": manifest.get("shared_eval_valid_frac", np.nan),
        "gate": manifest.get("gate", {}),
    }
    result["all_methods"] = {}
    result["affine_candidates"] = []
    result["nonrigid_candidates"] = []
    
    # Add metadata from manifest
    result["row_i"] = int(manifest.get("row_i", -1))
    result["winner"] = manifest.get("winner", None)
    result["session_key"] = manifest.get("session_key", None)
    result["plane_path"] = manifest.get("plane_path", None)
    result["plane_id"] = manifest.get("plane_id", None)
    result["processed_name"] = manifest.get("processed_name", None)
    result["source_result_file"] = str(metadata_path)
    
    return result


def load_result_from_npy_bundle(bundle_path: Path | str) -> dict[str, Any]:
    """Backward-compatible alias for older notebooks/scripts."""
    return load_result_from_bundle(bundle_path)


def _infer_bundle_metadata_path(result_path: Path) -> Path | None:
    if result_path.name == "metadata.json":
        return result_path

    if result_path.name.endswith("_metadata.json"):
        return result_path

    if result_path.suffix.lower() == ".json" and result_path.parent.name == "per_plane":
        per_plane_dir = result_path.parent
        candidate = per_plane_dir.parent / "per_plane_full" / result_path.stem / "metadata.json"
        return candidate

    return None


def prepare_visualization_result_from_file(
    result_file: Path | str,
    *,
    nonrigid_block_sizes: list[tuple[int, int]] | None = None,
    nonrigid_maxregshift_values: list[int] | None = None,
) -> tuple[dict[str, Any], bool]:
    """Load visualization payload from file, or re-run registration from plane_path.

    If the file already contains the full visualization payload (transforms and
    images), it is returned directly. Otherwise, ``plane_path`` stored in the
    metadata is used to re-run :func:`register_local_zstack_to_fov`.

    Returns
    -------
    result : dict
        Full result payload compatible with :func:`save_method_comparison_figure`.
    reprepared : bool
        True if registration was re-run.
    """
    result_path = Path(result_file)

    # Prefer direct bundle load when metadata/h5 pair file is explicitly provided.
    if (
        result_path.name == "metadata.json"
        or result_path.name == "result.h5"
        or result_path.name.endswith("_metadata.json")
        or result_path.name.endswith("_results.h5")
    ):
        bundled = load_result_from_bundle(result_path)
        if _has_visualization_payload(bundled):
            return bundled, False
        raise ValueError(f"Bundle file exists but has no visualization payload: {result_path}")

    with open(result_path, "r") as f:
        summary_or_full = json.load(f)

    if _has_visualization_payload(summary_or_full):
        return summary_or_full, False

    # For summary-only per_plane json, try loading full payload from bundle first.
    bundle_metadata = _infer_bundle_metadata_path(result_path)
    if bundle_metadata is not None and bundle_metadata.exists():
        bundled = load_result_from_bundle(bundle_metadata)
        if _has_visualization_payload(bundled):
            return bundled, False

    plane_path_str = summary_or_full.get("plane_path", None)
    if not plane_path_str:
        raise KeyError(f"Cannot re-prepare: no plane_path in result file: {result_path}")

    mod_name = "register_fov_local_zstack"
    try:
        reg_mod = importlib.import_module(mod_name)
    except ModuleNotFoundError:
        this_dir = str(Path(__file__).resolve().parent)
        if this_dir not in sys.path:
            sys.path.append(this_dir)
        reg_mod = importlib.import_module(mod_name)

    reg_kwargs: dict[str, Any] = {"plane_path": Path(plane_path_str)}
    if nonrigid_block_sizes is not None:
        reg_kwargs["nonrigid_block_sizes"] = tuple(nonrigid_block_sizes)
    if nonrigid_maxregshift_values is not None:
        reg_kwargs["nonrigid_maxregshift_values"] = tuple(nonrigid_maxregshift_values)

    rebuilt = reg_mod.register_local_zstack_to_fov(**reg_kwargs)
    rebuilt["source_result_file"] = str(result_path)
    return rebuilt, True


def save_method_comparison_figure_from_result_file(
    result_file: Path | str,
    out_path: Path | str,
    *,
    nonrigid_block_sizes: list[tuple[int, int]] | None = None,
    nonrigid_maxregshift_values: list[int] | None = None,
) -> bool:
    """Save comparison figure, loading or re-running registration as needed.

    Returns
    -------
    bool
        True if registration was re-run, False if loaded from file.
    """
    result, reprepared = prepare_visualization_result_from_file(
        result_file=result_file,
        nonrigid_block_sizes=nonrigid_block_sizes,
        nonrigid_maxregshift_values=nonrigid_maxregshift_values,
    )
    session_key = result.get("session_key", "NA")
    plane_id = result.get("plane_id", "NA")
    row_i = int(result.get("row_i", -1))
    save_method_comparison_figure(result=result, row_i=row_i, out_path=Path(out_path))
    return reprepared


def visualize_method_comparison(
    plane_path: Path | str,
    register_fn,
    method_order: tuple[str, ...] = ("translation", "affine_after_translation", "nonrigid_after_affine"),
) -> tuple[Any, dict[str, Any]] | dict[str, Any]:
    """Render in-notebook method comparison and return the registration result dict."""
    r = register_fn(plane_path)
    fov_raw = np.asarray(r["fov_mean_cropped"], dtype=np.float32)
    methods = r["all_methods"]
    session_key = r.get("session_key", "NA")
    plane_id = r.get("plane_id", "NA")
    selected_method = r.get("summary", {}).get("selected_method", "NA")
    row_i = -1

    fig = _build_method_comparison_figure(
        fov_raw, methods, row_i, session_key, plane_id, selected_method, method_order
    )
    if fig is None:
        return r
    return fig, r


def visualize_pairwise_method_differences(result: dict[str, Any]) -> tuple[Any, list[tuple[str, str]]]:
    """Plot pairwise residual differences in fixed method order for QC comparison.

    Uses fixed order (translation, affine, nonrigid, nonrigid_after_affine).
    Plots all available methods for visual QC inspection.
    Returns the figure and plotted method pairs in display order.
    """
    methods = result.get("all_methods", {})
    if not methods:
        return None, []

    # Fixed method order for consistent QC layout across all rows.
    fixed_method_order = ["translation", "affine_after_translation", "nonrigid", "nonrigid_after_affine"]
    all_methods = [m for m in fixed_method_order if m in methods]

    pairs = []
    for i in range(len(all_methods)):
        for j in range(i + 1, len(all_methods)):
            pairs.append((all_methods[i], all_methods[j]))

    if not pairs:
        return None, []

    ncols = min(3, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axs = np.atleast_1d(axs).ravel()

    for ax, (ma, mb) in zip(axs, pairs):
        a_raw = methods[ma]["registered_mean_cropped"]
        b_raw = methods[mb]["registered_mean_cropped"]
        mask_a = np.asarray(methods[ma].get("valid_mask", np.ones_like(a_raw, dtype=bool)), dtype=bool)
        mask_b = np.asarray(methods[mb].get("valid_mask", np.ones_like(b_raw, dtype=bool)), dtype=bool)
        pair_mask = mask_a & mask_b
        pair_mask &= np.isfinite(np.asarray(a_raw, dtype=np.float32))
        pair_mask &= np.isfinite(np.asarray(b_raw, dtype=np.float32))
        pair_mask &= (np.asarray(a_raw, dtype=np.float32) > 0)
        pair_mask &= (np.asarray(b_raw, dtype=np.float32) > 0)
        lo, hi = _bounds_from_masked_values(a_raw, b_raw, pair_mask)
        a = _scale_for_display(a_raw, lo, hi)
        b = _scale_for_display(b_raw, lo, hi)
        diff = a - b
        mad = float(np.mean(np.abs(diff)))

        ax.imshow(diff, cmap="bwr", vmin=-0.5, vmax=0.5)
        ax.set_title(f"{ma} - {mb} | mean|diff|={mad:.4f}")
        ax.axis("off")

    for ax in axs[len(pairs) :]:
        ax.axis("off")

    fig.suptitle(
        (
            f"Pairwise differences across all methods (fixed order QC)\n"
            f"session_key={result.get('session_key', 'NA')} | plane_id={result.get('plane_id', 'NA')}"
        ),
        y=1.02,
        fontsize=15,
    )
    fig.tight_layout()
    return fig, pairs
