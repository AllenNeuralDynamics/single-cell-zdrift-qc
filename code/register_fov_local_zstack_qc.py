"""QC utilities for single-plane local z-stack to FOV registration.

Provides:
- z-drift intensity-profile analysis (ROI and neuropil signal vs z)
- Registration QC image: ROI/neuropil mask outlines overlaid on mean registered z-stack
- QC GIF: animated registered z-stack matched range with colour-coded ROI outlines
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont
from skimage import measure

from lamf_analysis.code_ocean import capsule_data_utils as cdu
from lamf_analysis.ophys import roi_utils


# ---------------------------------------------------------------------------
# z-drift matched index helpers
# ---------------------------------------------------------------------------

def get_binned_zdrift_minmax(result: dict[str, Any], window_size: int = 5) -> tuple[int, int]:
    """Return the (min, max) smoothed z-drift plane indices over a rolling window.

    Uses a median rolling window of ``window_size`` frames (approx. 5 min at
    typical frame rate) on ``matched_plane_indices``.
    """
    filtered = (
        pd.Series(result["matched_plane_indices"])
        .rolling(window=window_size, min_periods=window_size, center=True)
        .median()
        .dropna()
        .values.astype(int)
    )
    return int(filtered.min()), int(filtered.max())


def get_zdrift_matched_inds(result: dict[str, Any], zdrift_calc_bin: int = 5) -> tuple[int, int]:
    """Return (start, end) indices into the padded registered z-stack
    corresponding to the smoothed z-drift min/max.

    These index into ``result['registered_zstack']`` (the padded stack).
    """
    padded_indices = result["padded_plane_indices"]
    z_min, z_max = get_binned_zdrift_minmax(result, window_size=zdrift_calc_bin)
    start = int(np.where(padded_indices == z_min)[0][0])
    end = int(np.where(padded_indices == z_max)[0][0])
    return start, end


def get_matched_registered_zstack(result: dict[str, Any], zdrift_calc_bin: int = 5) -> np.ndarray:
    """Return the slice of the registered z-stack spanning the z-drift matched range."""
    start, end = get_zdrift_matched_inds(result, zdrift_calc_bin)
    return result["registered_zstack"][start : end + 1]


def get_mean_registered_zstack(result: dict[str, Any], zdrift_calc_bin: int = 5) -> np.ndarray:
    """Return the mean image over the z-drift matched planes."""
    matched = get_matched_registered_zstack(result, zdrift_calc_bin)
    return np.mean(matched, axis=0)


# ---------------------------------------------------------------------------
# ROI intensity analysis
# ---------------------------------------------------------------------------

def calculate_roi_intensity_by_z(row: pd.Series, zstack: np.ndarray) -> np.ndarray:
    """Mean soma pixel intensity at each z-plane, masked to valid (non-zero) pixels."""
    mask = row.mask_matrix
    binary_mask = (mask > 0) & (np.mean(zstack, axis=0) > 0)
    if binary_mask.sum() == 0:
        return np.full(zstack.shape[0], np.nan)
    masked = zstack * binary_mask[None, :, :]
    return masked.sum(axis=(1, 2)) / binary_mask.sum()


def calculate_neuropil_intensity(row: pd.Series, zstack_mean: np.ndarray) -> float:
    """Mean neuropil pixel intensity from the mean z-stack image."""
    mask = (row.np_mask > 0) & (zstack_mean > 0)
    if mask.sum() == 0:
        return float("nan")
    return float((zstack_mean * mask).sum() / mask.sum())


def get_valid_roi_table(result: dict[str, Any],
                        zdrift_calc_bin: int = 5) -> pd.DataFrame:
    """Build a valid-ROI table annotated with z-dependent intensity metrics.

    Adds columns:
    - ``zdrift_matched_ind_min/max``: indices into registered z-stack
    - ``soma_intensity_by_z``: per-z soma intensity array
    - ``np_intensity``: scalar neuropil intensity (mean from matched planes mean)
    - ``np_subtracted_soma_intensity_by_z``: neuropil-subtracted, smoothed per-z signal
    - ``np_subtracted_soma_intensity_by_z_matched``: slice over matched range
    - ``intensity_min_max_ratio``: min/max ratio (≤1, negative for inverted)
    - ``intensity_top_bottom_ratio``: top/bottom ratio (≤1, negative for inverted)
    """
    reg_zstack = result["registered_zstack"]
    mean_zstack = get_mean_registered_zstack(result, zdrift_calc_bin)
    start_ind, end_ind = get_zdrift_matched_inds(result, zdrift_calc_bin)

    plane_path = result.get("plane_path", "")
    roi_table = cdu.get_roi_table_from_plane_path(plane_path)
    roi_table = roi_utils.append_neuropil_masks_to_roi_table(roi_table, plane_path)
    valid_roi_table = roi_table[roi_table["valid_roi"]].copy()

    valid_roi_table["zdrift_matched_ind_min"] = start_ind
    valid_roi_table["zdrift_matched_ind_max"] = end_ind

    valid_roi_table["soma_intensity_by_z"] = valid_roi_table.apply(
        calculate_roi_intensity_by_z, axis=1, zstack=reg_zstack
    )
    valid_roi_table["np_intensity"] = valid_roi_table.apply(
        calculate_neuropil_intensity, axis=1, zstack_mean=mean_zstack
    )
    valid_roi_table["np_subtracted_soma_intensity_by_z"] = valid_roi_table.apply(
        lambda row: (
            row.soma_intensity_by_z - row.np_intensity
            if not np.isnan(row.np_intensity)
            else np.full_like(row.soma_intensity_by_z, np.nan)
        ),
        axis=1,
    )
    # Smooth with rolling window of 3
    valid_roi_table["np_subtracted_soma_intensity_by_z"] = valid_roi_table[
        "np_subtracted_soma_intensity_by_z"
    ].apply(
        lambda x: pd.Series(x).rolling(window=3, min_periods=1, center=True).mean().values
    )
    valid_roi_table["np_subtracted_soma_intensity_by_z_matched"] = valid_roi_table.apply(
        lambda row: (
            row.np_subtracted_soma_intensity_by_z[
                row.zdrift_matched_ind_min : row.zdrift_matched_ind_max + 1
            ]
            if not np.isnan(row.np_intensity)
            else np.full(
                row.zdrift_matched_ind_max - row.zdrift_matched_ind_min + 1, np.nan
            )
        ),
        axis=1,
    )

    def _min_max_ratio(x: np.ndarray) -> float:
        v = float(np.min(x) / np.max(x)) if np.max(x) != 0 else float("nan")
        return v if v <= 1 else -v

    def _top_bottom_ratio(x: np.ndarray) -> float:
        if x[-1] > x[0]:
            v = float(x[0] / x[-1])
        else:
            v = float(x[-1] / x[0]) if x[0] != 0 else float("nan")
        return v if v <= 1 else -v

    valid_roi_table["intensity_min_max_ratio"] = valid_roi_table[
        "np_subtracted_soma_intensity_by_z_matched"
    ].apply(_min_max_ratio)
    valid_roi_table["intensity_top_bottom_ratio"] = valid_roi_table[
        "np_subtracted_soma_intensity_by_z_matched"
    ].apply(_top_bottom_ratio)

    valid_roi_table.reset_index(drop=True, inplace=True)
    valid_roi_table.reset_index(inplace=True)
    valid_roi_table.rename(columns={"index": "roi_ind"}, inplace=True)
    return valid_roi_table


# ---------------------------------------------------------------------------
# QC image: ROI/neuropil overlays on mean z-stack
# ---------------------------------------------------------------------------

def make_roi_overlay_image(
    result: dict[str, Any],
    valid_roi_table: pd.DataFrame,
    *,
    zdrift_calc_bin: int = 5,
    roi_color: tuple[int, int, int] = (255, 0, 0),
    neuropil_color: tuple[int, int, int] = (255, 255, 0),
    show_neuropil: bool = True,
) -> np.ndarray:
    """RGB image of the mean registered z-stack with ROI and neuropil mask outlines.

    Color coding:
    - ROI outlines      : red by default
    - Neuropil outlines : yellow by default

    Parameters
    ----------
    result : dict
        Registration result from :func:`register_fov_local_zstack_to_fov`.
    valid_roi_table : DataFrame
        Output of :func:`get_valid_roi_table`.
    Returns
    -------
    np.ndarray  (H, W, 3)  uint8
    """
    mean_img = get_mean_registered_zstack(result, zdrift_calc_bin)

    valid_px = mean_img[mean_img > 1]
    if valid_px.size == 0:
        valid_px = mean_img.ravel()
    lo = float(np.percentile(valid_px, 1))
    hi = float(np.percentile(valid_px, 99.5))
    norm = ((mean_img - lo) / max(hi - lo, 1e-6) * 255).clip(0, 255).astype(np.uint8)
    rgb = np.stack([norm, norm, norm], axis=-1)

    for _, row in valid_roi_table.iterrows():
        mask = (row.mask_matrix > 0).astype(np.uint8)
        for contour in measure.find_contours(mask, 0.5):
            c = contour.astype(int)
            for r, col in zip(c[:, 0], c[:, 1]):
                if 0 <= r < rgb.shape[0] and 0 <= col < rgb.shape[1]:
                    rgb[r, col] = list(roi_color)

        if show_neuropil and hasattr(row, "np_mask") and row.np_mask is not None:
            np_mask = (row.np_mask > 0).astype(np.uint8)
            for contour in measure.find_contours(np_mask, 0.5):
                c = contour.astype(int)
                for r, col in zip(c[:, 0], c[:, 1]):
                    if 0 <= r < rgb.shape[0] and 0 <= col < rgb.shape[1]:
                        rgb[r, col] = list(neuropil_color)

    return rgb


def save_roi_overlay_image(
    result: dict[str, Any],
    valid_roi_table: pd.DataFrame,
    save_path: Path | str,
    *,
    zdrift_calc_bin: int = 5,
    **kwargs: Any,
) -> Path:
    """Save the ROI overlay image as a PNG. Returns the save path."""
    rgb = make_roi_overlay_image(result, valid_roi_table, zdrift_calc_bin=zdrift_calc_bin, **kwargs)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(save_path), rgb)
    return save_path


# ---------------------------------------------------------------------------
# QC GIF: animated registered z-stack matched range with ROI outlines
# ---------------------------------------------------------------------------

def _pil_font(size: int = 12) -> ImageFont.ImageFont:
    """Load a PIL font at the requested size, with robust fallbacks."""
    import glob as _glob
    # Candidate filenames in priority order
    candidates = [
        "DejaVuSansMono.ttf",
        "DejaVuSans.ttf",
        "LiberationMono-Regular.ttf",
        "FreeMono.ttf",
    ]
    # Build search dirs: system dirs + matplotlib bundled fonts
    search_dirs = ["/usr/share/fonts", "/usr/local/share/fonts"]
    try:
        import matplotlib
        mpl_font_dir = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf")
        search_dirs.insert(0, mpl_font_dir)
    except Exception:
        pass

    for name in candidates:
        # Try PIL's own font lookup first (works if font is on system PATH)
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            pass
        # Try explicit paths in known dirs
        for d in search_dirs:
            full = str(Path(d) / name)
            try:
                return ImageFont.truetype(full, size)
            except (IOError, OSError):
                pass
        # Try recursive glob under each search dir
        for d in search_dirs:
            for hit in _glob.glob(f"{d}/**/{name}", recursive=True):
                try:
                    return ImageFont.truetype(hit, size)
                except (IOError, OSError):
                    pass

    # Final fallback: Pillow ≥ 10 supports size parameter
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def make_qc_gif_frames(
    result: dict[str, Any],
    valid_roi_table: pd.DataFrame,
    *,
    zdrift_calc_bin: int = 5,
    intensity_threshold: float = 0.5,
    roi_color_pass: tuple[int, int, int] = (255, 0, 0),
    roi_color_negative: tuple[int, int, int] = (0, 255, 255),
    roi_color_fail: tuple[int, int, int] = (255, 255, 0),
    font_size: int = 28,
) -> list[np.ndarray]:
    """Build per-frame RGB images for the QC GIF.

    Layout (top to bottom):
    1. Title header  – session key | plane ID | drift planes (centred, white)
    2. Column headers – one per panel: preset label + z=XX (yellow)
    3. Image panels   – three side-by-side contrast normalisations

    Returns
    -------
    list of np.ndarray  (H_total, W_total, 3)  uint8
    """
    reg_zstack_matched = get_matched_registered_zstack(result, zdrift_calc_bin)

    valid_px = reg_zstack_matched[reg_zstack_matched > 1]
    if valid_px.size == 0:
        valid_px = reg_zstack_matched.ravel()

    # Three contrast presets: (lo_pct, hi_pct, label)
    contrast_presets = [
        (0.1, 98.0, "lower (p0.1-p98)"),
        (1.0, 99.5, "medium (p1-p99.5)"),
        (3.0, 99.99, "higher (p3-p99.99)"),
    ]
    stacks_u8 = []
    for lo_pct, hi_pct, _ in contrast_presets:
        lo = float(np.percentile(valid_px, lo_pct))
        hi = float(np.percentile(valid_px, hi_pct))
        s = ((reg_zstack_matched - lo) / max(hi - lo, 1e-6) * 255).clip(0, 255).astype(np.uint8)
        stacks_u8.append(s)

    # Metadata
    session_key = result.get("session_key", "NA")
    plane_id = result.get("plane_id", "NA")
    start_ind, end_ind = get_zdrift_matched_inds(result, zdrift_calc_bin)
    total_drift_planes = end_ind - start_ind + 1
    padded_indices = np.asarray(result.get("padded_plane_indices", []))

    title_line = f"{session_key}  |  {plane_id}  |  drift planes: {total_drift_planes}"
    font_title = _pil_font(font_size)
    font_col = _pil_font(max(font_size - 2, 16))   # column header font

    def _text_size(draw: ImageDraw.Draw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
        try:
            bb = draw.textbbox((0, 0), text, font=font)
            return bb[2] - bb[0], bb[3] - bb[1]
        except AttributeError:
            return draw.textsize(text, font=font)

    def _text_draw_top(draw: ImageDraw.Draw, text: str, font: ImageFont.ImageFont) -> int:
        """Return the actual top pixel offset when drawing at y=0 (can be negative)."""
        try:
            bb = draw.textbbox((0, 0), text, font=font)
            return bb[1]  # top offset (<=0 typically)
        except AttributeError:
            return 0

    PAD = max(font_size // 3, 8)  # vertical padding on each side of a text row

    # Measure true rendered heights using a dummy image
    _tmp_img = Image.new("RGB", (10, 10))
    _tmp_d = ImageDraw.Draw(_tmp_img)
    _, title_h = _text_size(_tmp_d, title_line, font_title)
    title_top_off = _text_draw_top(_tmp_d, title_line, font_title)
    # Banner heights: text height + top-offset correction + 2× padding
    title_banner_h = title_h - title_top_off + 2 * PAD

    # Column header: measure the longest possible label
    sample_col_text = f"higher (p10-p99.99)  |  z=000"
    _, col_h = _text_size(_tmp_d, sample_col_text, font_col)
    col_top_off = _text_draw_top(_tmp_d, sample_col_text, font_col)
    col_banner_h = col_h - col_top_off + 2 * PAD

    panel_w = stacks_u8[0].shape[2]  # width of one panel

    # Pre-compute binary masks and colours once
    roi_masks = []
    roi_colors = []
    for _, row in valid_roi_table.iterrows():
        ratio = row.get("intensity_top_bottom_ratio", float("nan"))
        if ratio >= intensity_threshold:
            color = list(roi_color_pass)
        elif ratio < -1:
            color = list(roi_color_negative)
        else:
            color = list(roi_color_fail)
        mask = (row.mask_matrix > 0).astype(np.uint8)
        roi_masks.append(mask)
        roi_colors.append(color)

    frames: list[np.ndarray] = []
    for frame_i in range(len(stacks_u8[0])):
        # z value for this frame
        if padded_indices.size > 0:
            z_global = int(padded_indices[start_ind + frame_i])
        else:
            z_global = start_ind + frame_i
        z_label = f"z={z_global}"

        # Build each image panel
        panels = []
        for stack_idx in range(len(contrast_presets)):
            z_img = stacks_u8[stack_idx][frame_i]
            rgb = np.stack([z_img, z_img, z_img], axis=-1)
            for mask, color in zip(roi_masks, roi_colors):
                for contour in measure.find_contours(mask, 0.5):
                    c = contour.astype(int)
                    for r, col_i in zip(c[:, 0], c[:, 1]):
                        if 0 <= r < rgb.shape[0] and 0 <= col_i < rgb.shape[1]:
                            rgb[r, col_i] = color
            panels.append(rgb)

        # Concatenate panels side by side
        combined = np.concatenate(panels, axis=1)  # (H, W*3, 3)
        img_h, W_total = combined.shape[:2]

        # Full canvas: title banner + column-header banner + image panels
        total_h = title_banner_h + col_banner_h + img_h
        canvas = np.zeros((total_h, W_total, 3), dtype=np.uint8)
        canvas[title_banner_h + col_banner_h:] = combined

        pil_canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_canvas)

        # ── Title banner (centred) ──────────────────────────────────────────
        t_w, _ = _text_size(draw, title_line, font_title)
        title_y = PAD - title_top_off  # compensate for font's top offset
        draw.text(((W_total - t_w) // 2, title_y), title_line,
                  fill=(255, 255, 255), font=font_title)

        # ── Column headers (one per panel) ─────────────────────────────────
        col_y = title_banner_h + PAD - col_top_off  # starts after title banner
        for col_idx, (_, _, preset_label) in enumerate(contrast_presets):
            col_text = f"{preset_label}  |  {z_label}"
            col_x_center = col_idx * panel_w + panel_w // 2
            cw, _ = _text_size(draw, col_text, font_col)
            draw.text((col_x_center - cw // 2, col_y), col_text,
                      fill=(255, 255, 0), font=font_col)

        frames.append(np.asarray(pil_canvas, dtype=np.uint8))

    return frames


def save_qc_gif(
    result: dict[str, Any],
    valid_roi_table: pd.DataFrame,
    save_path: Path | str,
    *,
    fps: int = 3,
    **kwargs: Any,
) -> Path:
    """Save the QC GIF with colour-coded ROI contours on the registered z-stack.

    Parameters
    ----------
    result : dict
        Registration result.
    valid_roi_table : DataFrame
        Output of :func:`get_valid_roi_table`.
    save_path : Path or str
        Output GIF path.
    fps : int
        Frames per second. Default 3.

    Returns
    -------
    Path
        Path to the saved GIF.
    """
    frames = make_qc_gif_frames(result, valid_roi_table, **kwargs)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(save_path), frames, fps=fps, loop=0)
    return save_path


# ---------------------------------------------------------------------------
# Full QC pipeline for a single plane
# ---------------------------------------------------------------------------

def run_qc(
    result: dict[str, Any],
    save_dir: Path | str,
    *,
    gif_fps: int = 3,
    intensity_threshold: float = 0.5,
    zdrift_calc_bin: int = 5,
) -> dict[str, Any]:
    """Run the full QC pipeline for one registration result.

    Computes ROI intensity metrics, saves an overlay PNG, and saves a GIF.

    Parameters
    ----------
    result : dict
        Output of :func:`register_fov_local_zstack_to_fov`.
    save_dir : Path or str
        Directory for QC outputs.
    gif_fps : int
        GIF frames per second. Default 3.
    intensity_threshold : float
        Pass/fail threshold for ``intensity_top_bottom_ratio``. Default 0.5.
    zdrift_calc_bin : int
        Bin size (in minutes) for calculating z-drift min/max. Default 5.

    Returns
    -------
    dict with keys:
        ``valid_roi_table`` (DataFrame),
        ``overlay_image_path`` (Path),
        ``gif_path`` (Path).
    """
    session_key = result.get("session_key", "NA")
    plane_id = result.get("plane_id", "NA")
    save_dir = Path(save_dir)

    valid_roi_table = get_valid_roi_table(result, zdrift_calc_bin=zdrift_calc_bin)
    valid_roi_table["negative_intensity"] = valid_roi_table["intensity_top_bottom_ratio"] < -1
    valid_roi_table["drift_pass"] = valid_roi_table["intensity_top_bottom_ratio"] >= intensity_threshold
    valid_roi_table["binary_mask_matrix"] = valid_roi_table["mask_matrix"].apply(
        lambda x: (x > 0).astype(np.uint8)
    )

    overlay_path = save_dir / f"{session_key}_{plane_id}_registered_mean_zstack_roi_overlay.png"
    save_roi_overlay_image(result, valid_roi_table, overlay_path, zdrift_calc_bin=zdrift_calc_bin)

    gif_path = save_dir / f"{session_key}_{plane_id}_registered_zstack_with_roi_outlines.gif"
    save_qc_gif(result, valid_roi_table, gif_path, fps=gif_fps, intensity_threshold=intensity_threshold, zdrift_calc_bin=zdrift_calc_bin)

    return {
        "valid_roi_table": valid_roi_table,
        "overlay_image_path": overlay_path,
        "gif_path": gif_path,
    }

