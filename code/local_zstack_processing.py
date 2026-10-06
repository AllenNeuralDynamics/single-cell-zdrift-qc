"""Rebuilds local z-stack registration for processed ophys assets that predate
the ``movie_qc`` pipeline step.

Some older processed data assets have no ``*_z_stack_local_reg.h5`` (and therefore
no ``movie_qc/*_z_drift_evaluation.json`` either), which is what
``register_fov_local_zstack.prepare_local_zstack``/``prepare_plane_data`` require.
Comparing such an asset's directory tree against one from a session the
single-cell-zdrift-qc capsule already succeeds on shows the only difference is a
missing ``movie_qc/`` folder and the two files in it -- every other input
(extraction, motion_correction, decrosstalk) is already present and used
successfully elsewhere in this same pipeline.

Both missing files can be rebuilt from data that *is* present, using the exact
lamf_analysis building blocks the production pipeline itself uses:

- ``*_z_stack_local_reg.h5``: the within-plane-registered-and-averaged,
  between-plane-registered local z-stack. Rebuilt from the raw interleaved
  ``*_z_stack_local.h5`` (found via ``zstack_utils.get_local_zstack_filepath``,
  which already knows to fall back from the processed asset to the raw asset)
  via ``zstack.register_local_z_stack``, which de-interleaves repeats-per-slice,
  registers+averages each slice's repeats, then registers between slices.
- ``movie_qc/*_z_drift_evaluation.json`` (``matched_plane_indices``): rebuilt by
  comparing one-minute mean FOVs of the plane's decrosstalked movie against the
  z-stack above -- this is the exact fallback
  ``capsule_data_utils.get_zdrift_matched_plane_indices(..., run_z_drift_estimation_if_not_found=True)``
  implements, inlined here (a) to reuse the z-stack already computed above
  instead of recomputing it, and (b) because that function's own
  ``next(glob(...))`` raises ``StopIteration`` and returns ``None`` *before* ever
  checking the ``run_z_drift_estimation_if_not_found`` flag when the json is
  missing outright (rather than merely missing the key) -- which is exactly our
  case.

This module does not touch ``register_fov_local_zstack.py``. It instead builds a
writable "shadow" copy of the affected plane's session directory -- symlinks to
every existing file, plus a real ``movie_qc/`` with the two rebuilt artifacts --
so the existing pipeline can run against it completely unmodified. The shadow
requires the plane's raw data asset to be attached next to its processed asset
under the same ``/data`` directory (see README "Testing local z-stack processing").
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from lamf_analysis.code_ocean import capsule_data_utils as cdu
import lamf_analysis.utils as lamf_utils
from lamf_analysis.ophys import zdrift as zdrift_mod
from lamf_analysis.ophys import zstack as zstack_mod
from lamf_analysis.ophys import zstack_utils

LOCAL_ZSTACK_PROCESSING_NOTE = (
    "This session predates the movie_qc pipeline step. The local z-stack was "
    "de-interleaved, within-plane registered/averaged and between-plane registered "
    "from the raw '*_z_stack_local.h5' (lamf_analysis.ophys.zstack.register_local_z_stack), "
    "and matched_plane_indices were estimated from one-minute mean FOVs of the "
    "decrosstalked movie against that z-stack (lamf_analysis.ophys.zdrift.get_matched_plane_indices) "
    "-- see single-cell-zdrift-qc, code/local_zstack_processing.py."
)


def needs_local_zstack_processing(plane_path: Path | str) -> bool:
    """True if ``plane_path`` has no ``*_z_stack_local_reg.h5`` anywhere under it."""
    plane_path = Path(plane_path)
    return next(plane_path.rglob("*_z_stack_local_reg.h5"), None) is None


# ---------------------------------------------------------------------------
# Shadow plane directory: symlink farm + the two rebuilt files
# ---------------------------------------------------------------------------

def _symlink_tree(src: Path, dst: Path, skip_names: frozenset[str] = frozenset()) -> None:
    """Shallow-symlink every entry of ``src`` into ``dst`` (idempotent).

    Safe only when nothing will ``rglob``/recursively search *through* one of
    these entries from an ancestor outside it (see ``_mirror_tree`` otherwise).
    """
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.name in skip_names:
            continue
        link = dst / entry.name
        if not link.exists():
            # .resolve(): same reasoning as build_shadow_plane_path's
            # plane_path normalization -- os.symlink's target is interpreted
            # relative to link's directory on dereference, not src's.
            os.symlink(entry.resolve(), link, target_is_directory=entry.is_dir())


def _mirror_tree(src: Path, dst: Path, skip_names: frozenset[str] = frozenset()) -> None:
    """Recursively mirror ``src`` into ``dst``: real directories, symlinked files.

    ``pathlib.Path.rglob`` refuses to descend into a symlinked *directory*
    encountered while recursing from a real ancestor (it only follows a
    symlink when that symlink is the glob's own root) -- so a single
    directory-level symlink for e.g. ``extraction/`` silently breaks every
    downstream ``plane_path.rglob('*_extraction.h5')``-style lookup into it.
    Mirroring with real directories and symlinks only at the leaf (file) level
    avoids that entirely, since symlinked *files* match ``rglob`` patterns
    exactly like real ones.

    If ``dst`` itself is already a whole-directory symlink -- e.g. a second
    plane in the same session being mirrored after
    ``build_shadow_plane_path``'s own ``_symlink_tree`` pass over "every
    *other* plane" already symlinked this one whole, back when it was the
    other plane being skipped -- that symlink is removed first. Otherwise
    ``dst.mkdir(exist_ok=True)`` is a silent no-op (the symlink already
    resolves to a directory), every entry below looks like it "already
    exists" through that symlink, and the caller ends up writing into the
    original read-only asset.
    """
    if dst.is_symlink():
        dst.unlink()
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.name in skip_names:
            continue
        target = dst / entry.name
        if target.exists():
            continue
        if entry.is_dir() and not entry.is_symlink():
            _mirror_tree(entry, target)
        else:
            os.symlink(entry.resolve(), target)


def build_shadow_plane_path(plane_path: Path | str, shadow_root: Path | str) -> Path:
    """Build a writable shadow of ``plane_path``'s session under ``shadow_root``.

    Everything is symlinked from the original (read-only) asset -- including the
    corresponding raw asset, found next to the processed one exactly the way
    ``capsule_data_utils.get_raw_path_from_plane_path`` expects -- except the
    target plane's own directory, which is real (writable) so a ``movie_qc/``
    can be added to it.

    Parameters
    ----------
    plane_path : Path or str
        The original (read-only) processed plane directory, e.g.
        ``/root/capsule/data/multiplane-ophys_<subject>_<date>_processed_.../VISp_0``.
    shadow_root : Path or str
        Writable scratch directory to build the shadow under.

    Returns
    -------
    Path
        The shadow plane path -- a drop-in substitute for ``plane_path``.

    Raises
    ------
    FileNotFoundError
        If the raw asset for this session is not attached next to the processed
        one under the same data directory.
    """
    # Absolute, not resolved: os.symlink writes its target argument verbatim,
    # and a relative one is interpreted relative to the *link's* directory
    # (under shadow_root) when later dereferenced, not relative to this
    # process's cwd -- so a relative plane_path here would silently produce
    # symlinks that point nowhere real. .absolute() (not .resolve()) keeps
    # any intentional symlinks in the original /data mount intact.
    plane_path = Path(plane_path).absolute()
    processed_dir = plane_path.parent
    data_root = processed_dir.parent
    session_name = processed_dir.name.split("_processed")[0]
    raw_dir = data_root / session_name
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Raw asset '{session_name}' not found next to the processed asset under "
            f"{data_root} -- attach it first (see README 'Testing local z-stack processing')."
        )

    shadow_root = Path(shadow_root)
    shadow_data_root = shadow_root / "data"
    shadow_processed_dir = shadow_data_root / processed_dir.name
    shadow_raw_dir = shadow_data_root / session_name
    shadow_plane_path = shadow_processed_dir / plane_path.name

    shadow_data_root.mkdir(parents=True, exist_ok=True)
    if not shadow_raw_dir.exists():
        os.symlink(raw_dir, shadow_raw_dir, target_is_directory=True)

    # Other planes in the session are symlinked whole (nothing rglobs into them
    # from the target plane's path). The target plane itself is mirrored with
    # real directories + leaf-file symlinks (see _mirror_tree) so that
    # plane_path.rglob(...) -- used throughout capsule_data_utils and
    # register_fov_local_zstack -- keeps working, plus gets its own writable
    # movie_qc/ (populated below).
    _symlink_tree(processed_dir, shadow_processed_dir, skip_names=frozenset({plane_path.name}))
    _mirror_tree(plane_path, shadow_plane_path, skip_names=frozenset({"movie_qc"}))
    return shadow_plane_path


# ---------------------------------------------------------------------------
# Rebuilding the two missing artifacts
# ---------------------------------------------------------------------------

def _rebuild_registered_zstack(shadow_plane_path: Path) -> np.ndarray:
    """De-interleave + register the raw local z-stack for this plane.

    Finds ``*_z_stack_local.h5`` (processed path first, falling back to the raw
    asset -- the exact lookup the production pipeline uses) and runs it through
    ``zstack.register_local_z_stack``, which splits the raw ``(n_slices *
    n_repeats, H, W)`` stack by slice, registers+averages each slice's repeats
    (within-plane), then registers the resulting per-slice means to each other
    (between-plane).
    """
    local_zstack_path = zstack_utils.get_local_zstack_filepath(shadow_plane_path)
    registered = zstack_mod.register_local_z_stack(local_zstack_path)
    return np.asarray(registered, dtype=np.float64)


def _one_minute_mean_fovs_streaming(
    movie_path: Path, frame_rate: float, threshold_sec: float = 30
) -> np.ndarray:
    """Chunked equivalent of ``zdrift.get_one_minute_mean_fovs`` that never loads
    the whole movie into memory -- full-session decrosstalked movies here run
    ~100k+ frames at (512, 512) int16, i.e. tens of GB, which does not fit.

    Reproduces that function's divider logic exactly, but reads and averages one
    ~one-minute chunk directly from the HDF5 dataset at a time.
    """
    with h5py.File(movie_path, "r") as f:
        dset = f["data"]
        num_frames, ny, nx = dset.shape

        one_minute_frames = max(1, int(round(frame_rate * 60)))
        last_minute_threshold = frame_rate * threshold_sec
        dividers = np.arange(0, num_frames, one_minute_frames, dtype=int)

        if dividers.size == 0:
            dividers = np.array([0], dtype=int)
        if dividers[-1] != num_frames:
            trailing_frames = num_frames - dividers[-1]
            if trailing_frames < last_minute_threshold and dividers.size > 1:
                dividers[-1] = num_frames
            else:
                dividers = np.append(dividers, num_frames)
        elif dividers.size == 1:
            dividers = np.append(dividers, num_frames)

        emf = np.zeros((len(dividers) - 1, ny, nx), dtype=np.float32)
        for i in range(len(dividers) - 1):
            chunk = dset[dividers[i]:dividers[i + 1], :, :]
            emf[i, :, :] = np.mean(chunk, axis=0)
    return emf


def _rebuild_matched_plane_indices(
    shadow_plane_path: Path, registered_zstack: np.ndarray
) -> np.ndarray:
    """Estimate matched_plane_indices the same way the production fallback does.

    Inlined from ``capsule_data_utils.get_zdrift_matched_plane_indices``'s
    ``run_z_drift_estimation_if_not_found`` branch, reusing the z-stack already
    computed above instead of recomputing it, and streaming the movie from disk
    (see ``_one_minute_mean_fovs_streaming``) instead of loading it whole.
    """
    frame_rate = cdu.get_frame_rate_from_plane_path(shadow_plane_path)
    movie_path = cdu.get_decrosstalked_movie_file(shadow_plane_path)

    one_min_emf = _one_minute_mean_fovs_streaming(movie_path, frame_rate)

    range_y, range_x = lamf_utils.get_motion_correction_crop_xy_range(shadow_plane_path)
    ref_zstack_crop = registered_zstack[:, range_y[0]:range_y[1], range_x[0]:range_x[1]]
    episodic_mean_fovs_crop = one_min_emf[:, range_y[0]:range_y[1], range_x[0]:range_x[1]]

    matched_plane_indices = zdrift_mod.get_matched_plane_indices(
        ref_zstack_crop, episodic_mean_fovs_crop
    )
    return np.asarray(matched_plane_indices).astype(int)


def populate_movie_qc(shadow_plane_path: Path | str) -> dict[str, Any]:
    """Rebuild and write both missing artifacts into ``shadow_plane_path/movie_qc/``.

    After this, ``cdu.get_local_zstack_reg(shadow_plane_path)`` and the
    ``movie_qc/*_z_drift_evaluation.json`` read in
    ``register_fov_local_zstack.prepare_plane_data`` both resolve normally, so
    the rest of the existing pipeline runs unmodified.

    Returns
    -------
    dict with keys ``z_stack_local_reg_h5``, ``z_drift_evaluation_json`` (paths),
    and ``matched_plane_indices`` (the array actually written).
    """
    shadow_plane_path = Path(shadow_plane_path)
    plane_id = shadow_plane_path.name
    movie_qc_dir = shadow_plane_path / "movie_qc"
    movie_qc_dir.mkdir(parents=True, exist_ok=True)

    reg_h5_path = movie_qc_dir / f"{plane_id}_z_stack_local_reg.h5"
    evaluation_json_path = movie_qc_dir / f"{plane_id}_z_drift_evaluation.json"
    tmp_reg_h5_path = reg_h5_path.with_suffix(reg_h5_path.suffix + ".tmp")
    tmp_evaluation_json_path = evaluation_json_path.with_suffix(evaluation_json_path.suffix + ".tmp")

    # Both artifacts are written under temp names and only published (renamed
    # to their final names) once everything has succeeded, h5 last. Without
    # this, a failure between the two writes (e.g. matched_plane_indices
    # estimation raising) leaves a complete *_z_stack_local_reg.h5 on disk
    # with no evaluation json -- needs_local_zstack_processing only checks
    # for the h5, so a later retry against the same shadow would see it and
    # skip rebuilding entirely, permanently "poisoning" this shadow plane
    # path with an incomplete movie_qc/ that later fails inside
    # prepare_plane_data instead of here. Publishing the h5 last means its
    # final name never exists unless the json already does too, so that
    # check stays a valid completion signal either way.
    try:
        registered_zstack = _rebuild_registered_zstack(shadow_plane_path)
        with h5py.File(tmp_reg_h5_path, "w") as f:
            f.create_dataset(
                "data", data=registered_zstack, compression="gzip", compression_opts=4
            )

        matched_plane_indices = _rebuild_matched_plane_indices(shadow_plane_path, registered_zstack)
        if matched_plane_indices.size == 0:
            raise RuntimeError(f"Estimated matched_plane_indices is empty for {plane_id}")

        evaluation_data = {
            "matched_plane_indices": matched_plane_indices.tolist(),
            "local_zstack_processing_generated": True,
            "processing_note": LOCAL_ZSTACK_PROCESSING_NOTE,
        }
        with open(tmp_evaluation_json_path, "w") as f:
            json.dump(evaluation_data, f, indent=2)

        os.replace(tmp_evaluation_json_path, evaluation_json_path)
        os.replace(tmp_reg_h5_path, reg_h5_path)
    finally:
        tmp_reg_h5_path.unlink(missing_ok=True)
        tmp_evaluation_json_path.unlink(missing_ok=True)

    return {
        "z_stack_local_reg_h5": reg_h5_path,
        "z_drift_evaluation_json": evaluation_json_path,
        "matched_plane_indices": matched_plane_indices,
    }


# ---------------------------------------------------------------------------
# One-call entry point
# ---------------------------------------------------------------------------

def ensure_plane_path(plane_path: Path | str, shadow_root: Path | str) -> Path:
    """Return a plane_path that's safe to hand to ``register_fov_local_zstack``.

    If ``plane_path`` already has a local z-stack registration file, it's
    returned unchanged (no shadow is built). Otherwise a shadow is built and
    populated, and the shadow path is returned.
    """
    plane_path = Path(plane_path)
    if not needs_local_zstack_processing(plane_path):
        return plane_path

    shadow_plane_path = build_shadow_plane_path(plane_path, shadow_root)
    if needs_local_zstack_processing(shadow_plane_path):
        populate_movie_qc(shadow_plane_path)
    return shadow_plane_path
