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

def _symlink_if_absent(target: Path, link: Path, *, target_is_directory: bool = False) -> None:
    """``os.symlink(target, link)`` unless ``link`` already exists.

    The existence check and the symlink call are two separate syscalls, so a
    second plane from the same session being processed concurrently (the
    sequential path never does this today, but nothing stops a future
    parallel one) can slip in between them and create ``link`` first --
    ``os.symlink`` has no atomic "create if absent" mode, only an
    after-the-fact ``FileExistsError``, which is what's caught here. Safe to
    ignore: by the time it's raised, the link exists either way, which is all
    every caller here checks for.
    """
    if link.is_symlink() and not link.exists():
        link.unlink(missing_ok=True)
    if link.exists():
        return
    try:
        os.symlink(target, link, target_is_directory=target_is_directory)
    except FileExistsError:
        pass


def _symlink_tree(src: Path, dst: Path, skip_names: frozenset[str] = frozenset()) -> None:
    """Shallow-symlink every entry of ``src`` into ``dst`` (idempotent).

    Safe only when nothing will ``rglob``/recursively search *through* one of
    these entries from an ancestor outside it (see ``_mirror_tree`` otherwise).
    """
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.name in skip_names:
            continue
        # .resolve(): same reasoning as build_shadow_plane_path's plane_path
        # normalization -- os.symlink's target is interpreted relative to
        # link's directory on dereference, not src's.
        _symlink_if_absent(entry.resolve(), dst / entry.name, target_is_directory=entry.is_dir())


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

    Idempotent and retry-safe: a leaf file/symlink that already exists is
    left alone (a single ``os.symlink`` call is atomic, so an existing one is
    never partial), but an existing *directory* is always recursed into
    rather than short-circuited -- a directory can easily be left
    half-populated by an interrupted previous run (e.g. ``extraction/``
    created but only some of its files linked in before a crash), and
    skipping it on sight would make that partial state permanent across
    every subsequent retry.
    """
    if dst.is_symlink():
        dst.unlink()
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.name in skip_names:
            continue
        target = dst / entry.name
        if entry.is_dir() and not entry.is_symlink():
            _mirror_tree(entry, target)
        else:
            _symlink_if_absent(entry.resolve(), target)


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
    _symlink_if_absent(raw_dir, shadow_raw_dir, target_is_directory=True)

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

# Keyed by raw tif path -> (registered_list_or_array, channels_saved), so that
# two planes sharing one multi-channel raw tif (see _rebuild_registered_zstack)
# only pay for its registration once. Never evicted, but bounded in practice:
# one capsule run processes exactly one session's planes, so this holds at
# most one entry's worth of data (~2 channels) per process lifetime, not one
# per session ever processed.
_TIF_ZSTACK_CACHE: dict[str, tuple] = {}


def _rebuild_registered_zstack(shadow_plane_path: Path) -> np.ndarray:
    """De-interleave + register the raw local z-stack for this plane.

    Finds ``*_z_stack_local.h5`` (processed path first, falling back to the raw
    asset -- the exact lookup the production pipeline uses) via
    ``zstack_utils.get_local_zstack_filepath``, which can return either of two
    formats depending on rig/pipeline vintage:

    - ``.h5``: one file per plane, raw frames only (mesoscope-style). Handled by
      ``zstack.register_local_z_stack``, which splits the raw ``(n_slices *
      n_repeats, H, W)`` stack by slice, registers+averages each slice's
      repeats (within-plane), then registers the resulting per-slice means to
      each other (between-plane).
    - ``.tif``/``.tiff``: one file per *session* (seen on at least one
      non-mesoscope rig in this cohort), with this plane and its
      simultaneously-acquired sibling multiplexed as separate ScanImage
      "channels" rather than as separate files. Handled by
      ``zstack.register_local_zstack_from_raw_tif``, which does the same
      per-slice within/between-plane registration but returns one stack per
      channel (a bare array if there's only one channel, else a list). The
      plane to pick is ``channels_saved[i]`` where ``i`` is this plane's
      position among the session's imaging planes -- which is exactly the
      plane id's own trailing index (``VISp_0`` -> 0, ``VISp_1`` -> 1, ...):
      that index *is* ``fov['index']`` from ``session.json``'s ``ophys_fovs``
      (see ``capsule_data_utils.get_intended_depth``'s
      ``plane_names_in_metadata``), and ``ophys_fovs`` and ``channels_saved``
      are both ordered by acquisition/channel order, so the two indices
      coincide. Confirmed against session.json depths + channel count for the
      one session in this cohort that hit this path (759732_2024-12-12:
      fov 0 = 175 um = VISp_0, fov 1 = 275 um = VISp_1, channels_saved=[1, 2]).

    In the ``.tif`` case, the registration itself (the expensive part --
    within-plane repeat-registration for every slice of every channel) runs
    once for *all* of this session's planes at once, since they share one
    file; a session-scoped cache (``_TIF_ZSTACK_CACHE``) means the second
    plane's call reuses that result instead of silently repeating the whole
    thing just to throw away the other channel again.
    """
    local_zstack_path = zstack_utils.get_local_zstack_filepath(shadow_plane_path)
    suffix = local_zstack_path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        cache_key = str(local_zstack_path)
        cached = _TIF_ZSTACK_CACHE.get(cache_key)
        if cached is not None:
            registered, channels_saved = cached
        else:
            registered, channels_saved = zstack_mod.register_local_zstack_from_raw_tif(local_zstack_path)
            _TIF_ZSTACK_CACHE[cache_key] = (registered, channels_saved)
        if isinstance(registered, list):
            plane_index = int(shadow_plane_path.name.rsplit("_", 1)[-1])
            if plane_index >= len(registered):
                raise ValueError(
                    f"Raw z-stack tif at {local_zstack_path} has {len(registered)} channel(s) "
                    f"({channels_saved}), but plane {shadow_plane_path.name} needs index {plane_index}"
                )
            registered = registered[plane_index]
    else:
        registered = zstack_mod.register_local_z_stack(local_zstack_path)
    return np.asarray(registered, dtype=np.float64)


def _minute_dividers(num_frames: int, frame_rate: float, threshold_sec: float = 30) -> np.ndarray:
    """Pure copy of ``zdrift.get_one_minute_mean_fovs``'s divider logic.

    Split out so it can run on a bare frame count -- no array, no disk access
    -- both for the streaming chunker below and for ``_check_divider_parity``.
    """
    if frame_rate <= 0:
        raise ValueError(f"frame_rate must be > 0, got {frame_rate}")

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
    return dividers


_DIVIDER_PARITY_CHECKED = False


def _check_divider_parity() -> None:
    """Confirm ``_minute_dividers`` still matches the installed
    ``lamf_analysis.ophys.zdrift.get_one_minute_mean_fovs`` it was copied from.

    ``environment/postInstall`` clones ``lamf-analysis`` from its default
    branch with no pinned ref, so the installed copy can change under a
    capsule rebuild with no signal here otherwise -- this module would then
    silently chunk the movie differently than the function it claims to be
    equivalent to, producing different (wrong) ``matched_plane_indices`` with
    no error. Checked once per process, against tiny (1-pixel) synthetic
    movies spanning the edge cases in the divider logic above, comparing only
    the resulting *chunk count* -- cheap enough to run every time rather than
    trust a one-off manual check to stay valid.
    """
    global _DIVIDER_PARITY_CHECKED
    if _DIVIDER_PARITY_CHECKED:
        return

    frame_rate = 10.0
    one_minute_frames = int(round(frame_rate * 60))
    cases = [
        one_minute_frames,  # exact multiple
        one_minute_frames * 3,
        one_minute_frames * 3 + 1,  # tiny trailing remainder (below threshold)
        one_minute_frames * 3 + int(frame_rate * 30) + 5,  # trailing remainder above threshold
        1,  # degenerate
    ]
    for num_frames in cases:
        expected = len(zdrift_mod.get_one_minute_mean_fovs(
            np.zeros((num_frames, 1, 1), dtype=np.float32), frame_rate
        ))
        actual = len(_minute_dividers(num_frames, frame_rate)) - 1
        if actual != expected:
            raise RuntimeError(
                "local_zstack_processing._minute_dividers has drifted from the installed "
                "lamf_analysis.ophys.zdrift.get_one_minute_mean_fovs "
                f"(num_frames={num_frames}, frame_rate={frame_rate}: "
                f"got {actual} chunks, upstream gives {expected}). "
                "lamf-analysis is cloned unpinned in environment/postInstall, so this can "
                "happen after any rebuild -- re-sync _minute_dividers with the current "
                "upstream implementation before trusting matched_plane_indices from this module."
            )
    _DIVIDER_PARITY_CHECKED = True


def _one_minute_mean_fovs_streaming(
    movie_path: Path, frame_rate: float, threshold_sec: float = 30
) -> np.ndarray:
    """Chunked equivalent of ``zdrift.get_one_minute_mean_fovs`` that never loads
    the whole movie into memory -- full-session decrosstalked movies here run
    ~100k+ frames at (512, 512) int16, i.e. tens of GB, which does not fit.

    Reproduces that function's divider logic exactly (see ``_minute_dividers``,
    including the ``frame_rate > 0`` validation -- with a zero/invalid frame
    rate, ``one_minute_frames`` would silently floor to 1, making ``emf``
    roughly ``num_frames`` long instead of ~60x shorter: a multi-hundred-GB
    allocation attempt instead of a clear error), but reads and averages one
    ~one-minute chunk directly from the HDF5 dataset at a time. Verifies that
    reproduction is still accurate (``_check_divider_parity``) before relying
    on it.
    """
    if frame_rate <= 0:
        # Fail before opening movie_path, not after: _minute_dividers below
        # checks this too, but only once num_frames is already in hand, which
        # needs the (potentially huge, slow-to-open-over-S3) file open first.
        raise ValueError(f"frame_rate must be > 0, got {frame_rate}")
    _check_divider_parity()

    with h5py.File(movie_path, "r") as f:
        dset = f["data"]
        num_frames, ny, nx = dset.shape
        dividers = _minute_dividers(num_frames, frame_rate, threshold_sec)

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
    # PID-suffixed: two processes racing on the same plane (not possible via
    # today's sequential-only wiring, but this module makes no such
    # assumption otherwise) would share a fixed temp name and interleave
    # writes into it; each gets its own here instead.
    tmp_tag = f".{os.getpid()}.tmp"
    tmp_reg_h5_path = reg_h5_path.with_suffix(reg_h5_path.suffix + tmp_tag)
    tmp_evaluation_json_path = evaluation_json_path.with_suffix(evaluation_json_path.suffix + tmp_tag)

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
# Provenance: letting callers (and published results) tell generated apart
# from genuine on-rig movie_qc
# ---------------------------------------------------------------------------

def read_provenance(plane_path: Path | str) -> dict[str, Any] | None:
    """Return this module's provenance record for ``plane_path``, or ``None``.

    ``register_fov_local_zstack.prepare_plane_data`` only ever extracts
    ``matched_plane_indices`` out of ``movie_qc/*_z_drift_evaluation.json`` --
    the ``local_zstack_processing_generated``/``processing_note`` fields
    ``populate_movie_qc`` writes alongside it are otherwise never looked at
    again, including by the final published result/processing metadata. This
    is how a caller (see ``run_capsule.py``) recovers that record itself, to
    fold it into the output it does control, so a generated estimate stays
    distinguishable from genuine on-rig ``movie_qc`` after publication rather
    than silently looking identical to it.
    """
    plane_path = Path(plane_path)
    evaluation_path = next(plane_path.rglob("*_z_drift_evaluation.json"), None)
    if evaluation_path is None:
        return None
    with open(evaluation_path) as f:
        evaluation_data = json.load(f)
    if not evaluation_data.get("local_zstack_processing_generated"):
        return None
    return {
        "local_zstack_processing_generated": True,
        "processing_note": evaluation_data.get("processing_note"),
        "source_evaluation_json": str(evaluation_path),
    }


# ---------------------------------------------------------------------------
# A second, unrelated compatibility fix: some plane's extraction file is
# unprefixed. Lives here (not in register_fov_local_zstack.py / capsule_data_
# utils.py) purely because this module already has the shadow/symlink
# machinery to apply it without touching read-only originals -- it has
# nothing to do with movie_qc or z-stacks.
# ---------------------------------------------------------------------------

def _ensure_prefixed_extraction_alias(plane_path: Path) -> None:
    """Add a ``<plane_id>_extraction.h5`` alias if only the bare name exists.

    ``capsule_data_utils.load_projection_image`` looks for
    ``*_extraction.h5`` (``rglob``, underscore required, not optional) --
    one session in this cohort has its extraction file saved as plain
    ``extraction.h5`` instead, a one-off naming quirk from whatever
    processing run produced that specific asset. Confirmed narrow: checked
    every plane in this cohort -- only that one session's two planes are
    affected, and no other file in their tree has the same problem (e.g.
    ``dff.h5`` is matched by a looser ``*dff.h5`` glob elsewhere, so an
    unprefixed name there is already fine).

    Only acts on a writable ``plane_path`` (in practice: a shadow already
    built by ``ensure_plane_path`` for the z-stack fix) -- it does not build
    one itself, so this does NOT help a plane whose extraction file is
    unprefixed but whose ``movie_qc`` is otherwise fine (no such plane
    exists in this cohort; worth knowing if this module is ever reused on a
    different one).
    """
    extraction_dir = plane_path / "extraction"
    if not extraction_dir.is_dir():
        return
    plane_id = plane_path.name
    if list(extraction_dir.glob(f"{plane_id}_extraction.h5")):
        return  # already prefixed; nothing to do
    bare = extraction_dir / "extraction.h5"
    if not bare.exists():
        return  # no bare file either -- not this quirk, leave it alone
    try:
        _symlink_if_absent(bare.resolve(), extraction_dir / f"{plane_id}_extraction.h5")
    except OSError:
        pass  # plane_path isn't writable (not a shadow) -- can't help here


# ---------------------------------------------------------------------------
# One-call entry point
# ---------------------------------------------------------------------------

def ensure_plane_path(plane_path: Path | str, shadow_root: Path | str) -> Path:
    """Return a plane_path that's safe to hand to ``register_fov_local_zstack``.

    If ``plane_path`` already has a local z-stack registration file, it's
    returned unchanged (no shadow is built). Otherwise a shadow is built and
    populated (plus the unrelated extraction-naming fix, see
    ``_ensure_prefixed_extraction_alias``), and the shadow path is returned.
    """
    plane_path = Path(plane_path)
    if not needs_local_zstack_processing(plane_path):
        return plane_path

    shadow_plane_path = build_shadow_plane_path(plane_path, shadow_root)
    if needs_local_zstack_processing(shadow_plane_path):
        populate_movie_qc(shadow_plane_path)
    _ensure_prefixed_extraction_alias(shadow_plane_path)
    return shadow_plane_path
