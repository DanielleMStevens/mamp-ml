"""Tests for :mod:`mamp_ml.lrr_features` (checkpoint 4e).

Two layers of coverage:

1. Unit tests for the small helpers (filter construction, normalisation,
   zero-crossing detection) plus a synthetic per-protein round-trip.

2. A golden DataFrame-equivalence test for the full
   :func:`compute_bfactor_lrr_segments` run, comparing against the
   committed ``tests/fixtures/golden/bfactor_winding_lrr_segments.csv``.
   The comparison uses ``pd.testing.assert_frame_equal`` with
   ``rtol=1e-10`` — strict enough to catch any algorithmic divergence but
   lax enough to tolerate the last-digit float-roundoff in the
   ``Winding Number`` column that comes from ``np.random.rand`` in the
   shared ``compute_winding`` (the surrounding gradient descent still
   converges to the same minimum, so breakpoints + filtered B-factors +
   repeat numbers all stay byte-stable).

Tests that need real PDBs use the ``colabfold_outputs_dir`` fixture from
``conftest.py``, which skips cleanly when the fold artefacts are absent.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mamp_ml.lrr_features import (
    _butterworth_bandpass_sos,
    _compute_zero_crossing_repeats,
    _normalize_filtered_bfactor,
    compute_bfactor_lrr_segments,
    write_bfactor_lrr_segments,
)


# ---------------------------------------------------------------------------
# _butterworth_bandpass_sos
# ---------------------------------------------------------------------------


def test_butterworth_sos_default_shape() -> None:
    """Default tuning produces a usable SOS filter matrix."""
    sos = _butterworth_bandpass_sos(period=25, filter_order=10)
    # SOS sections are 6-column rows; bandpass produces order // 2 * 2 sections.
    assert sos.shape[1] == 6
    assert sos.shape[0] >= 1


def test_butterworth_sos_rejects_zero_period() -> None:
    """A zero period makes the cutoff infinite — caller's bug."""
    with pytest.raises(ZeroDivisionError):
        _butterworth_bandpass_sos(period=0, filter_order=10)


def test_butterworth_sos_clips_high_to_nyquist() -> None:
    """Very short period would push highcut past Nyquist; we clip instead."""
    # period=2 gives high=1.0 which exceeds Nyquist (0.5). The helper must
    # clip silently.
    sos = _butterworth_bandpass_sos(period=2, filter_order=4)
    assert sos.shape[1] == 6


# ---------------------------------------------------------------------------
# _normalize_filtered_bfactor
# ---------------------------------------------------------------------------


def test_normalize_scales_to_unit_max_abs() -> None:
    out = _normalize_filtered_bfactor(np.array([1.0, -2.0, 0.5]))
    assert np.allclose(out, [0.5, -1.0, 0.25])


def test_normalize_flat_signal_returns_zeros() -> None:
    out = _normalize_filtered_bfactor(np.zeros(5))
    assert np.array_equal(out, np.zeros(5))


def test_normalize_does_not_mutate_input() -> None:
    arr = np.array([2.0, -1.0])
    _ = _normalize_filtered_bfactor(arr)
    # Input untouched.
    assert np.array_equal(arr, np.array([2.0, -1.0]))


# ---------------------------------------------------------------------------
# _compute_zero_crossing_repeats
# ---------------------------------------------------------------------------


def test_zero_crossings_no_initial_positive() -> None:
    """No initial positive lobe -> repeat counter stays at 0 until first
    rising zero crossing at idx=2."""
    bff = np.array([-1.0, -0.5, 0.3, 0.2, -0.1])
    reps = _compute_zero_crossing_repeats(bff)
    assert reps.tolist() == [0, 0, 1, 1, 1]


def test_zero_crossings_initial_positive() -> None:
    """When bff[0] > 0 the seeded repeat counter is 1, but position 0 in
    the output is still 0 by convention (the legacy never writes
    ``repeat_numbers[0]`` explicitly — it stays at the np.zeros init)."""
    bff = np.array([0.5, 0.4, -0.2, 0.1])
    reps = _compute_zero_crossing_repeats(bff)
    # idx 0: untouched -> 0
    # idx 1, 2: still in repeat 1 (counter seeded at 1)
    # idx 3: negative-to-positive crossing -> repeat 2
    assert reps.tolist() == [0, 1, 1, 2]


def test_zero_crossings_empty_array() -> None:
    reps = _compute_zero_crossing_repeats(np.array([], dtype=float))
    assert reps.shape == (0,)


# ---------------------------------------------------------------------------
# compute_bfactor_lrr_segments — missing inputs
# ---------------------------------------------------------------------------


def test_compute_bfactor_segments_missing_pdb_dir(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    # We don't even need a breakpoints.pickle — the pdb_dir check fires first.
    with pytest.raises(FileNotFoundError, match="PDB directory"):
        compute_bfactor_lrr_segments(tmp_path / "nope_pdbs", cache)


def test_compute_bfactor_segments_missing_cache_dir(tmp_path: Path) -> None:
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Cache directory"):
        compute_bfactor_lrr_segments(pdb_dir, tmp_path / "nope_cache")


def test_compute_bfactor_segments_missing_breakpoints_pickle(tmp_path: Path) -> None:
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(FileNotFoundError, match="breakpoints.pickle"):
        compute_bfactor_lrr_segments(pdb_dir, cache)


# ---------------------------------------------------------------------------
# Golden equivalence: compute_bfactor_lrr_segments against legacy output
# ---------------------------------------------------------------------------


def test_compute_bfactor_lrr_segments_matches_golden(
    tmp_path: Path,
    repo_root: Path,
    colabfold_outputs_dir: Path,
) -> None:
    """The new implementation must produce a DataFrame numerically equal to
    the committed golden bfactor_winding_lrr_segments.csv.

    Important
    ---------
    The golden was generated by running the legacy
    ``analyze_bfactor_peaks.py`` against the **production breakpoints
    cache** at ``src/mamp_ml/lrr_annotation/cache/`` (the same cache the
    real pipeline always reads from, and the same cache the model's
    ``BFactorWeightGenerator`` indirectly depends on through breakpoint
    boundaries). The production cache carries Solanum breakpoints
    ``[112, 636]`` whereas a *fresh* regression on our locally folded PDB
    would give ``[112, 646]``. To make the equivalence check meaningful we
    point the new code at the production cache here.

    Comparison uses ``pd.testing.assert_frame_equal`` with ``rtol=1e-10``
    — strict enough to catch any algorithmic drift but lax enough to
    tolerate the ~15-significant-digit float roundoff in the
    ``Winding Number`` column caused by ``np.random.rand`` in the shared
    ``compute_winding`` initialiser. ``Filtered B-Factor`` and
    ``LRR Repeat Number`` are byte-identical across runs.
    """
    production_cache = repo_root / "src" / "mamp_ml" / "lrr_annotation" / "cache"
    assert (production_cache / "breakpoints.pickle").is_file(), (
        "production cache pickle missing"
    )

    # Stage the PDB(s) the legacy bfactor stage would have seen.
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    src_pdbs = list(
        colabfold_outputs_dir.glob(
            "*_unrelaxed_rank_001_alphafold2_ptm_model_*_seed_*.pdb"
        )
    )
    assert src_pdbs, "no ColabFold PDB files in fixture dir"
    for src in src_pdbs:
        receptor = src.name.split("_unrelaxed_rank_")[0]
        (pdb_dir / f"{receptor}.pdb").write_bytes(src.read_bytes())

    new_df = compute_bfactor_lrr_segments(pdb_dir, production_cache)
    assert len(new_df) > 0

    golden_csv = (
        repo_root / "tests" / "fixtures" / "golden" / "bfactor_winding_lrr_segments.csv"
    )
    golden_df = pd.read_csv(golden_csv)

    assert list(new_df.columns) == list(golden_df.columns)
    assert len(new_df) == len(golden_df)

    pd.testing.assert_frame_equal(
        new_df.reset_index(drop=True),
        golden_df.reset_index(drop=True),
        check_dtype=False,    # int64 vs int may differ across pandas versions
        check_like=False,
        rtol=1e-10,
        atol=1e-12,
    )


def test_write_bfactor_lrr_segments_round_trips(
    tmp_path: Path,
    repo_root: Path,
    colabfold_outputs_dir: Path,
) -> None:
    """The on-disk CSV must parse back to the same DataFrame the writer returned."""
    production_cache = repo_root / "src" / "mamp_ml" / "lrr_annotation" / "cache"
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    for src in colabfold_outputs_dir.glob(
        "*_unrelaxed_rank_001_alphafold2_ptm_model_*_seed_*.pdb"
    ):
        receptor = src.name.split("_unrelaxed_rank_")[0]
        (pdb_dir / f"{receptor}.pdb").write_bytes(src.read_bytes())

    out_csv = tmp_path / "out.csv"
    df_returned = write_bfactor_lrr_segments(pdb_dir, production_cache, out_csv)
    df_reread = pd.read_csv(out_csv)
    pd.testing.assert_frame_equal(
        df_returned.reset_index(drop=True),
        df_reread.reset_index(drop=True),
        check_dtype=False,
    )


def test_compute_bfactor_segments_skips_receptors_without_breakpoints(
    tmp_path: Path,
    repo_root: Path,
    colabfold_outputs_dir: Path,
) -> None:
    """Receptors with PDB files but no cached breakpoints are silently
    skipped (the legacy behaviour, implemented here via set intersection
    in compute_bfactor_lrr_segments)."""
    production_cache = repo_root / "src" / "mamp_ml" / "lrr_annotation" / "cache"
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    for src in colabfold_outputs_dir.glob(
        "*_unrelaxed_rank_001_alphafold2_ptm_model_*_seed_*.pdb"
    ):
        receptor = src.name.split("_unrelaxed_rank_")[0]
        (pdb_dir / f"{receptor}.pdb").write_bytes(src.read_bytes())

    # Add a fake PDB that has no breakpoints entry — must NOT crash the
    # batch; the real receptor still produces rows.
    (pdb_dir / "ghost_receptor.pdb").write_text("HEADER stub\nEND\n")

    df = compute_bfactor_lrr_segments(pdb_dir, production_cache)
    assert (df["Protein Key"] != "ghost_receptor").all()
    assert len(df) > 0
