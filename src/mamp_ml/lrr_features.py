"""mamp_ml.lrr_features — B-factor bandpass analysis stage.

Fresh Python replacement for
``src/mamp_ml/lrr_annotation/analyze_bfactor_peaks.py``.

For every PDB structure produced by :mod:`mamp_ml.structure`, this module:

1. Reads the CA coordinates AND per-residue B-factors in one PDB pass
   (the legacy code parsed each PDB twice — once for structure, once for
   B-factors — which doubled I/O cost on large batches).
2. Computes the cumulative winding number via the shared
   :func:`mamp_ml.lrr_annotation.geom_lrr.analyzer.compute_winding`
   algorithm.
3. Bandpass-filters the B-factor signal between the cached LRR start and
   end positions using a Butterworth filter tuned to the expected LRR
   repeat period (Butterworth order 10, lowcut = 0.5/period,
   highcut = 2.0/period).
4. Locates negative-to-positive zero crossings of the filtered signal —
   these are the boundaries between consecutive LRR repeats.
5. Emits one row per residue in the LRR region with columns
   ``Protein Key``, ``Residue Index``, ``Filtered B-Factor``,
   ``Winding Number``, and ``LRR Repeat Number`` — the schema the
   downstream :class:`mamp_ml.models.esm_positon_weighted.BFactorWeightGenerator`
   reads.

This module reuses ``mamp_ml.lrr_annotation.geom_lrr.compute_winding`` (the
shared numerical-algorithm library, per interpretation (i)) but contains no
orchestration code in common with the legacy
``analyze_bfactor_peaks.py`` — the PDB I/O, breakpoint loading, filter
construction, zero-crossing detection and result assembly are all
implemented fresh here.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

__all__ = [
    "DEFAULT_PERIOD",
    "DEFAULT_FILTER_ORDER",
    "compute_bfactor_lrr_segments",
    "write_bfactor_lrr_segments",
]

# Filter tuning defaults inherited from the legacy script. The period is the
# approximate LRR-repeat length in residues; the bandpass lowcut/highcut are
# 0.5/period and 2/period respectively, so a period of 25 isolates spatial
# frequencies between 1/50 and 1/12.5 cycles per residue.
DEFAULT_PERIOD: int = 25
DEFAULT_FILTER_ORDER: int = 10

# Column order written to disk; the model dataset reads these names exactly.
_CSV_COLUMN_ORDER: List[str] = [
    "Protein Key",
    "Residue Index",
    "Filtered B-Factor",
    "Winding Number",
    "LRR Repeat Number",
]


# =============================================================================
# Section 1 :: PDB I/O (CA coords + B-factors in a single parse)
# =============================================================================


def _read_pdb_ca_data(pdb_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read Cα coordinates and Cα B-factors from a single-chain PDB file.

    The legacy ``analyze_bfactor_peaks.py`` parsed each PDB twice (once via
    ``Loader.load_batch`` for structure, once via ``Loader.load_single`` for
    B-factors); we do both in a single Biopython pass. The B-factor column
    holds AlphaFold's per-residue pLDDT confidence score.

    Parameters
    ----------
    pdb_path
        Path to a PDB file containing a single protein chain.

    Returns
    -------
    coords : ndarray, shape (n_residues, 3)
        Cα coordinates in Å.
    bfactors : ndarray, shape (n_residues,)
        Cα B-factors (pLDDT for ColabFold/AlphaFold outputs).
    """
    # Local import keeps this module's import cost cheap when callers only
    # need the public functions (Biopython is heavy on first import).
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    chain = next(structure.get_chains())
    residues = list(chain.get_residues())
    coords = np.array(
        [np.array(list(r["CA"].get_vector())) for r in residues],
        dtype=float,
    )
    bfactors = np.array([r["CA"].get_bfactor() for r in residues], dtype=float)
    return coords, bfactors


def _load_cached_breakpoints(cache_dir: Path) -> Dict[str, np.ndarray]:
    """Load the regression breakpoints pickle produced by the structure stage.

    The structure-stage cache directory contains ``breakpoints.pickle`` (and
    a sibling ``slopes.pickle`` we don't need here). This helper reads the
    breakpoints dict directly — we don't need an :class:`Analyzer` instance
    for the bfactor stage, just the breakpoint arrays.

    Parameters
    ----------
    cache_dir
        Directory containing ``breakpoints.pickle``.

    Returns
    -------
    dict
        ``{protein_key: breakpoints_array}``.

    Raises
    ------
    FileNotFoundError
        If ``breakpoints.pickle`` is not present in ``cache_dir``.
    """
    pickle_path = cache_dir / "breakpoints.pickle"
    if not pickle_path.is_file():
        raise FileNotFoundError(
            f"Cached breakpoints pickle not found: {pickle_path}"
        )
    with open(pickle_path, "rb") as fh:
        return pickle.load(fh)


# =============================================================================
# Section 2 :: Butterworth bandpass filter + zero-crossing logic
# =============================================================================


def _butterworth_bandpass_sos(period: int, filter_order: int):
    """Build the Butterworth bandpass SOS filter matrix tuned to LRR period.

    The cutoff frequencies are inherited from the legacy script:
    ``low = 0.5 / period``, ``high = 2.0 / period``. If ``high`` would
    exceed the Nyquist frequency (0.5) it is clipped just below.

    Parameters
    ----------
    period
        Approximate LRR repeat length in residues (typically 25 for plant
        LRR-receptor kinases).
    filter_order
        Order of the Butterworth filter (default 10).

    Returns
    -------
    ndarray
        Filter coefficients in second-order-sections form, suitable for
        :func:`scipy.signal.sosfiltfilt`.

    Raises
    ------
    ValueError
        If the cutoff frequencies are mathematically invalid for the
        requested period.
    """
    from scipy import signal

    nyquist = 0.5
    low = 0.5 / period
    high = 2.0 / period
    if high >= nyquist:
        high = nyquist * 0.99
    if low <= 0:
        raise ValueError(f"low cutoff must be positive; got {low}")
    if low >= high:
        raise ValueError(f"low cutoff {low} >= high cutoff {high}")
    return signal.butter(
        filter_order, [low, high], btype="bandpass", output="sos", fs=1.0
    )


def _normalize_filtered_bfactor(bff: np.ndarray) -> np.ndarray:
    """Divide the filtered signal by its peak absolute value (range [-1, 1]).

    Returns an all-zero array when the signal is numerically flat (peak
    magnitude < 1e-9). The caller can then read every residue's filtered
    B-factor as ``0`` without dividing by zero.
    """
    out = np.asarray(bff, dtype=float).copy()
    max_abs = float(np.max(np.abs(out))) if out.size else 0.0
    if max_abs > 1e-9:
        out /= max_abs
    else:
        out[:] = 0.0
    return out


def _compute_zero_crossing_repeats(bff: np.ndarray) -> np.ndarray:
    """Assign each residue a cumulative LRR-repeat index.

    A new repeat begins at every negative-to-positive zero crossing of the
    filtered signal. The first residue's repeat index is 1 if
    ``bff[0] > 0`` and 0 otherwise — i.e. residues before the first
    positive lobe are tagged as "pre-repeat-1".

    Parameters
    ----------
    bff
        Filtered B-factor signal, length ``n_segment_residues``.

    Returns
    -------
    ndarray of int, shape (n_segment_residues,)
        Per-residue repeat index in the order ``bff`` was given.
    """
    n = len(bff)
    repeat_numbers = np.zeros(n, dtype=int)
    # Match the legacy convention exactly: repeat_numbers[0] is ALWAYS 0
    # (the loop below begins at index 1). The current_repeat counter is
    # nevertheless seeded to 1 when bff[0] > 0 so that subsequent residues
    # before the first negative-to-positive crossing land in repeat 1.
    current_repeat = 1 if (n > 0 and bff[0] > 0) else 0
    for i in range(1, n):
        # A "negative-to-positive" zero crossing increments the repeat count.
        if bff[i - 1] <= 0 and bff[i] > 0:
            current_repeat += 1
        repeat_numbers[i] = current_repeat
    return repeat_numbers


# =============================================================================
# Section 3 :: Per-protein analysis
# =============================================================================


def _analyze_single_protein(
    key: str,
    structure: np.ndarray,
    bfactor: np.ndarray,
    breakpoints,
    *,
    period: int,
    filter_order: int,
) -> Optional[pd.DataFrame]:
    """Run the full bfactor analysis for one protein; return one DataFrame.

    Returns ``None`` when the protein is skipped (length mismatch,
    insufficient breakpoints, segment too short to filter, etc. — these
    correspond to the various ``continue`` branches in the legacy script).
    A warning is raised in every skip case so the caller can audit.

    Parameters
    ----------
    key
        Protein identifier; written into the ``Protein Key`` column.
    structure, bfactor
        Arrays of Cα coordinates and Cα B-factors (matching residue count).
    breakpoints
        Iterable of ints from the cached regression; only the first and
        last value are used here to delimit the overall LRR region.
    period, filter_order
        Tuning for the Butterworth filter (see :func:`_butterworth_bandpass_sos`).

    Returns
    -------
    pandas.DataFrame or None
        One row per residue in the LRR region, with the columns listed in
        :data:`_CSV_COLUMN_ORDER`.
    """
    # Local imports keep module-load time small.
    from scipy import signal as sp_signal

    from mamp_ml.lrr_annotation.geom_lrr.analyzer import compute_winding

    # compute_winding returns one fewer value than there are residues, since
    # winding is a cumulative integral along the chain.
    winding_result = compute_winding(structure)
    winding: np.ndarray = winding_result["winding"]

    if len(bfactor) != len(winding) + 1:
        warnings.warn(
            f"{key}: length mismatch — bfactor has {len(bfactor)} residues, "
            f"compute_winding produced {len(winding)} (expected "
            f"{len(bfactor) - 1}); skipping.",
            stacklevel=2,
        )
        return None

    bps = list(int(bp) for bp in np.asarray(breakpoints).tolist())
    if len(bps) < 2:
        warnings.warn(
            f"{key}: need at least two breakpoints to define the LRR region; "
            f"got {bps}; skipping.",
            stacklevel=2,
        )
        return None

    # Only the first and last breakpoints are used — the cached pickle may
    # carry intermediate ones from the four-breakpoint subdivision path in
    # the regression, but the LRR overall region is always [first, last).
    start = bps[0]
    end = bps[-1]
    if start < 0 or end > len(bfactor) or start >= end:
        warnings.warn(
            f"{key}: invalid LRR region [{start}, {end}) "
            f"for bfactor of length {len(bfactor)}; skipping.",
            stacklevel=2,
        )
        return None

    bfactor_segment = bfactor[start:end]
    # sosfiltfilt requires a minimum signal length of ~3 * filter_order; we
    # use the same threshold the legacy did (filter_order * 2 + 1) which is
    # conservative.
    min_len_for_filter = filter_order * 2 + 1
    if len(bfactor_segment) < min_len_for_filter:
        return None

    sos = _butterworth_bandpass_sos(period, filter_order)
    try:
        bff = sp_signal.sosfiltfilt(sos, bfactor_segment)
    except ValueError as exc:
        warnings.warn(
            f"{key}: Butterworth filter failed on segment of length "
            f"{len(bfactor_segment)}: {exc}; skipping.",
            stacklevel=2,
        )
        return None

    bff = _normalize_filtered_bfactor(bff)
    repeat_numbers = _compute_zero_crossing_repeats(bff)

    n = len(bff)
    residue_indices = np.arange(start, start + n)

    # Per the legacy convention, the winding-number column is indexed at
    # ``residue_idx - 1`` (because winding has length n_residues - 1 and is
    # tied to inter-residue intervals). Residues whose offset falls outside
    # the winding array — i.e. ``residue_idx == 0`` — are tagged NaN.
    winding_indices = residue_indices - 1
    winding_values = np.full(n, np.nan, dtype=float)
    valid_mask = (winding_indices >= 0) & (winding_indices < len(winding))
    winding_values[valid_mask] = winding[winding_indices[valid_mask]]

    return pd.DataFrame(
        {
            "Protein Key": [key] * n,
            "Residue Index": residue_indices,
            "Filtered B-Factor": bff,
            "Winding Number": winding_values,
            "LRR Repeat Number": repeat_numbers,
        }
    )


# =============================================================================
# Section 4 :: Batch driver + on-disk writer
# =============================================================================


def compute_bfactor_lrr_segments(
    pdb_dir: PathLike,
    cache_dir: PathLike,
    *,
    period: int = DEFAULT_PERIOD,
    filter_order: int = DEFAULT_FILTER_ORDER,
) -> pd.DataFrame:
    """Compute the B-factor bandpass analysis for every PDB in ``pdb_dir``.

    Iterates the input directory in **sorted filename order** so output rows
    are deterministic across machines and Python versions (the legacy used
    ``os.listdir`` ordering, which is platform-dependent).

    Parameters
    ----------
    pdb_dir
        Directory of PDB files, one per protein, named ``{key}.pdb``.
    cache_dir
        Directory containing the ``breakpoints.pickle`` produced by the
        structure stage.
    period, filter_order
        Tuning for the Butterworth filter; defaults match the legacy
        script.

    Returns
    -------
    pandas.DataFrame
        Concatenation of per-protein DataFrames in sorted-key order. Empty
        if no protein produced any rows.

    Raises
    ------
    FileNotFoundError
        If ``pdb_dir`` or ``cache_dir`` (or ``cache_dir/breakpoints.pickle``)
        is missing.
    """
    pdb_dir = Path(pdb_dir)
    cache_dir = Path(cache_dir)

    if not pdb_dir.is_dir():
        raise FileNotFoundError(f"PDB directory not found: {pdb_dir}")
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    breakpoints_by_key = _load_cached_breakpoints(cache_dir)

    # Iterate sorted filenames to keep row order stable.
    pdb_paths = sorted(pdb_dir.glob("*.pdb"))

    per_protein_dfs: List[pd.DataFrame] = []
    for pdb_path in pdb_paths:
        key = pdb_path.stem
        if key not in breakpoints_by_key:
            # Common case for partial structure-stage runs: a PDB was
            # produced but its breakpoints weren't cached (e.g. because the
            # regression failed). Skip silently — the legacy did this via a
            # set-intersection.
            continue
        try:
            coords, bfactors = _read_pdb_ca_data(pdb_path)
        except Exception as exc:
            warnings.warn(
                f"{key}: failed to read PDB {pdb_path.name}: {exc}; skipping.",
                stacklevel=2,
            )
            continue

        df = _analyze_single_protein(
            key,
            coords,
            bfactors,
            breakpoints_by_key[key],
            period=period,
            filter_order=filter_order,
        )
        if df is not None and len(df) > 0:
            per_protein_dfs.append(df)

    if not per_protein_dfs:
        return pd.DataFrame(columns=_CSV_COLUMN_ORDER)
    combined = pd.concat(per_protein_dfs, ignore_index=True)
    return combined[_CSV_COLUMN_ORDER]


def write_bfactor_lrr_segments(
    pdb_dir: PathLike,
    cache_dir: PathLike,
    output_csv: PathLike,
    *,
    period: int = DEFAULT_PERIOD,
    filter_order: int = DEFAULT_FILTER_ORDER,
) -> pd.DataFrame:
    """Run :func:`compute_bfactor_lrr_segments` and persist as CSV.

    Parameters
    ----------
    pdb_dir, cache_dir, period, filter_order
        See :func:`compute_bfactor_lrr_segments`.
    output_csv
        Where to write the CSV. Parent directories are created on demand.

    Returns
    -------
    pandas.DataFrame
        The DataFrame written to disk.
    """
    df = compute_bfactor_lrr_segments(
        pdb_dir, cache_dir, period=period, filter_order=filter_order
    )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
