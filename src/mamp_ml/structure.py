"""mamp_ml.structure — structure-prediction post-processing stage.

Fresh Python replacement for ``scripts/02_alphafold_to_lrr_annotation.py``.
The legacy script glued together three jobs that this module exposes as
discrete, individually-testable functions:

1. :func:`parse_colabfold_log` — parse the ColabFold ``log.txt`` into a
   ``{receptor: {model_num: ModelScore}}`` mapping, keeping the highest
   pLDDT per ``(receptor, model_num)`` pair (the log can carry multiple
   entries per model — initial pass, per-recycle, ``rank_001`` — and the
   final post-processing pass typically has the highest score).
2. :func:`write_alphafold_scores` — emit the legacy ``alphafold_scores.txt``
   format and return ``{receptor: best_model_num}``.
3. :func:`select_best_pdb_files` + :func:`copy_best_pdbs` — locate each
   receptor's best model PDB in the ColabFold output dir and copy it into
   a flat target dir keyed on the receptor name.
4. :func:`annotate_lrr_regions` — run the geometric LRR annotation
   (winding-number → piecewise-linear regression → breakpoints → sequence
   extraction) and write the legacy ``lrr_annotation_results.txt``.

The whole pipeline is also exposed as :func:`run_structure_stage` for the
common case where a caller wants to drive all four steps end-to-end.

This module reuses ``mamp_ml.lrr_annotation.geom_lrr`` (Loader, Analyzer,
Plotter, compute_winding, compute_regression — the *shared* numerical
algorithm library) but contains no code in common with
``scripts/02_alphafold_to_lrr_annotation.py`` or with
``mamp_ml.lrr_annotation.extract_lrr_sequences``: PDB I/O, log parsing,
file selection and result formatting are all reimplemented here.
"""

from __future__ import annotations

import re
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, NamedTuple, Optional, Union

PathLike = Union[str, Path]

__all__ = [
    "ModelScore",
    "parse_colabfold_log",
    "write_alphafold_scores",
    "select_best_pdb_files",
    "copy_best_pdbs",
    "annotate_lrr_regions",
    "run_structure_stage",
]


# =============================================================================
# Section 1 :: ColabFold log parsing
# =============================================================================


class ModelScore(NamedTuple):
    """Final pLDDT and pTM scores for a single AlphaFold2 model run."""

    plddt: float
    ptm: float


# Compiled at module load — each regex is exercised on every log line so
# avoiding re-compilation is worth the explicit module-level binding.
_QUERY_LINE_RE = re.compile(r"Query \d+/\d+:\s+(\S+)\s+\(length \d+\)")
_MODEL_NUM_RE = re.compile(r"alphafold2_ptm_model_(\d+)")
_PLDDT_RE = re.compile(r"pLDDT=(\d+(?:\.\d+)?)")
_PTM_RE = re.compile(r"pTM=(\d+(?:\.\d+)?)")


def parse_colabfold_log(
    log_path: PathLike,
) -> Dict[str, Dict[str, ModelScore]]:
    """Parse a ColabFold ``log.txt`` into per-receptor best-model scores.

    Walks the log line-by-line, alternating between receptor identification
    (``Query N/M: <name> (length L)``) and per-model score lines
    (``alphafold2_ptm_model_<N> ... pLDDT=<x> pTM=<y>``). When the same
    ``(receptor, model_num)`` pair appears multiple times (the log records
    every recycle plus a final ``rank_001`` line), only the entry with the
    highest pLDDT is kept — that is the final post-processing score the
    legacy parser would have surfaced.

    Lines tagged with ``took`` (timing summaries) are skipped because their
    pLDDT field, if present, is a re-printed snapshot — never higher than
    the matching recycle/rank line.

    Parameters
    ----------
    log_path
        Path to the ColabFold log file.

    Returns
    -------
    dict
        Nested mapping ``{receptor_name: {model_num: ModelScore}}``.
        Receptors with no parsed model scores still appear with an empty
        inner dict so a caller can tell "we know of this receptor" from
        "we never saw a Query for it".

    Raises
    ------
    FileNotFoundError
        If ``log_path`` does not exist.
    """
    log_path = Path(log_path)
    if not log_path.is_file():
        raise FileNotFoundError(f"ColabFold log file not found: {log_path}")

    results: Dict[str, Dict[str, ModelScore]] = {}
    current_receptor: Optional[str] = None

    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            query_match = _QUERY_LINE_RE.search(line)
            if query_match:
                current_receptor = query_match.group(1)
                results.setdefault(current_receptor, {})
                continue

            # Skip per-model timing summaries — they re-emit pLDDT but never
            # at a higher value than the matching recycle/rank line we'd
            # already have seen.
            if "took" in line:
                continue

            model_match = _MODEL_NUM_RE.search(line)
            if not model_match or current_receptor is None:
                continue

            plddt_match = _PLDDT_RE.search(line)
            ptm_match = _PTM_RE.search(line)
            if not (plddt_match and ptm_match):
                continue

            plddt = float(plddt_match.group(1))
            ptm = float(ptm_match.group(1))
            model_num = model_match.group(1)

            existing = results[current_receptor].get(model_num)
            if existing is None or plddt > existing.plddt:
                results[current_receptor][model_num] = ModelScore(plddt, ptm)

    return results


# =============================================================================
# Section 2 :: alphafold_scores.txt summary writer
# =============================================================================

# Column widths and underline width that the legacy writer used.
_RECEPTOR_COL_WIDTH = 100
_MODEL_COL_WIDTH = 15
_PLDDT_COL_WIDTH = 10
# Note: the legacy underline is intentionally shorter than the header (125
# vs 128 chars) — we preserve that for byte-identical output.
_SEPARATOR_WIDTH = _RECEPTOR_COL_WIDTH + 25


def write_alphafold_scores(
    results: Mapping[str, Mapping[str, ModelScore]],
    output_path: PathLike,
) -> Dict[str, str]:
    """Write a fixed-width best-model summary and return ``{receptor: model_num}``.

    The output format matches the legacy ``alphafold_scores.txt`` exactly:
    a fixed-width header line, a dash-rule underline, and one data row per
    receptor (alphabetically sorted) showing the highest-pLDDT model.
    pLDDT is formatted as ``%.1f`` and pTM as ``%.3f``.

    Receptors whose inner dict is empty (e.g. a ``Query`` line was seen but
    no matching model-score line) are silently skipped — they have no best
    model to report.

    Parameters
    ----------
    results
        Output of :func:`parse_colabfold_log`.
    output_path
        Where to write the summary. Parent directories are created on
        demand. Uses unconditional LF line endings.

    Returns
    -------
    dict
        ``{receptor: best_model_num}`` for every receptor that produced a
        data row in the summary. Useful as the input to
        :func:`select_best_pdb_files`.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_models: Dict[str, str] = {}

    with open(output_path, "w", newline="\n", encoding="utf-8") as fh:
        fh.write(
            "Receptor".ljust(_RECEPTOR_COL_WIDTH)
            + "Best Model".ljust(_MODEL_COL_WIDTH)
            + "pLDDT".ljust(_PLDDT_COL_WIDTH)
            + "pTM\n"
        )
        fh.write("-" * _SEPARATOR_WIDTH + "\n")

        for receptor in sorted(results.keys()):
            models = results[receptor]
            if not models:
                continue
            # Tie-breaker: when multiple models have identical pLDDT, the
            # ``max`` builtin keeps the first one encountered in iteration
            # order, which for a regular dict is insertion order — which is
            # numeric-ascending in practice because models are logged in
            # number order. We don't depend on this anywhere.
            best_model_num = max(models, key=lambda m: models[m].plddt)
            score = models[best_model_num]
            best_models[receptor] = best_model_num

            fh.write(
                receptor.ljust(_RECEPTOR_COL_WIDTH)
                + f"model_{best_model_num}".ljust(_MODEL_COL_WIDTH)
                + f"{score.plddt:.1f}".ljust(_PLDDT_COL_WIDTH)
                + f"{score.ptm:.3f}\n"
            )

    return best_models


# =============================================================================
# Section 3 :: Best-PDB selection + copy
# =============================================================================


def select_best_pdb_files(
    best_models: Mapping[str, str],
    colabfold_dir: PathLike,
) -> Dict[str, Path]:
    """Locate each receptor's best-model PDB inside the ColabFold output dir.

    The ColabFold naming convention puts the receptor name (FASTA header
    with underscores), a rank index (``rank_001``, ``rank_002``...), the
    model number (``alphafold2_ptm_model_N``), and a seed index in the PDB
    filename. We search for the file matching the receptor and chosen
    model number; if multiple ranks of the same model are present, the
    lowest-rank file (highest-confidence) is preferred via ``sorted``.

    Receptors with no matching PDB are skipped with a runtime warning —
    this is the common case when a structure-prediction run aborted mid-
    way for one of several receptors.

    Parameters
    ----------
    best_models
        ``{receptor: best_model_num}`` (output of
        :func:`write_alphafold_scores`).
    colabfold_dir
        Directory containing the raw ColabFold PDB files.

    Returns
    -------
    dict
        ``{receptor: source_pdb_path}`` for receptors with matching files.
    """
    colabfold_dir = Path(colabfold_dir)
    selected: Dict[str, Path] = {}

    for receptor, model_num in best_models.items():
        pattern = (
            f"{receptor}_unrelaxed_rank_[0-9][0-9][0-9]"
            f"_alphafold2_ptm_model_{model_num}_seed_*.pdb"
        )
        matches = sorted(colabfold_dir.glob(pattern))
        if not matches:
            warnings.warn(
                f"No PDB file matching {pattern!r} in {colabfold_dir}; "
                f"receptor {receptor} will be skipped.",
                stacklevel=2,
            )
            continue
        if len(matches) > 1:
            # Prefer the lowest rank index (typically the highest pLDDT
            # already by colabfold convention). The sort above puts
            # rank_001 before rank_002.
            warnings.warn(
                f"Multiple PDB files matched {pattern!r}; using "
                f"{matches[0].name}.",
                stacklevel=2,
            )
        selected[receptor] = matches[0]

    return selected


def copy_best_pdbs(
    best_pdb_files: Mapping[str, Path],
    target_dir: PathLike,
) -> List[Path]:
    """Copy each receptor's chosen PDB into ``target_dir/{receptor}.pdb``.

    Parameters
    ----------
    best_pdb_files
        Output of :func:`select_best_pdb_files`.
    target_dir
        Destination directory (created on demand).

    Returns
    -------
    list of Path
        Absolute paths of the files written, in receptor-iteration order.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    for receptor, source in best_pdb_files.items():
        target = target_dir / f"{receptor}.pdb"
        shutil.copy2(source, target)
        copied.append(target)
    return copied


# =============================================================================
# Section 4 :: LRR-region annotation (geometric algorithm + sequence slicing)
# =============================================================================
#
# This is the only section that calls into the shared geom_lrr math library.
# The PDB → sequence projection and the breakpoint-pair slicing logic are
# implemented locally rather than imported from
# ``mamp_ml.lrr_annotation.extract_lrr_sequences``; the new code shares no
# orchestration helpers with the legacy.


def _read_pdb_sequence(pdb_path: Path) -> str:
    """Extract the single-letter amino-acid sequence from a PDB file.

    Uses Biopython's polypeptide builder, which segments the structure into
    contiguous peptide stretches (handling chain breaks gracefully) and
    concatenates them. The resulting string is the indexing reference for
    the breakpoint-pair slicing below.

    Parameters
    ----------
    pdb_path
        Path to a single-chain PDB file (the format produced by ColabFold
        and copied by :func:`copy_best_pdbs`).

    Returns
    -------
    str
        Concatenated single-letter sequence of all polypeptide segments.
    """
    # Local imports to keep the module's import cost low when callers only
    # need the log-parsing pieces.
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import PPBuilder

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    ppb = PPBuilder()
    return "".join(str(pp.get_sequence()) for pp in ppb.build_peptides(structure))


def _slice_lrr_regions(
    sequence: str,
    breakpoints,
) -> List[tuple]:
    """Slice ``sequence`` into LRR-region (sub-sequence, (start, end)) pairs.

    Breakpoints are taken pairwise (``[start_0, end_0, start_1, end_1, ...]``).
    Out-of-range indices are clamped to the sequence bounds, matching the
    convention used by the upstream geometric analyzer.

    Parameters
    ----------
    sequence
        Full amino-acid sequence to slice.
    breakpoints
        Iterable of integers (or numpy array of integers) describing the
        region boundaries in pairs.

    Returns
    -------
    list of (str, (int, int))
        ``(sub_sequence, (start, end))`` per LRR region.
    """
    bps = [int(bp) for bp in breakpoints]
    regions: List[tuple] = []
    # ``range(0, len(bps) - 1, 2)`` yields the index of each pair's start;
    # a stray final breakpoint with no partner is silently ignored.
    for i in range(0, len(bps) - 1, 2):
        start = max(0, bps[i])
        end = min(len(sequence), bps[i + 1])
        regions.append((sequence[start:end], (start, end)))
    return regions


def annotate_lrr_regions(
    pdb_dir: PathLike,
    output_results_path: PathLike,
    *,
    cache_dir: Optional[PathLike] = None,
    plot_dir: Optional[PathLike] = None,
) -> int:
    """Annotate LRR regions for every PDB in ``pdb_dir``; write the legacy TSV.

    For each PDB file the geometric analyzer computes a winding number,
    fits a piecewise-linear regression to determine the LRR-region
    boundaries, and this function then slices the corresponding amino-acid
    sub-sequences out of the PDB. The tab-separated output matches the
    legacy ``lrr_annotation_results.txt`` schema (PDB filename, region
    index, start, end, sub-sequence length, full sequence length, total
    region count, sequence).

    PDBs are processed in **sorted filename order** so the output is
    deterministic across runs and operating systems — the legacy script
    relied on ``os.listdir`` ordering which is platform-dependent.

    Parameters
    ----------
    pdb_dir
        Directory containing one PDB per receptor, named
        ``{receptor}.pdb``.
    output_results_path
        Where to write the TSV. Parent directories are created on demand.
    cache_dir
        If provided, the structures + geometry + regression pickles are
        cached here for use by the downstream B-factor analysis stage.
    plot_dir
        If provided, the per-receptor regression plots are written here.

    Returns
    -------
    int
        Number of LRR-region rows written to ``output_results_path``.
    """
    pdb_dir = Path(pdb_dir)
    output_results_path = Path(output_results_path)
    output_results_path.parent.mkdir(parents=True, exist_ok=True)

    # Local import — keeps the cost of `import mamp_ml.structure` low when
    # callers only need the log-parsing helpers.
    from mamp_ml.lrr_annotation import Analyzer, Loader, Plotter

    loader = Loader()
    loader.load_batch(str(pdb_dir))

    analyzer = Analyzer()
    analyzer.load_structures(loader.structures)
    analyzer.compute_windings()
    analyzer.compute_regressions()

    n_written = 0
    sorted_keys = sorted(analyzer.breakpoints.keys())

    with open(output_results_path, "w", newline="\n", encoding="utf-8") as fh:
        fh.write(
            "PDB_Filename\tRegion_Number\tStart_Position\tEnd_Position\t"
            "Sequence_Length\tFull_Sequence_Length\tTotal_LRR_Regions\t"
            "Sequence\n"
        )

        for pdb_id in sorted_keys:
            breakpoints = analyzer.breakpoints[pdb_id]
            pdb_file = pdb_dir / f"{pdb_id}.pdb"
            if not pdb_file.is_file():
                warnings.warn(
                    f"Expected PDB file not found: {pdb_file}; skipping.",
                    stacklevel=2,
                )
                continue

            full_sequence = _read_pdb_sequence(pdb_file)
            regions = _slice_lrr_regions(full_sequence, breakpoints)
            num_regions = len(regions)
            full_len = len(full_sequence)

            for region_idx, (seq, (start, end)) in enumerate(regions, start=1):
                fh.write(
                    f"{pdb_file.name}\t{region_idx}\t{start}\t{end}\t"
                    f"{len(seq)}\t{full_len}\t{num_regions}\t{seq}\n"
                )
                n_written += 1

    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        loader.cache(str(cache_path))
        analyzer.cache_geometry(str(cache_path))
        analyzer.cache_regressions(str(cache_path))

    if plot_dir is not None:
        plot_path = Path(plot_dir)
        plot_path.mkdir(parents=True, exist_ok=True)
        plotter = Plotter()
        plotter.load(analyzer.windings, analyzer.breakpoints, analyzer.slopes)
        plotter.plot_regressions(save=True, directory=str(plot_path))

    return n_written


# =============================================================================
# Section 5 :: One-shot orchestrator
# =============================================================================


def run_structure_stage(
    colabfold_dir: PathLike,
    scores_path: PathLike,
    pdb_target_dir: PathLike,
    lrr_results_path: PathLike,
    *,
    log_filename: str = "log.txt",
    cache_dir: Optional[PathLike] = None,
    plot_dir: Optional[PathLike] = None,
) -> int:
    """Drive the full structure-stage pipeline from a ColabFold output dir.

    Equivalent to running, in order:

    1. :func:`parse_colabfold_log` on ``{colabfold_dir}/{log_filename}``
    2. :func:`write_alphafold_scores` -> ``scores_path``
    3. :func:`select_best_pdb_files` + :func:`copy_best_pdbs` ->
       ``pdb_target_dir``
    4. :func:`annotate_lrr_regions` -> ``lrr_results_path`` (with optional
       cache and plot directories).

    Returns the number of LRR-region rows written.
    """
    colabfold_dir = Path(colabfold_dir)
    log_path = colabfold_dir / log_filename

    parsed = parse_colabfold_log(log_path)
    best_models = write_alphafold_scores(parsed, scores_path)
    selected = select_best_pdb_files(best_models, colabfold_dir)
    copy_best_pdbs(selected, pdb_target_dir)
    return annotate_lrr_regions(
        pdb_target_dir,
        lrr_results_path,
        cache_dir=cache_dir,
        plot_dir=plot_dir,
    )
