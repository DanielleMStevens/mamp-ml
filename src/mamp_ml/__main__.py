"""mamp_ml — package-level CLI dispatcher.

The primary user entry point is the **``prepare``** command, which runs the
full preparation pipeline in one shot:

    python -m mamp_ml prepare INPUT.xlsx

This invocation:

1. Builds the receptor FASTA from ``INPUT.xlsx``.
2. Checks whether ColabFold has already been run for these receptors.
   * If yes (the ``intermediate_files/receptor_only/log.txt`` is present),
     continues to step 3 silently.
   * If no, but a ``colabfold_batch`` install is discoverable on the host,
     runs it automatically (by absolute path, so no env activation is
     needed) and continues to step 3 with the fresh outputs.
   * If no install can be found, prints the exact ``colabfold_batch``
     command to run next and exits non-zero so the workflow stops cleanly.
3. Runs the structure-stage (LRR-annotation), B-factor analysis,
   test-data assembly, and chemical-features stages in sequence, producing
   ``intermediate_files/ready_test_data.csv`` ready for model evaluation.

The six per-stage subcommands (``prepare-fasta``, ``structure-stage``,
``lrr-domain-fasta``, ``bfactor``, ``assemble-test-data``,
``chemical-features``) remain available as escape hatches for debugging or
when only a subset of the pipeline needs to be re-run.

Subcommands
-----------

prepare (top-level)
    Run the full preparation pipeline in one shot.

prepare-fasta
    Convert an input ``.xlsx`` to a deduplicated receptor FASTA.

structure-stage
    Drive the structure-prediction post-processing: parse the ColabFold
    ``log.txt``, copy each receptor's best PDB, and run the geometric LRR
    annotation, optionally caching the geometry pickles and writing
    per-receptor regression plots.

lrr-domain-fasta
    Join LRR-annotation results with the receptor FASTA to produce an
    LRR-domain FASTA keyed on ``species|locus|receptor|LRR_domain``.

bfactor
    Run the B-factor bandpass analysis over the per-receptor PDBs using
    cached breakpoints, writing the ``bfactor_winding_lrr_segments.csv``
    consumed by the model's :class:`BFactorWeightGenerator`.

assemble-test-data
    Join the per-row receptor/ligand pairs from the spreadsheet with the
    LRR-domain FASTA to produce ``test_data.csv``.

chemical-features
    Append per-residue bulkiness / charge / hydrophobicity columns,
    producing ``ready_test_data.csv``.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


# Shared help text for the --cache-dir flag on the model-loading subcommands.
_CACHE_DIR_HELP = (
    "Directory for the HuggingFace model cache (ESM-2 / ESMFold weights). "
    "Defaults to a `model_cache/` folder next to the mamp-ml install — i.e. "
    "the filesystem you installed onto. Override here (or via $HF_HOME) to use "
    "a different location, e.g. if HOME is over quota on a cluster."
)


# Worked-example block surfaced at the bottom of both `mamp-ml --help` and
# `mamp-ml predict --help` via argparse's `epilog`. Mirrors the canonical
# workflow documented in the README so users get the answer to "how do I
# actually run this" without leaving the terminal.
_USAGE_EXAMPLES = """\
Example usage
-------------

  # Smoke-test your install on the bundled sample (no data of your own needed)
  mamp-ml predict --example --device cuda

  # Full pipeline: spreadsheet -> predictions.csv
  mamp-ml predict input_data.xlsx --device cuda

  # Use ESMFold instead of ColabFold (no separate conda env needed)
  mamp-ml predict input_data.xlsx --structure esmfold --device cuda

  # Use a custom-trained model instead of the bundled weights
  mamp-ml predict input_data.xlsx --weights /path/to/checkpoint.pth

  # Keep every intermediate file (default keeps only predictions + plots)
  mamp-ml predict input_data.xlsx --keep all

Output
------
  predict writes an `<output-name>/` folder (default: a unique
  `output_<timestamp>/`) containing:
    - predictions.csv      one row per receptor-ligand pair, with the predicted
                           class (Immunogenic / Non-Immunogenic / Weakly
                           Immunogenic) and the per-class probabilities
    - lrr_annotation_plots/  per-receptor LRR regression plots

See the README at https://github.com/DanielleMStevens/mamp-ml for the
full workflow + input spreadsheet format.
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mamp_ml",
        description=(
            "MAMP-ml CLI. Use `predict` for the one-shot end-to-end pipeline; "
            "per-stage subcommands are escape hatches for debugging."
        ),
        epilog=_USAGE_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    from mamp_ml import __version__

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"mamp-ml {__version__}",
        help="Print the installed mamp-ml version and exit.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="SUBCOMMAND")

    # prepare (one-shot top-level) -------------------------------------
    sp = sub.add_parser(
        "prepare",
        help=(
            "Run the full preparation pipeline in one shot "
            "(xlsx -> ready_test_data.csv)."
        ),
        description=(
            "Runs every preparation stage in sequence: receptor FASTA "
            "generation, structure-stage post-processing, LRR-domain FASTA "
            "assembly, B-factor bandpass analysis, test-data assembly, and "
            "chemical-feature annotation. If ColabFold has not been run yet "
            "the command auto-runs a discovered colabfold_batch install (or, "
            "if none is found, writes the receptor FASTA and prints the "
            "colabfold_batch invocation, exiting cleanly so the user can run "
            "ColabFold and re-invoke this command)."
        ),
    )
    sp.add_argument("xlsx", help="Path to input .xlsx file")
    sp.add_argument(
        "--out-dir",
        default="intermediate_files",
        help=(
            "Directory for pipeline intermediates and the final "
            "ready_test_data.csv (default: intermediate_files)."
        ),
    )
    sp.add_argument(
        "--colabfold-dir",
        default=None,
        help=(
            "Directory containing colabfold log.txt + raw PDBs. "
            "Defaults to <out_dir>/receptor_only."
        ),
    )
    sp.add_argument(
        "--structure-cache-dir",
        default="LRR_Annotation/cache",
        help=(
            "Directory for the structure-stage geometry pickles "
            "(default: LRR_Annotation/cache). Matches legacy convention."
        ),
    )
    sp.add_argument(
        "--bfactor-cache-dir",
        default=None,
        help=(
            "Directory the bfactor stage reads breakpoints from. "
            "Default: the production cache shipped with mamp_ml "
            "(src/mamp_ml/lrr_annotation/cache). Use the structure-cache-dir "
            "value for fresh receptors not in the production cache."
        ),
    )
    sp.add_argument(
        "--sheet-name",
        default="Sheet1",
        help="Sheet name to read from the workbook (default: Sheet1)",
    )
    sp.add_argument(
        "--structure",
        default="colabfold",
        choices=["colabfold", "esmfold"],
        help=(
            "Structure-prediction tool to use. `colabfold` (default) auto-runs "
            "a discovered colabfold_batch install when its outputs are missing "
            "(falling back to a copy-paste hint only when none is found). "
            "`esmfold` auto-runs HuggingFace facebook/esmfold_v1 in-process "
            "(requires `pip install mamp-ml[esmfold]`)."
        ),
    )
    sp.add_argument(
        "--device",
        default=None,
        help=(
            "Torch device for in-process ESMFold (cpu / cuda / mps). "
            "Defaults to cuda if available else cpu. Ignored for the "
            "colabfold structure tool."
        ),
    )
    sp.add_argument(
        "--cache-dir",
        default=None,
        help=_CACHE_DIR_HELP,
    )
    sp.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "ESMFold trunk chunk size (typical: 128 / 64 / 32). Splits the "
            "folding trunk's triangular attention into chunks of that many "
            "tokens, dramatically lowering peak VRAM at the cost of some "
            "wall-clock. By default an appropriate value is auto-picked from "
            "the host's free VRAM on CUDA devices; pass an explicit integer "
            "to override. Ignored for --structure colabfold."
        ),
    )
    sp.add_argument(
        "--max-length",
        type=int,
        default=1300,
        help=(
            "Fold only the first N residues of each receptor with ColabFold "
            "(default: 1300). ColabFold/AlphaFold2 memory scales steeply with "
            "length, so long receptors can OOM-kill colabfold_batch (status "
            "-9); the LRR ectodomain is N-terminal, so truncating preserves "
            "it. The full-length FASTA is kept intact for header lookup. Pass "
            "0 to disable truncation. Ignored for --structure esmfold (which "
            "applies its own positional-embedding cap)."
        ),
    )

    # find-colabfold ---------------------------------------------------
    sp = sub.add_parser(
        "find-colabfold",
        help="Locate existing colabfold_batch installations on this system.",
        description=(
            "Searches common locations for `colabfold_batch` — the user's "
            "$PATH, the conda envs implied by $CONDA_PREFIX, the standard "
            "conda root directories (~/anaconda3, /opt/miniconda3, ...), "
            "and known localcolabfold paths — and prints each install "
            "along with the `export PATH=...` line needed to activate it. "
            "Useful on cluster hosts where ColabFold is already installed "
            "but the user doesn't know where."
        ),
    )

    # example (drop the bundled sample dataset) ---------------------------
    sp = sub.add_parser(
        "example",
        help="Copy the bundled example_data.xlsx sample into the current dir.",
        description=(
            "Writes the example_data.xlsx sample that ships inside the "
            "installed package into the working directory so a pip-installed "
            "user can smoke-test their install without cloning the repo:\n\n"
            "  mamp-ml example\n"
            "  mamp-ml predict example_data.xlsx --device cuda\n\n"
            "Pass --path to print the bundled file's location instead of "
            "copying it (handy for scripting)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sp.add_argument(
        "--out-dir",
        default=".",
        help="Directory to copy the sample into (default: current directory).",
    )
    sp.add_argument(
        "--path",
        action="store_true",
        help="Print the bundled sample's absolute path and exit (no copy).",
    )
    sp.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing example_data.xlsx in the target directory.",
    )

    # fold (standalone folding step) --------------------------------------
    sp = sub.add_parser(
        "fold",
        help="Fold receptor structures with the chosen structure tool.",
        description=(
            "Runs the structure-prediction tool on the receptor FASTA "
            "without touching the downstream pipeline. Useful for users who "
            "want to inspect / replace structures before running prediction. "
            "For `--structure colabfold`, prints the colabfold_batch "
            "invocation to run (we don't auto-shell-out to it). For "
            "`--structure esmfold`, runs facebook/esmfold_v1 in-process."
        ),
    )
    sp.add_argument(
        "fasta",
        help=(
            "Receptor FASTA produced by `mamp-ml prepare-fasta` "
            "(intermediate_files/receptor_full_length.fasta by default)."
        ),
    )
    sp.add_argument(
        "output_dir",
        help=(
            "Where to write the folded PDBs + log.txt "
            "(intermediate_files/receptor_only by convention)."
        ),
    )
    sp.add_argument(
        "--structure",
        default="esmfold",
        choices=["colabfold", "esmfold"],
        help=(
            "Structure-prediction tool; `esmfold` (default for the fold "
            "subcommand) runs in-process, `colabfold` prints the "
            "colabfold_batch invocation to run externally."
        ),
    )
    sp.add_argument(
        "--device",
        default=None,
        help="Torch device (cpu / cuda / mps); only relevant for esmfold.",
    )
    sp.add_argument(
        "--cache-dir",
        default=None,
        help=_CACHE_DIR_HELP,
    )
    sp.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help=(
            "Truncate input sequences to this many residues (ESMFold has a "
            "1024-AA positional-embedding cap; the LRR ectodomain is "
            "N-terminal so the truncation preserves it)."
        ),
    )
    sp.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "ESMFold trunk chunk size (typical: 128 / 64 / 32). Lowers peak "
            "VRAM at the cost of some wall-clock. By default an appropriate "
            "value is auto-picked from the host's free VRAM on CUDA devices; "
            "pass an explicit integer to override."
        ),
    )

    # predict (one-shot top-level, end-to-end including model inference) ---
    sp = sub.add_parser(
        "predict",
        help=(
            "Run the full pipeline plus model inference "
            "(xlsx -> predictions CSV)."
        ),
        description=(
            "Runs `prepare` to produce ready_test_data.csv, then loads the "
            "bundled MAMP-ml checkpoint and runs ESM-2 inference, writing the "
            "predictions CSV the user actually wants — one row per "
            "receptor-ligand pair with the predicted class (Immunogenic / "
            "Non-Immunogenic / Weakly Immunogenic) and per-class probabilities. "
            "If ColabFold has not been run yet the command auto-runs a "
            "discovered colabfold_batch install (or prints the invocation and "
            "exits cleanly when none is found, so the user can run ColabFold "
            "and re-invoke this command)."
        ),
        epilog=_USAGE_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sp.add_argument(
        "xlsx",
        nargs="?",
        default=None,
        help=(
            "Path to input .xlsx file. Omit and pass --example to run on the "
            "bundled sample dataset instead."
        ),
    )
    sp.add_argument(
        "--example",
        action="store_true",
        help=(
            "Run on the example_data.xlsx sample bundled inside the package so "
            "you can smoke-test your install without supplying real data. "
            "Overrides the xlsx positional if both are given."
        ),
    )
    sp.add_argument(
        "--out-dir",
        default="intermediate_files",
        help="Directory for pipeline intermediates (default: intermediate_files).",
    )
    sp.add_argument(
        "--colabfold-dir",
        default=None,
        help=(
            "Directory containing colabfold log.txt + raw PDBs. "
            "Defaults to <out_dir>/receptor_only."
        ),
    )
    sp.add_argument(
        "--structure-cache-dir",
        default="LRR_Annotation/cache",
        help="Directory for the structure-stage geometry pickles.",
    )
    sp.add_argument(
        "--bfactor-cache-dir",
        default=None,
        help="Override for the bfactor-stage breakpoints cache directory.",
    )
    sp.add_argument(
        "--weights",
        default=None,
        help=(
            "Path to a custom model weights file (.pth). Defaults to the "
            "bundled mamp_ml_weights.pth shipped inside the package; "
            "override with a path to your own trained checkpoint to "
            "predict against an alternative model."
        ),
    )
    sp.add_argument(
        "--device",
        default="cpu",
        help=(
            "Torch device to run inference on (default: cpu). "
            "Pass 'cuda' on a GPU host for ~50-100x speedup. "
            "Also used for ESMFold folding when --structure esmfold is selected."
        ),
    )
    sp.add_argument(
        "--cache-dir",
        default=None,
        help=_CACHE_DIR_HELP,
    )
    sp.add_argument(
        "--sheet-name",
        default="Sheet1",
        help="Sheet name to read from the workbook (default: Sheet1)",
    )
    sp.add_argument(
        "--structure",
        default="colabfold",
        choices=["colabfold", "esmfold"],
        help=(
            "Structure-prediction tool (default: colabfold, which auto-runs a "
            "discovered colabfold_batch install when its outputs are missing). "
            "Set to `esmfold` to auto-run facebook/esmfold_v1 in-process "
            "instead."
        ),
    )
    sp.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "ESMFold trunk chunk size (typical: 128 / 64 / 32). Lowers peak "
            "VRAM at the cost of some wall-clock. By default an appropriate "
            "value is auto-picked from the host's free VRAM on CUDA devices; "
            "pass an explicit integer to override. Ignored for --structure "
            "colabfold."
        ),
    )
    sp.add_argument(
        "--max-length",
        type=int,
        default=1300,
        help=(
            "Fold only the first N residues of each receptor with ColabFold "
            "(default: 1300). ColabFold/AlphaFold2 memory scales steeply with "
            "length, so long receptors can OOM-kill colabfold_batch (status "
            "-9); the LRR ectodomain is N-terminal, so truncating preserves "
            "it. The full-length FASTA is kept intact for header lookup. Pass "
            "0 to disable truncation. Ignored for --structure esmfold (which "
            "applies its own positional-embedding cap)."
        ),
    )
    sp.add_argument(
        "--output-name",
        default=None,
        help=(
            "Name of the output folder this run writes into; it contains "
            "predictions.csv and lrr_annotation_plots/. Defaults to a unique, "
            "timestamped name like `output_2026-06-15_20-30-00` so repeated "
            "runs don't overwrite each other. Applies to the default --keep mode."
        ),
    )
    sp.add_argument(
        "--keep",
        default="default",
        choices=["default", "all"],
        help=(
            "Whether to keep the intermediate_files/ working directory. Either "
            "way the outputs (predictions.csv + lrr_annotation_plots/) are moved "
            "into a labeled <output-name>/ folder in the current directory. The "
            "default removes intermediate_files/; pass `all` to keep it (useful "
            "for debugging or for re-running prediction without re-folding)."
        ),
    )

    # prepare-fasta -----------------------------------------------------
    sp = sub.add_parser(
        "prepare-fasta",
        help="Convert input xlsx to receptor full-length FASTA.",
    )
    sp.add_argument("xlsx", help="Path to input .xlsx file")
    sp.add_argument("out_fasta", help="Output FASTA path")
    sp.add_argument(
        "--sheet-name",
        default="Sheet1",
        help="Sheet name to read from the workbook (default: Sheet1)",
    )

    # structure-stage --------------------------------------------------
    sp = sub.add_parser(
        "structure-stage",
        help="Parse ColabFold log, copy best PDBs, and run LRR annotation.",
    )
    sp.add_argument(
        "colabfold_dir",
        help="Directory containing colabfold log.txt and raw PDBs",
    )
    sp.add_argument(
        "scores_path",
        help="Output path for the alphafold_scores.txt summary",
    )
    sp.add_argument(
        "pdb_target_dir",
        help="Output directory for best-model PDBs renamed to {receptor}.pdb",
    )
    sp.add_argument(
        "lrr_results_path",
        help="Output path for the lrr_annotation_results.txt TSV",
    )
    sp.add_argument(
        "--cache-dir",
        default=None,
        help="If set, structure/winding/regression pickles are written here",
    )
    sp.add_argument(
        "--plot-dir",
        default=None,
        help="If set, per-receptor regression plots are written here",
    )

    # lrr-domain-fasta -------------------------------------------------
    sp = sub.add_parser(
        "lrr-domain-fasta",
        help="Build LRR-domain FASTA from annotation results + receptor FASTA.",
    )
    sp.add_argument("lrr_annotation_results")
    sp.add_argument("receptor_full_length_fasta")
    sp.add_argument("out_fasta")

    # bfactor ----------------------------------------------------------
    sp = sub.add_parser(
        "bfactor",
        help="Compute B-factor bandpass LRR-repeat annotation CSV.",
    )
    sp.add_argument("pdb_dir", help="Directory of {receptor}.pdb files")
    sp.add_argument(
        "cache_dir",
        help="Directory containing breakpoints.pickle",
    )
    sp.add_argument("out_csv", help="Output CSV path")

    # assemble-test-data ----------------------------------------------
    sp = sub.add_parser(
        "assemble-test-data",
        help="Assemble per-row test_data.csv from xlsx + LRR-domain FASTA.",
    )
    sp.add_argument("xlsx", help="Path to input .xlsx file")
    sp.add_argument("lrr_domain_fasta")
    sp.add_argument("out_csv")
    sp.add_argument(
        "--sheet-name",
        default="Sheet1",
        help="Sheet name to read from the workbook (default: Sheet1)",
    )

    # chemical-features ------------------------------------------------
    sp = sub.add_parser(
        "chemical-features",
        help="Append per-residue chemical-feature columns to test_data.csv.",
    )
    sp.add_argument("in_csv")
    sp.add_argument("out_csv")

    return parser


def _resolve_default_bfactor_cache() -> "Path":
    """Locate the production breakpoints cache shipped with the package.

    Resolved at runtime so an installed wheel finds it next to the package
    sources, not next to the user's CWD. The cache holds the breakpoints
    the model was originally trained against; for receptors not in it, pass
    ``--bfactor-cache-dir`` pointing at a freshly-generated cache instead.
    """
    from pathlib import Path

    return Path(__file__).resolve().parent / "lrr_annotation" / "cache"


def _pipeline_subtitle(args) -> str:
    """One-line context shown under the run banner: input + device + tool."""
    bits = [f"input: {args.xlsx}"]
    device = getattr(args, "device", None)
    if device:
        bits.append(f"device: {device}")
    bits.append(f"structure: {getattr(args, 'structure', 'colabfold')}")
    return "  ·  ".join(bits)


def _default_model_cache_dir() -> "Path":
    """Default HuggingFace model-cache location: a ``model_cache/`` directory
    next to the mamp-ml install.

    This lands the (multi-GB) ESM-2 / ESMFold weights on whatever filesystem
    mamp-ml was installed to — which avoids a small-quota HOME ``~/.cache`` for
    cluster users who install onto scratch — without hardcoding any
    site-specific path. Override with ``--cache-dir`` or ``$HF_HOME``.
    """
    from pathlib import Path

    import mamp_ml

    return Path(mamp_ml.__file__).resolve().parent / "model_cache"


def _configure_model_cache(args) -> None:
    """Point the HuggingFace cache at a writable location, before any load.

    Resolution order (highest priority first):

    1. ``--cache-dir`` on the CLI,
    2. an ``HF_HOME`` / ``HF_HUB_CACHE`` the user already exported (respected,
       never overridden),
    3. :func:`_default_model_cache_dir` — next to the install.

    Sets ``HF_HOME`` in the process environment so it is inherited by both
    in-process model loads (ESM-2, ESMFold) and the ColabFold subprocess. Must
    run *before* ``transformers`` / ``huggingface_hub`` are imported — we call
    it at the top of :func:`main`. No-op (leaving HF's own default) if the
    directory can't be created, e.g. a read-only install.
    """
    import os
    from pathlib import Path

    cache_dir = getattr(args, "cache_dir", None)
    if cache_dir is None:
        if os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE"):
            return  # respect the user's explicit environment
        cache_dir = _default_model_cache_dir()
    cache_path = Path(cache_dir)
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    os.environ["HF_HOME"] = str(cache_path)


def _is_disk_full_error(exc: OSError) -> bool:
    """True if ``exc`` is an out-of-space / over-quota filesystem error.

    Covers ``ENOSPC`` (disk full), ``EDQUOT`` (over quota — the usual HPC HOME
    case), and a message fallback for environments that don't surface a clean
    errno. Used to turn the HuggingFace model-download failure into an
    actionable "point HF_HOME at scratch" message instead of a raw traceback.
    """
    import errno

    quota_errnos = {errno.ENOSPC, errno.EDQUOT}
    if getattr(exc, "errno", None) in quota_errnos:
        return True
    message = str(exc).lower()
    return "quota exceeded" in message or "no space left" in message


def _count_csv_rows(csv_path: "Path") -> int:
    """Best-effort count of data rows in a CSV (excludes the header).

    Returns 0 if the file is missing or unreadable — this is only used for a
    human-readable progress summary, never for control flow.
    """
    try:
        with open(csv_path, encoding="utf-8") as fh:
            total = sum(1 for _ in fh)
    except OSError:
        return 0
    return max(0, total - 1)


def _resolve_output_dirname(args) -> str:
    """Name of the folder this run's outputs are written into.

    The folder holds ``predictions.csv`` and ``lrr_annotation_plots/``. Uses
    ``--output-name`` if given, otherwise a unique, timestamped default like
    ``output_2026-06-15_20-30-00`` so repeated runs don't clobber each other.
    """
    import datetime

    name = getattr(args, "output_name", None)
    if name:
        return name
    return "output_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _run_prepare(args, *, progress=None) -> int:
    """One-shot pipeline orchestrator behind the ``prepare`` subcommand.

    Mirrors the legacy ``prepare_input_data.sh`` + ``run_preparation_pipeline.sh``
    flow but in a single Python process so the user only invokes one command.

    Parameters
    ----------
    progress
        An existing :class:`mamp_ml.progress.PipelineProgress` to report into
        (passed by :func:`_run_predict` so prep + inference share one banner and
        total timer). When ``None`` (the standalone ``prepare`` command) this
        function owns the reporter and prints its own banner + closing summary.

    Returns
    -------
    int
        Process exit code: ``0`` if the full pipeline completed, ``2`` if
        ColabFold hadn't yet been run for the current input (in which case
        the receptor FASTA was still written and the user was shown the
        exact command to run next).
    """
    from pathlib import Path

    from mamp_ml.progress import PipelineProgress

    own_progress = progress is None
    if own_progress:
        progress = PipelineProgress(6)
        progress.banner(f"mamp-ml {getattr(args, 'cmd', 'prepare')}", _pipeline_subtitle(args))

    from mamp_ml.preprocess import (
        add_chemical_features,
        assemble_test_data,
        build_lrr_domain_fasta,
        write_truncated_fasta,
        xlsx_to_receptor_fasta,
    )
    from mamp_ml.structure import run_structure_stage
    from mamp_ml.lrr_features import write_bfactor_lrr_segments

    xlsx_path = Path(args.xlsx)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    colabfold_dir = (
        Path(args.colabfold_dir)
        if args.colabfold_dir
        else out_dir / "receptor_only"
    )
    colabfold_dir.mkdir(parents=True, exist_ok=True)

    structure_cache_dir = Path(args.structure_cache_dir)
    bfactor_cache_dir = (
        Path(args.bfactor_cache_dir)
        if args.bfactor_cache_dir
        else _resolve_default_bfactor_cache()
    )

    receptor_fasta = out_dir / "receptor_full_length.fasta"
    scores_path = out_dir / "alphafold_scores.txt"
    pdb_target_dir = out_dir / "pdb_for_lrr_annotator"
    lrr_results = out_dir / "lrr_annotation_results.txt"
    lrr_domain_fasta = out_dir / "lrr_domain_sequences.fasta"
    bfactor_csv = out_dir / "bfactor_winding_lrr_segments.csv"
    plot_dir = out_dir / "lrr_annotation_plots"
    test_data_csv = out_dir / "test_data.csv"
    ready_csv = out_dir / "ready_test_data.csv"

    # ---- Step 1/6: receptor FASTA ----
    step = progress.start("Receptor FASTA", estimate="<5s")
    n = xlsx_to_receptor_fasta(
        xlsx_path, receptor_fasta, sheet_name=args.sheet_name
    )
    step.done(f"{n} unique receptor record(s)", target=receptor_fasta)

    # ---- Folding phase (not one of the 6 numbered steps) ----
    # Both tools end up writing into the same colabfold_dir (the same PDB
    # filename convention + log.txt schema), so the gating logic only
    # differs in *what to do when the outputs are missing*: for colabfold we
    # run the discovered binary (or print a hint and exit), for esmfold we
    # auto-run the in-process tool.
    log_path = colabfold_dir / "log.txt"
    structure_tool = getattr(args, "structure", "colabfold")
    if not log_path.is_file():
        if structure_tool == "esmfold":
            fold = progress.start(
                "Fold receptors · ESMFold (in-process)",
                estimate=f"~1–3 min/receptor on GPU, ~20–40 min on CPU × {n}",
                numbered=False,
            )
            device = _resolve_torch_device(getattr(args, "device", None))
            from mamp_ml.fold.esmfold import fold_with_esmfold

            pdbs = fold_with_esmfold(
                receptor_fasta,
                colabfold_dir,
                device=device,
                chunk_size=getattr(args, "chunk_size", None),
            )
            fold.done(f"{len(pdbs)} PDB(s) + log.txt", target=colabfold_dir)
        else:
            from mamp_ml.fold.colabfold import (
                find_colabfold_installs,
                format_activation_hint,
                run_colabfold_batch,
            )

            # ColabFold/AlphaFold2 memory scales steeply with sequence length,
            # so very long receptors can OOM-kill colabfold_batch (status -9).
            # The LRR ectodomain is N-terminal, so we fold only the first
            # `max_length` residues. This is ColabFold-specific: ESMFold keeps
            # its own positional-embedding cap and folds the full FASTA. We
            # write the truncated copy to a separate file so the canonical
            # receptor_full_length.fasta (used for the downstream header
            # lookup) stays intact.
            colabfold_max_length = getattr(args, "max_length", None)
            fold_input_fasta = receptor_fasta
            if colabfold_max_length:
                fold_input_fasta = out_dir / "receptor_for_folding.fasta"
                n_records, n_trunc = write_truncated_fasta(
                    receptor_fasta, fold_input_fasta, colabfold_max_length
                )
                if n_trunc:
                    print(
                        f"  truncating {n_trunc}/{n_records} receptor(s) longer "
                        f"than {colabfold_max_length} aa to their N-terminal "
                        f"{colabfold_max_length} residues for folding "
                        f"(LRR ectodomain is N-terminal); full-length FASTA "
                        f"preserved at {receptor_fasta}"
                    )

            existing = find_colabfold_installs()
            if existing:
                # A reachable colabfold_batch was found -> just run it. The
                # binary is invoked by absolute path, so no `export PATH` /
                # `conda activate` is needed even when it lives in another env.
                binary, source = existing[0]
                fold = progress.start(
                    "Fold receptors · ColabFold (auto-detected)",
                    estimate=f"~2–10 min/receptor on GPU × {n}",
                    numbered=False,
                )
                # NB: keep the literal phrase below — tests assert on it.
                print(
                    "  running the discovered colabfold_batch automatically:"
                )
                fold.info(f"{binary}  ({source})")
                if len(existing) > 1:
                    fold.info(
                        f"{len(existing) - 1} other install(s) found; the "
                        "$PATH-preferred one above is used (run `mamp-ml "
                        "find-colabfold` to list them)"
                    )
                print()
                rc = run_colabfold_batch(
                    binary,
                    fold_input_fasta,
                    colabfold_dir,
                    num_models=1,
                    num_recycle=1,
                )
                if rc != 0:
                    fold.fail(f"ColabFold exited with status {rc}")
                    print(
                        f"ColabFold exited with status {rc} (see its output "
                        "above). Fix the error and re-run this command, or run "
                        "ColabFold manually:"
                    )
                    print(f"  {format_activation_hint(binary)}")
                    print(
                        f"  colabfold_batch --num-models 1 --num-recycle 1 \\\n"
                        f"      {fold_input_fasta} \\\n"
                        f"      {colabfold_dir}"
                    )
                    return 2
                if not log_path.is_file():
                    fold.fail("ColabFold wrote no log.txt")
                    print(
                        "ColabFold finished (status 0) but wrote no log.txt to "
                        f"{colabfold_dir}; cannot continue. Check ColabFold's "
                        "output above for warnings."
                    )
                    return 2
                # NB: keep the literal "ColabFold finished ->" — tests assert it.
                fold.done("ColabFold finished ->", target=colabfold_dir)
                # Fall through to the structure stage with the fresh outputs.
            else:
                print()
                print("ColabFold has not been run yet for this input.")
                print(
                    "Run ColabFold on the receptor FASTA above, then re-invoke "
                    "this command. Suggested invocation:"
                )
                print()
                print(
                    f"  colabfold_batch --num-models 1 --num-recycle 1 \\\n"
                    f"      {fold_input_fasta} \\\n"
                    f"      {colabfold_dir}"
                )
                print()
                print(
                    "(No existing colabfold_batch found. See "
                    "scripts/install_colabbatch_linux.sh or "
                    "scripts/install_colabbatch_mac.sh to install ColabFold "
                    "locally, or pass --structure esmfold to fold in-process. "
                    "Run `mamp-ml find-colabfold` later if you install or "
                    "load a ColabFold module.)"
                )
                return 2

    # ---- Step 2/6: structure stage (LRR annotation) ----
    step = progress.start("Structure analysis · LRR annotation", estimate="~15–90s")
    n_lrr = run_structure_stage(
        colabfold_dir,
        scores_path,
        pdb_target_dir,
        lrr_results,
        cache_dir=structure_cache_dir,
        plot_dir=plot_dir,
    )
    step.done(f"{n_lrr} LRR region row(s)", target=lrr_results)

    # ---- Step 3/6: LRR-domain FASTA ----
    step = progress.start("LRR-domain FASTA", estimate="<5s")
    n_dom = build_lrr_domain_fasta(lrr_results, receptor_fasta, lrr_domain_fasta)
    step.done(f"{n_dom} sequence(s)", target=lrr_domain_fasta)

    # ---- Step 4/6: B-factor analysis ----
    step = progress.start("B-factor winding analysis", estimate="~10–60s")
    bfactor_df = write_bfactor_lrr_segments(
        pdb_target_dir, bfactor_cache_dir, bfactor_csv
    )
    step.done(f"{len(bfactor_df)} row(s)", target=bfactor_csv)

    # ---- Step 5/6: test-data assembly ----
    step = progress.start("Test-data assembly", estimate="<10s")
    test_df = assemble_test_data(
        xlsx_path,
        lrr_domain_fasta,
        test_data_csv,
        sheet_name=args.sheet_name,
    )
    step.done(f"{len(test_df)} row(s)", target=test_data_csv)

    # ---- Step 6/6: chemical features ----
    step = progress.start("Chemical-feature annotation", estimate="<10s")
    ready_df = add_chemical_features(test_data_csv, ready_csv)
    step.done(f"{len(ready_df)} row(s)", target=ready_csv)

    if own_progress:
        progress.complete(
            "Preparation complete",
            outputs=[
                ("Model-ready CSV", ready_csv),
                ("LRR annotation plots", f"{plot_dir}/"),
                ("All intermediates", f"{out_dir}/"),
            ],
        )
    return 0


def _run_predict(args) -> int:
    """Implementation of ``python -m mamp_ml predict``.

    Two-stage flow:

    1. Delegate to :func:`_run_prepare` with the same data-pipeline args
       so the user gets the standard prepare-time output (one ``[N/6]``
       line per stage, ColabFold gating if needed). If prepare returns a
       non-zero exit code we propagate it without attempting inference.
    2. Hand off to :func:`mamp_ml.train.main` in eval-only mode, with the
       resolved checkpoint path and device. ``mamp_ml.train.main`` reuses
       the same architecture initialisation + model_dict the original
       legacy trainer used, so the inference call surface is unchanged
       (only its driver moved).

    Returns
    -------
    int
        Exit code: ``0`` on success, ``2`` if prepare hit the ColabFold
        gate (FASTA was written; user was shown next steps), or whatever
        ``mamp_ml.train.main`` returns (typically None, treated as 0).
    """
    from pathlib import Path

    # The directory the user invoked from — where the final deliverables land
    # under the default --keep mode (captured before any chdir).
    invocation_dir = Path.cwd()

    # Resolve the input spreadsheet: --example swaps in the bundled sample so
    # pip users can smoke-test their install with no data of their own. A bare
    # `predict` with neither a path nor --example is a usage error.
    if getattr(args, "example", False):
        from mamp_ml import example_data_path

        args.xlsx = str(example_data_path())
        print(f"Using bundled example dataset: {args.xlsx}")
    elif not args.xlsx:
        print(
            "Error: no input spreadsheet given. Pass a path to an .xlsx file, "
            "or use --example to run on the bundled sample dataset."
        )
        return 2

    # One shared progress reporter spans the 6 prep steps + the inference phase
    # so the banner and total timer cover the whole `predict` run.
    from mamp_ml.progress import PipelineProgress

    progress = PipelineProgress(6)
    progress.banner("mamp-ml predict", _pipeline_subtitle(args))

    # The prepare args parser declares a superset of what we need here, but
    # the same attribute names — so feed args directly into _run_prepare
    # without rebuilding a namespace.
    prepare_rc = _run_prepare(args, progress=progress)
    if prepare_rc != 0:
        return prepare_rc

    from mamp_ml import train
    from mamp_ml.weights import default_weights_path

    out_dir = Path(args.out_dir)
    ready_csv = out_dir / "ready_test_data.csv"
    weights = Path(args.weights) if args.weights else default_weights_path()
    if not weights.is_file():
        print(
            f"Error: model weights not found at {weights}. "
            "Pass --weights with an explicit path to a custom .pth file, or "
            "reinstall the package to restore the bundled weights.",
        )
        return 3
    if not ready_csv.is_file():
        # Should not happen — prepare returns 0 only after producing this file.
        print(f"Error: ready_test_data.csv missing at {ready_csv}.")
        return 4

    infer = progress.start(
        "Predict · ESM-2 inference",
        estimate="~30s–3 min on GPU, longer on CPU",
        numbered=False,
    )
    infer.info(f"weights: {weights}")

    eval_argv = [
        "--model", "esm2_bfactor_weighted",
        "--eval_only_data_path", str(ready_csv.resolve()),
        "--model_checkpoint_path", str(weights.resolve()),
        "--device", args.device,
        "--disable_wandb",
    ]
    train_args = train.get_args_parser().parse_args(eval_argv)

    # train.main() reads `args.output_dir` directly (the legacy __main__ block
    # set it after parsing; main() does not). We mirror that here so eval mode
    # has a place to write test_preds.pth.
    train_args.output_dir = out_dir.resolve()

    # The model's get_stats() writes 'predictions.csv' to the current working
    # directory (hardcoded relative path inside esm_positon_weighted.py). We
    # temporarily chdir into out_dir so the file lands alongside the other
    # intermediate artefacts; chdir is restored on the way out so the caller
    # sees no environmental change.
    import os

    prev_cwd = Path.cwd()
    try:
        os.chdir(out_dir)
        train.main(train_args)
    except OSError as exc:
        if _is_disk_full_error(exc):
            infer.fail("disk quota / no space while caching model weights")
            print()
            print(
                "ESM-2 weights are cached to disk and the target filesystem is "
                "out of space / over quota:"
            )
            print(f"  {exc}")
            print()
            print(
                "By default mamp-ml caches model weights next to its install "
                f"({_default_model_cache_dir()}). Point the cache at a "
                "filesystem with more room and re-run — either pass --cache-dir, "
                "or export HF_HOME:"
            )
            print()
            print(
                f"  mamp-ml predict {args.xlsx} --device {args.device} "
                "--cache-dir /path/with/room"
            )
            print("  # or, once for the whole session:")
            print("  export HF_HOME=/path/with/room")
            return 5
        raise
    finally:
        os.chdir(prev_cwd)

    predictions_csv = out_dir / "predictions.csv"
    plots_dir = out_dir / "lrr_annotation_plots"
    infer.done(f"{_count_csv_rows(predictions_csv)} prediction(s)", target=predictions_csv)

    # --keep default: the two real deliverables (predictions.csv + the LRR
    # plots) are promoted to the invocation directory and the whole
    # intermediate_files/ scratch dir is removed, so the user's outputs aren't
    # buried under a folder named "intermediate_files".
    # --keep all: leave everything in out_dir untouched (useful for debugging
    # or re-running prediction on a different ligand sheet without re-folding).
    # Always bundle the deliverables into a labeled output folder in the
    # invocation dir: <name>/predictions.csv + <name>/lrr_annotation_plots/.
    # --keep then only governs whether the intermediate_files/ working dir
    # survives (default: removed; all: kept for debugging / re-running without
    # re-folding).
    results_dir = invocation_dir / _resolve_output_dirname(args)
    final_predictions = _promote_output(
        predictions_csv, results_dir / "predictions.csv"
    )
    final_plots = _promote_output(plots_dir, results_dir / "lrr_annotation_plots")
    outputs = [
        ("Output folder", f"{results_dir}/"),
        ("  predictions", (final_predictions or predictions_csv).name),
        ("  LRR annotation plots", f"{(final_plots or plots_dir).name}/"),
    ]

    keep_mode = getattr(args, "keep", "default")
    if keep_mode == "all":
        outputs.append(("Intermediates kept", f"{out_dir}/"))
    else:
        # Remove the intermediate_files working directory now that the
        # deliverables are out of it — unless --out-dir points at the invocation
        # dir or the results folder itself (don't delete what we just wrote).
        protected = {invocation_dir.resolve(), results_dir.resolve()}
        if out_dir.resolve() not in protected:
            import shutil

            shutil.rmtree(out_dir, ignore_errors=True)
        outputs.append(
            ("(intermediates removed", "rerun with `--keep all` to retain them)")
        )
    progress.complete("Prediction complete", outputs=outputs)
    return 0


def _promote_output(src: "Path", dest: "Path") -> "Path | None":
    """Move a final deliverable ``src`` to ``dest``, replacing ``dest`` if present.

    Used to lift ``predictions.csv`` and the ``lrr_annotation_plots/`` dir out
    of the scratch ``intermediate_files/`` directory and into the directory the
    user invoked from, so the real outputs aren't buried under a folder named
    "intermediate_files".

    Returns the destination path on success, the source path if it is already
    at the destination, or ``None`` if ``src`` doesn't exist (e.g. the model
    didn't write predictions for some reason — we don't want the relocation to
    hard-fail in that case).
    """
    import shutil
    from pathlib import Path

    src = Path(src)
    dest = Path(dest)
    if not src.exists():
        return None
    try:
        if src.resolve() == dest.resolve():
            return src  # already in place (e.g. --out-dir is the invocation dir)
    except (OSError, RuntimeError):
        pass
    if dest.exists():
        if dest.is_dir() and not dest.is_symlink():
            shutil.rmtree(dest, ignore_errors=True)
        else:
            try:
                dest.unlink()
            except OSError:
                pass
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    return dest


def _resolve_torch_device(device_arg: Optional[str]) -> str:
    """Pick a torch device for ESMFold given the user's --device hint.

    If the user passed something explicit, use it as-is. Otherwise prefer
    cuda > mps > cpu so users on a GPU host or an Apple Silicon laptop get
    the fast path automatically.
    """
    if device_arg:
        return device_arg
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()  # type: ignore[attr-defined]
        ):
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _run_find_colabfold(args) -> int:
    """Implementation of ``python -m mamp_ml find-colabfold``.

    Lists every reachable ``colabfold_batch`` install. Returns 0 when at
    least one was found, 1 when the search came up empty. Cluster users
    typically have ColabFold installed somewhere but don't remember
    where; this avoids them having to guess.
    """
    from mamp_ml.fold.colabfold import (
        find_colabfold_installs,
        format_activation_hint,
    )

    found = find_colabfold_installs()
    if not found:
        print("No `colabfold_batch` install found on this system.")
        print()
        print("Common installs:")
        print("  bash scripts/install_colabbatch_linux.sh   # Linux + NVIDIA GPU")
        print("  bash scripts/install_colabbatch_mac.sh     # macOS (dev only)")
        print()
        print(
            "If your cluster provides ColabFold as a module, run "
            "`module load colabfold` (or equivalent) and re-run `mamp-ml "
            "find-colabfold` — it'll then appear on $PATH."
        )
        return 1

    print(f"Found {len(found)} `colabfold_batch` install(s):")
    print()
    for path, source in found:
        print(f"  {path}")
        print(f"    source : {source}")
        print(f"    activate: {format_activation_hint(path)}")
        print()
    return 0


def _run_example(args) -> int:
    """Implementation of ``python -m mamp_ml example``.

    Copies the bundled ``example_data.xlsx`` into ``--out-dir`` (cwd by
    default) so a pip-installed user can try the pipeline without cloning the
    repo, or — with ``--path`` — just prints where the bundled file lives.

    Returns
    -------
    int
        ``0`` on success (file copied, already present, or path printed),
        ``1`` if the target already exists and ``--force`` was not given,
        ``3`` if the bundled sample is missing (a broken install).
    """
    import shutil
    from pathlib import Path

    from mamp_ml import example_data_path

    try:
        source = example_data_path()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 3

    if args.path:
        # Scripting hook: `mamp-ml predict "$(mamp-ml example --path)"`.
        print(source)
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / source.name

    if target.exists() and not args.force:
        print(
            f"{target} already exists; not overwriting. "
            "Pass --force to replace it, or --path to print the bundled "
            "sample's location instead."
        )
        return 1

    shutil.copyfile(source, target)
    print(f"Wrote {source.name} to {target}")
    print()
    print("Try the pipeline on it with:")
    print(f"  mamp-ml predict {target} --device cuda")
    return 0


def _run_fold(args) -> int:
    """Implementation of ``python -m mamp_ml fold``.

    Standalone wrapper around either the ESMFold backend or a ColabFold
    invocation hint. Useful when the user wants to fold a custom FASTA
    without running the rest of the preparation pipeline.
    """
    from pathlib import Path

    fasta = Path(args.fasta)
    output_dir = Path(args.output_dir)

    if not fasta.is_file():
        print(f"Error: input FASTA not found: {fasta}")
        return 3

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.structure == "esmfold":
        from mamp_ml.fold.esmfold import fold_with_esmfold

        device = _resolve_torch_device(args.device)
        print(f"Folding {fasta} with ESMFold on device='{device}' ...")
        pdbs = fold_with_esmfold(
            fasta,
            output_dir,
            device=device,
            max_length=args.max_length,
            chunk_size=getattr(args, "chunk_size", None),
        )
        print(
            f"Wrote {len(pdbs)} PDB(s) + log.txt to {output_dir}/"
        )
        return 0

    # colabfold path: print the recommended invocation.
    print(
        "To fold with ColabFold, run the command below in an environment "
        "where colabfold_batch is on $PATH:"
    )
    print()
    print(
        f"  colabfold_batch --num-models 1 --num-recycle 1 \\\n"
        f"      {fasta} \\\n"
        f"      {output_dir}"
    )
    return 2


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for ``python -m mamp_ml``.

    Returns the process exit code (0 on success). Subcommand errors raise as
    usual; this wrapper only handles dispatch.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Steer the HuggingFace model cache to a writable location before any of
    # the model-loading subcommands import transformers / huggingface_hub.
    if args.cmd in ("predict", "prepare", "fold"):
        _configure_model_cache(args)

    if args.cmd == "prepare":
        return _run_prepare(args)

    if args.cmd == "predict":
        return _run_predict(args)

    if args.cmd == "fold":
        return _run_fold(args)

    if args.cmd == "find-colabfold":
        return _run_find_colabfold(args)

    if args.cmd == "example":
        return _run_example(args)

    if args.cmd == "prepare-fasta":
        from mamp_ml.preprocess import xlsx_to_receptor_fasta

        n = xlsx_to_receptor_fasta(
            args.xlsx, args.out_fasta, sheet_name=args.sheet_name
        )
        print(f"prepare-fasta: wrote {n} records to {args.out_fasta}")
        return 0

    if args.cmd == "structure-stage":
        from mamp_ml.structure import run_structure_stage

        n = run_structure_stage(
            args.colabfold_dir,
            args.scores_path,
            args.pdb_target_dir,
            args.lrr_results_path,
            cache_dir=args.cache_dir,
            plot_dir=args.plot_dir,
        )
        print(
            f"structure-stage: wrote {n} LRR-region rows to {args.lrr_results_path}"
        )
        return 0

    if args.cmd == "lrr-domain-fasta":
        from mamp_ml.preprocess import build_lrr_domain_fasta

        n = build_lrr_domain_fasta(
            args.lrr_annotation_results,
            args.receptor_full_length_fasta,
            args.out_fasta,
        )
        print(f"lrr-domain-fasta: wrote {n} records to {args.out_fasta}")
        return 0

    if args.cmd == "bfactor":
        from mamp_ml.lrr_features import write_bfactor_lrr_segments

        df = write_bfactor_lrr_segments(args.pdb_dir, args.cache_dir, args.out_csv)
        print(f"bfactor: wrote {len(df)} rows to {args.out_csv}")
        return 0

    if args.cmd == "assemble-test-data":
        from mamp_ml.preprocess import assemble_test_data

        df = assemble_test_data(
            args.xlsx,
            args.lrr_domain_fasta,
            args.out_csv,
            sheet_name=args.sheet_name,
        )
        print(f"assemble-test-data: wrote {len(df)} rows to {args.out_csv}")
        return 0

    if args.cmd == "chemical-features":
        from mamp_ml.preprocess import add_chemical_features

        df = add_chemical_features(args.in_csv, args.out_csv)
        print(f"chemical-features: wrote {len(df)} rows to {args.out_csv}")
        return 0

    # argparse with required=True should make this unreachable.
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
