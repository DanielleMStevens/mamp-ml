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
import shlex
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

  # Preflight the install (PyTorch/GPU compatibility, ColabFold, weights)
  mamp-ml install-check

  # Smoke-test your install on the bundled sample (no data of your own needed)
  mamp-ml predict --example --device cuda

  # Full pipeline: spreadsheet -> predictions.csv
  mamp-ml predict input_data.xlsx --device cuda

  # Use ESMFold instead of ColabFold (no separate conda env needed)
  mamp-ml predict input_data.xlsx --structure esmfold --device cuda

  # Use a custom-trained model instead of the bundled weights
  mamp-ml predict input_data.xlsx --weights /path/to/checkpoint.pth

  # Keep every intermediate file (default keeps only the output folder)
  mamp-ml predict input_data.xlsx --keep all

  # Reuse a previous run's folds to try different weights (skips folding)
  mamp-ml predict input_data.xlsx \
      --structures output_input_data_.../structures --weights other.pth

Output
------
  predict writes an `<output-name>/` folder (default: a unique per-run
  `output_<input>_<timestamp-or-SLURM-job>_<random>/`) containing:
    - predictions.csv      one row per receptor-ligand pair, with the predicted
                           class (Immunogenic / Non-Immunogenic / Weakly
                           Immunogenic) and the per-class probabilities
    - lrr_annotation_plots/  per-receptor LRR regression plots
    - structures/          the folded receptor structures (PDBs + log.txt);
                           pass this folder to --structures on a later run to
                           skip folding
    - mamp-ml-run.log      full run transcript (command, version, every step,
                           and the complete ColabFold/ESMFold output) — attach
                           this when reporting an error

Concurrency: each run uses a unique working dir (intermediate_files/<token>/)
and a unique output folder, so multiple runs from the same directory — e.g.
SLURM array jobs — never collide. Works the same off-cluster; no SLURM needed.
Pass --out-dir / --output-name to pin explicit names.

The folding step shows a compact per-receptor progress bar; the backend's
verbose output is written to mamp-ml-run.log instead of the terminal.

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
        default=None,
        help=(
            "Directory for pipeline intermediates and the final "
            "ready_test_data.csv. Defaults to a unique per-run directory "
            "intermediate_files/<input>_<timestamp-or-SLURM-job>_<random> so "
            "concurrent runs (e.g. SLURM array jobs) never clobber each other; "
            "pass an explicit path to override."
        ),
    )
    sp.add_argument(
        "--structures",
        default=None,
        help=(
            "Reuse existing folded structures from this directory (the "
            "structures/ folder a previous run produced) and SKIP the "
            "structural-modeling stage. Handy for re-running with different "
            "--weights without re-folding."
        ),
    )
    # Back-compat alias for --structures; hidden from help to avoid two names
    # for the same thing.
    sp.add_argument(
        "--colabfold-dir",
        default=None,
        help=argparse.SUPPRESS,
    )
    sp.add_argument(
        "--structure-cache-dir",
        default=None,
        help=(
            "Directory for the structure-stage geometry + breakpoints pickles. "
            "Defaults to a per-run <out-dir>/lrr_cache/ so the B-factor stage "
            "reads the breakpoints this run actually computed (the breakpoints "
            "are an output of the LRR-annotation stage)."
        ),
    )
    sp.add_argument(
        "--bfactor-cache-dir",
        default=None,
        help=(
            "Directory the B-factor stage reads breakpoints from. Defaults to "
            "the structure-cache-dir for this run (i.e. the freshly-computed "
            "breakpoints). Point this at the shipped production cache "
            "(src/mamp_ml/lrr_annotation/cache) only to reproduce the original "
            "training set."
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

    # install-check ----------------------------------------------------
    sp = sub.add_parser(
        "install-check",
        help="Verify the install can run (PyTorch/GPU, ColabFold, weights).",
        description=(
            "Preflight the install before submitting a long job: detects the GPU "
            "via nvidia-smi, reports the mamp-ml version and whether the installed "
            "PyTorch can actually drive that GPU (catching the V100/sm_70 'no "
            "kernel image' trap before it crashes mid-run), runs a live CUDA op, "
            "checks ColabFold discovery, and confirms the bundled weights. When "
            "PyTorch can't use the GPU it prints the exact `pip install torch` "
            "line matching the driver's CUDA version."
        ),
    )
    sp.add_argument(
        "--install-torch",
        action="store_true",
        help=(
            "If the installed PyTorch can't drive the detected GPU, run the "
            "recommended `pip install torch --index-url …` automatically instead "
            "of only printing it."
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
        default=None,
        help=(
            "Directory for pipeline intermediates. Defaults to a unique "
            "per-run directory intermediate_files/<input>_<timestamp-or-SLURM-"
            "job>_<random> so concurrent runs (e.g. SLURM array jobs) never "
            "clobber each other; pass an explicit path to override."
        ),
    )
    sp.add_argument(
        "--structures",
        default=None,
        help=(
            "Reuse existing folded structures from this directory (the "
            "structures/ folder a previous run produced) and SKIP the "
            "structural-modeling stage. Lets you re-predict with different "
            "--weights without re-running ColabFold/ESMFold."
        ),
    )
    # Back-compat alias for --structures; hidden from help.
    sp.add_argument(
        "--colabfold-dir",
        default=None,
        help=argparse.SUPPRESS,
    )
    sp.add_argument(
        "--structure-cache-dir",
        default=None,
        help=(
            "Directory for the structure-stage geometry + breakpoints pickles "
            "(default: a per-run <out-dir>/lrr_cache/). The B-factor stage reads "
            "this run's breakpoints from here."
        ),
    )
    sp.add_argument(
        "--bfactor-cache-dir",
        default=None,
        help=(
            "Directory the B-factor stage reads breakpoints from (default: the "
            "structure-cache-dir for this run). Point at the shipped production "
            "cache only to reproduce the original training set."
        ),
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


def _sanitize_token_part(text: str, *, max_length: int = 40) -> str:
    """Make ``text`` safe + tidy for use inside a directory name.

    Keeps alphanumerics, dot, dash and underscore; collapses every other run of
    characters to a single underscore; trims leading/trailing separators; and
    caps the length so a long spreadsheet name doesn't produce an unwieldy
    folder. Falls back to ``"run"`` if nothing usable remains.
    """
    import re

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    cleaned = cleaned[:max_length].strip("._-")
    return cleaned or "run"


def _run_discriminator() -> str:
    """A human-readable, per-run discriminator that needs no scheduler.

    Prefers SLURM identifiers when present (so a folder maps straight back to a
    job in ``sacct``), and otherwise falls back to a wall-clock timestamp. This
    is only the *readable* part of the run token — uniqueness is guaranteed by
    the random suffix appended in :func:`_make_run_token`, so this works
    identically on a laptop, a workstation, the cloud, or an HPC node.
    """
    import datetime
    import os

    array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_job and array_task:
        return f"job{array_job}_{array_task}"
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        return f"job{job}"
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _make_run_token(args) -> str:
    """Build the shared, human-readable run token (without creating anything).

    Form: ``<input-stem>_<slurm-id-or-timestamp>_<4 random hex>`` — e.g.
    ``Cabernet_filtered_RLKs_2026-06-22_14-03-01_a4f9`` locally or
    ``Cabernet_filtered_RLKs_job4471823_a4f9`` under SLURM. The working
    directory and the final output folder share this token so they correspond,
    and the random suffix makes the name unique on any setup with no config.
    """
    import secrets
    from pathlib import Path

    xlsx = getattr(args, "xlsx", None)
    stem = _sanitize_token_part(Path(xlsx).stem) if xlsx else "run"
    return f"{stem}_{_run_discriminator()}_{secrets.token_hex(2)}"


def _resolve_working_dir(args):
    """Resolve (and create) this run's working directory, uniquely by default.

    If ``--out-dir`` was given explicitly, it is honoured verbatim. Otherwise a
    unique per-run directory ``intermediate_files/<run-token>`` is created
    atomically (``mkdir`` with ``exist_ok=False``, retrying with a fresh random
    suffix on the astronomically rare collision), so concurrent runs — e.g.
    SLURM array tasks launched from the same directory — never share a working
    dir and can't clobber each other or pick up each other's folds.

    Idempotent: once resolved, ``args.out_dir`` is set, so a second call (the
    standalone ``prepare`` path re-invoking after ``predict`` already resolved
    it) returns the same directory. The chosen run token is cached on
    ``args._run_token`` so the output folder can share it.
    """
    import os
    from pathlib import Path

    existing = getattr(args, "out_dir", None)
    if existing:
        out_dir = Path(existing)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    parent = Path("intermediate_files")
    last_exc: "OSError | None" = None
    for _ in range(64):
        token = _make_run_token(args)
        candidate = parent / token
        try:
            os.makedirs(candidate, exist_ok=False)
        except FileExistsError as exc:
            last_exc = exc
            continue
        args._run_token = token
        args.out_dir = str(candidate)
        return candidate
    raise RuntimeError(
        f"Could not create a unique working directory under {parent}/"
    ) from last_exc


def _resolve_output_dirname(args) -> str:
    """Name of the folder this run's outputs are written into.

    The folder holds ``predictions.csv``, ``lrr_annotation_plots/``,
    ``structures/`` and ``mamp-ml-run.log``. Uses ``--output-name`` if given,
    otherwise ``output_<run-token>`` — sharing the token with the working
    directory so the two correspond, and inheriting its per-run uniqueness so
    concurrent runs never collide on the output folder either.
    """
    name = getattr(args, "output_name", None)
    if name:
        return name
    token = getattr(args, "_run_token", None)
    if not token:
        # Standalone / direct call without a resolved working dir: mint a token
        # on the spot so the name is still unique.
        token = _make_run_token(args)
    return f"output_{token}"


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

    from mamp_ml.progress import FoldProgressBar, PipelineProgress, RunLogger

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
    # Resolve (and create) a unique per-run working directory unless the user
    # gave an explicit --out-dir. Idempotent: predict resolves this before
    # calling us, so this just returns the already-created dir.
    out_dir = _resolve_working_dir(args)

    # Open the run log inside the (always-created) working directory so it
    # survives even an early hard failure like an OOM-killed ColabFold; it is
    # copied into the final output folder on success. Everything the reporter
    # emits, plus the full fold-backend output, is teed here.
    from mamp_ml import __version__ as _mamp_version

    log = RunLogger(
        out_dir / "mamp-ml-run.log",
        command=getattr(args, "command_line", "mamp-ml"),
        version=_mamp_version,
        context=[
            ("input", str(xlsx_path)),
            ("device", str(getattr(args, "device", "") or "")),
            ("structure", str(getattr(args, "structure", "") or "")),
            ("out_dir", str(out_dir)),
        ],
    )
    progress.attach_logger(log)

    # Reuse pre-folded structures when the user points us at them (--structures,
    # or the hidden back-compat --colabfold-dir): we then skip the structural-
    # modeling stage entirely. Otherwise structures live under this run's
    # working dir and are folded fresh.
    reuse_dir = getattr(args, "structures", None) or getattr(args, "colabfold_dir", None)
    colabfold_dir = Path(reuse_dir) if reuse_dir else out_dir / "receptor_only"
    colabfold_dir.mkdir(parents=True, exist_ok=True)

    # The breakpoints the B-factor stage needs are an OUTPUT of the structure
    # (LRR-annotation) stage, so by default both share a per-run cache inside
    # this run's working dir. This makes the B-factor weighting work for novel
    # receptors (the shipped cache only holds the original training set) and is
    # collision-safe across concurrent runs. Explicit overrides still win;
    # point --bfactor-cache-dir at the shipped cache only to reproduce training.
    structure_cache_dir = (
        Path(args.structure_cache_dir)
        if getattr(args, "structure_cache_dir", None)
        else out_dir / "lrr_cache"
    )
    bfactor_cache_dir = (
        Path(args.bfactor_cache_dir)
        if getattr(args, "bfactor_cache_dir", None)
        else structure_cache_dir
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

            bar = FoldProgressBar(n, stream=sys.stdout, label="folding")

            def _esm_progress(idx: int, total: int, name: str) -> None:
                bar.update(idx, total=total, label=name)

            pdbs = fold_with_esmfold(
                receptor_fasta,
                colabfold_dir,
                device=device,
                chunk_size=getattr(args, "chunk_size", None),
                on_progress=_esm_progress,
                log=log.write_line,
            )
            bar.finish()
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
                    progress.note(
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
                progress.note(
                    "  running the discovered colabfold_batch automatically:"
                )
                fold.info(f"{binary}  ({source})")
                if len(existing) > 1:
                    fold.info(
                        f"{len(existing) - 1} other install(s) found; the "
                        "$PATH-preferred one above is used (run `mamp-ml "
                        "find-colabfold` to list them)"
                    )
                progress.note(
                    f"  full ColabFold output is being written to {log.path}"
                )

                # Stream ColabFold's (very verbose) output into the run log and
                # show only a compact per-receptor bar on the terminal. The bar
                # is driven by ColabFold's own "Query i/N: <name> (length L)"
                # lines; the total is seeded from the receptor count.
                from mamp_ml.fold.colabfold import COLABFOLD_QUERY_RE

                bar = FoldProgressBar(n, stream=sys.stdout, label="folding")

                def _colabfold_line(line: str) -> None:
                    log.write_line(line)
                    m = COLABFOLD_QUERY_RE.search(line)
                    if m:
                        bar.update(
                            int(m.group("i")),
                            total=int(m.group("n")),
                            label=f"{m.group('name')} ({m.group('length')} aa)",
                        )

                rc = run_colabfold_batch(
                    binary,
                    fold_input_fasta,
                    colabfold_dir,
                    num_models=1,
                    num_recycle=1,
                    on_line=_colabfold_line,
                )
                bar.finish()
                if rc != 0:
                    fold.fail(f"ColabFold exited with status {rc}")
                    progress.note(
                        f"ColabFold exited with status {rc} (full output is in "
                        f"the run log: {log.path}). A status of -9 means the OS "
                        "OOM-killed it — lower --max-length and re-run. Fix the "
                        "error and re-run this command, or run ColabFold "
                        "manually:"
                    )
                    progress.note(f"  {format_activation_hint(binary)}")
                    progress.note(
                        f"  colabfold_batch --num-models 1 --num-recycle 1 \\\n"
                        f"      {fold_input_fasta} \\\n"
                        f"      {colabfold_dir}"
                    )
                    return 2
                if not log_path.is_file():
                    fold.fail("ColabFold wrote no log.txt")
                    progress.note(
                        "ColabFold finished (status 0) but wrote no log.txt to "
                        f"{colabfold_dir}; cannot continue. Check the run log "
                        f"({log.path}) for warnings."
                    )
                    return 2
                # NB: keep the literal "ColabFold finished ->" — tests assert it.
                fold.done("ColabFold finished ->", target=colabfold_dir)
                # Fall through to the structure stage with the fresh outputs.
            else:
                progress.note("")
                progress.note("ColabFold has not been run yet for this input.")
                progress.note(
                    "Run ColabFold on the receptor FASTA above, then re-invoke "
                    "this command. Suggested invocation:"
                )
                progress.note("")
                progress.note(
                    f"  colabfold_batch --num-models 1 --num-recycle 1 \\\n"
                    f"      {fold_input_fasta} \\\n"
                    f"      {colabfold_dir}"
                )
                progress.note("")
                progress.note(
                    "(No existing colabfold_batch found. See "
                    "scripts/install_colabbatch_linux.sh or "
                    "scripts/install_colabbatch_mac.sh to install ColabFold "
                    "locally, or pass --structure esmfold to fold in-process. "
                    "Run `mamp-ml find-colabfold` later if you install or "
                    "load a ColabFold module.)"
                )
                log.close()
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
    if len(bfactor_df) == 0:
        # No rows means the model will fall back to UNIFORM (default) weights —
        # the structure-derived B-factor weighting is silently lost. Surface it
        # loudly; the usual cause is a breakpoints cache that doesn't cover
        # these receptors (e.g. --bfactor-cache-dir pointing at the shipped
        # training cache instead of this run's freshly-computed breakpoints).
        progress.note(
            "  ⚠ B-factor analysis produced 0 rows — the model will use "
            "UNIFORM weights, not the structure-derived B-factor weighting. "
            f"Check that {bfactor_cache_dir}/breakpoints.pickle covers these "
            "receptors (by default it is this run's own lrr_cache/)."
        )

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
                ("Folded structures", f"{colabfold_dir}/"),
                ("All intermediates", f"{out_dir}/"),
                ("Run log", log.path),
            ],
        )
        progress.note(
            f"  reuse these structures later with: --structures {colabfold_dir}"
        )
        log.close()
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

    # Resolve the unique per-run working directory up front (before prepare, and
    # before computing the output-folder name) so the working dir and the output
    # folder share one run token and concurrent runs never collide.
    _resolve_working_dir(args)

    # The prepare args parser declares a superset of what we need here, but
    # the same attribute names — so feed args directly into _run_prepare
    # without rebuilding a namespace.
    prepare_rc = _run_prepare(args, progress=progress)
    if prepare_rc != 0:
        # prepare hit the ColabFold gate / a fold failure. The run log lives in
        # the (retained) intermediate dir for the error report; flush + close.
        if progress.logger is not None:
            progress.logger.close()
        return prepare_rc

    from mamp_ml import train
    from mamp_ml.weights import default_weights_path

    out_dir = Path(args.out_dir)
    ready_csv = out_dir / "ready_test_data.csv"
    weights = Path(args.weights) if args.weights else default_weights_path()
    if not weights.is_file():
        progress.note(
            f"Error: model weights not found at {weights}. "
            "Pass --weights with an explicit path to a custom .pth file, or "
            "reinstall the package to restore the bundled weights."
        )
        if progress.logger is not None:
            progress.logger.close()
        return 3
    if not ready_csv.is_file():
        # Should not happen — prepare returns 0 only after producing this file.
        progress.note(f"Error: ready_test_data.csv missing at {ready_csv}.")
        if progress.logger is not None:
            progress.logger.close()
        return 4

    # Validate the requested device against the installed PyTorch BEFORE the
    # expensive inference: a torch build without kernels for this GPU's compute
    # capability otherwise crashes deep in the forward pass
    # (cudaErrorNoKernelImageForDevice). Fall back to CPU with a clear warning
    # rather than dying minutes in.
    device = _resolve_inference_device(getattr(args, "device", "cpu"), progress)

    infer = progress.start(
        "Predict · ESM-2 inference",
        estimate="~30s–3 min on GPU, longer on CPU",
        numbered=False,
    )
    infer.info(f"weights: {weights}")

    bfactor_csv = out_dir / "bfactor_winding_lrr_segments.csv"
    eval_argv = [
        "--model", "esm2_bfactor_weighted",
        "--eval_only_data_path", str(ready_csv.resolve()),
        "--model_checkpoint_path", str(weights.resolve()),
        "--device", device,
        "--disable_wandb",
    ]
    train_args = train.get_args_parser().parse_args(eval_argv)

    # Tell the model exactly where this run's B-factor CSV is. The model's own
    # fallback search uses a relative path that breaks once we chdir into
    # out_dir, so without this the structure-derived weighting is silently lost.
    train_args.bfactor_csv_path = str(bfactor_csv.resolve()) if bfactor_csv.is_file() else None

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
            progress.note("")
            progress.note(
                "ESM-2 weights are cached to disk and the target filesystem is "
                "out of space / over quota:"
            )
            progress.note(f"  {exc}")
            progress.note("")
            progress.note(
                "By default mamp-ml caches model weights next to its install "
                f"({_default_model_cache_dir()}). Point the cache at a "
                "filesystem with more room and re-run — either pass --cache-dir, "
                "or export HF_HOME:"
            )
            progress.note("")
            progress.note(
                f"  mamp-ml predict {args.xlsx} --device {args.device} "
                "--cache-dir /path/with/room"
            )
            progress.note("  # or, once for the whole session:")
            progress.note("  export HF_HOME=/path/with/room")
            if progress.logger is not None:
                progress.logger.close()
            return 5
        raise
    finally:
        os.chdir(prev_cwd)

    predictions_csv = out_dir / "predictions.csv"
    plots_dir = out_dir / "lrr_annotation_plots"
    infer.done(f"{_count_csv_rows(predictions_csv)} prediction(s)", target=predictions_csv)

    # Always bundle the deliverables into a labeled output folder in the
    # invocation dir. The folder is self-contained: predictions.csv, the LRR
    # plots, the folded structures/ (so they can be reused with other weights
    # without re-folding), and the full run log. `--keep` then governs only
    # whether the *remaining* working-dir scratch survives:
    #   default  -> structures + log promoted, the rest of the working dir removed
    #   all      -> additionally retain the working dir (fasta, csvs, pdbs) for
    #               debugging / re-running on a different ligand sheet.
    results_dir = invocation_dir / _resolve_output_dirname(args)
    final_predictions = _promote_output(
        predictions_csv, results_dir / "predictions.csv"
    )
    final_plots = _promote_output(plots_dir, results_dir / "lrr_annotation_plots")

    # Promote the folded structures unless they were *reused* from an external
    # directory the user pointed at (in which case they already live there and
    # we must not move/destroy that source).
    reuse_dir = getattr(args, "structures", None) or getattr(args, "colabfold_dir", None)
    structures_out = results_dir / "structures"
    if reuse_dir:
        final_structures = Path(reuse_dir)
    else:
        final_structures = _promote_output(out_dir / "receptor_only", structures_out)

    log_dest = results_dir / "mamp-ml-run.log"
    outputs = [
        ("Output folder", f"{results_dir}/"),
        ("  predictions", (final_predictions or predictions_csv).name),
        ("  LRR annotation plots", f"{(final_plots or plots_dir).name}/"),
    ]
    if final_structures is not None and not reuse_dir:
        outputs.append(("  structures", f"{structures_out.name}/"))
    outputs.append(("  run log", log_dest.name))

    keep_mode = getattr(args, "keep", "default")
    if keep_mode == "all":
        outputs.append(("Intermediates kept", f"{out_dir}/"))
    else:
        outputs.append(
            ("(remaining intermediates removed", "rerun with `--keep all` to retain them)")
        )
    progress.complete("Prediction complete", outputs=outputs)

    # Surface the one-liner that reuses these structures with different weights
    # (skips the structural-modeling stage). Points at the promoted copy when we
    # saved one, else at the directory the user reused.
    reuse_target = structures_out if final_structures is not None and not reuse_dir else final_structures
    if reuse_target is not None:
        progress.note(
            "  reuse these structures with different weights (skips folding):"
        )
        progress.note(
            f"    mamp-ml predict {args.xlsx} --structures {reuse_target} "
            "--weights <other.pth>"
        )

    # Finalise the run log: flush the closing summary, then copy it into the
    # final output folder so the deliverables are self-describing (command +
    # version + full transcript). Do this *before* removing the working dir.
    import shutil

    if progress.logger is not None:
        progress.logger.close()
    src_log = out_dir / "mamp-ml-run.log"
    try:
        if src_log.is_file() and src_log.resolve() != log_dest.resolve():
            log_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_log, log_dest)
    except OSError:
        pass

    if keep_mode != "all":
        # Remove the working directory now that the deliverables (structures +
        # run log included) are out of it — unless --out-dir points at the
        # invocation dir or the results folder itself (don't delete what we just
        # wrote). Then tidy up the now-empty intermediate_files/ parent so the
        # cwd stays clean.
        protected = {invocation_dir.resolve(), results_dir.resolve()}
        if out_dir.resolve() not in protected:
            shutil.rmtree(out_dir, ignore_errors=True)
            parent = out_dir.parent
            try:
                if parent.name == "intermediate_files" and not any(parent.iterdir()):
                    parent.rmdir()
            except OSError:
                pass
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


def _gpu_compatibility_problem(device_str: "str | None") -> "str | None":
    """Why the requested CUDA device can't be used by the installed torch.

    Returns a human-readable reason, or ``None`` if the device is fine (or isn't
    a CUDA device). Catches the common cluster failure where a PyTorch build
    ships no kernels for the GPU's compute capability — which otherwise crashes
    deep in the forward pass with ``cudaErrorNoKernelImageForDevice`` (e.g. a
    Tesla V100, CC 7.0 / sm_70, against a torch built only for sm_75+).
    """
    if not device_str or not str(device_str).startswith("cuda"):
        return None
    try:
        import torch
    except ImportError:
        return "PyTorch is not installed"
    if not torch.cuda.is_available():
        return (
            "torch.cuda.is_available() is False (no CUDA driver/runtime visible, "
            "or this is a CPU-only PyTorch build)"
        )
    try:
        major, minor = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        arches = list(torch.cuda.get_arch_list())  # e.g. ['sm_75', 'sm_80', ...]
        sm = f"sm_{major}{minor}"
        if arches and sm not in arches:
            return (
                f"GPU '{name}' is compute capability {major}.{minor} ({sm}), but "
                f"this PyTorch was built only for [{', '.join(arches)}]. Install "
                "a torch build that includes this GPU's architecture."
            )
    except Exception as exc:  # pragma: no cover - exotic driver states
        return f"could not query the GPU ({exc})"
    return None


def _resolve_inference_device(requested: "str | None", progress=None) -> str:
    """Return a device that the installed torch can actually run on.

    If the requested CUDA device is unusable, fall back to CPU with a clear
    warning rather than crashing minutes into inference. The bundled model is
    small, so CPU is an acceptable default.
    """
    problem = _gpu_compatibility_problem(requested)
    if problem is None:
        return requested or "cpu"
    msg = (
        f"requested --device {requested}, but it is unusable: {problem} "
        "Falling back to --device cpu. Run `mamp-ml install-check` to diagnose, "
        "and the README's install section for a GPU-compatible PyTorch."
    )
    if progress is not None:
        progress.note(f"  ⚠ {msg}")
    else:
        print(f"Warning: {msg}")
    return "cpu"


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


def _detect_nvidia_gpu() -> "dict | None":
    """Detect the GPU(s) and the driver's max CUDA version via ``nvidia-smi``.

    Deliberately does NOT use torch — this is what lets ``install-check``
    recommend the right PyTorch wheel even when the *currently installed* torch
    is broken, CPU-only, or absent. Returns a dict with ``names``,
    ``compute_caps``, ``driver_version`` and ``driver_max_cuda`` (the CUDA
    version the driver supports, e.g. ``"12.4"``), or ``None`` if ``nvidia-smi``
    isn't present (e.g. a CPU-only host or a login node without GPUs).
    """
    import re
    import shutil
    import subprocess

    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    info: dict = {
        "names": [],
        "compute_caps": [],
        "driver_version": None,
        "driver_max_cuda": None,
    }
    # Structured query — `compute_cap` is supported on reasonably recent drivers.
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,compute_cap,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0 and out.stdout.strip():
            for line in out.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if parts and parts[0]:
                    info["names"].append(parts[0])
                if len(parts) >= 2 and parts[1] and parts[1].replace(".", "").isdigit():
                    info["compute_caps"].append(parts[1])
                if len(parts) >= 3 and parts[2] and not info["driver_version"]:
                    info["driver_version"] = parts[2]
    except (OSError, subprocess.SubprocessError):
        pass
    # The driver's max CUDA version is printed in the plain `nvidia-smi` header.
    try:
        out = subprocess.run([smi], capture_output=True, text=True, timeout=15)
        m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", out.stdout)
        if m:
            info["driver_max_cuda"] = m.group(1)
    except (OSError, subprocess.SubprocessError):
        pass
    if not info["names"] and not info["driver_max_cuda"]:
        return None
    return info


def _recommend_torch_index_url(driver_max_cuda: "str | None") -> "str | None":
    """Pick a PyTorch wheel index URL from the driver's max CUDA version.

    Conservative: a wheel's CUDA runtime must be <= what the driver supports.
    Returns ``None`` when we can't tell or the driver is too old to map cleanly
    (the caller then advises a manual choice).
    """
    if not driver_max_cuda:
        return None
    try:
        major, minor = (int(x) for x in driver_max_cuda.split(".")[:2])
    except (ValueError, TypeError):
        return None
    cuda = major * 100 + minor  # "12.4" -> 1204
    if cuda >= 1201:
        return "https://download.pytorch.org/whl/cu121"
    if cuda >= 1108:
        return "https://download.pytorch.org/whl/cu118"
    return None


def _recommended_torch_command(gpu_info: "dict | None") -> "str | None":
    """The exact ``pip install`` line that installs the right PyTorch.

    GPU present + a known driver CUDA -> the matching ``cuXXX`` wheel. No GPU at
    all -> the CPU-only wheel. GPU present but driver too old / unmappable ->
    ``None`` (caller advises picking a wheel manually).
    """
    index = _recommend_torch_index_url((gpu_info or {}).get("driver_max_cuda"))
    if index is not None:
        return f"pip install --force-reinstall --no-cache-dir torch --index-url {index}"
    if not gpu_info:
        return "pip install --force-reinstall --no-cache-dir torch"
    return None


def _pip_install_torch(cmd: str) -> int:
    """Run a ``pip install ... torch ...`` line via the current interpreter."""
    import subprocess
    import sys

    parts = cmd.split()
    if parts and parts[0] == "pip":
        parts = [sys.executable, "-m", "pip"] + parts[1:]
    try:
        return subprocess.run(parts).returncode
    except OSError as exc:
        print(f"Failed to run pip: {exc}")
        return 1


def _run_install_check(args) -> int:
    """Implementation of ``python -m mamp_ml install-check``.

    A preflight that verifies the install can actually run, BEFORE a user
    submits a long job: mamp-ml version, the PyTorch build + whether it can use
    the visible GPU (the V100/sm_70 trap), a tiny live CUDA op, ColabFold
    discovery, and the bundled weights. Prints a PASS/WARN/FAIL line per check
    and returns non-zero if any hard check failed.
    """
    from mamp_ml import __version__

    ok = "  [ OK ]"
    warn = "  [WARN]"
    fail = "  [FAIL]"
    n_fail = 0
    n_warn = 0

    print(f"mamp-ml install-check  (mamp-ml {__version__})")
    print("=" * 50)

    # --- GPU hardware (detected WITHOUT torch, via nvidia-smi) ----------
    # This is what lets us recommend the right PyTorch even when the currently
    # installed torch is broken, CPU-only, or absent.
    gpu_info = _detect_nvidia_gpu()
    if gpu_info:
        names = ", ".join(gpu_info["names"] or ["(unknown)"])
        caps = gpu_info["compute_caps"]
        cap_str = f", compute capability {'/'.join(caps)}" if caps else ""
        drv = gpu_info["driver_version"] or "?"
        cu = gpu_info["driver_max_cuda"] or "?"
        print(f"{ok} GPU (nvidia-smi): {names}{cap_str}; driver {drv}, supports up to CUDA {cu}")
    else:
        print(f"{warn} No NVIDIA GPU detected by nvidia-smi (CPU-only host / login node).")
        n_warn += 1

    # --- PyTorch + whether it can drive the detected GPU ----------------
    torch_problem = False  # set when a (re)install of torch is warranted
    try:
        import torch

        print(f"{ok} PyTorch {torch.__version__} (CUDA build: {torch.version.cuda or 'cpu-only'})")
    except ImportError:
        print(f"{fail} PyTorch is not installed — inference cannot run")
        torch = None
        n_fail += 1
        torch_problem = True

    if torch is not None:
        if not torch.cuda.is_available():
            if gpu_info:
                print(
                    f"{fail} A GPU is present but this PyTorch can't use it "
                    "(torch.cuda.is_available() is False — likely a CPU-only build)."
                )
                n_fail += 1
                torch_problem = True
            else:
                print(
                    f"{warn} No usable CUDA GPU; inference will run on CPU — "
                    "fine for the bundled 8M model."
                )
                n_warn += 1
        else:
            count = torch.cuda.device_count()
            arches = list(torch.cuda.get_arch_list())
            print(f"{ok} CUDA available: {count} device(s); torch built for [{', '.join(arches) or '?'}]")
            for i in range(count):
                name = torch.cuda.get_device_name(i)
                major, minor = torch.cuda.get_device_capability(i)
                sm = f"sm_{major}{minor}"
                if arches and sm not in arches:
                    print(
                        f"{fail} GPU{i} {name} (CC {major}.{minor}/{sm}) is NOT supported "
                        "by this PyTorch — would crash with cudaErrorNoKernelImageForDevice."
                    )
                    n_fail += 1
                    torch_problem = True
                else:
                    print(f"{ok} GPU{i} {name} (CC {major}.{minor}/{sm}) supported")
            # Live smoke test: a real kernel launch catches mismatches the arch
            # list alone might miss.
            try:
                _ = (torch.ones(8, 8, device="cuda") @ torch.ones(8, 8, device="cuda")).sum().item()
                print(f"{ok} Live CUDA op succeeded")
            except Exception as exc:
                print(f"{fail} Live CUDA op failed: {type(exc).__name__}: {exc}")
                n_fail += 1
                torch_problem = True

    # --- Recommend (or install) a matching PyTorch ----------------------
    if torch_problem:
        cmd = _recommended_torch_command(gpu_info)
        print()
        if cmd is None:
            print(
                "  → Couldn't auto-pick a PyTorch wheel for this driver. Choose the "
                "index URL matching your driver's CUDA version at "
                "https://pytorch.org/get-started/locally/"
            )
        else:
            if gpu_info and gpu_info.get("driver_max_cuda"):
                print(
                    f"  → Recommended PyTorch for your GPU (driver supports up to "
                    f"CUDA {gpu_info['driver_max_cuda']}):"
                )
            else:
                print("  → Recommended PyTorch:")
            print(f"      {cmd}")
            if getattr(args, "install_torch", False):
                print("  Installing it now (--install-torch)…")
                rc = _pip_install_torch(cmd)
                if rc == 0:
                    print(
                        f"{ok} torch (re)installed. Re-run `mamp-ml install-check` to verify."
                    )
                else:
                    print(f"{fail} torch install failed (exit {rc}); run the command above manually.")
            else:
                print(
                    "  Add --install-torch to run this automatically: "
                    "`mamp-ml install-check --install-torch`"
                )

    # --- ColabFold ------------------------------------------------------
    from mamp_ml.fold.colabfold import find_colabfold_installs

    installs = find_colabfold_installs()
    if installs:
        print(f"{ok} ColabFold found: {installs[0][0]} ({installs[0][1]})")
    else:
        print(
            f"{warn} No colabfold_batch found. Folding needs ColabFold (or pass "
            "--structure esmfold). Reused structures via --structures don't need it."
        )
        n_warn += 1

    # --- Bundled weights ------------------------------------------------
    try:
        from mamp_ml.weights import default_weights_path

        wp = default_weights_path()
        if wp.is_file():
            print(f"{ok} Bundled model weights present: {wp.name}")
        else:
            print(f"{fail} Bundled weights missing at {wp} — reinstall mamp-ml")
            n_fail += 1
    except Exception as exc:
        print(f"{fail} Could not locate bundled weights: {exc}")
        n_fail += 1

    print("=" * 50)
    if n_fail:
        print(f"{n_fail} check(s) FAILED, {n_warn} warning(s). See messages above.")
        return 1
    if n_warn:
        print(f"All hard checks passed, {n_warn} warning(s).")
        return 0
    print("All checks passed.")
    return 0


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

    # Reconstruct the command line so the run log records exactly how the tool
    # was invoked (and therefore which version/flags produced its outputs).
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    args.command_line = "mamp-ml " + " ".join(shlex.quote(a) for a in raw_argv)

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

    if args.cmd == "install-check":
        return _run_install_check(args)

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
