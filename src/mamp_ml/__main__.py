"""mamp_ml — package-level CLI dispatcher.

The primary user entry point is the **``prepare``** command, which runs the
full preparation pipeline in one shot:

    python -m mamp_ml prepare INPUT.xlsx

This invocation:

1. Builds the receptor FASTA from ``INPUT.xlsx``.
2. Checks whether ColabFold has already been run for these receptors.
   * If yes (the ``intermediate_files/receptor_only/log.txt`` is present),
     continues to step 3 silently.
   * If no, prints the exact ``colabfold_batch`` command the user should
     run next, and exits with a non-zero code so the workflow stops cleanly.
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mamp_ml",
        description=(
            "MAMP-ml CLI. Use `prepare` for the one-shot pipeline; "
            "the per-stage subcommands are escape hatches for debugging."
        ),
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
            "the command writes the receptor FASTA, prints the colabfold_batch "
            "invocation, and exits cleanly so the user can run ColabFold and "
            "re-invoke this command."
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

    # predict (one-shot top-level, end-to-end including model inference) ---
    sp = sub.add_parser(
        "predict",
        help=(
            "Run the full pipeline plus model inference "
            "(xlsx -> predictions CSV)."
        ),
        description=(
            "Runs `prepare` to produce ready_test_data.csv, then loads the "
            "bundled MAMP-ml checkpoint and runs ESM-2 inference, writing "
            "the predictions CSV the user actually wants. If ColabFold has "
            "not been run yet the command prints the colabfold_batch "
            "invocation and exits cleanly so the user can run ColabFold "
            "and re-invoke this command."
        ),
    )
    sp.add_argument("xlsx", help="Path to input .xlsx file")
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
        "--checkpoint",
        default=None,
        help=(
            "Path to the model checkpoint .pth file. Defaults to the "
            "bundled mamp_ml_weights.pth shipped inside the package."
        ),
    )
    sp.add_argument(
        "--device",
        default="cpu",
        help=(
            "Torch device to run inference on (default: cpu). "
            "Pass 'cuda' on a GPU host for ~50-100x speedup."
        ),
    )
    sp.add_argument(
        "--sheet-name",
        default="Sheet1",
        help="Sheet name to read from the workbook (default: Sheet1)",
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


def _run_prepare(args) -> int:
    """One-shot pipeline orchestrator behind the ``prepare`` subcommand.

    Mirrors the legacy ``prepare_input_data.sh`` + ``run_preparation_pipeline.sh``
    flow but in a single Python process so the user only invokes one command.

    Returns
    -------
    int
        Process exit code: ``0`` if the full pipeline completed, ``2`` if
        ColabFold hadn't yet been run for the current input (in which case
        the receptor FASTA was still written and the user was shown the
        exact command to run next).
    """
    from pathlib import Path

    from mamp_ml.preprocess import (
        add_chemical_features,
        assemble_test_data,
        build_lrr_domain_fasta,
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

    # ---- Stage 1/6 ----
    n = xlsx_to_receptor_fasta(
        xlsx_path, receptor_fasta, sheet_name=args.sheet_name
    )
    print(f"[1/6] receptor FASTA: wrote {n} unique records -> {receptor_fasta}")

    # ---- ColabFold gate ----
    log_path = colabfold_dir / "log.txt"
    if not log_path.is_file():
        print()
        print("ColabFold has not been run yet for this input.")
        print(
            "Run ColabFold on the receptor FASTA above, then re-invoke "
            "this command. Suggested invocation:"
        )
        print()
        print(
            f"  colabfold_batch --num-models 1 --num-recycle 1 \\\n"
            f"      {receptor_fasta} \\\n"
            f"      {colabfold_dir}"
        )
        print()
        print(
            "(See scripts/install_colabbatch_linux.sh or "
            "scripts/install_colabbatch_mac.sh to install ColabFold locally.)"
        )
        return 2

    # ---- Stage 2/6 ----
    n_lrr = run_structure_stage(
        colabfold_dir,
        scores_path,
        pdb_target_dir,
        lrr_results,
        cache_dir=structure_cache_dir,
        plot_dir=plot_dir,
    )
    print(
        f"[2/6] structure-stage: {n_lrr} LRR region rows -> {lrr_results}"
    )

    # ---- Stage 3/6 ----
    n_dom = build_lrr_domain_fasta(lrr_results, receptor_fasta, lrr_domain_fasta)
    print(
        f"[3/6] LRR-domain FASTA: {n_dom} sequences -> {lrr_domain_fasta}"
    )

    # ---- Stage 4/6 ----
    bfactor_df = write_bfactor_lrr_segments(
        pdb_target_dir, bfactor_cache_dir, bfactor_csv
    )
    print(
        f"[4/6] B-factor analysis: {len(bfactor_df)} rows -> {bfactor_csv}"
    )

    # ---- Stage 5/6 ----
    test_df = assemble_test_data(
        xlsx_path,
        lrr_domain_fasta,
        test_data_csv,
        sheet_name=args.sheet_name,
    )
    print(
        f"[5/6] test-data assembly: {len(test_df)} rows -> {test_data_csv}"
    )

    # ---- Stage 6/6 ----
    ready_df = add_chemical_features(test_data_csv, ready_csv)
    print(
        f"[6/6] chemical features: {len(ready_df)} rows -> {ready_csv}"
    )

    print()
    print(f"Preparation complete. Final output: {ready_csv}")
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

    # The prepare args parser declares a superset of what we need here, but
    # the same attribute names — so feed args directly into _run_prepare
    # without rebuilding a namespace.
    prepare_rc = _run_prepare(args)
    if prepare_rc != 0:
        return prepare_rc

    from mamp_ml import train
    from mamp_ml.weights import default_weights_path

    out_dir = Path(args.out_dir)
    ready_csv = out_dir / "ready_test_data.csv"
    checkpoint = Path(args.checkpoint) if args.checkpoint else default_weights_path()
    if not checkpoint.is_file():
        print(
            f"Error: model checkpoint not found at {checkpoint}. "
            "Pass --checkpoint with an explicit path, or reinstall the "
            "package to restore the bundled weights.",
        )
        return 3
    if not ready_csv.is_file():
        # Should not happen — prepare returns 0 only after producing this file.
        print(f"Error: ready_test_data.csv missing at {ready_csv}.")
        return 4

    print()
    print(f"Running ESM-2 inference on {ready_csv} (device: {args.device}) ...")

    eval_argv = [
        "--model", "esm2_bfactor_weighted",
        "--eval_only_data_path", str(ready_csv.resolve()),
        "--model_checkpoint_path", str(checkpoint.resolve()),
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
    finally:
        os.chdir(prev_cwd)

    print()
    print(f"Predictions written to {out_dir / 'predictions.csv'}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for ``python -m mamp_ml``.

    Returns the process exit code (0 on success). Subcommand errors raise as
    usual; this wrapper only handles dispatch.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "prepare":
        return _run_prepare(args)

    if args.cmd == "predict":
        return _run_predict(args)

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
