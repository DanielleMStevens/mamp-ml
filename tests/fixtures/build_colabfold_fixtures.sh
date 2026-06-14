#!/usr/bin/env bash
# =============================================================================
# build_colabfold_fixtures.sh
# -----------------------------------------------------------------------------
# Regenerate the ColabFold structure-prediction artefacts used by the
# checkpoint-4 golden tests, from the canonical example_data.xlsx.
#
# These outputs are environment-dependent (require a working colabfold_batch
# install with the AlphaFold2 weights), tens of megabytes per receptor, and
# fully reproducible from this script. They are therefore NOT committed to
# git (see .gitignore) — every developer regenerates locally once on a
# machine where colabfold_batch is functional, typically a Linux/GPU box.
#
# What gets produced
# ------------------
#   tests/fixtures/colabfold_output/
#     receptor_full_length.fasta         (from xlsx via mamp_ml.preprocess)
#     log.txt                            (ColabFold per-model scores)
#     <receptor1>_unrelaxed_rank_001_*.pdb
#     <receptor2>_unrelaxed_rank_001_*.pdb
#     ... plus auxiliary ColabFold files
#
# How to use
# ----------
#   # 1. Activate the localfold env (built by install_software.sh)
#   conda activate localfold
#   # 2. Make sure colabfold_batch is reachable
#   which colabfold_batch
#   # 3. Run from the repository root, or from this directory
#   bash tests/fixtures/build_colabfold_fixtures.sh
#
# Cost
# ----
# Two ~1000-AA receptors with --num-models 1 typically takes 30-60 min on a
# modern GPU and several hours on CPU. Network access is required (ColabFold
# queries the MSA API).
#
# After this script completes
# ----------------------------
# Run `python tests/fixtures/build_golden_outputs.py` (added in checkpoint 4b)
# to drive the legacy preparation pipeline once and capture the small
# kilobyte-sized downstream artefacts (test_data.csv, ready_test_data.csv,
# lrr_*.fasta/.txt, bfactor_winding_lrr_segments.csv). Those files ARE
# committed and serve as the gold standard against which the new
# mamp_ml.preprocess / mamp_ml.structure / mamp_ml.lrr_features modules are
# validated in checkpoints 4b through 4e.
# =============================================================================

set -euo pipefail

# Resolve repository root regardless of where this script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FIXTURES_DIR="$SCRIPT_DIR"
OUTPUT_DIR="$FIXTURES_DIR/colabfold_output"
EXAMPLE_XLSX="$REPO_ROOT/example_data.xlsx"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [ ! -f "$EXAMPLE_XLSX" ]; then
    echo "Error: example data not found at $EXAMPLE_XLSX" >&2
    exit 1
fi

if ! command -v colabfold_batch >/dev/null 2>&1; then
    cat >&2 <<'EOF'
Error: colabfold_batch is not on $PATH.

This fixture-generation script relies on the same colabfold_batch entry point
that run_preparation_pipeline.sh invokes. Activate the localfold conda env
(or whichever env you installed colabfold into) before re-running:

    conda activate localfold
    bash tests/fixtures/build_colabfold_fixtures.sh

If you have not installed ColabFold yet, see install_software.sh and
scripts/install_colabbatch_linux.sh.
EOF
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1 — Build the receptor full-length FASTA from example_data.xlsx
# Uses the new mamp_ml.preprocess.xlsx_to_receptor_fasta function so the
# FASTA is generated exactly the same way as in the production pipeline
# (instead of duplicating logic here).
# ---------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"
RECEPTOR_FASTA="$OUTPUT_DIR/receptor_full_length.fasta"

echo "Generating receptor FASTA from $EXAMPLE_XLSX ..."
PYTHONPATH="$REPO_ROOT/src" python - <<PY
from mamp_ml.preprocess import xlsx_to_receptor_fasta
n = xlsx_to_receptor_fasta(r"""$EXAMPLE_XLSX""", r"""$RECEPTOR_FASTA""")
print(f"  Wrote {n} unique receptor records to $RECEPTOR_FASTA")
PY

# ---------------------------------------------------------------------------
# Step 2 — Run ColabFold (one model per receptor)
# Matches the canonical invocation in run_preparation_pipeline.sh:76.
# ---------------------------------------------------------------------------
echo "Running colabfold_batch (--num-models 1) ..."
colabfold_batch --num-models 1 "$RECEPTOR_FASTA" "$OUTPUT_DIR"

echo ""
echo "ColabFold fixtures generated under: $OUTPUT_DIR"
echo ""
echo "Next step:"
echo "  python tests/fixtures/build_golden_outputs.py"
echo "(That script lands in checkpoint 4b — it runs the legacy preparation"
echo " pipeline against these outputs and captures the small downstream"
echo " artefacts that DO get committed as the gold standard.)"
