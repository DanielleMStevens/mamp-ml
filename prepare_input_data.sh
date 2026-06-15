#!/bin/bash
#
# prepare_input_data.sh
# ---------------------
# Stage 1 of the legacy pipeline: just convert the input xlsx to the receptor
# FASTA needed by ColabFold. The full pipeline lives behind
# `python -m mamp_ml prepare`, which automatically handles this step plus
# everything downstream of the ColabFold run.
#
# This script is kept for backwards compatibility; new users should invoke
# `python -m mamp_ml prepare-fasta` (or `mamp-ml prepare`) directly.
#
# Usage:
#   bash prepare_input_data.sh <input_excel_file>

set -e

if [ $# -eq 0 ]; then
    echo "Error: No input file provided"
    echo "Usage: bash prepare_input_data.sh <input_excel_file>"
    exit 1
fi

cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

mkdir -p intermediate_files intermediate_files/receptor_only

exec python -m mamp_ml prepare-fasta \
    "$1" \
    intermediate_files/receptor_full_length.fasta
