#!/bin/bash
#
# run_preparation_pipeline.sh
# ---------------------------
# Thin wrapper around `python -m mamp_ml prepare`. Runs the full preparation
# pipeline in one shot (xlsx -> ready_test_data.csv), assuming ColabFold has
# already been run and its outputs are in intermediate_files/receptor_only/.
#
# This script is kept around for users with muscle memory from the legacy
# workflow; new users should invoke `python -m mamp_ml prepare` (or, after
# `pip install`, `mamp-ml prepare`) directly.
#
# Usage:
#   bash run_preparation_pipeline.sh <input_excel_file>

set -e

if [ $# -eq 0 ]; then
    echo "Error: No input file provided"
    echo "Usage: bash run_preparation_pipeline.sh <input_excel_file>"
    exit 1
fi

cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

exec python -m mamp_ml prepare "$1"
