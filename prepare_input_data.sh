#!/bin/bash

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo "Error: No input file provided"
    echo "Usage: bash run_prediction_pipeline.sh <input_excel_file>"
    exit 1
fi

INPUT_FILE=$1

# Check if file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' does not exist"
    exit 1
fi

echo "Running data preparation pipeline with input file: $INPUT_FILE"

# Change to the main project directory
cd "$(dirname "$0")"

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to run a script and check its exit status
run_script() {
    local script=$1
    shift  # Remove the first argument (script name) so $@ contains only the additional arguments
    
    # Create appropriate log file name based on script type
    local log_file
    if [[ $script == *.R ]]; then
        log_file="logs/$(basename "$script" .R).R.log"
    fi
    
    echo "Running $script..."
    
    if [[ $script == *.R ]]; then
        Rscript "scripts/$script" "$@" 2>&1 | tee "$log_file"
    fi
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Error: $script failed. Check $log_file for details."
        exit 1
    fi
    
    echo "Completed $script successfully."
    echo "----------------------------------------"
}


# Run scripts in order
echo "Starting MAMP prediction pipeline..."
echo "----------------------------------------"

# create intermediate files for model prediction
echo "Creating fasta file for AlphaFold modeling..."
mkdir -p intermediate_files
mkdir -p intermediate_files/receptor_only
run_script "01_convert_sheet_to_fasta.R" "$INPUT_FILE"