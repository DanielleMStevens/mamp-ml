#!/bin/bash
echo "Running data preparation pipeline..."

# Change to the main project directory
cd "$(dirname "$0")"

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to run a script and check its exit status
run_script() {
    local script=$1
    local log_file="logs/$(basename "$script" .py).log"
    
    echo "Running $script..."
    
    if [[ $script == *.R ]]; then
        Rscript "scripts/$script" 2>&1 | tee "$log_file"
    else
        python "scripts/$script" 2>&1 | tee "$log_file"
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
mkdir -p intermediate_files
run_script "01_convert_sheet_to_fasta.R"

# run AlphaFold locally to extract ectodomin of receptor sequence
echo "Running AlphaFold to model the receptor sequence..."
mkdir -p intermediate_files/receptor_only

conda activate alphafold
colabfold_batch --num-models 1 ./intermediate_files/receptor_full_length.fasta ./intermediate_files/receptor_only/





run_script "02_alphafold_to_lrr_annotation.py"
run_script "03_parse_lrr_annotations.py"
run_script "04_chemical_conversion.R"
run_script "05_data_prep_for_training.py"

echo "Pipeline preparationcompleted successfully!" 