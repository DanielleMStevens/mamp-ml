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
        Rscript "06_scripts_ml/$script" 2>&1 | tee "$log_file"
    else
        python "06_scripts_ml/$script" 2>&1 | tee "$log_file"
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

run_script "00_visualize_input_data.R"
run_script "01_prep_receptor_sequences_for_modeling.R"
run_script "02_alphafold_to_lrr_annotation.py"
run_script "03_parse_lrr_annotations.py"
run_script "04_chemical_conversion.R"
run_script "05_data_prep_for_training.py"

echo "Pipeline preparationcompleted successfully!" 