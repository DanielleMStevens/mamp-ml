#!/usr/bin/env python3
#-----------------------------------------------------------------------------------------------
# Krasileva Lab - Plant & Microbial Biology Department UC Berkeley
# Original Author: Danielle M. Stevens
# Python Version by: Assistant
# Script Purpose: Convert Excel sheet data to FASTA format
# Inputs: Excel file (.xlsx) with Sheet1 containing plant_species, locus_id, receptor, and receptor_sequence columns
# Outputs: FASTA file with formatted sequences
#-----------------------------------------------------------------------------------------------

import pandas as pd
import sys
import os

def write_fasta(data, filename):
    """
    Write data to a FASTA file where data contains sequence names and sequences
    Args:
        data: DataFrame with columns ['Locus_Tag_Name', 'Sequence']
        filename: Output FASTA file path
    """
    with open(filename, 'w') as file:
        for _, row in data.iterrows():
            file.write(f"{row['Locus_Tag_Name']}\n")
            file.write(f"{row['Sequence']}\n")

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        sys.exit("Example: python3 convert_sheet_to_fasta.py input_file.xlsx")

    # Load input file
    input_file = sys.argv[1]
    load_data = pd.read_excel(input_file, sheet_name="Sheet1")

    # Initialize empty DataFrame for receptor sequences
    receptor_full_length = pd.DataFrame(columns=['Locus_Tag_Name', 'Sequence'])

    # Process each row
    for _, row in load_data.iterrows():
        locus_tag = f">{row['plant_species']}|{row['locus_id']}|{row['receptor']}"
        sequence = row['receptor_sequence']
        
        receptor_full_length = pd.concat([
            receptor_full_length,
            pd.DataFrame({
                'Locus_Tag_Name': [locus_tag],
                'Sequence': [sequence]
            })
        ], ignore_index=True)

    # Remove duplicates based on Sequence
    receptor_full_length = receptor_full_length.drop_duplicates(subset=['Sequence'], keep='first')

    # Ensure output directory exists
    os.makedirs("intermediate_files", exist_ok=True)
    
    # Write to FASTA file
    write_fasta(receptor_full_length, "./intermediate_files/receptor_full_length.fasta")

if __name__ == "__main__":
    main()