#-----------------------------------------------------------------------------------------------
# Krasileva Lab - Plant & Microbial Biology Department UC Berkeley
# Author: Danielle M. Stevens
# Last Updated: 07/06/2020
# Script Purpose: 
# Inputs: 
# Outputs: 
#-----------------------------------------------------------------------------------------------


def normalize_header(header):
    """Convert a header by replacing spaces and pipes with underscores"""
    return header.replace(' ', '_').replace('|', '_')

def read_receptor_sequences(fasta_file):
    """Read the receptor FASTA file and create a mapping of normalized headers to (original header, sequence)"""
    header_to_seq = {}
    current_header = ""
    current_sequence = ""
    
    with open(fasta_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Store previous sequence if it exists
                if current_sequence:
                    # Remove the '>' and normalize the header
                    orig_header = current_header[1:]
                    norm_header = normalize_header(orig_header)
                    header_to_seq[norm_header] = (orig_header, current_sequence)
                current_header = line
                current_sequence = ""
            else:
                current_sequence += line
        
        # Store the last sequence
        if current_sequence:
            orig_header = current_header[1:]
            norm_header = normalize_header(orig_header)
            header_to_seq[norm_header] = (orig_header, current_sequence)
    
    return header_to_seq

def find_best_match(pdb_header, header_map):
    """Find the matching header in the full sequence headers using normalized headers"""
    # Normalize the PDB header
    norm_pdb_header = normalize_header(pdb_header)
    
    # Look for an exact match of the normalized PDB header in the normalized full headers
    if norm_pdb_header in header_map:
        orig_header, _ = header_map[norm_pdb_header]
        return orig_header
    
    # If exact match fails, optionally try substring matching as a fallback (or remove if not desired)
    # Fallback: Check if normalized PDB header is a substring of any normalized full header
    # for norm_header, (orig_header, _) in header_map.items():
    #     if norm_pdb_header in norm_header:
    #         print(f"Debug: Found substring match for {norm_pdb_header} in {norm_header}") # Optional debug print
    #         return orig_header

    return None


def parse_lrr_results(lrr_file, header_map):
    """Parse LRR annotation results and match with full sequences"""
    sequences = []
    
    with open(lrr_file) as f:
        # Skip header line
        next(f)
        
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 8: # Basic check for enough fields
                print(f"Warning: Skipping malformed line: {line.strip()}")
                continue
                
            pdb_filename_header = fields[0]  # Get the PDB filename header from first column
            lrr_sequence = fields[7]
            
            # --- Modification Start ---
            # Remove the .pdb suffix to get the base identifier
            if pdb_filename_header.endswith('.pdb'):
                base_header_id = pdb_filename_header[:-4] 
            else:
                base_header_id = pdb_filename_header # Use as is if no .pdb suffix
            # --- Modification End ---

            # Find matching header using the base identifier
            full_header = find_best_match(base_header_id, header_map)
            
            if full_header:
                sequences.append((f">{full_header}", lrr_sequence))
            else:
                # Use the original filename header in the warning for clarity
                print(f"Warning: No matching header found for PDB: {pdb_filename_header}") 
                # Show the identifier we actually tried to match
                print(f"Attempted to match identifier: {normalize_header(base_header_id)}") 
    
    return sequences

def write_fasta(sequences, output_file):
    """Write sequences in FASTA format"""
    with open(output_file, 'w') as f:
        for header, sequence in sequences:
            header = header.replace(' ', '_')
            # Add |LRR_domain to the header
            header = header + "|LRR_domain"
            f.write(f"{header}\n")
            f.write(f"{sequence}\n")

def main():
    # Input files
    receptor_fasta = "09_testing_and_dropout/test_data_set/receptor_full_length_model_validation.fasta"
    lrr_results = "09_testing_and_dropout/test_data_set/lrr_annotation_results_independent_test.txt"
    
    # Output files
    output_fasta = "09_testing_and_dropout/test_data_set/lrr_domain_sequences_independent_test.fasta"
    
    # Read receptor sequences
    sequence_map = read_receptor_sequences(receptor_fasta)
    
    # Parse LRR results and match with sequences
    sequences = parse_lrr_results(lrr_results, sequence_map)
    
    # Write output FASTA file
    write_fasta(sequences, output_fasta)
    
    print(f"Created {output_fasta} with {len(sequences)} sequences")

if __name__ == "__main__":
    main()