#-----------------------------------------------------------------------------------------------
# Krasileva Lab - Plant & Microbial Biology Department UC Berkeley
# Author: Danielle M. Stevens
# Last Updated: 07/06/2020
# Script Purpose: 
# Inputs: 
# Outputs: 
#-----------------------------------------------------------------------------------------------

from pathlib import Path
import re
import shutil
import sys
import os   

project_root = Path(__file__).parent.parent
lrr_annotation_path = project_root / "01_LRR_Annotation"
sys.path.append(str(lrr_annotation_path))

from geom_lrr import Loader, Analyzer, Plotter
from extract_lrr_sequences import LRRSequenceExtractor


def parse_alphafold_log(log_file):
    """
    Parse AlphaFold log file to extract final pLDDT and pTM scores for each model.
    
    Args:
        log_file (Path): Path to the log.txt file
        
    Returns:
        dict: Dictionary with results organized by receptor and model
    """
    results = {}
    current_receptor = None
    
    print(f"Parsing AlphaFold log file: {log_file}")
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
            
        for line in lines:
            if "Query" in line and "length" in line:
                current_receptor = line.split("Query")[1].split(":")[1].split("(")[0].strip()
                results[current_receptor] = {}
                continue
                
            if "alphafold2_ptm_model_" in line and "took" not in line:
                model_match = re.search(r'model_(\d+)', line)
                if not model_match:
                    continue
                    
                model_num = model_match.group(1)
                
                plddt_match = re.search(r'pLDDT=(\d+\.?\d*)', line)
                ptm_match = re.search(r'pTM=(\d+\.?\d*)', line)
                
                if plddt_match and ptm_match:
                    plddt = float(plddt_match.group(1))
                    ptm = float(ptm_match.group(1))
                    
                    if model_num not in results[current_receptor] or \
                       plddt > results[current_receptor][model_num]['plddt']:
                        results[current_receptor][model_num] = {
                            'plddt': plddt,
                            'ptm': ptm
                        }
    
    return results

def write_results(results, output_file):
    """Write results to a clear text file showing only the best model."""
    receptor_width = 100
    
    with open(output_file, 'w') as f:
        f.write("Receptor".ljust(receptor_width))
        f.write("Best Model".ljust(15))
        f.write("pLDDT".ljust(10))
        f.write("pTM\n")
        f.write("-" * (receptor_width + 25) + "\n")
        
        for receptor in sorted(results.keys()):
            models = results[receptor]
            best_model = max(models.items(), key=lambda x: x[1]['plddt'])
            model_num = best_model[0]
            scores = best_model[1]
            
            f.write(f"{receptor.ljust(receptor_width)}")
            f.write(f"model_{model_num}".ljust(15))
            f.write(f"{scores['plddt']:.1f}".ljust(10))
            f.write(f"{scores['ptm']:.3f}\n")
    
    return {receptor: max(models.items(), key=lambda x: x[1]['plddt'])[0] 
            for receptor, models in results.items()}

def copy_best_models(best_models, source_dir, target_dir):
    """
    Copy the best model PDB files to the target directory.
    
    Args:
        best_models (dict): Dictionary mapping receptor names to their best model numbers
        source_dir (Path): Directory containing AlphaFold output
        target_dir (Path): Directory to copy best models to
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # First, get all PDB files in source directory
    all_pdb_files = list(source_dir.glob("*.pdb"))
    print(f"\nFound {len(all_pdb_files)} total PDB files in {source_dir}")
    print("\nExpected to find files for these receptors:")
    for receptor in sorted(best_models.keys()):
        print(f"- {receptor} (model_{best_models[receptor]})")
    
    # Continue with copying
    for receptor, model_num in best_models.items():
        # Modified pattern to handle rank with leading zeros
        file_pattern = f"{receptor}_unrelaxed_rank_[0-9][0-9][0-9]_alphafold2_ptm_model_{model_num}_seed_*.pdb"
        
        matching_files = list(source_dir.glob(file_pattern))
        
        if not matching_files:
            print(f"Warning: Could not find model file for {receptor} matching pattern {file_pattern}")
            continue
            
        if len(matching_files) > 1:
            print(f"Warning: Multiple matching files found for {receptor}, using first match")
            
        source_path = matching_files[0]
        target_path = target_dir / f"{receptor}.pdb"
        shutil.copy2(source_path, target_path)

def run_lrr_annotation(pdb_directory):
    """Run LRR annotation on the PDB files."""
    # Initialize objects
    L = Loader()
    A = Analyzer()
    P = Plotter()
    sequence_extractor = LRRSequenceExtractor()

    # Load structures
    L.load_batch(str(pdb_directory))
    
    # Analyze geometry
    A.load_structures(L.structures)
    A.compute_windings()
    A.compute_regressions()

    # Extract LRR sequences
    output_file = Path('./09_testing_and_dropout/test_data_set/lrr_annotation_results_independent_test.txt')
    
    with open(output_file, 'w') as f:
        f.write("PDB_Filename\tRegion_Number\tStart_Position\tEnd_Position\tSequence_Length\tFull_Sequence_Length\tTotal_LRR_Regions\tSequence\n")
    
    for pdb_id, breakpoints in A.breakpoints.items():
        pdb_file = pdb_directory / f"{pdb_id}.pdb"
        pdb_filename = pdb_file.name
        
        if not pdb_file.exists():
            print(f"Warning: PDB file not found for {pdb_id}")
            continue
            
        results = sequence_extractor.analyze_lrr_regions(str(pdb_file), breakpoints)
        
        with open(output_file, 'a') as f:
            for i, (seq, (start, end)) in enumerate(zip(results['lrr_sequences'], 
                                                       results['lrr_positions'])):
                f.write(f"{pdb_filename}\t{i+1}\t{start}\t{end}\t{len(seq)}\t{results['sequence_length']}\t{results['num_lrr_regions']}\t{seq}\n")

    # Cache data
    cache_dir = Path('./01_LRR_Annotation/cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    L.cache(str(cache_dir))
    A.cache_geometry(str(cache_dir))
    A.cache_regressions(str(cache_dir))

    # Generate plots
    P.load(A.windings, A.breakpoints, A.slopes)
    plot_dir = Path('./09_testing_and_dropout/test_data_set/lrr_annotation_plots')
    plot_dir.mkdir(parents=True, exist_ok=True)
    P.plot_regressions(save=True, directory=str(plot_dir))

def main():
    # Set up paths
    project_root = Path(__file__).parent.parent
    log_file = Path(sys.argv[1]) if len(sys.argv) > 1 else sys.exit("Error: Please provide the path to the ColabFold log.txt file as an argument")
    scores_file = project_root / "09_testing_and_dropout" / "test_data_set" / "alphafold_scores.txt"
    
    # Source and target directories for PDB files
    source_dir = project_root / "09_testing_and_dropout" / "test_data_set" / "receptor_only"
    target_dir = project_root / "09_testing_and_dropout" / "test_data_set" / "pdb_for_lrr_annotator"
    
    if not log_file.exists():
        print(f"Error: Log file not found at {log_file}")
        return
        
    # Step 1: Parse AlphaFold results and get best models
    results = parse_alphafold_log(log_file)
    best_models = write_results(results, scores_file)
    print(f"AlphaFold scores written to {scores_file}")
    
    # Step 2: Copy best models to target directory
    copy_best_models(best_models, source_dir, target_dir)
    print("Best models copied to target directory")
    
    # Step 3: Run LRR annotation
    run_lrr_annotation(target_dir)
    print("LRR annotation completed")

if __name__ == '__main__':
    main() 