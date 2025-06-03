import os
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm
import pickle # Added for potential loading errors

# Assuming geom_lrr is in the same parent directory or Python path
try:
    from geom_lrr.loader import Loader
    # Import Analyzer and compute_winding function
    # Assuming Analyzer might have a method to compute breakpoints
    from geom_lrr.analyzer import Analyzer, compute_winding
except ImportError:
    print("Error: Could not import Loader, Analyzer, or compute_winding from geom_lrr.")
    print("Make sure 'geom_lrr' directory is accessible and contains analyzer.py with compute_winding.")
    exit() # Or handle the error appropriately
except Exception as e:
    print(f"Error during geom_lrr import: {e}") # Catch other potential import errors
    exit()

def analyze_lrr_bfactor_peaks(pdb_dir, period=25, filter_order=10, cache_dir=None):
    """
    Analyzes B-factor peaks within LRR regions.
    Computes winding numbers and LRR repeat breakpoints on the fly from loaded structures.
    Loads structures via load_batch and B-factors via load_single.

    Args:
        pdb_dir (str): Path to the directory containing PDB files.
        period (int, optional): Approximate period for Butterworth filter. Defaults to 25.
        filter_order (int, optional): Order for Butterworth filter. Defaults to 10.
        cache_dir (str, optional): Path to the directory containing cached regression data.

    Returns:
        pandas.DataFrame: A DataFrame containing peak information with columns:
                          'Protein Key', 'Residue Index', 'Filtered B-Factor',
                          'Winding Number', 'LRR Repeat Number'.
                          Returns an empty DataFrame if essential data is missing.
    """
    print(f"Loading Structures via load_batch from PDBs in: {pdb_dir}")
    loader = Loader()
    keys_loaded = []
    try:
        # Use load_batch to get structures and the list of keys
        loader.load_batch(pdb_dir, progress=True)
        structures = loader.structures
        if not structures:
            print("Warning: No structures loaded via load_batch. Check PDB directory and files.")
            return pd.DataFrame()
        keys_loaded = list(structures.keys()) # Get keys from successfully loaded structures
        print(f"Found {len(keys_loaded)} structures via load_batch.")

    except FileNotFoundError:
        print(f"Error: PDB directory not found: {pdb_dir}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error during load_batch for structures: {e}")
        return pd.DataFrame()

    # Now, load B-factors individually using load_single for the keys found
    print(f"Loading B-factors via load_single for {len(keys_loaded)} keys...")
    bfactors = {} # Initialize bfactors dictionary
    for key in tqdm(keys_loaded, desc="Loading B-factors"):
        try:
            loader.bfactors = {} # Clear before each call
            loader.load_single(pdb_dir, f"{key}.pdb") # Assumes key is filename without .pdb
            if key in loader.bfactors:
                 bfactors[key] = loader.bfactors[key]
        except Exception as e:
            print(f"\nError calling load_single for {key}.pdb: {e}")
            continue # Skip to next key

    loader.bfactors = {} # Clear loader's dict
    print(f"Finished loading B-factors. Found B-factors for {len(bfactors)} keys.")

    if not bfactors:
         print("Error: Failed to load any B-factors using load_single.")
         return pd.DataFrame()

    # --- Load Cached Breakpoints ---
    print(f"Loading cached regression data (breakpoints) from: {cache_dir}")
    analyzer = Analyzer()
    try:
        # This still loads the original breakpoints (likely just start/end)
        analyzer.retrieve_regressions(cache_dir)
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Error: Cache directory or essential regression files not found/corrupt in {cache_dir}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading cached regression data: {e}")
        return pd.DataFrame()

    analyzer.load_structures(structures)
    analyzer.load_bfactors(bfactors)

    results = []
    # Check we have breakpoints, structures, bfactors
    required_data = {'breakpoints', 'structures', 'bfactors'}
    if not required_data.issubset(analyzer.__dict__.keys()) or \
       not analyzer.breakpoints or not analyzer.structures or not analyzer.bfactors:
         print(f"Error: Analyzer is missing essential data.")
         print(f"Breakpoints: {len(analyzer.breakpoints)}, Structures: {len(analyzer.structures)}, B-factors: {len(analyzer.bfactors)}")
         return pd.DataFrame()

    # --- Filter Setup ---
    nyquist = 0.5
    low_cutoff = 0.5 / period
    high_cutoff = 2.0 / period
    if high_cutoff >= nyquist:
        print(f"Warning: High cutoff frequency ({high_cutoff}) >= Nyquist ({nyquist}). Adjusting.")
        high_cutoff = nyquist * 0.99 # Ensure it's strictly less
    if low_cutoff <= 0:
        print(f"Error: Low cutoff ({low_cutoff}) must be positive.")
        return pd.DataFrame()
    if low_cutoff >= high_cutoff:
        print(f"Error: Low cutoff ({low_cutoff}) >= high cutoff ({high_cutoff}). Cannot create filter.")
        return pd.DataFrame()
    try:
        sos = signal.butter(filter_order, [low_cutoff, high_cutoff], 'bandpass', output='sos', fs=1.0)
    except ValueError as e:
        print(f"Error creating Butterworth filter: {e}")
        return pd.DataFrame()
    # --- End Filter Setup ---

    # --- Define Approximate Repeat Length ---
    APPROX_REPEAT_LENGTH = 25 # Or your best estimate
    print(f"Using approximate repeat length: {APPROX_REPEAT_LENGTH} for LRR numbering.")
    # --- End Define Approximate Repeat Length ---

    print("Analyzing B-factor peaks in LRR regions (computing windings, using cached breakpoints)...")
    # Intersect keys from all three sources (structure, bfactor, cached breakpoints)
    valid_keys = set(analyzer.structures.keys()) & set(analyzer.bfactors.keys()) & set(analyzer.breakpoints.keys())
    print(f"Processing {len(valid_keys)} proteins with structures, b-factors, and cached breakpoints.")

    if not valid_keys:
         print("No proteins found with all required data (structure, b-factor, cached breakpoints).")
         return pd.DataFrame()

    for key in tqdm(list(valid_keys), desc="Processing Proteins"):
        structure = analyzer.structures[key]
        bfactor = analyzer.bfactors[key]
        # Use the breakpoints loaded from the cache
        breakpoints = analyzer.breakpoints[key]

        # Compute winding number (still useful for the 'Winding Number' column)
        try:
            winding_res = compute_winding(structure)
            winding = winding_res["winding"]
        except Exception as e:
            print(f"Warning: Skipping {key}, failed to compute winding number: {e}")
            continue

        # Basic check: winding array is typically N-1, bfactor N.
        if len(bfactor) != len(winding) + 1:
             print(f"Warning: Skipping {key}, length mismatch between B-factor ({len(bfactor)}) and computed winding ({len(winding)}).")
             continue

        # *** Use ONLY the first and last breakpoint to define the overall LRR region ***
        if len(breakpoints) < 2:
            print(f"Warning: Skipping {key}, fewer than 2 breakpoints loaded from cache: {breakpoints}")
            continue # Need at least start and end

        lrr_start_index = breakpoints[0]
        lrr_end_index = breakpoints[-1] # Use the last one, assumes format [start, ..., end] or just [start, end]

        if lrr_start_index < 0 or lrr_end_index > len(bfactor) or lrr_start_index >= lrr_end_index:
             print(f"Warning: Skipping LRR region [{lrr_start_index}, {lrr_end_index}) for {key} due to invalid indices from cache (B-factor length: {len(bfactor)}).")
             continue

        # Process the single overall LRR segment defined by cache
        a = lrr_start_index
        b = lrr_end_index

        bfactor_segment = bfactor[a:b]

        min_len_for_filter = filter_order * 2 + 1 # Heuristic minimum length
        if len(bfactor_segment) < min_len_for_filter:
            # print(f"Debug: Skipping LRR region [{a}, {b}) for {key}, too short ({len(bfactor_segment)}) for filter (min: {min_len_for_filter}).")
            continue # Skip if the whole LRR region is too short

        try:
            bff = signal.sosfiltfilt(sos, bfactor_segment)
        except ValueError as e:
             print(f"Warning: Could not filter LRR segment [{a}, {b}) for {key} (length {len(bfactor_segment)}): {e}")
             continue

        # Normalize
        bff_max = np.max(np.abs(bff))
        if bff_max > 1e-9:
            bff /= bff_max
        else:
            bff[:] = 0

        # Find zero crossings (negative to positive)
        zero_crossings = []
        current_repeat = 0
        repeat_numbers = np.zeros(len(bff), dtype=int)
        
        # Initialize first repeat number
        if bff[0] > 0:
            current_repeat = 1
            zero_crossings.append(0)
            
        # Find all zero crossings (negative to positive transitions)
        for i in range(1, len(bff)):
            if bff[i-1] <= 0 and bff[i] > 0:
                zero_crossings.append(i)
                current_repeat += 1
            repeat_numbers[i] = current_repeat

        # Record information for all residues in the segment
        for i in range(len(bff)):
            residue_idx = a + i # Absolute residue index in the full protein
            filtered_b_val = bff[i] # Filtered B-factor relative to segment start

            # Use the repeat number we calculated based on zero crossings
            lrr_repeat_number = repeat_numbers[i]

            # Winding number calculation (remains the same)
            winding_idx = residue_idx - 1
            if 0 <= winding_idx < len(winding):
                winding_val = winding[winding_idx]
            else:
                winding_val = np.nan

            results.append({
                'Protein Key': key,
                'Residue Index': residue_idx,
                'Filtered B-Factor': filtered_b_val,
                'Winding Number': winding_val,
                'LRR Repeat Number': lrr_repeat_number
            })

        # Optional: Print debug information about the zero crossings
        print(f"Found {len(zero_crossings)} zero crossings for {key}")

    if not results:
        print("No LRR segments processed or no data generated.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    # --- Configuration ---
    PDB_DIRECTORY = "./03_out_data/modeled_structures/pdb_for_lrr_annotator/"
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    CACHE_DIRECTORY = os.path.join(SCRIPT_DIR, "cache")
    # --- End Configuration ---

    if not os.path.isdir(PDB_DIRECTORY):
        print(f"Error: PDB Directory not found: {PDB_DIRECTORY}")
        exit()
    if not os.path.isdir(CACHE_DIRECTORY):
         print(f"Error: Cache Directory not found: {CACHE_DIRECTORY}")
         exit()

    # Fix: Pass cache_dir as a named argument
    peak_data = analyze_lrr_bfactor_peaks(pdb_dir=PDB_DIRECTORY, cache_dir=CACHE_DIRECTORY)

    if not peak_data.empty:
        print("\n--- B-Factor Peak Analysis Results ---")
        output_csv = os.path.join("04_Preprocessing_results", "bfactor_winding_lrr_segments.csv")
        peak_data.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")
    else:
        print("\nAnalysis finished, but no segment data was generated.")
