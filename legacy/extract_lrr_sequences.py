import numpy as np
from Bio import PDB
from Bio.PDB import *
from Bio import SeqIO

class LRRSequenceExtractor:
    def __init__(self):
        self.parser = PDB.PDBParser(QUIET=True)
        
    def extract_sequence_from_pdb(self, pdb_file):
        """Extract the full sequence from a PDB file"""
        structure = self.parser.get_structure('protein', pdb_file)
        ppb = PPBuilder()
        seq = ""
        for pp in ppb.build_peptides(structure):
            seq += str(pp.get_sequence())
        return seq

    def extract_lrr_regions(self, sequence, breakpoints):
        """
        Extract sequences between breakpoints
        
        Parameters
        ----------
        sequence: str
            Full protein sequence
        breakpoints: list or numpy.ndarray
            List/array of breakpoint positions defining LRR regions
            
        Returns
        -------
        dict: Dictionary with:
            - 'lrr_sequences': List of sequences in LRR regions
            - 'lrr_positions': List of tuples with (start, end) positions
        """
        lrr_sequences = []
        lrr_positions = []
        
        # Convert breakpoints to list if it's a numpy array
        if isinstance(breakpoints, np.ndarray):
            breakpoints = breakpoints.tolist()
        
        # Ensure breakpoints are integers
        breakpoints = [int(bp) for bp in breakpoints]
        
        # Process breakpoints in pairs
        for i in range(0, len(breakpoints)-1, 2):
            start = breakpoints[i]
            end = breakpoints[i+1]
            
            # Ensure start and end are within sequence bounds
            start = max(0, start)
            end = min(len(sequence), end)
            
            # Extract sequence between breakpoints
            lrr_seq = sequence[start:end]
            lrr_sequences.append(lrr_seq)
            lrr_positions.append((start, end))
            
        return {
            'lrr_sequences': lrr_sequences,
            'lrr_positions': lrr_positions
        }

    def analyze_lrr_regions(self, pdb_file, breakpoints):
        """
        Analyze LRR regions for a given PDB file and breakpoints
        
        Parameters
        ----------
        pdb_file: str
            Path to PDB file
        breakpoints: list or numpy.ndarray
            List/array of breakpoint positions
            
        Returns
        -------
        dict: Dictionary containing analysis results
        """
        # Get full sequence
        full_sequence = self.extract_sequence_from_pdb(pdb_file)
        
        # Extract LRR regions
        lrr_data = self.extract_lrr_regions(full_sequence, breakpoints)
        
        # Add analysis information
        results = {
            'full_sequence': full_sequence,
            'sequence_length': len(full_sequence),
            'num_lrr_regions': len(lrr_data['lrr_sequences']),
            **lrr_data
        }
        
        return results 