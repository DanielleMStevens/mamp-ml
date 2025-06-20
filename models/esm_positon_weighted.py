#-----------------------------------------------------------------------------------------------
# Krasileva Lab - Plant & Microbial Biology Department UC Berkeley
# Author: MAMP-ML Project Team
# Last Updated: 2024
# Script Purpose: ESM2-based peptide-receptor interaction prediction with B-factor weighting
# Inputs: 
#   - Peptide and receptor protein sequences
#   - Chemical features (bulkiness, charge, hydrophobicity)
#   - B-factor structural data for position weighting
#   - Training labels (interaction classes: 0, 1, 2)
# Outputs: 
#   - Trained model for peptide-receptor interaction prediction
#   - Prediction probabilities for each interaction class
#   - Evaluation metrics and CSV files with predictions
#-----------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os

######################################################################
# FiLM (Feature-wise Linear Modulation) Layer for Chemical Conditioning
######################################################################

class FiLMWithConcatenation(nn.Module):
    """
    FiLM layer that conditions sequence representation with chemical features.
    
    FiLM (Feature-wise Linear Modulation) is a conditional normalization method
    that modulates neural network activations using learned parameters generated
    from conditioning information. Here, we use this technique to condition protein
    sequence embeddings with chemical property features.
    
    This implementation:
    - Processes 3D chemical features (bulkiness, charge, hydrophobicity)
    - Projects them to the same dimension as sequence embeddings
    - Generates conditioning parameters (gamma and beta) via FiLM
    - Applies element-wise modulation to sequence representations
    """
    def __init__(self, feature_dim):
        """
        Initialize the FiLM conditioning layer.
        
        Args:
            feature_dim (int): Dimension of the sequence embeddings to be conditioned
        """
        super().__init__()
        # Process chemical features with a small MLP network
        # The network projects 3D chemical features (bulkiness, charge, hydrophobicity)
        # into the same dimension as the sequence embeddings
        self.chemical_proj = nn.Sequential(
            nn.Linear(3, 64),  # Project 3 chemical features to hidden dimension
            nn.ReLU(),         # Non-linearity for feature transformation
            nn.Linear(64, feature_dim),  # Project to final dimension matching sequence features
            nn.LayerNorm(feature_dim)    # Normalize the outputs
        )
        
        # FiLM layer - generates scaling (gamma) and shift (beta) parameters
        # These parameters are used to modulate the sequence embeddings
        self.film_layer = nn.Linear(feature_dim * 2, feature_dim * 2)  # For gamma and beta
        self.layer_norm = nn.LayerNorm(feature_dim)  # Normalize sequence embeddings
        self.dropout = nn.Dropout(0.1)  # Regularization to prevent overfitting

    def forward(self, x, z, chemical_features=None):
        """
        Apply FiLM conditioning to sequence embeddings using chemical features.
        
        The conditioning process:
        1. Normalize sequence embeddings
        2. Process chemical features through MLP
        3. Generate FiLM parameters (gamma, beta)
        4. Apply modulation: gamma * x + beta
        
        Args:
            x: Sequence embeddings (batch_size, seq_len, feature_dim)
               These are the embeddings from the ESM protein language model
            z: Pooled context vector (batch_size, feature_dim)
               This is a global context vector summarizing the entire sequence
            chemical_features: Combined chemical features (batch_size, seq_len, 6) 
                             [3 for sequence, 3 for receptor]
                             These are the physicochemical properties of residues
        
        Returns:
            Conditioned sequence embeddings (batch_size, seq_len, feature_dim)
        """
        batch_size, seq_len, feature_dim = x.shape
        
        # Apply layer normalization to sequence embeddings
        x = self.layer_norm(x)
        
        # Expand the pooled context vector (z) to match sequence length
        # This allows the context to be applied at each position
        z = z.unsqueeze(1).expand(-1, seq_len, -1)  # Shape: (batch_size, seq_len, feature_dim)
        
        # Process chemical features if provided
        if chemical_features is not None:
            # Split combined features into sequence and receptor components
            seq_features, rec_features = torch.split(chemical_features, 3, dim=-1)
            
            # Project each set of chemical features to embedding space
            seq_chem = self.chemical_proj(seq_features)  # Process sequence chemical features
            rec_chem = self.chemical_proj(rec_features)  # Process receptor chemical features
            
            # Add chemical information to the context vector
            # This integrates chemical properties with sequence information
            z = z + seq_chem + rec_chem
        
        # Concatenate sequence embeddings with the context vector
        combined = torch.cat([x, z], dim=-1)  # Shape: (batch_size, seq_len, feature_dim*2)
        
        # Generate FiLM conditioning parameters (gamma and beta)
        gamma_beta = self.film_layer(combined)  # Shape: (batch_size, seq_len, feature_dim*2)
        
        # Split into separate scaling (gamma) and shift (beta) parameters
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)  # Each shape: (batch_size, seq_len, feature_dim)
        
        # Apply FiLM conditioning: element-wise multiply by gamma and add beta
        # This is the core of FiLM conditioning: output = gamma * x + beta
        output = gamma * x + beta
        
        # Apply dropout for regularization
        return self.dropout(output)

######################################################################
# B-Factor Weight Generator for Structural Position Weighting
######################################################################

class BFactorWeightGenerator:
    """
    Generates weights based on B-factors from preprocessed data.
    
    The B-factors here represent positions along a transformed LRR coil structure.
    Each Leucine-Rich Repeat (LRR) is mapped onto a standardized coil, where the
    B-factor values indicate the numerical position along this coil. This allows
    us to weight different regions of the LRR based on their relative position
    in the repeating structural motif.
    
    Key functionality:
    - Loads B-factor data from CSV files
    - Maps protein identifiers to structural weights
    - Normalizes weights to specified range
    - Handles missing data gracefully with default weights
    """
    def __init__(self, bfactor_csv_path=None, min_weight=0.5, max_weight=2):
        """
        Initialize the B-factor weight generator.
        
        Args:
            bfactor_csv_path (str, optional): Path to CSV file containing B-factor data
            min_weight (float): Minimum weight to assign (default: 0.5)
            max_weight (float): Maximum weight to assign (default: 2.0)
        """
        # Store weight range parameters
        self.min_weight = min_weight  # Minimum weight for normalization
        self.max_weight = max_weight  # Maximum weight for normalization
        
        # Make the path parameter optional with a default fallback
        if bfactor_csv_path is None:
            # Try different common paths
            possible_paths = [
                "intermediate_files/bfactor_winding_lrr_segments.csv"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    bfactor_csv_path = path
                    break
            
            if bfactor_csv_path is None:
                print("Warning: B-factor CSV file not found. Using default weights.")
                self.bfactor_data = {}
                return

        try:
            # Load B-factor data from CSV file
            self.bfactor_data = self._load_bfactor_data(bfactor_csv_path)
        except Exception as e:
            print(f"Warning: Failed to load B-factor data: {e}. Using default weights.")
            self.bfactor_data = {}
        
        # Store weight range parameters
        self.min_weight = min_weight  # Minimum weight for normalization
        self.max_weight = max_weight  # Maximum weight for normalization
        
    def _load_bfactor_data(self, csv_path):
        """
        Load and process B-factor data from CSV.
        
        This method handles the conversion between different protein identifier formats
        to ensure compatibility with the training data.
        
        Args:
            csv_path (str): Path to CSV file with B-factor data
            
        Returns:
            dict: Dictionary mapping protein keys to their B-factor data
        """
        # Load CSV data
        df = pd.read_csv(csv_path)
        
        # Group by protein to create a dictionary of protein-specific data
        protein_data = {}
        for protein_key, group in df.groupby('Protein Key'):
            # Convert protein key format to match training data format
            # Assuming protein_key is in format "Species_LocusID_Receptor"
            parts = protein_key.split('_')
            if len(parts) >= 3:
                # Extract and reformat components
                species = parts[0].replace('_', ' ')
                locus = parts[1]
                receptor = parts[2]
                # Create training format key: "Species|LocusID|Receptor"
                training_key = f"{species}|{locus}|{receptor}"
                
                # Store both indices and B-factor values
                protein_data[training_key] = {
                    'residue_idx': group['Residue Index'].values,  # Position indices
                    'bfactors': group['Filtered B-Factor'].values  # B-factor values
                }
                
                # Also store with original key as fallback for compatibility
                protein_data[protein_key] = {
                    'residue_idx': group['Residue Index'].values,
                    'bfactors': group['Filtered B-Factor'].values
                }
        return protein_data
    
    def get_weights(self, protein_key, sequence_length):
        """
        Generate position-specific weights for a protein sequence based on B-factors.
        
        Higher weights emphasize regions with higher B-factors (more flexible),
        while lower weights de-emphasize more rigid regions.
        
        Args:
            protein_key (str): Key identifying the protein
            sequence_length (int): Length of the sequence to generate weights for
            
        Returns:
            torch.Tensor: Tensor of position-specific weights (length = sequence_length)
        """
        # Default weights - start with minimum weight for all positions
        weights = torch.ones(sequence_length) * self.min_weight
        
        # Try to find the protein in the B-factor data
        # First try with the provided key
        if protein_key in self.bfactor_data:
            data = self.bfactor_data[protein_key]
        else:
            # If not found, try converting the training format to B-factor format
            converted_key = protein_key.replace('|', '_').replace(' ', '_')
            if converted_key in self.bfactor_data:
                data = self.bfactor_data[converted_key]
            else:
                # If still not found, return default weights
                return weights
        
        # Extract B-factors and corresponding residue indices
        bfactors = data['bfactors']
        residue_idx = data['residue_idx']
        
        # Generate weights only for positions with positive B-factors
        pos_mask = bfactors > 0
        if pos_mask.any():
            # Extract positive B-factors
            pos_bfactors = bfactors[pos_mask]
            
            # Normalize B-factors to the specified weight range
            # Higher B-factors result in higher weights
            pos_weights = self.min_weight + (self.max_weight - self.min_weight) * (
                pos_bfactors / pos_bfactors.max()
            )
            
            # Assign weights to corresponding positions in the sequence
            for idx, weight in zip(residue_idx[pos_mask], pos_weights):
                if idx < sequence_length:
                    weights[idx] = weight
        
        return weights

######################################################################
# Main ESM Model with B-Factor Weighting and Chemical Feature Integration
######################################################################

class ESMBfactorWeightedFeatures(nn.Module):
    """
    ESM model with B-factor weighted features for peptide-receptor interaction prediction.
    
    This model uses the ESM2 protein language model as a backbone and enhances it with:
    1. B-factor weighting to emphasize structurally important regions
    2. Chemical feature integration via FiLM conditioning
    3. Targeted fine-tuning by freezing early layers
    
    The model is designed for peptide-receptor interaction prediction with 3 output classes:
    - Class 0: No interaction
    - Class 1: Weak interaction  
    - Class 2: Strong interaction
    
    Architecture overview:
    - ESM2 backbone for sequence embeddings
    - B-factor weight generator for structural emphasis
    - FiLM layer for chemical feature conditioning
    - Classification head for final predictions
    """
    def __init__(self, args, num_classes=3):
        """
        Initialize the ESM model with B-factor weighted features.
        
        Args:
            args: Configuration arguments containing model hyperparameters
            num_classes (int): Number of output classes (default: 3)
        """
        super().__init__()
        
        # Load pretrained ESM2 model and tokenizer
        self.esm = AutoModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        
        # notes: other size models from ESM2:
        # Checkpoint name	Num layers	Num parameters
        # esm2_t48_15B_UR50D	48	15B
        # esm2_t36_3B_UR50D	36	3B
        # esm2_t33_650M_UR50D	33	650M
        # esm2_t30_150M_UR50D	30	150M
        # esm2_t12_35M_UR50D	12	35M
        # esm2_t6_8M_UR50D	6	8M

        # --- Debug Tokenizer Info ---
        # This helps diagnose tokenization issues during development
        print(f"DEBUG Tokenizer Info:")
        print(f"  - eos_token: {self.tokenizer.eos_token}, eos_token_id: {self.tokenizer.eos_token_id}")
        print(f"  - cls_token: {self.tokenizer.cls_token}, cls_token_id: {self.tokenizer.cls_token_id}")
        print(f"  - pad_token: {self.tokenizer.pad_token}, pad_token_id: {self.tokenizer.pad_token_id}")
        print(f"  - unk_token: {self.tokenizer.unk_token}, unk_token_id: {self.tokenizer.unk_token_id}")
        # --- End Debug Tokenizer ---

        # Define the separator token ID for splitting peptide and receptor sequences
        # Default to EOS (End-of-Sequence) token
        self.separator_token_id = self.tokenizer.eos_token_id
        if self.separator_token_id is None:
             # Fallback logic if EOS is None (shouldn't happen for ESM2)
             print("WARNING: EOS token ID is None. Check tokenizer configuration.")
             # Could add: raise ValueError("Could not find a suitable separator token ID")

        # Freeze early layers of the ESM model to preserve learned protein representations
        # Only fine-tune the later layers for the specific task
        modules_to_freeze = [
            self.esm.embeddings,  # Freeze embedding layer
            *self.esm.encoder.layer[:5]  # Freeze first 5 transformer layers
        ]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
        
        # Get feature dimension from ESM model
        self.hidden_size = self.esm.config.hidden_size
        
        # Initialize FiLM layer for feature conditioning
        self.film = FiLMWithConcatenation(self.hidden_size)
        
        # Classification head - transforms ESM features to class predictions
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),  # Reduce dimension
            nn.LayerNorm(self.hidden_size // 2),  # Normalize features
            nn.ReLU(),  # Non-linearity
            nn.Dropout(0.2),  # Regularization
            nn.Linear(self.hidden_size // 2, num_classes)  # Output layer
        )
        
        # Initialize B-factor weight generator
        bfactor_path = getattr(args, 'bfactor_csv_path', None)  # Get path from args if provided
        self.bfactor_weights = BFactorWeightGenerator(
            bfactor_csv_path=bfactor_path,
            min_weight=0.5,  # Minimum position weight
            max_weight=2.0   # Maximum position weight
        )
        
        # Loss function with label smoothing for better generalization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.losses = ["ce"]  # Track which loss types are used
        
        # Save configuration hyperparameters
        self.save_hyperparameters(args)

    def save_hyperparameters(self, args):
        """
        Save model hyperparameters for reproducibility and checkpointing.
        
        Args:
            args: Configuration arguments to save
        """
        self.hparams = args

    def forward(self, batch_x):
        """
        Forward pass applying B-factor weighting to both embeddings and chemical features.
        
        This method implements the core functionality:
        1. ESM embedding generation for combined peptide-receptor sequences
        2. B-factor weight calculation and application
        3. Chemical feature processing and weighting
        4. FiLM conditioning for feature integration
        5. Final classification
        
        Args:
            batch_x: Input batch with tokenized sequences and chemical features
            
        Returns:
            torch.Tensor: Classification logits (batch_size, num_classes)
        """
        # Handle different input formats
        if isinstance(batch_x, dict) and 'x' in batch_x:
            batch_x = batch_x['x']
        
        # Extract inputs and prepare dimensions
        combined_tokens = batch_x['combined_tokens']  # Tokenized combined sequences
        combined_mask = batch_x['combined_mask'].bool()  # Attention mask (token vs padding)
        batch_size = combined_tokens.shape[0]  # Number of sequences in batch
        seq_len = combined_mask.shape[1]  # Length of padded sequences
        device = combined_mask.device  # Device (CPU/GPU) of input tensors
        
        # Get ESM embeddings - forward pass through the ESM model
        outputs = self.esm(
            input_ids=combined_tokens,
            attention_mask=combined_mask,
            output_hidden_states=True  # Get all hidden states
        )
        sequence_output = outputs.last_hidden_state  # Last layer embeddings
        
        # Generate B-factor based weights for each receptor in the batch
        # These weights emphasize structurally important regions
        all_receptor_weights = []
        receptor_ids = batch_x['receptor_id']  # List of receptor identifiers
        for i in range(batch_size):
            single_receptor_id = receptor_ids[i]
            # Get weights based on B-factors for this specific receptor
            weights = self.bfactor_weights.get_weights(single_receptor_id, seq_len)
            all_receptor_weights.append(weights)
            
        # Stack weights into a single tensor [batch_size, seq_len]
        receptor_weights = torch.stack(all_receptor_weights).to(device)
        
        # Locate the separator token positions to distinguish peptide from receptor
        separator_token_id_to_use = self.separator_token_id
        if separator_token_id_to_use is None:
            raise ValueError("Separator token ID is None during forward pass. Check initialization.")
        
        # Create a mask showing where the separator token appears
        separator_mask = (combined_tokens == separator_token_id_to_use)
        if not isinstance(separator_mask, torch.Tensor):
             raise TypeError(f"Comparison resulted in type {type(separator_mask)}, expected torch.Tensor.")
        
        # Get the column indices where separator tokens appear for each sequence
        sep_positions = separator_mask.nonzero(as_tuple=True)[1]
        
        # Create a mask for receptor positions (tokens after the separator)
        # This distinguishes peptide from receptor portions in the combined sequence
        receptor_mask = torch.zeros_like(combined_mask, dtype=torch.bool)
        for i in range(batch_size):
            # Ensure sep_positions has an entry for batch item i
            if i < len(sep_positions) and sep_positions[i] < seq_len - 1:
                # Mark all positions after separator as receptor positions
                receptor_mask[i, sep_positions[i]+1:] = True
            elif i >= len(sep_positions):
                 print(f"Warning: Separator token not found for batch item {i}. Receptor mask will be empty.")
            # Skip if separator is the last token (no receptor portion)
        
        # Apply B-factor weights to receptor portion of sequence embeddings
        # This emphasizes structurally important regions
        weighted_sequence_output = sequence_output.clone()
        weighted_sequence_output[receptor_mask] = (
            sequence_output[receptor_mask] * 
            receptor_weights[receptor_mask].unsqueeze(-1)  # Expand to match feature dimension
        )
        
        # Pool for context vector using weighted embeddings
        # This creates a single vector representation of the entire sequence
        masked_output = weighted_sequence_output.masked_fill(~combined_mask.unsqueeze(-1), -torch.inf)
        pooled_output, _ = torch.max(masked_output, dim=1)  # Max pooling across sequence
        
        # Apply B-factor weights to receptor chemical features
        # This emphasizes chemical properties in structurally important regions
        chemical_features = []
        for feat_name in ['bulkiness', 'charge', 'hydrophobicity']:
            # Extract peptide and receptor features
            seq_feat = batch_x[f'seq_{feat_name}']  # Peptide features
            rec_feat = batch_x[f'rec_{feat_name}']  # Receptor features
            
            # Apply B-factor weights to receptor features
            weighted_rec_feat = rec_feat * receptor_weights  # Element-wise multiplication
            
            # Apply only to receptor positions (using the receptor mask)
            weighted_rec_feat = torch.where(receptor_mask, weighted_rec_feat, rec_feat)
            
            # Add both peptide and weighted receptor features to the list
            chemical_features.extend([seq_feat, weighted_rec_feat])
        
        # Stack all chemical features into a single tensor [batch_size, seq_len, 6]
        chemical_features = torch.stack(chemical_features, dim=-1)
        
        # Apply FiLM conditioning with weighted sequence output and chemical features
        # This modulates sequence features with chemical properties
        conditioned_output = self.film(weighted_sequence_output, pooled_output, chemical_features)
        
        # Pool the conditioned output for classification
        # This aggregates the conditioned features into a single vector per sequence
        masked_conditioned = conditioned_output.masked_fill(~combined_mask.unsqueeze(-1), -torch.inf)
        final_pooled, _ = torch.max(masked_conditioned, dim=1)  # Max pooling
        
        # Classify the pooled features into interaction classes
        logits = self.classifier(final_pooled)
        return logits

    def training_step(self, batch, batch_idx):
        """
        Training step with L2 regularization to prevent overfitting.
        
        Args:
            batch: Batch of training data containing inputs and labels
            batch_idx: Index of the current batch (for logging purposes)
            
        Returns:
            torch.Tensor: Loss value for this batch
        """
        # Forward pass to get logits
        logits = self(batch['x'])
        # Extract ground truth labels
        labels = batch['y']
        
        # Add L2 regularization to prevent overfitting
        l2_lambda = 0.01  # Regularization strength
        l2_reg = torch.tensor(0., device=logits.device, requires_grad=True)
        for param in self.parameters():
            l2_reg = l2_reg + torch.norm(param)  # Sum of parameter norms
        
        # Combine cross-entropy loss with L2 regularization
        loss = self.criterion(logits, labels) + l2_lambda * l2_reg
        return loss

    ######################################################################
    # Data Processing and Tokenization Functions
    ######################################################################

    def collate_fn(self, batch):
        """
        Collate function for batching data during training and evaluation.
        
        This function handles the complex data preprocessing pipeline:
        1. Combines peptide and receptor sequences with a separator
        2. Tokenizes the combined sequences using ESM tokenizer
        3. Processes chemical features to match tokenized sequence length
        4. Stores metadata for later use in evaluation and prediction saving
        
        Args:
            batch: List of individual data samples from the dataset
                
        Returns:
            dict: Batch dictionary with processed inputs ready for model forward pass
        """
        # Use the model's tokenizer
        tokenizer = self.tokenizer 
        separator_token = tokenizer.eos_token  # Use EOS token as separator

        if separator_token is None:
            raise ValueError("EOS token is None in collate_fn. Check tokenizer configuration.")

        # Extract sequences and metadata
        sequences = [str(item['peptide_x']) for item in batch]  # Ligand sequences
        receptors = [str(item['receptor_x']) for item in batch]  # Receptor sequences
        
        # Store all metadata for later use in get_stats
        self.header_names = [item.get('Header_Name', '') for item in batch]
        self.plant_species = [item.get('plant_species', '') for item in batch]
        self.receptors_meta = [item.get('receptor', '') for item in batch]  # Renamed to avoid conflict
        self.locus_ids = [item.get('locus_id', '') for item in batch]
        self.epitope_seqs = sequences
        self.receptor_seqs = receptors
        
        # Create receptor IDs for B-factor weight lookup using the format: "plant_species|locus_id|receptor"
        receptor_ids = [
            f"{item['plant_species']}|{item['locus_id']}|{item['receptor']}" 
            for item in batch
        ]
        
        # Combine ligand and receptor sequences with separator token
        combined = [f"{seq} {separator_token} {rec}" for seq, rec in zip(sequences, receptors)]
        
        # Tokenize with padding and truncation
        encoded = tokenizer(
            combined,
            padding=True,  # Pad shorter sequences
            truncation=True,  # Truncate longer sequences
            max_length=1024,  # Maximum sequence length
            return_tensors='pt'  # Return PyTorch tensors
        )
        
        # Process chemical features to match tokenized sequence length
        def process_features(batch, prefix):
            """
            Helper function to process chemical features for either peptide or receptor.
            
            This function handles the conversion of chemical feature data from the input
            format to tensors that match the tokenized sequence length.
            """
            features = {}
            for feat in ['bulkiness', 'charge', 'hydrophobicity']:
                # Map to the actual column names from the R script
                if prefix == 'sequence':
                    key = f"Sequence_{feat.capitalize()}"  # e.g., "Sequence_Bulkiness"
                else:  # receptor
                    key = f"Receptor_{feat.capitalize()}"  # e.g., "Receptor_Bulkiness"
                    
                # Get tokenized sequence length to match feature dimensions
                feature_length = encoded['input_ids'].size(1)
                feature_list = []
                
                for item in batch:
                    if key in item:
                        # Convert feature string to tensor if it's a comma-separated string
                        if isinstance(item[key], str):
                            feature_values = [float(x) for x in item[key].split(',')]
                            item_feature = torch.tensor(feature_values)
                        else:
                            item_feature = torch.tensor(item[key])
                            
                        # Pad or truncate to match tokenized length
                        if len(item_feature) < feature_length:
                            padding = torch.zeros(feature_length - len(item_feature))
                            item_feature = torch.cat([item_feature, padding])
                        elif len(item_feature) > feature_length:
                            item_feature = item_feature[:feature_length]
                        feature_list.append(item_feature)
                    else:
                        feature_list.append(torch.zeros(feature_length))
                features[feat] = torch.stack(feature_list)
            return features
        
        # Process features for both peptide and receptor
        seq_features = process_features(batch, 'sequence')  # Peptide features
        rec_features = process_features(batch, 'receptor')  # Receptor features
        
        # Prepare the output dictionary
        batch_output = {
            'x': {
                'combined_tokens': encoded['input_ids'],
                'combined_mask': encoded['attention_mask'],
                'seq_bulkiness': seq_features['bulkiness'],
                'seq_charge': seq_features['charge'],
                'seq_hydrophobicity': seq_features['hydrophobicity'],
                'rec_bulkiness': rec_features['bulkiness'],
                'rec_charge': rec_features['charge'],
                'rec_hydrophobicity': rec_features['hydrophobicity'],
                'receptor_id': receptor_ids,
            }
        }

        # Include labels if they exist in the batch (for training/evaluation)
        if 'y' in batch[0]:
            labels = [item['y'] for item in batch]
            batch_output['y'] = torch.tensor(labels, dtype=torch.long)

        return batch_output

    ######################################################################
    # Model Utility Functions
    ######################################################################

    def get_tokenizer(self):
        """
        Get the model's tokenizer for external use.
        
        Returns:
            Tokenizer: The ESM tokenizer used by this model
        """
        return self.tokenizer

    def batch_decode(self, batch):
        """
        Decode tokenized sequences back to text for analysis and debugging.
        
        Args:
            batch: Batch containing tokenized sequences
            
        Returns:
            list: List of decoded sequences split into peptide and receptor parts
        """
        # Extract tokens based on input format
        if isinstance(batch, dict) and 'x' in batch:
            tokens = batch['x']['combined_tokens']
        else:
            tokens = batch['combined_tokens']
            
        # Decode tokens to text
        decoded = self.tokenizer.batch_decode(
            tokens,
            skip_special_tokens=True  # Remove special tokens like [PAD]
        )
        
        # ESM2 doesn't have a dedicated separator token (sep_token is None)
        # We previously defaulted to using EOS token (<eos>) as separator
        # in self.separator_token_id
        
        # Split each sequence at the EOS token to get peptide and receptor parts
        # Note: self.tokenizer.eos_token is "<eos>"
        #return [seq.split(self.tokenizer.sep_token) for seq in decoded]
        return [seq.split(self.tokenizer.eos_token) for seq in decoded]

    def get_pr(self, logits):
        """
        Convert logits to class probabilities using softmax.
        
        Args:
            logits: Raw output logits from the model classifier
            
        Returns:
            torch.Tensor: Softmax probabilities for each class
        """
        return torch.softmax(logits, dim=-1)

    ######################################################################
    # Evaluation and Prediction Analysis Functions
    ######################################################################

    def get_stats(self, pr, gt=None, train=False, sequences=None, metadata=None):
        """
        Calculate evaluation metrics and save predictions with original metadata.
        
        This function computes comprehensive evaluation metrics and saves detailed
        predictions to CSV files for further analysis. It handles both training
        evaluation (with ground truth) and inference (prediction only).
        
        Args:
            pr: Predicted probabilities from the model
            gt: Ground truth labels (optional, for evaluation)
            train: Whether these are training or test metrics
            sequences: List of decoded sequences (optional, for debugging)
            metadata: Dictionary containing all metadata from all batches
            
        Returns:
            dict: Dictionary containing evaluation metrics and prediction info
        """
        # Get predicted class labels
        pred_labels = pr.argmax(dim=-1)
        
        # Convert predictions to numpy arrays
        probs = pr.cpu().numpy()
        
        # Create DataFrame with probabilities and predicted labels
        results_df = pd.DataFrame(probs, columns=['prob_class0', 'prob_class1', 'prob_class2'])
        results_df['predicted_label'] = pred_labels.cpu().numpy()

        if gt is not None:
            results_df['ground_truth'] = gt.cpu().numpy()
        
        num_predictions = len(results_df)
        
        # Use passed metadata if available
        if metadata and all(key in metadata for key in ['header_names', 'plant_species', 'receptors_meta', 'locus_ids', 'epitope_seqs', 'receptor_seqs']):
            if len(metadata['epitope_seqs']) == num_predictions:
                results_df['Header_Name'] = metadata['header_names']
                results_df['plant_species'] = metadata['plant_species']
                results_df['receptor'] = metadata['receptors_meta']
                results_df['locus_id'] = metadata['locus_ids']
                results_df['Sequence'] = metadata['epitope_seqs']
                results_df['receptor_sequence'] = metadata['receptor_seqs']
                print(f"Successfully added metadata for {num_predictions} predictions")
            else:
                print(f"Warning: Metadata length mismatch. Predictions: {num_predictions}, Metadata: {len(metadata['epitope_seqs'])}")
                # Fill with truncated or padded metadata
                for key, col_name in [('header_names', 'Header_Name'), ('plant_species', 'plant_species'), 
                                      ('receptors_meta', 'receptor'), ('locus_ids', 'locus_id'),
                                      ('epitope_seqs', 'Sequence'), ('receptor_seqs', 'receptor_sequence')]:
                    if len(metadata[key]) >= num_predictions:
                        results_df[col_name] = metadata[key][:num_predictions]
                    else:
                        padded_data = metadata[key] + [""] * (num_predictions - len(metadata[key]))
                        results_df[col_name] = padded_data
        else:
            # Use stored metadata if available (fallback for single batch)
            if (hasattr(self, 'epitope_seqs') and hasattr(self, 'receptor_seqs') and 
                hasattr(self, 'header_names') and hasattr(self, 'plant_species') and
                hasattr(self, 'receptors_meta') and hasattr(self, 'locus_ids') and
                len(self.epitope_seqs) == num_predictions):
                
                results_df['Header_Name'] = self.header_names
                results_df['plant_species'] = self.plant_species
                results_df['receptor'] = self.receptors_meta
                results_df['locus_id'] = self.locus_ids
                results_df['Sequence'] = self.epitope_seqs
                results_df['receptor_sequence'] = self.receptor_seqs
                print(f"Successfully added stored metadata for {num_predictions} predictions")
            else:
                # No metadata available or mismatch
                results_df['Header_Name'] = [""] * num_predictions
                results_df['plant_species'] = [""] * num_predictions
                results_df['receptor'] = [""] * num_predictions
                results_df['locus_id'] = [""] * num_predictions
                results_df['Sequence'] = [""] * num_predictions
                results_df['receptor_sequence'] = [""] * num_predictions
                print("Warning: No matching metadata available")
        
        # Save predictions to CSV
        results_df.to_csv('predictions.csv', index=False)
        print(f"Saved predictions to predictions.csv with {len(results_df)} rows")
        
        # Calculate proper evaluation metrics if ground truth is available
        if gt is not None:
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
            
            y_true = gt.cpu().numpy()
            y_pred = pred_labels.cpu().numpy()
            y_prob = probs
            
            # Calculate metrics
            metrics = {}
            prefix = "train_" if train else "test_"
            
            # Basic classification metrics
            metrics[f"{prefix}acc"] = accuracy_score(y_true, y_pred)
            metrics[f"{prefix}f1_macro"] = f1_score(y_true, y_pred, average='macro')
            metrics[f"{prefix}f1_weighted"] = f1_score(y_true, y_pred, average='weighted')
            
            # Multi-class ROC AUC
            try:
                metrics[f"{prefix}auroc"] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            except:
                metrics[f"{prefix}auroc"] = 0.0
                
            # Per-class AUPRC
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
            for i in range(3):
                try:
                    metrics[f"{prefix}auprc_class{i}"] = average_precision_score(y_true_bin[:, i], y_prob[:, i])
                except:
                    metrics[f"{prefix}auprc_class{i}"] = 0.0
            
            # Add loss if available
            metrics[f"{prefix}loss"] = 0.0  # Placeholder
            
            return metrics
        else:
            # No ground truth available
            return {
                'predictions_saved': True,
                'num_predictions': len(results_df)
            }

######################################################################
# Dataset Class for Peptide-Receptor Interaction Data
######################################################################

class PeptideSeqWithReceptorDataset(torch.utils.data.Dataset):
    """
    Dataset class for handling peptide-receptor interaction data.
    
    This dataset handles:
    - Peptide and receptor protein sequences
    - Associated metadata (species, locus, receptor type)
    - Interaction labels (if available for training/evaluation)
    
    The dataset is designed to work with the collate_fn method of the model
    to properly format data for training and inference.
    """
    def __init__(self, df):
        """
        Initialize the dataset from a pandas DataFrame.
        
        Args:
            df: DataFrame containing columns:
                - Sequence: peptide sequences
                - receptor_sequence: receptor sequences  
                - plant_species: species information
                - locus_id: locus identifiers
                - receptor: receptor type/name
                - Header_Name: sequence identifiers (optional)
                - y: interaction labels (optional, for training)
        """
        self.peptide_x = df['Sequence']
        self.receptor_x = df['receptor_sequence']
        self.plant_species = df['plant_species']
        self.locus_id = df['locus_id']
        self.receptor = df['receptor']
        self.header_name = df['Header_Name'] if 'Header_Name' in df else df.index.astype(str)
        self.name = "PeptideSeqWithReceptorDataset"
        self.y = df['y'] if 'y' in df else None

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.peptide_x)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            dict: Dictionary containing sample data and metadata
        """
        item = {
            'peptide_x': self.peptide_x.iloc[idx],
            'receptor_x': self.receptor_x.iloc[idx],
            'plant_species': self.plant_species.iloc[idx],
            'locus_id': self.locus_id.iloc[idx],
            'receptor': self.receptor.iloc[idx],
            'Header_Name': self.header_name.iloc[idx]
        }
        if self.y is not None:
            item['y'] = self.y.iloc[idx]
        return item