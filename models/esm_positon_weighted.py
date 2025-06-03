import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os

class FiLMWithConcatenation(nn.Module):
    """
    FiLM layer that conditions sequence representation with chemical features.
    
    FiLM (Feature-wise Linear Modulation) is a conditional normalization method
    that modulates neural network activations using learned parameters generated
    from conditioning information. Here, we use this technique to condition protein
    sequence embeddings with chemical property features.
    """
    def __init__(self, feature_dim):
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

class BFactorWeightGenerator:
    """
    Generates weights based on B-factors from preprocessed data.
    
    The B-factors here represent positions along a transformed LRR coil structure.
    Each Leucine-Rich Repeat (LRR) is mapped onto a standardized coil, where the
    B-factor values indicate the numerical position along this coil. This allows
    us to weight different regions of the LRR based on their relative position
    in the repeating structural motif.
    """
    def __init__(self, bfactor_csv_path=None, min_weight=0.5, max_weight=2):
        """
        Initialize the B-factor weight generator.
        
        Args:
            bfactor_csv_path (str, optional): Path to CSV file containing B-factor data
            min_weight (float): Minimum weight to assign (default: 0.5)
            max_weight (float): Maximum weight to assign (default: 2.0)
        """
        # Make the path parameter optional with a default fallback
        if bfactor_csv_path is None:
            # Try different common paths
            possible_paths = [
                "04_Preprocessing_results/bfactor_winding_lrr_segments.csv"
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

class ESMBfactorWeightedFeatures(nn.Module):
    """
    ESM model with B-factor weighted features for peptide-receptor interaction prediction.
    
    This model uses the ESM2 protein language model as a backbone and enhances it with:
    1. B-factor weighting to emphasize structurally important regions
    2. Chemical feature integration via FiLM conditioning
    3. Targeted fine-tuning by freezing early layers
    
    The model is designed for peptide-receptor interaction prediction with 3 output classes.
    """
    def __init__(self, args, num_classes=3):
        """
        Initialize the ESM model with B-factor weighted features.
        
        Args:
            args: Configuration arguments
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
        Save model hyperparameters.
        
        Args:
            args: Configuration arguments to save
        """
        self.hparams = args

    def forward(self, batch_x):
        """
        Forward pass applying B-factor weighting to both embeddings and chemical features.
        
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
            batch: Batch of training data
            batch_idx: Index of the current batch
            
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

    def collate_fn(self, batch):
        """
        Collate function for batching data during training and evaluation.
        
        This function:
        1. Combines peptide and receptor sequences with a separator
        2. Tokenizes the combined sequences
        3. Processes chemical features to match tokenized sequence length
        
        Args:
            batch: List of individual data samples
            
        Returns:
            dict: Batch dictionary with processed inputs and labels
        """
        # Use the model's tokenizer
        tokenizer = self.tokenizer 
        separator_token = tokenizer.eos_token  # Use EOS token as separator

        if separator_token is None:
            raise ValueError("EOS token is None in collate_fn. Check tokenizer configuration.")

        # Extract sequences and labels
        sequences = [str(item['peptide_x']) for item in batch]  # Peptide sequences
        receptors = [str(item['receptor_x']) for item in batch]  # Receptor sequences
        labels = torch.tensor([item['y'] for item in batch])  # Ground truth labels
        
        # Get receptor IDs for B-factor weight lookup
        receptor_ids = [str(item.get('receptor_id', '')) for item in batch]
        
        # Combine peptide and receptor sequences with separator token between them
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
            """Helper function to process chemical features for either peptide or receptor."""
            features = {}
            for feat in ['bulkiness', 'charge', 'hydrophobicity']:
                key = f"{prefix}_{feat}"
                # Get tokenized sequence length to match feature dimensions
                feature_length = encoded['input_ids'].size(1)
                feature_list = []
                
                for item in batch:
                    if key in item:
                         # Ensure the feature tensor matches tokenized sequence length
                         item_feature = torch.tensor(item[key])
                         
                         # Pad or truncate features to match tokenized length
                         if len(item_feature) < feature_length:
                              # Pad with zeros if feature is shorter
                              padding = torch.zeros(feature_length - len(item_feature))
                              item_feature = torch.cat([item_feature, padding])
                         elif len(item_feature) > feature_length:
                              # Truncate if feature is longer
                              item_feature = item_feature[:feature_length]
                              
                         feature_list.append(item_feature)
                    else:
                         # If feature not found, use zeros
                         feature_list.append(torch.zeros(feature_length))
                         
                # Stack features into tensor [batch_size, seq_len]
                features[feat] = torch.stack(feature_list)
            return features
        
        # Process features for both peptide and receptor
        seq_features = process_features(batch, 'sequence')  # Peptide features
        rec_features = process_features(batch, 'receptor')  # Receptor features
        
        # Return formatted batch dictionary
        return {
            'x': {
                'combined_tokens': encoded['input_ids'],  # Tokenized sequences
                'combined_mask': encoded['attention_mask'],  # Attention mask
                'seq_bulkiness': seq_features['bulkiness'],  # Peptide bulkiness
                'seq_charge': seq_features['charge'],  # Peptide charge
                'seq_hydrophobicity': seq_features['hydrophobicity'],  # Peptide hydrophobicity
                'rec_bulkiness': rec_features['bulkiness'],  # Receptor bulkiness
                'rec_charge': rec_features['charge'],  # Receptor charge
                'rec_hydrophobicity': rec_features['hydrophobicity'],  # Receptor hydrophobicity
                'receptor_id': receptor_ids,  # Receptor identifiers for B-factor lookup
            },
            'y': labels  # Ground truth class labels
        }

    def get_tokenizer(self):
        """
        Get the model's tokenizer.
        
        Returns:
            Tokenizer: The ESM tokenizer
        """
        return self.tokenizer

    def batch_decode(self, batch):
        """
        Decode tokenized sequences back to text.
        
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
        Convert logits to class probabilities.
        
        Args:
            logits: Raw output logits from the model
            
        Returns:
            torch.Tensor: Softmax probabilities
        """
        return torch.softmax(logits, dim=-1)

    def get_stats(self, gt, pr, train=False):
        """
        Calculate evaluation metrics and include predicted probabilities.
        
        Args:
            gt: Ground truth labels
            pr: Predicted probabilities
            train: Whether these are training or test metrics
            
        Returns:
            dict: Dictionary of evaluation metrics including probabilities
        """
        # Set prefix for metric names
        prefix = "train" if train else "test"
        
        # Get predicted class labels
        pred_labels = pr.argmax(dim=-1)
        
        # Calculate standard classification metrics
        stats = {
            f"{prefix}_acc": accuracy_score(gt.cpu(), pred_labels.cpu()),  # Accuracy
            f"{prefix}_f1_macro": f1_score(gt.cpu(), pred_labels.cpu(), average='macro'),  # Macro F1
            f"{prefix}_f1_weighted": f1_score(gt.cpu(), pred_labels.cpu(), average='weighted')  # Weighted F1
        }
        
        try:
            # Calculate area under ROC curve (multi-class)
            stats[f"{prefix}_auroc"] = roc_auc_score(gt.cpu(), pr.cpu(), multi_class='ovr')
            
            # Convert ground truth to one-hot encoding for per-class metrics
            gt_onehot = np.eye(3)[gt.cpu()]  # One-hot encoding for 3 classes
            pr_np = pr.cpu().numpy()  # Convert predictions to numpy
            
            # Calculate precision-recall AUC for each class
            for i in range(3):
                precision, recall, _ = precision_recall_curve(gt_onehot[:, i], pr_np[:, i])
                stats[f"{prefix}_auprc_class{i}"] = auc(recall, precision)
            
            # Calculate macro-average precision-recall AUC
            stats[f"{prefix}_auprc_macro"] = np.mean([stats[f"{prefix}_auprc_class{i}"] for i in range(3)])
            
        except:
            # Handle errors (e.g., if only one class is present)
            stats[f"{prefix}_auroc"] = 0.0
            stats[f"{prefix}_auprc_macro"] = 0.0
            for i in range(3):
                stats[f"{prefix}_auprc_class{i}"] = 0.0

        # Save test probabilities to CSV if this is test data and final epoch
        #if not train and hasattr(self, 'current_epoch') and self.current_epoch == self.hparams.epochs - 1:
        if not train:
            # Convert predictions and ground truth to numpy arrays
            probs = pr.cpu().numpy()
            labels = gt.cpu().numpy()
            
            # Create DataFrame with probabilities and true labels
            # Create DataFrame with probabilities, labels and sequences
            results_df = pd.DataFrame(probs, columns=['prob_class0', 'prob_class1', 'prob_class2'])
            if labels is not None:
                results_df['true_label'] = labels
            results_df['predicted_label'] = pred_labels.cpu().numpy()
            
            # Add receptor and epitope sequences
            if hasattr(self, 'receptor_seqs') and hasattr(self, 'epitope_seqs'):
                results_df['receptor_sequence'] = self.receptor_seqs
                results_df['epitope_sequence'] = self.epitope_seqs
                
            # Save to CSV
            results_df.to_csv('test_predictions.csv', index=False)
            
        # Keep probabilities in stats dictionary for consistency
        #stats[f"{prefix}_probabilities"] = pr.cpu().numpy()
            
        return stats