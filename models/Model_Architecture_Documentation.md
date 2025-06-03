# Model Architecture Documentation

## ESMBfactorWeightedFeatures

### Base Architecture
- **Base Model**: ESM2-t30 with selective fine-tuning
- **Input**: Sequences with B-factor and chemical features
- **Architecture**:
  - ESM2 encoder with configurable layer freezing
  - B-factor based position weighting
  - Enhanced FiLMWithConcatenation for feature integration
  - Comprehensive classification head

### Key Components
- **B-Factor Integration**:
  - BFactorWeightGenerator for position-specific weighting
  - Configurable weight range (default: 0.5 to 2.0)
  - Handles missing B-factor data gracefully
- **Chemical Feature Processing**:
  - Separate processing for sequence and receptor features
  - Enhanced projection network with LayerNorm
  - Position-wise feature modulation

### Features
- Structural information integration through B-factors
- Robust error handling for missing structural data
- Enhanced chemical feature processing
- Support for both training and evaluation modes
- Comprehensive evaluation metrics

## Common Features Across All Models

### Base Architecture
- All models use ESM2-t8 as the backbone
- Implement 3-class classification for immune response prediction
- Use FiLM-based feature modulation
- Support distributed training

### Training Features
- Comprehensive evaluation metrics
- Checkpoint saving and loading
- WandB integration for experiment tracking
- Proper handling of variable-length sequences

### Evaluation Metrics
- Accuracy
- Macro and weighted F1 scores
- ROC AUC (one-vs-rest)
- PR AUC for each class
- Prediction saving capabilities

### Data Processing
- Efficient batching with custom collate functions
- Support for chemical feature integration
- Proper sequence masking and padding
- Tokenization using ESM2 tokenizer

