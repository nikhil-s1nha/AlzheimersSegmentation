# Temporally-Aware NeuroToken Model (Attempt 3)

This directory contains the implementation of a **hierarchical GRU model** that makes explicit use of longitudinal time information for Alzheimer's detection using neurotoken sequences.

## 🧠 Model Architecture

The temporal model implements a **three-level hierarchical architecture**:

```
1. Session Encoder: GRU processes each session's tokens (28 tokens → 1 session embedding)
2. Time Embedding: Linear layer encodes delay (normalized 0-1)
3. Subject Encoder: GRU processes session embeddings over time
4. Classification: Linear head for binary classification (CN vs Impaired)
```

### Key Features

- **Temporal Awareness**: Incorporates actual time delays between MRI sessions
- **Hierarchical Processing**: Respects the natural structure of longitudinal data
- **Two-Level Padding**: Sessions (outer) and tokens within sessions (inner)
- **Attention Mechanism**: Captures which sessions are most important for classification

## 📁 File Structure

```
temporal_attempt/
├── temporal_data_processor.py      # Converts data to temporal format
├── hierarchical_gru_model.py       # Model architecture
├── temporal_dataset.py             # Dataset with two-level padding
├── train_temporal.py               # Training script
├── eval_temporal.py                # Evaluation script
├── README.md                       # This file
├── temporal_sequences.jsonl        # Generated temporal data
├── models/                         # Training outputs
│   ├── checkpoints/
│   ├── metrics.json
│   ├── training_curves.png
│   └── confusion_matrix.png
└── evaluation/                     # Evaluation outputs
    ├── test_confusion_matrix.png
    ├── test_attention_analysis.png
    ├── test_probability_distribution.png
    └── test_evaluation_report.json
```

## 🚀 Quick Start

### 1. Process Temporal Data

```bash
cd /Volumes/SEAGATE_NIKHIL/neurotokens_project/temporal_attempt
python temporal_data_processor.py
```

This will:
- Load OASIS demographics to extract session timing
- Convert existing neurotoken sequences to temporal format
- Create `temporal_sequences.jsonl` with session-level structure

### 2. Train the Model

```bash
python train_temporal.py
```

This will:
- Train the hierarchical GRU model for 30 epochs
- Use BCEWithLogitsLoss and AdamW optimizer
- Apply early stopping with patience=10
- Save best model and training metrics

### 3. Evaluate the Model

```bash
python eval_temporal.py
```

This will:
- Load the trained model
- Evaluate on test and validation sets
- Generate confusion matrices and attention analysis
- Create comprehensive evaluation reports

## 📊 Data Format

### Input Format

Each subject is represented as:

```json
{
  "subject_id": "OAS2_0001",
  "sessions": [
    { "tokens": [1, 5, 12, ...], "delay": 0.0 },
    { "tokens": [3, 8, 15, ...], "delay": 0.5 },
    ...
  ],
  "label": 1
}
```

Where:
- `tokens`: List of 28 token IDs for that session
- `delay`: Normalized time delay (0-1) since first scan
- `label`: 0 (CN) or 1 (Impaired = MCI + AD)

### Model Input

The model expects:
- `input_ids`: [batch_size, max_sessions=5, max_tokens=28]
- `delays`: [batch_size, max_sessions=5]
- `attention_mask`: [batch_size, max_sessions=5, max_tokens=28]

## 🔧 Model Configuration

```python
config = HierarchicalGRUConfig(
    vocab_size=32,              # Number of unique neurotokens
    token_emb_dim=32,           # Token embedding dimension
    session_hidden_dim=64,      # Session-level GRU hidden size
    subject_hidden_dim=128,     # Subject-level GRU hidden size
    time_emb_dim=16,            # Time embedding dimension
    num_layers=2,               # Number of GRU layers
    dropout=0.3,                # Dropout rate
    max_sessions=5,             # Maximum sessions per subject
    max_tokens=28               # Maximum tokens per session
)
```

## 🎯 Training Parameters

- **Batch Size**: 8 (smaller for hierarchical model)
- **Learning Rate**: 1e-4
- **Weight Decay**: 1e-3
- **Epochs**: 30
- **Early Stopping**: Patience=10
- **Loss Function**: BCEWithLogitsLoss
- **Optimizer**: AdamW
- **Scheduler**: CosineAnnealingLR

## 📈 Expected Performance

Based on the temporal architecture, we expect:

- **Better Performance**: Temporal information should improve classification accuracy
- **Interpretability**: Attention weights show which sessions are most important
- **Robustness**: Hierarchical structure handles variable session counts
- **Clinical Relevance**: Time-aware predictions align with disease progression

## 🔍 Key Innovations

### 1. Temporal Embedding
- Encodes actual time delays between MRI sessions
- Normalized to 0-1 range per subject
- Combined with session embeddings

### 2. Hierarchical Processing
- Session-level GRU processes tokens within each session
- Subject-level GRU processes sessions over time
- Respects natural data hierarchy

### 3. Two-Level Padding
- **Session-level**: Pad to max_sessions=5
- **Token-level**: Pad to max_tokens=28 within each session
- Maintains temporal structure

### 4. Attention Analysis
- Session-level attention shows which timepoints are most important
- Token-level attention within sessions
- Provides interpretability for clinical decisions

## 🧪 Experimental Design

### Why This Approach?

1. **Longitudinal Nature**: Alzheimer's is a progressive disease - temporal information is crucial
2. **Session Boundaries**: MRI sessions have natural temporal structure
3. **Clinical Timing**: Time intervals between scans provide valuable information
4. **Hierarchical Data**: Sessions contain tokens, subjects contain sessions

### Comparison to Previous Attempts

| Aspect | Previous (Binary GRU) | Current (Temporal) |
|--------|----------------------|-------------------|
| **Data Structure** | Flat token sequence | Hierarchical sessions |
| **Temporal Info** | None | Explicit time delays |
| **Model Architecture** | Single GRU | Hierarchical GRU |
| **Padding** | Single level | Two levels |
| **Interpretability** | Limited | Session attention |

## 📋 Usage Examples

### Load and Test Model

```python
from hierarchical_gru_model import HierarchicalGRUConfig, create_hierarchical_model
from temporal_dataset import TemporalNeuroTokenDataset

# Create model
config = HierarchicalGRUConfig()
model = create_hierarchical_model(config)

# Load data
dataset = TemporalNeuroTokenDataset(temporal_sequences, max_sessions=5, max_tokens=28)
sample = dataset[0]

# Forward pass
logits = model(sample['input_ids'], sample['delays'], sample['attention_mask'])
probabilities = torch.sigmoid(logits)
```

### Analyze Attention

```python
# Get attention weights
attention_weights = model.get_attention_weights(input_ids, delays, attention_mask)

# attention_weights shape: [batch_size, max_sessions]
# Shows which sessions the model focuses on
```

## 🎯 Clinical Applications

This temporal model is particularly suited for:

1. **Early Detection**: Temporal patterns may reveal early disease progression
2. **Progression Monitoring**: Track changes over multiple timepoints
3. **Personalized Medicine**: Individual temporal trajectories
4. **Clinical Decision Support**: Interpretable session-level attention

## 📊 Expected Outputs

After training, you'll get:

1. **Model Checkpoints**: Best and final models
2. **Training Curves**: Loss, accuracy, learning rate plots
3. **Confusion Matrices**: Test and validation performance
4. **Attention Analysis**: Which sessions are most important
5. **Probability Distributions**: Model confidence by class
6. **Evaluation Reports**: Comprehensive JSON reports

## 🔬 Future Work

Potential extensions:

1. **Multi-modal Integration**: Combine with other biomarkers
2. **Advanced Temporal Models**: LSTM, Transformer variants
3. **Clinical Validation**: Test on external datasets
4. **Real-time Prediction**: Online learning for new patients
5. **Causal Inference**: Understand temporal causality

## 📞 Support

For questions or issues:

1. Check the logs in the training output
2. Verify data paths and file permissions
3. Ensure sufficient GPU memory for batch processing
4. Review the evaluation reports for detailed metrics

---

**Note**: This temporal approach represents a significant advancement over previous attempts by explicitly modeling the longitudinal nature of Alzheimer's disease progression. 