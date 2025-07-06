# Neurotokenization of Brain MRI for Early Alzheimer's Detection

This project implements a novel methodology for Alzheimer's disease detection using structured neurotokens derived from T1-weighted MRI images processed through FreeSurfer. The approach generates region-specific quantitative descriptors and processes them through transformer architectures for binary classification.

## Overview

Traditional deep learning approaches for Alzheimer's detection rely on raw MRI voxel data and large convolutional networks, requiring substantial computational resources and often lacking interpretability. This study introduces neurotokens: structured, region-specific quantitative descriptors that enable efficient training on smaller datasets while maintaining interpretability.

## Methodology

1. **Data Preprocessing**: T1-weighted MRI images are processed through FreeSurfer's `recon-all` pipeline
2. **Feature Extraction**: Cortical thickness, hippocampal volume, ventricle size, and surface curvature are extracted for curated brain regions
3. **Neurotokenization**: Each anatomical region becomes a token with a fixed-length vector of attributes
4. **Model Architecture**: Transformer encoder model ingests these tokens with positional embeddings reflecting anatomical adjacency
5. **Training**: Cross-entropy loss with stratified sampling, 80/20 train/test split

## Data Organization

The project uses the OASIS-2 dataset with the following structure:

```
OASIS_Processed/
├── OAS2_0001/
│   ├── session_1/
│   │   └── T1_avg.mgz
│   ├── session_2/
│   │   └── T1_avg.mgz
│   └── metadata.json
├── OAS2_0002/
└── master_index.csv
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Data Processing

```bash
# Convert NIfTI files to FreeSurfer format
python nifti_to_freesurfer.py /path/to/nifti/files /path/to/output

# Organize OASIS-2 data
./batch_organize_oasis.sh

# Generate metadata
python -c "import oasis2_utils as o2u; df = o2u.parse_oasis_demographics('Oasis Longitudinal Demographics.xlsx'); o2u.create_subject_metadata_files(df, '/path/to/processed'); o2u.create_master_index_csv(df, '/path/to/processed', '/path/to/processed/master_index.csv')"
```

### FreeSurfer Processing

```bash
# Run FreeSurfer recon-all on processed images
recon-all -i /path/to/T1_avg.mgz -s subject_id -all
```

### Neurotoken Generation

```python
from freesurfer_parser import FreeSurferParser

parser = FreeSurferParser("/path/to/subjects")
stats = parser.parse_subject("subject_id")
tokens = parser.generate_neurotokens(stats, "subject_id")
parser.save_tokens(tokens, "output.json")
```

### Transformer Training

```python
from oasis2_neurotokens import OASIS2NeuroTokenProcessor

processor = OASIS2NeuroTokenProcessor("config.json")
train_dataset, test_dataset = processor.prepare_transformer_dataset(tokens_list, labels)
model = processor.train_transformer(train_dataset, test_dataset)
```

## Results

- Structured transformers on neurotokens outperformed traditional CNNs trained on voxel maps
- Improved interpretability with attention maps showing high weights for hippocampus and entorhinal cortex
- Achieved >90% accuracy on binary classification with reduced data requirements

## Technical Challenges

- **NIfTI Compatibility**: Resolved format confusion between `.nii.gz` and legacy `.nifti.img/.hdr` files
- **External Drive Compatibility**: Addressed macOS read/write permission issues with Seagate external drives
- **FreeSurfer Resource Limits**: Bypassed `mri_synthstrip` memory issues on ARM chips using alternative preprocessing

## Files

- `freesurfer_parser.py`: Core parser for FreeSurfer output files
- `oasis2_neurotokens.py`: OASIS-2 specific processor with transformer implementation
- `oasis2_utils.py`: Utility functions for metadata processing
- `nifti_to_freesurfer.py`: NIfTI to FreeSurfer format converter
- `batch_organize_oasis.sh`: Batch processing script for OASIS-2 data
- `config.py`: Configuration parameters
- `requirements.txt`: Python dependencies

## Citation

This work presents a scalable, interpretable, and performant alternative to raw-image-based Alzheimer's detection, suitable for clinical applications and longitudinal modeling.