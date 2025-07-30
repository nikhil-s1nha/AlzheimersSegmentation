# Attempt 1: Multi-class Transformer for NeuroToken Classification

## 🧠 Overview

This is the first attempt at implementing neurotoken-based Alzheimer's detection using a transformer architecture for multi-class classification (CN vs MCI vs AD).

## 📊 Results

- **Accuracy**: 50%
- **F1-Score**: ~33%
- **Key Issue**: Severe class imbalance - model predicted majority class (CN) for most cases

## 🔧 Architecture

- **Model**: Transformer encoder with self-attention
- **Classification**: Multi-class (CN, MCI, AD)
- **Input**: Flattened neurotoken sequences
- **Output**: 3-class probabilities

## 📁 Files

- `train.py` - Training script
- `eval.py` - Evaluation script
- `transformer_model.py` - Transformer architecture
- `dataset.py` - Data loading and preprocessing
- `create_subject_labels.py` - Label generation from demographics

## 🚀 Usage

```bash
python train.py
python eval.py
```

## 🎯 Key Learnings

1. **Neurotoken Concept**: Successfully established the feasibility of token-based MRI analysis
2. **Class Imbalance**: Identified as a major challenge in multi-class classification
3. **Transformer Limitations**: May be too complex for small datasets

## 📈 Next Steps

This attempt led to the development of binary classification approaches in subsequent attempts. 