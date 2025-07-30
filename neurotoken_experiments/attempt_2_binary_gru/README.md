# Attempt 2: Binary GRU with Class Balancing

## 🧠 Overview

This attempt addresses the class imbalance issues from Attempt 1 by implementing binary classification (CN vs Impaired, where Impaired = MCI + AD) with aggressive class balancing through downsampling.

## 📊 Results

- **Accuracy**: 65%
- **F1-Score**: ~60%
- **Improvement**: 15% accuracy improvement over transformer approach
- **Key Success**: Better class balance and generalization

## 🔧 Architecture

- **Model**: GRU with attention mechanism
- **Classification**: Binary (CN vs Impaired)
- **Class Balancing**: Downsampling of majority class
- **Input**: Neurotoken sequences
- **Output**: Binary probabilities

## 📁 Files

- `train_binary.py` - Training script with class balancing
- `dataset_binary.py` - Binary dataset implementation
- `gru_model.py` - GRU model architecture
- `eval.py` - Evaluation script
- `extract_results.py` - Results analysis

## 🚀 Usage

```bash
python train_binary.py
python eval.py
```

## 🎯 Key Improvements

1. **Binary Classification**: Simplified problem from 3-class to 2-class
2. **Class Balancing**: Downsampling addresses imbalance issues
3. **GRU Architecture**: More suitable for smaller datasets than transformer
4. **Better Generalization**: Reduced overfitting

## 📈 Limitations

- **No Temporal Information**: Doesn't utilize longitudinal data
- **Loss of Detail**: MCI and AD are combined into "Impaired" class

## 🔬 Research Value

This attempt established the foundation for more sophisticated approaches and demonstrated the importance of proper class balancing in medical AI applications. 