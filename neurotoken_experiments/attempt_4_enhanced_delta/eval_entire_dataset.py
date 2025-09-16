import os
import json
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from enhanced_dataset import (
    load_enhanced_tokens,
    load_subject_labels,
    create_enhanced_sequences,
    split_enhanced_data,
)
from enhanced_dataset_discrete import EnhancedNeuroTokenDatasetDiscrete
from enhanced_model import create_enhanced_model


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits, _ = model(batch)
            probabilities = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            labels = batch['label'].squeeze()
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
    
    accuracy = correct / total if total > 0 else 0.0
    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])
    all_probabilities = np.concatenate(all_probabilities) if all_probabilities else np.array([])
    
    # Precision/Recall/F1 (binary)
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    tn = np.sum((all_preds == 0) & (all_labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Calculate AUC manually
    if len(np.unique(all_labels)) == 2:
        # Sort by probability of positive class
        sorted_indices = np.argsort(all_probabilities[:, 1])
        sorted_labels = all_labels[sorted_indices]
        sorted_probs = all_probabilities[sorted_indices, 1]
        
        n_pos = np.sum(all_labels == 1)
        n_neg = np.sum(all_labels == 0)
        
        if n_pos > 0 and n_neg > 0:
            tpr = []
            fpr = []
            
            for i in range(len(sorted_probs)):
                threshold = sorted_probs[i]
                tp_count = np.sum((sorted_probs >= threshold) & (sorted_labels == 1))
                fp_count = np.sum((sorted_probs >= threshold) & (sorted_labels == 0))
                
                tpr.append(tp_count / n_pos)
                fpr.append(fp_count / n_neg)
            
            # Calculate AUC using trapezoidal rule
            auc = 0
            for i in range(1, len(fpr)):
                auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
        else:
            auc = 0.5
    else:
        auc = 0.5
    
    # Confusion matrix
    confusion_matrix = np.array([[tn, fp], [fn, tp]])
    
    return accuracy, precision, recall, f1, auc, confusion_matrix, all_preds, all_labels, all_probabilities


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Must match the training config
    config = {
        "token_file": "enhanced_tokens.json",
        "labels_file": "subject_labels.csv",
        "output_dir": "models",
        "max_sessions": 5,
        "max_tokens": 28,
        "batch_size": 16,
        "test_size": 0.2,
        "val_size": 0.2,
        "random_state": 42,
        "num_workers": 0,  # Set to 0 to avoid multiprocessing issues
    }

    logger.info("Setting up data for ENTIRE DATASET evaluation...")

    sequences = load_enhanced_tokens(config['token_file'])
    subject_labels = load_subject_labels(config['labels_file'])
    sequences = create_enhanced_sequences(sequences, subject_labels)
    
    logger.info(f"Total sequences available: {len(sequences)}")
    
    # Use ALL sequences for evaluation (no train/test split)
    all_sequences = sequences

    transformers_path = os.path.join(config['output_dir'], 'transformers_discrete.pkl')
    best_model_path = os.path.join(config['output_dir'], 'best_model_discrete.pt')

    # Build dataset using ALL data
    all_dataset = EnhancedNeuroTokenDatasetDiscrete(
        all_sequences,
        max_sessions=config['max_sessions'],
        max_tokens=config['max_tokens'],
        fit_transformers=False,
        transformers_path=transformers_path,
    )
    all_loader = DataLoader(all_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # Infer feature dims from one batch
    sample_batch = next(iter(all_loader))
    level_token_dim = int(sample_batch['level_tokens'].max().item()) + 1
    delta_token_dim = int(sample_batch['delta_tokens'].max().item()) + 1
    harmonized_dim = int(sample_batch['harmonized_features'].size(-1))
    region_embedding_dim = int(sample_batch['region_embeddings'].size(-1))
    delta_t_bucket_dim = int(sample_batch['delta_t_buckets'].size(-1))

    logger.info(f"Feature dimensions:")
    logger.info(f"  - Level tokens: {level_token_dim}")
    logger.info(f"  - Delta tokens: {delta_token_dim}")
    logger.info(f"  - Harmonized features: {harmonized_dim}")
    logger.info(f"  - Region embeddings: {region_embedding_dim}")
    logger.info(f"  - Delta-t buckets: {delta_t_bucket_dim}")

    logger.info("Initializing model for evaluation...")
    model = create_enhanced_model(
        model_type='gru',
        max_sessions=config['max_sessions'],
        max_tokens=config['max_tokens'],
        level_token_dim=level_token_dim,
        delta_token_dim=delta_token_dim,
        harmonized_dim=harmonized_dim,
        region_embedding_dim=region_embedding_dim,
        delta_t_bucket_dim=delta_t_bucket_dim,
        hidden_dim=128,
        num_layers=2,
        num_heads=8,
        dropout=0.3,
        num_classes=2,
    ).to(device)

    # Load the model checkpoint
    state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state)
    logger.info(f"Loaded model from {best_model_path}")

    logger.info("Evaluating model on ENTIRE DATASET...")
    acc, prec, rec, f1, auc, cm, preds, labels, probs = evaluate_model(model, all_loader, device)
    
    logger.info("="*60)
    logger.info("ENTIRE DATASET EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"Total samples: {len(labels)}")
    logger.info(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    logger.info(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
    logger.info(f"Recall: {rec:.4f} ({rec*100:.2f}%)")
    logger.info(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")
    logger.info(f"AUC-ROC: {auc:.4f} ({auc*100:.2f}%)")
    logger.info("")
    logger.info("Confusion Matrix:")
    logger.info("                Predicted")
    logger.info("              Normal  Impaired")
    logger.info(f"True Normal    {cm[0,0]:4d}    {cm[0,1]:4d}")
    logger.info(f"True Impaired  {cm[1,0]:4d}    {cm[1,1]:4d}")
    logger.info("")
    
    # Class distribution
    normal_count = np.sum(labels == 0)
    impaired_count = np.sum(labels == 1)
    logger.info(f"Class distribution:")
    logger.info(f"  Normal: {normal_count} ({normal_count/len(labels)*100:.1f}%)")
    logger.info(f"  Impaired: {impaired_count} ({impaired_count/len(labels)*100:.1f}%)")
    logger.info("="*60)

    # Save comprehensive results
    results = {
        "total_samples": len(labels),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auc_roc": auc,
        "confusion_matrix": cm.tolist(),
        "class_distribution": {
            "normal": int(normal_count),
            "impaired": int(impaired_count)
        },
        "predictions": preds.tolist(),
        "true_labels": labels.tolist(),
        "probabilities": probs.tolist()
    }
    
    # Save metrics
    metrics_path = os.path.join(config['output_dir'], 'entire_dataset_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved comprehensive results to {metrics_path}")
    
    # Save detailed CSV
    csv_path = 'entire_dataset_results.csv'
    with open(csv_path, 'w') as f:
        f.write("Subject_Index,True_Label,Predicted_Label,Probability_Normal,Probability_Impaired\n")
        for i in range(len(labels)):
            f.write(f"{i},{labels[i]},{preds[i]},{probs[i][0]:.6f},{probs[i][1]:.6f}\n")
    logger.info(f"Saved detailed results to {csv_path}")
    
    # Save summary report
    report_path = 'entire_dataset_summary.txt'
    with open(report_path, 'w') as f:
        f.write("ENTIRE DATASET EVALUATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total samples evaluated: {len(labels)}\n")
        f.write(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"Precision: {prec:.4f} ({prec*100:.2f}%)\n")
        f.write(f"Recall: {rec:.4f} ({rec*100:.2f}%)\n")
        f.write(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)\n")
        f.write(f"AUC-ROC: {auc:.4f} ({auc*100:.2f}%)\n\n")
        f.write("Confusion Matrix:\n")
        f.write("                Predicted\n")
        f.write("              Normal  Impaired\n")
        f.write(f"True Normal    {cm[0,0]:4d}    {cm[0,1]:4d}\n")
        f.write(f"True Impaired  {cm[1,0]:4d}    {cm[1,1]:4d}\n\n")
        f.write(f"Class distribution:\n")
        f.write(f"  Normal: {normal_count} ({normal_count/len(labels)*100:.1f}%)\n")
        f.write(f"  Impaired: {impaired_count} ({impaired_count/len(labels)*100:.1f}%)\n\n")
        f.write("Model Performance Analysis:\n")
        if acc > 0.8:
            f.write("- Good overall accuracy\n")
        elif acc > 0.7:
            f.write("- Moderate accuracy\n")
        else:
            f.write("- Low accuracy - consider more training data\n")
        
        if auc > 0.8:
            f.write("- Good discrimination ability\n")
        elif auc > 0.7:
            f.write("- Moderate discrimination ability\n")
        else:
            f.write("- Poor discrimination ability\n")
        
        if abs(prec - rec) < 0.1:
            f.write("- Balanced precision-recall tradeoff\n")
        else:
            f.write("- Imbalanced precision-recall tradeoff\n")
    
    logger.info(f"Saved summary report to {report_path}")


if __name__ == "__main__":
    main()
