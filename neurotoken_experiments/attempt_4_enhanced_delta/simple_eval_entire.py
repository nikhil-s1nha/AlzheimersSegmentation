#!/usr/bin/env python3
"""
Simple evaluation script for entire dataset
No external dependencies beyond torch and numpy
"""

import os
import json
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import our modules
from enhanced_dataset_discrete import EnhancedNeuroTokenDatasetDiscrete
from enhanced_model import create_enhanced_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_simple_data(token_file, labels_file):
    """Load data without pandas dependency"""
    logger.info("Loading token data...")
    
    # Load tokens
    with open(token_file, 'r') as f:
        tokens_data = [json.loads(line) for line in f]
    
    logger.info(f"Loaded {len(tokens_data)} token records")
    
    # Load labels (simple CSV parsing)
    subject_labels = {}
    with open(labels_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            parts = line.strip().split(',')
            if len(parts) >= 2:
                subject_labels[parts[0]] = int(parts[1])
    
    logger.info(f"Loaded {len(subject_labels)} subject labels")
    
    # Group tokens by subject
    subject_tokens = {}
    for token_record in tokens_data:
        subject_id = token_record['subject_id']
        if subject_id not in subject_tokens:
            subject_tokens[subject_id] = []
        subject_tokens[subject_id].append(token_record)
    
    logger.info(f"Found {len(subject_tokens)} subjects with token data")
    
    # Create sequences
    sequences = []
    for subject_id, sessions in subject_tokens.items():
        if subject_id in subject_labels:
            sequences.append({
                'subject_id': subject_id,
                'label': subject_labels[subject_id],
                'sessions': sessions
            })
    
    logger.info(f"Created {len(sequences)} complete sequences")
    return sequences

def evaluate_model(model, data_loader, device):
    """Evaluate model and return comprehensive metrics"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
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
            
            if batch_idx % 10 == 0:
                logger.info(f"Processed batch {batch_idx}/{len(data_loader)}")
    
    accuracy = correct / total if total > 0 else 0.0
    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])
    all_probabilities = np.concatenate(all_probabilities) if all_probabilities else np.array([])
    
    # Calculate binary classification metrics
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
    logger.info(f"Using device: {device}")

    # File paths
    token_file = 'enhanced_tokens.json'
    labels_file = 'subject_labels.csv'
    transformers_path = 'models/transformers_discrete.pkl'
    best_model_path = 'models/best_model_discrete.pt'
    
    # Check if files exist
    for file_path in [token_file, labels_file, transformers_path, best_model_path]:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return

    # Load data
    sequences = load_simple_data(token_file, labels_file)
    
    if len(sequences) == 0:
        logger.error("No sequences found!")
        return

    # Create dataset
    logger.info("Creating evaluation dataset...")
    dataset = EnhancedNeuroTokenDatasetDiscrete(
        sequences,
        max_sessions=5,
        max_tokens=28,
        fit_transformers=False,
        transformers_path=transformers_path,
    )
    
    # Create data loader
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Get sample batch to determine dimensions
    sample_batch = next(iter(data_loader))
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

    # Create model
    logger.info("Creating model...")
    model = create_enhanced_model(
        model_type='gru',
        max_sessions=5,
        max_tokens=28,
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

    # Load trained weights
    logger.info("Loading trained model...")
    state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state)
    logger.info(f"Loaded model from {best_model_path}")

    # Evaluate model
    logger.info("Evaluating model on entire dataset...")
    acc, prec, rec, f1, auc, cm, preds, labels, probs = evaluate_model(model, data_loader, device)
    
    # Print results
    print("\n" + "="*60)
    print("ENTIRE DATASET EVALUATION RESULTS")
    print("="*60)
    print(f"Total samples: {len(labels)}")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall: {rec:.4f} ({rec*100:.2f}%)")
    print(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")
    print(f"AUC-ROC: {auc:.4f} ({auc*100:.2f}%)")
    print("")
    print("Confusion Matrix:")
    print("                Predicted")
    print("              Normal  Impaired")
    print(f"True Normal    {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"True Impaired  {cm[1,0]:4d}    {cm[1,1]:4d}")
    print("")
    
    # Class distribution
    normal_count = np.sum(labels == 0)
    impaired_count = np.sum(labels == 1)
    print(f"Class distribution:")
    print(f"  Normal: {normal_count} ({normal_count/len(labels)*100:.1f}%)")
    print(f"  Impaired: {impaired_count} ({impaired_count/len(labels)*100:.1f}%)")
    print("="*60)

    # Save results
    logger.info("Saving results...")
    
    # Save comprehensive results as JSON
    results = {
        "total_samples": len(labels),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "auc_roc": float(auc),
        "confusion_matrix": cm.tolist(),
        "class_distribution": {
            "normal": int(normal_count),
            "impaired": int(impaired_count)
        }
    }
    
    with open('entire_dataset_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save detailed CSV
    with open('entire_dataset_detailed.csv', 'w') as f:
        f.write("Index,True_Label,Predicted_Label,Probability_Normal,Probability_Impaired\n")
        for i in range(len(labels)):
            f.write(f"{i},{labels[i]},{preds[i]},{probs[i][0]:.6f},{probs[i][1]:.6f}\n")
    
    # Save summary report
    with open('entire_dataset_summary.txt', 'w') as f:
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
    
    logger.info("Results saved to:")
    logger.info("- entire_dataset_results.json")
    logger.info("- entire_dataset_detailed.csv")
    logger.info("- entire_dataset_summary.txt")

if __name__ == "__main__":
    main()
