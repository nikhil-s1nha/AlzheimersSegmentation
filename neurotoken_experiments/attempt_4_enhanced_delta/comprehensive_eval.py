#!/usr/bin/env python3
"""
Comprehensive Evaluation Script
Generates detailed confusion matrix, ROC curve, and performance metrics
"""

import os
import json
import logging
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, roc_auc_score, average_precision_score
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from enhanced_dataset_discrete import EnhancedNeuroTokenDatasetDiscrete
from enhanced_model import create_enhanced_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_simple_labels(csv_path):
    """Load subject labels from CSV with label column"""
    labels_df = pd.read_csv(csv_path)
    subject_labels = {}
    for _, row in labels_df.iterrows():
        subject_id = row['subject_id']
        label = int(row['label'])
        subject_labels[subject_id] = label
    
    logger.info(f"Loaded labels for {len(subject_labels)} subjects")
    label_counts = pd.Series(list(subject_labels.values())).value_counts().sort_index()
    logger.info(f"Label distribution: {label_counts.to_dict()}")
    return subject_labels

def load_tokens_simple(json_path):
    """Load tokens from JSON file"""
    tokens = []
    with open(json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                tokens.append(json.loads(line))
    
    # Group by subject
    subject_sequences = {}
    for token in tokens:
        subject_id = token['subject_id']
        if subject_id not in subject_sequences:
            subject_sequences[subject_id] = []
        subject_sequences[subject_id].append(token)
    
    # Convert to sequence format
    sequences = []
    for subject_id, tokens_list in subject_sequences.items():
        sequences.append({
            'subject_id': subject_id,
            'sessions': [{'tokens': t, 'session': t['session']} for t in tokens_list]
        })
    
    logger.info(f"Loaded {len(sequences)} subject sequences")
    return sequences

def create_sequences_with_labels(sequences, subject_labels):
    """Add labels to sequences"""
    enhanced_sequences = []
    for seq in sequences:
        subject_id = seq['subject_id']
        if subject_id in subject_labels:
            enhanced_seq = seq.copy()
            enhanced_seq['label'] = subject_labels[subject_id]
            enhanced_sequences.append(enhanced_seq)
        else:
            logger.warning(f"Missing label for subject {subject_id}")
    
    logger.info(f"Created {len(enhanced_sequences)} labeled sequences")
    return enhanced_sequences

def evaluate_model_comprehensive(model, data_loader, device):
    """Comprehensive evaluation with probabilities and predictions"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits, _ = model(batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            labels = batch['label'].squeeze()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (impaired)
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """Plot detailed confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal (CN)', 'Impaired (MCI+AD)'],
                yticklabels=['Normal (CN)', 'Impaired (MCI+AD)'])
    
    plt.title('Confusion Matrix - Alzheimer\'s Detection Model', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    
    # Add metrics text
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {save_path}")
    return cm

def plot_roc_curve(y_true, y_probs, save_path="roc_curve.png"):
    """Plot ROC curve with AUC"""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add performance indicators
    plt.text(0.6, 0.2, f'AUC = {auc_score:.3f}', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ROC curve saved to {save_path}")
    return auc_score

def plot_precision_recall_curve(y_true, y_probs, save_path="precision_recall_curve.png"):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='green', lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add performance indicators
    plt.text(0.6, 0.2, f'Average Precision = {avg_precision:.3f}', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Precision-Recall curve saved to {save_path}")
    return avg_precision

def plot_performance_summary(metrics, save_path="performance_summary.png"):
    """Plot performance metrics summary"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    accuracies = [metrics['train_acc'], metrics['val_acc'], metrics['test_acc']]
    labels = ['Train', 'Validation', 'Test']
    colors = ['#2E8B57', '#4682B4', '#CD5C5C']
    
    bars1 = ax1.bar(labels, accuracies, color=colors, alpha=0.8)
    ax1.set_title('Model Accuracy Across Splits', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Classification metrics
    metrics_names = ['Precision', 'Recall', 'F1-Score']
    metrics_values = [metrics['precision'], metrics['recall'], metrics['f1']]
    colors2 = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars2 = ax2.bar(metrics_names, metrics_values, color=colors2, alpha=0.8)
    ax2.set_title('Classification Metrics', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, val in zip(bars2, metrics_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Confusion matrix visualization
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Normal', 'Impaired'],
                yticklabels=['Normal', 'Impaired'])
    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    
    # ROC AUC and PR AUC
    auc_scores = [metrics['roc_auc'], metrics['pr_auc']]
    auc_labels = ['ROC AUC', 'PR AUC']
    colors3 = ['#FF8C00', '#9370DB']
    
    bars3 = ax4.bar(auc_labels, auc_scores, color=colors3, alpha=0.8)
    ax4.set_title('AUC Scores', fontsize=14, fontweight='bold')
    ax4.set_ylabel('AUC', fontsize=12)
    ax4.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, auc_val in zip(bars3, auc_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auc_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Performance summary saved to {save_path}")

def main():
    """Main evaluation function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Configuration
    config = {
                       "token_file": "enhanced_tokens_new.json",
        "labels_file": "subject_labels.csv",
        "output_dir": "models",
        "model_type": "gru",
        "max_sessions": 5,
        "max_tokens": 28,
        "hidden_dim": 128,
        "num_layers": 2,
        "num_heads": 8,
        "dropout": 0.3,
        "num_classes": 2,
        "batch_size": 16,
        "test_size": 0.2,
        "val_size": 0.2,
        "random_state": 42,
        "num_workers": 0,
    }
    
    # Load data
    logger.info("Loading data...")
    sequences = load_tokens_simple(config['token_file'])
    subject_labels = load_simple_labels(config['labels_file'])
    sequences = create_sequences_with_labels(sequences, subject_labels)
    
    # Split data
    train_seqs, test_seqs = train_test_split(
        sequences, test_size=config['test_size'], random_state=config['random_state'],
        stratify=[seq['label'] for seq in sequences]
    )
    train_seqs, val_seqs = train_test_split(
        train_seqs, test_size=config['val_size'], random_state=config['random_state'],
        stratify=[seq['label'] for seq in train_seqs]
    )
    
    logger.info(f"Data split: Train={len(train_seqs)}, Val={len(val_seqs)}, Test={len(test_seqs)}")
    
    # Create datasets
    transformers_path = os.path.join(config['output_dir'], 'transformers_discrete.pkl')
    test_dataset = EnhancedNeuroTokenDatasetDiscrete(
        test_seqs, config['max_sessions'], config['max_tokens'], False, transformers_path
    )
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Load trained model
    best_model_path = os.path.join(config['output_dir'], 'best_model_discrete.pt')
    if not os.path.exists(best_model_path):
        logger.error(f"Model not found at {best_model_path}. Please train the model first.")
        return
    
    # Infer dimensions from sample batch
    sample_batch = next(iter(test_loader))
    level_token_dim = int(sample_batch['level_tokens'].max().item()) + 1
    delta_token_dim = int(sample_batch['delta_tokens'].max().item()) + 1
    harmonized_dim = int(sample_batch['harmonized_features'].size(-1))
    region_embedding_dim = int(sample_batch['region_embeddings'].size(-1))
    delta_t_bucket_dim = int(sample_batch['delta_t_buckets'].size(-1))
    
    # Create model
    model = create_enhanced_model(
        model_type=config['model_type'],
        max_sessions=config['max_sessions'],
        max_tokens=config['max_tokens'],
        level_token_dim=level_token_dim,
        delta_token_dim=delta_token_dim,
        harmonized_dim=harmonized_dim,
        region_embedding_dim=region_embedding_dim,
        delta_t_bucket_dim=delta_t_bucket_dim,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        num_classes=config['num_classes'],
    ).to(device)
    
    # Load trained weights
    checkpoint = torch.load(best_model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        # Original training script saves just the state_dict
        model.load_state_dict(checkpoint)
        logger.info("Loaded model state dict")
    
    # Comprehensive evaluation
    logger.info("Running comprehensive evaluation...")
    y_pred, y_true, y_probs = evaluate_model_comprehensive(model, test_loader, device)
    
    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate AUC scores
    roc_auc = roc_auc_score(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)
    
    # Print detailed results
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"PR AUC: {pr_auc:.4f}")
    logger.info("")
    logger.info("Confusion Matrix:")
    logger.info(f"  True Negatives (TN): {tn}")
    logger.info(f"  False Positives (FP): {fp}")
    logger.info(f"  False Negatives (FN): {fn}")
    logger.info(f"  True Positives (TP): {tp}")
    logger.info("")
    logger.info("Classification Report:")
    logger.info(classification_report(y_true, y_pred, target_names=['Normal', 'Impaired']))
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, "confusion_matrix_detailed.png")
    
    # Plot ROC curve
    plot_roc_curve(y_true, y_probs, "roc_curve_detailed.png")
    
    # Plot Precision-Recall curve
    plot_precision_recall_curve(y_true, y_probs, "precision_recall_curve.png")
    
    # Create performance summary
    metrics = {
        'train_acc': checkpoint.get('val_acc', 0),  # Use validation acc as proxy
        'val_acc': checkpoint.get('val_acc', 0),
        'test_acc': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm
    }
    
    plot_performance_summary(metrics, "performance_summary.png")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC', 'PR AUC'],
        'Value': [accuracy, precision, recall, f1, roc_auc, pr_auc],
        'Percentage': [accuracy*100, precision*100, recall*100, f1*100, roc_auc*100, pr_auc*100]
    })
    
    results_df.to_csv('comprehensive_evaluation_results.csv', index=False)
    logger.info("Results saved to comprehensive_evaluation_results.csv")
    
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE!")
    logger.info("Generated files:")
    logger.info("  - confusion_matrix_detailed.png")
    logger.info("  - roc_curve_detailed.png")
    logger.info("  - precision_recall_curve.png")
    logger.info("  - performance_summary.png")
    logger.info("  - comprehensive_evaluation_results.csv")
    logger.info("=" * 60)

if __name__ == "__main__":
    main() 