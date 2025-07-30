#!/usr/bin/env python3
"""
Temporal Model Evaluation Script
Loads trained temporal models and performs comprehensive evaluation and analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import our custom modules
from temporal_dataset import load_temporal_sequences, create_temporal_data_splits, TemporalNeuroTokenDataset
from hierarchical_gru_model import HierarchicalGRUConfig, create_hierarchical_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_trained_model(model_path, config):
    """
    Load a trained temporal model
    
    Args:
        model_path: Path to the saved model checkpoint
        config: Model configuration
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading trained model from {model_path}")
    
    # Create model
    model = create_hierarchical_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
    logger.info(f"Model metrics: {checkpoint['metrics']}")
    
    return model, checkpoint

def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataset
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_attention_weights = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            delays = batch['delays'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            session_mask = batch['session_mask'].to(device)
            labels = batch['label'].to(device).float()
            
            # Forward pass
            logits = model(input_ids, delays, attention_mask, session_mask).squeeze(-1)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > 0.5).long()
            
            # Get attention weights
            attention_weights = model.get_attention_weights(input_ids, delays, attention_mask, session_mask)
            
            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy().astype(int))
            all_probabilities.extend(probabilities.cpu().numpy())
            all_attention_weights.extend(attention_weights.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='binary')
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_predictions, average=None
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Classification report
    report = classification_report(all_labels, all_predictions, target_names=['CN', 'Impaired'], output_dict=True)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'support_per_class': support_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'attention_weights': all_attention_weights
    }
    
    return results

def plot_confusion_matrix(cm, class_names, save_path, title="Temporal Model Confusion Matrix"):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {save_path}")

def plot_attention_analysis(attention_weights, save_path):
    """Plot attention weight analysis"""
    attention_weights = np.array(attention_weights)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Average attention per session
    avg_attention = np.mean(attention_weights, axis=0)
    axes[0, 0].bar(range(1, len(avg_attention) + 1), avg_attention)
    axes[0, 0].set_title('Average Attention Weights per Session')
    axes[0, 0].set_xlabel('Session')
    axes[0, 0].set_ylabel('Attention Weight')
    axes[0, 0].grid(True)
    
    # Attention distribution
    axes[0, 1].hist(attention_weights.flatten(), bins=50, alpha=0.7)
    axes[0, 1].set_title('Distribution of Attention Weights')
    axes[0, 1].set_xlabel('Attention Weight')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True)
    
    # Attention heatmap (first 20 samples)
    if len(attention_weights) > 20:
        sample_attention = attention_weights[:20]
    else:
        sample_attention = attention_weights
    
    sns.heatmap(sample_attention, ax=axes[1, 0], cmap='YlOrRd')
    axes[1, 0].set_title('Attention Weights Heatmap (First 20 Samples)')
    axes[1, 0].set_xlabel('Session')
    axes[1, 0].set_ylabel('Sample')
    
    # Attention vs prediction confidence
    # This would need to be implemented with probabilities
    axes[1, 1].scatter(attention_weights.flatten(), np.random.rand(len(attention_weights.flatten())), alpha=0.5)
    axes[1, 1].set_title('Attention vs Random (Placeholder)')
    axes[1, 1].set_xlabel('Attention Weight')
    axes[1, 1].set_ylabel('Random Value')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved attention analysis to {save_path}")

def plot_probability_distribution(probabilities, labels, save_path):
    """Plot probability distribution by class"""
    plt.figure(figsize=(10, 6))
    
    cn_probs = [prob for prob, label in zip(probabilities, labels) if label == 0]
    impaired_probs = [prob for prob, label in zip(probabilities, labels) if label == 1]
    
    plt.hist(cn_probs, bins=30, alpha=0.7, label='CN', color='blue')
    plt.hist(impaired_probs, bins=30, alpha=0.7, label='Impaired', color='red')
    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
    
    plt.title('Prediction Probability Distribution by Class')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved probability distribution to {save_path}")

def create_evaluation_report(results, model_info, save_path):
    """Create a comprehensive evaluation report"""
    report = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'model_info': model_info,
        'metrics': {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1'],
            'precision_per_class': results['precision_per_class'],
            'recall_per_class': results['recall_per_class'],
            'f1_per_class': results['f1_per_class'],
            'support_per_class': results['support_per_class']
        },
        'confusion_matrix': results['confusion_matrix'],
        'classification_report': results['classification_report']
    }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Saved evaluation report to {save_path}")
    return report

def main():
    """Main evaluation function"""
    # Configuration
    config = HierarchicalGRUConfig(
        vocab_size=32,
        token_emb_dim=32,
        session_hidden_dim=64,
        subject_hidden_dim=128,
        time_emb_dim=16,
        num_layers=2,
        dropout=0.3,
        max_sessions=5,
        max_tokens=28
    )
    
    # Paths
    model_dir = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/temporal_attempt/models"
    temporal_sequences_file = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/temporal_attempt/temporal_sequences.jsonl"
    output_dir = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/temporal_attempt/evaluation"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load temporal sequences
    logger.info("Loading temporal sequences...")
    temporal_sequences = load_temporal_sequences(temporal_sequences_file)
    
    # Create data splits
    train_sequences, val_sequences, test_sequences = create_temporal_data_splits(
        temporal_sequences, test_size=0.2, val_size=0.2
    )
    
    # Create datasets
    test_dataset = TemporalNeuroTokenDataset(test_sequences, config.max_sessions, config.max_tokens)
    val_dataset = TemporalNeuroTokenDataset(val_sequences, config.max_sessions, config.max_tokens)
    
    # Create dataloaders
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Evaluate best model
    best_model_path = os.path.join(model_dir, "checkpoints", "best_model.pt")
    if os.path.exists(best_model_path):
        logger.info("Evaluating best model...")
        best_model, best_checkpoint = load_trained_model(best_model_path, config)
        best_model = best_model.to(device)
        
        # Evaluate on test set
        test_results = evaluate_model(best_model, test_loader, device)
        
        # Evaluate on validation set
        val_results = evaluate_model(best_model, val_loader, device)
        
        # Create plots
        class_names = ['CN', 'Impaired']
        
        # Test set plots
        test_cm_path = os.path.join(output_dir, "test_confusion_matrix.png")
        plot_confusion_matrix(test_results['confusion_matrix'], class_names, test_cm_path, 
                            "Temporal Model - Test Set Confusion Matrix")
        
        test_attention_path = os.path.join(output_dir, "test_attention_analysis.png")
        plot_attention_analysis(test_results['attention_weights'], test_attention_path)
        
        test_prob_path = os.path.join(output_dir, "test_probability_distribution.png")
        plot_probability_distribution(test_results['probabilities'], test_results['labels'], test_prob_path)
        
        # Validation set plots
        val_cm_path = os.path.join(output_dir, "val_confusion_matrix.png")
        plot_confusion_matrix(val_results['confusion_matrix'], class_names, val_cm_path,
                            "Temporal Model - Validation Set Confusion Matrix")
        
        # Create evaluation report
        model_info = {
            'model_type': 'HierarchicalGRU',
            'config': config.to_dict(),
            'checkpoint_epoch': best_checkpoint['epoch'],
            'checkpoint_metrics': best_checkpoint['metrics']
        }
        
        test_report_path = os.path.join(output_dir, "test_evaluation_report.json")
        test_report = create_evaluation_report(test_results, model_info, test_report_path)
        
        val_report_path = os.path.join(output_dir, "val_evaluation_report.json")
        val_report = create_evaluation_report(val_results, model_info, val_report_path)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("TEMPORAL MODEL EVALUATION COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"Test Set Results:")
        logger.info(f"  Accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"  Precision: {test_results['precision']:.4f}")
        logger.info(f"  Recall: {test_results['recall']:.4f}")
        logger.info(f"  F1 Score: {test_results['f1']:.4f}")
        logger.info(f"")
        logger.info(f"Validation Set Results:")
        logger.info(f"  Accuracy: {val_results['accuracy']:.4f}")
        logger.info(f"  Precision: {val_results['precision']:.4f}")
        logger.info(f"  Recall: {val_results['recall']:.4f}")
        logger.info(f"  F1 Score: {val_results['f1']:.4f}")
        logger.info(f"")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 60)
        
    else:
        logger.error(f"Best model not found at {best_model_path}")
        logger.info("Please run training first or check the model path")
    
    # Evaluate final model if different from best
    final_model_path = os.path.join(model_dir, "checkpoints", "final_model.pt")
    if os.path.exists(final_model_path) and final_model_path != best_model_path:
        logger.info("Evaluating final model...")
        final_model, final_checkpoint = load_trained_model(final_model_path, config)
        final_model = final_model.to(device)
        
        final_test_results = evaluate_model(final_model, test_loader, device)
        
        final_cm_path = os.path.join(output_dir, "final_test_confusion_matrix.png")
        plot_confusion_matrix(final_test_results['confusion_matrix'], class_names, final_cm_path,
                            "Temporal Model - Final Model Test Set Confusion Matrix")
        
        logger.info(f"Final Model Test Results:")
        logger.info(f"  Accuracy: {final_test_results['accuracy']:.4f}")
        logger.info(f"  F1 Score: {final_test_results['f1']:.4f}")

if __name__ == "__main__":
    main() 