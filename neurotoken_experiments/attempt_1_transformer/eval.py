#!/usr/bin/env python3
"""
NeuroToken Transformer Evaluation Script
Load trained models and perform detailed evaluation and visualization.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import os
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Import our custom modules
from dataset import load_and_prepare_data, create_data_splits, NeuroTokenDataset
from transformer_model import NeuroTokenTransformerConfig, create_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_trained_model(checkpoint_path, device):
    """Load a trained model from checkpoint"""
    logger.info(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = NeuroTokenTransformerConfig.from_dict(checkpoint['config'])
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
    logger.info(f"Model metrics: {checkpoint['metrics']}")
    
    return model, config, checkpoint

def evaluate_model(model, dataloader, device, label_map):
    """Evaluate model on a dataset"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_embeddings = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            # Get CLS token embeddings
            batch_size, seq_len = input_ids.shape
            x = model.embed(input_ids)
            x = x + model.pos_embed[:, :seq_len, :]
            cls_token = model.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
            attn_mask = torch.cat([cls_mask, attention_mask], dim=1)
            x = model.dropout(x)
            x = model.transformer(x, src_key_padding_mask=~attn_mask)
            cls_output = x[:, 0, :]
            
            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_embeddings.append(cls_output.cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    return all_predictions, all_labels, all_probabilities, all_embeddings

def compute_detailed_metrics(y_true, y_pred, y_proba, class_names):
    """Compute detailed classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'support_per_class': support_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    return metrics

def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
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

def plot_tsne_embeddings(embeddings, labels, class_names, save_path, title="t-SNE Visualization"):
    """Plot t-SNE visualization of embeddings"""
    logger.info("Computing t-SNE...")
    
    # Convert to numpy
    embeddings_np = embeddings.numpy()
    labels_np = np.array(labels)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_np)-1))
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'orange', 'red']
    
    for i, class_name in enumerate(class_names):
        mask = labels_np == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=colors[i], label=class_name, alpha=0.7, s=50)
    
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved t-SNE visualization to {save_path}")

def plot_attention_weights(model, dataloader, device, save_path, num_samples=5):
    """Plot attention weights for sample sequences"""
    logger.info("Computing attention weights...")
    
    model.eval()
    attention_weights_list = []
    sample_input_ids = []
    sample_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Get attention weights
            attention_weights = model.get_attention_weights(input_ids, attention_mask)
            
            attention_weights_list.append(attention_weights)
            sample_input_ids.append(input_ids.cpu())
            sample_labels.append(labels.cpu())
    
    # Plot attention weights for each layer
    num_layers = len(attention_weights_list[0])
    fig, axes = plt.subplots(num_samples, num_layers, figsize=(4*num_layers, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx in range(num_samples):
        for layer_idx in range(num_layers):
            # Get attention weights for this sample and layer
            attn_weights = attention_weights_list[sample_idx][layer_idx][0]  # First head
            
            # Plot heatmap
            im = axes[sample_idx, layer_idx].imshow(attn_weights.cpu().numpy(), cmap='Blues')
            axes[sample_idx, layer_idx].set_title(f'Sample {sample_idx+1}, Layer {layer_idx+1}')
            axes[sample_idx, layer_idx].set_xlabel('Key Position')
            axes[sample_idx, layer_idx].set_ylabel('Query Position')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[sample_idx, layer_idx])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved attention weights to {save_path}")

def analyze_predictions(predictions, labels, probabilities, subject_ids, class_names, save_path):
    """Analyze prediction patterns"""
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'subject_id': subject_ids,
        'true_label': [class_names[label] for label in labels],
        'predicted_label': [class_names[pred] for pred in predictions],
        'confidence': np.max(probabilities, axis=1),
        'correct': np.array(predictions) == np.array(labels)
    })
    
    # Add probability columns
    for i, class_name in enumerate(class_names):
        df[f'prob_{class_name}'] = probabilities[:, i]
    
    # Save detailed analysis
    df.to_csv(save_path, index=False)
    logger.info(f"Saved prediction analysis to {save_path}")
    
    # Print summary statistics
    logger.info("Prediction Analysis Summary:")
    logger.info(f"Overall accuracy: {df['correct'].mean():.4f}")
    logger.info(f"Average confidence: {df['confidence'].mean():.4f}")
    logger.info(f"Confidence for correct predictions: {df[df['correct']]['confidence'].mean():.4f}")
    logger.info(f"Confidence for incorrect predictions: {df[~df['correct']]['confidence'].mean():.4f}")
    
    # Per-class analysis
    for class_name in class_names:
        class_df = df[df['true_label'] == class_name]
        if len(class_df) > 0:
            class_acc = class_df['correct'].mean()
            class_conf = class_df['confidence'].mean()
            logger.info(f"{class_name}: Accuracy={class_acc:.4f}, Avg Confidence={class_conf:.4f}")

def main():
    """Main evaluation function"""
    # Configuration
    checkpoint_path = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/models/checkpoints/best_model.pt"
    output_dir = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/models"
    
    # Data paths
    token_sequences_file = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/token_sequences.jsonl"
    subject_labels_file = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/subject_labels.csv"
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model, config, checkpoint = load_trained_model(checkpoint_path, device)
    
    # Load data
    logger.info("Loading and preparing data...")
    input_ids, labels, label_map, subject_ids = load_and_prepare_data(
        token_sequences_file, subject_labels_file, config.max_len
    )
    
    # Create data splits (same as training)
    train_data, val_data, test_data = create_data_splits(
        input_ids, labels, subject_ids, test_size=0.2, val_size=0.2
    )
    
    # Create test dataset
    test_dataset = NeuroTokenDataset(test_data['input_ids'], test_data['labels'], config.max_len)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Class names
    class_names = ['CN', 'MCI', 'AD']
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    predictions, true_labels, probabilities, embeddings = evaluate_model(
        model, test_loader, device, label_map
    )
    
    # Compute metrics
    metrics = compute_detailed_metrics(true_labels, predictions, probabilities, class_names)
    
    # Print results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info("=" * 60)
    
    # Per-class metrics
    logger.info("Per-class metrics:")
    for i, class_name in enumerate(class_names):
        logger.info(f"{class_name}:")
        logger.info(f"  Precision: {metrics['precision_per_class'][i]:.4f}")
        logger.info(f"  Recall: {metrics['recall_per_class'][i]:.4f}")
        logger.info(f"  F1: {metrics['f1_per_class'][i]:.4f}")
        logger.info(f"  Support: {metrics['support_per_class'][i]}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved evaluation metrics to {metrics_path}")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    
    # Confusion matrix
    cm_path = os.path.join(output_dir, "evaluation_confusion_matrix.png")
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, cm_path, "Test Set Confusion Matrix")
    
    # t-SNE visualization
    tsne_path = os.path.join(output_dir, "evaluation_tsne.png")
    plot_tsne_embeddings(embeddings, true_labels, class_names, tsne_path, "Test Set t-SNE")
    
    # Attention weights
    attention_path = os.path.join(output_dir, "attention_weights.png")
    plot_attention_weights(model, test_loader, device, attention_path)
    
    # Prediction analysis
    analysis_path = os.path.join(output_dir, "prediction_analysis.csv")
    analyze_predictions(predictions, true_labels, probabilities, 
                       test_data['subject_ids'], class_names, analysis_path)
    
    # Save embeddings
    embeddings_path = os.path.join(output_dir, "evaluation_embeddings.pt")
    torch.save({
        'embeddings': embeddings,
        'labels': true_labels,
        'predictions': predictions,
        'probabilities': probabilities,
        'subject_ids': test_data['subject_ids'],
        'class_names': class_names
    }, embeddings_path)
    logger.info(f"Saved embeddings to {embeddings_path}")
    
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETED!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main() 