#!/usr/bin/env python3
"""
NeuroToken Transformer Training Script
Complete training pipeline with validation, early stopping, and metrics tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import os
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time

# Import our custom modules
from dataset import load_and_prepare_data, create_data_splits, NeuroTokenDataset
from transformer_model import NeuroTokenTransformerConfig, create_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


class MetricsTracker:
    """Track training and validation metrics"""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        
    def update(self, train_loss, val_loss, train_acc, val_acc, lr):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
    
    def get_best_metrics(self):
        best_val_loss_idx = np.argmin(self.val_losses)
        return {
            'best_epoch': best_val_loss_idx + 1,
            'best_val_loss': self.val_losses[best_val_loss_idx],
            'best_val_acc': self.val_accuracies[best_val_loss_idx],
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'final_train_acc': self.train_accuracies[-1],
            'final_val_acc': self.val_accuracies[-1]
        }


def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist()
    }
    
    return metrics


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Collect metrics
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # Collect metrics
            total_loss += loss.item()
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy, all_predictions, all_labels, all_probabilities


def save_checkpoint(model, optimizer, epoch, metrics, config, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config.to_dict()
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Saved checkpoint to {filepath}")


def plot_training_curves(metrics_tracker, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(metrics_tracker.train_losses) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, metrics_tracker.train_losses, 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, metrics_tracker.val_losses, 'r-', label='Val Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy curves
    axes[0, 1].plot(epochs, metrics_tracker.train_accuracies, 'b-', label='Train Accuracy')
    axes[0, 1].plot(epochs, metrics_tracker.val_accuracies, 'r-', label='Val Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate curve
    axes[1, 0].plot(epochs, metrics_tracker.learning_rates, 'g-')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].grid(True)
    
    # Loss difference
    loss_diff = [abs(t - v) for t, v in zip(metrics_tracker.train_losses, metrics_tracker.val_losses)]
    axes[1, 1].plot(epochs, loss_diff, 'purple')
    axes[1, 1].set_title('Train-Val Loss Difference')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('|Train Loss - Val Loss|')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training curves to {save_path}")


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {save_path}")


def main():
    """Main training function"""
    # Configuration
    config = NeuroTokenTransformerConfig(
        vocab_size=32,
        emb_dim=64,
        max_len=224,
        num_classes=3,
        num_layers=2,
        num_heads=4,
        dim_feedforward=256,
        dropout=0.1
    )
    
    # Training hyperparameters
    batch_size = 32
    learning_rate = 1e-4
    weight_decay = 1e-2
    epochs = 30
    patience = 7
    
    # Data paths
    token_sequences_file = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/token_sequences.jsonl"
    subject_labels_file = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/subject_labels.csv"
    
    # Output directory
    output_dir = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/models"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    input_ids, labels, label_map, subject_ids = load_and_prepare_data(
        token_sequences_file, subject_labels_file, config.max_len
    )
    
    # Create data splits
    train_data, val_data, test_data = create_data_splits(
        input_ids, labels, subject_ids, test_size=0.2, val_size=0.2
    )
    
    # Create datasets
    train_dataset = NeuroTokenDataset(train_data['input_ids'], train_data['labels'], config.max_len)
    val_dataset = NeuroTokenDataset(val_data['input_ids'], val_data['labels'], config.max_len)
    test_dataset = NeuroTokenDataset(test_data['input_ids'], test_data['labels'], config.max_len)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels, val_probs = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Update metrics tracker
        metrics_tracker.update(train_loss, val_loss, train_acc, val_acc, current_lr)
        
        # Log progress
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{epochs} ({epoch_time:.2f}s)")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        logger.info(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(output_dir, "checkpoints", "best_model.pt")
            save_checkpoint(model, optimizer, epoch+1, 
                          {'val_loss': val_loss, 'val_acc': val_acc}, config, best_model_path)
        
        # Early stopping
        if early_stopping(val_loss, model):
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_loss, test_acc, test_preds, test_labels, test_probs = validate_epoch(
        model, test_loader, criterion, device
    )
    
    # Compute final metrics
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)
    
    # Save final model
    final_model_path = os.path.join(output_dir, "checkpoints", "final_model.pt")
    save_checkpoint(model, optimizer, epochs, 
                   {'test_loss': test_loss, 'test_acc': test_acc, **test_metrics}, config, final_model_path)
    
    # Save metrics
    best_metrics = metrics_tracker.get_best_metrics()
    final_results = {
        'best_metrics': best_metrics,
        'test_metrics': test_metrics,
        'training_curves': {
            'train_losses': [float(x) for x in metrics_tracker.train_losses],
            'val_losses': [float(x) for x in metrics_tracker.val_losses],
            'train_accuracies': [float(x) for x in metrics_tracker.train_accuracies],
            'val_accuracies': [float(x) for x in metrics_tracker.val_accuracies],
            'learning_rates': [float(x) for x in metrics_tracker.learning_rates]
        },
        'config': config.to_dict(),
        'label_map': label_map
    }
    
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Plot training curves
    curves_path = os.path.join(output_dir, "training_curves.png")
    plot_training_curves(metrics_tracker, curves_path)
    
    # Plot confusion matrix
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    class_names = ['CN', 'MCI', 'AD']
    plot_confusion_matrix(test_metrics['confusion_matrix'], class_names, cm_path)
    
    # Save CLS embeddings for visualization
    cls_embeddings = []
    cls_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
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
            
            cls_embeddings.append(cls_output.cpu())
            cls_labels.append(labels.cpu())
    
    cls_embeddings = torch.cat(cls_embeddings, dim=0)
    cls_labels = torch.cat(cls_labels, dim=0)
    
    embeddings_path = os.path.join(output_dir, "cls_embeddings.pt")
    torch.save({
        'embeddings': cls_embeddings,
        'labels': cls_labels,
        'class_names': class_names
    }, embeddings_path)
    
    # Final summary
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 60)
    logger.info(f"Best validation accuracy: {best_metrics['best_val_acc']:.4f}")
    logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1 score: {test_metrics['f1']:.4f}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main() 