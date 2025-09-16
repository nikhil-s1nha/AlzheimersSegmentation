#!/usr/bin/env python3
"""
Training Script for Enhanced NeuroToken Model
Implements the suggested improvements with proper train-only fitting
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle

# Import our custom modules
from enhanced_dataset import (
    load_enhanced_tokens, load_subject_labels, create_enhanced_sequences,
    split_enhanced_data, EnhancedNeuroTokenDataset
)
from enhanced_model import create_enhanced_model, count_parameters

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedTrainer:
    """
    Enhanced trainer class with proper train-only fitting
    """
    
    def __init__(self, config):
        """
        Initialize the enhanced trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        
        # Create output directory
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize model and training components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Best model tracking
        self.best_val_accuracy = 0.0
        self.best_model_path = None
        
    def setup_data(self):
        """Setup data loaders with proper train-only fitting"""
        logger.info("Setting up data loaders...")
        
        # Load data
        token_sequences = load_enhanced_tokens(self.config['token_file'])
        subject_labels = load_subject_labels(self.config['labels_file'])
        
        # Create enhanced sequences
        enhanced_sequences = create_enhanced_sequences(token_sequences, subject_labels)
        
        # Split data (stratified)
        train_sequences, val_sequences, test_sequences = split_enhanced_data(
            enhanced_sequences,
            test_size=self.config['test_size'],
            val_size=self.config['val_size'],
            random_state=self.config['random_state']
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"  - Train: {len(train_sequences)} samples")
        logger.info(f"  - Validation: {len(val_sequences)} samples")
        logger.info(f"  - Test: {len(test_sequences)} samples")
        
        # Create datasets with proper transformer fitting
        # Train dataset: fit new transformers
        self.train_dataset = EnhancedNeuroTokenDataset(
            train_sequences,
            max_sessions=self.config['max_sessions'],
            max_tokens=self.config['max_tokens'],
            fit_transformers=True
        )
        
        # Save transformers for validation and test
        transformers_path = os.path.join(self.output_dir, 'transformers.pkl')
        with open(transformers_path, 'wb') as f:
            pickle.dump(self.train_dataset.transformers, f)
        logger.info(f"Saved transformers to {transformers_path}")
        
        # Validation dataset: use fitted transformers
        self.val_dataset = EnhancedNeuroTokenDataset(
            val_sequences,
            max_sessions=self.config['max_sessions'],
            max_tokens=self.config['max_tokens'],
            fit_transformers=False,
            transformers_path=transformers_path
        )
        
        # Test dataset: use fitted transformers
        self.test_dataset = EnhancedNeuroTokenDataset(
            test_sequences,
            max_sessions=self.config['max_sessions'],
            max_tokens=self.config['max_tokens'],
            fit_transformers=False,
            transformers_path=transformers_path
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers']
        )
        
        logger.info("Data loaders setup completed")
        
        # Show data dimensions
        sample_batch = next(iter(self.train_loader))
        logger.info(f"Sample batch dimensions:")
        for key, value in sample_batch.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: {value.shape}")
            else:
                logger.info(f"  {key}: {value}")
    
    def setup_model(self):
        """Setup the enhanced model"""
        logger.info("Setting up enhanced model...")
        
        # Get feature dimensions from dataset
        sample_batch = next(iter(self.train_loader))
        
        # Extract dimensions
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
        self.model = create_enhanced_model(
            model_type=self.config['model_type'],
            max_sessions=self.config['max_sessions'],
            max_tokens=self.config['max_tokens'],
            level_token_dim=level_token_dim,
            delta_token_dim=delta_token_dim,
            harmonized_dim=harmonized_dim,
            region_embedding_dim=region_embedding_dim,
            delta_t_bucket_dim=delta_t_bucket_dim,
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            dropout=self.config['dropout'],
            num_classes=self.config['num_classes']
        )
        
        self.model = self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
        # Setup loss function
        if self.config['use_class_weights']:
            # Calculate class weights
            train_labels = [data['label'].item() for data in self.train_dataset]
            class_counts = np.bincount(train_labels)
            class_weights = 1.0 / class_counts
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info(f"Using class weights: {class_weights.cpu().numpy()}")
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Model setup completed. Parameters: {count_parameters(self.model):,}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            # Move data to device
            batch_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch_data.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(batch_data)
            labels = batch_data['label'].squeeze()
            
            # Calculate loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Log progress
            if batch_idx % self.config['log_interval'] == 0:
                logger.info(f"Train batch {batch_idx}/{len(self.train_loader)}: "
                          f"Loss: {loss.item():.4f}, "
                          f"Accuracy: {100.0 * correct_predictions / total_predictions:.2f}%")
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_accuracy = 100.0 * correct_predictions / total_predictions
        
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                # Move data to device
                batch_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in batch_data.items()}
                
                # Forward pass
                logits, _ = self.model(batch_data)
                labels = batch_data['label'].squeeze()
                
                # Calculate loss
                loss = self.criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)
                
                # Store predictions and labels
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = total_loss / len(self.val_loader)
        epoch_accuracy = 100.0 * correct_predictions / total_predictions
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary', zero_division=0
        )
        
        return epoch_loss, epoch_accuracy, precision, recall, f1
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': self.best_val_accuracy
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if this is the best so far
        if is_best:
            best_model_path = os.path.join(self.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_model_path)
            self.best_model_path = best_model_path
            logger.info(f"New best model saved: {best_model_path}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Loss difference
        loss_diff = [abs(t - v) for t, v in zip(self.train_losses, self.val_losses)]
        ax3.plot(loss_diff, label='|Train Loss - Val Loss|')
        ax3.set_title('Loss Difference (Overfitting Indicator)')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss Difference')
        ax3.legend()
        ax3.grid(True)
        
        # Accuracy difference
        acc_diff = [abs(t - v) for t, v in zip(self.train_accuracies, self.val_accuracies)]
        ax4.plot(acc_diff, label='|Train Acc - Val Acc|')
        ax4.set_title('Accuracy Difference (Overfitting Indicator)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy Difference (%)')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plot saved to {plot_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting enhanced training...")
        
        # Setup data and model
        self.setup_data()
        self.setup_model()
        
        # Training loop
        for epoch in range(1, self.config['num_epochs'] + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{self.config['num_epochs']}")
            logger.info(f"{'='*50}")
            
            # Train
            train_loss, train_accuracy = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            
            # Validate
            val_loss, val_accuracy, precision, recall, f1 = self.validate_epoch()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Log results
            logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
            logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_accuracy)
            
            # Save checkpoint
            is_best = val_accuracy > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_accuracy
            
            if epoch % self.config['save_interval'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping check
            if len(self.val_accuracies) >= self.config['patience']:
                recent_accuracies = self.val_accuracies[-self.config['patience']:]
                if all(acc <= self.best_val_accuracy for acc in recent_accuracies):
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
        
        # Save final model
        self.save_checkpoint(self.config['num_epochs'])
        
        # Plot training history
        self.plot_training_history()
        
        # Save training metrics
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': self.best_val_accuracy,
            'final_epoch': len(self.train_losses)
        }
        
        metrics_path = os.path.join(self.output_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Training completed! Best validation accuracy: {self.best_val_accuracy:.2f}%")
        logger.info(f"Training metrics saved to {metrics_path}")

def main():
    """Main function"""
    # Configuration
    config = {
        # Data paths
        'token_file': '/Volumes/SEAGATE_NIKHIL/neurotokens_project/enhanced_attempt/enhanced_tokens.jsonl',
        'labels_file': '/Volumes/SEAGATE_NIKHIL/neurotokens_project/subject_labels.csv',
        'output_dir': '/Volumes/SEAGATE_NIKHIL/neurotokens_project/enhanced_attempt/models',
        
        # Model parameters
        'model_type': 'gru',  # 'gru' or 'transformer'
        'max_sessions': 5,
        'max_tokens': 28,
        'hidden_dim': 128,
        'num_layers': 2,
        'num_heads': 8,
        'dropout': 0.3,
        'num_classes': 2,
        
        # Training parameters
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'max_grad_norm': 1.0,
        'patience': 10,
        
        # Data split
        'test_size': 0.2,
        'val_size': 0.2,
        'random_state': 42,
        
        # Data loading
        'num_workers': 4,
        
        # Logging and saving
        'log_interval': 10,
        'save_interval': 10,
        
        # Class balancing
        'use_class_weights': True
    }
    
    # Create trainer and start training
    trainer = EnhancedTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main() 