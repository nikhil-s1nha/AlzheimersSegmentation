#!/usr/bin/env python3
"""
Simple training script for discrete tokens
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from enhanced_dataset_discrete import EnhancedNeuroTokenDatasetDiscrete
from enhanced_model import create_enhanced_model, count_parameters

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_simple_labels(csv_path):
    """Load subject labels from CSV with label column"""
    import pandas as pd
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

def main():
    # Configuration
    config = {
        "token_file": "enhanced_tokens.json",
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
        "num_epochs": 50,  # Reduced for faster training
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "max_grad_norm": 1.0,
        "patience": 10,
        "test_size": 0.2,
        "val_size": 0.2,
        "random_state": 42,
        "num_workers": 0,  # Reduced for compatibility
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
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
    os.makedirs(config['output_dir'], exist_ok=True)
    
    train_dataset = EnhancedNeuroTokenDatasetDiscrete(
        train_seqs, config['max_sessions'], config['max_tokens'], True, transformers_path
    )
    val_dataset = EnhancedNeuroTokenDatasetDiscrete(
        val_seqs, config['max_sessions'], config['max_tokens'], False, transformers_path
    )
    test_dataset = EnhancedNeuroTokenDatasetDiscrete(
        test_seqs, config['max_sessions'], config['max_tokens'], False, transformers_path
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Infer dimensions from sample batch
    sample_batch = next(iter(train_loader))
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
    
    logger.info(f"Model created with {count_parameters(model):,} parameters")
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Class weights for imbalanced data - simplified
    # Based on the label distribution we saw: 70 normal, 80 impaired
    weights = torch.FloatTensor([1.0/70, 1.0/80]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # Training loop
    best_val_acc = 0.0
    patience_left = config['patience']
    best_model_path = os.path.join(config['output_dir'], 'best_model_discrete.pt')
    
    for epoch in range(1, config['num_epochs'] + 1):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            logits, _ = model(batch)
            labels = batch['label'].squeeze()
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                logits, _ = model(batch)
                labels = batch['label'].squeeze()
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_loss = val_loss / len(val_loader)
        
        scheduler.step(val_acc)
        
        logger.info(f"Epoch {epoch:2d} | Train: {train_acc*100:5.2f}% | Val: {val_acc*100:5.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_left = config['patience']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'config': config
            }, best_model_path)
            logger.info(f"New best model saved with validation accuracy: {val_acc*100:.2f}%")
        else:
            patience_left -= 1
            if patience_left == 0:
                logger.info("Early stopping triggered")
                break
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc*100:.2f}%")
    
    # Test evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=device)['model_state_dict'])
    model.eval()
    
    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits, _ = model(batch)
            labels = batch['label'].squeeze()
            preds = torch.argmax(logits, dim=1)
            
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = test_correct / test_total if test_total > 0 else 0.0
    logger.info(f"Test accuracy: {test_acc*100:.2f}%")
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    tn = np.sum((all_preds == 0) & (all_labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    logger.info(f"Test Metrics:")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  Confusion Matrix:")
    logger.info(f"    TN: {tn}, FP: {fp}")
    logger.info(f"    FN: {fn}, TP: {tp}")

if __name__ == "__main__":
    main() 