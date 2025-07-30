#!/usr/bin/env python3
"""
NeuroToken Dataset
Custom PyTorch dataset for handling neurotoken sequences and diagnostic labels.
"""

import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeuroTokenDataset(Dataset):
    """Custom PyTorch dataset for neurotoken sequences"""
    
    def __init__(self, input_ids, labels, max_len=224):
        """
        Initialize the dataset
        
        Args:
            input_ids: List of token sequences (each sequence is a list of integers)
            labels: List of integer labels (0=CN, 1=MCI, 2=AD)
            max_len: Maximum sequence length for padding/truncation
        """
        self.input_ids = input_ids
        self.labels = labels
        self.max_len = max_len
        
        logger.info(f"Initialized dataset with {len(input_ids)} samples, max_len={max_len}")
    
    def __len__(self):
        """Return the number of samples"""
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with input_ids, attention_mask, and label tensors
        """
        x = self.input_ids[idx]
        label = self.labels[idx]
        
        # Truncate if longer than max_len
        x = x[:self.max_len]
        
        # Create attention mask (1s for real tokens, 0s for padding)
        mask = [1] * len(x)
        
        # Calculate padding length
        pad_len = self.max_len - len(x)
        
        # Pad with zeros if shorter than max_len
        if pad_len > 0:
            x += [0] * pad_len
            mask += [0] * pad_len
        
        return {
            "input_ids": torch.tensor(x, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.bool),
            "label": torch.tensor(label, dtype=torch.long)
        }


def load_and_prepare_data(token_sequences_file, subject_labels_file, max_len=224):
    """
    Load and prepare data for training
    
    Args:
        token_sequences_file: Path to token_sequences.jsonl
        subject_labels_file: Path to subject_labels.csv
        max_len: Maximum sequence length
        
    Returns:
        input_ids, labels, label_map
    """
    logger.info("Loading and preparing data...")
    
    # Load token sequences
    token_sequences = []
    with open(token_sequences_file, 'r') as f:
        for line in f:
            token_sequences.append(json.loads(line))
    
    # Load subject labels
    labels_df = pd.read_csv(subject_labels_file)
    
    # Create label mapping
    label_map = {"CN": 0, "MCI": 1, "AD": 2}
    
    # Create mapping from subject_id to label
    subject_to_label = dict(zip(labels_df['subject_id'], labels_df['class']))
    
    # Join token sequences with labels
    input_ids = []
    labels = []
    subject_ids = []
    
    for seq in token_sequences:
        subject_id = seq['subject_id']
        if subject_id in subject_to_label:
            input_ids.append(seq['token_sequence'])
            labels.append(label_map[subject_to_label[subject_id]])
            subject_ids.append(subject_id)
        else:
            logger.warning(f"Subject {subject_id} not found in labels, skipping")
    
    logger.info(f"Prepared {len(input_ids)} samples for training")
    logger.info(f"Label distribution: {np.bincount(labels)}")
    
    return input_ids, labels, label_map, subject_ids


def create_data_splits(input_ids, labels, subject_ids, test_size=0.2, val_size=0.2, random_state=42):
    """
    Create train/validation/test splits using stratified sampling
    
    Args:
        input_ids: List of token sequences
        labels: List of integer labels
        subject_ids: List of subject IDs
        test_size: Fraction of data for test set
        val_size: Fraction of remaining data for validation set
        random_state: Random seed for reproducibility
        
    Returns:
        train_data, val_data, test_data (each containing input_ids, labels, subject_ids)
    """
    logger.info("Creating stratified train/validation/test splits...")
    
    # First split: train+val vs test
    train_val_input_ids, test_input_ids, train_val_labels, test_labels, train_val_subjects, test_subjects = train_test_split(
        input_ids, labels, subject_ids, 
        test_size=test_size, 
        stratify=labels, 
        random_state=random_state
    )
    
    # Second split: train vs val
    train_input_ids, val_input_ids, train_labels, val_labels, train_subjects, val_subjects = train_test_split(
        train_val_input_ids, train_val_labels, train_val_subjects,
        test_size=val_size,
        stratify=train_val_labels,
        random_state=random_state
    )
    
    # Create data dictionaries
    train_data = {
        'input_ids': train_input_ids,
        'labels': train_labels,
        'subject_ids': train_subjects
    }
    
    val_data = {
        'input_ids': val_input_ids,
        'labels': val_labels,
        'subject_ids': val_subjects
    }
    
    test_data = {
        'input_ids': test_input_ids,
        'labels': test_labels,
        'subject_ids': test_subjects
    }
    
    logger.info(f"Train set: {len(train_data['input_ids'])} samples")
    logger.info(f"Validation set: {len(val_data['input_ids'])} samples")
    logger.info(f"Test set: {len(test_data['input_ids'])} samples")
    
    # Show class distribution in each split
    for split_name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        class_counts = np.bincount(split_data['labels'])
        logger.info(f"{split_name} class distribution: CN={class_counts[0]}, MCI={class_counts[1]}, AD={class_counts[2]}")
    
    return train_data, val_data, test_data


def create_cross_validation_splits(input_ids, labels, subject_ids, n_splits=5, random_state=42):
    """
    Create cross-validation splits using StratifiedKFold
    
    Args:
        input_ids: List of token sequences
        labels: List of integer labels
        subject_ids: List of subject IDs
        n_splits: Number of CV folds
        random_state: Random seed for reproducibility
        
    Returns:
        List of (train_data, val_data) tuples for each fold
    """
    logger.info(f"Creating {n_splits}-fold cross-validation splits...")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_splits = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(input_ids, labels)):
        # Split the data
        train_input_ids = [input_ids[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        train_subjects = [subject_ids[i] for i in train_idx]
        
        val_input_ids = [input_ids[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        val_subjects = [subject_ids[i] for i in val_idx]
        
        # Create data dictionaries
        train_data = {
            'input_ids': train_input_ids,
            'labels': train_labels,
            'subject_ids': train_subjects
        }
        
        val_data = {
            'input_ids': val_input_ids,
            'labels': val_labels,
            'subject_ids': val_subjects
        }
        
        cv_splits.append((train_data, val_data))
        
        logger.info(f"Fold {fold+1}: Train={len(train_data['input_ids'])}, Val={len(val_data['input_ids'])}")
    
    return cv_splits


if __name__ == "__main__":
    # Test the dataset
    token_sequences_file = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/token_sequences.jsonl"
    subject_labels_file = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/subject_labels.csv"
    
    # Load data
    input_ids, labels, label_map, subject_ids = load_and_prepare_data(
        token_sequences_file, subject_labels_file
    )
    
    # Create splits
    train_data, val_data, test_data = create_data_splits(input_ids, labels, subject_ids)
    
    # Create datasets
    train_dataset = NeuroTokenDataset(train_data['input_ids'], train_data['labels'])
    val_dataset = NeuroTokenDataset(val_data['input_ids'], val_data['labels'])
    test_dataset = NeuroTokenDataset(test_data['input_ids'], test_data['labels'])
    
    # Test a sample
    sample = train_dataset[0]
    logger.info(f"Sample keys: {sample.keys()}")
    logger.info(f"Input shape: {sample['input_ids'].shape}")
    logger.info(f"Attention mask shape: {sample['attention_mask'].shape}")
    logger.info(f"Label: {sample['label']}")
    
    logger.info("Dataset creation completed successfully!") 