#!/usr/bin/env python3
"""
Binary Classification Dataset for NeuroTokens
Handles CN vs Impaired (MCI+AD) classification with class balancing.
"""

import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import resample
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinaryNeuroTokenDataset(Dataset):
    """Custom PyTorch dataset for binary neurotoken classification"""
    
    def __init__(self, input_ids, labels, max_len=224):
        """
        Initialize the dataset
        
        Args:
            input_ids: List of token sequences (each sequence is a list of integers)
            labels: List of binary labels (0=CN, 1=Impaired)
            max_len: Maximum sequence length for padding/truncation
        """
        self.input_ids = input_ids
        self.labels = labels
        self.max_len = max_len
        
        logger.info(f"Initialized binary dataset with {len(input_ids)} samples, max_len={max_len}")
        logger.info(f"Class distribution: CN={sum(1 for x in labels if x == 0)}, Impaired={sum(1 for x in labels if x == 1)}")
    
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


def load_and_prepare_binary_data(token_sequences_file, subject_labels_file, max_len=224):
    """
    Load and prepare data for binary classification (CN vs Impaired)
    
    Args:
        token_sequences_file: Path to token_sequences.jsonl
        subject_labels_file: Path to subject_labels.csv
        max_len: Maximum sequence length
        
    Returns:
        input_ids, labels, label_map, subject_ids
    """
    logger.info("Loading and preparing binary classification data...")
    
    # Load token sequences
    token_sequences = []
    with open(token_sequences_file, 'r') as f:
        for line in f:
            token_sequences.append(json.loads(line))
    
    # Load subject labels
    labels_df = pd.read_csv(subject_labels_file)
    
    # Create binary label mapping: CN=0, MCI+AD=1
    binary_label_map = {"CN": 0, "MCI": 1, "AD": 1}
    
    # Create mapping from subject_id to binary label
    subject_to_label = dict(zip(labels_df['subject_id'], labels_df['class']))
    
    # Join token sequences with labels
    input_ids = []
    labels = []
    subject_ids = []
    
    for seq in token_sequences:
        subject_id = seq['subject_id']
        if subject_id in subject_to_label:
            input_ids.append(seq['token_sequence'])
            labels.append(binary_label_map[subject_to_label[subject_id]])
            subject_ids.append(subject_id)
        else:
            logger.warning(f"Subject {subject_id} not found in labels, skipping")
    
    logger.info(f"Prepared {len(input_ids)} samples for binary classification")
    logger.info(f"Original class distribution: CN={sum(1 for x in labels if x == 0)}, Impaired={sum(1 for x in labels if x == 1)}")
    
    return input_ids, labels, binary_label_map, subject_ids


def balance_classes(input_ids, labels, subject_ids, method='downsample'):
    """
    Balance classes using downsampling or upsampling
    
    Args:
        input_ids: List of token sequences
        labels: List of binary labels
        subject_ids: List of subject IDs
        method: 'downsample' or 'upsample'
        
    Returns:
        balanced_input_ids, balanced_labels, balanced_subject_ids
    """
    logger.info(f"Balancing classes using {method} method...")
    
    # Convert to numpy arrays for easier manipulation
    input_ids_np = np.array(input_ids, dtype=object)
    labels_np = np.array(labels)
    subject_ids_np = np.array(subject_ids)
    
    # Separate classes
    cn_mask = labels_np == 0
    impaired_mask = labels_np == 1
    
    cn_input_ids = input_ids_np[cn_mask]
    cn_labels = labels_np[cn_mask]
    cn_subjects = subject_ids_np[cn_mask]
    
    impaired_input_ids = input_ids_np[impaired_mask]
    impaired_labels = labels_np[impaired_mask]
    impaired_subjects = subject_ids_np[impaired_mask]
    
    logger.info(f"Before balancing: CN={len(cn_input_ids)}, Impaired={len(impaired_input_ids)}")
    
    if method == 'downsample':
        # Downsample CN to match impaired count
        if len(cn_input_ids) > len(impaired_input_ids):
            # Randomly sample CN subjects to match impaired count
            n_samples = len(impaired_input_ids)
            indices = np.random.choice(len(cn_input_ids), n_samples, replace=False)
            
            cn_input_ids = cn_input_ids[indices]
            cn_labels = cn_labels[indices]
            cn_subjects = cn_subjects[indices]
            
            logger.info(f"Downsampled CN to {len(cn_input_ids)} samples")
    
    elif method == 'upsample':
        # Upsample impaired to match CN count
        if len(impaired_input_ids) < len(cn_input_ids):
            # Randomly sample with replacement to match CN count
            n_samples = len(cn_input_ids)
            indices = np.random.choice(len(impaired_input_ids), n_samples, replace=True)
            
            impaired_input_ids = impaired_input_ids[indices]
            impaired_labels = impaired_labels[indices]
            impaired_subjects = impaired_subjects[indices]
            
            logger.info(f"Upsampled Impaired to {len(impaired_input_ids)} samples")
    
    # Combine balanced classes
    balanced_input_ids = np.concatenate([cn_input_ids, impaired_input_ids])
    balanced_labels = np.concatenate([cn_labels, impaired_labels])
    balanced_subject_ids = np.concatenate([cn_subjects, impaired_subjects])
    
    # Shuffle the data
    indices = np.random.permutation(len(balanced_input_ids))
    balanced_input_ids = balanced_input_ids[indices]
    balanced_labels = balanced_labels[indices]
    balanced_subject_ids = balanced_subject_ids[indices]
    
    logger.info(f"After balancing: CN={sum(1 for x in balanced_labels if x == 0)}, Impaired={sum(1 for x in balanced_labels if x == 1)}")
    
    return balanced_input_ids.tolist(), balanced_labels.tolist(), balanced_subject_ids.tolist()


def create_binary_data_splits(input_ids, labels, subject_ids, test_size=0.2, val_size=0.2, random_state=42):
    """
    Create train/validation/test splits for binary classification using stratified sampling
    
    Args:
        input_ids: List of token sequences
        labels: List of binary labels
        subject_ids: List of subject IDs
        test_size: Fraction of data for test set
        val_size: Fraction of remaining data for validation set
        random_state: Random seed for reproducibility
        
    Returns:
        train_data, val_data, test_data (each containing input_ids, labels, subject_ids)
    """
    logger.info("Creating stratified train/validation/test splits for binary classification...")
    
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
        cn_count = sum(1 for x in split_data['labels'] if x == 0)
        impaired_count = sum(1 for x in split_data['labels'] if x == 1)
        logger.info(f"{split_name} class distribution: CN={cn_count}, Impaired={impaired_count}")
    
    return train_data, val_data, test_data


if __name__ == "__main__":
    # Test the binary dataset
    token_sequences_file = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/token_sequences.jsonl"
    subject_labels_file = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/subject_labels.csv"
    
    # Load data
    input_ids, labels, label_map, subject_ids = load_and_prepare_binary_data(
        token_sequences_file, subject_labels_file
    )
    
    # Balance classes
    balanced_input_ids, balanced_labels, balanced_subject_ids = balance_classes(
        input_ids, labels, subject_ids, method='downsample'
    )
    
    # Create splits
    train_data, val_data, test_data = create_binary_data_splits(
        balanced_input_ids, balanced_labels, balanced_subject_ids
    )
    
    # Create datasets
    train_dataset = BinaryNeuroTokenDataset(train_data['input_ids'], train_data['labels'])
    val_dataset = BinaryNeuroTokenDataset(val_data['input_ids'], val_data['labels'])
    test_dataset = BinaryNeuroTokenDataset(test_data['input_ids'], test_data['labels'])
    
    # Test a sample
    sample = train_dataset[0]
    logger.info(f"Sample keys: {sample.keys()}")
    logger.info(f"Input shape: {sample['input_ids'].shape}")
    logger.info(f"Attention mask shape: {sample['attention_mask'].shape}")
    logger.info(f"Label: {sample['label']}")
    
    logger.info("Binary dataset creation completed successfully!") 