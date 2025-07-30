#!/usr/bin/env python3
"""
Temporal Dataset for Hierarchical GRU Model
Handles two-level padding: sessions (outer) and tokens within sessions (inner).
"""

import torch
from torch.utils.data import Dataset
import json
import numpy as np
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TemporalNeuroTokenDataset(Dataset):
    """
    Custom PyTorch dataset for temporally-aware neurotoken sequences
    
    Returns:
        - input_ids: List[List[int]] - tokens per session
        - delays: List[float] - normalized delay (0 to 1)
        - label: int - binary classification label
    """
    
    def __init__(self, temporal_sequences, max_sessions=5, max_tokens=28):
        """
        Initialize the temporal dataset
        
        Args:
            temporal_sequences: List of temporal sequence dictionaries
            max_sessions: Maximum number of sessions to pad to
            max_tokens: Maximum number of tokens per session to pad to
        """
        self.temporal_sequences = temporal_sequences
        self.max_sessions = max_sessions
        self.max_tokens = max_tokens
        
        # Process sequences into padded format
        self.processed_data = self._process_sequences()
        
        logger.info(f"Initialized temporal dataset with {len(self.processed_data)} samples")
        logger.info(f"Max sessions: {max_sessions}, Max tokens: {max_tokens}")
        
        # Show class distribution
        labels = [data['label'] for data in self.processed_data]
        cn_count = sum(1 for x in labels if x == 0)
        impaired_count = sum(1 for x in labels if x == 1)
        logger.info(f"Class distribution: CN={cn_count}, Impaired={impaired_count}")
    
    def _process_sequences(self):
        """Process temporal sequences into padded format"""
        processed_data = []
        
        for seq in self.temporal_sequences:
            sessions = seq['sessions']
            label = seq['label']
            
            # Extract tokens and delays from sessions
            session_tokens = []
            session_delays = []
            
            for session in sessions:
                tokens = session['tokens']
                delay = session['delay']
                
                # Pad tokens to max_tokens
                if len(tokens) < self.max_tokens:
                    tokens = tokens + [0] * (self.max_tokens - len(tokens))
                else:
                    tokens = tokens[:self.max_tokens]
                
                session_tokens.append(tokens)
                session_delays.append(delay)
            
            # Pad sessions to max_sessions
            while len(session_tokens) < self.max_sessions:
                session_tokens.append([1] * self.max_tokens)  # Use token 1 instead of 0 for padding
                session_delays.append(0.0)
            
            # Truncate if too many sessions
            if len(session_tokens) > self.max_sessions:
                session_tokens = session_tokens[:self.max_sessions]
                session_delays = session_delays[:self.max_sessions]
            
            processed_data.append({
                'input_ids': session_tokens,
                'delays': session_delays,
                'label': label,
                'subject_id': seq['subject_id']
            })
        
        return processed_data
    
    def __len__(self):
        """Return the number of samples"""
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with input_ids, delays, attention_mask, and label tensors
        """
        data = self.processed_data[idx]
        
        # Convert to tensors
        input_ids = torch.tensor(data['input_ids'], dtype=torch.long)  # [max_sessions, max_tokens]
        delays = torch.tensor(data['delays'], dtype=torch.float)  # [max_sessions]
        label = torch.tensor(data['label'], dtype=torch.long)  # scalar
        
        # Create attention mask (1 for real tokens, 0 for padding)
        # For session-level mask: 1 if session has any non-padding tokens
        session_mask = torch.any(input_ids != 1, dim=1)  # [max_sessions]
        
        # For token-level mask: 1 for real tokens, 0 for padding
        attention_mask = (input_ids != 1)  # [max_sessions, max_tokens]
        
        return {
            "input_ids": input_ids,
            "delays": delays,
            "attention_mask": attention_mask,
            "session_mask": session_mask,
            "label": label
        }


def load_temporal_sequences(jsonl_path):
    """
    Load temporal sequences from JSONL file
    
    Args:
        jsonl_path: Path to temporal_sequences.jsonl
        
    Returns:
        List of temporal sequence dictionaries
    """
    logger.info(f"Loading temporal sequences from {jsonl_path}")
    
    temporal_sequences = []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            temporal_sequences.append(data)
    
    logger.info(f"Loaded {len(temporal_sequences)} temporal sequences")
    
    # Show some statistics
    session_counts = [len(seq['sessions']) for seq in temporal_sequences]
    token_counts = []
    for seq in temporal_sequences:
        for session in seq['sessions']:
            token_counts.append(len(session['tokens']))
    
    logger.info(f"Average sessions per subject: {np.mean(session_counts):.1f}")
    logger.info(f"Average tokens per session: {np.mean(token_counts):.1f}")
    logger.info(f"Min sessions: {min(session_counts)}, Max sessions: {max(session_counts)}")
    logger.info(f"Min tokens: {min(token_counts)}, Max tokens: {max(token_counts)}")
    
    return temporal_sequences


def create_temporal_data_splits(temporal_sequences, test_size=0.2, val_size=0.2, random_state=42):
    """
    Create train/validation/test splits for temporal sequences using stratified sampling
    
    Args:
        temporal_sequences: List of temporal sequence dictionaries
        test_size: Fraction of data for test set
        val_size: Fraction of remaining data for validation set
        random_state: Random seed for reproducibility
        
    Returns:
        train_data, val_data, test_data (each containing temporal sequences)
    """
    logger.info("Creating stratified train/validation/test splits for temporal sequences...")
    
    # Extract labels for stratification
    labels = [seq['label'] for seq in temporal_sequences]
    
    # First split: train+val vs test
    train_val_sequences, test_sequences, train_val_labels, test_labels = train_test_split(
        temporal_sequences, labels, 
        test_size=test_size, 
        stratify=labels, 
        random_state=random_state
    )
    
    # Second split: train vs val
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        train_val_sequences, train_val_labels,
        test_size=val_size,
        stratify=train_val_labels,
        random_state=random_state
    )
    
    logger.info(f"Train set: {len(train_sequences)} samples")
    logger.info(f"Validation set: {len(val_sequences)} samples")
    logger.info(f"Test set: {len(test_sequences)} samples")
    
    # Show class distribution in each split
    for split_name, split_sequences in [('Train', train_sequences), ('Val', val_sequences), ('Test', test_sequences)]:
        split_labels = [seq['label'] for seq in split_sequences]
        cn_count = sum(1 for x in split_labels if x == 0)
        impaired_count = sum(1 for x in split_labels if x == 1)
        logger.info(f"{split_name} class distribution: CN={cn_count}, Impaired={impaired_count}")
    
    return train_sequences, val_sequences, test_sequences


def create_cross_validation_splits(temporal_sequences, n_splits=5, random_state=42):
    """
    Create cross-validation splits using StratifiedKFold
    
    Args:
        temporal_sequences: List of temporal sequence dictionaries
        n_splits: Number of CV folds
        random_state: Random seed for reproducibility
        
    Returns:
        List of (train_data, val_data) tuples for each fold
    """
    logger.info(f"Creating {n_splits}-fold cross-validation splits...")
    
    # Extract labels for stratification
    labels = [seq['label'] for seq in temporal_sequences]
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_splits = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(temporal_sequences, labels)):
        # Split the data
        train_sequences = [temporal_sequences[i] for i in train_idx]
        val_sequences = [temporal_sequences[i] for i in val_idx]
        
        cv_splits.append((train_sequences, val_sequences))
        
        logger.info(f"Fold {fold+1}: Train={len(train_sequences)}, Val={len(val_sequences)}")
    
    return cv_splits


if __name__ == "__main__":
    # Test the temporal dataset
    temporal_sequences_file = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/temporal_attempt/temporal_sequences.jsonl"
    
    # Load temporal sequences
    temporal_sequences = load_temporal_sequences(temporal_sequences_file)
    
    # Create data splits
    train_sequences, val_sequences, test_sequences = create_temporal_data_splits(
        temporal_sequences, test_size=0.2, val_size=0.2
    )
    
    # Create datasets
    train_dataset = TemporalNeuroTokenDataset(train_sequences, max_sessions=5, max_tokens=28)
    val_dataset = TemporalNeuroTokenDataset(val_sequences, max_sessions=5, max_tokens=28)
    test_dataset = TemporalNeuroTokenDataset(test_sequences, max_sessions=5, max_tokens=28)
    
    # Test a sample
    sample = train_dataset[0]
    logger.info(f"Sample keys: {sample.keys()}")
    logger.info(f"Input shape: {sample['input_ids'].shape}")
    logger.info(f"Delays shape: {sample['delays'].shape}")
    logger.info(f"Attention mask shape: {sample['attention_mask'].shape}")
    logger.info(f"Label: {sample['label']}")
    
    logger.info("Temporal dataset creation completed successfully!") 