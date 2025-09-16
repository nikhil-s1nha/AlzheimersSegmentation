#!/usr/bin/env python3
"""
Enhanced Dataset for Enhanced Neurotoken Model
Handles the new token structure with:
- Delta-tokens (quantile-binned)
- Level-tokens (reduced codebook size)
- Harmonized features (site-wise)
- Region embeddings (consistent order)
- Delta-t embeddings (temporal buckets)
"""

import torch
from torch.utils.data import Dataset
import json
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedNeuroTokenDataset(Dataset):
    """
    Enhanced PyTorch dataset for the new neurotoken structure
    
    Returns:
        - level_tokens: torch.Tensor - level tokens per session
        - delta_tokens: torch.Tensor - delta tokens per session  
        - harmonized_features: torch.Tensor - site-harmonized features
        - region_embeddings: torch.Tensor - region embeddings
        - delta_t_buckets: torch.Tensor - temporal bucket embeddings
        - label: int - binary classification label
    """
    
    def __init__(self, token_sequences, max_sessions=5, max_tokens=28, 
                 fit_transformers=True, transformers_path=None):
        """
        Initialize the enhanced dataset
        
        Args:
            token_sequences: List of token sequence dictionaries
            max_sessions: Maximum number of sessions to pad to
            max_tokens: Maximum number of tokens per session to pad to
            fit_transformers: Whether to fit new transformers or load existing ones
            transformers_path: Path to saved transformers (if not fitting new ones)
        """
        self.token_sequences = token_sequences
        self.max_sessions = max_sessions
        self.max_tokens = max_tokens
        self.fit_transformers = fit_transformers
        self.transformers_path = transformers_path
        
        # Process sequences and fit/load transformers
        self.processed_data, self.transformers = self._process_sequences()
        
        logger.info(f"Initialized enhanced dataset with {len(self.processed_data)} samples")
        logger.info(f"Max sessions: {max_sessions}, Max tokens: {max_tokens}")
        
        # Show class distribution
        labels = [data['label'] for data in self.processed_data]
        cn_count = sum(1 for x in labels if x == 0)
        impaired_count = sum(1 for x in labels if x == 1)
        logger.info(f"Class distribution: CN={cn_count}, Impaired={impaired_count}")
        
        # Show feature dimensions
        sample_data = self.processed_data[0]
        logger.info(f"Feature dimensions:")
        logger.info(f"  - Level tokens: {sample_data['level_tokens'].shape}")
        logger.info(f"  - Delta tokens: {sample_data['delta_tokens'].shape}")
        logger.info(f"  - Harmonized features: {sample_data['harmonized_features'].shape}")
        logger.info(f"  - Region embeddings: {sample_data['region_embeddings'].shape}")
        logger.info(f"  - Delta-t buckets: {sample_data['delta_t_buckets'].shape}")
    
    def _process_sequences(self):
        """Process token sequences and fit/load transformers"""
        processed_data = []
        
        # Extract all features for transformer fitting
        all_level_tokens = []
        all_delta_tokens = []
        all_harmonized_features = []
        all_region_embeddings = []
        all_delta_t_buckets = []
        
        for seq in self.token_sequences:
            sessions = seq['sessions']
            label = seq['label']
            
            # Extract tokens and features from sessions
            session_level_tokens = []
            session_delta_tokens = []
            session_harmonized_features = []
            session_region_embeddings = []
            session_delta_t_buckets = []
            
            for session in sessions:
                tokens = session['tokens']
                
                # Extract different token types
                level_tokens = []
                delta_tokens = []
                harmonized_features = []
                region_embeddings = []
                delta_t_bucket = 0
                
                # Parse the token dictionary to extract different types
                for key, value in tokens.items():
                    if key.startswith('level_'):
                        level_tokens.append(value)
                    elif key.startswith('binned_delta_'):
                        delta_tokens.append(value)
                    elif key.startswith('harmonized_'):
                        harmonized_features.append(value)
                    elif key.startswith('region_') and key.endswith('_embedding'):
                        region_embeddings.append(value)
                    elif key == 'delta_t_bucket':
                        delta_t_bucket = value
                
                # Pad tokens to max_tokens
                if len(level_tokens) < self.max_tokens:
                    level_tokens = level_tokens + [0] * (self.max_tokens - len(level_tokens))
                else:
                    level_tokens = level_tokens[:self.max_tokens]
                
                if len(delta_tokens) < self.max_tokens:
                    delta_tokens = delta_tokens + [0] * (self.max_tokens - len(delta_tokens))
                else:
                    delta_tokens = delta_tokens[:self.max_tokens]
                
                if len(harmonized_features) < self.max_tokens:
                    harmonized_features = harmonized_features + [0.0] * (self.max_tokens - len(harmonized_features))
                else:
                    harmonized_features = harmonized_features[:self.max_tokens]
                
                if len(region_embeddings) < self.max_tokens:
                    region_embeddings = region_embeddings + [0.0] * (self.max_tokens - len(region_embeddings))
                else:
                    region_embeddings = region_embeddings[:self.max_tokens]
                
                session_level_tokens.append(level_tokens)
                session_delta_tokens.append(delta_tokens)
                session_harmonized_features.append(harmonized_features)
                session_region_embeddings.append(region_embeddings)
                session_delta_t_buckets.append(delta_t_bucket)
            
            # Pad sessions to max_sessions
            while len(session_level_tokens) < self.max_sessions:
                session_level_tokens.append([0] * self.max_tokens)
                session_delta_tokens.append([0] * self.max_tokens)
                session_harmonized_features.append([0.0] * self.max_tokens)
                session_region_embeddings.append([0.0] * self.max_tokens)
                session_delta_t_buckets.append(0)
            
            # Truncate if too many sessions
            if len(session_level_tokens) > self.max_sessions:
                session_level_tokens = session_level_tokens[:self.max_sessions]
                session_delta_tokens = session_delta_tokens[:self.max_sessions]
                session_harmonized_features = session_harmonized_features[:self.max_sessions]
                session_region_embeddings = session_region_embeddings[:self.max_sessions]
                session_delta_t_buckets = session_delta_t_buckets[:self.max_sessions]
            
            # Store for transformer fitting
            all_level_tokens.extend(session_level_tokens)
            all_delta_tokens.extend(session_delta_tokens)
            all_harmonized_features.extend(session_harmonized_features)
            all_region_embeddings.extend(session_region_embeddings)
            all_delta_t_buckets.extend(session_delta_t_buckets)
            
            processed_data.append({
                'level_tokens': session_level_tokens,
                'delta_tokens': session_delta_tokens,
                'harmonized_features': session_harmonized_features,
                'region_embeddings': session_region_embeddings,
                'delta_t_buckets': session_delta_t_buckets,
                'label': label,
                'subject_id': seq['subject_id']
            })
        
        # Fit or load transformers
        if self.fit_transformers:
            transformers = self._fit_transformers(
                all_level_tokens, all_delta_tokens, all_harmonized_features, 
                all_region_embeddings, all_delta_t_buckets
            )
        else:
            transformers = self._load_transformers()
        
        # Apply transformers to processed data
        processed_data = self._apply_transformers(processed_data, transformers)
        
        return processed_data, transformers
    
    def _fit_transformers(self, level_tokens, delta_tokens, harmonized_features, 
                         region_embeddings, delta_t_buckets):
        """Fit transformers on the training data"""
        logger.info("Fitting transformers on training data...")
        
        transformers = {}
        
        # Convert to numpy arrays
        level_tokens = np.array(level_tokens)
        delta_tokens = np.array(delta_tokens)
        harmonized_features = np.array(harmonized_features)
        region_embeddings = np.array(region_embeddings)
        delta_t_buckets = np.array(delta_t_buckets)
        
        # 1. Level tokens: StandardScaler + QuantileTransformer
        level_scaler = StandardScaler()
        level_scaled = level_scaler.fit_transform(level_tokens.reshape(-1, level_tokens.shape[-1]))
        
        level_quantizer = QuantileTransformer(n_quantiles=10, output_distribution='uniform')
        level_quantized = level_quantizer.fit_transform(level_scaled)
        
        transformers['level_scaler'] = level_scaler
        transformers['level_quantizer'] = level_quantizer
        
        # 2. Delta tokens: QuantileTransformer (already binned, but normalize)
        delta_normalizer = StandardScaler()
        delta_normalized = delta_normalizer.fit_transform(delta_tokens.reshape(-1, delta_tokens.shape[-1]))
        
        transformers['delta_normalizer'] = delta_normalizer
        
        # 3. Harmonized features: StandardScaler
        harmonized_scaler = StandardScaler()
        harmonized_scaled = harmonized_scaler.fit_transform(harmonized_features.reshape(-1, harmonized_features.shape[-1]))
        
        transformers['harmonized_scaler'] = harmonized_scaler
        
        # 4. Region embeddings: StandardScaler
        region_scaler = StandardScaler()
        region_scaled = region_scaler.fit_transform(region_embeddings.reshape(-1, region_embeddings.shape[-1]))
        
        transformers['region_scaler'] = region_scaler
        
        # 5. Delta-t buckets: One-hot encoding
        # Already categorical, no transformation needed
        
        logger.info("Transformers fitted successfully")
        return transformers
    
    def _load_transformers(self):
        """Load pre-fitted transformers"""
        logger.info("Loading pre-fitted transformers...")
        
        if not os.path.exists(self.transformers_path):
            raise FileNotFoundError(f"Transformers file not found: {self.transformers_path}")
        
        with open(self.transformers_path, 'rb') as f:
            transformers = pickle.load(f)
        
        logger.info("Transformers loaded successfully")
        return transformers
    
    def _apply_transformers(self, processed_data, transformers):
        """Apply fitted transformers to the data"""
        logger.info("Applying transformers to data...")
        
        for data in processed_data:
            # Convert to numpy arrays
            level_tokens = np.array(data['level_tokens'])
            delta_tokens = np.array(data['delta_tokens'])
            harmonized_features = np.array(data['harmonized_features'])
            region_embeddings = np.array(data['region_embeddings'])
            
            # Apply transformations
            # Level tokens
            level_scaled = transformers['level_scaler'].transform(
                level_tokens.reshape(-1, level_tokens.shape[-1])
            )
            level_quantized = transformers['level_quantizer'].transform(level_scaled)
            data['level_tokens'] = torch.FloatTensor(level_quantized.reshape(level_tokens.shape))
            
            # Delta tokens
            delta_normalized = transformers['delta_normalizer'].transform(
                delta_tokens.reshape(-1, delta_tokens.shape[-1])
            )
            data['delta_tokens'] = torch.FloatTensor(delta_normalized.reshape(delta_tokens.shape))
            
            # Harmonized features
            harmonized_scaled = transformers['harmonized_scaler'].transform(
                harmonized_features.reshape(-1, harmonized_features.shape[-1])
            )
            data['harmonized_features'] = torch.FloatTensor(harmonized_scaled.reshape(harmonized_features.shape))
            
            # Region embeddings
            region_scaled = transformers['region_scaler'].transform(
                region_embeddings.reshape(-1, region_embeddings.shape[-1])
            )
            data['region_embeddings'] = torch.FloatTensor(region_scaled.reshape(region_embeddings.shape))
            
            # Delta-t buckets (convert to one-hot)
            delta_t_buckets = np.array(data['delta_t_buckets']).astype(int)
            delta_t_onehot = np.eye(4)[delta_t_buckets]  # 4 buckets
            data['delta_t_buckets'] = torch.FloatTensor(delta_t_onehot)
            
            # Convert label to tensor
            data['label'] = torch.LongTensor([data['label']])
        
        logger.info("Transformers applied successfully")
        return processed_data
    
    def __len__(self):
        """Return the number of samples"""
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        data = self.processed_data[idx]
        
        return {
            'level_tokens': data['level_tokens'],
            'delta_tokens': data['delta_tokens'],
            'harmonized_features': data['harmonized_features'],
            'region_embeddings': data['region_embeddings'],
            'delta_t_buckets': data['delta_t_buckets'],
            'label': data['label'],
            'subject_id': data['subject_id']
        }

def load_enhanced_tokens(token_file_path):
    """
    Load enhanced tokens from JSONL file
    
    Args:
        token_file_path: Path to enhanced_tokens.jsonl
        
    Returns:
        List of token sequence dictionaries
    """
    logger.info(f"Loading enhanced tokens from {token_file_path}")
    
    token_sequences = []
    
    with open(token_file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Group by subject_id to create sequences
            subject_id = data['subject_id']
            session = data['session']
            
            # Find existing sequence or create new one
            sequence_found = False
            for seq in token_sequences:
                if seq['subject_id'] == subject_id:
                    seq['sessions'].append({
                        'tokens': data,
                        'session': session
                    })
                    sequence_found = True
                    break
            
            if not sequence_found:
                token_sequences.append({
                    'subject_id': subject_id,
                    'sessions': [{
                        'tokens': data,
                        'session': session
                    }]
                })
    
    # Sort sessions within each sequence
    for seq in token_sequences:
        seq['sessions'].sort(key=lambda x: x['session'])
    
    logger.info(f"Loaded token sequences for {len(token_sequences)} subjects")
    return token_sequences

def load_subject_labels(csv_path):
    """
    Load subject labels for binary classification
    
    Args:
        csv_path: Path to subject_labels.csv
        
    Returns:
        Dictionary mapping subject_id to binary label (0=CN, 1=Impaired)
    """
    logger.info(f"Loading subject labels from {csv_path}")
    
    labels_df = pd.read_csv(csv_path)
    
    # Check if we have the new format with 'label' column or old format with 'class' column
    if 'label' in labels_df.columns:
        # New format: direct binary labels
        subject_labels = {}
        for _, row in labels_df.iterrows():
            subject_id = row['subject_id']
            binary_label = int(row['label'])
            subject_labels[subject_id] = binary_label
        
        logger.info(f"Loaded labels for {len(subject_labels)} subjects (new format)")
    else:
        # Old format: CN/MCI/AD classes
        binary_label_map = {"CN": 0, "MCI": 1, "AD": 1}
        subject_labels = {}
        for _, row in labels_df.iterrows():
            subject_id = row['subject_id']
            class_label = row['class']
            binary_label = binary_label_map[class_label]
            subject_labels[subject_id] = binary_label
        
        logger.info(f"Loaded labels for {len(subject_labels)} subjects (old format)")
    
    # Show label distribution
    label_counts = pd.Series(list(subject_labels.values())).value_counts().sort_index()
    logger.info(f"Label distribution: CN={label_counts.get(0, 0)}, Impaired={label_counts.get(1, 0)}")
    
    return subject_labels

def create_enhanced_sequences(token_sequences, subject_labels):
    """
    Create enhanced sequences with labels
    
    Args:
        token_sequences: List of token sequence dictionaries
        subject_labels: Dictionary mapping subject_id to binary label
        
    Returns:
        List of enhanced sequence dictionaries
    """
    logger.info("Creating enhanced sequences with labels...")
    
    enhanced_sequences = []
    
    for seq in token_sequences:
        subject_id = seq['subject_id']
        
        if subject_id not in subject_labels:
            logger.warning(f"Missing label for subject {subject_id}, skipping")
            continue
        
        # Add label to sequence
        enhanced_seq = seq.copy()
        enhanced_seq['label'] = subject_labels[subject_id]
        enhanced_sequences.append(enhanced_seq)
    
    logger.info(f"Created enhanced sequences for {len(enhanced_sequences)} subjects")
    
    # Show label distribution
    label_counts = {}
    for seq in enhanced_sequences:
        label = seq['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    logger.info(f"Label distribution: CN={label_counts.get(0, 0)}, Impaired={label_counts.get(1, 0)}")
    
    return enhanced_sequences

def split_enhanced_data(enhanced_sequences, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split enhanced data into train/val/test sets
    
    Args:
        enhanced_sequences: List of enhanced sequence dictionaries
        test_size: Fraction of data for test set
        val_size: Fraction of remaining data for validation set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_sequences, val_sequences, test_sequences)
    """
    logger.info("Splitting enhanced data into train/val/test sets...")
    
    # First split: train+val vs test
    train_val_sequences, test_sequences = train_test_split(
        enhanced_sequences, 
        test_size=test_size, 
        random_state=random_state,
        stratify=[seq['label'] for seq in enhanced_sequences]
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    train_sequences, val_sequences = train_test_split(
        train_val_sequences,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=[seq['label'] for seq in train_val_sequences]
    )
    
    logger.info(f"Data split completed:")
    logger.info(f"  - Train: {len(train_sequences)} samples")
    logger.info(f"  - Validation: {len(val_sequences)} samples")
    logger.info(f"  - Test: {len(test_sequences)} samples")
    
    return train_sequences, val_sequences, test_sequences

def main():
    """Main function to test the enhanced dataset"""
    # Example usage
    token_file = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/enhanced_attempt/enhanced_tokens.jsonl"
    labels_file = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/subject_labels.csv"
    
    try:
        # Load data
        token_sequences = load_enhanced_tokens(token_file)
        subject_labels = load_subject_labels(labels_file)
        
        # Create enhanced sequences
        enhanced_sequences = create_enhanced_sequences(token_sequences, subject_labels)
        
        # Split data
        train_sequences, val_sequences, test_sequences = split_enhanced_data(enhanced_sequences)
        
        # Create datasets
        train_dataset = EnhancedNeuroTokenDataset(train_sequences, fit_transformers=True)
        val_dataset = EnhancedNeuroTokenDataset(val_sequences, fit_transformers=False)
        test_dataset = EnhancedNeuroTokenDataset(test_sequences, fit_transformers=False)
        
        logger.info("Enhanced dataset creation completed successfully!")
        
        # Show sample data
        sample_data = train_dataset[0]
        logger.info(f"Sample data keys: {sample_data.keys()}")
        logger.info(f"Sample level_tokens shape: {sample_data['level_tokens'].shape}")
        logger.info(f"Sample label: {sample_data['label']}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main() 