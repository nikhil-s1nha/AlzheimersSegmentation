#!/usr/bin/env python3
"""
OASIS-2 FreeSurfer NeuroTokens Generator
Processes FreeSurfer output files and generates structured NeuroTokens for Transformer models
"""

import os
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import re
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class NeuroTokenDataset(Dataset):
    """Dataset for neurotoken-based transformer training."""
    
    def __init__(self, tokens_list: List[Dict], labels: List[int], max_regions: int = 100):
        self.tokens_list = tokens_list
        self.labels = labels
        self.max_regions = max_regions
        self.scaler = StandardScaler()
        
        # Prepare features
        self.features = self._prepare_features()
        
    def _prepare_features(self) -> np.ndarray:
        """Convert neurotokens to feature matrix."""
        feature_vectors = []
        
        for tokens in self.tokens_list:
            region_features = []
            for region_name, region_data in tokens['regions'].items():
                if isinstance(region_data, dict):
                    feature_vector = [
                        region_data.get('volume', 0),
                        region_data.get('thickness', 0),
                        region_data.get('area', 0)
                    ]
                else:
                    feature_vector = [region_data]
                
                region_features.extend(feature_vector)
            
            # Pad to max_regions
            while len(region_features) < self.max_regions * 3:
                region_features.append(0.0)
            
            region_features = region_features[:self.max_regions * 3]
            feature_vectors.append(region_features)
        
        features = np.array(feature_vectors)
        
        # Standardize features
        features = self.scaler.fit_transform(features)
        
        return features
    
    def __len__(self):
        return len(self.tokens_list)
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])
        return features, label

class NeuroTokenTransformer(nn.Module):
    """Transformer model for neurotoken classification."""
    
    def __init__(self, input_dim: int, num_classes: int, d_model: int = 128, 
                 nhead: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x

class OASIS2NeuroTokenProcessor:
    """Processor for OASIS-2 neurotoken generation and analysis."""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.subjects_dir = self.config['subjects_dir']
        self.output_dir = self.config['output_dir']
        
    def load_metadata(self, metadata_path: str) -> pd.DataFrame:
        """Load OASIS-2 metadata."""
        return pd.read_csv(metadata_path)
    
    def extract_freesurfer_features(self, subject_id: str) -> Dict:
        """Extract FreeSurfer features for a subject."""
        subject_dir = os.path.join(self.subjects_dir, subject_id)
        
        features = {}
        
        # Parse aseg.stats
        aseg_path = os.path.join(subject_dir, 'stats', 'aseg.stats')
        if os.path.exists(aseg_path):
            features['aseg'] = self._parse_aseg_stats(aseg_path)
        
        # Parse aparc.stats
        for hemi in ['lh', 'rh']:
            aparc_path = os.path.join(subject_dir, 'stats', f'{hemi}.aparc.stats')
            if os.path.exists(aparc_path):
                features[f'{hemi}_aparc'] = self._parse_aparc_stats(aparc_path)
        
        return features
    
    def _parse_aseg_stats(self, aseg_path: str) -> Dict:
        """Parse aseg.stats file."""
        stats = {}
        with open(aseg_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    region = parts[4]
                    volume = float(parts[3])
                    stats[region] = volume
        return stats
    
    def _parse_aparc_stats(self, aparc_path: str) -> Dict:
        """Parse aparc.stats file."""
        stats = {}
        with open(aparc_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    region = parts[4]
                    thickness = float(parts[2])
                    area = float(parts[3])
                    volume = float(parts[4])
                    stats[region] = {
                        'thickness': thickness,
                        'area': area,
                        'volume': volume
                    }
        return stats
    
    def generate_neurotokens(self, features: Dict, subject_id: str) -> Dict:
        """Generate neurotokens from FreeSurfer features."""
        tokens = {
            'subject_id': subject_id,
            'regions': {}
        }
        
        # Subcortical regions
        if 'aseg' in features:
            for region, volume in features['aseg'].items():
                tokens['regions'][f'aseg_{region}'] = {
                    'volume': volume,
                    'type': 'subcortical'
                }
        
        # Cortical regions
        for hemi in ['lh', 'rh']:
            key = f'{hemi}_aparc'
            if key in features:
                for region, metrics in features[key].items():
                    tokens['regions'][f'{hemi}_{region}'] = {
                        'thickness': metrics['thickness'],
                        'area': metrics['area'],
                        'volume': metrics['volume'],
                        'type': f'cortical_{hemi}'
                    }
        
        return tokens
    
    def compute_z_scores(self, tokens_list: List[Dict], region: str, metric: str) -> List[float]:
        """Compute z-scores for a region and metric."""
        values = []
        for tokens in tokens_list:
            if region in tokens['regions'] and metric in tokens['regions'][region]:
                values.append(tokens['regions'][region][metric])
        
        if not values:
            return []
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return [0.0] * len(values)
        
        z_scores = [(v - mean_val) / std_val for v in values]
        return z_scores
    
    def prepare_transformer_dataset(self, tokens_list: List[Dict], labels: List[int]) -> Tuple[NeuroTokenDataset, NeuroTokenDataset]:
        """Prepare train/test datasets for transformer."""
        X_train, X_test, y_train, y_test = train_test_split(
            tokens_list, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        train_dataset = NeuroTokenDataset(X_train, y_train)
        test_dataset = NeuroTokenDataset(X_test, y_test)
        
        return train_dataset, test_dataset
    
    def train_transformer(self, train_dataset: NeuroTokenDataset, test_dataset: NeuroTokenDataset,
                         num_epochs: int = 100, learning_rate: float = 0.001) -> NeuroTokenTransformer:
        """Train the transformer model."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = NeuroTokenTransformer(
            input_dim=train_dataset.features.shape[1],
            num_classes=len(set(train_dataset.labels))
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            
            for features, labels in train_loader:
                features, labels = features.to(device), labels.squeeze().to(device)
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            if epoch % 10 == 0:
                model.eval()
                test_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for features, labels in test_loader:
                        features, labels = features.to(device), labels.squeeze().to(device)
                        outputs = model(features)
                        loss = criterion(outputs, labels)
                        test_loss += loss.item()
                        
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                print(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, '
                      f'Test Loss: {test_loss/len(test_loader):.4f}, '
                      f'Test Acc: {100*correct/total:.2f}%')
        
        return model
    
    def save_results(self, tokens_list: List[Dict], model: NeuroTokenTransformer, output_dir: str):
        """Save processing results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save neurotokens
        with open(os.path.join(output_dir, 'neurotokens.json'), 'w') as f:
            json.dump(tokens_list, f, indent=2)
        
        # Save model
        torch.save(model.state_dict(), os.path.join(output_dir, 'transformer_model.pth'))
        
        # Save metadata
        metadata = {
            'num_subjects': len(tokens_list),
            'num_regions': len(tokens_list[0]['regions']) if tokens_list else 0
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OASIS-2 NeuroToken Processor')
    parser.add_argument('config', help='Path to configuration file')
    parser.add_argument('--metadata', help='Path to metadata CSV file')
    parser.add_argument('--output', help='Output directory')
    
    args = parser.parse_args()
    
    processor = OASIS2NeuroTokenProcessor(args.config)
    
    if args.metadata:
        metadata = processor.load_metadata(args.metadata)
        print(f"Loaded metadata for {len(metadata)} subjects")
    
    if args.output:
        processor.output_dir = args.output 