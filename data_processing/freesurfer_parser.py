#!/usr/bin/env python3
"""
FreeSurfer Output Parser for OASIS-2 Dataset
Generates NeuroTokens from aseg.stats and aparc.stats files
"""

import os
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FreeSurferParser:
    """Parser for FreeSurfer output files to extract neuroimaging metrics."""
    
    def __init__(self, subjects_dir: str):
        self.subjects_dir = subjects_dir
        
    def parse_subject(self, subject_id: str) -> Dict:
        """Parse all FreeSurfer files for a single subject."""
        subject_dir = os.path.join(self.subjects_dir, subject_id)
        if not os.path.exists(subject_dir):
            raise FileNotFoundError(f"Subject directory not found: {subject_dir}")
        
        stats = {}
        
        # Parse aseg.stats (subcortical volumes)
        aseg_path = os.path.join(subject_dir, 'stats', 'aseg.stats')
        if os.path.exists(aseg_path):
            stats['aseg'] = self._parse_aseg_stats(aseg_path)
        
        # Parse lh.aparc.stats (left hemisphere cortical)
        lh_aparc_path = os.path.join(subject_dir, 'stats', 'lh.aparc.stats')
        if os.path.exists(lh_aparc_path):
            stats['lh_aparc'] = self._parse_aparc_stats(lh_aparc_path)
        
        # Parse rh.aparc.stats (right hemisphere cortical)
        rh_aparc_path = os.path.join(subject_dir, 'stats', 'rh.aparc.stats')
        if os.path.exists(rh_aparc_path):
            stats['rh_aparc'] = self._parse_aparc_stats(rh_aparc_path)
        
        return stats
    
    def _parse_aseg_stats(self, aseg_path: str) -> Dict:
        """Parse aseg.stats file for subcortical volumes."""
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
        """Parse aparc.stats file for cortical measurements."""
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
    
    def generate_neurotokens(self, fs_stats: Dict, subject_id: str) -> Dict:
        """Generate neurotokens from FreeSurfer statistics."""
        tokens = {
            'subject_id': subject_id,
            'regions': {}
        }
        
        # Subcortical regions
        if 'aseg' in fs_stats:
            for region, volume in fs_stats['aseg'].items():
                tokens['regions'][f'aseg_{region}'] = {
                    'volume': volume,
                    'type': 'subcortical'
                }
        
        # Left hemisphere cortical regions
        if 'lh_aparc' in fs_stats:
            for region, metrics in fs_stats['lh_aparc'].items():
                tokens['regions'][f'lh_{region}'] = {
                    'thickness': metrics['thickness'],
                    'area': metrics['area'],
                    'volume': metrics['volume'],
                    'type': 'cortical_left'
                }
        
        # Right hemisphere cortical regions
        if 'rh_aparc' in fs_stats:
            for region, metrics in fs_stats['rh_aparc'].items():
                tokens['regions'][f'rh_{region}'] = {
                    'thickness': metrics['thickness'],
                    'area': metrics['area'],
                    'volume': metrics['volume'],
                    'type': 'cortical_right'
                }
        
        return tokens
    
    def compute_z_scores(self, tokens_list: List[Dict], region: str, metric: str) -> List[float]:
        """Compute z-scores for a specific region and metric across subjects."""
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
    
    def save_tokens(self, tokens: Dict, output_path: str) -> None:
        """Save neurotokens to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(tokens, f, indent=2)
    
    def load_tokens(self, input_path: str) -> Dict:
        """Load neurotokens from JSON file."""
        with open(input_path, 'r') as f:
            return json.load(f)
    
    def batch_process(self, subject_ids: List[str], output_dir: str) -> None:
        """Process multiple subjects and save their neurotokens."""
        os.makedirs(output_dir, exist_ok=True)
        
        for subject_id in subject_ids:
            try:
                fs_stats = self.parse_subject(subject_id)
                tokens = self.generate_neurotokens(fs_stats, subject_id)
                
                output_path = os.path.join(output_dir, f'{subject_id}_neurotokens.json')
                self.save_tokens(tokens, output_path)
                
            except Exception as e:
                print(f"Error processing {subject_id}: {e}")
    
    def create_dataset_csv(self, tokens_list: List[Dict], output_path: str) -> None:
        """Create CSV dataset from neurotokens."""
        rows = []
        
        for tokens in tokens_list:
            row = {'subject_id': tokens['subject_id']}
            
            for region, data in tokens['regions'].items():
                if isinstance(data, dict):
                    for metric, value in data.items():
                        if metric != 'type':
                            row[f'{region}_{metric}'] = value
                else:
                    row[region] = data
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # Example usage
    parser = FreeSurferParser("/path/to/subjects")
    
    # Process a single subject
    stats = parser.parse_subject("sub-001")
    tokens = parser.generate_neurotokens(stats, "sub-001")
    parser.save_tokens(tokens, "sub-001_tokens.json") 