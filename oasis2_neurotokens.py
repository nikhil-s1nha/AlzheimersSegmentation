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
import logging
from collections import defaultdict
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OASIS2NeuroTokensProcessor:
    """Comprehensive processor for OASIS-2 FreeSurfer data to NeuroTokens."""
    
    def __init__(self, data_root: str, config: Optional[Dict] = None):
        """
        Initialize the OASIS-2 processor.
        
        Args:
            data_root: Root directory containing OASIS-2 data
            config: Optional configuration dictionary
        """
        self.data_root = Path(data_root)
        self.config = config or self._get_default_config()
        self.subjects = []
        self.region_stats = {}
        self.diagnosis_data = {}
        self.tokenizer = None
        
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'subjects_dir': 'subjects',
            'stats_dir': 'stats',
            'aseg_file': 'aseg.stats',
            'lh_aparc_file': 'lh.aparc.stats',
            'rh_aparc_file': 'rh.aparc.stats',
            'diagnosis_file': 'clinical_data.csv',
            'output_dir': 'neurotokens_output',
            'subject_prefix': 'sub-',
            'aseg_columns': {
                'region_name': 4,
                'volume': 3
            },
            'aparc_columns': {
                'region_name': 0,
                'surface_area': 2,
                'thickness': 4
            },
            'hemisphere_names': {
                'lh': 'Left',
                'rh': 'Right'
            },
            'token_format': {
                'value_precision': 1,
                'z_score_precision': 1
            },
            'transformer_config': {
                'max_tokens': 200,
                'pad_token': '[PAD]',
                'unk_token': '[UNK]',
                'sep_token': '[SEP]'
            }
        }
    
    def find_subjects(self) -> List[str]:
        """Find all subject directories in the OASIS-2 dataset."""
        subjects = []
        subjects_path = self.data_root / self.config['subjects_dir']
        
        if not subjects_path.exists():
            logger.error(f"Subjects directory {subjects_path} does not exist")
            return subjects
            
        for item in subjects_path.iterdir():
            if item.is_dir() and item.name.startswith(self.config['subject_prefix']):
                subjects.append(item.name)
                
        logger.info(f"Found {len(subjects)} subjects: {subjects[:5]}...")
        return subjects
    
    def load_diagnosis_data(self) -> Dict[str, Dict]:
        """
        Load clinical diagnosis data for subjects.
        
        Returns:
            Dictionary mapping subject IDs to diagnosis information
        """
        diagnosis_file = self.data_root / self.config['diagnosis_file']
        diagnosis_data = {}
        
        if not diagnosis_file.exists():
            logger.warning(f"Diagnosis file {diagnosis_file} not found")
            return diagnosis_data
        
        try:
            df = pd.read_csv(diagnosis_file)
            logger.info(f"Loaded diagnosis data for {len(df)} subjects")
            
            # Expected columns: subject_id, cdr_score, diagnosis, age, sex, etc.
            for _, row in df.iterrows():
                subject_id = row.get('subject_id', row.get('Subject', ''))
                if subject_id:
                    diagnosis_data[subject_id] = {
                        'cdr_score': row.get('cdr_score', row.get('CDR', np.nan)),
                        'diagnosis': row.get('diagnosis', row.get('Diagnosis', 'Unknown')),
                        'age': row.get('age', row.get('Age', np.nan)),
                        'sex': row.get('sex', row.get('Sex', 'Unknown')),
                        'mmse': row.get('mmse', row.get('MMSE', np.nan))
                    }
                    
        except Exception as e:
            logger.error(f"Error loading diagnosis data: {e}")
            
        return diagnosis_data
    
    def read_aseg_stats(self, subject_path: Path) -> Dict[str, float]:
        """
        Read aseg.stats file and extract volume measurements.
        
        Args:
            subject_path: Path to subject directory
            
        Returns:
            Dictionary mapping region names to volumes
        """
        aseg_file = subject_path / self.config['stats_dir'] / self.config['aseg_file']
        if not aseg_file.exists():
            logger.warning(f"aseg.stats not found for {subject_path.name}")
            return {}
            
        volumes = {}
        try:
            with open(aseg_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('Measure'):
                        parts = line.split()
                        if len(parts) >= max(self.config['aseg_columns'].values()) + 1:
                            region_name = parts[self.config['aseg_columns']['region_name']]
                            volume = float(parts[self.config['aseg_columns']['volume']])
                            volumes[f"[{region_name}]"] = volume
        except Exception as e:
            logger.error(f"Error reading aseg.stats for {subject_path.name}: {e}")
            
        return volumes
    
    def read_aparc_stats(self, subject_path: Path, hemisphere: str) -> Dict[str, Dict[str, float]]:
        """
        Read aparc.stats file and extract thickness and surface area measurements.
        
        Args:
            subject_path: Path to subject directory
            hemisphere: 'lh' or 'rh'
            
        Returns:
            Dictionary mapping region names to thickness and surface area
        """
        aparc_file = subject_path / self.config['stats_dir'] / self.config[f'{hemisphere}_aparc_file']
        if not aparc_file.exists():
            logger.warning(f"{hemisphere}.aparc.stats not found for {subject_path.name}")
            return {}
            
        measurements = {}
        try:
            with open(aparc_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('Measure'):
                        parts = line.split()
                        if len(parts) >= max(self.config['aparc_columns'].values()) + 1:
                            region_name = parts[self.config['aparc_columns']['region_name']]
                            thickness = float(parts[self.config['aparc_columns']['thickness']])
                            surface_area = float(parts[self.config['aparc_columns']['surface_area']])
                            
                            # Add hemisphere prefix to region name
                            hem_prefix = self.config['hemisphere_names'][hemisphere]
                            full_region_name = f"[{hem_prefix} {region_name}]"
                            
                            measurements[full_region_name] = {
                                'thickness': thickness,
                                'surface_area': surface_area
                            }
        except Exception as e:
            logger.error(f"Error reading {hemisphere}.aparc.stats for {subject_path.name}: {e}")
            
        return measurements
    
    def collect_all_measurements(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Collect measurements from all subjects.
        
        Returns:
            Dictionary mapping subject IDs to their measurements
        """
        self.subjects = self.find_subjects()
        all_measurements = {}
        
        for subject in self.subjects:
            subject_path = self.data_root / self.config['subjects_dir'] / subject
            logger.info(f"Processing subject: {subject}")
            
            # Read aseg.stats (volumes)
            volumes = self.read_aseg_stats(subject_path)
            
            # Read aparc.stats (thickness and surface area)
            lh_measurements = self.read_aparc_stats(subject_path, "lh")
            rh_measurements = self.read_aparc_stats(subject_path, "rh")
            
            # Combine all measurements
            subject_measurements = {}
            
            # Add volumes
            for region, volume in volumes.items():
                subject_measurements[region] = {'volume': volume}
            
            # Add left hemisphere measurements
            for region, measures in lh_measurements.items():
                if region in subject_measurements:
                    subject_measurements[region].update(measures)
                else:
                    subject_measurements[region] = measures
            
            # Add right hemisphere measurements
            for region, measures in rh_measurements.items():
                if region in subject_measurements:
                    subject_measurements[region].update(measures)
                else:
                    subject_measurements[region] = measures
            
            all_measurements[subject] = subject_measurements
            
        return all_measurements
    
    def compute_region_statistics(self, all_measurements: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compute mean and standard deviation for each region and measurement type.
        
        Args:
            all_measurements: Dictionary of all subject measurements
            
        Returns:
            Dictionary mapping regions to their statistics
        """
        region_stats = {}
        
        # Collect all unique regions and measurement types
        all_regions = set()
        measurement_types = set()
        
        for subject_data in all_measurements.values():
            for region, measurements in subject_data.items():
                all_regions.add(region)
                measurement_types.update(measurements.keys())
        
        # Compute statistics for each region and measurement type
        for region in all_regions:
            region_stats[region] = {}
            
            for measure_type in measurement_types:
                values = []
                
                for subject_data in all_measurements.values():
                    if region in subject_data and measure_type in subject_data[region]:
                        values.append(subject_data[region][measure_type])
                
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    region_stats[region][measure_type] = {
                        'mean': mean_val,
                        'std': std_val,
                        'count': len(values)
                    }
        
        self.region_stats = region_stats
        return region_stats
    
    def generate_neurotokens(self, subject_measurements: Dict[str, Dict[str, float]], 
                           region_stats: Dict[str, Dict[str, Dict[str, float]]]) -> List[str]:
        """
        Generate NeuroTokens for a single subject.
        
        Args:
            subject_measurements: Measurements for the subject
            region_stats: Statistics for all regions
            
        Returns:
            List of NeuroToken strings
        """
        neurotokens = []
        
        for region, measurements in subject_measurements.items():
            if region not in region_stats:
                continue
                
            for measure_type, value in measurements.items():
                if measure_type not in region_stats[region]:
                    continue
                    
                stats = region_stats[region][measure_type]
                mean_val = stats['mean']
                std_val = stats['std']
                
                if std_val > 0:  # Avoid division by zero
                    z_score = (value - mean_val) / std_val
                else:
                    z_score = 0.0
                
                # Format the NeuroToken
                value_precision = self.config['token_format']['value_precision']
                z_precision = self.config['token_format']['z_score_precision']
                
                neurotoken = f"{region}: {measure_type}={value:.{value_precision}f}, z={z_score:.{z_precision}f}"
                neurotokens.append(neurotoken)
        
        return neurotokens
    
    def create_tokenizer(self, all_neurotokens: Dict[str, List[str]]) -> Dict[str, int]:
        """
        Create a simple tokenizer for NeuroTokens.
        
        Args:
            all_neurotokens: Dictionary of NeuroTokens by subject
            
        Returns:
            Dictionary mapping tokens to indices
        """
        # Collect all unique tokens
        all_tokens = set()
        for tokens in all_neurotokens.values():
            for token in tokens:
                all_tokens.add(token)
        
        # Create token to index mapping
        token_to_idx = {
            self.config['transformer_config']['pad_token']: 0,
            self.config['transformer_config']['unk_token']: 1,
            self.config['transformer_config']['sep_token']: 2
        }
        
        # Add all unique tokens
        for token in sorted(all_tokens):
            token_to_idx[token] = len(token_to_idx)
        
        self.tokenizer = token_to_idx
        logger.info(f"Created tokenizer with {len(token_to_idx)} tokens")
        return token_to_idx
    
    def tokenize_sequence(self, neurotokens: List[str], max_length: Optional[int] = None) -> List[int]:
        """
        Convert NeuroToken sequence to numeric indices.
        
        Args:
            neurotokens: List of NeuroToken strings
            max_length: Maximum sequence length (will pad or truncate)
            
        Returns:
            List of token indices
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized. Call create_tokenizer first.")
        
        max_len = max_length or self.config['transformer_config']['max_tokens']
        
        # Convert tokens to indices
        indices = []
        for token in neurotokens:
            if token in self.tokenizer:
                indices.append(self.tokenizer[token])
            else:
                indices.append(self.tokenizer[self.config['transformer_config']['unk_token']])
        
        # Pad or truncate
        if len(indices) < max_len:
            indices.extend([self.tokenizer[self.config['transformer_config']['pad_token']]] * (max_len - len(indices)))
        else:
            indices = indices[:max_len]
        
        return indices
    
    def process_all_subjects(self) -> Dict[str, List[str]]:
        """
        Process all subjects and generate NeuroTokens.
        
        Returns:
            Dictionary mapping subject IDs to their NeuroToken lists
        """
        logger.info("Loading diagnosis data...")
        self.diagnosis_data = self.load_diagnosis_data()
        
        logger.info("Collecting measurements from all subjects...")
        all_measurements = self.collect_all_measurements()
        
        if not all_measurements:
            logger.error("No measurements found for any subjects")
            return {}
        
        logger.info("Computing region statistics...")
        region_stats = self.compute_region_statistics(all_measurements)
        
        logger.info("Generating NeuroTokens for all subjects...")
        all_neurotokens = {}
        
        for subject, measurements in all_measurements.items():
            neurotokens = self.generate_neurotokens(measurements, region_stats)
            all_neurotokens[subject] = neurotokens
            logger.info(f"Generated {len(neurotokens)} NeuroTokens for {subject}")
        
        # Create tokenizer
        logger.info("Creating tokenizer...")
        self.create_tokenizer(all_neurotokens)
        
        return all_neurotokens
    
    def save_results(self, neurotokens: Dict[str, List[str]], output_format: str = 'json'):
        """
        Save NeuroTokens and diagnosis data to files.
        
        Args:
            neurotokens: Dictionary of NeuroTokens by subject
            output_format: 'json' or 'csv'
        """
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Save NeuroTokens
        if output_format.lower() == 'json':
            # Save individual JSON files per subject
            for subject, tokens in neurotokens.items():
                subject_file = output_dir / f"{subject}_neurotokens.json"
                with open(subject_file, 'w') as f:
                    json.dump({
                        'subject_id': subject,
                        'neurotokens': tokens,
                        'diagnosis': self.diagnosis_data.get(subject, {}),
                        'token_count': len(tokens)
                    }, f, indent=2)
            
            # Save combined file
            combined_file = output_dir / "all_neurotokens.json"
            with open(combined_file, 'w') as f:
                json.dump(neurotokens, f, indent=2)
            
            logger.info(f"Saved NeuroTokens to {output_dir}")
            
        elif output_format.lower() == 'csv':
            # Flatten the data for CSV
            rows = []
            for subject, tokens in neurotokens.items():
                diagnosis = self.diagnosis_data.get(subject, {})
                for token in tokens:
                    rows.append({
                        'subject_id': subject,
                        'neurotoken': token,
                        'cdr_score': diagnosis.get('cdr_score', np.nan),
                        'diagnosis': diagnosis.get('diagnosis', 'Unknown'),
                        'age': diagnosis.get('age', np.nan),
                        'sex': diagnosis.get('sex', 'Unknown'),
                        'mmse': diagnosis.get('mmse', np.nan)
                    })
            
            df = pd.DataFrame(rows)
            csv_file = output_dir / "neurotokens_with_diagnosis.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"Saved NeuroTokens to {csv_file}")
        
        # Save diagnosis summary
        diagnosis_summary = []
        for subject in neurotokens.keys():
            diagnosis = self.diagnosis_data.get(subject, {})
            diagnosis_summary.append({
                'subject_id': subject,
                'cdr_score': diagnosis.get('cdr_score', np.nan),
                'diagnosis': diagnosis.get('diagnosis', 'Unknown'),
                'age': diagnosis.get('age', np.nan),
                'sex': diagnosis.get('sex', 'Unknown'),
                'mmse': diagnosis.get('mmse', np.nan),
                'token_count': len(neurotokens[subject])
            })
        
        diagnosis_df = pd.DataFrame(diagnosis_summary)
        diagnosis_file = output_dir / "subjects_diagnosis_summary.csv"
        diagnosis_df.to_csv(diagnosis_file, index=False)
        logger.info(f"Saved diagnosis summary to {diagnosis_file}")
        
        # Save region statistics
        stats_file = output_dir / "region_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(self.region_stats, f, indent=2)
        logger.info(f"Saved region statistics to {stats_file}")
        
        # Save tokenizer
        if self.tokenizer:
            tokenizer_file = output_dir / "tokenizer.json"
            with open(tokenizer_file, 'w') as f:
                json.dump(self.tokenizer, f, indent=2)
            logger.info(f"Saved tokenizer to {tokenizer_file}")
    
    def prepare_transformer_data(self, neurotokens: Dict[str, List[str]], 
                               max_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for Transformer model input.
        
        Args:
            neurotokens: Dictionary of NeuroTokens by subject
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (token_sequences, labels) for Transformer training
        """
        sequences = []
        labels = []
        
        for subject, tokens in neurotokens.items():
            # Tokenize sequence
            sequence = self.tokenize_sequence(tokens, max_length)
            sequences.append(sequence)
            
            # Get label (CDR score or diagnosis)
            diagnosis = self.diagnosis_data.get(subject, {})
            cdr_score = diagnosis.get('cdr_score', np.nan)
            
            if not np.isnan(cdr_score):
                labels.append(cdr_score)
            else:
                # Use diagnosis as categorical label
                diagnosis_label = diagnosis.get('diagnosis', 'Unknown')
                labels.append(diagnosis_label)
        
        return np.array(sequences), np.array(labels)


def main():
    """Main function to run the OASIS-2 processor."""
    parser = argparse.ArgumentParser(description='Process OASIS-2 FreeSurfer data and generate NeuroTokens')
    parser.add_argument('data_root', help='Root directory containing OASIS-2 data')
    parser.add_argument('--output-format', choices=['json', 'csv'], default='json',
                       help='Output format for NeuroTokens (default: json)')
    parser.add_argument('--config', help='Path to configuration JSON file')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize processor
    processor = OASIS2NeuroTokensProcessor(args.data_root, config)
    
    # Process all subjects
    neurotokens = processor.process_all_subjects()
    
    if neurotokens:
        # Save results
        processor.save_results(neurotokens, args.output_format)
        
        # Prepare Transformer data
        sequences, labels = processor.prepare_transformer_data(neurotokens)
        logger.info(f"Prepared Transformer data: {sequences.shape} sequences, {len(labels)} labels")
        
        logger.info("Processing completed successfully!")
    else:
        logger.error("No NeuroTokens were generated. Please check your data.")


if __name__ == "__main__":
    main() 