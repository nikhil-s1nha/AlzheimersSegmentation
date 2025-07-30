#!/usr/bin/env python3
"""
Neurotoken Extractor for FreeSurfer Outputs
Extracts structured features from FreeSurfer recon-all outputs for Alzheimer's detection.
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SUBJECTS_DIR = "/Volumes/SEAGATE_NIKHIL/subjects"
OUTPUT_DIR = "/Volumes/SEAGATE_NIKHIL/neurotokens_project"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "all_features_combined.csv")

# Desikan-Killiany cortical regions (34 regions)
CORTICAL_REGIONS = [
    'bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'entorhinal',
    'fusiform', 'inferiorparietal', 'inferiortemporal', 'insula',
    'lateraloccipital', 'lateralorbitofrontal', 'lingual', 'medialorbitofrontal',
    'middletemporal', 'paracentral', 'parahippocampal', 'parsopercularis',
    'parsorbitalis', 'parstriangularis', 'pericalcarine', 'postcentral',
    'posteriorcingulate', 'precentral', 'precuneus', 'rostralanteriorcingulate',
    'rostralmiddlefrontal', 'superiorfrontal', 'superiorparietal', 'superiortemporal',
    'supramarginal', 'temporalpole', 'transversetemporal'
]

# Subcortical regions to extract from aseg.stats
SUBCORTICAL_REGIONS = [
    'Left-Hippocampus', 'Right-Hippocampus', 'Left-Amygdala', 'Right-Amygdala',
    'Left-Lateral-Ventricle', 'Right-Lateral-Ventricle', 'EstimatedTotalIntraCranialVol'
]

# Cortical metrics to extract
CORTICAL_METRICS = ['MeanThickness', 'SurfArea', 'GrayVol']


def parse_subject_session(folder_name):
    """Extract subject_id and session from folder name like 'OAS2_0001_session_1'"""
    match = re.match(r'(OAS2_\d+)_session_(\d+)', folder_name)
    if match:
        subject_id = match.group(1)
        session = int(match.group(2))
        return subject_id, session
    else:
        logger.warning(f"Could not parse folder name: {folder_name}")
        return None, None


def parse_stats_file(file_path, region_column='StructName', value_column='Volume'):
    """Parse FreeSurfer stats file and return dictionary of region:value pairs"""
    if not os.path.exists(file_path):
        logger.warning(f"Stats file not found: {file_path}")
        return {}
    
    try:
        # Read the stats file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find the data section (after the header)
        data_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith('#') and 'StructName' in line:
                data_start = i + 1
                break
        
        if data_start is None:
            logger.warning(f"Could not find data section in {file_path}")
            return {}
        
        # Parse the data
        stats_dict = {}
        for line in lines[data_start:]:
            line = line.strip()
            if not line:
                continue
            
            # Split by whitespace and extract values
            parts = line.split()
            if len(parts) >= 2:
                region_name = parts[0]
                try:
                    value = float(parts[1])
                    stats_dict[region_name] = value
                except ValueError:
                    continue
        
        return stats_dict
    
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
        return {}


def extract_aseg_features(stats_dir):
    """Extract subcortical volumes from aseg.stats"""
    aseg_file = os.path.join(stats_dir, 'aseg.stats')
    aseg_stats = parse_stats_file(aseg_file)
    
    features = {}
    
    # Extract raw volumes
    for region in SUBCORTICAL_REGIONS:
        if region in aseg_stats:
            features[region] = aseg_stats[region]
        else:
            features[region] = np.nan
            logger.warning(f"Region {region} not found in aseg.stats")
    
    # Calculate normalized volumes
    eTIV = aseg_stats.get('EstimatedTotalIntraCranialVol', np.nan)
    features['eTIV'] = eTIV
    
    if not np.isnan(eTIV) and eTIV > 0:
        for region in SUBCORTICAL_REGIONS:
            if region != 'EstimatedTotalIntraCranialVol' and not np.isnan(features[region]):
                features[f"{region}_norm"] = features[region] / eTIV
            else:
                features[f"{region}_norm"] = np.nan
    else:
        for region in SUBCORTICAL_REGIONS:
            if region != 'EstimatedTotalIntraCranialVol':
                features[f"{region}_norm"] = np.nan
    
    return features


def extract_cortical_features(stats_dir):
    """Extract cortical metrics from lh.aparc.stats and rh.aparc.stats"""
    features = {}
    
    for hemi in ['lh', 'rh']:
        aparc_file = os.path.join(stats_dir, f'{hemi}.aparc.stats')
        aparc_stats = parse_stats_file(aparc_file, 'StructName', 'MeanThickness')
        
        for region in CORTICAL_REGIONS:
            for metric in CORTICAL_METRICS:
                # Look for the metric in the stats
                if metric in aparc_stats:
                    features[f"{hemi}_{region}_{metric.lower()}"] = aparc_stats[metric]
                else:
                    # Try to find it in the parsed data
                    found = False
                    for line in open(aparc_file, 'r'):
                        if region in line and metric in line:
                            try:
                                value = float(line.split()[-1])
                                features[f"{hemi}_{region}_{metric.lower()}"] = value
                                found = True
                                break
                            except (ValueError, IndexError):
                                continue
                    
                    if not found:
                        features[f"{hemi}_{region}_{metric.lower()}"] = np.nan
                        logger.warning(f"Metric {metric} for region {region} in {hemi} not found")
    
    return features


def process_subject_session(subject_session_dir):
    """Process a single subject-session directory"""
    folder_name = os.path.basename(subject_session_dir)
    subject_id, session = parse_subject_session(folder_name)
    
    if subject_id is None:
        return None
    
    logger.info(f"Processing {subject_id} session {session}")
    
    # Initialize features dictionary
    features = {
        'subject_id': subject_id,
        'session': session
    }
    
    # Check if stats directory exists
    stats_dir = os.path.join(subject_session_dir, 'stats')
    if not os.path.exists(stats_dir):
        logger.warning(f"Stats directory not found for {subject_id} session {session}")
        return features
    
    # Extract aseg features (subcortical volumes)
    aseg_features = extract_aseg_features(stats_dir)
    features.update(aseg_features)
    
    # Extract cortical features
    cortical_features = extract_cortical_features(stats_dir)
    features.update(cortical_features)
    
    return features


def main():
    """Main function to extract all neurotokens"""
    logger.info("Starting neurotoken extraction...")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all subject-session directories
    if not os.path.exists(SUBJECTS_DIR):
        logger.error(f"Subjects directory not found: {SUBJECTS_DIR}")
        return
    
    subject_dirs = [d for d in os.listdir(SUBJECTS_DIR) 
                   if os.path.isdir(os.path.join(SUBJECTS_DIR, d)) 
                   and d.startswith('OAS2_')]
    
    logger.info(f"Found {len(subject_dirs)} subject-session directories")
    
    # Process each subject-session
    all_features = []
    for subject_dir in subject_dirs:
        subject_path = os.path.join(SUBJECTS_DIR, subject_dir)
        features = process_subject_session(subject_path)
        
        if features:
            all_features.append(features)
    
    # Create DataFrame and save to CSV
    if all_features:
        df = pd.DataFrame(all_features)
        df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Saved {len(df)} records to {OUTPUT_CSV}")
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
    else:
        logger.error("No features extracted!")
    
    logger.info("Neurotoken extraction completed!")


if __name__ == "__main__":
    main() 