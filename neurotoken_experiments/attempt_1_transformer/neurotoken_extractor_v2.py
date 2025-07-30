#!/usr/bin/env python3
"""
Neurotoken Extractor v2 for FreeSurfer Outputs
Improved version with better stats file parsing and robust error handling.
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


def parse_aseg_stats(file_path):
    """Parse aseg.stats file and return dictionary of region:volume pairs"""
    if not os.path.exists(file_path):
        logger.warning(f"aseg.stats file not found: {file_path}")
        return {}
    
    try:
        stats_dict = {}
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                # Parse aseg.stats format: region_name volume
                parts = line.split()
                if len(parts) >= 2:
                    region_name = parts[0]
                    try:
                        volume = float(parts[1])
                        stats_dict[region_name] = volume
                    except ValueError:
                        continue
        
        return stats_dict
    
    except Exception as e:
        logger.error(f"Error parsing aseg.stats {file_path}: {e}")
        return {}


def parse_aparc_stats(file_path):
    """Parse aparc.stats file and return dictionary of region:metrics pairs"""
    if not os.path.exists(file_path):
        logger.warning(f"aparc.stats file not found: {file_path}")
        return {}
    
    try:
        stats_dict = {}
        current_region = None
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                # Look for region headers
                if line.startswith('#'):
                    # Extract region name from comments like "# Measure MeanThickness, Region entorhinal"
                    if 'Region' in line:
                        match = re.search(r'Region\s+(\w+)', line)
                        if match:
                            current_region = match.group(1)
                    continue
                
                # Parse data lines
                parts = line.split()
                if len(parts) >= 2 and current_region:
                    try:
                        value = float(parts[1])
                        # Store as region_metric
                        if 'MeanThickness' in line:
                            stats_dict[f"{current_region}_meanthickness"] = value
                        elif 'SurfArea' in line:
                            stats_dict[f"{current_region}_surfarea"] = value
                        elif 'GrayVol' in line:
                            stats_dict[f"{current_region}_grayvol"] = value
                    except ValueError:
                        continue
        
        return stats_dict
    
    except Exception as e:
        logger.error(f"Error parsing aparc.stats {file_path}: {e}")
        return {}


def extract_aseg_features(stats_dir):
    """Extract subcortical volumes from aseg.stats"""
    aseg_file = os.path.join(stats_dir, 'aseg.stats')
    aseg_stats = parse_aseg_stats(aseg_file)
    
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
        aparc_stats = parse_aparc_stats(aparc_file)
        
        # Extract features for each region and metric
        for region in CORTICAL_REGIONS:
            for metric in ['meanthickness', 'surfarea', 'grayvol']:
                key = f"{region}_{metric}"
                if key in aparc_stats:
                    features[f"{hemi}_{region}_{metric}"] = aparc_stats[key]
                else:
                    features[f"{hemi}_{region}_{metric}"] = np.nan
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
    logger.info("Starting neurotoken extraction v2...")
    
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
    for i, subject_dir in enumerate(subject_dirs, 1):
        logger.info(f"Processing {i}/{len(subject_dirs)}: {subject_dir}")
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
        logger.info(f"Number of columns: {len(df.columns)}")
        
        # Show sample of columns
        sample_cols = df.columns[:10].tolist()
        logger.info(f"Sample columns: {sample_cols}")
        
        # Show summary statistics
        logger.info(f"Summary of extracted data:")
        logger.info(f"  - Subjects: {df['subject_id'].nunique()}")
        logger.info(f"  - Sessions: {df['session'].nunique()}")
        logger.info(f"  - Total records: {len(df)}")
        
    else:
        logger.error("No features extracted!")
    
    logger.info("Neurotoken extraction completed!")


if __name__ == "__main__":
    main() 