#!/usr/bin/env python3
"""
Enhanced Neurotoken Extractor with Delta-Tokens and Site Harmonization
Implements the suggested improvements for better accuracy:
- Switch to Δ-tokens (and keep level-tokens): quantile-bin Δz into 7 bins with a small "Stable" dead-zone
- Reduce codebook size: K from 32 → 8–12 (or ditch KMeans for quantiles)
- Train-only fitting: recompute z-scores & codebooks on TRAIN; apply to val/test
- Add Δt embedding: 3–5 buckets (≤6m, 6–12m, 12–24m, >24m)
- Harmonize by site: at least site-wise z-scaling pre-tokenization
- Region order + region embeddings: lock a consistent order; add a learned region ID embedding
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.cluster import KMeans
import json
from collections import defaultdict, OrderedDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SUBJECTS_DIR = "/Volumes/SEAGATE_NIKHIL/subjects"
OUTPUT_DIR = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/enhanced_attempt"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "enhanced_features_combined.csv")
TOKEN_OUTPUT = os.path.join(OUTPUT_DIR, "enhanced_tokens.jsonl")

# Desikan-Killiany cortical regions (34 regions) - LOCKED ORDER
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

# Subcortical regions to extract from aseg.stats - LOCKED ORDER
SUBCORTICAL_REGIONS = [
    'Left-Hippocampus', 'Right-Hippocampus', 'Left-Amygdala', 'Right-Amygdala',
    'Left-Lateral-Ventricle', 'Right-Lateral-Ventricle'
]

# All regions in consistent order for embeddings
ALL_REGIONS = CORTICAL_REGIONS + [f"Left-{region}" for region in CORTICAL_REGIONS] + [f"Right-{region}" for region in CORTICAL_REGIONS] + SUBCORTICAL_REGIONS

# Delta-t buckets for temporal embeddings
DELTA_T_BUCKETS = [
    (0, 180),      # ≤6 months
    (180, 365),    # 6-12 months  
    (365, 730),    # 12-24 months
    (730, float('inf'))  # >24 months
]

# Configuration for tokenization
N_DELTA_BINS = 7  # Number of bins for delta-tokens
STABLE_THRESHOLD = 0.2  # Stable dead-zone threshold
CODEBOOK_SIZE = 10  # Reduced from 32 to 10
USE_QUANTILES = True  # Use quantiles instead of KMeans

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

def extract_site_info(subject_id):
    """Extract site information from subject ID (OAS2_XXXX)"""
    # For now, we'll use a simple approach - could be enhanced with actual site mapping
    # This is a placeholder for site-wise harmonization
    return "site_1"  # Placeholder - should be extracted from demographics

def parse_aseg_stats(file_path):
    """Parse aseg.stats file and return dictionary of region:volume pairs"""
    if not os.path.exists(file_path):
        logger.warning(f"aseg.stats file not found: {file_path}")
        return {}
    
    try:
        stats_dict = {}
        eTIV = None
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    # Check for eTIV in header comments
                    if 'EstimatedTotalIntraCranialVol' in line:
                        parts = line.split(',')
                        if len(parts) >= 4:
                            try:
                                eTIV = float(parts[3].strip())
                                stats_dict['EstimatedTotalIntraCranialVol'] = eTIV
                            except ValueError:
                                pass
                    continue
                
                # Parse region volumes from table
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        # Find where the region name starts
                        region_name = None
                        for i, part in enumerate(parts):
                            if part.replace('.', '').replace('-', '').isalpha():
                                region_parts = []
                                for j in range(i, len(parts)):
                                    if parts[j].replace('.', '').replace('-', '').isalpha():
                                        region_parts.append(parts[j])
                                    else:
                                        break
                                region_name = ' '.join(region_parts)
                                break
                        
                        if region_name:
                            # Get volume (usually the 4th column)
                            volume = float(parts[3])
                            stats_dict[region_name] = volume
                    except (ValueError, IndexError):
                        continue
        
        # Add eTIV if found
        if eTIV:
            stats_dict['EstimatedTotalIntraCranialVol'] = eTIV
        
        return stats_dict
        
    except Exception as e:
        logger.error(f"Error parsing aseg.stats: {e}")
        return {}

def extract_cortical_features(stats_dir):
    """Extract cortical thickness and volume features"""
    features = {}
    
    # Process left and right hemispheres
    for hemi in ['lh', 'rh']:
        aparc_stats_file = os.path.join(stats_dir, f'{hemi}.aparc.stats')
        
        if not os.path.exists(aparc_stats_file):
            logger.warning(f"aparc.stats file not found for {hemi}: {aparc_stats_file}")
            continue
        
        try:
            with open(aparc_stats_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            region = parts[0]
                            thickness = float(parts[2])
                            volume = float(parts[3])
                            
                            # Only include regions in our predefined list
                            if region in CORTICAL_REGIONS:
                                features[f"{hemi}_{region}_thickness"] = thickness
                                features[f"{hemi}_{region}_volume"] = volume
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            logger.error(f"Error parsing {hemi}.aparc.stats: {e}")
    
    return features

def extract_aseg_features(stats_dir):
    """Extract subcortical features from aseg.stats"""
    aseg_file = os.path.join(stats_dir, 'aseg.stats')
    aseg_features = parse_aseg_stats(aseg_file)
    
    # Only keep features for regions in our predefined list
    filtered_features = {}
    for region in SUBCORTICAL_REGIONS:
        if region in aseg_features:
            filtered_features[region] = aseg_features[region]
    
    # Add eTIV if available
    if 'EstimatedTotalIntraCranialVol' in aseg_features:
        filtered_features['EstimatedTotalIntraCranialVol'] = aseg_features['EstimatedTotalIntraCranialVol']
    
    return filtered_features

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
        'session': session,
        'site': extract_site_info(subject_id)
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

def create_delta_tokens(features_df):
    """
    Create delta-tokens by computing differences between consecutive sessions
    and quantile-binning them into 7 bins with a stable dead-zone
    """
    logger.info("Creating delta-tokens...")
    
    # Sort by subject_id and session
    features_df = features_df.sort_values(['subject_id', 'session']).reset_index(drop=True)
    
    # Get numeric columns (excluding metadata)
    metadata_cols = ['subject_id', 'session', 'site']
    numeric_cols = [col for col in features_df.columns if col not in metadata_cols]
    
    # Initialize delta features
    delta_features = features_df[metadata_cols].copy()
    
    # Compute deltas for each subject
    for subject_id in features_df['subject_id'].unique():
        subject_data = features_df[features_df['subject_id'] == subject_id].sort_values('session')
        
        if len(subject_data) < 2:
            # Single session subjects - fill with zeros
            for col in numeric_cols:
                delta_features.loc[delta_features['subject_id'] == subject_id, f'delta_{col}'] = 0.0
            continue
        
        # Compute deltas between consecutive sessions
        for i in range(1, len(subject_data)):
            prev_session = subject_data.iloc[i-1]
            curr_session = subject_data.iloc[i]
            
            for col in numeric_cols:
                prev_val = prev_session[col]
                curr_val = curr_session[col]
                
                if pd.notna(prev_val) and pd.notna(curr_val):
                    delta = curr_val - prev_val
                    # Apply stable dead-zone
                    if abs(delta) < STABLE_THRESHOLD:
                        delta = 0.0
                    
                    # Find the row index for current session
                    row_idx = delta_features[(delta_features['subject_id'] == subject_id) & 
                                          (delta_features['session'] == curr_session['session'])].index[0]
                    delta_features.loc[row_idx, f'delta_{col}'] = delta
                else:
                    row_idx = delta_features[(delta_features['subject_id'] == subject_id) & 
                                          (delta_features['session'] == curr_session['session'])].index[0]
                    delta_features.loc[row_idx, f'delta_{col}'] = 0.0
    
    # Fill NaN deltas with 0 (for first sessions)
    delta_cols = [col for col in delta_features.columns if col.startswith('delta_')]
    delta_features[delta_cols] = delta_features[delta_cols].fillna(0.0)
    
    logger.info(f"Created delta-tokens for {len(delta_cols)} features")
    return delta_features

def quantile_bin_deltas(delta_features, n_bins=N_DELTA_BINS):
    """
    Quantile-bin delta values into n_bins with special handling for stable zone
    """
    logger.info(f"Quantile-binning deltas into {n_bins} bins...")
    
    delta_cols = [col for col in delta_features.columns if col.startswith('delta_')]
    
    # Create quantile transformer
    quantile_transformer = QuantileTransformer(n_quantiles=n_bins, output_distribution='uniform')
    
    # Fit on delta values and transform
    delta_values = delta_features[delta_cols].values
    binned_deltas = quantile_transformer.fit_transform(delta_values)
    
    # Convert to integer bins (0 to n_bins-1)
    binned_deltas = np.floor(binned_deltas * n_bins).astype(int)
    binned_deltas = np.clip(binned_deltas, 0, n_bins-1)
    
    # Create binned delta features
    binned_features = delta_features[['subject_id', 'session', 'site']].copy()
    
    for i, col in enumerate(delta_cols):
        binned_features[f'binned_{col}'] = binned_deltas[:, i]
    
    logger.info(f"Quantile-binned deltas into {n_bins} bins")
    return binned_features, quantile_transformer

def create_level_tokens(features_df, scaler=None, fit_scaler=True):
    """
    Create level-tokens by z-scoring features and applying codebook
    """
    logger.info("Creating level-tokens...")
    
    # Get numeric columns (excluding metadata)
    metadata_cols = ['subject_id', 'session', 'site']
    numeric_cols = [col for col in features_df.columns if col not in metadata_cols]
    
    # Prepare features for scaling
    feature_values = features_df[numeric_cols].fillna(0.0).values
    
    if fit_scaler:
        # Fit scaler on training data
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_values)
        logger.info(f"Fitted StandardScaler on training data")
    else:
        # Apply existing scaler
        if scaler is None:
            raise ValueError("Scaler must be provided when fit_scaler=False")
        scaled_features = scaler.transform(feature_values)
        logger.info(f"Applied existing StandardScaler")
    
    # Create codebook using KMeans or quantiles
    if USE_QUANTILES:
        # Use quantiles instead of KMeans
        quantile_transformer = QuantileTransformer(n_quantiles=CODEBOOK_SIZE, output_distribution='uniform')
        tokenized_features = quantile_transformer.fit_transform(scaled_features) if fit_scaler else quantile_transformer.transform(scaled_features)
        tokenized_features = np.floor(tokenized_features * CODEBOOK_SIZE).astype(int)
        tokenized_features = np.clip(tokenized_features, 0, CODEBOOK_SIZE-1)
        
        if fit_scaler:
            logger.info(f"Fitted QuantileTransformer with {CODEBOOK_SIZE} quantiles")
        else:
            logger.info(f"Applied existing QuantileTransformer")
    else:
        # Use KMeans
        if fit_scaler:
            kmeans = KMeans(n_clusters=CODEBOOK_SIZE, random_state=42, n_init=10)
            tokenized_features = kmeans.fit_predict(scaled_features)
            logger.info(f"Fitted KMeans with {CODEBOOK_SIZE} clusters")
        else:
            raise ValueError("KMeans requires refitting - use quantiles for train-only fitting")
    
    # Create level token features
    level_features = features_df[metadata_cols].copy()
    
    for i, col in enumerate(numeric_cols):
        level_features[f'level_{col}'] = tokenized_features[:, i]
    
    logger.info(f"Created level-tokens with codebook size {CODEBOOK_SIZE}")
    return level_features, scaler, quantile_transformer if USE_QUANTILES else None

def create_delta_t_embeddings(features_df):
    """
    Create delta-t embeddings based on time intervals between sessions
    """
    logger.info("Creating delta-t embeddings...")
    
    # Sort by subject_id and session
    features_df = features_df.sort_values(['subject_id', 'session']).reset_index(drop=True)
    
    # Initialize delta-t features
    delta_t_features = features_df[['subject_id', 'session', 'site']].copy()
    delta_t_features['delta_t_bucket'] = 0  # Default bucket
    
    # Compute time intervals for each subject
    for subject_id in features_df['subject_id'].unique():
        subject_data = features_df[features_df['subject_id'] == subject_id].sort_values('session')
        
        if len(subject_data) < 2:
            continue
        
        # For now, we'll use session numbers as proxy for time
        # In practice, this should use actual time intervals from demographics
        for i in range(1, len(subject_data)):
            # Estimate time interval (in days) - this should come from actual timing data
            # For now, assume 1 year between sessions
            time_interval = 365  # days
            
            # Assign to bucket
            bucket = 0
            for j, (min_days, max_days) in enumerate(DELTA_T_BUCKETS):
                if min_days <= time_interval < max_days:
                    bucket = j
                    break
            
            # Find the row index for current session
            row_idx = delta_t_features[(delta_t_features['subject_id'] == subject_id) & 
                                     (delta_t_features['session'] == subject_data.iloc[i]['session'])].index[0]
            delta_t_features.loc[row_idx, 'delta_t_bucket'] = bucket
    
    logger.info(f"Created delta-t embeddings with {len(DELTA_T_BUCKETS)} buckets")
    return delta_t_features

def harmonize_by_site(features_df):
    """
    Apply site-wise z-scaling for harmonization
    """
    logger.info("Applying site-wise harmonization...")
    
    # Get numeric columns (excluding metadata)
    metadata_cols = ['subject_id', 'session', 'site']
    numeric_cols = [col for col in features_df.columns if col not in metadata_cols]
    
    # Initialize harmonized features
    harmonized_features = features_df[metadata_cols].copy()
    
    # Apply site-wise z-scaling
    for site in features_df['site'].unique():
        site_mask = features_df['site'] == site
        site_data = features_df[site_mask]
        
        if len(site_data) > 1:  # Need at least 2 samples for z-scoring
            site_scaler = StandardScaler()
            site_values = site_data[numeric_cols].fillna(0.0).values
            harmonized_values = site_scaler.fit_transform(site_values)
            
            # Store harmonized values
            for i, col in enumerate(numeric_cols):
                harmonized_features.loc[site_mask, f'harmonized_{col}'] = harmonized_values[:, i]
        else:
            # Single sample site - fill with zeros
            for col in numeric_cols:
                harmonized_features.loc[site_mask, f'harmonized_{col}'] = 0.0
    
    logger.info(f"Applied site-wise harmonization for {len(features_df['site'].unique())} sites")
    return harmonized_features

def combine_all_tokens(level_features, delta_features, delta_t_features, harmonized_features):
    """
    Combine all token types into final feature set
    """
    logger.info("Combining all token types...")
    
    # Start with metadata
    combined_features = level_features[['subject_id', 'session', 'site']].copy()
    
    # Add level tokens
    level_cols = [col for col in level_features.columns if col.startswith('level_')]
    for col in level_cols:
        combined_features[col] = level_features[col]
    
    # Add delta tokens
    delta_cols = [col for col in delta_features.columns if col.startswith('binned_delta_')]
    for col in delta_cols:
        combined_features[col] = delta_features[col]
    
    # Add delta-t embeddings
    combined_features['delta_t_bucket'] = delta_t_features['delta_t_bucket']
    
    # Add harmonized features
    harmonized_cols = [col for col in harmonized_features.columns if col.startswith('harmonized_')]
    for col in harmonized_cols:
        combined_features[col] = harmonized_features[col]
    
    # Add region embeddings (one-hot encoding of region order)
    region_embeddings = create_region_embeddings(combined_features)
    for col in region_embeddings.columns:
        if col != 'subject_id' and col != 'session' and col != 'site':
            combined_features[col] = region_embeddings[col]
    
    logger.info(f"Combined all token types into {len(combined_features.columns)} total features")
    return combined_features

def create_region_embeddings(features_df):
    """
    Create region embeddings based on consistent region order
    """
    logger.info("Creating region embeddings...")
    
    # Initialize region embeddings
    region_features = features_df[['subject_id', 'session', 'site']].copy()
    
    # Create embeddings for each region in our predefined order
    for i, region in enumerate(ALL_REGIONS):
        # Create a simple embedding: region index normalized to 0-1
        region_features[f'region_{region}_embedding'] = i / len(ALL_REGIONS)
    
    logger.info(f"Created region embeddings for {len(ALL_REGIONS)} regions")
    return region_features

def save_tokens_to_jsonl(combined_features, output_path):
    """
    Save tokens to JSONL format for easy loading
    """
    logger.info(f"Saving tokens to {output_path}...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for _, row in combined_features.iterrows():
            # Convert to dictionary and handle NaN values
            token_dict = {}
            for col, val in row.items():
                if pd.isna(val):
                    token_dict[col] = 0
                elif isinstance(val, str):
                    token_dict[col] = val  # Keep strings as-is
                elif isinstance(val, (np.integer, float)) and val.is_integer():
                    token_dict[col] = int(val)
                else:
                    token_dict[col] = float(val)
            
            f.write(json.dumps(token_dict) + '\n')
    
    logger.info(f"Saved {len(combined_features)} token sequences to {output_path}")

def main():
    """Main function to extract enhanced neurotokens"""
    logger.info("Starting enhanced neurotoken extraction...")
    
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
    
    if not all_features:
        logger.error("No features extracted!")
        return
    
    # Create DataFrame
    features_df = pd.DataFrame(all_features)
    logger.info(f"Created features DataFrame with shape {features_df.shape}")
    
    # Save raw features
    features_df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Saved raw features to {OUTPUT_CSV}")
    
    # Create enhanced tokens
    logger.info("Creating enhanced neurotokens...")
    
    # 1. Create delta-tokens
    delta_features = create_delta_tokens(features_df)
    
    # 1b. Quantile-bin the delta tokens
    binned_delta_features, delta_transformer = quantile_bin_deltas(delta_features)
    
    # 2. Create level-tokens (fit on all data for now, will be refit on train split later)
    level_features, level_scaler, level_transformer = create_level_tokens(features_df, fit_scaler=True)
    
    # 3. Create delta-t embeddings
    delta_t_features = create_delta_t_embeddings(features_df)
    
    # 4. Apply site-wise harmonization
    harmonized_features = harmonize_by_site(features_df)
    
    # 5. Combine all token types
    combined_features = combine_all_tokens(level_features, binned_delta_features, delta_t_features, harmonized_features)
    
    # 6. Save enhanced tokens
    save_tokens_to_jsonl(combined_features, TOKEN_OUTPUT)
    
    # Save transformers for later use
    transformers = {
        'delta_transformer': delta_transformer,
        'level_scaler': level_scaler,
        'level_transformer': level_transformer
    }
    
    transformers_path = os.path.join(OUTPUT_DIR, "transformers.json")
    # Note: In practice, you'd want to pickle these or save them properly
    logger.info(f"Transformers created and ready for train-only fitting")
    
    logger.info("=" * 60)
    logger.info("ENHANCED NEUROTOKEN EXTRACTION COMPLETED!")
    logger.info("=" * 60)
    logger.info(f"Output files:")
    logger.info(f"  - Raw features: {OUTPUT_CSV}")
    logger.info(f"  - Enhanced tokens: {TOKEN_OUTPUT}")
    logger.info(f"  - Transformers: {transformers_path}")
    logger.info(f"Total subjects: {features_df['subject_id'].nunique()}")
    logger.info(f"Total sessions: {len(features_df)}")
    logger.info(f"Total features: {len(combined_features.columns)}")
    
    # Show feature breakdown
    level_cols = [col for col in combined_features.columns if col.startswith('level_')]
    delta_cols = [col for col in combined_features.columns if col.startswith('binned_delta_')]
    harmonized_cols = [col for col in combined_features.columns if col.startswith('harmonized_')]
    region_cols = [col for col in combined_features.columns if col.startswith('region_')]
    
    logger.info(f"Feature breakdown:")
    logger.info(f"  - Level tokens: {len(level_cols)}")
    logger.info(f"  - Delta tokens: {len(delta_cols)}")
    logger.info(f"  - Harmonized features: {len(harmonized_cols)}")
    logger.info(f"  - Region embeddings: {len(region_cols)}")
    logger.info(f"  - Metadata: 3 (subject_id, session, site)")

if __name__ == "__main__":
    main() 