#!/usr/bin/env python3
"""
Neurotoken Generator
Converts continuous brain features into discrete tokens using KMeans clustering.
This is the next step after feature extraction for Alzheimer's detection.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
INPUT_CSV = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/all_features_combined.csv"
OUTPUT_DIR = "/Volumes/SEAGATE_NIKHIL/neurotokens_project"

# Features to use for neurotokenization (26 features total)
FEATURES_TO_USE = [
    # Normalized subcortical volumes (6 features)
    "Left-Hippocampus_norm",
    "Right-Hippocampus_norm", 
    "Left-Amygdala_norm",
    "Right-Amygdala_norm",
    "Left-Lateral-Ventricle_norm",
    "Right-Lateral-Ventricle_norm",
    
    # Cortical thickness features (20 features) - key regions for Alzheimer's
    "lh_entorhinal_meanthickness",
    "rh_entorhinal_meanthickness",
    "lh_parahippocampal_meanthickness", 
    "rh_parahippocampal_meanthickness",
    "lh_precuneus_meanthickness",
    "rh_precuneus_meanthickness",
    "lh_inferiortemporal_meanthickness",
    "rh_inferiortemporal_meanthickness",
    "lh_middletemporal_meanthickness",
    "rh_middletemporal_meanthickness",
    "lh_superiortemporal_meanthickness",
    "rh_superiortemporal_meanthickness",
    "lh_fusiform_meanthickness",
    "rh_fusiform_meanthickness",
    "lh_insula_meanthickness",
    "rh_insula_meanthickness",
    "lh_posteriorcingulate_meanthickness",
    "rh_posteriorcingulate_meanthickness",
    "lh_rostralmiddlefrontal_meanthickness",
    "rh_rostralmiddlefrontal_meanthickness",
    "lh_supramarginal_meanthickness",
    "rh_supramarginal_meanthickness"
]

# KMeans parameters
N_CLUSTERS = 32
RANDOM_STATE = 42


def load_and_prepare_data():
    """Load the CSV data and prepare it for tokenization"""
    logger.info(f"Loading data from {INPUT_CSV}")
    
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
    
    # Load the data
    df = pd.read_csv(INPUT_CSV)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Check if all required features exist
    missing_features = [f for f in FEATURES_TO_USE if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in CSV: {missing_features}")
    
    # Select only the features we want to use
    feature_df = df[['subject_id', 'session'] + FEATURES_TO_USE].copy()
    
    # Drop rows with missing values
    initial_count = len(feature_df)
    feature_df = feature_df.dropna()
    final_count = len(feature_df)
    
    if final_count < initial_count:
        logger.warning(f"Dropped {initial_count - final_count} rows with missing values")
        logger.info(f"Remaining data: {final_count} records")
    
    return feature_df


def fit_kmeans_models(feature_df):
    """Fit KMeans models for each feature"""
    logger.info(f"Fitting KMeans models for {len(FEATURES_TO_USE)} features")
    
    tokenizers = {}
    feature_stats = {}
    
    for feature in FEATURES_TO_USE:
        logger.info(f"Fitting KMeans for {feature}")
        
        # Get all values for this feature across all subjects and sessions
        values = feature_df[feature].values.reshape(-1, 1)
        
        # Store statistics for later analysis
        feature_stats[feature] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'n_values': len(values)
        }
        
        # Fit KMeans model
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(values)
        
        tokenizers[feature] = kmeans
        
        # Log some info about the clustering
        cluster_centers = kmeans.cluster_centers_.flatten()
        logger.info(f"  {feature}: {len(values)} values → {N_CLUSTERS} clusters")
        logger.info(f"  Cluster centers range: {cluster_centers.min():.4f} to {cluster_centers.max():.4f}")
    
    return tokenizers, feature_stats


def create_token_sequences(feature_df, tokenizers):
    """Create token sequences for each subject"""
    logger.info("Creating token sequences for each subject")
    
    # Sort by subject_id and session to ensure consistent ordering
    feature_df = feature_df.sort_values(['subject_id', 'session']).reset_index(drop=True)
    
    token_sequences = []
    
    # Group by subject
    for subject_id, subject_data in feature_df.groupby('subject_id'):
        logger.info(f"Processing subject {subject_id} with {len(subject_data)} sessions")
        
        subject_tokens = []
        
        # Process each session in order
        for _, session_data in subject_data.iterrows():
            session_tokens = []
            
            # Convert each feature value to a token
            for feature in FEATURES_TO_USE:
                value = session_data[feature]
                kmeans = tokenizers[feature]
                
                # Predict cluster for this value
                cluster_idx = kmeans.predict([[value]])[0]
                session_tokens.append(int(cluster_idx))
            
            subject_tokens.extend(session_tokens)
        
        # Create the final token sequence for this subject
        token_sequences.append({
            "subject_id": subject_id,
            "token_sequence": subject_tokens
        })
    
    logger.info(f"Created token sequences for {len(token_sequences)} subjects")
    return token_sequences


def save_outputs(token_sequences, tokenizers, feature_stats):
    """Save all output files"""
    logger.info("Saving output files")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Save token sequences
    token_sequences_file = os.path.join(OUTPUT_DIR, "token_sequences.jsonl")
    with open(token_sequences_file, 'w') as f:
        for sequence in token_sequences:
            f.write(json.dumps(sequence) + '\n')
    logger.info(f"Saved token sequences to {token_sequences_file}")
    
    # 2. Save feature order
    feature_order_file = os.path.join(OUTPUT_DIR, "feature_order.json")
    with open(feature_order_file, 'w') as f:
        json.dump(FEATURES_TO_USE, f, indent=2)
    logger.info(f"Saved feature order to {feature_order_file}")
    
    # 3. Save KMeans models
    tokenizers_file = os.path.join(OUTPUT_DIR, "tokenizers.pkl")
    with open(tokenizers_file, 'wb') as f:
        pickle.dump(tokenizers, f)
    logger.info(f"Saved KMeans models to {tokenizers_file}")
    
    # 4. Save feature statistics for analysis
    stats_file = os.path.join(OUTPUT_DIR, "feature_statistics.json")
    with open(stats_file, 'w') as f:
        json.dump(feature_stats, f, indent=2)
    logger.info(f"Saved feature statistics to {stats_file}")
    
    return {
        'token_sequences_file': token_sequences_file,
        'feature_order_file': feature_order_file,
        'tokenizers_file': tokenizers_file,
        'stats_file': stats_file
    }


def analyze_results(token_sequences, feature_stats):
    """Analyze the generated token sequences"""
    logger.info("Analyzing results")
    
    # Calculate sequence lengths
    sequence_lengths = [len(seq['token_sequence']) for seq in token_sequences]
    
    # Calculate token distribution
    all_tokens = []
    for seq in token_sequences:
        all_tokens.extend(seq['token_sequence'])
    
    token_counts = pd.Series(all_tokens).value_counts().sort_index()
    
    # Print analysis
    logger.info(f"Token sequence analysis:")
    logger.info(f"  Number of subjects: {len(token_sequences)}")
    logger.info(f"  Average sequence length: {np.mean(sequence_lengths):.1f}")
    logger.info(f"  Min sequence length: {min(sequence_lengths)}")
    logger.info(f"  Max sequence length: {max(sequence_lengths)}")
    logger.info(f"  Total tokens generated: {len(all_tokens)}")
    logger.info(f"  Token range: {min(all_tokens)} to {max(all_tokens)}")
    logger.info(f"  Token distribution:")
    for token, count in token_counts.head(10).items():
        logger.info(f"    Token {token}: {count} occurrences")
    
    # Show some example sequences
    logger.info(f"Example token sequences:")
    for i, seq in enumerate(token_sequences[:3]):
        logger.info(f"  {seq['subject_id']}: {seq['token_sequence'][:10]}... (length: {len(seq['token_sequence'])})")
    
    return {
        'n_subjects': len(token_sequences),
        'avg_sequence_length': np.mean(sequence_lengths),
        'total_tokens': len(all_tokens),
        'token_distribution': token_counts.to_dict()
    }


def main():
    """Main function to run the neurotoken generation pipeline"""
    logger.info("Starting neurotoken generation pipeline")
    
    try:
        # Step 1: Load and prepare data
        feature_df = load_and_prepare_data()
        
        # Step 2: Fit KMeans models for each feature
        tokenizers, feature_stats = fit_kmeans_models(feature_df)
        
        # Step 3: Create token sequences
        token_sequences = create_token_sequences(feature_df, tokenizers)
        
        # Step 4: Save outputs
        output_files = save_outputs(token_sequences, tokenizers, feature_stats)
        
        # Step 5: Analyze results
        analysis = analyze_results(token_sequences, feature_stats)
        
        # Final summary
        logger.info("=" * 50)
        logger.info("NEUROTOKEN GENERATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        logger.info(f"Generated {analysis['n_subjects']} token sequences")
        logger.info(f"Average sequence length: {analysis['avg_sequence_length']:.1f} tokens")
        logger.info(f"Total tokens: {analysis['total_tokens']}")
        logger.info(f"Features used: {len(FEATURES_TO_USE)}")
        logger.info(f"KMeans clusters per feature: {N_CLUSTERS}")
        logger.info("=" * 50)
        logger.info("Output files:")
        for key, path in output_files.items():
            logger.info(f"  {key}: {path}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error in neurotoken generation: {e}")
        raise


if __name__ == "__main__":
    main() 