#!/usr/bin/env python3
"""
Create Subject Labels from OASIS Demographics
Process the OASIS longitudinal demographics Excel file to create subject labels
for classification training based on CDR (Clinical Dementia Rating) scores.
"""

import pandas as pd
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
INPUT_EXCEL = "/Volumes/SEAGATE_NIKHIL/Oasis Longitudinal Demographics.xlsx"
OUTPUT_CSV = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/subject_labels.csv"

def load_demographics_data():
    """Load the OASIS demographics Excel file"""
    logger.info(f"Loading demographics data from {INPUT_EXCEL}")
    
    if not os.path.exists(INPUT_EXCEL):
        raise FileNotFoundError(f"Demographics Excel file not found: {INPUT_EXCEL}")
    
    try:
        # Read the Excel file
        df = pd.read_excel(INPUT_EXCEL)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Show column names for debugging
        logger.info(f"Available columns: {list(df.columns)}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        raise

def clean_and_prepare_data(df):
    """Clean and prepare the demographics data"""
    logger.info("Cleaning and preparing data")
    
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    
    # Check for required columns
    required_columns = ['Subject ID', 'Visit', 'CDR']
    missing_columns = [col for col in required_columns if col not in df_clean.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        logger.info(f"Available columns: {list(df_clean.columns)}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Show initial data info
    logger.info(f"Initial data shape: {df_clean.shape}")
    logger.info(f"CDR value counts before cleaning:")
    logger.info(df_clean['CDR'].value_counts().sort_index())
    
    # Drop rows with missing CDR values
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['CDR'])
    final_count = len(df_clean)
    
    if final_count < initial_count:
        logger.warning(f"Dropped {initial_count - final_count} rows with missing CDR values")
    
    logger.info(f"Data shape after cleaning: {df_clean.shape}")
    logger.info(f"CDR value counts after cleaning:")
    logger.info(df_clean['CDR'].value_counts().sort_index())
    
    return df_clean

def create_subject_labels(df_clean):
    """Create subject labels based on maximum CDR score per subject"""
    logger.info("Creating subject labels based on maximum CDR scores")
    
    # Group by Subject ID and compute maximum CDR
    subject_max_cdr = df_clean.groupby('Subject ID')['CDR'].max().reset_index()
    subject_max_cdr.columns = ['subject_id', 'max_cdr']
    
    logger.info(f"Found {len(subject_max_cdr)} unique subjects")
    logger.info(f"Maximum CDR distribution:")
    logger.info(subject_max_cdr['max_cdr'].value_counts().sort_index())
    
    # Map CDR scores to class labels
    def map_cdr_to_class(cdr):
        if cdr == 0.0:
            return "CN"  # Cognitively Normal
        elif cdr == 0.5:
            return "MCI"  # Mild Cognitive Impairment
        elif cdr >= 1.0:
            return "AD"   # Alzheimer's Disease
        else:
            logger.warning(f"Unexpected CDR value: {cdr}")
            return "UNKNOWN"
    
    # Apply the mapping
    subject_max_cdr['class'] = subject_max_cdr['max_cdr'].apply(map_cdr_to_class)
    
    # Show class distribution
    logger.info(f"Class distribution:")
    logger.info(subject_max_cdr['class'].value_counts())
    
    # Create final output DataFrame
    output_df = subject_max_cdr[['subject_id', 'class']].copy()
    
    # Sort by subject_id for consistency
    output_df = output_df.sort_values('subject_id').reset_index(drop=True)
    
    return output_df

def validate_with_token_sequences(output_df):
    """Validate that our labels match the token sequences"""
    logger.info("Validating labels against token sequences")
    
    token_sequences_file = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/token_sequences.jsonl"
    
    if not os.path.exists(token_sequences_file):
        logger.warning("Token sequences file not found, skipping validation")
        return
    
    # Load token sequences to get subject IDs
    import json
    token_subjects = set()
    with open(token_sequences_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            token_subjects.add(data['subject_id'])
    
    label_subjects = set(output_df['subject_id'])
    
    # Find overlaps and differences
    common_subjects = token_subjects.intersection(label_subjects)
    only_in_tokens = token_subjects - label_subjects
    only_in_labels = label_subjects - token_subjects
    
    logger.info(f"Subjects in token sequences: {len(token_subjects)}")
    logger.info(f"Subjects in labels: {len(label_subjects)}")
    logger.info(f"Common subjects: {len(common_subjects)}")
    logger.info(f"Only in tokens: {len(only_in_tokens)}")
    logger.info(f"Only in labels: {len(only_in_labels)}")
    
    if only_in_tokens:
        logger.warning(f"Subjects in tokens but not in labels: {sorted(list(only_in_tokens))[:10]}...")
    
    if only_in_labels:
        logger.warning(f"Subjects in labels but not in tokens: {sorted(list(only_in_labels))[:10]}...")

def save_labels(output_df):
    """Save the subject labels to CSV"""
    logger.info(f"Saving subject labels to {OUTPUT_CSV}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    # Save to CSV
    output_df.to_csv(OUTPUT_CSV, index=False)
    
    logger.info(f"Successfully saved {len(output_df)} subject labels")
    
    # Show sample of the output
    logger.info("Sample of generated labels:")
    logger.info(output_df.head(10).to_string(index=False))
    
    return OUTPUT_CSV

def main():
    """Main function to create subject labels"""
    logger.info("Starting subject label creation process")
    
    try:
        # Step 1: Load demographics data
        df = load_demographics_data()
        
        # Step 2: Clean and prepare data
        df_clean = clean_and_prepare_data(df)
        
        # Step 3: Create subject labels
        output_df = create_subject_labels(df_clean)
        
        # Step 4: Validate against token sequences
        validate_with_token_sequences(output_df)
        
        # Step 5: Save labels
        output_file = save_labels(output_df)
        
        # Final summary
        logger.info("=" * 60)
        logger.info("SUBJECT LABEL CREATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Output file: {output_file}")
        logger.info(f"Total subjects: {len(output_df)}")
        logger.info(f"Class distribution:")
        class_counts = output_df['class'].value_counts()
        for class_name, count in class_counts.items():
            percentage = (count / len(output_df)) * 100
            logger.info(f"  {class_name}: {count} ({percentage:.1f}%)")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in subject label creation: {e}")
        raise

if __name__ == "__main__":
    main() 