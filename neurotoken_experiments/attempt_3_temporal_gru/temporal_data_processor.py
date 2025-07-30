#!/usr/bin/env python3
"""
Temporal Data Processor
Converts neurotoken sequences into temporally-aware format with session timing.
"""

import json
import pandas as pd
import numpy as np
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_oasis_demographics(excel_path):
    """
    Load OASIS demographics data to get session timing information
    
    Args:
        excel_path: Path to OASIS Longitudinal Demographics.xlsx
        
    Returns:
        DataFrame with subject session timing data
    """
    logger.info(f"Loading OASIS demographics from {excel_path}")
    
    try:
        df = pd.read_excel(excel_path)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Show available columns for debugging
        logger.info(f"Available columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        raise

def extract_session_timing(demographics_df):
    """
    Extract session timing information for each subject
    
    Args:
        demographics_df: OASIS demographics DataFrame
        
    Returns:
        Dictionary mapping subject_id to list of (session, delay) tuples
    """
    logger.info("Extracting session timing information")
    
    # Expected columns for timing
    timing_columns = ['Subject ID', 'Visit', 'MR Delay']
    
    # Check if required columns exist
    missing_columns = [col for col in timing_columns if col not in demographics_df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        logger.info(f"Available columns: {list(demographics_df.columns)}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Group by subject and extract timing
    subject_timing = defaultdict(list)
    
    for _, row in demographics_df.iterrows():
        subject_id = row['Subject ID']
        visit = row['Visit']
        delay = row['MR Delay']
        
        # Convert delay to float, handle NaN
        if pd.isna(delay):
            delay = 0.0
        else:
            delay = float(delay)
        
        subject_timing[subject_id].append((visit, delay))
    
    # Sort sessions by delay for each subject
    for subject_id in subject_timing:
        subject_timing[subject_id].sort(key=lambda x: x[1])
    
    logger.info(f"Extracted timing for {len(subject_timing)} subjects")
    
    # Show some examples
    for i, (subject_id, sessions) in enumerate(list(subject_timing.items())[:3]):
        logger.info(f"Subject {subject_id}: {len(sessions)} sessions")
        for visit, delay in sessions:
            logger.info(f"  Visit {visit}: {delay} days")
    
    return subject_timing

def load_token_sequences(jsonl_path):
    """
    Load existing token sequences
    
    Args:
        jsonl_path: Path to token_sequences.jsonl
        
    Returns:
        Dictionary mapping subject_id to token sequences
    """
    logger.info(f"Loading token sequences from {jsonl_path}")
    
    token_sequences = {}
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            subject_id = data['subject_id']
            token_sequence = data['token_sequence']
            token_sequences[subject_id] = token_sequence
    
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
    
    # Create binary label mapping: CN=0, MCI+AD=1
    binary_label_map = {"CN": 0, "MCI": 1, "AD": 1}
    
    subject_labels = {}
    for _, row in labels_df.iterrows():
        subject_id = row['subject_id']
        class_label = row['class']
        binary_label = binary_label_map[class_label]
        subject_labels[subject_id] = binary_label
    
    logger.info(f"Loaded labels for {len(subject_labels)} subjects")
    
    # Show label distribution
    label_counts = pd.Series(list(subject_labels.values())).value_counts().sort_index()
    logger.info(f"Label distribution: CN={label_counts.get(0, 0)}, Impaired={label_counts.get(1, 0)}")
    
    return subject_labels

def create_temporal_sequences(token_sequences, subject_timing, subject_labels, max_sessions=5):
    """
    Create temporally-aware sequences for each subject
    
    Args:
        token_sequences: Dict of subject_id -> token_sequence
        subject_timing: Dict of subject_id -> list of (visit, delay) tuples
        subject_labels: Dict of subject_id -> binary label
        max_sessions: Maximum number of sessions to include
        
    Returns:
        List of temporal sequence dictionaries
    """
    logger.info("Creating temporally-aware sequences")
    
    temporal_sequences = []
    
    for subject_id in token_sequences:
        if subject_id not in subject_timing or subject_id not in subject_labels:
            logger.warning(f"Missing timing or label for subject {subject_id}, skipping")
            continue
        
        # Get timing information for this subject
        timing_info = subject_timing[subject_id]
        
        # Get token sequence
        full_token_sequence = token_sequences[subject_id]
        
        # Split token sequence into sessions
        # Assuming tokens are ordered by session, we need to split them
        # For now, we'll create equal-length sessions
        tokens_per_session = 28
        num_sessions = len(full_token_sequence) // tokens_per_session
        
        if num_sessions == 0:
            logger.warning(f"Subject {subject_id} has insufficient tokens, skipping")
            continue
        
        # Create sessions
        sessions = []
        for i in range(min(num_sessions, max_sessions)):
            start_idx = i * tokens_per_session
            end_idx = start_idx + tokens_per_session
            session_tokens = full_token_sequence[start_idx:end_idx]
            
            # Get delay for this session (or use timing info if available)
            if i < len(timing_info):
                _, delay = timing_info[i]
            else:
                # If we don't have timing info for this session, estimate
                delay = i * 365  # Assume 1 year between sessions
            
            sessions.append({
                "tokens": session_tokens,
                "delay": delay
            })
        
        # Create temporal sequence
        temporal_sequence = {
            "subject_id": subject_id,
            "sessions": sessions,
            "label": subject_labels[subject_id]
        }
        
        temporal_sequences.append(temporal_sequence)
    
    logger.info(f"Created temporal sequences for {len(temporal_sequences)} subjects")
    
    # Show some examples
    for i, seq in enumerate(temporal_sequences[:2]):
        logger.info(f"Example {i+1}: Subject {seq['subject_id']}")
        logger.info(f"  Label: {seq['label']} ({'CN' if seq['label'] == 0 else 'Impaired'})")
        logger.info(f"  Sessions: {len(seq['sessions'])}")
        for j, session in enumerate(seq['sessions']):
            logger.info(f"    Session {j+1}: {len(session['tokens'])} tokens, delay={session['delay']} days")
    
    return temporal_sequences

def normalize_delays(temporal_sequences):
    """
    Normalize delays to 0-1 range for each subject
    
    Args:
        temporal_sequences: List of temporal sequence dictionaries
        
    Returns:
        List of temporal sequences with normalized delays
    """
    logger.info("Normalizing delays to 0-1 range")
    
    for seq in temporal_sequences:
        sessions = seq['sessions']
        if not sessions:
            continue
        
        # Find min and max delays for this subject
        delays = [session['delay'] for session in sessions]
        min_delay = min(delays)
        max_delay = max(delays)
        
        # Normalize delays
        for session in sessions:
            if max_delay > min_delay:
                session['delay'] = (session['delay'] - min_delay) / (max_delay - min_delay)
            else:
                session['delay'] = 0.0
    
    logger.info("Delay normalization completed")
    return temporal_sequences

def save_temporal_sequences(temporal_sequences, output_path):
    """
    Save temporal sequences to JSON file
    
    Args:
        temporal_sequences: List of temporal sequence dictionaries
        output_path: Path to save the JSON file
    """
    logger.info(f"Saving temporal sequences to {output_path}")
    
    with open(output_path, 'w') as f:
        for seq in temporal_sequences:
            json.dump(seq, f)
            f.write('\n')
    
    logger.info(f"Saved {len(temporal_sequences)} temporal sequences")

def main():
    """Main function to process temporal data"""
    # File paths
    token_sequences_file = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/token_sequences.jsonl"
    subject_labels_file = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/subject_labels.csv"
    demographics_file = "/Volumes/SEAGATE_NIKHIL/Oasis Longitudinal Demographics.xlsx"
    output_file = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/temporal_attempt/temporal_sequences.jsonl"
    
    try:
        # Step 1: Load demographics data
        demographics_df = load_oasis_demographics(demographics_file)
        
        # Step 2: Extract session timing
        subject_timing = extract_session_timing(demographics_df)
        
        # Step 3: Load token sequences
        token_sequences = load_token_sequences(token_sequences_file)
        
        # Step 4: Load subject labels
        subject_labels = load_subject_labels(subject_labels_file)
        
        # Step 5: Create temporal sequences
        temporal_sequences = create_temporal_sequences(
            token_sequences, subject_timing, subject_labels, max_sessions=5
        )
        
        # Step 6: Normalize delays
        temporal_sequences = normalize_delays(temporal_sequences)
        
        # Step 7: Save temporal sequences
        save_temporal_sequences(temporal_sequences, output_file)
        
        logger.info("=" * 60)
        logger.info("TEMPORAL DATA PROCESSING COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"Output file: {output_file}")
        logger.info(f"Total subjects: {len(temporal_sequences)}")
        
        # Show statistics
        label_counts = {}
        session_counts = []
        
        for seq in temporal_sequences:
            label = seq['label']
            label_counts[label] = label_counts.get(label, 0) + 1
            session_counts.append(len(seq['sessions']))
        
        logger.info(f"Label distribution: CN={label_counts.get(0, 0)}, Impaired={label_counts.get(1, 0)}")
        logger.info(f"Average sessions per subject: {np.mean(session_counts):.1f}")
        logger.info(f"Min sessions: {min(session_counts)}, Max sessions: {max(session_counts)}")
        
    except Exception as e:
        logger.error(f"Error in temporal data processing: {e}")
        raise

if __name__ == "__main__":
    main() 