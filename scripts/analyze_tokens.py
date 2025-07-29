#!/usr/bin/env python3
"""
Analyze Neurotokens
Quick analysis script to examine the generated token sequences and their properties.
"""

import json
import pandas as pd
import numpy as np
from collections import Counter

# Configuration
TOKEN_SEQUENCES_FILE = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/token_sequences.jsonl"
FEATURE_ORDER_FILE = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/feature_order.json"
STATS_FILE = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/feature_statistics.json"

def load_data():
    """Load the generated token data"""
    print("Loading token sequences...")
    
    # Load token sequences
    sequences = []
    with open(TOKEN_SEQUENCES_FILE, 'r') as f:
        for line in f:
            sequences.append(json.loads(line))
    
    # Load feature order
    with open(FEATURE_ORDER_FILE, 'r') as f:
        feature_order = json.load(f)
    
    # Load feature statistics
    with open(STATS_FILE, 'r') as f:
        feature_stats = json.load(f)
    
    return sequences, feature_order, feature_stats

def analyze_sequences(sequences, feature_order):
    """Analyze the token sequences"""
    print("\n" + "="*60)
    print("NEUROTOKEN SEQUENCE ANALYSIS")
    print("="*60)
    
    # Basic statistics
    n_subjects = len(sequences)
    sequence_lengths = [len(seq['token_sequence']) for seq in sequences]
    n_features = len(feature_order)
    
    print(f"Number of subjects: {n_subjects}")
    print(f"Number of features per session: {n_features}")
    print(f"Average sequence length: {np.mean(sequence_lengths):.1f} tokens")
    print(f"Min sequence length: {min(sequence_lengths)} tokens")
    print(f"Max sequence length: {max(sequence_lengths)} tokens")
    print(f"Standard deviation: {np.std(sequence_lengths):.1f} tokens")
    
    # Sessions per subject
    sessions_per_subject = [len(seq['token_sequence']) // n_features for seq in sequences]
    print(f"\nSessions per subject:")
    print(f"  Average: {np.mean(sessions_per_subject):.1f}")
    print(f"  Min: {min(sessions_per_subject)}")
    print(f"  Max: {max(sessions_per_subject)}")
    
    # Token distribution
    all_tokens = []
    for seq in sequences:
        all_tokens.extend(seq['token_sequence'])
    
    token_counts = Counter(all_tokens)
    print(f"\nToken distribution (0-31):")
    print(f"  Total tokens: {len(all_tokens)}")
    print(f"  Unique tokens: {len(token_counts)}")
    print(f"  Most common token: {token_counts.most_common(1)[0]}")
    print(f"  Least common token: {min(token_counts.items(), key=lambda x: x[1])}")
    
    # Show distribution
    print(f"\nToken frequency distribution:")
    for token in range(32):
        count = token_counts.get(token, 0)
        percentage = (count / len(all_tokens)) * 100
        print(f"  Token {token:2d}: {count:4d} ({percentage:5.1f}%)")
    
    return {
        'n_subjects': n_subjects,
        'n_features': n_features,
        'sequence_lengths': sequence_lengths,
        'sessions_per_subject': sessions_per_subject,
        'token_counts': token_counts,
        'all_tokens': all_tokens
    }

def analyze_feature_tokens(sequences, feature_order):
    """Analyze tokens by feature position"""
    print(f"\n" + "="*60)
    print("FEATURE-SPECIFIC TOKEN ANALYSIS")
    print("="*60)
    
    n_features = len(feature_order)
    
    # Analyze each feature position
    for i, feature in enumerate(feature_order):
        feature_tokens = []
        for seq in sequences:
            # Extract tokens for this feature across all sessions
            for j in range(0, len(seq['token_sequence']), n_features):
                if j + i < len(seq['token_sequence']):
                    feature_tokens.append(seq['token_sequence'][j + i])
        
        token_counts = Counter(feature_tokens)
        most_common = token_counts.most_common(1)[0]
        least_common = min(token_counts.items(), key=lambda x: x[1])
        
        print(f"{feature[:30]:30s}: Most common token {most_common[0]} ({most_common[1]} times), "
              f"Least common token {least_common[0]} ({least_common[1]} times)")

def show_example_sequences(sequences, feature_order, n_examples=3):
    """Show detailed examples of token sequences"""
    print(f"\n" + "="*60)
    print("EXAMPLE TOKEN SEQUENCES")
    print("="*60)
    
    n_features = len(feature_order)
    
    for i, seq in enumerate(sequences[:n_examples]):
        subject_id = seq['subject_id']
        tokens = seq['token_sequence']
        n_sessions = len(tokens) // n_features
        
        print(f"\nSubject: {subject_id} ({n_sessions} sessions, {len(tokens)} tokens)")
        
        # Show tokens by session
        for session in range(n_sessions):
            start_idx = session * n_features
            end_idx = start_idx + n_features
            session_tokens = tokens[start_idx:end_idx]
            
            print(f"  Session {session + 1}: {session_tokens}")
            
            # Show feature names for first session
            if session == 0:
                print("           Features: ", end="")
                for j, feature in enumerate(feature_order):
                    if j < 5:  # Show first 5 features
                        print(f"{feature.split('_')[0]:8s}", end=" ")
                    elif j == 5:
                        print("...", end=" ")
                        break
                print()

def main():
    """Main analysis function"""
    try:
        # Load data
        sequences, feature_order, feature_stats = load_data()
        
        # Analyze sequences
        analysis = analyze_sequences(sequences, feature_order)
        
        # Analyze by feature
        analyze_feature_tokens(sequences, feature_order)
        
        # Show examples
        show_example_sequences(sequences, feature_order)
        
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("The neurotokens are ready for transformer training!")
        print("Each subject now has a discrete token sequence representing")
        print("their brain structure across multiple sessions.")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        raise

if __name__ == "__main__":
    main() 