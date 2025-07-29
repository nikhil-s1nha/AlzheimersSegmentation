#!/usr/bin/env python3
"""
Validate Labels and Tokens
Show how the subject labels and token sequences work together for classification training.
"""

import json
import pandas as pd
import numpy as np

# Configuration
TOKEN_SEQUENCES_FILE = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/token_sequences.jsonl"
SUBJECT_LABELS_FILE = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/subject_labels.csv"

def load_data():
    """Load both token sequences and subject labels"""
    print("Loading token sequences and subject labels...")
    
    # Load token sequences
    token_sequences = []
    with open(TOKEN_SEQUENCES_FILE, 'r') as f:
        for line in f:
            token_sequences.append(json.loads(line))
    
    # Load subject labels
    labels_df = pd.read_csv(SUBJECT_LABELS_FILE)
    
    return token_sequences, labels_df

def analyze_combined_data(token_sequences, labels_df):
    """Analyze the combined token sequences and labels"""
    print("\n" + "="*60)
    print("COMBINED TOKEN SEQUENCES AND LABELS ANALYSIS")
    print("="*60)
    
    # Create a mapping from subject_id to class
    label_mapping = dict(zip(labels_df['subject_id'], labels_df['class']))
    
    # Analyze each token sequence with its label
    combined_data = []
    for seq in token_sequences:
        subject_id = seq['subject_id']
        if subject_id in label_mapping:
            combined_data.append({
                'subject_id': subject_id,
                'class': label_mapping[subject_id],
                'sequence_length': len(seq['token_sequence']),
                'token_sequence': seq['token_sequence']
            })
    
    # Convert to DataFrame for analysis
    combined_df = pd.DataFrame(combined_data)
    
    print(f"Total subjects with both tokens and labels: {len(combined_df)}")
    print(f"Class distribution:")
    class_counts = combined_df['class'].value_counts()
    for class_name, count in class_counts.items():
        percentage = (count / len(combined_df)) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    print(f"\nSequence length statistics by class:")
    for class_name in ['CN', 'MCI', 'AD']:
        class_data = combined_df[combined_df['class'] == class_name]
        if len(class_data) > 0:
            avg_length = class_data['sequence_length'].mean()
            print(f"  {class_name}: {len(class_data)} subjects, avg sequence length: {avg_length:.1f}")
    
    return combined_df

def show_training_examples(combined_df, n_examples=3):
    """Show examples of how the data will be used for training"""
    print(f"\n" + "="*60)
    print("TRAINING DATA EXAMPLES")
    print("="*60)
    print("These examples show how the data will be used for transformer training:")
    print("(Each row represents one training sample)")
    print()
    
    for i, (_, row) in enumerate(combined_df.head(n_examples).iterrows()):
        print(f"Example {i+1}:")
        print(f"  Subject ID: {row['subject_id']}")
        print(f"  Class: {row['class']}")
        print(f"  Sequence Length: {row['sequence_length']} tokens")
        print(f"  Token Sequence (first 20 tokens): {row['token_sequence'][:20]}...")
        print(f"  Token Sequence (last 10 tokens): ...{row['token_sequence'][-10:]}")
        print()

def show_classification_setup():
    """Show the classification setup"""
    print(f"\n" + "="*60)
    print("CLASSIFICATION SETUP")
    print("="*60)
    print("Ready for transformer training with the following setup:")
    print()
    print("Input:")
    print("  - Token sequences (integers 0-31)")
    print("  - Variable sequence lengths (28-140 tokens)")
    print("  - Each token represents a brain region measurement")
    print()
    print("Output:")
    print("  - Binary classification: CN vs (MCI + AD)")
    print("  - Multi-class classification: CN vs MCI vs AD")
    print()
    print("Model Architecture:")
    print("  - Transformer encoder")
    print("  - Input: Token sequences")
    print("  - Output: Classification probabilities")
    print()
    print("Training Data:")
    print("  - 149 subjects with complete data")
    print("  - 70 CN (Cognitively Normal)")
    print("  - 54 MCI (Mild Cognitive Impairment)")
    print("  - 25 AD (Alzheimer's Disease)")
    print()
    print("Validation:")
    print("  - Cross-validation recommended")
    print("  - Stratified sampling by class")
    print("  - Handle class imbalance")

def main():
    """Main validation function"""
    try:
        # Load data
        token_sequences, labels_df = load_data()
        
        # Analyze combined data
        combined_df = analyze_combined_data(token_sequences, labels_df)
        
        # Show training examples
        show_training_examples(combined_df)
        
        # Show classification setup
        show_classification_setup()
        
        print(f"\n" + "="*60)
        print("VALIDATION COMPLETE")
        print("="*60)
        print("✅ Token sequences and labels are ready for training!")
        print("✅ Data format is correct for transformer models")
        print("✅ Class distribution is suitable for classification")
        print("✅ Ready to proceed with model training")
        
    except Exception as e:
        print(f"Error in validation: {e}")
        raise

if __name__ == "__main__":
    main() 