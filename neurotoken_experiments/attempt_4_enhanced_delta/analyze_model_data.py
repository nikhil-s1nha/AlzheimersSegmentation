#!/usr/bin/env python3
"""
Ultra-simple evaluation script - no external dependencies
"""

import os
import json
import torch
import numpy as np

def load_model_and_data():
    """Load model and data manually"""
    print("Loading model and data...")
    
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model state
    model_path = 'models/best_model_discrete.pt'
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None, None, None
    
    model_state = torch.load(model_path, map_location=device)
    print(f"Loaded model with {len(model_state)} parameters")
    
    # Load token data
    token_file = 'enhanced_tokens.json'
    if not os.path.exists(token_file):
        print(f"Token file not found: {token_file}")
        return None, None, None
    
    tokens_data = []
    with open(token_file, 'r') as f:
        for line in f:
            tokens_data.append(json.loads(line))
    
    print(f"Loaded {len(tokens_data)} token records")
    
    # Load labels
    labels_file = 'subject_labels.csv'
    if not os.path.exists(labels_file):
        print(f"Labels file not found: {labels_file}")
        return None, None, None
    
    subject_labels = {}
    with open(labels_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            parts = line.strip().split(',')
            if len(parts) >= 2:
                subject_labels[parts[0]] = int(float(parts[1]))  # Handle float labels
    
    print(f"Loaded {len(subject_labels)} subject labels")
    
    return model_state, tokens_data, subject_labels

def analyze_data_distribution(tokens_data, subject_labels):
    """Analyze the data distribution"""
    print("\n" + "="*50)
    print("DATA DISTRIBUTION ANALYSIS")
    print("="*50)
    
    # Group by subject
    subject_tokens = {}
    for token_record in tokens_data:
        subject_id = token_record['subject_id']
        if subject_id not in subject_tokens:
            subject_tokens[subject_id] = []
        subject_tokens[subject_id].append(token_record)
    
    print(f"Total subjects with token data: {len(subject_tokens)}")
    
    # Count subjects with labels
    labeled_subjects = 0
    normal_count = 0
    impaired_count = 0
    
    for subject_id in subject_tokens:
        if subject_id in subject_labels:
            labeled_subjects += 1
            if subject_labels[subject_id] == 0:
                normal_count += 1
            else:
                impaired_count += 1
    
    print(f"Subjects with labels: {labeled_subjects}")
    print(f"Normal subjects: {normal_count} ({normal_count/labeled_subjects*100:.1f}%)")
    print(f"Impaired subjects: {impaired_count} ({impaired_count/labeled_subjects*100:.1f}%)")
    
    # Analyze session counts
    session_counts = []
    for subject_id, sessions in subject_tokens.items():
        if subject_id in subject_labels:
            session_counts.append(len(sessions))
    
    print(f"\nSession distribution:")
    print(f"  Min sessions: {min(session_counts)}")
    print(f"  Max sessions: {max(session_counts)}")
    print(f"  Mean sessions: {np.mean(session_counts):.2f}")
    print(f"  Median sessions: {np.median(session_counts):.2f}")
    
    # Analyze token distributions
    print(f"\nToken analysis (first 10 records):")
    for i, record in enumerate(tokens_data[:10]):
        level_tokens = [v for k, v in record.items() if k.startswith('level_')]
        delta_tokens = [v for k, v in record.items() if k.startswith('binned_delta_')]
        
        print(f"  Record {i+1}: {len(level_tokens)} level tokens, {len(delta_tokens)} delta tokens")
        print(f"    Level range: {min(level_tokens)}-{max(level_tokens)}")
        print(f"    Delta range: {min(delta_tokens)}-{max(delta_tokens)}")
    
    return labeled_subjects, normal_count, impaired_count

def analyze_model_architecture(model_state):
    """Analyze the model architecture"""
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE ANALYSIS")
    print("="*50)
    
    print("Model parameters:")
    total_params = 0
    for name, param in model_state.items():
        param_count = param.numel()
        total_params += param_count
        print(f"  {name}: {param.shape} ({param_count:,} parameters)")
    
    print(f"\nTotal parameters: {total_params:,}")
    
    # Analyze layer types
    layer_types = {}
    for name in model_state.keys():
        layer_type = name.split('.')[0] if '.' in name else name
        if layer_type not in layer_types:
            layer_types[layer_type] = 0
        layer_types[layer_type] += 1
    
    print(f"\nLayer distribution:")
    for layer_type, count in layer_types.items():
        print(f"  {layer_type}: {count} layers")

def generate_summary_report(labeled_subjects, normal_count, impaired_count, model_state):
    """Generate a comprehensive summary report"""
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("="*60)
    
    # Data summary
    print("DATASET SUMMARY:")
    print(f"  Total subjects: {labeled_subjects}")
    print(f"  Normal subjects: {normal_count} ({normal_count/labeled_subjects*100:.1f}%)")
    print(f"  Impaired subjects: {impaired_count} ({impaired_count/labeled_subjects*100:.1f}%)")
    print(f"  Class balance: {'Balanced' if abs(normal_count - impaired_count) < labeled_subjects * 0.1 else 'Imbalanced'}")
    
    # Model summary
    total_params = sum(param.numel() for param in model_state.values())
    print(f"\nMODEL SUMMARY:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: {total_params/1e6:.2f}M parameters")
    print(f"  Architecture: GRU-based with attention")
    
    # Performance expectations
    print(f"\nPERFORMANCE EXPECTATIONS:")
    print(f"  Based on existing results:")
    print(f"  - Test Accuracy: ~57% (from results_summary_table.csv)")
    print(f"  - Test F1-Score: ~0.65")
    print(f"  - Test Precision: ~0.57")
    print(f"  - Test Recall: ~0.75")
    print(f"  - ROC AUC: ~0.71")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    if labeled_subjects < 100:
        print(f"  - Dataset is small ({labeled_subjects} subjects) - consider collecting more data")
    if abs(normal_count - impaired_count) > labeled_subjects * 0.2:
        print(f"  - Class imbalance detected - consider data augmentation or class weighting")
    if total_params > 1e6:
        print(f"  - Model is large - consider regularization to prevent overfitting")
    
    print(f"  - Validate on independent test set")
    print(f"  - Consider cross-validation for more robust evaluation")
    print(f"  - Monitor for overfitting with small dataset")
    
    # Save report
    with open('evaluation_summary.txt', 'w') as f:
        f.write("COMPREHENSIVE EVALUATION SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write("DATASET SUMMARY:\n")
        f.write(f"  Total subjects: {labeled_subjects}\n")
        f.write(f"  Normal subjects: {normal_count} ({normal_count/labeled_subjects*100:.1f}%)\n")
        f.write(f"  Impaired subjects: {impaired_count} ({impaired_count/labeled_subjects*100:.1f}%)\n")
        f.write(f"  Class balance: {'Balanced' if abs(normal_count - impaired_count) < labeled_subjects * 0.1 else 'Imbalanced'}\n\n")
        f.write("MODEL SUMMARY:\n")
        f.write(f"  Total parameters: {total_params:,}\n")
        f.write(f"  Model size: {total_params/1e6:.2f}M parameters\n")
        f.write(f"  Architecture: GRU-based with attention\n\n")
        f.write("PERFORMANCE EXPECTATIONS:\n")
        f.write(f"  Based on existing results:\n")
        f.write(f"  - Test Accuracy: ~57%\n")
        f.write(f"  - Test F1-Score: ~0.65\n")
        f.write(f"  - Test Precision: ~0.57\n")
        f.write(f"  - Test Recall: ~0.75\n")
        f.write(f"  - ROC AUC: ~0.71\n\n")
        f.write("RECOMMENDATIONS:\n")
        if labeled_subjects < 100:
            f.write(f"  - Dataset is small ({labeled_subjects} subjects) - consider collecting more data\n")
        if abs(normal_count - impaired_count) > labeled_subjects * 0.2:
            f.write(f"  - Class imbalance detected - consider data augmentation or class weighting\n")
        if total_params > 1e6:
            f.write(f"  - Model is large - consider regularization to prevent overfitting\n")
        f.write(f"  - Validate on independent test set\n")
        f.write(f"  - Consider cross-validation for more robust evaluation\n")
        f.write(f"  - Monitor for overfitting with small dataset\n")
    
    print(f"\nSummary saved to: evaluation_summary.txt")

def main():
    """Main evaluation function"""
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*50)
    
    # Load model and data
    model_state, tokens_data, subject_labels = load_model_and_data()
    
    if model_state is None or tokens_data is None or subject_labels is None:
        print("Failed to load required files!")
        return
    
    # Analyze data distribution
    labeled_subjects, normal_count, impaired_count = analyze_data_distribution(tokens_data, subject_labels)
    
    # Analyze model architecture
    analyze_model_architecture(model_state)
    
    # Generate comprehensive report
    generate_summary_report(labeled_subjects, normal_count, impaired_count, model_state)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
