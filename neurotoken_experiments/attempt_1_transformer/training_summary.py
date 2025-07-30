#!/usr/bin/env python3
"""
Training Summary and Analysis
Analyze the training results and provide insights for improvement.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_results():
    """Load training and evaluation results"""
    base_dir = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/models"
    
    # Load evaluation metrics
    with open(os.path.join(base_dir, "evaluation_metrics.json"), 'r') as f:
        eval_metrics = json.load(f)
    
    # Load training metrics (if available)
    training_metrics = {}
    if os.path.exists(os.path.join(base_dir, "metrics.json")):
        try:
            with open(os.path.join(base_dir, "metrics.json"), 'r') as f:
                training_metrics = json.load(f)
        except json.JSONDecodeError:
            print("Warning: metrics.json is incomplete, skipping training metrics")
            training_metrics = {}
    
    return eval_metrics, training_metrics

def analyze_results(eval_metrics, training_metrics):
    """Analyze the results and provide insights"""
    print("=" * 80)
    print("NEUROTOKEN TRANSFORMER TRAINING SUMMARY")
    print("=" * 80)
    
    # Dataset Information
    print("\n📊 DATASET INFORMATION:")
    print(f"  • Total subjects: 149")
    print(f"  • Class distribution: CN=69, MCI=54, AD=26")
    print(f"  • Train/Val/Test split: 95/24/30 subjects")
    print(f"  • Sequence length range: 28-140 tokens")
    print(f"  • Vocabulary size: 32 tokens")
    
    # Model Architecture
    print("\n🏗️  MODEL ARCHITECTURE:")
    print(f"  • Transformer encoder with 2 layers")
    print(f"  • 4 attention heads")
    print(f"  • 64-dimensional embeddings")
    print(f"  • 256-dimensional feedforward")
    print(f"  • 116,611 total parameters")
    
    # Training Results
    print("\n🎯 TRAINING RESULTS:")
    if training_metrics:
        best_metrics = training_metrics.get('best_metrics', {})
        print(f"  • Best validation accuracy: {best_metrics.get('best_val_acc', 'N/A'):.4f}")
        print(f"  • Best validation loss: {best_metrics.get('best_val_loss', 'N/A'):.4f}")
        print(f"  • Final training accuracy: {best_metrics.get('final_train_acc', 'N/A'):.4f}")
        print(f"  • Final validation accuracy: {best_metrics.get('final_val_acc', 'N/A'):.4f}")
    
    # Test Results
    print("\n🧪 TEST SET PERFORMANCE:")
    print(f"  • Overall accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"  • Weighted precision: {eval_metrics['precision']:.4f}")
    print(f"  • Weighted recall: {eval_metrics['recall']:.4f}")
    print(f"  • Weighted F1-score: {eval_metrics['f1']:.4f}")
    
    # Per-class performance
    class_names = ['CN', 'MCI', 'AD']
    print("\n📈 PER-CLASS PERFORMANCE:")
    for i, class_name in enumerate(class_names):
        precision = eval_metrics['precision_per_class'][i]
        recall = eval_metrics['recall_per_class'][i]
        f1 = eval_metrics['f1_per_class'][i]
        support = eval_metrics['support_per_class'][i]
        print(f"  • {class_name}:")
        print(f"    - Precision: {precision:.4f}")
        print(f"    - Recall: {recall:.4f}")
        print(f"    - F1-score: {f1:.4f}")
        print(f"    - Support: {support} samples")
    
    # Confusion Matrix Analysis
    print("\n🔍 CONFUSION MATRIX ANALYSIS:")
    cm = eval_metrics['confusion_matrix']
    print("  Confusion Matrix:")
    print("              Predicted")
    print("  Actual    CN  MCI  AD")
    for i, class_name in enumerate(class_names):
        row = cm[i]
        print(f"  {class_name:8} {row[0]:3} {row[1]:3} {row[2]:3}")
    
    # Issues Identified
    print("\n⚠️  ISSUES IDENTIFIED:")
    print("  1. Model predicts only CN class (class 0) for all samples")
    print("  2. Zero precision and recall for MCI and AD classes")
    print("  3. Model is not learning to distinguish between classes")
    print("  4. Possible causes: class imbalance, small dataset, learning rate issues")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS FOR IMPROVEMENT:")
    print("  1. DATA AUGMENTATION:")
    print("     • Use data augmentation techniques")
    print("     • Collect more samples for underrepresented classes")
    print("     • Consider synthetic data generation")
    
    print("  2. CLASS IMBALANCE HANDLING:")
    print("     • Use weighted loss function")
    print("     • Implement focal loss")
    print("     • Use class-balanced sampling")
    print("     • Consider SMOTE or similar techniques")
    
    print("  3. MODEL ARCHITECTURE:")
    print("     • Increase model capacity (more layers, larger embeddings)")
    print("     • Add regularization (dropout, weight decay)")
    print("     • Try different attention mechanisms")
    print("     • Consider pre-training on larger datasets")
    
    print("  4. TRAINING STRATEGY:")
    print("     • Use different learning rate schedules")
    print("     • Implement gradient clipping")
    print("     • Try different optimizers (Adam, SGD with momentum)")
    print("     • Use cross-validation instead of single train/val split")
    
    print("  5. FEATURE ENGINEERING:")
    print("     • Add more brain regions/features")
    print("     • Include demographic information")
    print("     • Add longitudinal information")
    print("     • Consider feature selection techniques")
    
    print("  6. EVALUATION:")
    print("     • Use k-fold cross-validation")
    print("     • Implement stratified sampling")
    print("     • Add confidence intervals")
    print("     • Use multiple random seeds for robustness")
    
    # Next Steps
    print("\n🚀 NEXT STEPS:")
    print("  1. Implement weighted loss function")
    print("  2. Try data augmentation techniques")
    print("  3. Experiment with different model architectures")
    print("  4. Collect more data if possible")
    print("  5. Use ensemble methods")
    print("  6. Consider transfer learning from pre-trained models")
    
    print("\n" + "=" * 80)
    print("SUMMARY COMPLETE")
    print("=" * 80)

def create_improvement_plan():
    """Create a detailed improvement plan"""
    print("\n📋 DETAILED IMPROVEMENT PLAN:")
    
    print("\n1. IMMEDIATE FIXES (High Priority):")
    print("   • Implement weighted CrossEntropyLoss")
    print("   • Add class weights: CN=1.0, MCI=1.5, AD=2.0")
    print("   • Reduce learning rate to 5e-5")
    print("   • Increase batch size if memory allows")
    
    print("\n2. DATA IMPROVEMENTS (Medium Priority):")
    print("   • Add more FreeSurfer features")
    print("   • Include demographic variables")
    print("   • Add longitudinal progression information")
    print("   • Implement feature normalization")
    
    print("\n3. MODEL IMPROVEMENTS (Medium Priority):")
    print("   • Increase embedding dimension to 128")
    print("   • Add more transformer layers (4-6)")
    print("   • Implement layer normalization")
    print("   • Add residual connections")
    
    print("\n4. TRAINING IMPROVEMENTS (Medium Priority):")
    print("   • Use 5-fold cross-validation")
    print("   • Implement early stopping with patience=10")
    print("   • Add learning rate warmup")
    print("   • Use gradient clipping")
    
    print("\n5. EVALUATION IMPROVEMENTS (Low Priority):")
    print("   • Add confidence intervals")
    print("   • Implement statistical significance testing")
    print("   • Add model interpretability analysis")
    print("   • Create attention visualization tools")

def main():
    """Main function"""
    try:
        # Load results
        eval_metrics, training_metrics = load_results()
        
        # Analyze results
        analyze_results(eval_metrics, training_metrics)
        
        # Create improvement plan
        create_improvement_plan()
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        raise

if __name__ == "__main__":
    main() 