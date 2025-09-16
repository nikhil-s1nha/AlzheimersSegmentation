#!/usr/bin/env python3
"""
Comprehensive Model Evaluation on Entire Dataset
Tests the trained model on all available data and provides detailed results
"""

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging

# Import our modules
from enhanced_dataset_discrete import EnhancedNeuroTokenDatasetDiscrete
from enhanced_model import create_enhanced_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    def __init__(self, model_path, token_file, labels_file, transformers_path):
        self.model_path = model_path
        self.token_file = token_file
        self.labels_file = labels_file
        self.transformers_path = transformers_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        
    def load_data(self):
        """Load all available data"""
        logger.info("Loading token data...")
        
        # Load tokens
        with open(self.token_file, 'r') as f:
            tokens_data = [json.loads(line) for line in f]
        
        logger.info(f"Loaded {len(tokens_data)} token records")
        
        # Load labels
        labels_df = pd.read_csv(self.labels_file)
        logger.info(f"Loaded {len(labels_df)} subject labels")
        
        # Create subject labels mapping
        subject_labels = {}
        for _, row in labels_df.iterrows():
            subject_labels[row['subject_id']] = row['label']
        
        # Group tokens by subject
        subject_tokens = {}
        for token_record in tokens_data:
            subject_id = token_record['subject_id']
            if subject_id not in subject_tokens:
                subject_tokens[subject_id] = []
            subject_tokens[subject_id].append(token_record)
        
        logger.info(f"Found {len(subject_tokens)} subjects with token data")
        
        # Create sequences
        sequences = []
        for subject_id, sessions in subject_tokens.items():
            if subject_id in subject_labels:
                sequences.append({
                    'subject_id': subject_id,
                    'label': subject_labels[subject_id],
                    'sessions': sessions
                })
        
        logger.info(f"Created {len(sequences)} complete sequences")
        
        return sequences
    
    def create_dataset(self, sequences):
        """Create dataset for evaluation"""
        logger.info("Creating evaluation dataset...")
        
        # Use the same transformers that were fitted during training
        dataset = EnhancedNeuroTokenDatasetDiscrete(
            sequences, 
            max_sessions=5, 
            max_tokens=28, 
            fit_transformers=False,  # Don't fit new transformers
            transformers_path=self.transformers_path
        )
        
        return dataset
    
    def load_model(self, dataset):
        """Load the trained model"""
        logger.info("Loading trained model...")
        
        # Get sample batch to determine dimensions
        sample_batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
        
        level_token_dim = int(sample_batch['level_tokens'].max().item()) + 1
        delta_token_dim = int(sample_batch['delta_tokens'].max().item()) + 1
        harmonized_dim = int(sample_batch['harmonized_features'].size(-1))
        region_embedding_dim = int(sample_batch['region_embeddings'].size(-1))
        delta_t_bucket_dim = int(sample_batch['delta_t_buckets'].size(-1))
        
        logger.info(f"Model dimensions:")
        logger.info(f"  Level tokens: {level_token_dim}")
        logger.info(f"  Delta tokens: {delta_token_dim}")
        logger.info(f"  Harmonized features: {harmonized_dim}")
        logger.info(f"  Region embeddings: {region_embedding_dim}")
        logger.info(f"  Delta-t buckets: {delta_t_bucket_dim}")
        
        # Create model
        model = create_enhanced_model(
            model_type='gru',
            max_sessions=5,
            max_tokens=28,
            level_token_dim=level_token_dim,
            delta_token_dim=delta_token_dim,
            harmonized_dim=harmonized_dim,
            region_embedding_dim=region_embedding_dim,
            delta_t_bucket_dim=delta_t_bucket_dim,
            hidden_dim=128,
            num_layers=2,
            num_heads=8,
            dropout=0.3,
            num_classes=2
        )
        
        # Load trained weights
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        logger.info("Model loaded successfully")
        return model
    
    def evaluate_model(self, model, dataset):
        """Comprehensive model evaluation"""
        logger.info("Starting comprehensive evaluation...")
        
        # Create data loader
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                logits, _ = model(batch)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    logger.info(f"Processed batch {batch_idx}/{len(dataloader)}")
        
        return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)
    
    def calculate_metrics(self, predictions, labels, probabilities):
        """Calculate comprehensive metrics"""
        logger.info("Calculating metrics...")
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted')
        recall = recall_score(labels, predictions, average='weighted')
        f1 = f1_score(labels, predictions, average='weighted')
        
        # AUC (binary classification)
        if len(np.unique(labels)) == 2:
            auc = roc_auc_score(labels, probabilities[:, 1])
        else:
            auc = roc_auc_score(labels, probabilities, multi_class='ovr', average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Classification report
        class_names = ['Normal', 'Impaired'] if len(np.unique(labels)) == 2 else ['CN', 'MCI', 'AD']
        report = classification_report(labels, predictions, target_names=class_names, output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': predictions,
            'labels': labels,
            'probabilities': probabilities
        }
        
        return metrics
    
    def create_visualizations(self, metrics):
        """Create comprehensive visualizations"""
        logger.info("Creating visualizations...")
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = metrics['confusion_matrix']
        class_names = ['Normal', 'Impaired'] if cm.shape[0] == 2 else ['CN', 'MCI', 'AD']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Entire Dataset', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=14, fontweight='bold')
        plt.ylabel('True Class', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('entire_dataset_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Performance Metrics Bar Chart
        plt.figure(figsize=(10, 6))
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        metric_values = [metrics['accuracy'], metrics['precision'], 
                        metrics['recall'], metrics['f1_score'], metrics['auc_roc']]
        
        bars = plt.bar(metric_names, metric_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6F42C1'], alpha=0.8)
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.ylabel('Performance Score', fontsize=14, fontweight='bold')
        plt.title('Model Performance on Entire Dataset', fontsize=16, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig('entire_dataset_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Prediction Distribution
        plt.figure(figsize=(12, 5))
        
        # True labels distribution
        plt.subplot(1, 2, 1)
        label_counts = Counter(metrics['labels'])
        plt.bar(class_names, [label_counts[i] for i in range(len(class_names))], 
                color=['#28A745', '#DC3545'], alpha=0.7)
        plt.title('True Label Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('Count', fontsize=12, fontweight='bold')
        
        # Predicted labels distribution
        plt.subplot(1, 2, 2)
        pred_counts = Counter(metrics['predictions'])
        plt.bar(class_names, [pred_counts[i] for i in range(len(class_names))], 
                color=['#28A745', '#DC3545'], alpha=0.7)
        plt.title('Predicted Label Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('Count', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('entire_dataset_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Probability Distribution
        plt.figure(figsize=(10, 6))
        probabilities = metrics['probabilities']
        
        # Plot probability distributions for each class
        for i in range(len(class_names)):
            class_probs = probabilities[metrics['labels'] == i, 1]  # Probability of class 1
            plt.hist(class_probs, bins=20, alpha=0.6, label=f'True {class_names[i]}', density=True)
        
        plt.xlabel('Predicted Probability of Impaired Class', fontsize=12, fontweight='bold')
        plt.ylabel('Density', fontsize=12, fontweight='bold')
        plt.title('Probability Distribution by True Class', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('entire_dataset_probabilities.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, metrics):
        """Generate comprehensive text report"""
        logger.info("Generating comprehensive report...")
        
        report = f"""
COMPREHENSIVE MODEL EVALUATION REPORT
=====================================

Dataset Information:
- Total samples evaluated: {len(metrics['labels'])}
- Classes: {len(np.unique(metrics['labels']))}
- Class distribution: {dict(Counter(metrics['labels']))}

Performance Metrics:
- Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
- Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)
- Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)
- F1-Score: {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)
- AUC-ROC: {metrics['auc_roc']:.4f} ({metrics['auc_roc']*100:.2f}%)

Confusion Matrix:
{metrics['confusion_matrix']}

Per-Class Performance:
"""
        
        # Add per-class metrics
        for class_name, class_metrics in metrics['classification_report'].items():
            if isinstance(class_metrics, dict) and 'precision' in class_metrics:
                report += f"\n{class_name}:"
                report += f"  Precision: {class_metrics['precision']:.4f}"
                report += f"  Recall: {class_metrics['recall']:.4f}"
                report += f"  F1-Score: {class_metrics['f1-score']:.4f}"
                report += f"  Support: {class_metrics['support']}"
        
        report += f"""

Model Analysis:
- The model achieves {metrics['accuracy']*100:.2f}% accuracy on the entire dataset
- {'Good' if metrics['auc_roc'] > 0.8 else 'Moderate' if metrics['auc_roc'] > 0.7 else 'Poor'} discrimination ability (AUC = {metrics['auc_roc']:.3f})
- {'Balanced' if abs(metrics['precision'] - metrics['recall']) < 0.1 else 'Imbalanced'} precision-recall tradeoff
- {'Strong' if metrics['f1_score'] > 0.8 else 'Moderate' if metrics['f1_score'] > 0.7 else 'Weak'} overall performance

Recommendations:
"""
        
        if metrics['accuracy'] < 0.8:
            report += "- Consider collecting more training data\n"
        if metrics['auc_roc'] < 0.8:
            report += "- Model may benefit from feature engineering\n"
        if abs(metrics['precision'] - metrics['recall']) > 0.1:
            report += "- Address class imbalance in training\n"
        
        report += "- Validate on independent test set\n"
        report += "- Consider ensemble methods for improved robustness\n"
        
        return report
    
    def save_results(self, metrics, report):
        """Save all results to files"""
        logger.info("Saving results...")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
            'Value': [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                     metrics['f1_score'], metrics['auc_roc']],
            'Percentage': [metrics['accuracy']*100, metrics['precision']*100, 
                          metrics['recall']*100, metrics['f1_score']*100, metrics['auc_roc']*100]
        })
        metrics_df.to_csv('entire_dataset_metrics.csv', index=False)
        
        # Save detailed results
        results_df = pd.DataFrame({
            'True_Label': metrics['labels'],
            'Predicted_Label': metrics['predictions'],
            'Probability_Normal': metrics['probabilities'][:, 0],
            'Probability_Impaired': metrics['probabilities'][:, 1]
        })
        results_df.to_csv('entire_dataset_predictions.csv', index=False)
        
        # Save report
        with open('entire_dataset_report.txt', 'w') as f:
            f.write(report)
        
        logger.info("Results saved to:")
        logger.info("- entire_dataset_metrics.csv")
        logger.info("- entire_dataset_predictions.csv") 
        logger.info("- entire_dataset_report.txt")
        logger.info("- entire_dataset_confusion_matrix.png")
        logger.info("- entire_dataset_performance.png")
        logger.info("- entire_dataset_distributions.png")
        logger.info("- entire_dataset_probabilities.png")
    
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        logger.info("Starting comprehensive evaluation on entire dataset...")
        
        # Load data
        sequences = self.load_data()
        
        # Create dataset
        dataset = self.create_dataset(sequences)
        
        # Load model
        model = self.load_model(dataset)
        
        # Evaluate model
        predictions, labels, probabilities = self.evaluate_model(model, dataset)
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, labels, probabilities)
        
        # Create visualizations
        self.create_visualizations(metrics)
        
        # Generate report
        report = self.generate_report(metrics)
        
        # Save results
        self.save_results(metrics, report)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total samples: {len(labels)}")
        print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"F1-Score: {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f} ({metrics['auc_roc']*100:.2f}%)")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("\nDetailed report saved to: entire_dataset_report.txt")
        print("="*60)
        
        return metrics

def main():
    """Main evaluation function"""
    # File paths
    model_path = 'models/best_model_discrete.pt'
    token_file = 'enhanced_tokens.json'
    labels_file = 'subject_labels.csv'
    transformers_path = 'models/transformers_discrete.pkl'
    
    # Check if files exist
    for file_path in [model_path, token_file, labels_file, transformers_path]:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(model_path, token_file, labels_file, transformers_path)
    
    # Run evaluation
    metrics = evaluator.run_evaluation()
    
    return metrics

if __name__ == "__main__":
    main()
