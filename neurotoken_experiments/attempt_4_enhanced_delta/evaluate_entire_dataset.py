#!/usr/bin/env python3
"""
Comprehensive Model Evaluation on Entire Dataset
Tests the trained model on all available data and provides detailed results
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
        
        # Load labels (simple CSV parsing)
        subject_labels = {}
        with open(self.labels_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    subject_labels[parts[0]] = int(parts[1])
        
        logger.info(f"Loaded {len(subject_labels)} subject labels")
        
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
        accuracy = np.mean(predictions == labels)
        
        # Precision, Recall, F1 for each class
        unique_labels = np.unique(labels)
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for label in unique_labels:
            tp = np.sum((predictions == label) & (labels == label))
            fp = np.sum((predictions == label) & (labels != label))
            fn = np.sum((predictions != label) & (labels == label))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        
        # Weighted averages
        class_counts = [np.sum(labels == label) for label in unique_labels]
        total_samples = len(labels)
        
        precision_weighted = sum(p * c for p, c in zip(precision_scores, class_counts)) / total_samples
        recall_weighted = sum(r * c for r, c in zip(recall_scores, class_counts)) / total_samples
        f1_weighted = sum(f * c for f, c in zip(f1_scores, class_counts)) / total_samples
        
        # AUC (binary classification)
        if len(unique_labels) == 2:
            # Use probability of positive class
            pos_probs = probabilities[:, 1]
            auc = self.calculate_auc(labels, pos_probs)
        else:
            # Multi-class AUC (one-vs-rest)
            auc_scores = []
            for i, label in enumerate(unique_labels):
                binary_labels = (labels == label).astype(int)
                binary_probs = probabilities[:, i]
                auc_scores.append(self.calculate_auc(binary_labels, binary_probs))
            auc = np.mean(auc_scores)
        
        # Confusion matrix
        cm = self.calculate_confusion_matrix(labels, predictions, unique_labels)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision_weighted,
            'recall': recall_weighted,
            'f1_score': f1_weighted,
            'auc_roc': auc,
            'confusion_matrix': cm,
            'class_precision': precision_scores,
            'class_recall': recall_scores,
            'class_f1': f1_scores,
            'unique_labels': unique_labels,
            'class_counts': class_counts,
            'predictions': predictions,
            'labels': labels,
            'probabilities': probabilities
        }
        
        return metrics
    
    def calculate_auc(self, labels, probabilities):
        """Calculate AUC using trapezoidal rule"""
        # Sort by probabilities
        sorted_indices = np.argsort(probabilities)
        sorted_labels = labels[sorted_indices]
        sorted_probs = probabilities[sorted_indices]
        
        # Calculate TPR and FPR
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        tpr = []
        fpr = []
        
        for i in range(len(sorted_probs)):
            threshold = sorted_probs[i]
            tp = np.sum((sorted_probs >= threshold) & (sorted_labels == 1))
            fp = np.sum((sorted_probs >= threshold) & (sorted_labels == 0))
            
            tpr.append(tp / n_pos)
            fpr.append(fp / n_neg)
        
        # Calculate AUC using trapezoidal rule
        auc = 0
        for i in range(1, len(fpr)):
            auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
        
        return auc
    
    def calculate_confusion_matrix(self, labels, predictions, unique_labels):
        """Calculate confusion matrix"""
        n_classes = len(unique_labels)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        for i in range(len(labels)):
            true_idx = np.where(unique_labels == labels[i])[0][0]
            pred_idx = np.where(unique_labels == predictions[i])[0][0]
            cm[true_idx, pred_idx] += 1
        
        return cm
    
    def generate_report(self, metrics):
        """Generate comprehensive text report"""
        logger.info("Generating comprehensive report...")
        
        class_names = ['Normal', 'Impaired'] if len(metrics['unique_labels']) == 2 else ['CN', 'MCI', 'AD']
        
        report = f"""
COMPREHENSIVE MODEL EVALUATION REPORT
=====================================

Dataset Information:
- Total samples evaluated: {len(metrics['labels'])}
- Classes: {len(metrics['unique_labels'])}
- Class distribution: {dict(zip(class_names, metrics['class_counts']))}

Overall Performance Metrics:
- Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)
- Precision (Weighted): {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)
- Recall (Weighted): {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)
- F1-Score (Weighted): {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)
- AUC-ROC: {metrics['auc_roc']:.4f} ({metrics['auc_roc']*100:.2f}%)

Confusion Matrix:
{metrics['confusion_matrix']}

Per-Class Performance:
"""
        
        # Add per-class metrics
        for i, class_name in enumerate(class_names):
            report += f"\n{class_name}:"
            report += f"  Precision: {metrics['class_precision'][i]:.4f}"
            report += f"  Recall: {metrics['class_recall'][i]:.4f}"
            report += f"  F1-Score: {metrics['class_f1'][i]:.4f}"
            report += f"  Support: {metrics['class_counts'][i]}"
        
        report += f"""

Model Analysis:
- The model achieves {metrics['accuracy']*100:.2f}% accuracy on the entire dataset
- {'Good' if metrics['auc_roc'] > 0.8 else 'Moderate' if metrics['auc_roc'] > 0.7 else 'Poor'} discrimination ability (AUC = {metrics['auc_roc']:.3f})
- {'Balanced' if abs(metrics['precision'] - metrics['recall']) < 0.1 else 'Imbalanced'} precision-recall tradeoff
- {'Strong' if metrics['f1_score'] > 0.8 else 'Moderate' if metrics['f1_score'] > 0.7 else 'Weak'} overall performance

Detailed Results:
- True positives: {np.sum((metrics['predictions'] == 1) & (metrics['labels'] == 1))}
- False positives: {np.sum((metrics['predictions'] == 1) & (metrics['labels'] == 0))}
- True negatives: {np.sum((metrics['predictions'] == 0) & (metrics['labels'] == 0))}
- False negatives: {np.sum((metrics['predictions'] == 0) & (metrics['labels'] == 1))}

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
        with open('entire_dataset_metrics.csv', 'w') as f:
            f.write("Metric,Value,Percentage\n")
            f.write(f"Accuracy,{metrics['accuracy']:.6f},{metrics['accuracy']*100:.2f}\n")
            f.write(f"Precision,{metrics['precision']:.6f},{metrics['precision']*100:.2f}\n")
            f.write(f"Recall,{metrics['recall']:.6f},{metrics['recall']*100:.2f}\n")
            f.write(f"F1-Score,{metrics['f1_score']:.6f},{metrics['f1_score']*100:.2f}\n")
            f.write(f"AUC-ROC,{metrics['auc_roc']:.6f},{metrics['auc_roc']*100:.2f}\n")
        
        # Save detailed results
        with open('entire_dataset_predictions.csv', 'w') as f:
            f.write("True_Label,Predicted_Label,Probability_Normal,Probability_Impaired\n")
            for i in range(len(metrics['labels'])):
                f.write(f"{metrics['labels'][i]},{metrics['predictions'][i]},{metrics['probabilities'][i][0]:.6f},{metrics['probabilities'][i][1]:.6f}\n")
        
        # Save confusion matrix
        with open('entire_dataset_confusion_matrix.csv', 'w') as f:
            f.write("True\\Predicted,")
            class_names = ['Normal', 'Impaired'] if len(metrics['unique_labels']) == 2 else ['CN', 'MCI', 'AD']
            f.write(",".join(class_names) + "\n")
            for i, row in enumerate(metrics['confusion_matrix']):
                f.write(f"{class_names[i]},")
                f.write(",".join(map(str, row)) + "\n")
        
        # Save report
        with open('entire_dataset_report.txt', 'w') as f:
            f.write(report)
        
        logger.info("Results saved to:")
        logger.info("- entire_dataset_metrics.csv")
        logger.info("- entire_dataset_predictions.csv") 
        logger.info("- entire_dataset_confusion_matrix.csv")
        logger.info("- entire_dataset_report.txt")
    
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
