import os
import json
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from enhanced_dataset import (
    load_enhanced_tokens,
    load_subject_labels,
    create_enhanced_sequences,
    split_enhanced_data,
    EnhancedNeuroTokenDataset,
)
from enhanced_model import create_enhanced_model


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits, _ = model(batch)
            preds = torch.argmax(logits, dim=1)
            labels = batch['label'].squeeze()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    accuracy = correct / total if total > 0 else 0.0
    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])
    # Precision/Recall/F1 (binary)
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return accuracy, precision, recall, f1


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Must match the training config
    config = {
        "token_file": "/Volumes/SEAGATE_NIKHIL/neurotokens_project/enhanced_attempt/enhanced_tokens.jsonl",
        "labels_file": "/Volumes/SEAGATE_NIKHIL/neurotokens_project/subject_labels.csv",
        "output_dir": "/Volumes/SEAGATE_NIKHIL/neurotokens_project/enhanced_attempt/models",
        "max_sessions": 5,
        "max_tokens": 28,
        "batch_size": 16,
        "test_size": 0.2,
        "val_size": 0.2,
        "random_state": 42,
        "num_workers": 4,
    }

    logger.info("Setting up data for evaluation...")

    sequences = load_enhanced_tokens(config['token_file'])
    subject_labels = load_subject_labels(config['labels_file'])
    sequences = create_enhanced_sequences(sequences, subject_labels)
    train_seqs, val_seqs, test_seqs = split_enhanced_data(
        sequences,
        test_size=config['test_size'],
        val_size=config['val_size'],
        random_state=config['random_state']
    )

    transformers_path = os.path.join(config['output_dir'], 'transformers.pkl')
    best_model_path = os.path.join(config['output_dir'], 'best_model.pt')

    # Build datasets/loaders (load existing transformers)
    test_dataset = EnhancedNeuroTokenDataset(
        test_seqs,
        max_sessions=config['max_sessions'],
        max_tokens=config['max_tokens'],
        fit_transformers=False,
        transformers_path=transformers_path,
    )
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # Infer feature dims from one batch
    sample_batch = next(iter(test_loader))
    level_token_dim = int(sample_batch['level_tokens'].max().item()) + 1
    delta_token_dim = int(sample_batch['delta_tokens'].max().item()) + 1
    harmonized_dim = int(sample_batch['harmonized_features'].size(-1))
    region_embedding_dim = int(sample_batch['region_embeddings'].size(-1))
    delta_t_bucket_dim = int(sample_batch['delta_t_buckets'].size(-1))

    logger.info("Initializing model for evaluation...")
    model = create_enhanced_model(
        model_type='gru',
        max_sessions=config['max_sessions'],
        max_tokens=config['max_tokens'],
        level_token_dim=level_token_dim,
        delta_token_dim=delta_token_dim,
        harmonized_dim=harmonized_dim,
        region_embedding_dim=region_embedding_dim,
        delta_t_bucket_dim=delta_t_bucket_dim,
        hidden_dim=128,
        num_layers=2,
        num_heads=8,
        dropout=0.3,
        num_classes=2,
    ).to(device)

    state = torch.load(best_model_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    elif isinstance(state, dict) and 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)

    logger.info("Evaluating on test set...")
    acc, prec, rec, f1 = evaluate_model(model, test_loader, device)
    logger.info(f"Test Accuracy: {acc*100:.2f}% | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    # Save metrics
    metrics_path = os.path.join(config['output_dir'], 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}, f, indent=2)
    logger.info(f"Saved test metrics to {metrics_path}")


if __name__ == "__main__":
    main()

