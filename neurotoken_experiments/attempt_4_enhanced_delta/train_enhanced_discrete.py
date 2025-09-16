import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from enhanced_dataset import load_enhanced_tokens, load_subject_labels, create_enhanced_sequences, split_enhanced_data
from enhanced_dataset_discrete import EnhancedNeuroTokenDatasetDiscrete
from enhanced_model import create_enhanced_model, count_parameters


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DiscreteTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def setup_data(self):
        sequences = load_enhanced_tokens(self.config['token_file'])
        labels = load_subject_labels(self.config['labels_file'])
        sequences = create_enhanced_sequences(sequences, labels)
        train_seqs, val_seqs, test_seqs = split_enhanced_data(
            sequences,
            test_size=self.config['test_size'],
            val_size=self.config['val_size'],
            random_state=self.config['random_state'],
        )

        transformers_path = os.path.join(self.config['output_dir'], 'transformers_discrete.pkl')
        self.train_dataset = EnhancedNeuroTokenDatasetDiscrete(train_seqs, self.config['max_sessions'], self.config['max_tokens'], True, transformers_path)
        self.val_dataset = EnhancedNeuroTokenDatasetDiscrete(val_seqs, self.config['max_sessions'], self.config['max_tokens'], False, transformers_path)
        self.test_dataset = EnhancedNeuroTokenDatasetDiscrete(test_seqs, self.config['max_sessions'], self.config['max_tokens'], False, transformers_path)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers'])
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'])
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'])

    def setup_model(self):
        sample_batch = next(iter(self.train_loader))
        level_token_dim = int(sample_batch['level_tokens'].max().item()) + 1
        delta_token_dim = int(sample_batch['delta_tokens'].max().item()) + 1
        harmonized_dim = int(sample_batch['harmonized_features'].size(-1))
        region_embedding_dim = int(sample_batch['region_embeddings'].size(-1))
        delta_t_bucket_dim = int(sample_batch['delta_t_buckets'].size(-1))

        self.model = create_enhanced_model(
            model_type=self.config['model_type'],
            max_sessions=self.config['max_sessions'],
            max_tokens=self.config['max_tokens'],
            level_token_dim=level_token_dim,
            delta_token_dim=delta_token_dim,
            harmonized_dim=harmonized_dim,
            region_embedding_dim=region_embedding_dim,
            delta_t_bucket_dim=delta_t_bucket_dim,
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            dropout=self.config['dropout'],
            num_classes=self.config['num_classes'],
        ).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)

        # class weights
        labels = [d['label'].item() for d in self.train_dataset]
        counts = np.bincount(labels)
        weights = torch.FloatTensor(1.0 / counts).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        logger.info(f"Model setup completed. Parameters: {count_parameters(self.model):,}")

    def train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in self.train_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            self.optimizer.zero_grad()
            logits, _ = self.model(batch)
            labels = batch['label'].squeeze()
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
            self.optimizer.step()
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return total_loss / len(self.train_loader), correct / total

    @torch.no_grad()
    def validate_epoch(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits, _ = self.model(batch)
            labels = batch['label'].squeeze()
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return total_loss / len(self.val_loader), correct / total

    def train(self):
        self.setup_data()
        self.setup_model()
        best_val = 0.0
        patience_left = self.config['patience']
        os.makedirs(self.config['output_dir'], exist_ok=True)
        best_path = os.path.join(self.config['output_dir'], 'best_model_discrete.pt')

        for epoch in range(1, self.config['num_epochs'] + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()
            self.scheduler.step(val_acc)
            logger.info(f"[Discrete] Epoch {epoch} | Train {train_acc*100:.2f}% | Val {val_acc*100:.2f}%")
            if val_acc > best_val:
                best_val = val_acc
                patience_left = self.config['patience']
                torch.save(self.model.state_dict(), best_path)
            else:
                patience_left -= 1
                if patience_left == 0:
                    logger.info("[Discrete] Early stopping")
                    break


def main():
    config = {
        "token_file": "enhanced_tokens_new.json",
        "labels_file": "subject_labels.csv",
        "output_dir": "models",
        "model_type": "gru",
        "max_sessions": 5,
        "max_tokens": 28,
        "hidden_dim": 128,
        "num_layers": 2,
        "num_heads": 8,
        "dropout": 0.3,
        "num_classes": 2,
        "batch_size": 16,
        "num_epochs": 100,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "max_grad_norm": 1.0,
        "patience": 10,
        "test_size": 0.2,
        "val_size": 0.2,
        "random_state": 42,
        "num_workers": 4,
    }
    trainer = DiscreteTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

