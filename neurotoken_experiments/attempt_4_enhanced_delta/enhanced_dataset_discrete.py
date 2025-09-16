import os
import pickle
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from enhanced_dataset import (
    load_enhanced_tokens,
    load_subject_labels,
    create_enhanced_sequences,
    split_enhanced_data,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedNeuroTokenDatasetDiscrete(Dataset):
    """Dataset variant that keeps level/delta tokens as discrete indices (no scaling)."""

    def __init__(self, token_sequences, max_sessions=5, max_tokens=28, fit_transformers=True, transformers_path=None):
        self.token_sequences = token_sequences
        self.max_sessions = max_sessions
        self.max_tokens = max_tokens
        self.fit_transformers = fit_transformers
        self.transformers_path = transformers_path

        self.processed_data, self.transformers = self._process_sequences()
        logger.info(f"Initialized discrete dataset with {len(self.processed_data)} samples")

    def _process_sequences(self):
        processed = []
        all_harmonized = []
        all_region = []

        for seq in self.token_sequences:
            subject_id = seq['subject_id']
            label = seq['label']
            sessions = seq['sessions'][: self.max_sessions]

            level_tokens_list = []
            delta_tokens_list = []
            harmonized_list = []
            region_list = []
            delta_t_list = []

            for session in sessions:
                tokens = session['tokens']
                level_vals, delta_vals, harm_vals, region_vals = [], [], [], []
                delta_t_bucket = 0
                for k, v in tokens.items():
                    if k.startswith('level_'):
                        level_vals.append(int(v))
                    elif k.startswith('binned_delta_'):
                        delta_vals.append(int(v))
                    elif k.startswith('harmonized_'):
                        harm_vals.append(float(v))
                    elif k.startswith('region_') and k.endswith('_embedding'):
                        region_vals.append(float(v))
                    elif k == 'delta_t_bucket':
                        delta_t_bucket = int(v)

                # pad/truncate
                level_vals = (level_vals + [0] * self.max_tokens)[: self.max_tokens]
                delta_vals = (delta_vals + [0] * self.max_tokens)[: self.max_tokens]
                # Pad/truncate continuous features
                harm_vals = (harm_vals + [0.0] * self.max_tokens)[: self.max_tokens]
                region_vals = (region_vals + [0.0] * self.max_tokens)[: self.max_tokens]

                level_tokens_list.append(level_vals)
                delta_tokens_list.append(delta_vals)
                harmonized_list.append(harm_vals)
                region_list.append(region_vals)
                delta_t_list.append(delta_t_bucket)

            # pad sessions
            while len(level_tokens_list) < self.max_sessions:
                level_tokens_list.append([0] * self.max_tokens)
                delta_tokens_list.append([0] * self.max_tokens)
                harmonized_list.append([0.0] * self.max_tokens)
                region_list.append([0.0] * self.max_tokens)
                delta_t_list.append(0)

            data_item = {
                'subject_id': subject_id,
                'label': torch.LongTensor([label]),
                'level_tokens': np.array(level_tokens_list),
                'delta_tokens': np.array(delta_tokens_list),
                'harmonized_features': np.array(harmonized_list),
                'region_embeddings': np.array(region_list),
                'delta_t_buckets': np.array(delta_t_list),
            }
            processed.append(data_item)
            all_harmonized.extend(harmonized_list)
            all_region.extend(region_list)

        # Fit/load transformers for continuous features only
        if self.fit_transformers:
            scaler_h = StandardScaler().fit(np.array(all_harmonized).reshape(-1, self.max_tokens))
            scaler_r = StandardScaler().fit(np.array(all_region).reshape(-1, self.max_tokens))
            transformers = {
                'harmonized_scaler': scaler_h,
                'region_scaler': scaler_r,
            }
            if self.transformers_path:
                os.makedirs(os.path.dirname(self.transformers_path), exist_ok=True)
                with open(self.transformers_path, 'wb') as f:
                    pickle.dump(transformers, f)
        else:
            with open(self.transformers_path, 'rb') as f:
                transformers = pickle.load(f)

        # Apply only to continuous features; keep tokens discrete and cast later
        for d in processed:
            d['harmonized_features'] = torch.FloatTensor(transformers['harmonized_scaler'].transform(d['harmonized_features']))
            d['region_embeddings'] = torch.FloatTensor(transformers['region_scaler'].transform(d['region_embeddings']))
            d['level_tokens'] = torch.LongTensor(d['level_tokens'])
            d['delta_tokens'] = torch.LongTensor(d['delta_tokens'])
            onehot = np.eye(4)[d['delta_t_buckets'].astype(int)]
            d['delta_t_buckets'] = torch.FloatTensor(onehot)

        return processed, transformers

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]

