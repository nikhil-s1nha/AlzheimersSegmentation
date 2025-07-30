#!/usr/bin/env python3
"""
Lightweight GRU Model for Binary NeuroToken Classification
More suitable for smaller datasets than transformer models.
"""

import torch
import torch.nn as nn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeuroTokenGRU(nn.Module):
    """Lightweight GRU model for binary neurotoken classification"""
    
    def __init__(self, vocab_size=32, emb_dim=32, hidden_dim=64, num_layers=2, 
                 dropout=0.3, max_len=224, num_classes=2):
        """
        Initialize the GRU model
        
        Args:
            vocab_size: Number of unique tokens (32 for our neurotokens)
            emb_dim: Embedding dimension
            hidden_dim: Hidden dimension of GRU
            num_layers: Number of GRU layers
            dropout: Dropout rate
            max_len: Maximum sequence length
            num_classes: Number of output classes (2 for binary)
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Token embeddings
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized NeuroTokenGRU with {self._count_parameters()} parameters")
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize embeddings
        nn.init.normal_(self.embed.weight, mean=0, std=0.02)
        
        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize attention and classifier
        for module in [self.attention, self.classifier]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
    
    def _count_parameters(self):
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Token sequences [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            logits: Classification logits [batch_size, num_classes]
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.embed(input_ids)  # [batch_size, seq_len, emb_dim]
        
        # Apply attention mask to embeddings (zero out padding)
        x = x * attention_mask.unsqueeze(-1).float()
        
        # Pack padded sequences for GRU
        lengths = attention_mask.sum(dim=1).cpu()
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        
        # Pass through GRU
        packed_output, hidden = self.gru(packed_x)
        
        # Unpack sequences
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=seq_len
        )
        
        # Apply attention mask to output
        output = output * attention_mask.unsqueeze(-1).float()
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(output), dim=1)  # [batch_size, seq_len, 1]
        attended_output = torch.sum(attention_weights * output, dim=1)  # [batch_size, hidden_dim*2]
        
        # Apply dropout
        attended_output = self.dropout(attended_output)
        
        # Classification
        logits = self.classifier(attended_output)  # [batch_size, num_classes]
        
        return logits
    
    def get_attention_weights(self, input_ids, attention_mask):
        """
        Get attention weights for visualization
        
        Args:
            input_ids: Token sequences [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            attention_weights: Attention weights [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.embed(input_ids)
        x = x * attention_mask.unsqueeze(-1).float()
        
        # Pack and pass through GRU
        lengths = attention_mask.sum(dim=1).cpu()
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.gru(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=seq_len
        )
        output = output * attention_mask.unsqueeze(-1).float()
        
        # Get attention weights
        attention_weights = torch.softmax(self.attention(output), dim=1).squeeze(-1)
        
        return attention_weights


class NeuroTokenGRUConfig:
    """Configuration class for NeuroTokenGRU"""
    
    def __init__(self, 
                 vocab_size=32,
                 emb_dim=32,
                 hidden_dim=64,
                 num_layers=2,
                 dropout=0.3,
                 max_len=224,
                 num_classes=2):
        
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_len = max_len
        self.num_classes = num_classes
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'vocab_size': self.vocab_size,
            'emb_dim': self.emb_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'max_len': self.max_len,
            'num_classes': self.num_classes
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        return cls(**config_dict)


def create_gru_model(config):
    """
    Create a NeuroTokenGRU model from config
    
    Args:
        config: NeuroTokenGRUConfig object
        
    Returns:
        NeuroTokenGRU model
    """
    return NeuroTokenGRU(
        vocab_size=config.vocab_size,
        emb_dim=config.emb_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        max_len=config.max_len,
        num_classes=config.num_classes
    )


if __name__ == "__main__":
    # Test the GRU model
    config = NeuroTokenGRUConfig()
    model = create_gru_model(config)
    
    # Test forward pass
    batch_size = 4
    seq_len = 100
    input_ids = torch.randint(0, 32, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Forward pass
    logits = model(input_ids, attention_mask)
    
    logger.info(f"Model output shape: {logits.shape}")
    logger.info(f"Expected shape: [{batch_size}, {config.num_classes}]")
    
    # Test attention weights
    attention_weights = model.get_attention_weights(input_ids, attention_mask)
    logger.info(f"Attention weights shape: {attention_weights.shape}")
    
    logger.info("GRU model test completed successfully!") 