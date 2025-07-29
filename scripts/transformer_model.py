#!/usr/bin/env python3
"""
NeuroToken Transformer Model
PyTorch transformer model for classifying neurotoken sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NeuroTokenTransformer(nn.Module):
    """Transformer model for neurotoken sequence classification"""
    
    def __init__(self, vocab_size=32, emb_dim=64, max_len=224, num_classes=3, 
                 num_layers=2, num_heads=4, dim_feedforward=256, dropout=0.1):
        """
        Initialize the transformer model
        
        Args:
            vocab_size: Number of unique tokens (32 for our neurotokens)
            emb_dim: Embedding dimension
            max_len: Maximum sequence length
            num_classes: Number of output classes (3: CN, MCI, AD)
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.num_classes = num_classes
        
        # Token embeddings
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        # Positional embeddings (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, emb_dim))
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Use batch_first=True for easier handling
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Linear(emb_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized NeuroTokenTransformer with {self._count_parameters()} parameters")
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize embeddings
        nn.init.normal_(self.embed.weight, mean=0, std=0.02)
        nn.init.normal_(self.pos_embed, mean=0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0, std=0.02)
        
        # Initialize transformer layers
        for module in self.transformer.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
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
        
        # Add positional embeddings
        x = x + self.pos_embed[:, :seq_len, :]
        
        # Add CLS token at the beginning
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, emb_dim]
        x = torch.cat([cls_token, x], dim=1)  # [batch_size, seq_len+1, emb_dim]
        
        # Update attention mask to include CLS token
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=attention_mask.device)
        attn_mask = torch.cat([cls_mask, attention_mask], dim=1)  # [batch_size, seq_len+1]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through transformer encoder
        # Note: src_key_padding_mask expects True for padding tokens (opposite of attention_mask)
        x = self.transformer(x, src_key_padding_mask=~attn_mask)
        
        # Extract CLS token output for classification
        cls_output = x[:, 0, :]  # [batch_size, emb_dim]
        
        # Classification
        logits = self.classifier(cls_output)  # [batch_size, num_classes]
        
        return logits
    
    def get_attention_weights(self, input_ids, attention_mask):
        """
        Get attention weights for visualization
        
        Args:
            input_ids: Token sequences [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            attention_weights: Attention weights from all layers
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.embed(input_ids)
        x = x + self.pos_embed[:, :seq_len, :]
        
        # Add CLS token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Update attention mask
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=attention_mask.device)
        attn_mask = torch.cat([cls_mask, attention_mask], dim=1)
        
        # Collect attention weights from each layer
        attention_weights = []
        
        # Pass through transformer layers and collect attention weights
        for layer in self.transformer.layers:
            # Get attention weights from the self-attention layer
            attn_output, attn_weights = layer.self_attn(
                x, x, x, 
                key_padding_mask=~attn_mask,
                need_weights=True
            )
            attention_weights.append(attn_weights)
            
            # Continue with the rest of the layer (simplified to avoid internal method issues)
            x = layer.norm1(x + attn_output)
            x = layer.norm2(x + layer._ff_block(x))
        
        return attention_weights


class NeuroTokenTransformerConfig:
    """Configuration class for NeuroTokenTransformer"""
    
    def __init__(self, 
                 vocab_size=32,
                 emb_dim=64,
                 max_len=224,
                 num_classes=3,
                 num_layers=2,
                 num_heads=4,
                 dim_feedforward=256,
                 dropout=0.1):
        
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'vocab_size': self.vocab_size,
            'emb_dim': self.emb_dim,
            'max_len': self.max_len,
            'num_classes': self.num_classes,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        return cls(**config_dict)


def create_model(config):
    """
    Create a NeuroTokenTransformer model from config
    
    Args:
        config: NeuroTokenTransformerConfig object
        
    Returns:
        NeuroTokenTransformer model
    """
    return NeuroTokenTransformer(
        vocab_size=config.vocab_size,
        emb_dim=config.emb_dim,
        max_len=config.max_len,
        num_classes=config.num_classes,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout
    )


if __name__ == "__main__":
    # Test the model
    config = NeuroTokenTransformerConfig()
    model = create_model(config)
    
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
    logger.info(f"Number of attention layers: {len(attention_weights)}")
    logger.info(f"Attention weights shape: {attention_weights[0].shape}")
    
    logger.info("Model test completed successfully!") 