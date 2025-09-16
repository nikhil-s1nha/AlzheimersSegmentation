#!/usr/bin/env python3
"""
Enhanced Model Architecture for Enhanced Neurotokens
Handles the new token structure with:
- Delta-tokens (quantile-binned)
- Level-tokens (reduced codebook size)
- Harmonized features (site-wise)
- Region embeddings (consistent order)
- Delta-t embeddings (temporal buckets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedNeuroTokenModel(nn.Module):
    """
    Enhanced model architecture for the new neurotoken structure
    
    Architecture:
    1. Separate encoders for each token type
    2. Multi-head attention for session-level processing
    3. Hierarchical GRU for subject-level processing
    4. Final classification head with dropout
    """
    
    def __init__(self, 
                 max_sessions=5,
                 max_tokens=28,
                 level_token_dim=10,
                 delta_token_dim=7,
                 harmonized_dim=28,
                 region_embedding_dim=28,
                 delta_t_bucket_dim=4,
                 hidden_dim=128,
                 num_layers=2,
                 num_heads=8,
                 dropout=0.3,
                 num_classes=2):
        """
        Initialize the enhanced model
        
        Args:
            max_sessions: Maximum number of sessions
            max_tokens: Maximum number of tokens per session
            level_token_dim: Dimension of level tokens (codebook size)
            delta_token_dim: Dimension of delta tokens (number of bins)
            harmonized_dim: Dimension of harmonized features
            region_embedding_dim: Dimension of region embeddings
            delta_t_bucket_dim: Dimension of delta-t bucket embeddings
            hidden_dim: Hidden dimension for GRU and attention
            num_layers: Number of GRU layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            num_classes: Number of output classes
        """
        super(EnhancedNeuroTokenModel, self).__init__()
        
        self.max_sessions = max_sessions
        self.max_tokens = max_tokens
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        logger.info(f"Initializing Enhanced NeuroToken Model:")
        logger.info(f"  - Max sessions: {max_sessions}")
        logger.info(f"  - Max tokens: {max_tokens}")
        logger.info(f"  - Hidden dimension: {hidden_dim}")
        logger.info(f"  - Number of layers: {num_layers}")
        logger.info(f"  - Number of attention heads: {num_heads}")
        
        # 1. Token Embeddings
        self.level_token_embedding = nn.Embedding(level_token_dim, hidden_dim)
        self.delta_token_embedding = nn.Embedding(delta_token_dim, hidden_dim)
        
        # 2. Feature Projections
        self.harmonized_projection = nn.Linear(harmonized_dim, hidden_dim)
        self.region_embedding_projection = nn.Linear(region_embedding_dim, hidden_dim)
        self.delta_t_bucket_projection = nn.Linear(delta_t_bucket_dim, hidden_dim)
        
        # 3. Session-level Processing
        self.session_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.session_norm1 = nn.LayerNorm(hidden_dim)
        self.session_norm2 = nn.LayerNorm(hidden_dim)
        self.session_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 4. Subject-level Processing
        self.subject_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # 5. Final Classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional GRU
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 6. Additional components
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        self._init_weights()
        
        logger.info("Model initialization completed")
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, batch_data):
        """
        Forward pass through the enhanced model
        
        Args:
            batch_data: Dictionary containing:
                - level_tokens: [batch_size, max_sessions, max_tokens]
                - delta_tokens: [batch_size, max_sessions, max_tokens]
                - harmonized_features: [batch_size, max_sessions, harmonized_dim]
                - region_embeddings: [batch_size, max_sessions, region_embedding_dim]
                - delta_t_buckets: [batch_size, max_sessions, delta_t_bucket_dim]
        
        Returns:
            logits: [batch_size, num_classes]
            attention_weights: [batch_size, max_sessions] (optional)
        """
        batch_size = batch_data['level_tokens'].size(0)
        
        # 1. Process each session
        session_representations = []
        session_attention_weights = []
        
        for session_idx in range(self.max_sessions):
            session_repr = self._process_session(
                batch_data, session_idx, batch_size
            )
            session_representations.append(session_repr)
        
        # Stack session representations
        session_representations = torch.stack(session_representations, dim=1)  # [batch_size, max_sessions, hidden_dim]
        
        # 2. Apply session-level attention
        session_representations, session_attn_weights = self.session_attention(
            session_representations, session_representations, session_representations
        )
        
        # Add residual connection and normalization
        session_representations = self.session_norm1(session_representations + session_representations)
        
        # Apply FFN
        ffn_output = self.session_ffn(session_representations)
        session_representations = self.session_norm2(session_representations + ffn_output)
        
        # 3. Subject-level processing with GRU
        gru_output, _ = self.subject_gru(session_representations)  # [batch_size, max_sessions, hidden_dim*2]
        
        # 4. Global pooling (mean across sessions)
        subject_representation = torch.mean(gru_output, dim=1)  # [batch_size, hidden_dim*2]
        
        # 5. Final classification
        logits = self.classifier(subject_representation)
        
        return logits, session_attn_weights
    
    def _process_session(self, batch_data, session_idx, batch_size):
        """
        Process a single session to create a session representation
        
        Args:
            batch_data: Batch data dictionary
            session_idx: Index of the session to process
            batch_size: Batch size
            
        Returns:
            session_repr: [batch_size, hidden_dim] - Session representation
        """
        # Extract session data
        level_tokens = batch_data['level_tokens'][:, session_idx, :]  # [batch_size, max_tokens]
        delta_tokens = batch_data['delta_tokens'][:, session_idx, :]  # [batch_size, max_tokens]
        harmonized_features = batch_data['harmonized_features'][:, session_idx, :]  # [batch_size, harmonized_dim]
        region_embeddings = batch_data['region_embeddings'][:, session_idx, :]  # [batch_size, region_embedding_dim]
        delta_t_buckets = batch_data['delta_t_buckets'][:, session_idx, :]  # [batch_size, delta_t_bucket_dim]
        
        # 1. Token embeddings (convert to long for embedding layers)
        level_embeddings = self.level_token_embedding(level_tokens.long())  # [batch_size, max_tokens, hidden_dim]
        delta_embeddings = self.delta_token_embedding(delta_tokens.long())  # [batch_size, max_tokens, hidden_dim]
        
        # 2. Feature projections
        harmonized_proj = self.harmonized_projection(harmonized_features)  # [batch_size, hidden_dim]
        region_proj = self.region_embedding_projection(region_embeddings)  # [batch_size, hidden_dim]
        delta_t_proj = self.delta_t_bucket_projection(delta_t_buckets)  # [batch_size, hidden_dim]
        
        # 3. Combine token embeddings (mean across tokens)
        level_repr = torch.mean(level_embeddings, dim=1)  # [batch_size, hidden_dim]
        delta_repr = torch.mean(delta_embeddings, dim=1)  # [batch_size, hidden_dim]
        
        # 4. Combine all representations
        session_repr = (
            level_repr + 
            delta_repr + 
            harmonized_proj + 
            region_proj + 
            delta_t_proj
        )
        
        # 5. Normalize and apply dropout
        session_repr = self.layer_norm(session_repr)
        session_repr = self.dropout(session_repr)
        
        return session_repr
    
    def get_attention_weights(self, batch_data):
        """
        Get attention weights for interpretability
        
        Args:
            batch_data: Batch data dictionary
            
        Returns:
            attention_weights: [batch_size, max_sessions] - Attention weights per session
        """
        with torch.no_grad():
            _, session_attn_weights = self.forward(batch_data)
            
            # Extract attention weights (mean across attention heads)
            if session_attn_weights is not None:
                # session_attn_weights shape: [batch_size, num_heads, max_sessions, max_sessions]
                # We want the diagonal (session attending to itself) averaged across heads
                batch_size, num_heads, max_sessions, _ = session_attn_weights.shape
                diag_weights = torch.diagonal(session_attn_weights, dim1=2, dim2=3)  # [batch_size, num_heads, max_sessions]
                attention_weights = torch.mean(diag_weights, dim=1)  # [batch_size, max_sessions]
            else:
                # If no attention weights, return uniform weights
                attention_weights = torch.ones(batch_data['level_tokens'].size(0), self.max_sessions) / self.max_sessions
                attention_weights = attention_weights.to(batch_data['level_tokens'].device)
            
            return attention_weights

class EnhancedNeuroTokenModelV2(nn.Module):
    """
    Alternative enhanced model with transformer-based architecture
    
    This version uses a pure transformer approach instead of GRU
    """
    
    def __init__(self,
                 max_sessions=5,
                 max_tokens=28,
                 level_token_dim=10,
                 delta_token_dim=7,
                 harmonized_dim=28,
                 region_embedding_dim=28,
                 delta_t_bucket_dim=4,
                 hidden_dim=128,
                 num_layers=3,
                 num_heads=8,
                 dropout=0.3,
                 num_classes=2):
        """
        Initialize the transformer-based enhanced model
        """
        super(EnhancedNeuroTokenModelV2, self).__init__()
        
        self.max_sessions = max_sessions
        self.max_tokens = max_tokens
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        logger.info(f"Initializing Enhanced NeuroToken Model V2 (Transformer):")
        logger.info(f"  - Max sessions: {max_sessions}")
        logger.info(f"  - Max tokens: {max_tokens}")
        logger.info(f"  - Hidden dimension: {hidden_dim}")
        logger.info(f"  - Number of layers: {num_layers}")
        logger.info(f"  - Number of attention heads: {num_heads}")
        
        # Token embeddings and projections (same as V1)
        self.level_token_embedding = nn.Embedding(level_token_dim, hidden_dim)
        self.delta_token_embedding = nn.Embedding(delta_token_dim, hidden_dim)
        self.harmonized_projection = nn.Linear(harmonized_dim, hidden_dim)
        self.region_embedding_projection = nn.Linear(region_embedding_dim, hidden_dim)
        self.delta_t_bucket_projection = nn.Linear(delta_t_bucket_dim, hidden_dim)
        
        # Positional encoding for sessions
        self.session_pos_encoding = nn.Parameter(torch.randn(1, max_sessions, hidden_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Additional components
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        self._init_weights()
        
        logger.info("Model V2 initialization completed")
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
            elif isinstance(module, nn.TransformerEncoder):
                for layer in module.layers:
                    # Initialize attention weights
                    nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
                    nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
                    # Initialize FFN weights
                    nn.init.xavier_uniform_(layer.linear1.weight)
                    nn.init.xavier_uniform_(layer.linear2.weight)
    
    def forward(self, batch_data):
        """
        Forward pass through the transformer-based model
        """
        batch_size = batch_data['level_tokens'].size(0)
        
        # Process each session
        session_representations = []
        
        for session_idx in range(self.max_sessions):
            session_repr = self._process_session(
                batch_data, session_idx, batch_size
            )
            session_representations.append(session_repr)
        
        # Stack and add positional encoding
        session_representations = torch.stack(session_representations, dim=1)  # [batch_size, max_sessions, hidden_dim]
        session_representations = session_representations + self.session_pos_encoding
        
        # Apply transformer encoder
        encoded_sessions = self.transformer_encoder(session_representations)
        
        # Global pooling (mean across sessions)
        subject_representation = torch.mean(encoded_sessions, dim=1)
        
        # Final classification
        logits = self.classifier(subject_representation)
        
        return logits, None  # No attention weights for transformer version
    
    def _process_session(self, batch_data, session_idx, batch_size):
        """Process a single session (same as V1)"""
        # Extract session data
        level_tokens = batch_data['level_tokens'][:, session_idx, :]
        delta_tokens = batch_data['delta_tokens'][:, session_idx, :]
        harmonized_features = batch_data['harmonized_features'][:, session_idx, :]
        region_embeddings = batch_data['region_embeddings'][:, session_idx, :]
        delta_t_buckets = batch_data['delta_t_buckets'][:, session_idx, :]
        
        # Token embeddings
        level_embeddings = self.level_token_embedding(level_tokens)
        delta_embeddings = self.delta_token_embedding(delta_tokens)
        
        # Feature projections
        harmonized_proj = self.harmonized_projection(harmonized_features)
        region_proj = self.region_embedding_projection(region_embeddings)
        delta_t_proj = self.delta_t_bucket_projection(delta_t_buckets)
        
        # Combine token embeddings
        level_repr = torch.mean(level_embeddings, dim=1)
        delta_repr = torch.mean(delta_embeddings, dim=1)
        
        # Combine all representations
        session_repr = (
            level_repr + 
            delta_repr + 
            harmonized_proj + 
            region_proj + 
            delta_t_proj
        )
        
        # Normalize and apply dropout
        session_repr = self.layer_norm(session_repr)
        session_repr = self.dropout(session_repr)
        
        return session_repr

def create_enhanced_model(model_type='gru', **kwargs):
    """
    Factory function to create enhanced models
    
    Args:
        model_type: 'gru' or 'transformer'
        **kwargs: Model parameters
        
    Returns:
        EnhancedNeuroTokenModel or EnhancedNeuroTokenModelV2
    """
    if model_type.lower() == 'gru':
        return EnhancedNeuroTokenModel(**kwargs)
    elif model_type.lower() == 'transformer':
        return EnhancedNeuroTokenModelV2(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'gru' or 'transformer'")

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    """Test the enhanced model"""
    # Create a sample batch
    batch_size = 4
    max_sessions = 5
    max_tokens = 28
    
    sample_batch = {
        'level_tokens': torch.randint(0, 10, (batch_size, max_sessions, max_tokens)),
        'delta_tokens': torch.randint(0, 7, (batch_size, max_sessions, max_tokens)),
        'harmonized_features': torch.randn(batch_size, max_sessions, 28),
        'region_embeddings': torch.randn(batch_size, max_sessions, 28),
        'delta_t_buckets': torch.eye(4).unsqueeze(0).expand(batch_size, max_sessions, 4)
    }
    
    # Test GRU model
    logger.info("Testing GRU model...")
    gru_model = create_enhanced_model('gru', max_sessions=max_sessions, max_tokens=max_tokens)
    gru_logits, gru_attn = gru_model(sample_batch)
    logger.info(f"GRU model output shape: {gru_logits.shape}")
    logger.info(f"GRU model parameters: {count_parameters(gru_model):,}")
    
    # Test Transformer model
    logger.info("Testing Transformer model...")
    transformer_model = create_enhanced_model('transformer', max_sessions=max_sessions, max_tokens=max_tokens)
    transformer_logits, transformer_attn = transformer_model(sample_batch)
    logger.info(f"Transformer model output shape: {transformer_logits.shape}")
    logger.info(f"Transformer model parameters: {count_parameters(transformer_model):,}")
    
    # Test attention weights
    logger.info("Testing attention weights...")
    gru_attn_weights = gru_model.get_attention_weights(sample_batch)
    logger.info(f"GRU attention weights shape: {gru_attn_weights.shape}")
    
    logger.info("Model testing completed successfully!")

if __name__ == "__main__":
    main() 