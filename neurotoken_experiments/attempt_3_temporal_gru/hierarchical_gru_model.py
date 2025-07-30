#!/usr/bin/env python3
"""
Hierarchical GRU Model for Temporally-Aware NeuroToken Classification
Implements session encoder + time embedding + subject encoder architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HierarchicalGRU(nn.Module):
    """
    Hierarchical GRU model for temporally-aware neurotoken classification
    
    Architecture:
    1. Session Encoder: GRU processes each session's tokens
    2. Time Embedding: Linear layer encodes delay information
    3. Subject Encoder: GRU processes session embeddings over time
    4. Classification Head: Linear layer for binary classification
    """
    
    def __init__(self, 
                 vocab_size=32,
                 token_emb_dim=32,
                 session_hidden_dim=64,
                 subject_hidden_dim=128,
                 time_emb_dim=16,
                 num_layers=2,
                 dropout=0.3,
                 max_sessions=5,
                 max_tokens=28):
        """
        Initialize the hierarchical GRU model
        
        Args:
            vocab_size: Number of unique tokens (32 for neurotokens)
            token_emb_dim: Embedding dimension for tokens
            session_hidden_dim: Hidden dimension for session-level GRU
            subject_hidden_dim: Hidden dimension for subject-level GRU
            time_emb_dim: Embedding dimension for time delays
            num_layers: Number of layers in GRUs
            dropout: Dropout rate
            max_sessions: Maximum number of sessions per subject
            max_tokens: Maximum number of tokens per session
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.token_emb_dim = token_emb_dim
        self.session_hidden_dim = session_hidden_dim
        self.subject_hidden_dim = subject_hidden_dim
        self.time_emb_dim = time_emb_dim
        self.max_sessions = max_sessions
        self.max_tokens = max_tokens
        
        # 1. Session Encoder
        self.token_embedding = nn.Embedding(vocab_size, token_emb_dim, padding_idx=0)
        self.session_gru = nn.GRU(
            input_size=token_emb_dim,
            hidden_size=session_hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # 2. Time Embedding
        self.time_embedding = nn.Linear(1, time_emb_dim)
        
        # 3. Subject Encoder
        # Input: session_embedding + time_embedding
        subject_input_dim = session_hidden_dim * 2 + time_emb_dim  # *2 for bidirectional
        self.subject_gru = nn.GRU(
            input_size=subject_input_dim,
            hidden_size=subject_hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # 4. Classification Head
        classifier_input_dim = subject_hidden_dim * 2  # *2 for bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, subject_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(subject_hidden_dim, 1)  # Binary classification
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized HierarchicalGRU with {self._count_parameters()} parameters")
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        
        # Initialize GRU weights
        for gru in [self.session_gru, self.subject_gru]:
            for name, param in gru.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        # Initialize time embedding and classifier
        for module in [self.time_embedding, self.classifier]:
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
    
    def encode_session(self, session_tokens, session_mask):
        """
        Encode a single session's tokens
        
        Args:
            session_tokens: Token sequence [batch_size, max_tokens]
            session_mask: Mask for valid tokens [batch_size, max_tokens]
            
        Returns:
            session_embedding: Session representation [batch_size, session_hidden_dim*2]
        """
        batch_size = session_tokens.shape[0]
        
        # Token embeddings
        token_embeddings = self.token_embedding(session_tokens)  # [batch_size, max_tokens, token_emb_dim]
        
        # Apply mask
        token_embeddings = token_embeddings * session_mask.unsqueeze(-1).float()
        
        # Check if any sequence has zero length
        lengths = session_mask.sum(dim=1).cpu()
        if (lengths == 0).any():
            # Handle zero-length sequences by creating zero embeddings
            session_embedding = torch.zeros(batch_size, self.session_hidden_dim * 2, device=session_tokens.device)
            return session_embedding
        
        # Pack padded sequences
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            token_embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        
        # Pass through session GRU
        packed_output, hidden = self.session_gru(packed_embeddings)
        
        # Use the final hidden state from both directions
        # hidden shape: [num_layers * num_directions, batch_size, session_hidden_dim]
        final_hidden = hidden[-2:].transpose(0, 1).contiguous()  # [batch_size, 2, session_hidden_dim]
        session_embedding = final_hidden.view(batch_size, -1)  # [batch_size, session_hidden_dim*2]
        
        return session_embedding
    
    def forward(self, input_ids, delays, attention_mask, session_mask=None):
        """
        Forward pass through the hierarchical model
        
        Args:
            input_ids: Token sequences [batch_size, max_sessions, max_tokens]
            delays: Time delays [batch_size, max_sessions]
            attention_mask: Attention mask [batch_size, max_sessions, max_tokens]
            session_mask: Session mask [batch_size, max_sessions] (optional)
            
        Returns:
            logits: Classification logits [batch_size, 1]
        """
        batch_size, max_sessions, max_tokens = input_ids.shape
        
        # Process each session
        session_embeddings = []
        
        for session_idx in range(max_sessions):
            session_tokens = input_ids[:, session_idx, :]  # [batch_size, max_tokens]
            token_mask = attention_mask[:, session_idx, :]  # [batch_size, max_tokens]
            
            # Skip encoding if session is all padding
            if session_mask is not None and not session_mask[:, session_idx].any():
                # Create zero embedding for padding sessions
                zero_embedding = torch.zeros(batch_size, self.session_hidden_dim * 2, device=input_ids.device)
                session_embeddings.append(zero_embedding)
            else:
                # Encode session
                session_embedding = self.encode_session(session_tokens, token_mask)
                session_embeddings.append(session_embedding)
        
        # Stack session embeddings
        session_embeddings = torch.stack(session_embeddings, dim=1)  # [batch_size, max_sessions, session_hidden_dim*2]
        
        # Time embeddings
        delays = delays.unsqueeze(-1)  # [batch_size, max_sessions, 1]
        time_embeddings = self.time_embedding(delays)  # [batch_size, max_sessions, time_emb_dim]
        
        # Combine session and time embeddings
        combined_embeddings = torch.cat([session_embeddings, time_embeddings], dim=-1)
        
        # Apply dropout
        combined_embeddings = self.dropout(combined_embeddings)
        
        # Use provided session mask or create from attention mask
        if session_mask is None:
            session_mask = attention_mask.any(dim=-1)  # [batch_size, max_sessions]
        
        # Pack sessions for subject-level GRU
        lengths = session_mask.sum(dim=1).cpu()
        packed_sessions = nn.utils.rnn.pack_padded_sequence(
            combined_embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        
        # Pass through subject-level GRU
        packed_output, hidden = self.subject_gru(packed_sessions)
        
        # Use final hidden state from both directions
        final_hidden = hidden[-2:].transpose(0, 1).contiguous()  # [batch_size, 2, subject_hidden_dim]
        subject_embedding = final_hidden.view(batch_size, -1)  # [batch_size, subject_hidden_dim*2]
        
        # Classification
        logits = self.classifier(subject_embedding)  # [batch_size, 1]
        
        return logits
    
    def get_attention_weights(self, input_ids, delays, attention_mask, session_mask=None):
        """
        Get attention weights for visualization (simplified version)
        
        Args:
            input_ids: Token sequences [batch_size, max_sessions, max_tokens]
            delays: Time delays [batch_size, max_sessions]
            attention_mask: Attention mask [batch_size, max_sessions, max_tokens]
            session_mask: Session mask [batch_size, max_sessions] (optional)
            
        Returns:
            session_attention: Session-level attention weights
        """
        batch_size, max_sessions, max_tokens = input_ids.shape
        
        # Process each session
        session_embeddings = []
        
        for session_idx in range(max_sessions):
            session_tokens = input_ids[:, session_idx, :]
            token_mask = attention_mask[:, session_idx, :]
            
            # Skip encoding if session is all padding
            if session_mask is not None and not session_mask[:, session_idx].any():
                zero_embedding = torch.zeros(batch_size, self.session_hidden_dim * 2, device=input_ids.device)
                session_embeddings.append(zero_embedding)
            else:
                session_embedding = self.encode_session(session_tokens, token_mask)
                session_embeddings.append(session_embedding)
        
        session_embeddings = torch.stack(session_embeddings, dim=1)
        
        # Time embeddings
        delays = delays.unsqueeze(-1)
        time_embeddings = self.time_embedding(delays)
        
        # Combine embeddings
        combined_embeddings = torch.cat([session_embeddings, time_embeddings], dim=-1)
        
        # Simple attention: use the magnitude of embeddings as attention weights
        session_attention = torch.norm(combined_embeddings, dim=-1)  # [batch_size, max_sessions]
        session_attention = F.softmax(session_attention, dim=-1)
        
        return session_attention


class HierarchicalGRUConfig:
    """Configuration class for HierarchicalGRU"""
    
    def __init__(self,
                 vocab_size=32,
                 token_emb_dim=32,
                 session_hidden_dim=64,
                 subject_hidden_dim=128,
                 time_emb_dim=16,
                 num_layers=2,
                 dropout=0.3,
                 max_sessions=5,
                 max_tokens=28):
        
        self.vocab_size = vocab_size
        self.token_emb_dim = token_emb_dim
        self.session_hidden_dim = session_hidden_dim
        self.subject_hidden_dim = subject_hidden_dim
        self.time_emb_dim = time_emb_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_sessions = max_sessions
        self.max_tokens = max_tokens
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'vocab_size': self.vocab_size,
            'token_emb_dim': self.token_emb_dim,
            'session_hidden_dim': self.session_hidden_dim,
            'subject_hidden_dim': self.subject_hidden_dim,
            'time_emb_dim': self.time_emb_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'max_sessions': self.max_sessions,
            'max_tokens': self.max_tokens
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        return cls(**config_dict)


def create_hierarchical_model(config):
    """
    Create a HierarchicalGRU model from config
    
    Args:
        config: HierarchicalGRUConfig object
        
    Returns:
        HierarchicalGRU model
    """
    return HierarchicalGRU(
        vocab_size=config.vocab_size,
        token_emb_dim=config.token_emb_dim,
        session_hidden_dim=config.session_hidden_dim,
        subject_hidden_dim=config.subject_hidden_dim,
        time_emb_dim=config.time_emb_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        max_sessions=config.max_sessions,
        max_tokens=config.max_tokens
    )


if __name__ == "__main__":
    # Test the hierarchical model
    config = HierarchicalGRUConfig()
    model = create_hierarchical_model(config)
    
    # Test forward pass
    batch_size = 4
    max_sessions = 5
    max_tokens = 28
    
    input_ids = torch.randint(0, 32, (batch_size, max_sessions, max_tokens))
    delays = torch.rand(batch_size, max_sessions)
    attention_mask = torch.ones(batch_size, max_sessions, max_tokens, dtype=torch.bool)
    
    # Forward pass
    logits = model(input_ids, delays, attention_mask)
    
    logger.info(f"Model output shape: {logits.shape}")
    logger.info(f"Expected shape: [{batch_size}, 1]")
    
    # Test attention weights
    attention_weights = model.get_attention_weights(input_ids, delays, attention_mask)
    logger.info(f"Attention weights shape: {attention_weights.shape}")
    
    logger.info("Hierarchical GRU model test completed successfully!") 