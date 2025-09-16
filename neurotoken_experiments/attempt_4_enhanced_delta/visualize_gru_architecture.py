#!/usr/bin/env python3
"""
GRU/RNN Architecture Visualization for NeuroToken Model
Shows the hierarchical structure and data flow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def create_gru_architecture_diagram():
    """Create a comprehensive GRU/RNN architecture diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Title
    ax.text(10, 19.5, 'Hierarchical GRU Architecture for NeuroToken Processing', 
            fontsize=20, fontweight='bold', ha='center')
    
    # 1. Input Layer - NeuroTokens
    input_box = FancyBboxPatch((0.5, 16), 19, 2, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightblue', 
                               edgecolor='navy', 
                               linewidth=2)
    ax.add_patch(input_box)
    
    ax.text(10, 17.5, 'Input: Enhanced NeuroTokens (496 features)', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Token type breakdown
    ax.text(2, 16.5, '• Level Tokens (0-9): 131 features', fontsize=10, ha='left')
    ax.text(2, 16.2, '• Delta Tokens (0-6): 131 features', fontsize=10, ha='left')
    ax.text(2, 15.9, '• Harmonized Features: 131 features', fontsize=10, ha='left')
    ax.text(2, 15.6, '• Region Embeddings: 99 features', fontsize=10, ha='left')
    ax.text(2, 15.3, '• Metadata: 4 features', fontsize=10, ha='left')
    
    # 2. Token Embedding Layer
    embed_box = FancyBboxPatch((0.5, 13.5), 19, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightgreen', 
                               edgecolor='darkgreen', 
                               linewidth=2)
    ax.add_patch(embed_box)
    
    ax.text(10, 14.5, 'Token Embedding Layer', 
            fontsize=14, fontweight='bold', ha='center')
    ax.text(10, 13.8, 'Converts discrete tokens to dense vectors (hidden_dim=128)', 
            fontsize=10, ha='center')
    
    # 3. Hierarchical Processing
    # Session-level GRU
    session_gru = FancyBboxPatch((2, 11), 6, 1.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='lightcoral', 
                                 edgecolor='darkred', 
                                 linewidth=2)
    ax.add_patch(session_gru)
    
    ax.text(5, 12, 'Session-Level GRU', 
            fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 11.3, 'Processes tokens within each session', 
            fontsize=9, ha='center')
    
    # Subject-level GRU
    subject_gru = FancyBboxPatch((12, 11), 6, 1.5, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='gold', 
                                 edgecolor='orange', 
                                 linewidth=2)
    ax.add_patch(subject_gru)
    
    ax.text(15, 12, 'Subject-Level GRU', 
            fontsize=12, fontweight='bold', ha='center')
    ax.text(15, 12.7, 'Aggregates sessions per subject', 
            fontsize=9, ha='center')
    
    # 4. Attention Mechanism
    attention_box = FancyBboxPatch((7, 9), 6, 1.5, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor='plum', 
                                   edgecolor='purple', 
                                   linewidth=2)
    ax.add_patch(attention_box)
    
    ax.text(10, 10.5, 'Multi-Head Attention', 
            fontsize=12, fontweight='bold', ha='center')
    ax.text(10, 9.8, 'Weights sessions by importance', 
            fontsize=9, ha='center')
    
    # 5. Output Layers
    # Classification head
    class_box = FancyBboxPatch((7, 7), 6, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightyellow', 
                               edgecolor='goldenrod', 
                               linewidth=2)
    ax.add_patch(class_box)
    
    ax.text(10, 8.5, 'Classification Head', 
            fontsize=12, fontweight='bold', ha='center')
    ax.text(10, 7.8, 'Dense layers → Softmax → Alzheimer\'s Prediction', 
            fontsize=9, ha='center')
    
    # 6. Data Flow Arrows
    # Input to embedding
    arrow1 = FancyArrowPatch((10, 16), (10, 15), 
                             arrowstyle='->', mutation_scale=20, 
                             color='navy', linewidth=3)
    ax.add_patch(arrow1)
    
    # Embedding to session GRU
    arrow2 = FancyArrowPatch((5, 13.5), (5, 12.5), 
                             arrowstyle='->', mutation_scale=20, 
                             color='darkgreen', linewidth=3)
    ax.add_patch(arrow2)
    
    # Embedding to subject GRU
    arrow3 = FancyArrowPatch((15, 13.5), (15, 12.5), 
                             arrowstyle='->', mutation_scale=20, 
                             color='darkgreen', linewidth=3)
    ax.add_patch(arrow3)
    
    # Session GRU to attention
    arrow4 = FancyArrowPatch((5, 11), (10, 10.5), 
                             arrowstyle='->', mutation_scale=20, 
                             color='darkred', linewidth=3)
    ax.add_patch(arrow4)
    
    # Subject GRU to attention
    arrow5 = FancyArrowPatch((15, 11), (10, 10.5), 
                             arrowstyle='->', mutation_scale=20, 
                             color='orange', linewidth=3)
    ax.add_patch(arrow5)
    
    # Attention to classification
    arrow6 = FancyArrowPatch((10, 9), (10, 8.5), 
                             arrowstyle='->', mutation_scale=20, 
                             color='purple', linewidth=3)
    ax.add_patch(arrow6)
    
    # 7. Model Parameters
    param_box = FancyBboxPatch((0.5, 5), 19, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightgray', 
                               edgecolor='gray', 
                               linewidth=2)
    ax.add_patch(param_box)
    
    ax.text(10, 6.5, 'Model Architecture Details', 
            fontsize=14, fontweight='bold', ha='center')
    
    ax.text(2, 5.8, '• Hidden Dimensions: 128', fontsize=10, ha='left')
    ax.text(2, 5.5, '• GRU Layers: 2 (session + subject)', fontsize=10, ha='left')
    ax.text(2, 5.2, '• Attention Heads: 4', fontsize=10, ha='left')
    ax.text(2, 4.9, '• Dropout: 0.3', fontsize=10, ha='left')
    
    ax.text(10, 5.8, '• Total Parameters: ~2.1M', fontsize=10, ha='center')
    ax.text(10, 5.5, '• Input Features: 496', fontsize=10, ha='center')
    ax.text(10, 5.2, '• Output Classes: 2 (Control vs Alzheimer\'s)', fontsize=10, ha='center')
    
    ax.text(18, 5.8, '• Learning Rate: 1e-3', fontsize=10, ha='right')
    ax.text(18, 5.5, '• Batch Size: 16', fontsize=10, ha='right')
    ax.text(18, 5.2, '• Optimizer: AdamW', fontsize=10, ha='right')
    ax.text(18, 4.9, '• Loss: CrossEntropy', fontsize=10, ha='right')
    
    # 8. Data Processing Flow
    flow_box = FancyBboxPatch((0.5, 2.5), 19, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightcyan', 
                              edgecolor='teal', 
                              linewidth=2)
    ax.add_patch(flow_box)
    
    ax.text(10, 4, 'Data Processing Flow', 
            fontsize=14, fontweight='bold', ha='center')
    
    ax.text(2, 3.3, '1. Token Extraction → 2. Embedding → 3. Session GRU →', fontsize=10, ha='left')
    ax.text(2, 3, '4. Subject GRU → 5. Attention → 6. Classification', fontsize=10, ha='left')
    
    ax.text(10, 3.3, '• Sessions per subject: 1-4', fontsize=10, ha='center')
    ax.text(10, 3, '• Sequence length: Variable (max 4 sessions)', fontsize=10, ha='center')
    
    ax.text(18, 3.3, '• Memory efficient', fontsize=10, ha='right')
    ax.text(18, 3, '• Handles missing data', fontsize=10, ha='right')
    
    return fig

def create_gru_cell_diagram():
    """Create a detailed GRU cell diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(8, 11.5, 'GRU Cell Architecture', 
            fontsize=18, fontweight='bold', ha='center')
    
    # GRU Cell Box
    gru_box = FancyBboxPatch((2, 2), 12, 8, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightblue', 
                              edgecolor='navy', 
                              linewidth=3)
    ax.add_patch(gru_box)
    
    # Input
    ax.text(1, 9, 'x_t', fontsize=14, fontweight='bold', ha='center')
    ax.text(1, 8.5, '(Current\nInput)', fontsize=10, ha='center')
    
    # Previous hidden state
    ax.text(1, 5, 'h_{t-1}', fontsize=14, fontweight='bold', ha='center')
    ax.text(1, 4.5, '(Previous\nHidden)', fontsize=10, ha='center')
    
    # Reset Gate
    reset_gate = FancyBboxPatch((3, 7.5), 2, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightcoral', 
                                edgecolor='darkred', 
                                linewidth=2)
    ax.add_patch(reset_gate)
    ax.text(4, 8, 'Reset\nGate (r)', fontsize=10, fontweight='bold', ha='center')
    
    # Update Gate
    update_gate = FancyBboxPatch((3, 5.5), 2, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='lightgreen', 
                                 edgecolor='darkgreen', 
                                 linewidth=2)
    ax.add_patch(update_gate)
    ax.text(4, 6, 'Update\nGate (z)', fontsize=10, fontweight='bold', ha='center')
    
    # Candidate Hidden State
    candidate_box = FancyBboxPatch((3, 3.5), 2, 1, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor='gold', 
                                   edgecolor='orange', 
                                   linewidth=2)
    ax.add_patch(candidate_box)
    ax.text(4, 4, 'Candidate\nHidden (h̃)', fontsize=10, fontweight='bold', ha='center')
    
    # Output Hidden State
    output_box = FancyBboxPatch((11, 5.5), 2, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='plum', 
                                edgecolor='purple', 
                                linewidth=2)
    ax.add_patch(output_box)
    ax.text(12, 6, 'Output\nHidden (h_t)', fontsize=10, fontweight='bold', ha='center')
    
    # Arrows
    # Input to gates
    arrow1 = FancyArrowPatch((1.5, 9), (3, 8), 
                             arrowstyle='->', mutation_scale=15, 
                             color='navy', linewidth=2)
    ax.add_patch(arrow1)
    
    arrow2 = FancyArrowPatch((1.5, 9), (3, 6), 
                             arrowstyle='->', mutation_scale=15, 
                             color='navy', linewidth=2)
    ax.add_patch(arrow2)
    
    arrow3 = FancyArrowPatch((1.5, 9), (3, 4), 
                             arrowstyle='->', mutation_scale=15, 
                             color='navy', linewidth=2)
    ax.add_patch(arrow3)
    
    # Previous hidden to gates
    arrow4 = FancyArrowPatch((1.5, 5), (3, 8), 
                             arrowstyle='->', mutation_scale=15, 
                             color='darkred', linewidth=2)
    ax.add_patch(arrow4)
    
    arrow5 = FancyArrowPatch((1.5, 5), (3, 6), 
                             arrowstyle='->', mutation_scale=15, 
                             color='darkred', linewidth=2)
    ax.add_patch(arrow5)
    
    # Gates to candidate
    arrow6 = FancyArrowPatch((5, 7.5), (5, 4.5), 
                             arrowstyle='->', mutation_scale=15, 
                             color='darkred', linewidth=2)
    ax.add_patch(arrow6)
    
    # Candidate to output
    arrow7 = FancyArrowPatch((5, 4), (11, 6), 
                             arrowstyle='->', mutation_scale=15, 
                             color='orange', linewidth=2)
    ax.add_patch(arrow7)
    
    # Previous hidden to output
    arrow8 = FancyArrowPatch((1.5, 5), (11, 6), 
                             arrowstyle='->', mutation_scale=15, 
                             color='darkred', linewidth=2)
    ax.add_patch(arrow8)
    
    # Output
    ax.text(15, 6, 'h_t', fontsize=14, fontweight='bold', ha='center')
    ax.text(15, 5.5, '(Current\nHidden)', fontsize=10, ha='center')
    
    # Mathematical formulas
    ax.text(8, 1.5, 'GRU Equations:', fontsize=14, fontweight='bold', ha='center')
    ax.text(8, 1, 'r_t = σ(W_r · [x_t, h_{t-1}] + b_r)', fontsize=12, ha='center')
    ax.text(8, 0.5, 'z_t = σ(W_z · [x_t, h_{t-1}] + b_z)', fontsize=12, ha='center')
    ax.text(8, 0, 'h̃_t = tanh(W_h · [x_t, r_t ⊙ h_{t-1}] + b_h)', fontsize=12, ha='center')
    
    return fig

def create_attention_mechanism_diagram():
    """Create a multi-head attention mechanism diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(9, 13.5, 'Multi-Head Attention Mechanism', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Input representations
    input_box = FancyBboxPatch((1, 11.5), 16, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightblue', 
                               edgecolor='navy', 
                               linewidth=2)
    ax.add_patch(input_box)
    ax.text(9, 12, 'Input Representations (h_1, h_2, ..., h_n)', 
            fontsize=12, fontweight='bold', ha='center')
    
    # Attention heads
    for i in range(4):
        x_pos = 2 + i * 3.5
        head_box = FancyBboxPatch((x_pos, 9), 2.5, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='lightcoral', 
                                  edgecolor='darkred', 
                                  linewidth=2)
        ax.add_patch(head_box)
        ax.text(x_pos + 1.25, 9.75, f'Head {i+1}', 
                fontsize=10, fontweight='bold', ha='center')
        
        # Arrow from input to head
        arrow = FancyArrowPatch((9, 11.5), (x_pos + 1.25, 10.5), 
                               arrowstyle='->', mutation_scale=15, 
                               color='navy', linewidth=2)
        ax.add_patch(arrow)
    
    # Attention computation
    for i in range(4):
        x_pos = 2 + i * 3.5
        
        # Query, Key, Value boxes
        q_box = FancyBboxPatch((x_pos, 7), 0.8, 0.8, 
                               boxstyle="round,pad=0.05", 
                               facecolor='lightgreen', 
                               edgecolor='darkgreen', 
                               linewidth=1)
        ax.add_patch(q_box)
        ax.text(x_pos + 0.4, 7.4, 'Q', fontsize=8, fontweight='bold', ha='center')
        
        k_box = FancyBboxPatch((x_pos + 0.9, 7), 0.8, 0.8, 
                               boxstyle="round,pad=0.05", 
                               facecolor='gold', 
                               edgecolor='orange', 
                               linewidth=1)
        ax.add_patch(k_box)
        ax.text(x_pos + 1.3, 7.4, 'K', fontsize=8, fontweight='bold', ha='center')
        
        v_box = FancyBboxPatch((x_pos + 1.8, 7), 0.8, 0.8, 
                               boxstyle="round,pad=0.05", 
                               facecolor='plum', 
                               edgecolor='purple', 
                               linewidth=1)
        ax.add_patch(v_box)
        ax.text(x_pos + 2.2, 7.4, 'V', fontsize=8, fontweight='bold', ha='center')
        
        # Attention formula
        ax.text(x_pos + 1.25, 6.2, 'Attention(Q,K,V)', fontsize=8, ha='center')
        ax.text(x_pos + 1.25, 5.8, '= softmax(QK^T/√d_k)V', fontsize=8, ha='center')
    
    # Concatenation
    concat_box = FancyBboxPatch((7, 4), 4, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightcyan', 
                                edgecolor='teal', 
                                linewidth=2)
    ax.add_patch(concat_box)
    ax.text(9, 4.5, 'Concatenate All Heads', 
            fontsize=12, fontweight='bold', ha='center')
    
    # Arrows from heads to concat
    for i in range(4):
        x_pos = 2 + i * 3.5
        arrow = FancyArrowPatch((x_pos + 1.25, 9), (9, 5), 
                               arrowstyle='->', mutation_scale=15, 
                               color='darkred', linewidth=2)
        ax.add_patch(arrow)
    
    # Output projection
    output_box = FancyBboxPatch((7, 2), 4, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightyellow', 
                                edgecolor='goldenrod', 
                                linewidth=2)
    ax.add_patch(output_box)
    ax.text(9, 2.5, 'Output Projection', 
            fontsize=12, fontweight='bold', ha='center')
    
    # Arrow from concat to output
    arrow = FancyArrowPatch((9, 4), (9, 3), 
                           arrowstyle='->', mutation_scale=15, 
                           color='teal', linewidth=2)
    ax.add_patch(arrow)
    
    # Final output
    ax.text(9, 1, 'Attended Output', fontsize=12, fontweight='bold', ha='center')
    
    # Mathematical explanation
    ax.text(9, 0.5, 'Multi-Head Attention allows the model to focus on different aspects of the input simultaneously', 
            fontsize=10, ha='center', style='italic')
    
    return fig

def create_data_flow_diagram():
    """Create a data flow diagram showing how neurotokens move through the model"""
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(10, 11.5, 'NeuroToken Data Flow Through Hierarchical GRU', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Sample data flow
    # Subject 1
    subject1_box = FancyBboxPatch((0.5, 9), 3, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='lightblue', 
                                  edgecolor='navy', 
                                  linewidth=2)
    ax.add_patch(subject1_box)
    ax.text(2, 9.75, 'Subject 1', fontsize=12, fontweight='bold', ha='center')
    
    # Sessions for subject 1
    for i in range(3):
        session_box = FancyBboxPatch((5 + i * 1.5, 9), 1, 1.5, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor='lightgreen', 
                                    edgecolor='darkgreen', 
                                    linewidth=1)
        ax.add_patch(session_box)
        ax.text(5.5 + i * 1.5, 9.75, f'S{i+1}', fontsize=10, fontweight='bold', ha='center')
        
        # Arrow from subject to session
        arrow = FancyArrowPatch((3.5, 9.75), (5 + i * 1.5, 9.75), 
                               arrowstyle='->', mutation_scale=10, 
                               color='navy', linewidth=1)
        ax.add_patch(arrow)
    
    # Subject 2
    subject2_box = FancyBboxPatch((0.5, 7), 3, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='lightcoral', 
                                  edgecolor='darkred', 
                                  linewidth=2)
    ax.add_patch(subject2_box)
    ax.text(2, 7.75, 'Subject 2', fontsize=12, fontweight='bold', ha='center')
    
    # Sessions for subject 2
    for i in range(2):
        session_box = FancyBboxPatch((5 + i * 1.5, 7), 1, 1.5, 
                                    boxstyle="round,pad=0.05", 
                                    facecolor='lightgreen', 
                                    edgecolor='darkgreen', 
                                    linewidth=1)
        ax.add_patch(session_box)
        ax.text(5.5 + i * 1.5, 7.75, f'S{i+1}', fontsize=10, fontweight='bold', ha='center')
        
        # Arrow from subject to session
        arrow = FancyArrowPatch((3.5, 7.75), (5 + i * 1.5, 7.75), 
                               arrowstyle='->', mutation_scale=10, 
                               color='darkred', linewidth=1)
        ax.add_patch(arrow)
    
    # Token processing
    token_box = FancyBboxPatch((10, 8), 3, 2, 
                               boxstyle="round,pad=0.1", 
                               facecolor='gold', 
                               edgecolor='orange', 
                               linewidth=2)
    ax.add_patch(token_box)
    ax.text(11.5, 9.5, 'Token\nProcessing', fontsize=12, fontweight='bold', ha='center')
    ax.text(11.5, 8.5, '• Embedding\n• GRU layers\n• Attention', fontsize=9, ha='center')
    
    # Arrows from sessions to token processing
    for i in range(3):
        arrow = FancyArrowPatch((8 + i * 1.5, 9.75), (10, 9), 
                               arrowstyle='->', mutation_scale=10, 
                               color='darkgreen', linewidth=1)
        ax.add_patch(arrow)
    
    for i in range(2):
        arrow = FancyArrowPatch((8 + i * 1.5, 7.75), (10, 8), 
                               arrowstyle='->', mutation_scale=10, 
                               color='darkgreen', linewidth=1)
        ax.add_patch(arrow)
    
    # Output
    output_box = FancyBboxPatch((15, 8), 3, 2, 
                                boxstyle="round,pad=0.1", 
                                facecolor='plum', 
                                edgecolor='purple', 
                                linewidth=2)
    ax.add_patch(output_box)
    ax.text(16.5, 9.5, 'Classification\nOutput', fontsize=12, fontweight='bold', ha='center')
    ax.text(16.5, 8.5, '• Alzheimer\'s\n• Control\n• Probability', fontsize=9, ha='center')
    
    # Arrow from token processing to output
    arrow = FancyArrowPatch((13, 9), (15, 9), 
                           arrowstyle='->', mutation_scale=15, 
                           color='orange', linewidth=2)
    ax.add_patch(arrow)
    
    # Data dimensions
    dim_box = FancyBboxPatch((0.5, 4), 19, 2, 
                             boxstyle="round,pad=0.1", 
                             facecolor='lightgray', 
                             edgecolor='gray', 
                             linewidth=2)
    ax.add_patch(dim_box)
    
    ax.text(10, 5.5, 'Data Dimensions and Flow', 
            fontsize=14, fontweight='bold', ha='center')
    
    ax.text(2, 4.8, 'Input: 496 features per session', fontsize=10, ha='left')
    ax.text(2, 4.5, 'Embedding: 128 dimensions', fontsize=10, ha='left')
    ax.text(2, 4.2, 'GRU Hidden: 128 dimensions', fontsize=10, ha='left')
    
    ax.text(10, 4.8, 'Sessions per subject: Variable (1-4)', fontsize=10, ha='center')
    ax.text(10, 4.5, 'Total subjects: 149', fontsize=10, ha='center')
    ax.text(10, 4.2, 'Total sessions: 345', fontsize=10, ha='center')
    
    ax.text(18, 4.8, 'Output: 2 classes', fontsize=10, ha='right')
    ax.text(18, 4.5, 'Loss: Cross-entropy', fontsize=10, ha='right')
    ax.text(18, 4.2, 'Optimizer: AdamW', fontsize=10, ha='right')
    
    # Processing steps
    steps_box = FancyBboxPatch((0.5, 1), 19, 2, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightcyan', 
                               edgecolor='teal', 
                               linewidth=2)
    ax.add_patch(steps_box)
    
    ax.text(10, 2.5, 'Processing Steps', 
            fontsize=14, fontweight='bold', ha='center')
    
    ax.text(2, 1.8, '1. Token Extraction (MRI → Features → Tokens)', fontsize=10, ha='left')
    ax.text(2, 1.5, '2. Session Processing (GRU per session)', fontsize=10, ha='left')
    ax.text(2, 1.2, '3. Subject Aggregation (GRU across sessions)', fontsize=10, ha='left')
    
    ax.text(10, 1.8, '4. Attention Weighting (Focus on important sessions)', fontsize=10, ha='center')
    ax.text(10, 1.5, '5. Classification (Dense layers + Softmax)', fontsize=10, ha='center')
    ax.text(10, 1.2, '6. Output (Alzheimer\'s probability)', fontsize=10, ha='center')
    
    ax.text(18, 1.8, 'Memory Efficient', fontsize=10, ha='right')
    ax.text(18, 1.5, 'Handles Missing Data', fontsize=10, ha='right')
    ax.text(18, 1.2, 'Temporal Modeling', fontsize=10, ha='right')
    
    return fig

def main():
    """Main function to create all visualizations"""
    
    print("🎨 Creating GRU/RNN Architecture Visualizations...")
    
    # 1. Main architecture diagram
    print("📊 Creating main architecture diagram...")
    fig1 = create_gru_architecture_diagram()
    fig1.savefig('gru_architecture_overview.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: gru_architecture_overview.png")
    
    # 2. GRU cell diagram
    print("🧠 Creating GRU cell diagram...")
    fig2 = create_gru_cell_diagram()
    fig2.savefig('gru_cell_detailed.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: gru_cell_detailed.png")
    
    # 3. Attention mechanism diagram
    print("👁️ Creating attention mechanism diagram...")
    fig3 = create_attention_mechanism_diagram()
    fig3.savefig('attention_mechanism.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: attention_mechanism.png")
    
    # 4. Data flow diagram
    print("🔄 Creating data flow diagram...")
    fig4 = create_data_flow_diagram()
    fig4.savefig('neurotoken_data_flow.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: neurotoken_data_flow.png")
    
    print("\n🎯 GRU/RNN Architecture Visualization Complete!")
    print("📁 Generated files:")
    print("   • gru_architecture_overview.png - Complete model architecture")
    print("   • gru_cell_detailed.png - Individual GRU cell details")
    print("   • attention_mechanism.png - Multi-head attention explanation")
    print("   • neurotoken_data_flow.png - Data flow through the model")
    
    # Show the plots
    plt.show()
    
    print("\n✨ All visualizations saved! Check the PNG files for detailed GRU/RNN architecture.")

if __name__ == "__main__":
    main() 