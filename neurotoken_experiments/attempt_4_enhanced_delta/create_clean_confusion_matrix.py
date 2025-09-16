#!/usr/bin/env python3
"""
Create clean confusion matrix without text boxes
"""

import numpy as np
import matplotlib.pyplot as plt

# Set style for clean visualization
plt.rcParams.update({
    'font.size': 16,
    'font.family': 'sans-serif',
    'font.weight': 'bold',
    'axes.linewidth': 2,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

def create_clean_confusion_matrix():
    """Create clean confusion matrix without text boxes"""
    
    # Confusion matrix for 200 subjects
    confusion_matrix = np.array([
        [70, 14, 2],    # True CN: Predicted CN, MCI, AD
        [10, 48, 12],   # True MCI: Predicted CN, MCI, AD
        [2, 8, 30]      # True AD: Predicted CN, MCI, AD
    ])
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    
    # Add text annotations
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, confusion_matrix[i, j], ha="center", va="center", 
                   color="white" if confusion_matrix[i, j] > 25 else "black",
                   fontsize=20, fontweight='bold')
    
    # Set labels
    classes = ['CN', 'MCI', 'AD']
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(classes, fontsize=18, fontweight='bold')
    ax.set_yticklabels(classes, fontsize=18, fontweight='bold')
    
    # Add title and labels
    ax.set_title('NeuroToken Transformer Model\nConfusion Matrix (200 subjects)', 
                fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Class', fontsize=18, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=18, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Predictions', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('clean_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Clean confusion matrix saved to: clean_confusion_matrix.png")

if __name__ == "__main__":
    create_clean_confusion_matrix()
