#!/usr/bin/env python3
"""
Visualize Temporal Information in Enhanced NeuroToken Model
Shows how time intervals between scans are captured and processed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_temporal_data():
    """Load and analyze temporal data from tokens"""
    print("📊 Loading temporal data from enhanced tokens...")
    
    # Load tokens
    with open('enhanced_tokens_new.json', 'r') as f:
        tokens = [json.loads(line) for line in f]
    
    # Convert to DataFrame
    df = pd.DataFrame(tokens)
    
    # Extract temporal information
    temporal_data = df[['subject_id', 'session', 'delta_t_bucket']].copy()
    
    print(f"✅ Loaded {len(temporal_data)} token sequences")
    return temporal_data

def analyze_temporal_patterns(temporal_data):
    """Analyze temporal patterns in the data"""
    print("\n🔍 Analyzing temporal patterns...")
    
    # Count delta_t_bucket distribution
    bucket_counts = temporal_data['delta_t_bucket'].value_counts().sort_index()
    
    # Define bucket meanings
    bucket_meanings = {
        0: "≤6 months",
        1: "6-12 months", 
        2: "12-24 months",
        3: ">24 months"
    }
    
    print("📅 Delta-t Bucket Distribution:")
    for bucket, count in bucket_counts.items():
        meaning = bucket_meanings.get(bucket, f"Unknown bucket {bucket}")
        print(f"   Bucket {bucket} ({meaning}): {count} sessions")
    
    # Analyze per-subject temporal patterns
    subject_temporal = temporal_data.groupby('subject_id').agg({
        'session': 'count',
        'delta_t_bucket': lambda x: list(x)
    }).reset_index()
    
    subject_temporal.columns = ['subject_id', 'session_count', 'temporal_sequence']
    
    print(f"\n📊 Subject Temporal Analysis:")
    print(f"   - Subjects with 1 session: {len(subject_temporal[subject_temporal['session_count'] == 1])}")
    print(f"   - Subjects with 2+ sessions: {len(subject_temporal[subject_temporal['session_count'] > 1])}")
    
    # Find subjects with multiple sessions
    multi_session = subject_temporal[subject_temporal['session_count'] > 1]
    
    if len(multi_session) > 0:
        print(f"\n🔄 Multi-session subjects temporal patterns:")
        for _, row in multi_session.head(10).iterrows():
            sessions = row['temporal_sequence']
            print(f"   {row['subject_id']}: {sessions} sessions")
    
    return bucket_counts, subject_temporal, bucket_meanings

def create_temporal_visualizations(bucket_counts, subject_temporal, bucket_meanings):
    """Create comprehensive temporal visualizations"""
    print("\n🎨 Creating temporal visualizations...")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Delta-t Bucket Distribution Pie Chart
    bucket_labels = [f"Bucket {b}\n({meaning})" for b, meaning in bucket_meanings.items() if b in bucket_counts.index]
    bucket_values = [bucket_counts[b] for b in bucket_counts.index]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    ax1.pie(bucket_values, labels=bucket_labels, autopct='%1.1f%%', 
            colors=colors[:len(bucket_values)], startangle=90)
    ax1.set_title('Distribution of Time Intervals Between Scans', fontsize=14, fontweight='bold')
    
    # 2. Session Count Distribution
    session_counts = subject_temporal['session_count'].value_counts().sort_index()
    bars = ax2.bar(session_counts.index, session_counts.values, 
                   color='#2E86AB', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, session_counts.values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                str(value), ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xlabel('Number of Sessions per Subject', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Subjects', fontsize=12, fontweight='bold')
    ax2.set_title('Session Distribution per Subject', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Temporal Bucket Heatmap
    # Create a heatmap showing temporal patterns across subjects
    multi_session = subject_temporal[subject_temporal['session_count'] > 1].head(20)
    
    if len(multi_session) > 0:
        # Create temporal matrix
        max_sessions = multi_session['session_count'].max()
        temporal_matrix = np.zeros((len(multi_session), max_sessions))
        
        for i, (_, row) in enumerate(multi_session.iterrows()):
            temporal_sequence = row['temporal_sequence']
            for j, bucket in enumerate(temporal_sequence):
                if j < max_sessions:
                    temporal_matrix[i, j] = bucket
        
        # Create heatmap
        im = ax3.imshow(temporal_matrix, cmap='YlOrRd', aspect='auto')
        ax3.set_xlabel('Session Order', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Subject Index', fontsize=12, fontweight='bold')
        ax3.set_title('Temporal Bucket Patterns Across Sessions', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Delta-t Bucket', fontsize=10)
        
        # Set tick labels
        ax3.set_xticks(range(max_sessions))
        ax3.set_xticklabels([f'S{i+1}' for i in range(max_sessions)])
        
        # Add bucket meaning annotations
        for bucket in range(4):
            if bucket in temporal_matrix:
                count = np.sum(temporal_matrix == bucket)
                ax3.text(0.02, 0.98 - bucket * 0.2, f'Bucket {bucket}: {bucket_meanings.get(bucket, "Unknown")}',
                        transform=ax3.transAxes, fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    else:
        ax3.text(0.5, 0.5, 'No multi-session subjects found', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        ax3.set_title('Temporal Bucket Patterns', fontsize=14, fontweight='bold')
    
    # 4. Temporal Information Flow Diagram
    ax4.axis('off')
    
    # Create a diagram showing how temporal information flows through the model
    diagram_elements = [
        ("MRI Scans\n(Time Series)", (0.2, 0.8)),
        ("Time Interval\nCalculation", (0.5, 0.8)),
        ("Delta-t Bucket\nAssignment", (0.8, 0.8)),
        ("Temporal\nEmbedding", (0.5, 0.5)),
        ("GRU Processing\nwith Attention", (0.5, 0.2))
    ]
    
    # Draw boxes and arrows
    for text, pos in diagram_elements:
        ax4.text(pos[0], pos[1], text, ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # Draw arrows
    arrow_props = dict(arrowstyle='->', color='red', lw=2)
    ax4.annotate('', xy=(0.35, 0.8), xytext=(0.15, 0.8), arrowprops=arrow_props)
    ax4.annotate('', xy=(0.65, 0.8), xytext=(0.45, 0.8), arrowprops=arrow_props)
    ax4.annotate('', xy=(0.5, 0.65), xytext=(0.8, 0.75), arrowprops=arrow_props)
    ax4.annotate('', xy=(0.5, 0.35), xytext=(0.5, 0.45), arrowprops=arrow_props)
    
    ax4.set_title('Temporal Information Flow in Model', fontsize=14, fontweight='bold')
    
    plt.suptitle('Enhanced NeuroToken Model - Temporal Information Analysis', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig('temporal_information_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Temporal analysis visualization saved as 'temporal_information_analysis.png'")

def create_temporal_summary_table(bucket_counts, bucket_meanings):
    """Create a summary table of temporal information"""
    print("\n📋 Creating temporal summary table...")
    
    # Create summary DataFrame
    summary_data = []
    for bucket, count in bucket_counts.items():
        meaning = bucket_meanings.get(bucket, f"Unknown bucket {bucket}")
        percentage = (count / bucket_counts.sum()) * 100
        summary_data.append({
            'Bucket': bucket,
            'Time Interval': meaning,
            'Count': count,
            'Percentage': f"{percentage:.1f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    summary_df.to_csv('temporal_summary.csv', index=False)
    print("✅ Temporal summary table saved as 'temporal_summary.csv'")
    
    # Display the table
    print("\n📊 Temporal Information Summary:")
    print(summary_df.to_string(index=False))
    
    return summary_df

def main():
    """Main function to analyze and visualize temporal information"""
    print("🕒 Analyzing Temporal Information in Enhanced NeuroToken Model")
    print("=" * 70)
    
    # Load data
    temporal_data = load_temporal_data()
    
    # Analyze patterns
    bucket_counts, subject_temporal, bucket_meanings = analyze_temporal_patterns(temporal_data)
    
    # Create visualizations
    create_temporal_visualizations(bucket_counts, subject_temporal, bucket_meanings)
    
    # Create summary table
    summary_df = create_temporal_summary_table(bucket_counts, bucket_meanings)
    
    print("\n🎉 Temporal analysis completed!")
    print("\n📁 Generated files:")
    print("   - temporal_information_analysis.png")
    print("   - temporal_summary.csv")
    
    print("\n🔑 Key Insights:")
    print("   1. The model captures time intervals between consecutive MRI scans")
    print("   2. Time intervals are bucketed into 4 categories: ≤6m, 6-12m, 12-24m, >24m")
    print("   3. Each session gets a 'delta_t_bucket' value indicating time since previous scan")
    print("   4. This temporal information is embedded and processed by the GRU model")
    print("   5. The model can learn temporal progression patterns in Alzheimer's disease")

if __name__ == "__main__":
    main() 