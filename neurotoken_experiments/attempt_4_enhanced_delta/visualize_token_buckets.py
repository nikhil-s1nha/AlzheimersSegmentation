#!/usr/bin/env python3
"""
Enhanced NeuroToken Bucket Visualization
Shows the discrete token mapping in a clean, organized way
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_tokens(file_path):
    """Load neurotokens from JSONL file (one JSON object per line)"""
    tokens = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                tokens.append(json.loads(line))
    return tokens

def analyze_token_buckets(tokens):
    """Analyze how tokens are mapped to discrete buckets"""
    
    # Convert to DataFrame
    df = pd.DataFrame(tokens)
    
    # Separate different token types
    level_cols = [col for col in df.columns if col.startswith('level_')]
    delta_cols = [col for col in df.columns if col.startswith('binned_delta_')]
    harmonized_cols = [col for col in df.columns if col.startswith('harmonized_')]
    region_cols = [col for col in df.columns if col.startswith('region_') and 'embedding' in col]
    
    # Analyze bucket distributions
    bucket_analysis = {
        'level_tokens': analyze_bucket_distribution(df[level_cols], 'Level Tokens'),
        'delta_tokens': analyze_bucket_distribution(df[delta_cols], 'Delta Tokens'),
        'harmonized_features': analyze_continuous_distribution(df[harmonized_cols], 'Harmonized Features'),
        'region_embeddings': analyze_region_embeddings(df[region_cols], 'Region Embeddings')
    }
    
    return bucket_analysis, df

def analyze_bucket_distribution(data, name):
    """Analyze discrete bucket distribution"""
    # Get unique values and their counts
    unique_values = np.unique(data.values)
    value_counts = data.values.flatten()
    
    # Count occurrences of each value
    bucket_counts = {}
    for val in unique_values:
        if not pd.isna(val):
            bucket_counts[val] = np.sum(value_counts == val)
    
    return {
        'name': name,
        'unique_values': sorted(unique_values),
        'bucket_counts': bucket_counts,
        'total_samples': len(data),
        'total_features': len(data.columns),
        'data_type': 'discrete_buckets'
    }

def analyze_continuous_distribution(data, name):
    """Analyze continuous feature distribution"""
    # Get basic statistics
    stats = data.describe()
    
    return {
        'name': name,
        'mean': data.mean().mean(),
        'std': data.std().mean(),
        'min': data.min().min(),
        'max': data.max().max(),
        'total_samples': len(data),
        'total_features': len(data.columns),
        'data_type': 'continuous'
    }

def analyze_region_embeddings(data, name):
    """Analyze region embedding distribution"""
    unique_values = np.unique(data.values)
    
    return {
        'name': name,
        'unique_values': sorted(unique_values),
        'total_samples': len(data),
        'total_features': len(data.columns),
        'data_type': 'region_embeddings'
    }

def create_bucket_visualization(bucket_analysis, df):
    """Create comprehensive bucket visualization"""
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Level Token Buckets (Discrete 0-9)
    ax1 = plt.subplot(3, 2, 1)
    level_data = bucket_analysis['level_tokens']
    buckets = list(level_data['bucket_counts'].keys())
    counts = list(level_data['bucket_counts'].values())
    
    bars = plt.bar(buckets, counts, color='skyblue', alpha=0.8, edgecolor='navy')
    plt.title(f'{level_data["name"]} Distribution\n(Discrete Buckets 0-9)', fontsize=14, fontweight='bold')
    plt.xlabel('Bucket Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(buckets)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=10)
    
    # 2. Delta Token Buckets (Discrete 0-6 with stable dead-zone)
    ax2 = plt.subplot(3, 2, 2)
    delta_data = bucket_analysis['delta_tokens']
    buckets = list(delta_data['bucket_counts'].keys())
    counts = list(delta_data['bucket_counts'].values())
    
    bars = plt.bar(buckets, counts, color='lightcoral', alpha=0.8, edgecolor='darkred')
    plt.title(f'{delta_data["name"]} Distribution\n(Discrete Buckets 0-6, Stable Dead-Zone)', fontsize=14, fontweight='bold')
    plt.xlabel('Bucket Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(buckets)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=10)
    
    # 3. Harmonized Features (Continuous Z-scores)
    ax3 = plt.subplot(3, 2, 3)
    harm_data = bucket_analysis['harmonized_features']
    
    # Sample some features for visualization
    sample_features = ['harmonized_Left-Hippocampus', 'harmonized_Right-Hippocampus', 
                      'harmonized_Left-Amygdala', 'harmonized_Right-Amygdala']
    
    for feature in sample_features:
        if feature in df.columns:
            plt.hist(df[feature].dropna(), bins=30, alpha=0.6, label=feature.split('_')[-1])
    
    plt.title(f'{harm_data["name"]} Distribution\n(Continuous Z-scores)', fontsize=14, fontweight='bold')
    plt.xlabel('Z-score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Region Embeddings (Discrete 0.0-0.99)
    ax4 = plt.subplot(3, 2, 4)
    region_data = bucket_analysis['region_embeddings']
    
    # Sample some region embeddings
    sample_regions = ['region_Left-Hippocampus_embedding', 'region_Right-Hippocampus_embedding',
                     'region_Left-Amygdala_embedding', 'region_Right-Amygdala_embedding']
    
    for region in sample_regions:
        if region in df.columns:
            plt.hist(df[region].dropna(), bins=20, alpha=0.6, label=region.split('_')[1])
    
    plt.title(f'{region_data["name"]} Distribution\n(Discrete 0.0-0.99)', fontsize=14, fontweight='bold')
    plt.xlabel('Embedding Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Token Type Summary Table
    ax5 = plt.subplot(3, 2, 5)
    ax5.axis('tight')
    ax5.axis('off')
    
    # Create summary table
    summary_data = []
    for token_type, data in bucket_analysis.items():
        if data['data_type'] == 'discrete_buckets':
            summary_data.append([
                data['name'],
                f"Buckets: {min(data['unique_values'])}-{max(data['unique_values'])}",
                f"{len(data['unique_values'])} unique values",
                f"{data['total_features']} features",
                f"{data['total_samples']} samples"
            ])
        elif data['data_type'] == 'continuous':
            summary_data.append([
                data['name'],
                f"Range: {data['min']:.2f} to {data['max']:.2f}",
                f"Mean: {data['mean']:.2f}",
                f"{data['total_features']} features",
                f"{data['total_samples']} samples"
            ])
        else:
            summary_data.append([
                data['name'],
                f"Values: {min(data['unique_values']):.2f}-{max(data['unique_values']):.2f}",
                f"{len(data['unique_values'])} unique values",
                f"{data['total_features']} features",
                f"{data['total_samples']} samples"
            ])
    
    table = ax5.table(cellText=summary_data,
                      colLabels=['Token Type', 'Value Range', 'Unique Values', 'Features', 'Samples'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.25, 0.25, 0.2, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(summary_data) + 1):
        for j in range(5):
            if i == 0:  # Header
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:  # Data rows
                table[(i, j)].set_facecolor('#E8F5E8' if i % 2 == 0 else 'white')
    
    plt.title('NeuroToken Summary', fontsize=16, fontweight='bold', pad=20)
    
    # 6. Sample Token Values
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    # Show sample token values
    sample_tokens = df.iloc[0]  # First sample
    
    sample_data = [
        ['Subject ID', sample_tokens['subject_id']],
        ['Session', str(sample_tokens['session'])],
        ['Site', sample_tokens['site']],
        ['Level Left-Hippocampus', str(sample_tokens['level_Left-Hippocampus'])],
        ['Delta Left-Hippocampus', str(sample_tokens['binned_delta_Left-Hippocampus'])],
        ['Harmonized Left-Hippocampus', f"{sample_tokens['harmonized_Left-Hippocampus']:.3f}"],
        ['Region Left-Hippocampus', f"{sample_tokens['region_Left-Hippocampus_embedding']:.3f}"],
        ['Delta-t Bucket', str(sample_tokens['delta_t_bucket'])]
    ]
    
    sample_table = ax6.table(cellText=sample_data,
                             colLabels=['Feature', 'Value'],
                             cellLoc='left',
                             loc='center',
                             colWidths=[0.6, 0.4])
    
    sample_table.auto_set_font_size(False)
    sample_table.set_fontsize(10)
    sample_table.scale(1, 2)
    
    # Style the sample table
    for i in range(len(sample_data) + 1):
        for j in range(2):
            if i == 0:  # Header
                sample_table[(i, j)].set_facecolor('#2196F3')
                sample_table[(i, j)].set_text_props(weight='bold', color='white')
            else:  # Data rows
                sample_table[(i, j)].set_facecolor('#E3F2FD' if i % 2 == 0 else 'white')
    
    plt.title('Sample Token Values', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def create_detailed_bucket_analysis(bucket_analysis, df):
    """Create detailed analysis of each token type"""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Level Token Feature Breakdown
    ax1 = axes[0, 0]
    level_cols = [col for col in df.columns if col.startswith('level_')]
    
    # Count non-zero values for each feature
    feature_counts = []
    feature_names = []
    for col in level_cols[:20]:  # Show first 20 features
        non_zero = (df[col] > 0).sum()
        feature_counts.append(non_zero)
        feature_names.append(col.replace('level_', '').replace('_', '\n')[:20])
    
    bars = ax1.barh(range(len(feature_names)), feature_counts, color='lightblue', alpha=0.8)
    ax1.set_yticks(range(len(feature_names)))
    ax1.set_yticklabels(feature_names, fontsize=8)
    ax1.set_xlabel('Non-Zero Samples', fontsize=12)
    ax1.set_title('Level Token Feature Distribution\n(First 20 Features)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Delta Token Feature Breakdown
    ax2 = axes[0, 1]
    delta_cols = [col for col in df.columns if col.startswith('binned_delta_')]
    
    # Count non-zero values for each feature
    feature_counts = []
    feature_names = []
    for col in delta_cols[:20]:  # Show first 20 features
        non_zero = (df[col] > 0).sum()
        feature_counts.append(non_zero)
        feature_names.append(col.replace('binned_delta_', '').replace('_', '\n')[:20])
    
    bars = ax2.barh(range(len(feature_names)), feature_counts, color='lightcoral', alpha=0.8)
    ax2.set_yticks(range(len(feature_names)))
    ax2.set_yticklabels(feature_names, fontsize=8)
    ax2.set_xlabel('Non-Zero Samples', fontsize=12)
    ax2.set_title('Delta Token Feature Distribution\n(First 20 Features)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Token Value Heatmap (Sample)
    ax3 = axes[1, 0]
    
    # Sample a few subjects and features
    sample_subjects = df['subject_id'].unique()[:5]
    sample_features = ['level_Left-Hippocampus', 'level_Right-Hippocampus', 
                      'binned_delta_Left-Hippocampus', 'binned_delta_Right-Hippocampus',
                      'harmonized_Left-Hippocampus', 'harmonized_Right-Hippocampus']
    
    sample_data = df[df['subject_id'].isin(sample_subjects)][sample_features]
    
    im = ax3.imshow(sample_data.values, cmap='viridis', aspect='auto')
    ax3.set_xticks(range(len(sample_features)))
    ax3.set_xticklabels([f.replace('_', '\n') for f in sample_features], rotation=45, ha='right', fontsize=8)
    ax3.set_yticks(range(len(sample_subjects)))
    ax3.set_yticklabels(sample_subjects, fontsize=8)
    ax3.set_title('Token Value Heatmap\n(Sample Subjects & Features)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Token Value', fontsize=10)
    
    # 4. Token Type Comparison
    ax4 = axes[1, 1]
    
    # Compare token types
    token_types = ['Level Tokens', 'Delta Tokens', 'Harmonized Features', 'Region Embeddings']
    feature_counts = [
        len([col for col in df.columns if col.startswith('level_')]),
        len([col for col in df.columns if col.startswith('binned_delta_')]),
        len([col for col in df.columns if col.startswith('harmonized_')]),
        len([col for col in df.columns if col.startswith('region_') and 'embedding' in col])
    ]
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    bars = ax4.bar(token_types, feature_counts, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar, count in zip(bars, feature_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(feature_counts)*0.01,
                f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax4.set_ylabel('Number of Features', fontsize=12)
    ax4.set_title('Feature Count by Token Type', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    """Main function to create visualizations"""
    
    print("🧠 Loading neurotokens...")
    try:
        tokens = load_tokens('enhanced_tokens.json')
        print(f"✅ Loaded {len(tokens)} token sequences")
    except FileNotFoundError:
        print("❌ File 'enhanced_tokens.json' not found!")
        print("💡 Make sure you're in the correct directory or run the token extractor first")
        return
    
    print("📊 Analyzing token buckets...")
    bucket_analysis, df = analyze_token_buckets(tokens)
    
    print("🎨 Creating visualizations...")
    
    # Create main bucket visualization
    fig1 = create_bucket_visualization(bucket_analysis, df)
    fig1.savefig('token_buckets_overview.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: token_buckets_overview.png")
    
    # Create detailed analysis
    fig2 = create_detailed_bucket_analysis(bucket_analysis, df)
    fig2.savefig('token_buckets_detailed.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: token_buckets_detailed.png")
    
    # Print summary
    print("\n📋 NEUROTOKEN BUCKET SUMMARY:")
    print("=" * 50)
    
    for token_type, data in bucket_analysis.items():
        print(f"\n🔹 {data['name']}:")
        if data['data_type'] == 'discrete_buckets':
            print(f"   • Type: Discrete buckets")
            print(f"   • Range: {min(data['unique_values'])} to {max(data['unique_values'])}")
            print(f"   • Unique values: {len(data['unique_values'])}")
            print(f"   • Features: {data['total_features']}")
            print(f"   • Samples: {data['total_samples']}")
        elif data['data_type'] == 'continuous':
            print(f"   • Type: Continuous Z-scores")
            print(f"   • Range: {data['min']:.3f} to {data['max']:.3f}")
            print(f"   • Mean: {data['mean']:.3f}")
            print(f"   • Features: {data['total_features']}")
            print(f"   • Samples: {data['total_samples']}")
        else:
            print(f"   • Type: Region embeddings")
            print(f"   • Range: {min(data['unique_values']):.3f} to {max(data['unique_values']):.3f}")
            print(f"   • Features: {data['total_features']}")
            print(f"   • Samples: {data['total_samples']}")
    
    print(f"\n🎯 Total Features: {sum(data['total_features'] for data in bucket_analysis.values())}")
    print(f"📊 Total Samples: {len(tokens)}")
    
    plt.show()
    print("\n✨ Visualization complete! Check the generated PNG files.")

if __name__ == "__main__":
    main() 