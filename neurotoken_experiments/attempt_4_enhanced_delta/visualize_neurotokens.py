#!/usr/bin/env python3
"""
Visualize Actual NeuroTokens
Show the real token values, distributions, and examples from the enhanced approach
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class NeuroTokenVisualizer:
    def __init__(self, token_file_path):
        self.token_file_path = token_file_path
        self.tokens_data = None
        self.load_tokens()
        
    def load_tokens(self):
        """Load the enhanced tokens from JSONL file"""
        print(f"Loading tokens from: {self.token_file_path}")
        
        tokens_list = []
        with open(self.token_file_path, 'r') as f:
            for line in f:
                tokens_list.append(json.loads(line.strip()))
        
        self.tokens_data = tokens_list
        print(f"Loaded {len(tokens_list)} token samples")
        
    def analyze_token_types(self):
        """Analyze the different types of tokens in the data"""
        if not self.tokens_data:
            print("No tokens loaded!")
            return
            
        # Get sample tokens to analyze structure
        sample_tokens = self.tokens_data[0]
        
        print("\n🔍 TOKEN STRUCTURE ANALYSIS:")
        print("=" * 50)
        
        # Categorize tokens by type
        token_categories = {
            'level_tokens': [],
            'delta_tokens': [],
            'harmonized_features': [],
            'region_embeddings': [],
            'metadata': []
        }
        
        for key, value in sample_tokens.items():
            if key.startswith('level_'):
                token_categories['level_tokens'].append((key, value))
            elif key.startswith('binned_delta_'):
                token_categories['delta_tokens'].append((key, value))
            elif key.startswith('harmonized_'):
                token_categories['harmonized_features'].append((key, value))
            elif key.startswith('region_') and key.endswith('_embedding'):
                token_categories['region_embeddings'].append((key, value))
            else:
                token_categories['metadata'].append((key, value))
        
        # Print summary
        for category, tokens in token_categories.items():
            print(f"{category.upper()}: {len(tokens)} tokens")
            if tokens:
                print(f"  Examples: {tokens[:3]}")
            print()
        
        return token_categories
    
    def visualize_token_distributions(self):
        """Create visualizations of token distributions"""
        if not self.tokens_data:
            print("No tokens loaded!")
            return
            
        print("📊 Creating token distribution visualizations...")
        
        # Extract all token values by type
        all_level_tokens = []
        all_delta_tokens = []
        all_harmonized = []
        all_region_emb = []
        
        for sample in self.tokens_data:
            for key, value in sample.items():
                if key.startswith('level_'):
                    all_level_tokens.append(value)
                elif key.startswith('binned_delta_'):
                    all_delta_tokens.append(value)
                elif key.startswith('harmonized_'):
                    all_harmonized.append(value)
                elif key.startswith('region_') and key.endswith('_embedding'):
                    all_region_emb.append(value)
        
        # Create distribution plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Level tokens distribution
        ax1.hist(all_level_tokens, bins=20, alpha=0.7, color='#4CAF50', edgecolor='black')
        ax1.set_title('Level Tokens Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Token Values')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Delta tokens distribution
        ax2.hist(all_delta_tokens, bins=20, alpha=0.7, color='#2196F3', edgecolor='black')
        ax2.set_title('Delta Tokens Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Token Values')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Harmonized features distribution
        ax3.hist(all_harmonized, bins=30, alpha=0.7, color='#FF9800', edgecolor='black')
        ax3.set_title('Harmonized Features Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Feature Values')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Region embeddings distribution
        ax4.hist(all_region_emb, bins=30, alpha=0.7, color='#9C27B0', edgecolor='black')
        ax4.set_title('Region Embeddings Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Embedding Values')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results_visualizations/token_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Token distributions saved!")
        
        # Print statistics
        print(f"\n📈 TOKEN STATISTICS:")
        print(f"Level tokens: {len(all_level_tokens)} values, range: {min(all_level_tokens)} to {max(all_level_tokens)}")
        print(f"Delta tokens: {len(all_delta_tokens)} values, range: {min(all_delta_tokens)} to {max(all_delta_tokens)}")
        print(f"Harmonized: {len(all_harmonized)} values, range: {min(all_harmonized):.3f} to {max(all_harmonized):.3f}")
        print(f"Region emb: {len(all_region_emb)} values, range: {min(all_region_emb):.3f} to {max(all_region_emb):.3f}")
        
        return {
            'level': all_level_tokens,
            'delta': all_delta_tokens,
            'harmonized': all_harmonized,
            'region_emb': all_region_emb
        }
    
    def visualize_token_examples(self):
        """Show specific examples of tokens"""
        if not self.tokens_data:
            print("No tokens loaded!")
            return
            
        print("\n🔍 TOKEN EXAMPLES:")
        print("=" * 50)
        
        # Show first few samples
        for i, sample in enumerate(self.tokens_data[:3]):
            print(f"\n📋 SAMPLE {i+1}:")
            print("-" * 30)
            
            # Group tokens by type
            level_tokens = {k: v for k, v in sample.items() if k.startswith('level_')}
            delta_tokens = {k: v for k, v in sample.items() if k.startswith('binned_delta_')}
            harmonized = {k: v for k, v in sample.items() if k.startswith('harmonized_')}
            region_emb = {k: v for k, v in sample.items() if k.startswith('region_') and k.endswith('_embedding')}
            metadata = {k: v for k, v in sample.items() if not any(k.startswith(prefix) for prefix in ['level_', 'binned_delta_', 'harmonized_', 'region_'])}
            
            print(f"Level tokens ({len(level_tokens)}): {dict(list(level_tokens.items())[:5])}")
            print(f"Delta tokens ({len(delta_tokens)}): {dict(list(delta_tokens.items())[:5])}")
            print(f"Harmonized ({len(harmonized)}): {dict(list(harmonized.items())[:5])}")
            print(f"Region emb ({len(region_emb)}): {dict(list(region_emb.items())[:5])}")
            print(f"Metadata: {metadata}")
    
    def visualize_token_heatmap(self):
        """Create a heatmap showing token patterns across samples"""
        if not self.tokens_data:
            print("No tokens loaded!")
            return
            
        print("\n🔥 Creating token pattern heatmap...")
        
        # Extract level and delta tokens for heatmap
        level_data = []
        delta_data = []
        
        for sample in self.tokens_data[:20]:  # Use first 20 samples for visualization
            level_row = []
            delta_row = []
            
            for key, value in sample.items():
                if key.startswith('level_'):
                    level_row.append(value)
                elif key.startswith('binned_delta_'):
                    delta_row.append(value)
            
            if level_row:
                level_data.append(level_row)
            if delta_row:
                delta_data.append(delta_row)
        
        # Create heatmaps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        if level_data:
            level_array = np.array(level_data)
            im1 = ax1.imshow(level_array, cmap='viridis', aspect='auto')
            ax1.set_title('Level Tokens Pattern (First 20 Samples)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Token Index')
            ax1.set_ylabel('Sample Index')
            plt.colorbar(im1, ax=ax1, label='Token Value')
        
        if delta_data:
            delta_array = np.array(delta_data)
            im2 = ax2.imshow(delta_array, cmap='plasma', aspect='auto')
            ax2.set_title('Delta Tokens Pattern (First 20 Samples)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Token Index')
            ax2.set_ylabel('Sample Index')
            plt.colorbar(im2, ax=ax2, label='Token Value')
        
        plt.tight_layout()
        plt.savefig('results_visualizations/token_patterns_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Token patterns heatmap saved!")
    
    def visualize_token_statistics(self):
        """Create comprehensive token statistics"""
        if not self.tokens_data:
            print("No tokens loaded!")
            return
            
        print("\n📊 Creating comprehensive token statistics...")
        
        # Analyze token statistics
        token_stats = {
            'total_samples': len(self.tokens_data),
            'token_counts': {},
            'value_ranges': {},
            'unique_values': {}
        }
        
        # Count tokens by type
        for sample in self.tokens_data:
            for key, value in sample.items():
                token_type = key.split('_')[0] if '_' in key else 'other'
                if token_type not in token_stats['token_counts']:
                    token_stats['token_counts'][token_type] = 0
                token_stats['token_counts'][token_type] += 1
                
                # Track value ranges
                if token_type not in token_stats['value_ranges']:
                    token_stats['value_ranges'][token_type] = {'min': float('inf'), 'max': float('-inf')}
                
                try:
                    val = float(value)
                    token_stats['value_ranges'][token_type]['min'] = min(token_stats['value_ranges'][token_type]['min'], val)
                    token_stats['value_ranges'][token_type]['max'] = max(token_stats['value_ranges'][token_type]['max'], val)
                except:
                    pass
        
        # Create statistics visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Token counts pie chart
        token_types = list(token_stats['token_counts'].keys())
        token_counts = list(token_stats['token_counts'].values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(token_types)))
        
        ax1.pie(token_counts, labels=token_types, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Token Distribution by Type', fontsize=14, fontweight='bold')
        
        # Token counts bar chart
        ax2.bar(token_types, token_counts, color=colors, alpha=0.8)
        ax2.set_title('Token Counts by Type', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Tokens')
        ax2.tick_params(axis='x', rotation=45)
        
        # Value ranges
        valid_ranges = {k: v for k, v in token_stats['value_ranges'].items() 
                       if v['min'] != float('inf') and v['max'] != float('-inf')}
        
        if valid_ranges:
            ranges = [(v['max'] - v['min']) for v in valid_ranges.values()]
            ax3.bar(valid_ranges.keys(), ranges, color=colors[:len(valid_ranges)], alpha=0.8)
            ax3.set_title('Value Ranges by Token Type', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Range (max - min)')
            ax3.tick_params(axis='x', rotation=45)
        
        # Sample overview
        ax4.text(0.1, 0.9, f'Total Samples: {token_stats["total_samples"]}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.8, f'Total Tokens: {sum(token_counts)}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.7, f'Token Types: {len(token_types)}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.6, f'Avg Tokens/Sample: {sum(token_counts)/token_stats["total_samples"]:.1f}', fontsize=12, transform=ax4.transAxes)
        ax4.set_title('Dataset Overview', fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig('results_visualizations/token_statistics_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Token statistics overview saved!")
        
        # Print summary
        print(f"\n📈 TOKEN STATISTICS SUMMARY:")
        print(f"Total samples: {token_stats['total_samples']}")
        print(f"Total tokens: {sum(token_counts)}")
        print(f"Token types: {len(token_types)}")
        print(f"Average tokens per sample: {sum(token_counts)/token_stats['total_samples']:.1f}")
        
        return token_stats
    
    def generate_all_visualizations(self):
        """Generate all token visualizations"""
        print("🎨 Generating comprehensive neurotoken visualizations...")
        
        # Ensure output directory exists
        os.makedirs('results_visualizations', exist_ok=True)
        
        # Analyze token structure
        self.analyze_token_types()
        
        # Generate visualizations
        self.visualize_token_distributions()
        self.visualize_token_examples()
        self.visualize_token_heatmap()
        self.visualize_token_statistics()
        
        print(f"\n🎉 All neurotoken visualizations saved to 'results_visualizations/' directory!")
        print("\n📊 Generated files:")
        print("  • token_distributions.png - Token value distributions")
        print("  • token_patterns_heatmap.png - Token pattern heatmaps")
        print("  • token_statistics_overview.png - Comprehensive token statistics")
        
        print("\n💡 These visualizations show:")
        print("  • Actual token values and their distributions")
        print("  • How tokens vary across different samples")
        print("  • Token patterns and relationships")
        print("  • Statistical properties of the neurotokens")

def main():
    """Main function to visualize neurotokens"""
    # Path to the enhanced tokens file
    token_file = "/Volumes/SEAGATE_NIKHIL/neurotokens_project/enhanced_attempt/enhanced_tokens.jsonl"
    
    if not os.path.exists(token_file):
        print(f"❌ Token file not found: {token_file}")
        print("Please ensure the enhanced tokens have been generated first.")
        return
    
    visualizer = NeuroTokenVisualizer(token_file)
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main() 