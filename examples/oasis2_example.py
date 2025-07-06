#!/usr/bin/env python3
"""
OASIS-2 NeuroTokens Generator - Complete Example
Demonstrates the full pipeline from FreeSurfer data to Transformer-ready NeuroTokens
"""

import os
import sys
import json
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from oasis2_neurotokens import OASIS2NeuroTokensProcessor
from oasis2_utils import (
    create_sample_oasis2_data, 
    validate_oasis2_data, 
    analyze_neurotokens_with_diagnosis,
    create_diagnosis_plots,
    prepare_transformer_dataset
)

def main():
    """Complete OASIS-2 NeuroTokens pipeline example."""
    
    print("=== OASIS-2 NeuroTokens Generator - Complete Example ===\n")
    
    # Configuration
    data_root = "sample_oasis2_data"
    num_subjects = 20  # Create 20 sample subjects
    output_format = "json"
    
    # Step 1: Create sample OASIS-2 data
    print("1. Creating sample OASIS-2 data...")
    create_sample_oasis2_data(data_root, num_subjects)
    print(f"   Created sample data in '{data_root}' directory\n")
    
    # Step 2: Validate data structure
    print("2. Validating data structure...")
    validation = validate_oasis2_data(data_root)
    
    if validation['errors']:
        print("   ❌ Validation errors found:")
        for error in validation['errors']:
            print(f"      - {error}")
        return
    else:
        print("   ✅ Data validation passed")
        print(f"   Found {validation['subjects_found']} subjects")
        print(f"   {validation['subjects_with_aseg']} subjects have aseg.stats")
        print(f"   {validation['subjects_with_aparc']} subjects have aparc.stats")
    
    if validation['warnings']:
        print("   ⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"      - {warning}")
    print()
    
    # Step 3: Initialize and run the OASIS-2 processor
    print("3. Processing FreeSurfer data and generating NeuroTokens...")
    processor = OASIS2NeuroTokensProcessor(data_root)
    neurotokens = processor.process_all_subjects()
    
    if not neurotokens:
        print("   ❌ No NeuroTokens were generated. Please check your data.")
        return
    
    print(f"   ✅ Successfully processed {len(neurotokens)} subjects")
    total_tokens = sum(len(tokens) for tokens in neurotokens.values())
    print(f"   Generated {total_tokens} total NeuroTokens")
    print()
    
    # Step 4: Save results
    print("4. Saving results...")
    processor.save_results(neurotokens, output_format)
    print("   ✅ Results saved to neurotokens_output/ directory")
    print()
    
    # Step 5: Analyze results with diagnosis
    print("5. Analyzing NeuroTokens with diagnosis information...")
    neurotokens_file = "neurotokens_output/all_neurotokens.json"
    diagnosis_file = "neurotokens_output/subjects_diagnosis_summary.csv"
    
    if Path(neurotokens_file).exists() and Path(diagnosis_file).exists():
        analysis = analyze_neurotokens_with_diagnosis(neurotokens_file, diagnosis_file)
        
        print("   📊 Analysis Results:")
        print(f"      Total subjects: {analysis['total_subjects']}")
        print(f"      Diagnosis distribution: {analysis['diagnosis_distribution']}")
        
        if analysis['z_score_stats_by_diagnosis']:
            print("      Z-score statistics by diagnosis:")
            for diagnosis, stats in analysis['z_score_stats_by_diagnosis'].items():
                print(f"        {diagnosis}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
        
        # Create visualization plots
        create_diagnosis_plots(analysis, "neurotokens_output/plots")
        print("      📈 Plots saved to neurotokens_output/plots/")
    else:
        print("   ⚠️  Could not find output files for analysis")
    print()
    
    # Step 6: Prepare Transformer dataset
    print("6. Preparing Transformer dataset...")
    if Path(neurotokens_file).exists() and Path(diagnosis_file).exists():
        dataset = prepare_transformer_dataset(neurotokens_file, diagnosis_file, max_length=150)
        
        print("   🤖 Transformer Dataset Ready:")
        print(f"      Training samples: {dataset['X_train'].shape[0]}")
        print(f"      Test samples: {dataset['X_test'].shape[0]}")
        print(f"      Sequence length: {dataset['max_length']}")
        print(f"      Vocabulary size: {dataset['vocab_size']}")
        print(f"      Number of classes: {dataset['num_classes']}")
        print(f"      Class names: {dataset['class_names']}")
        
        # Save dataset
        import pickle
        with open("neurotokens_output/transformer_dataset.pkl", "wb") as f:
            pickle.dump(dataset, f)
        print("      💾 Dataset saved to neurotokens_output/transformer_dataset.pkl")
    else:
        print("   ⚠️  Could not prepare Transformer dataset")
    print()
    
    # Step 7: Show sample NeuroTokens
    print("7. Sample NeuroTokens:")
    for i, (subject, tokens) in enumerate(neurotokens.items()):
        if i >= 3:  # Show first 3 subjects
            break
        print(f"\n   {subject}:")
        for j, token in enumerate(tokens[:5]):  # Show first 5 tokens per subject
            print(f"     {j+1}. {token}")
        if len(tokens) > 5:
            print(f"     ... and {len(tokens) - 5} more tokens")
    
    print(f"\n=== Example completed successfully! ===")
    print(f"📁 Check the following directories and files:")
    print(f"   - {data_root}/: Sample OASIS-2 data")
    print(f"   - neurotokens_output/: Generated NeuroTokens and analysis")
    print(f"   - neurotokens_output/plots/: Visualization plots")
    print(f"   - neurotokens_output/transformer_dataset.pkl: Ready for Transformer training")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Replace sample data with real OASIS-2 data")
    print(f"   2. Adjust configuration in oasis2_config.json if needed")
    print(f"   3. Run: python oasis2_neurotokens.py /path/to/real/oasis2/data")
    print(f"   4. Use the generated NeuroTokens for Transformer model training")

def show_usage_examples():
    """Show usage examples for the OASIS-2 processor."""
    
    print("\n=== Usage Examples ===\n")
    
    print("1. Process real OASIS-2 data:")
    print("   python oasis2_neurotokens.py /path/to/oasis2/data")
    print()
    
    print("2. Process with custom configuration:")
    print("   python oasis2_neurotokens.py /path/to/oasis2/data --config oasis2_config.json")
    print()
    
    print("3. Output in CSV format:")
    print("   python oasis2_neurotokens.py /path/to/oasis2/data --output-format csv")
    print()
    
    print("4. Create sample data:")
    print("   python oasis2_utils.py create_sample sample_data 50")
    print()
    
    print("5. Validate data structure:")
    print("   python oasis2_utils.py validate /path/to/oasis2/data")
    print()
    
    print("6. Analyze results:")
    print("   python oasis2_utils.py analyze neurotokens.json diagnosis.csv")
    print()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "usage":
        show_usage_examples()
    else:
        main() 