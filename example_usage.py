#!/usr/bin/env python3
"""
Example usage of the FreeSurfer parser
"""

import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from freesurfer_parser import FreeSurferParser
from utils import create_sample_data, validate_neurotokens, create_summary_report

def main():
    """Example workflow demonstrating the FreeSurfer parser."""
    
    print("=== FreeSurfer NeuroTokens Generator - Example Usage ===\n")
    
    # Step 1: Create sample data
    print("1. Creating sample data...")
    sample_dir = "example_subjects"
    create_sample_data(sample_dir, num_subjects=3)
    print(f"   Created sample data in '{sample_dir}' directory\n")
    
    # Step 2: Initialize and run the parser
    print("2. Running FreeSurfer parser...")
    parser = FreeSurferParser(sample_dir)
    neurotokens = parser.process_all_subjects()
    
    if neurotokens:
        print(f"   Successfully processed {len(neurotokens)} subjects")
        print(f"   Generated {sum(len(tokens) for tokens in neurotokens.values())} total NeuroTokens\n")
        
        # Step 3: Save results
        print("3. Saving results...")
        parser.save_results(neurotokens, output_format='json')
        print("   Results saved to neurotokens.json and region_statistics.json\n")
        
        # Step 4: Validate results
        print("4. Validating results...")
        validation = validate_neurotokens("neurotokens.json")
        print(f"   Validation successful: {validation['valid_json']}")
        print(f"   Number of subjects: {validation['num_subjects']}")
        print(f"   Total tokens: {validation['total_tokens']}")
        print(f"   Format valid: {validation['token_format_valid']}\n")
        
        # Step 5: Generate summary report
        print("5. Generating summary report...")
        create_summary_report("neurotokens.json", "example_report.md")
        print("   Report saved to example_report.md\n")
        
        # Step 6: Show sample NeuroTokens
        print("6. Sample NeuroTokens:")
        for subject, tokens in neurotokens.items():
            print(f"\n   {subject}:")
            for token in tokens[:3]:  # Show first 3 tokens per subject
                print(f"     {token}")
            if len(tokens) > 3:
                print(f"     ... and {len(tokens) - 3} more tokens")
        
        print(f"\n=== Example completed successfully! ===")
        print(f"Check the following files:")
        print(f"  - neurotokens.json: Generated NeuroTokens")
        print(f"  - region_statistics.json: Region statistics")
        print(f"  - example_report.md: Summary report")
        
    else:
        print("   Error: No NeuroTokens were generated. Please check your data.")

if __name__ == "__main__":
    main() 