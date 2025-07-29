#!/usr/bin/env python3
"""
Test script to verify paths and show what the neurotoken extractor will process.
"""

import os
import re

# Configuration
SUBJECTS_DIR = "/Volumes/SEAGATE_NIKHIL/subjects"
OUTPUT_DIR = "/Volumes/SEAGATE_NIKHIL/neurotokens_project"

def test_paths():
    """Test if the required paths exist and show what will be processed"""
    print("=== Neurotoken Extractor Path Test ===\n")
    
    # Check if subjects directory exists
    print(f"1. Checking subjects directory: {SUBJECTS_DIR}")
    if os.path.exists(SUBJECTS_DIR):
        print("   ✓ Subjects directory exists")
        
        # List subject directories
        subject_dirs = [d for d in os.listdir(SUBJECTS_DIR) 
                       if os.path.isdir(os.path.join(SUBJECTS_DIR, d)) 
                       and d.startswith('OAS2_')]
        
        print(f"   Found {len(subject_dirs)} subject-session directories")
        
        if subject_dirs:
            print("   Sample directories:")
            for i, dir_name in enumerate(subject_dirs[:5]):
                print(f"     {i+1}. {dir_name}")
            
            if len(subject_dirs) > 5:
                print(f"     ... and {len(subject_dirs) - 5} more")
            
            # Check a sample directory structure
            sample_dir = subject_dirs[0]
            sample_path = os.path.join(SUBJECTS_DIR, sample_dir)
            stats_path = os.path.join(sample_path, 'stats')
            
            print(f"\n2. Checking sample directory structure: {sample_dir}")
            if os.path.exists(stats_path):
                print("   ✓ Stats directory exists")
                
                # Check for required stats files
                required_files = ['aseg.stats', 'lh.aparc.stats', 'rh.aparc.stats']
                for file_name in required_files:
                    file_path = os.path.join(stats_path, file_name)
                    if os.path.exists(file_path):
                        print(f"   ✓ {file_name} exists")
                    else:
                        print(f"   ✗ {file_name} missing")
            else:
                print("   ✗ Stats directory missing")
        else:
            print("   ✗ No OAS2_* directories found")
    else:
        print("   ✗ Subjects directory does not exist")
    
    # Check output directory
    print(f"\n3. Checking output directory: {OUTPUT_DIR}")
    if os.path.exists(OUTPUT_DIR):
        print("   ✓ Output directory exists")
    else:
        print("   ✗ Output directory does not exist (will be created)")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_paths() 