#!/usr/bin/env python3
"""
Test script for the FreeSurfer parser
"""

import os
import sys
import json
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from freesurfer_parser import FreeSurferParser
from utils import create_sample_data, validate_neurotokens

def test_sample_data_creation():
    """Test sample data creation."""
    print("Testing sample data creation...")
    
    test_dir = "test_subjects"
    create_sample_data(test_dir, num_subjects=2)
    
    # Check if files were created
    expected_files = [
        "test_subjects/sub-0001/stats/aseg.stats",
        "test_subjects/sub-0001/stats/lh.aparc.stats",
        "test_subjects/sub-0001/stats/rh.aparc.stats",
        "test_subjects/sub-0002/stats/aseg.stats",
        "test_subjects/sub-0002/stats/lh.aparc.stats",
        "test_subjects/sub-0002/stats/rh.aparc.stats"
    ]
    
    for file_path in expected_files:
        if not Path(file_path).exists():
            print(f"ERROR: Expected file {file_path} was not created")
            return False
    
    print("✓ Sample data creation test passed")
    return True

def test_parser_initialization():
    """Test parser initialization."""
    print("Testing parser initialization...")
    
    test_dir = "test_subjects"
    parser = FreeSurferParser(test_dir)
    
    if parser.subjects_dir != Path(test_dir):
        print("ERROR: Parser subjects directory not set correctly")
        return False
    
    print("✓ Parser initialization test passed")
    return True

def test_subject_discovery():
    """Test subject discovery."""
    print("Testing subject discovery...")
    
    test_dir = "test_subjects"
    parser = FreeSurferParser(test_dir)
    subjects = parser.find_subjects()
    
    expected_subjects = ["sub-0001", "sub-0002"]
    if set(subjects) != set(expected_subjects):
        print(f"ERROR: Expected subjects {expected_subjects}, got {subjects}")
        return False
    
    print("✓ Subject discovery test passed")
    return True

def test_neurotoken_generation():
    """Test NeuroToken generation."""
    print("Testing NeuroToken generation...")
    
    test_dir = "test_subjects"
    parser = FreeSurferParser(test_dir)
    neurotokens = parser.process_all_subjects()
    
    if not neurotokens:
        print("ERROR: No NeuroTokens were generated")
        return False
    
    if len(neurotokens) != 2:
        print(f"ERROR: Expected 2 subjects, got {len(neurotokens)}")
        return False
    
    # Check token format
    for subject, tokens in neurotokens.items():
        if not tokens:
            print(f"ERROR: No tokens generated for {subject}")
            return False
        
        for token in tokens:
            if not (':' in token and '=' in token and 'z=' in token):
                print(f"ERROR: Invalid token format: {token}")
                return False
    
    print("✓ NeuroToken generation test passed")
    return True

def test_output_saving():
    """Test output file saving."""
    print("Testing output file saving...")
    
    test_dir = "test_subjects"
    parser = FreeSurferParser(test_dir)
    neurotokens = parser.process_all_subjects()
    
    # Save results
    parser.save_results(neurotokens, output_format='json')
    
    # Check if files were created
    expected_files = [
        "test_subjects/neurotokens.json",
        "test_subjects/region_statistics.json"
    ]
    
    for file_path in expected_files:
        if not Path(file_path).exists():
            print(f"ERROR: Expected file {file_path} was not created")
            return False
    
    # Validate the JSON file
    validation = validate_neurotokens("test_subjects/neurotokens.json")
    if not validation['valid_json']:
        print(f"ERROR: Invalid JSON file: {validation['errors']}")
        return False
    
    print("✓ Output saving test passed")
    return True

def cleanup_test_files():
    """Clean up test files."""
    import shutil
    
    test_dirs = ["test_subjects", "example_subjects"]
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)
    
    # Remove output files
    output_files = ["neurotokens.json", "region_statistics.json", "example_report.md"]
    for file_path in output_files:
        if Path(file_path).exists():
            Path(file_path).unlink()

def main():
    """Run all tests."""
    print("=== FreeSurfer Parser Test Suite ===\n")
    
    tests = [
        test_sample_data_creation,
        test_parser_initialization,
        test_subject_discovery,
        test_neurotoken_generation,
        test_output_saving
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"ERROR: Test {test.__name__} failed with exception: {e}")
            print()
    
    print(f"=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")
    
    # Cleanup
    print("\nCleaning up test files...")
    cleanup_test_files()
    print("✓ Cleanup completed")

if __name__ == "__main__":
    main() 