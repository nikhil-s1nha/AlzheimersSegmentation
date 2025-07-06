#!/usr/bin/env python3
"""
Utility functions for FreeSurfer parser testing and data validation
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

def create_sample_data(subjects_dir: str, num_subjects: int = 5):
    """
    Create sample FreeSurfer output files for testing.
    
    Args:
        subjects_dir: Directory to create sample data in
        num_subjects: Number of sample subjects to create
    """
    subjects_path = Path(subjects_dir)
    subjects_path.mkdir(exist_ok=True)
    
    # Sample region names and typical values
    aseg_regions = [
        "Left-Hippocampus", "Right-Hippocampus", "Left-Amygdala", "Right-Amygdala",
        "Left-Thalamus", "Right-Thalamus", "Left-Caudate", "Right-Caudate",
        "Left-Putamen", "Right-Putamen", "Left-Pallidum", "Right-Pallidum",
        "Left-Accumbens", "Right-Accumbens", "Left-VentralDC", "Right-VentralDC"
    ]
    
    aparc_regions = [
        "bankssts", "caudal anterior cingulate", "caudal middle frontal",
        "cuneus", "entorhinal", "fusiform", "inferior parietal",
        "inferior temporal", "isthmus cingulate", "lateral occipital",
        "lateral orbitofrontal", "lingual", "medial orbitofrontal",
        "middle temporal", "parahippocampal", "paracentral",
        "pars opercularis", "pars orbitalis", "pars triangularis",
        "pericalcarine", "postcentral", "posterior cingulate",
        "precentral", "precuneus", "rostral anterior cingulate",
        "rostral middle frontal", "superior frontal", "superior parietal",
        "superior temporal", "supramarginal", "frontal pole", "temporal pole",
        "transverse temporal", "insula"
    ]
    
    for i in range(num_subjects):
        subject_id = f"sub-{i+1:04d}"
        subject_path = subjects_path / subject_id / "stats"
        subject_path.mkdir(parents=True, exist_ok=True)
        
        # Create aseg.stats
        aseg_file = subject_path / "aseg.stats"
        with open(aseg_file, 'w') as f:
            f.write("# Measure BrainSeg, BrainSegVol, BrainSegVolNotVent, BrainSegVolNotVentSurf\n")
            f.write("# BrainSeg  BrainSegVol  BrainSegVolNotVent  BrainSegVolNotVentSurf\n")
            f.write("  BrainSeg  1234567.0  1234567.0  1234567.0\n")
            f.write("# ColHeaders  Index  SegId  NVoxels  Volume_mm3  StructName  normMean  normStdDev  normMin  normMax\n")
            f.write("# ColHeaders  Index  SegId  NVoxels  Volume_mm3  StructName  normMean  normStdDev  normMin  normMax\n")
            
            for j, region in enumerate(aseg_regions):
                # Generate realistic volume values with some variation
                base_volume = np.random.normal(2000, 500)  # Typical brain region volumes
                volume = max(100, base_volume)  # Ensure positive
                f.write(f"  {j+1:3d}  {j+1:3d}  {int(volume/2):6d}  {volume:8.1f}  {region:20s}  0.0  0.0  0.0  0.0\n")
        
        # Create lh.aparc.stats
        lh_aparc_file = subject_path / "lh.aparc.stats"
        with open(lh_aparc_file, 'w') as f:
            f.write("# Table of FreeSurfer cortical parcellation statistics\n")
            f.write("# ColHeaders  StructName  NumVert  SurfArea  GrayVol  ThickAvg  ThickStd  MeanCurv  GausCurv  FoldInd  CurvInd\n")
            
            for j, region in enumerate(aparc_regions):
                # Generate realistic values
                thickness = np.random.normal(2.5, 0.3)  # Typical cortical thickness
                surface_area = np.random.normal(1500, 300)  # Typical surface area
                thickness = max(1.0, min(4.0, thickness))  # Clamp to reasonable range
                surface_area = max(100, surface_area)  # Ensure positive
                
                f.write(f"  {region:25s}  {np.random.randint(100, 1000):6d}  {surface_area:8.1f}  {surface_area*thickness:8.1f}  {thickness:7.3f}  {thickness*0.1:7.3f}  {np.random.normal(0, 0.1):8.3f}  {np.random.normal(0, 0.01):8.3f}  {np.random.randint(1, 10):8d}  {np.random.randint(1, 10):8d}\n")
        
        # Create rh.aparc.stats
        rh_aparc_file = subject_path / "rh.aparc.stats"
        with open(rh_aparc_file, 'w') as f:
            f.write("# Table of FreeSurfer cortical parcellation statistics\n")
            f.write("# ColHeaders  StructName  NumVert  SurfArea  GrayVol  ThickAvg  ThickStd  MeanCurv  GausCurv  FoldInd  CurvInd\n")
            
            for j, region in enumerate(aparc_regions):
                # Generate realistic values (slightly different from left hemisphere)
                thickness = np.random.normal(2.5, 0.3)
                surface_area = np.random.normal(1500, 300)
                thickness = max(1.0, min(4.0, thickness))
                surface_area = max(100, surface_area)
                
                f.write(f"  {region:25s}  {np.random.randint(100, 1000):6d}  {surface_area:8.1f}  {surface_area*thickness:8.1f}  {thickness:7.3f}  {thickness*0.1:7.3f}  {np.random.normal(0, 0.1):8.3f}  {np.random.normal(0, 0.01):8.3f}  {np.random.randint(1, 10):8d}  {np.random.randint(1, 10):8d}\n")
    
    print(f"Created sample data for {num_subjects} subjects in {subjects_dir}")

def validate_neurotokens(neurotokens_file: str) -> Dict[str, Any]:
    """
    Validate the generated NeuroTokens file.
    
    Args:
        neurotokens_file: Path to the NeuroTokens JSON file
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'file_exists': False,
        'valid_json': False,
        'num_subjects': 0,
        'total_tokens': 0,
        'token_format_valid': True,
        'errors': []
    }
    
    try:
        with open(neurotokens_file, 'r') as f:
            data = json.load(f)
        
        validation_results['file_exists'] = True
        validation_results['valid_json'] = True
        validation_results['num_subjects'] = len(data)
        
        for subject, tokens in data.items():
            validation_results['total_tokens'] += len(tokens)
            
            for token in tokens:
                # Check token format: [Region]: feature_type=value, z=Z
                if not (':' in token and '=' in token and 'z=' in token):
                    validation_results['token_format_valid'] = False
                    validation_results['errors'].append(f"Invalid token format: {token}")
        
        return validation_results
        
    except FileNotFoundError:
        validation_results['errors'].append(f"File not found: {neurotokens_file}")
        return validation_results
    except json.JSONDecodeError as e:
        validation_results['errors'].append(f"Invalid JSON: {e}")
        return validation_results

def analyze_neurotokens(neurotokens_file: str) -> Dict[str, Any]:
    """
    Analyze the generated NeuroTokens for insights.
    
    Args:
        neurotokens_file: Path to the NeuroTokens JSON file
        
    Returns:
        Dictionary with analysis results
    """
    with open(neurotokens_file, 'r') as f:
        data = json.load(f)
    
    analysis = {
        'total_subjects': len(data),
        'total_tokens': sum(len(tokens) for tokens in data.values()),
        'tokens_per_subject': {},
        'feature_types': set(),
        'regions': set(),
        'z_score_stats': {'min': float('inf'), 'max': float('-inf'), 'mean': 0, 'std': 0}
    }
    
    all_z_scores = []
    
    for subject, tokens in data.items():
        analysis['tokens_per_subject'][subject] = len(tokens)
        
        for token in tokens:
            # Extract feature type and z-score
            parts = token.split(': ')[1].split(', ')
            for part in parts:
                if part.startswith('z='):
                    z_score = float(part.split('=')[1])
                    all_z_scores.append(z_score)
                    analysis['z_score_stats']['min'] = min(analysis['z_score_stats']['min'], z_score)
                    analysis['z_score_stats']['max'] = max(analysis['z_score_stats']['max'], z_score)
                
                if '=' in part and not part.startswith('z='):
                    feature_type = part.split('=')[0]
                    analysis['feature_types'].add(feature_type)
            
            # Extract region name
            region = token.split(': ')[0]
            analysis['regions'].add(region)
    
    if all_z_scores:
        analysis['z_score_stats']['mean'] = np.mean(all_z_scores)
        analysis['z_score_stats']['std'] = np.std(all_z_scores)
    
    # Convert sets to lists for JSON serialization
    analysis['feature_types'] = list(analysis['feature_types'])
    analysis['regions'] = list(analysis['regions'])
    
    return analysis

def create_summary_report(neurotokens_file: str, output_file: Optional[str] = None):
    """
    Create a summary report of the NeuroTokens analysis.
    
    Args:
        neurotokens_file: Path to the NeuroTokens JSON file
        output_file: Optional output file for the report
    """
    validation = validate_neurotokens(neurotokens_file)
    analysis = analyze_neurotokens(neurotokens_file)
    
    report = f"""
# NeuroTokens Analysis Report

## Validation Results
- File exists: {validation['file_exists']}
- Valid JSON: {validation['valid_json']}
- Number of subjects: {validation['num_subjects']}
- Total tokens: {validation['total_tokens']}
- Token format valid: {validation['token_format_valid']}

## Analysis Results
- Total subjects: {analysis['total_subjects']}
- Total tokens: {analysis['total_tokens']}
- Average tokens per subject: {analysis['total_tokens'] / analysis['total_subjects']:.1f}
- Number of unique regions: {len(analysis['regions'])}
- Number of feature types: {len(analysis['feature_types'])}

## Feature Types
{chr(10).join(f"- {ft}" for ft in sorted(analysis['feature_types']))}

## Z-Score Statistics
- Minimum: {analysis['z_score_stats']['min']:.3f}
- Maximum: {analysis['z_score_stats']['max']:.3f}
- Mean: {analysis['z_score_stats']['mean']:.3f}
- Standard deviation: {analysis['z_score_stats']['std']:.3f}

## Sample Regions
{chr(10).join(f"- {region}" for region in sorted(analysis['regions'])[:10])}
... and {len(analysis['regions']) - 10} more regions

## Validation Errors
{chr(10).join(f"- {error}" for error in validation['errors']) if validation['errors'] else "- None"}
"""
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_file}")
    else:
        print(report)

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "create_sample":
            subjects_dir = sys.argv[2] if len(sys.argv) > 2 else "sample_subjects"
            num_subjects = int(sys.argv[3]) if len(sys.argv) > 3 else 5
            create_sample_data(subjects_dir, num_subjects)
        
        elif sys.argv[1] == "validate":
            neurotokens_file = sys.argv[2] if len(sys.argv) > 2 else "neurotokens.json"
            validation = validate_neurotokens(neurotokens_file)
            print(json.dumps(validation, indent=2))
        
        elif sys.argv[1] == "analyze":
            neurotokens_file = sys.argv[2] if len(sys.argv) > 2 else "neurotokens.json"
            analysis = analyze_neurotokens(neurotokens_file)
            print(json.dumps(analysis, indent=2))
        
        elif sys.argv[1] == "report":
            neurotokens_file = sys.argv[2] if len(sys.argv) > 2 else "neurotokens.json"
            output_file = sys.argv[3] if len(sys.argv) > 3 else None
            create_summary_report(neurotokens_file, output_file)
    
    else:
        print("Usage:")
        print("  python utils.py create_sample [subjects_dir] [num_subjects]")
        print("  python utils.py validate [neurotokens_file]")
        print("  python utils.py analyze [neurotokens_file]")
        print("  python utils.py report [neurotokens_file] [output_file]") 