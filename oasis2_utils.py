#!/usr/bin/env python3
"""
Utility functions for OASIS-2 FreeSurfer data processing
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import nibabel as nib

def create_sample_oasis2_data(data_root: str, num_subjects: int = 10):
    """
    Create sample OASIS-2 data structure for testing.
    
    Args:
        data_root: Root directory to create sample data
        num_subjects: Number of sample subjects to create
    """
    data_path = Path(data_root)
    data_path.mkdir(exist_ok=True)
    
    # Create subjects directory
    subjects_dir = data_path / "subjects"
    subjects_dir.mkdir(exist_ok=True)
    
    # Sample region names
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
    
    # Create clinical data
    clinical_data = []
    
    for i in range(num_subjects):
        subject_id = f"sub-{i+1:04d}"
        subject_path = subjects_dir / subject_id / "stats"
        subject_path.mkdir(parents=True, exist_ok=True)
        
        # Generate clinical data
        age = np.random.randint(60, 90)
        sex = np.random.choice(['M', 'F'])
        
        # Generate diagnosis with realistic distribution
        if i < num_subjects * 0.4:  # 40% normal
            cdr_score = 0.0
            diagnosis = "Normal"
            mmse = np.random.randint(25, 31)
        elif i < num_subjects * 0.7:  # 30% MCI
            cdr_score = 0.5
            diagnosis = "MCI"
            mmse = np.random.randint(20, 27)
        else:  # 30% AD
            cdr_score = np.random.choice([1.0, 2.0])
            diagnosis = "AD"
            mmse = np.random.randint(10, 24)
        
        clinical_data.append({
            'subject_id': subject_id,
            'age': age,
            'sex': sex,
            'cdr_score': cdr_score,
            'diagnosis': diagnosis,
            'mmse': mmse
        })
        
        # Create aseg.stats
        aseg_file = subject_path / "aseg.stats"
        with open(aseg_file, 'w') as f:
            f.write("# Measure BrainSeg, BrainSegVol, BrainSegVolNotVent, BrainSegVolNotVentSurf\n")
            f.write("# BrainSeg  BrainSegVol  BrainSegVolNotVent  BrainSegVolNotVentSurf\n")
            f.write("  BrainSeg  1234567.0  1234567.0  1234567.0\n")
            f.write("# ColHeaders  Index  SegId  NVoxels  Volume_mm3  StructName  normMean  normStdDev  normMin  normMax\n")
            
            for j, region in enumerate(aseg_regions):
                # Generate realistic volume values with some variation based on diagnosis
                base_volume = np.random.normal(2000, 500)
                if diagnosis == "AD":
                    base_volume *= 0.8  # AD subjects have smaller volumes
                elif diagnosis == "MCI":
                    base_volume *= 0.9  # MCI subjects have slightly smaller volumes
                
                volume = max(100, base_volume)
                f.write(f"  {j+1:3d}  {j+1:3d}  {int(volume/2):6d}  {volume:8.1f}  {region:20s}  0.0  0.0  0.0  0.0\n")
        
        # Create lh.aparc.stats
        lh_aparc_file = subject_path / "lh.aparc.stats"
        with open(lh_aparc_file, 'w') as f:
            f.write("# Table of FreeSurfer cortical parcellation statistics\n")
            f.write("# ColHeaders  StructName  NumVert  SurfArea  GrayVol  ThickAvg  ThickStd  MeanCurv  GausCurv  FoldInd  CurvInd\n")
            
            for j, region in enumerate(aparc_regions):
                # Generate realistic values with diagnosis-based variation
                thickness = np.random.normal(2.5, 0.3)
                surface_area = np.random.normal(1500, 300)
                
                if diagnosis == "AD":
                    thickness *= 0.85  # AD subjects have thinner cortex
                    surface_area *= 0.9  # AD subjects have smaller surface area
                elif diagnosis == "MCI":
                    thickness *= 0.92
                    surface_area *= 0.95
                
                thickness = max(1.0, min(4.0, thickness))
                surface_area = max(100, surface_area)
                
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
                
                if diagnosis == "AD":
                    thickness *= 0.85
                    surface_area *= 0.9
                elif diagnosis == "MCI":
                    thickness *= 0.92
                    surface_area *= 0.95
                
                thickness = max(1.0, min(4.0, thickness))
                surface_area = max(100, surface_area)
                
                f.write(f"  {region:25s}  {np.random.randint(100, 1000):6d}  {surface_area:8.1f}  {surface_area*thickness:8.1f}  {thickness:7.3f}  {thickness*0.1:7.3f}  {np.random.normal(0, 0.1):8.3f}  {np.random.normal(0, 0.01):8.3f}  {np.random.randint(1, 10):8d}  {np.random.randint(1, 10):8d}\n")
    
    # Save clinical data
    clinical_df = pd.DataFrame(clinical_data)
    clinical_file = data_path / "clinical_data.csv"
    clinical_df.to_csv(clinical_file, index=False)
    
    print(f"Created sample OASIS-2 data for {num_subjects} subjects in {data_root}")
    print(f"Clinical data saved to {clinical_file}")

def validate_oasis2_data(data_root: str) -> Dict[str, Any]:
    """
    Validate OASIS-2 data structure and files.
    
    Args:
        data_root: Root directory containing OASIS-2 data
        
    Returns:
        Dictionary with validation results
    """
    data_path = Path(data_root)
    validation_results = {
        'data_root_exists': False,
        'subjects_dir_exists': False,
        'clinical_data_exists': False,
        'subjects_found': 0,
        'subjects_with_stats': 0,
        'subjects_with_aseg': 0,
        'subjects_with_aparc': 0,
        'errors': [],
        'warnings': []
    }
    
    # Check data root
    if not data_path.exists():
        validation_results['errors'].append(f"Data root {data_root} does not exist")
        return validation_results
    
    validation_results['data_root_exists'] = True
    
    # Check subjects directory
    subjects_dir = data_path / "subjects"
    if not subjects_dir.exists():
        validation_results['errors'].append(f"Subjects directory {subjects_dir} does not exist")
        return validation_results
    
    validation_results['subjects_dir_exists'] = True
    
    # Check clinical data
    clinical_file = data_path / "clinical_data.csv"
    if clinical_file.exists():
        validation_results['clinical_data_exists'] = True
    else:
        validation_results['warnings'].append(f"Clinical data file {clinical_file} not found")
    
    # Check subjects
    subjects = []
    for item in subjects_dir.iterdir():
        if item.is_dir() and item.name.startswith('sub-'):
            subjects.append(item.name)
    
    validation_results['subjects_found'] = len(subjects)
    
    # Check stats files for each subject
    for subject in subjects:
        subject_path = subjects_dir / subject
        stats_dir = subject_path / "stats"
        
        if stats_dir.exists():
            validation_results['subjects_with_stats'] += 1
            
            # Check aseg.stats
            aseg_file = stats_dir / "aseg.stats"
            if aseg_file.exists():
                validation_results['subjects_with_aseg'] += 1
            else:
                validation_results['warnings'].append(f"aseg.stats missing for {subject}")
            
            # Check aparc.stats
            lh_aparc = stats_dir / "lh.aparc.stats"
            rh_aparc = stats_dir / "rh.aparc.stats"
            if lh_aparc.exists() and rh_aparc.exists():
                validation_results['subjects_with_aparc'] += 1
            else:
                validation_results['warnings'].append(f"aparc.stats missing for {subject}")
        else:
            validation_results['errors'].append(f"stats directory missing for {subject}")
    
    return validation_results

def analyze_neurotokens_with_diagnosis(neurotokens_file: str, diagnosis_file: str) -> Dict[str, Any]:
    """
    Analyze NeuroTokens with respect to diagnosis information.
    
    Args:
        neurotokens_file: Path to NeuroTokens JSON file
        diagnosis_file: Path to diagnosis summary CSV file
        
    Returns:
        Dictionary with analysis results
    """
    # Load data
    with open(neurotokens_file, 'r') as f:
        neurotokens = json.load(f)
    
    diagnosis_df = pd.read_csv(diagnosis_file)
    
    analysis = {
        'total_subjects': len(neurotokens),
        'diagnosis_distribution': {},
        'tokens_by_diagnosis': {},
        'z_score_stats_by_diagnosis': {},
        'region_importance': {}
    }
    
    # Analyze diagnosis distribution
    diagnosis_counts = diagnosis_df['diagnosis'].value_counts()
    analysis['diagnosis_distribution'] = diagnosis_counts.to_dict()
    
    # Analyze tokens by diagnosis
    for diagnosis in diagnosis_df['diagnosis'].unique():
        diagnosis_subjects = diagnosis_df[diagnosis_df['diagnosis'] == diagnosis]['subject_id'].tolist()
        diagnosis_tokens = []
        
        for subject in diagnosis_subjects:
            if subject in neurotokens:
                diagnosis_tokens.extend(neurotokens[subject])
        
        analysis['tokens_by_diagnosis'][diagnosis] = {
            'count': len(diagnosis_tokens),
            'avg_per_subject': len(diagnosis_tokens) / len(diagnosis_subjects) if diagnosis_subjects else 0
        }
    
    # Analyze z-scores by diagnosis
    for diagnosis in diagnosis_df['diagnosis'].unique():
        diagnosis_subjects = diagnosis_df[diagnosis_df['diagnosis'] == diagnosis]['subject_id'].tolist()
        z_scores = []
        
        for subject in diagnosis_subjects:
            if subject in neurotokens:
                for token in neurotokens[subject]:
                    # Extract z-score from token
                    if 'z=' in token:
                        z_part = token.split('z=')[1].split(',')[0]
                        try:
                            z_score = float(z_part)
                            z_scores.append(z_score)
                        except ValueError:
                            continue
        
        if z_scores:
            analysis['z_score_stats_by_diagnosis'][diagnosis] = {
                'mean': np.mean(z_scores),
                'std': np.std(z_scores),
                'min': np.min(z_scores),
                'max': np.max(z_scores),
                'count': len(z_scores)
            }
    
    return analysis

def create_diagnosis_plots(analysis_results: Dict[str, Any], output_dir: str):
    """
    Create visualization plots for diagnosis analysis.
    
    Args:
        analysis_results: Results from analyze_neurotokens_with_diagnosis
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Diagnosis distribution
    plt.figure(figsize=(10, 6))
    diagnosis_counts = analysis_results['diagnosis_distribution']
    plt.bar(diagnosis_counts.keys(), diagnosis_counts.values())
    plt.title('Distribution of Diagnoses')
    plt.xlabel('Diagnosis')
    plt.ylabel('Number of Subjects')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / 'diagnosis_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Z-score distribution by diagnosis
    if analysis_results['z_score_stats_by_diagnosis']:
        plt.figure(figsize=(12, 8))
        
        diagnoses = list(analysis_results['z_score_stats_by_diagnosis'].keys())
        means = [analysis_results['z_score_stats_by_diagnosis'][d]['mean'] for d in diagnoses]
        stds = [analysis_results['z_score_stats_by_diagnosis'][d]['std'] for d in diagnoses]
        
        x_pos = np.arange(len(diagnoses))
        plt.bar(x_pos, means, yerr=stds, capsize=5)
        plt.title('Mean Z-Scores by Diagnosis')
        plt.xlabel('Diagnosis')
        plt.ylabel('Mean Z-Score')
        plt.xticks(x_pos, diagnoses, rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'z_scores_by_diagnosis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Tokens per subject by diagnosis
    plt.figure(figsize=(10, 6))
    diagnoses = list(analysis_results['tokens_by_diagnosis'].keys())
    avg_tokens = [analysis_results['tokens_by_diagnosis'][d]['avg_per_subject'] for d in diagnoses]
    
    plt.bar(diagnoses, avg_tokens)
    plt.title('Average Number of NeuroTokens per Subject by Diagnosis')
    plt.xlabel('Diagnosis')
    plt.ylabel('Average Tokens per Subject')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / 'tokens_per_subject_by_diagnosis.png', dpi=300, bbox_inches='tight')
    plt.close()

def prepare_transformer_dataset(neurotokens_file: str, diagnosis_file: str, 
                              max_length: int = 200, test_size: float = 0.2) -> Dict[str, Any]:
    """
    Prepare dataset for Transformer model training.
    
    Args:
        neurotokens_file: Path to NeuroTokens JSON file
        diagnosis_file: Path to diagnosis summary CSV file
        max_length: Maximum sequence length
        test_size: Fraction of data to use for testing
        
    Returns:
        Dictionary containing train/test splits and metadata
    """
    # Load data
    with open(neurotokens_file, 'r') as f:
        neurotokens = json.load(f)
    
    diagnosis_df = pd.read_csv(diagnosis_file)
    
    # Prepare sequences and labels
    sequences = []
    labels = []
    subject_ids = []
    
    for _, row in diagnosis_df.iterrows():
        subject_id = row['subject_id']
        if subject_id in neurotokens:
            # Get tokens for this subject
            tokens = neurotokens[subject_id]
            
            # For now, we'll use a simple tokenization (you can enhance this)
            # Convert tokens to a simple numeric representation
            sequence = []
            for token in tokens[:max_length]:
                # Simple hash-based tokenization
                token_id = hash(token) % 10000  # Simple hash to numeric
                sequence.append(token_id)
            
            # Pad sequence
            while len(sequence) < max_length:
                sequence.append(0)  # Pad token
            
            sequences.append(sequence)
            
            # Get label (CDR score or diagnosis)
            cdr_score = row.get('cdr_score', np.nan)
            if not np.isnan(cdr_score):
                labels.append(cdr_score)
            else:
                # Use diagnosis as categorical label
                diagnosis = row.get('diagnosis', 'Unknown')
                labels.append(diagnosis)
            
            subject_ids.append(subject_id)
    
    # Convert to numpy arrays
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    # Split data
    X_train, X_test, y_train, y_test, train_subjects, test_subjects = train_test_split(
        sequences, labels, subject_ids, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Create label encoder for categorical labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    dataset = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_encoded': y_train_encoded,
        'y_test_encoded': y_test_encoded,
        'train_subjects': train_subjects,
        'test_subjects': test_subjects,
        'label_encoder': label_encoder,
        'vocab_size': 10000,  # Based on our simple tokenization
        'max_length': max_length,
        'num_classes': len(label_encoder.classes_),
        'class_names': label_encoder.classes_.tolist()
    }
    
    return dataset

def parse_oasis_demographics(excel_path: str) -> pd.DataFrame:
    """Parse OASIS Longitudinal Demographics Excel file."""
    df = pd.read_excel(excel_path)
    df.columns = [c.strip() for c in df.columns]
    return df

def create_subject_metadata_files(df: pd.DataFrame, processed_root: str) -> None:
    """Create metadata.json for each subject."""
    grouped = df.groupby('Subject ID')
    for subj_id, group in grouped:
        subj_dir = os.path.join(processed_root, subj_id)
        os.makedirs(subj_dir, exist_ok=True)
        sessions = group.to_dict(orient='records')
        with open(os.path.join(subj_dir, 'metadata.json'), 'w') as f:
            json.dump({'subject_id': subj_id, 'sessions': sessions}, f, indent=2)

def create_master_index_csv(df: pd.DataFrame, processed_root: str, out_csv_path: str) -> None:
    """Create master index CSV with all sessions and metadata."""
    rows = []
    for _, row in df.iterrows():
        subj_id = row['Subject ID']
        visit = row['Visit']
        session_idx = visit.replace('MR', '')
        
        # Construct path to T1_avg.mgz
        t1_path = os.path.join(processed_root, subj_id, f'session_{session_idx}', 'T1_avg.mgz')
        
        # Check if file exists
        if os.path.exists(t1_path):
            row_dict = row.to_dict()
            row_dict['t1_path'] = t1_path
            rows.append(row_dict)
    
    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_csv_path, index=False)

def extract_freesurfer_stats(subject_dir: str) -> Dict:
    """Extract FreeSurfer statistics from aseg.stats and aparc.stats files."""
    stats = {}
    
    # aseg.stats - subcortical volumes
    aseg_path = os.path.join(subject_dir, 'stats', 'aseg.stats')
    if os.path.exists(aseg_path):
        aseg_stats = parse_aseg_stats(aseg_path)
        stats['aseg'] = aseg_stats
    
    # lh.aparc.stats - left hemisphere cortical
    lh_aparc_path = os.path.join(subject_dir, 'stats', 'lh.aparc.stats')
    if os.path.exists(lh_aparc_path):
        lh_stats = parse_aparc_stats(lh_aparc_path)
        stats['lh_aparc'] = lh_stats
    
    # rh.aparc.stats - right hemisphere cortical
    rh_aparc_path = os.path.join(subject_dir, 'stats', 'rh.aparc.stats')
    if os.path.exists(rh_aparc_path):
        rh_stats = parse_aparc_stats(rh_aparc_path)
        stats['rh_aparc'] = rh_stats
    
    return stats

def parse_aseg_stats(aseg_path: str) -> Dict:
    """Parse aseg.stats file for subcortical volumes."""
    stats = {}
    with open(aseg_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                region = parts[4]
                volume = float(parts[3])
                stats[region] = volume
    return stats

def parse_aparc_stats(aparc_path: str) -> Dict:
    """Parse aparc.stats file for cortical measurements."""
    stats = {}
    with open(aparc_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                region = parts[4]
                thickness = float(parts[2])
                area = float(parts[3])
                volume = float(parts[4])
                stats[region] = {
                    'thickness': thickness,
                    'area': area,
                    'volume': volume
                }
    return stats

def generate_neurotokens(fs_stats: Dict, subject_id: str, session: str) -> Dict:
    """Generate neurotokens from FreeSurfer statistics."""
    tokens = {
        'subject_id': subject_id,
        'session': session,
        'timestamp': pd.Timestamp.now().isoformat(),
        'regions': {}
    }
    
    # Subcortical regions (aseg)
    if 'aseg' in fs_stats:
        for region, volume in fs_stats['aseg'].items():
            tokens['regions'][f'aseg_{region}'] = {
                'volume': volume,
                'type': 'subcortical'
            }
    
    # Left hemisphere cortical regions
    if 'lh_aparc' in fs_stats:
        for region, metrics in fs_stats['lh_aparc'].items():
            tokens['regions'][f'lh_{region}'] = {
                'thickness': metrics['thickness'],
                'area': metrics['area'],
                'volume': metrics['volume'],
                'type': 'cortical_left'
            }
    
    # Right hemisphere cortical regions
    if 'rh_aparc' in fs_stats:
        for region, metrics in fs_stats['rh_aparc'].items():
            tokens['regions'][f'rh_{region}'] = {
                'thickness': metrics['thickness'],
                'area': metrics['area'],
                'volume': metrics['volume'],
                'type': 'cortical_right'
            }
    
    return tokens

def compute_z_scores(tokens_list: List[Dict], region: str, metric: str) -> List[float]:
    """Compute z-scores for a specific region and metric across subjects."""
    values = []
    for tokens in tokens_list:
        if region in tokens['regions'] and metric in tokens['regions'][region]:
            values.append(tokens['regions'][region][metric])
    
    if not values:
        return []
    
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    if std_val == 0:
        return [0.0] * len(values)
    
    z_scores = [(v - mean_val) / std_val for v in values]
    return z_scores

def save_neurotokens(tokens: Dict, output_path: str) -> None:
    """Save neurotokens to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(tokens, f, indent=2)

def load_neurotokens(input_path: str) -> Dict:
    """Load neurotokens from JSON file."""
    with open(input_path, 'r') as f:
        return json.load(f)

def create_transformer_dataset(tokens_list: List[Dict], labels: List[int], 
                             output_dir: str) -> None:
    """Create dataset for transformer training."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract features and create token sequences
    features = []
    for tokens in tokens_list:
        region_features = []
        for region_name, region_data in tokens['regions'].items():
            if isinstance(region_data, dict):
                # For cortical regions with multiple metrics
                feature_vector = [
                    region_data.get('volume', 0),
                    region_data.get('thickness', 0),
                    region_data.get('area', 0)
                ]
            else:
                # For subcortical regions with just volume
                feature_vector = [region_data]
            
            region_features.append({
                'region': region_name,
                'features': feature_vector
            })
        
        features.append(region_features)
    
    # Save processed data
    dataset = {
        'features': features,
        'labels': labels,
        'num_regions': len(features[0]) if features else 0,
        'feature_dim': len(features[0][0]['features']) if features and features[0] else 0
    }
    
    with open(os.path.join(output_dir, 'transformer_dataset.json'), 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Save metadata
    metadata = {
        'num_subjects': len(tokens_list),
        'num_classes': len(set(labels)),
        'class_distribution': {label: labels.count(label) for label in set(labels)}
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "create_sample":
            data_root = sys.argv[2] if len(sys.argv) > 2 else "sample_oasis2_data"
            num_subjects = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            create_sample_oasis2_data(data_root, num_subjects)
        
        elif sys.argv[1] == "validate":
            data_root = sys.argv[2] if len(sys.argv) > 2 else "sample_oasis2_data"
            validation = validate_oasis2_data(data_root)
            print(json.dumps(validation, indent=2))
        
        elif sys.argv[1] == "analyze":
            neurotokens_file = sys.argv[2] if len(sys.argv) > 2 else "neurotokens_output/all_neurotokens.json"
            diagnosis_file = sys.argv[3] if len(sys.argv) > 3 else "neurotokens_output/subjects_diagnosis_summary.csv"
            analysis = analyze_neurotokens_with_diagnosis(neurotokens_file, diagnosis_file)
            print(json.dumps(analysis, indent=2))
    
    else:
        print("Usage:")
        print("  python oasis2_utils.py create_sample [data_root] [num_subjects]")
        print("  python oasis2_utils.py validate [data_root]")
        print("  python oasis2_utils.py analyze [neurotokens_file] [diagnosis_file]") 