#!/usr/bin/env python3
"""
FreeSurfer Output Parser for OASIS-2 Dataset
Generates NeuroTokens from aseg.stats and aparc.stats files
"""

import os
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FreeSurferParser:
    """Parser for FreeSurfer output files to generate NeuroTokens."""
    
    def __init__(self, subjects_dir: str):
        """
        Initialize the parser.
        
        Args:
            subjects_dir: Root directory containing subject folders
        """
        self.subjects_dir = Path(subjects_dir)
        self.subjects = []
        self.region_stats = {}  # Store mean/std for each region across subjects
        
    def find_subjects(self) -> List[str]:
        """Find all subject directories in the subjects directory."""
        subjects = []
        if not self.subjects_dir.exists():
            logger.error(f"Subjects directory {self.subjects_dir} does not exist")
            return subjects
            
        for item in self.subjects_dir.iterdir():
            if item.is_dir() and item.name.startswith('sub-'):
                subjects.append(item.name)
                
        logger.info(f"Found {len(subjects)} subjects: {subjects}")
        return subjects
    
    def read_aseg_stats(self, subject_path: Path) -> Dict[str, float]:
        """
        Read aseg.stats file and extract volume measurements.
        
        Args:
            subject_path: Path to subject directory
            
        Returns:
            Dictionary mapping region names to volumes
        """
        aseg_file = subject_path / "stats" / "aseg.stats"
        if not aseg_file.exists():
            logger.warning(f"aseg.stats not found for {subject_path.name}")
            return {}
            
        volumes = {}
        try:
            with open(aseg_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('Measure'):
                        parts = line.split()
                        if len(parts) >= 4:
                            region_name = parts[4]  # Region name is typically at index 4
                            volume = float(parts[3])  # Volume is typically at index 3
                            volumes[f"[{region_name}]"] = volume
        except Exception as e:
            logger.error(f"Error reading aseg.stats for {subject_path.name}: {e}")
            
        return volumes
    
    def read_aparc_stats(self, subject_path: Path, hemisphere: str) -> Dict[str, Dict[str, float]]:
        """
        Read aparc.stats file and extract thickness and surface area measurements.
        
        Args:
            subject_path: Path to subject directory
            hemisphere: 'lh' or 'rh'
            
        Returns:
            Dictionary mapping region names to thickness and surface area
        """
        aparc_file = subject_path / "stats" / f"{hemisphere}.aparc.stats"
        if not aparc_file.exists():
            logger.warning(f"{hemisphere}.aparc.stats not found for {subject_path.name}")
            return {}
            
        measurements = {}
        try:
            with open(aparc_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('Measure'):
                        parts = line.split()
                        if len(parts) >= 8:
                            region_name = parts[0]  # Region name is first column
                            thickness = float(parts[4])  # Thickness is typically at index 4
                            surface_area = float(parts[2])  # Surface area is typically at index 2
                            
                            # Add hemisphere prefix to region name
                            hem_prefix = "Left" if hemisphere == "lh" else "Right"
                            full_region_name = f"[{hem_prefix} {region_name}]"
                            
                            measurements[full_region_name] = {
                                'thickness': thickness,
                                'surface_area': surface_area
                            }
        except Exception as e:
            logger.error(f"Error reading {hemisphere}.aparc.stats for {subject_path.name}: {e}")
            
        return measurements
    
    def collect_all_measurements(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Collect measurements from all subjects.
        
        Returns:
            Dictionary mapping subject IDs to their measurements
        """
        self.subjects = self.find_subjects()
        all_measurements = {}
        
        for subject in self.subjects:
            subject_path = self.subjects_dir / subject
            logger.info(f"Processing subject: {subject}")
            
            # Read aseg.stats (volumes)
            volumes = self.read_aseg_stats(subject_path)
            
            # Read aparc.stats (thickness and surface area)
            lh_measurements = self.read_aparc_stats(subject_path, "lh")
            rh_measurements = self.read_aparc_stats(subject_path, "rh")
            
            # Combine all measurements
            subject_measurements = {}
            
            # Add volumes
            for region, volume in volumes.items():
                subject_measurements[region] = {'volume': volume}
            
            # Add left hemisphere measurements
            for region, measures in lh_measurements.items():
                if region in subject_measurements:
                    subject_measurements[region].update(measures)
                else:
                    subject_measurements[region] = measures
            
            # Add right hemisphere measurements
            for region, measures in rh_measurements.items():
                if region in subject_measurements:
                    subject_measurements[region].update(measures)
                else:
                    subject_measurements[region] = measures
            
            all_measurements[subject] = subject_measurements
            
        return all_measurements
    
    def compute_region_statistics(self, all_measurements: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compute mean and standard deviation for each region and measurement type.
        
        Args:
            all_measurements: Dictionary of all subject measurements
            
        Returns:
            Dictionary mapping regions to their statistics
        """
        region_stats = {}
        
        # Collect all unique regions and measurement types
        all_regions = set()
        measurement_types = set()
        
        for subject_data in all_measurements.values():
            for region, measurements in subject_data.items():
                all_regions.add(region)
                measurement_types.update(measurements.keys())
        
        # Compute statistics for each region and measurement type
        for region in all_regions:
            region_stats[region] = {}
            
            for measure_type in measurement_types:
                values = []
                
                for subject_data in all_measurements.values():
                    if region in subject_data and measure_type in subject_data[region]:
                        values.append(subject_data[region][measure_type])
                
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    region_stats[region][measure_type] = {
                        'mean': mean_val,
                        'std': std_val
                    }
        
        self.region_stats = region_stats
        return region_stats
    
    def generate_neurotokens(self, subject_measurements: Dict[str, Dict[str, float]], 
                           region_stats: Dict[str, Dict[str, Dict[str, float]]]) -> List[str]:
        """
        Generate NeuroTokens for a single subject.
        
        Args:
            subject_measurements: Measurements for the subject
            region_stats: Statistics for all regions
            
        Returns:
            List of NeuroToken strings
        """
        neurotokens = []
        
        for region, measurements in subject_measurements.items():
            if region not in region_stats:
                continue
                
            for measure_type, value in measurements.items():
                if measure_type not in region_stats[region]:
                    continue
                    
                stats = region_stats[region][measure_type]
                mean_val = stats['mean']
                std_val = stats['std']
                
                if std_val > 0:  # Avoid division by zero
                    z_score = (value - mean_val) / std_val
                else:
                    z_score = 0.0
                
                # Format the NeuroToken
                neurotoken = f"{region}: {measure_type}={value:.1f}, z={z_score:.1f}"
                neurotokens.append(neurotoken)
        
        return neurotokens
    
    def process_all_subjects(self) -> Dict[str, List[str]]:
        """
        Process all subjects and generate NeuroTokens.
        
        Returns:
            Dictionary mapping subject IDs to their NeuroToken lists
        """
        logger.info("Collecting measurements from all subjects...")
        all_measurements = self.collect_all_measurements()
        
        if not all_measurements:
            logger.error("No measurements found for any subjects")
            return {}
        
        logger.info("Computing region statistics...")
        region_stats = self.compute_region_statistics(all_measurements)
        
        logger.info("Generating NeuroTokens for all subjects...")
        all_neurotokens = {}
        
        for subject, measurements in all_measurements.items():
            neurotokens = self.generate_neurotokens(measurements, region_stats)
            all_neurotokens[subject] = neurotokens
            logger.info(f"Generated {len(neurotokens)} NeuroTokens for {subject}")
        
        return all_neurotokens
    
    def save_results(self, neurotokens: Dict[str, List[str]], output_format: str = 'json'):
        """
        Save NeuroTokens to file.
        
        Args:
            neurotokens: Dictionary of NeuroTokens by subject
            output_format: 'json' or 'csv'
        """
        if output_format.lower() == 'json':
            output_file = self.subjects_dir / "neurotokens.json"
            with open(output_file, 'w') as f:
                json.dump(neurotokens, f, indent=2)
            logger.info(f"Saved NeuroTokens to {output_file}")
            
        elif output_format.lower() == 'csv':
            output_file = self.subjects_dir / "neurotokens.csv"
            
            # Flatten the data for CSV
            rows = []
            for subject, tokens in neurotokens.items():
                for token in tokens:
                    rows.append({'subject': subject, 'neurotoken': token})
            
            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False)
            logger.info(f"Saved NeuroTokens to {output_file}")
        
        # Also save region statistics
        stats_file = self.subjects_dir / "region_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(self.region_stats, f, indent=2)
        logger.info(f"Saved region statistics to {stats_file}")


def main():
    """Main function to run the FreeSurfer parser."""
    parser = argparse.ArgumentParser(description='Parse FreeSurfer output and generate NeuroTokens')
    parser.add_argument('subjects_dir', help='Directory containing subject folders')
    parser.add_argument('--output-format', choices=['json', 'csv'], default='json',
                       help='Output format for NeuroTokens (default: json)')
    
    args = parser.parse_args()
    
    # Initialize parser
    freesurfer_parser = FreeSurferParser(args.subjects_dir)
    
    # Process all subjects
    neurotokens = freesurfer_parser.process_all_subjects()
    
    if neurotokens:
        # Save results
        freesurfer_parser.save_results(neurotokens, args.output_format)
        logger.info("Processing completed successfully!")
    else:
        logger.error("No NeuroTokens were generated. Please check your data.")


if __name__ == "__main__":
    main() 