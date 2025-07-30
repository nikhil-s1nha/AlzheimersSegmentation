#!/usr/bin/env python3
"""
NIfTI to FreeSurfer Converter
Converts .nii.gz files to FreeSurfer-compatible format and prepares directory structure
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NIfTIToFreeSurferConverter:
    """Converter for NIfTI files to FreeSurfer format."""
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the converter.
        
        Args:
            input_dir: Directory containing .nii.gz files
            output_dir: Directory to output FreeSurfer-compatible files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def find_nifti_files(self) -> List[Path]:
        """Find all .nii.gz files in the input directory."""
        nifti_files = []
        
        # Look for .nii.gz files
        for file_path in self.input_dir.rglob("*.nii.gz"):
            nifti_files.append(file_path)
        
        # Also look for .nii files
        for file_path in self.input_dir.rglob("*.nii"):
            nifti_files.append(file_path)
        
        logger.info(f"Found {len(nifti_files)} NIfTI files")
        return nifti_files
    
    def check_freesurfer_installation(self) -> bool:
        """Check if FreeSurfer is properly installed."""
        try:
            result = subprocess.run(['which', 'mri_convert'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("FreeSurfer mri_convert found")
                return True
            else:
                logger.warning("FreeSurfer mri_convert not found in PATH")
                return False
        except FileNotFoundError:
            logger.error("FreeSurfer not installed or not in PATH")
            return False
    
    def convert_nifti_to_mgz(self, nifti_file: Path, subject_id: str) -> Optional[Path]:
        """
        Convert a NIfTI file to FreeSurfer .mgz format.
        
        Args:
            nifti_file: Path to the NIfTI file
            subject_id: Subject identifier
            
        Returns:
            Path to the converted .mgz file
        """
        # Create subject directory
        subject_dir = self.output_dir / subject_id
        subject_dir.mkdir(exist_ok=True)
        
        # Determine output filename
        if "T1" in nifti_file.name or "t1" in nifti_file.name:
            output_file = subject_dir / "T1.mgz"
        elif "T2" in nifti_file.name or "t2" in nifti_file.name:
            output_file = subject_dir / "T2.mgz"
        else:
            # Use original name with .mgz extension
            output_file = subject_dir / f"{nifti_file.stem}.mgz"
        
        try:
            # Use FreeSurfer's mri_convert to convert NIfTI to MGZ
            cmd = [
                'mri_convert',
                str(nifti_file),
                str(output_file)
            ]
            
            logger.info(f"Converting {nifti_file.name} to {output_file.name}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully converted {nifti_file.name}")
                return output_file
            else:
                logger.error(f"Failed to convert {nifti_file.name}: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error converting {nifti_file.name}: {e}")
            return None
    
    def create_freesurfer_subject_structure(self, subject_id: str) -> Path:
        """
        Create the standard FreeSurfer subject directory structure.
        
        Args:
            subject_id: Subject identifier
            
        Returns:
            Path to the subject directory
        """
        subject_dir = self.output_dir / subject_id
        
        # Create standard FreeSurfer directories
        directories = [
            "mri",
            "surf", 
            "label",
            "stats",
            "scripts",
            "touch",
            "tmp",
            "trash"
        ]
        
        for dir_name in directories:
            (subject_dir / dir_name).mkdir(exist_ok=True)
        
        logger.info(f"Created FreeSurfer directory structure for {subject_id}")
        return subject_dir
    
    def copy_converted_files(self, subject_id: str, mgz_file: Path):
        """
        Copy converted .mgz files to the appropriate FreeSurfer directory.
        
        Args:
            subject_id: Subject identifier
            mgz_file: Path to the converted .mgz file
        """
        subject_dir = self.output_dir / subject_id
        mri_dir = subject_dir / "mri"
        
        # Copy T1.mgz to mri/T1.mgz
        if mgz_file.name == "T1.mgz":
            dest_file = mri_dir / "T1.mgz"
            shutil.copy2(mgz_file, dest_file)
            logger.info(f"Copied T1.mgz to {dest_file}")
        
        # Copy T2.mgz to mri/T2.mgz
        elif mgz_file.name == "T2.mgz":
            dest_file = mri_dir / "T2.mgz"
            shutil.copy2(mgz_file, dest_file)
            logger.info(f"Copied T2.mgz to {dest_file}")
        
        # For other files, copy to mri directory
        else:
            dest_file = mri_dir / mgz_file.name
            shutil.copy2(mgz_file, dest_file)
            logger.info(f"Copied {mgz_file.name} to {dest_file}")
    
    def create_subject_list(self, subjects: List[str]):
        """
        Create a list of subjects for FreeSurfer processing.
        
        Args:
            subjects: List of subject IDs
        """
        subjects_file = self.output_dir / "subjects_list.txt"
        with open(subjects_file, 'w') as f:
            for subject in subjects:
                f.write(f"{subject}\n")
        
        logger.info(f"Created subjects list: {subjects_file}")
    
    def create_processing_script(self, subjects: List[str]):
        """
        Create a script to run FreeSurfer processing.
        
        Args:
            subjects: List of subject IDs
        """
        script_file = self.output_dir / "run_freesurfer.sh"
        
        with open(script_file, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# FreeSurfer Processing Script\n")
            f.write("# Generated by NIfTI to FreeSurfer Converter\n\n")
            
            f.write("export SUBJECTS_DIR=$(pwd)\n\n")
            
            for subject in subjects:
                f.write(f"echo 'Processing subject: {subject}'\n")
                f.write(f"recon-all -subject {subject} -all\n")
                f.write(f"echo 'Completed processing: {subject}'\n\n")
        
        # Make script executable
        os.chmod(script_file, 0o755)
        logger.info(f"Created processing script: {script_file}")
    
    def convert_all_files(self, subject_mapping: Optional[Dict[str, str]] = None):
        """
        Convert all NIfTI files to FreeSurfer format.
        
        Args:
            subject_mapping: Optional mapping from filename to subject ID
        """
        nifti_files = self.find_nifti_files()
        
        if not nifti_files:
            logger.error("No NIfTI files found in input directory")
            return
        
        if not self.check_freesurfer_installation():
            logger.error("FreeSurfer not found. Please install FreeSurfer first.")
            return
        
        converted_subjects = []
        
        for nifti_file in nifti_files:
            # Determine subject ID
            if subject_mapping and nifti_file.name in subject_mapping:
                subject_id = subject_mapping[nifti_file.name]
            else:
                # Extract subject ID from filename
                subject_id = self.extract_subject_id(nifti_file.name)
            
            if not subject_id:
                logger.warning(f"Could not determine subject ID for {nifti_file.name}")
                continue
            
            # Create FreeSurfer directory structure
            self.create_freesurfer_subject_structure(subject_id)
            
            # Convert NIfTI to MGZ
            mgz_file = self.convert_nifti_to_mgz(nifti_file, subject_id)
            
            if mgz_file:
                # Copy to appropriate location
                self.copy_converted_files(subject_id, mgz_file)
                converted_subjects.append(subject_id)
        
        # Create subject list and processing script
        if converted_subjects:
            unique_subjects = list(set(converted_subjects))
            self.create_subject_list(unique_subjects)
            self.create_processing_script(unique_subjects)
            
            logger.info(f"Successfully converted {len(converted_subjects)} files for {len(unique_subjects)} subjects")
            logger.info(f"Output directory: {self.output_dir}")
            logger.info("Next steps:")
            logger.info("1. Run: cd output_directory")
            logger.info("2. Run: ./run_freesurfer.sh")
            logger.info("3. Wait for FreeSurfer processing to complete")
            logger.info("4. Use the generated stats files with the NeuroTokens pipeline")
    
    def extract_subject_id(self, filename: str) -> Optional[str]:
        """
        Extract subject ID from filename.
        
        Args:
            filename: Name of the NIfTI file
            
        Returns:
            Subject ID or None if not found
        """
        # Common patterns for subject IDs
        import re
        
        # Pattern 1: sub-XXXX format
        match = re.search(r'sub-(\d+)', filename)
        if match:
            return f"sub-{match.group(1)}"
        
        # Pattern 2: Just numbers
        match = re.search(r'(\d{3,4})', filename)
        if match:
            return f"sub-{match.group(1)}"
        
        # Pattern 3: OASIS format
        match = re.search(r'OAS2_(\d+)', filename)
        if match:
            return f"sub-{match.group(1)}"
        
        return None

def main():
    """Main function to run the NIfTI to FreeSurfer converter."""
    parser = argparse.ArgumentParser(description='Convert NIfTI files to FreeSurfer format')
    parser.add_argument('input_dir', help='Directory containing .nii.gz files')
    parser.add_argument('output_dir', help='Directory to output FreeSurfer-compatible files')
    parser.add_argument('--subject-mapping', help='JSON file mapping filenames to subject IDs')
    
    args = parser.parse_args()
    
    # Load subject mapping if provided
    subject_mapping = None
    if args.subject_mapping:
        import json
        with open(args.subject_mapping, 'r') as f:
            subject_mapping = json.load(f)
    
    # Initialize converter
    converter = NIfTIToFreeSurferConverter(args.input_dir, args.output_dir)
    
    # Convert all files
    converter.convert_all_files(subject_mapping)

if __name__ == "__main__":
    main() 