#!/bin/bash

# Set variables
RAW_ROOT="/Volumes/SEAGATE_NIKHIL/OAS2_RAW_PART1"
FS_ROOT="/Volumes/SEAGATE_NIKHIL/FS_SUBJECTS"

echo "Starting batch conversion of OAS2_RAW_PART1..."
echo "Raw data from: $RAW_ROOT"
echo "Output to: $FS_ROOT"

# Create output directory
mkdir -p "$FS_ROOT"

# Counter for progress
total=0
converted=0

# Loop through all subject folders
for subjdir in "$RAW_ROOT"/*; do
  if [ ! -d "$subjdir" ]; then
    continue
  fi
  
  subj=$(basename "$subjdir")
  total=$((total + 1))
  
  echo "Processing $subj ($total)..."
  
  # Find the best T1-weighted file (mpr-1.nii.gz is typically T1)
  nifti=""
  
  # Try .nii.gz first
  if [ -f "$subjdir/RAW/mpr-1.nii.gz" ]; then
    nifti="$subjdir/RAW/mpr-1.nii.gz"
  elif [ -f "$subjdir/RAW/mpr-2.nii.gz" ]; then
    nifti="$subjdir/RAW/mpr-2.nii.gz"
  elif [ -f "$subjdir/RAW/mpr-3.nii.gz" ]; then
    nifti="$subjdir/RAW/mpr-3.nii.gz"
  fi
  
  # If no .nii.gz, try .nifti.img/.hdr pair
  if [ -z "$nifti" ]; then
    if [ -f "$subjdir/RAW/mpr-1.nifti.img" ] && [ -f "$subjdir/RAW/mpr-1.nifti.hdr" ]; then
      nifti="$subjdir/RAW/mpr-1.nifti.img"
    elif [ -f "$subjdir/RAW/mpr-2.nifti.img" ] && [ -f "$subjdir/RAW/mpr-2.nifti.hdr" ]; then
      nifti="$subjdir/RAW/mpr-2.nifti.img"
    elif [ -f "$subjdir/RAW/mpr-3.nifti.img" ] && [ -f "$subjdir/RAW/mpr-3.nifti.hdr" ]; then
      nifti="$subjdir/RAW/mpr-3.nifti.img"
    fi
  fi
  
  if [ -z "$nifti" ]; then
    echo "  No suitable NIfTI file found for $subj"
    continue
  fi

  # Create FreeSurfer subject directory structure
  mkdir -p "$FS_ROOT/$subj/mri"

  # Convert to mgz
  if mri_convert "$nifti" "$FS_ROOT/$subj/mri/T1.mgz" > /dev/null 2>&1; then
    echo "  ✓ Converted $subj"
    converted=$((converted + 1))
  else
    echo "  ✗ Failed to convert $subj"
  fi
done

echo ""
echo "Batch conversion completed!"
echo "Total subjects processed: $total"
echo "Successfully converted: $converted"
echo "Output directory: $FS_ROOT"
