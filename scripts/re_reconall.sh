#!/bin/bash

# Set your FreeSurfer environment
export FREESURFER_HOME=/usr/local/freesurfer/8.0.0
source $FREESURFER_HOME/SetUpFreeSurfer.sh


INPUT_DIR=~/data/OASIS_Processed
SUBJECTS_DIR=~/subjects  # Where FreeSurfer output goes
export SUBJECTS_DIR

# Loop through all subjects
for SUBJECT in "$INPUT_DIR"/*; do
  SUBJECT_ID=$(basename "$SUBJECT")
  echo "Processing subject $SUBJECT_ID..."

  # Loop through session directories
  for SESSION in "$SUBJECT"/*; do
    SESSION_ID=$(basename "$SESSION")
    echo "  Found session $SESSION_ID"

    T1="$SESSION/T1_avg.mgz"
    if [ -f "$T1" ]; then
      OUTPUT_ID="${SUBJECT_ID}_${SESSION_ID}"

      echo "  Removing previous output for $OUTPUT_ID..."
      rm -rf "$SUBJECTS_DIR/$OUTPUT_ID"

      echo "  Running recon-all for $OUTPUT_ID..."
      recon-all -i "$T1" -s "$OUTPUT_ID" -all
    else
      echo "  ⚠️ No T1_avg.mgz found in $SESSION, skipping"
    fi
  done
done



