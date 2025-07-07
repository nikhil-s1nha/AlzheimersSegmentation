#!/bin/bash

# Batch skull-stripping script for OASIS-2 processed data
# Uses mri_synthstrip for robust brain extraction

PROCESSED_ROOT="/Volumes/SEAGATE_NIKHIL/OASIS_Processed"
LOG_FILE="$PROCESSED_ROOT/skullstrip.log"

echo "Starting batch skull-stripping of OASIS-2 data..."
echo "Processed data root: $PROCESSED_ROOT"
echo "Log file: $LOG_FILE"

# Initialize counters
total=0
processed=0
failed=0

# Find all T1_avg.mgz files
find "$PROCESSED_ROOT" -name "T1_avg.mgz" | while read -r t1_file; do
    total=$((total + 1))
    t1_dir=$(dirname "$t1_file")
    stripped_file="$t1_dir/T1_stripped.mgz"
    if [ -s "$stripped_file" ]; then
        echo "$stripped_file already exists, skipping." | tee -a "$LOG_FILE"
        continue
    fi
    echo "Skull-stripping $t1_file..." | tee -a "$LOG_FILE"
    mri_synthstrip -i "$t1_file" -o "$stripped_file" 2>> "$LOG_FILE"
    if [ $? -eq 0 ] && [ -s "$stripped_file" ]; then
        processed=$((processed + 1))
        echo "Success: $stripped_file" | tee -a "$LOG_FILE"
    else
        failed=$((failed + 1))
        echo "Failed: $t1_file" | tee -a "$LOG_FILE"
    fi
    # Optionally, fallback to mri_watershed if synthstrip fails
    # if [ ! -s "$stripped_file" ]; then
    #     echo "Fallback: mri_watershed for $t1_file..." | tee -a "$LOG_FILE"
    #     mri_watershed "$t1_file" "$stripped_file" 2>> "$LOG_FILE"
    # fi
done

echo "Skull-stripping complete!" | tee -a "$LOG_FILE"
echo "Total files: $total" | tee -a "$LOG_FILE"
echo "Successfully processed: $processed" | tee -a "$LOG_FILE"
echo "Failed: $failed" | tee -a "$LOG_FILE"
echo "Log saved to: $LOG_FILE" 