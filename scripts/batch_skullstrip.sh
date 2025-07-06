#!/bin/bash

# Batch skull-stripping script for OASIS-2 processed data
# Uses mri_synthstrip for robust brain extraction

PROCESSED_ROOT="/Volumes/SEAGATE_NIKHIL/OASIS_Processed"
LOG_FILE="/Volumes/SEAGATE_NIKHIL/OASIS_Processed/skullstrip.log"

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
    
    # Get directory and filename
    t1_dir=$(dirname "$t1_file")
    t1_name=$(basename "$t1_file")
    
    # Output file path
    stripped_file="$t1_dir/T1_stripped.mgz"
    
    echo "Processing $t1_file ($total)..."
    
    # Check if already processed
    if [ -f "$stripped_file" ]; then
        echo "  Already exists, skipping..."
        processed=$((processed + 1))
        continue
    fi
    
    # Run skull-stripping
    if mri_synthstrip -i "$t1_file" -o "$stripped_file" --robust >> "$LOG_FILE" 2>&1; then
        echo "  Successfully stripped"
        processed=$((processed + 1))
    else
        echo "  Failed to strip"
        failed=$((failed + 1))
        
        # Try alternative method if synthstrip fails
        echo "  Attempting alternative method..."
        if mri_watershed -atlas -T1 "$t1_file" "$stripped_file" >> "$LOG_FILE" 2>&1; then
            echo "  Alternative method succeeded"
            processed=$((processed + 1))
            failed=$((failed - 1))
        else
            echo "  Alternative method also failed"
        fi
    fi
done

echo "Skull-stripping complete!"
echo "Total files: $total"
echo "Successfully processed: $processed"
echo "Failed: $failed"
echo "Log saved to: $LOG_FILE" 