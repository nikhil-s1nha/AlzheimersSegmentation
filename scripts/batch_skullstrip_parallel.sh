#!/bin/bash

PROCESSED_ROOT="/Volumes/SEAGATE_NIKHIL/OASIS_Processed"
LOG_FILE="$PROCESSED_ROOT/skullstrip_parallel.log"

# Auto-detect number of CPU cores
N_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "Detected $N_JOBS CPU cores"

echo "Starting parallel batch skull-stripping of OASIS-2 data..."
echo "Processed data root: $PROCESSED_ROOT"
echo "Log file: $LOG_FILE"
echo "Using $N_JOBS parallel jobs"

# Initialize counters
total=0
processed=0
failed=0

# Function to process a single T1 file
process_t1() {
    local t1_file="$1"
    local t1_dir=$(dirname "$t1_file")
    local stripped_file="$t1_dir/T1_stripped.mgz"
    
    if [ -s "$stripped_file" ]; then
        echo "$stripped_file already exists, skipping."
        return 0
    fi
    
    echo "Skull-stripping $t1_file..."
    mri_synthstrip -i "$t1_file" -o "$stripped_file" > /dev/null 2>&1
    
    if [ $? -eq 0 ] && [ -s "$stripped_file" ]; then
        echo "Success: $stripped_file"
        return 0
    else
        echo "Failed: $t1_file"
        return 1
    fi
}

export -f process_t1
export LOG_FILE

# Find all T1_avg.mgz files and process them in parallel
find "$PROCESSED_ROOT" -name "T1_avg.mgz" | \
    xargs -P $N_JOBS -I {} bash -c 'process_t1 "{}" 2>&1 | tee -a "$LOG_FILE"'

echo "Parallel skull-stripping complete!"
echo "Log saved to: $LOG_FILE" 