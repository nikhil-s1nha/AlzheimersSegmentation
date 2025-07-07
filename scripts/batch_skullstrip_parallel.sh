#!/bin/bash

# Batch skull-stripping script with parallel processing (M1 MacBook safe version)
# This version uses fewer cores and includes memory management to prevent crashes

set -e  # Exit on any error

# Configuration
PROCESSED_DIR="/Users/NikhilSinha/Downloads/ASDRP/AlzheimersSegmentation/OASIS_Processed"
LOG_FILE="skullstrip_parallel.log"

# M1 MacBook safe settings - use fewer cores to prevent crashes
# Detect number of cores but limit to 4 for M1 safety
TOTAL_CORES=$(sysctl -n hw.ncpu)
SAFE_CORES=$((TOTAL_CORES > 4 ? 4 : TOTAL_CORES))
# Use even fewer cores if we have many
if [ $SAFE_CORES -gt 2 ]; then
    SAFE_CORES=$((SAFE_CORES - 1))
fi

echo "=== M1 MacBook Safe Skull-Stripping Script ===" | tee -a "$LOG_FILE"
echo "Detected $TOTAL_CORES total cores, using $SAFE_CORES for safety" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"

# Check if processed directory exists
if [ ! -d "$PROCESSED_DIR" ]; then
    echo "Error: Processed directory not found: $PROCESSED_DIR" | tee -a "$LOG_FILE"
    exit 1
fi

# Function to skull-strip a single file
skullstrip_file() {
    local file="$1"
    local session_dir=$(dirname "$file")
    local output_file="$session_dir/T1_stripped.mgz"
    
    # Skip if already processed
    if [ -f "$output_file" ]; then
        echo "Skipping $file (already processed)" | tee -a "$LOG_FILE"
        return 0
    fi
    
    echo "Processing: $file" | tee -a "$LOG_FILE"
    
    # Add memory management and error handling
    # Limit memory usage for M1 safety
    export MALLOC_ARENA_MAX=2
    
    # Run skull-stripping with timeout and memory limits
    timeout 300 mri_synthstrip -i "$file" -o "$output_file" 2>&1 | tee -a "$LOG_FILE"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Successfully processed: $file" | tee -a "$LOG_FILE"
    else
        echo "Error processing: $file (exit code: $exit_code)" | tee -a "$LOG_FILE"
        # Remove failed output file if it exists
        [ -f "$output_file" ] && rm "$output_file"
    fi
    
    # Small delay to prevent overwhelming the system
    sleep 1
}

# Export function for parallel processing
export -f skullstrip_file
export LOG_FILE

# Find all T1_avg.mgz files that need skull-stripping
echo "Finding T1_avg.mgz files..." | tee -a "$LOG_FILE"
find "$PROCESSED_DIR" -name "T1_avg.mgz" -type f | while read -r file; do
    session_dir=$(dirname "$file")
    output_file="$session_dir/T1_stripped.mgz"
    
    # Only process if T1_stripped.mgz doesn't exist
    if [ ! -f "$output_file" ]; then
        echo "$file"
    fi
done > /tmp/files_to_process.txt

# Count files to process
FILE_COUNT=$(wc -l < /tmp/files_to_process.txt)
echo "Found $FILE_COUNT files to process" | tee -a "$LOG_FILE"

if [ $FILE_COUNT -eq 0 ]; then
    echo "No files need processing. All done!" | tee -a "$LOG_FILE"
    rm -f /tmp/files_to_process.txt
    exit 0
fi

# Process files in parallel with safer settings
echo "Starting parallel processing with $SAFE_CORES cores..." | tee -a "$LOG_FILE"

# Use xargs with safer parallel processing
cat /tmp/files_to_process.txt | xargs -P "$SAFE_CORES" -I {} bash -c 'skullstrip_file "{}"'

# Clean up
rm -f /tmp/files_to_process.txt

echo "=== Skull-stripping completed ===" | tee -a "$LOG_FILE"
echo "Finished at: $(date)" | tee -a "$LOG_FILE"

# Final summary
echo "Checking results..." | tee -a "$LOG_FILE"
TOTAL_FILES=$(find "$PROCESSED_DIR" -name "T1_avg.mgz" -type f | wc -l)
PROCESSED_FILES=$(find "$PROCESSED_DIR" -name "T1_stripped.mgz" -type f | wc -l)
echo "Total T1_avg.mgz files: $TOTAL_FILES" | tee -a "$LOG_FILE"
echo "Successfully skull-stripped: $PROCESSED_FILES" | tee -a "$LOG_FILE"

if [ $TOTAL_FILES -eq $PROCESSED_FILES ]; then
    echo "✅ All files successfully processed!" | tee -a "$LOG_FILE"
else
    echo "⚠️  Some files may need manual processing" | tee -a "$LOG_FILE"
fi 