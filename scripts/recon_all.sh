#!/bin/bash

# Directory paths - Updated for macOS compatibility
DATA_DIR="/Users/NikhilSinha/Downloads/ASDRP/AlzheimersSegmentation/AlzheimersSegmentation/data"  # Update this path
SUBJECTS_DIR="$HOME/freesurfer_subjects"  # Use home directory for FreeSurfer subjects
LOG_DIR="logs"
N_JOBS=4  # Reduced for macOS - adjust based on your system

# Create necessary directories
mkdir -p "$SUBJECTS_DIR"
mkdir -p "$LOG_DIR"
export SUBJECTS_DIR

# Function to process a single subject
process_subject() {
    local sess_path="$1"
    local subj_id=$(basename "$sess_path")
    local log_file="$LOG_DIR/${subj_id}.log"
    local error_file="$LOG_DIR/${subj_id}_error.log"

    # Skip if already completed
    if [ -e "$SUBJECTS_DIR/$subj_id/mri/brain.mgz" ]; then
        echo "✅ Skipping $subj_id — already processed."
        return 0
    fi

    # Find T1_avg.mgz or T1.nii.gz
    local input_file=""
    if [ -f "$sess_path/T1_avg.mgz" ]; then
        input_file="$sess_path/T1_avg.mgz"
    elif [ -f "$sess_path/T1.nii.gz" ]; then
        input_file="$sess_path/T1.nii.gz"
    else
        echo "❌ No T1_avg.mgz or T1.nii.gz found for $subj_id in $sess_path"
        return 1
    fi

    echo "🚀 Starting $subj_id with input: $input_file"
    local start_time=$(date +%s)
    
    # Run recon-all with better error handling
    if recon-all -s "$subj_id" -i "$input_file" -all -openmp "$N_JOBS" > "$log_file" 2> "$error_file"; then
        local end_time=$(date +%s)
        local duration=$(( (end_time - start_time) / 60 ))
        echo "✅ $subj_id completed successfully in ${duration} minutes"
        echo "$subj_id: ${duration} minutes" >> "$LOG_DIR/time_summary.txt"
        return 0
    else
        echo "❌ $subj_id failed - check $error_file for details"
        echo "$subj_id: FAILED" >> "$LOG_DIR/time_summary.txt"
        return 1
    fi
}

export -f process_subject

# Check if DATA_DIR exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: DATA_DIR '$DATA_DIR' does not exist!"
    echo "Please update the DATA_DIR variable in this script to point to your OASIS data directory."
    exit 1
fi

# Check if GNU parallel is available
if ! command -v parallel &> /dev/null; then
    echo "⚠️  GNU parallel not found. Installing via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install parallel
    else
        echo "❌ Homebrew not found. Please install GNU parallel manually:"
        echo "   brew install parallel"
        exit 1
    fi
fi

echo "🔍 Searching for session directories in $DATA_DIR..."
echo "📁 Output directory: $SUBJECTS_DIR"
echo "📝 Log directory: $LOG_DIR"
echo "⚡ Parallel jobs: $N_JOBS"

# Find all session directories and process them
# Updated pattern to match OASIS-2 naming convention
find "$DATA_DIR" -type d -name "*session_*" | sort | \
  parallel -j "$N_JOBS" process_subject {}

echo "🎉 Processing complete! Check $LOG_DIR for detailed logs."
echo "📊 Summary of processing times:"
if [ -f "$LOG_DIR/time_summary.txt" ]; then
    cat "$LOG_DIR/time_summary.txt"
else
    echo "No time summary available."
fi