#!/bin/bash

# Directory paths
DATA_DIR="/mnt/data/oasis_data"  # Change if needed
SUBJECTS_DIR="/home/ubuntu/subjects"
LOG_DIR="logs"
N_JOBS=48  # Max parallel jobs for c7i.48xlarge

mkdir -p "$SUBJECTS_DIR"
mkdir -p "$LOG_DIR"
export SUBJECTS_DIR

process_subject() {
    sess_path="$1"
    subj_id=$(basename "$sess_path")

    # Skip if already completed
    if [ -e "$SUBJECTS_DIR/$subj_id/mri/brain.mgz" ]; then
        echo "✅ Skipping $subj_id — already processed."
        return
    fi

    # Find T1_avg.mgz
    input_mgz=$(find "$sess_path" -type f -name "T1_avg.mgz" | head -n 1)
    if [ ! -f "$input_mgz" ]; then
        echo "❌ No T1_avg.mgz found for $subj_id"
        return
    fi

    echo "🚀 Starting $subj_id"
    START=$(date +%s)
    recon-all -s "$subj_id" -i "$input_mgz" -all -openmp 48 > "$LOG_DIR/$subj_id.log" 2>&1
    END=$(date +%s)
    echo "⏱ $subj_id completed in $(( (END - START) / 60 )) min" >> "$LOG_DIR/time_summary.txt"
}

export -f process_subject

# Run all subjects in parallel, skipping those with brain.mgz
find "$DATA_DIR" -type d -name "OAS2_*_session_*" | sort | grep -v "session_1" | \
  parallel -j "$N_JOBS" process_subject {}