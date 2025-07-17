#!/bin/bash

# Batch FreeSurfer recon-all for all subjects in ~/data/OASIS_Processed on EC2
# Each subject folder contains session folders with T1_avg.mgz files
# Outputs go to ~/subjects, logs to ~/logs

set -e

DATA_DIR="$HOME/data/OASIS_Processed"
SUBJECTS_DIR="$HOME/subjects"
LOG_DIR="$HOME/logs"
N_JOBS=2  # Limit to 2 parallel jobs for safety

mkdir -p "$SUBJECTS_DIR"
mkdir -p "$LOG_DIR"

export SUBJECTS_DIR
export LOG_DIR

process_session() {
    session_path="$1"
    mgz_path="$session_path/T1_avg.mgz"
    
    if [ ! -f "$mgz_path" ]; then
        echo "[WARN] No T1_avg.mgz in $session_path, skipping."
        return
    fi
    
    subject_id=$(basename "$(dirname "$session_path")")_$(basename "$session_path")
    out_dir="$SUBJECTS_DIR/$subject_id"
    log_file="$LOG_DIR/${subject_id}.log"
    
    if [ -d "$out_dir" ]; then
        echo "[SKIP] $subject_id already exists, skipping." | tee -a "$log_file"
        return
    fi
    
    echo "[INFO] Processing $subject_id with $mgz_path" | tee "$log_file"
    
    recon-all -i "$mgz_path" \
        -subjid "$subject_id" \
        -all \
        -parallel \
        -openmp 4 \
        -no-isrunning \
        -cw256 \
        -motioncor \
        -T2pial \
        -bigventricles \
        -hires \
        -qcache \
        &>> "$log_file"
}

export -f process_session

# Find all session directories and run
find "$DATA_DIR" -mindepth 2 -maxdepth 2 -type d -name "session_*" | sort | \
    xargs -n 1 -P "$N_JOBS" -I {} bash -c 'process_session "$@"' _ {}

echo "[INFO] All recon-all jobs submitted. Check $SUBJECTS_DIR for results and $LOG_DIR for logs." 