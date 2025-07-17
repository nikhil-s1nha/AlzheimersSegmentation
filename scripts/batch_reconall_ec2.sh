#!/bin/bash

set -e

# === CONFIG ===
DATA_DIR="/home/ubuntu/data/OASIS_Processed"
SUBJECTS_DIR="/home/ubuntu/subjects"
LOG_DIR="$HOME/logs"
N_JOBS=2  # adjust this if needed

mkdir -p "$SUBJECTS_DIR"
mkdir -p "$LOG_DIR"

export SUBJECTS_DIR

process_subject() {
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
        -brainstem-structures \
        -qcache \
        -xopts \
        "-fix-ento-wm -transfer-base-bfs -fix-vsinus -fix-mca-dura -fix-ga -fix-acj -synthstrip -synthseg -synthmorph" \
        &>> "$log_file"
}

export -f process_subject

# === FIND ALL session directories and run ===
find "$DATA_DIR" -mindepth 2 -maxdepth 2 -type d -name "session_*" | sort | \
    xargs -n 1 -P "$N_JOBS" -I {} bash -c 'process_subject "$@"' _ {}