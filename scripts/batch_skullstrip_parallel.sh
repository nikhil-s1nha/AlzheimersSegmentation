#!/bin/bash

# Sequential batch skull-stripping for OASIS-2 processed data
# Usage: bash batch_skullstrip_parallel.sh /Volumes/SEAGATE_NIKHIL/OASIS_Processed

if [ $# -ne 1 ]; then
    echo "Usage: $0 <OASIS_Processed_Directory>"
    exit 1
fi

PROCESSED_DIR="$1"
LOGFILE="skullstrip_sequential.log"

echo "[INFO] Starting sequential skull-stripping..." | tee "$LOGFILE"

find "$PROCESSED_DIR" -type d -name 'session_*' | while read -r session_dir; do
    T1_AVG="$session_dir/T1_avg.mgz"
    T1_STRIPPED="$session_dir/T1_stripped.mgz"
    if [ ! -f "$T1_AVG" ]; then
        echo "[WARN] Missing $T1_AVG, skipping." | tee -a "$LOGFILE"
        continue
    fi
    if [ -f "$T1_STRIPPED" ]; then
        echo "[INFO] $T1_STRIPPED already exists, skipping." | tee -a "$LOGFILE"
        continue
    fi
    echo "[INFO] Stripping $T1_AVG..." | tee -a "$LOGFILE"
    mri_synthstrip -i "$T1_AVG" -o "$T1_STRIPPED"
    if [ $? -eq 0 ]; then
        echo "[SUCCESS] Stripped $T1_AVG -> $T1_STRIPPED" | tee -a "$LOGFILE"
    else
        echo "[ERROR] Failed to strip $T1_AVG" | tee -a "$LOGFILE"
    fi
    sync

done

echo "[INFO] Skull-stripping complete." | tee -a "$LOGFILE" 