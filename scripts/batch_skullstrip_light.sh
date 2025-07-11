#!/bin/bash

# Lightweight batch skull-stripping using FreeSurfer's older, more efficient tools
# This uses mri_watershed instead of mri_synthstrip for better M1 MacBook compatibility
# Usage: bash batch_skullstrip_light.sh /Volumes/SEAGATE_NIKHIL/OASIS_Processed

if [ $# -ne 1 ]; then
    echo "Usage: $0 <OASIS_Processed_Directory>"
    exit 1
fi

PROCESSED_DIR="$1"
LOGFILE="skullstrip_light.log"
QCLOG="skullstrip_qc.log"

THRESHOLD=0.85

echo "[INFO] Starting lightweight skull-stripping with mri_watershed..." | tee "$LOGFILE"

echo "[INFO] QC log for skull-stripping" > "$QCLOG"

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
    
    echo "[INFO] Stripping $T1_AVG with mri_watershed..." | tee -a "$LOGFILE"
    
    # Use mri_watershed (no -surf!)
    mri_watershed -T1 -atlas "$T1_AVG" "$session_dir/brainmask.mgz"
    
    if [ $? -eq 0 ] && [ -f "$session_dir/brainmask.mgz" ]; then
        # Apply the brain mask to get the skull-stripped image
        mri_mask "$T1_AVG" "$session_dir/brainmask.mgz" "$T1_STRIPPED"
        
        if [ $? -eq 0 ] && [ -f "$T1_STRIPPED" ]; then
            # QC: Compare nonzero voxels
            orig_vox=$(mri_stats --nonzero "$T1_AVG" | grep "Number of nonzero voxels" | awk '{print $6}')
            strip_vox=$(mri_stats --nonzero "$T1_STRIPPED" | grep "Number of nonzero voxels" | awk '{print $6}')
            
            if [ -z "$orig_vox" ] || [ -z "$strip_vox" ]; then
                echo "[ERROR] QC failed for $T1_AVG (could not count voxels)" | tee -a "$QCLOG"
            else
                ratio=$(echo "$strip_vox / $orig_vox" | bc -l)
                if (( \
$(echo "$ratio < $THRESHOLD" | bc -l) )); then
                    echo "[WARN] $T1_STRIPPED failed QC: $strip_vox/$orig_vox = $ratio" | tee -a "$QCLOG"
                else
                    echo "[SUCCESS] $T1_STRIPPED passed QC: $strip_vox/$orig_vox = $ratio" | tee -a "$QCLOG"
                fi
            fi
            # Clean up intermediate files
            rm -f "$session_dir/brainmask.mgz"
            echo "[SUCCESS] Stripped $T1_AVG -> $T1_STRIPPED" | tee -a "$LOGFILE"
        else
            echo "[ERROR] Failed to apply mask to $T1_AVG" | tee -a "$LOGFILE"
        fi
    else
        echo "[ERROR] Failed to create brain mask for $T1_AVG" | tee -a "$LOGFILE"
    fi
    
    # Small delay to prevent overwhelming the system
    sleep 0.5
    sync

done

echo "[INFO] Lightweight skull-stripping complete." | tee -a "$LOGFILE"
echo "[INFO] QC complete. See $QCLOG for details." | tee -a "$LOGFILE" 