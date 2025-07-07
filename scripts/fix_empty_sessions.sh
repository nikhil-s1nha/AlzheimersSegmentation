#!/bin/bash

PROCESSED_ROOT="/Volumes/SEAGATE_NIKHIL/OASIS_Processed"
RAW_ROOTS=("/Volumes/SEAGATE_NIKHIL/OAS2_RAW_PART1" "/Volumes/SEAGATE_NIKHIL/OAS2_RAW_PART2")

find "$PROCESSED_ROOT" -type d -name "session_*" | while read -r session_dir; do
    t1="$session_dir/T1_avg.mgz"
    if [ ! -s "$t1" ]; then
        subj=$(basename "$(dirname "$session_dir")")
        sess=$(basename "$session_dir" | sed 's/session_//')
        found_raw=""
        for raw_root in "${RAW_ROOTS[@]}"; do
            raw_dir="$raw_root/${subj}_MR${sess}/RAW"
            if [ -d "$raw_dir" ]; then
                found_raw="$raw_dir"
                break
            fi
        done
        if [ -z "$found_raw" ]; then
            echo "RAW folder not found for $subj session $sess"
            continue
        fi
        nii_files=("$found_raw"/mpr-*.nii.gz)
        if [ ${#nii_files[@]} -eq 0 ]; then
            echo "No NIfTI files found in $found_raw"
            continue
        fi
        echo "Averaging for $subj session $sess..."
        mri_robust_template --mov "${nii_files[@]}" --average 1 --template "$t1" --satit
    fi
    
    # Optionally, print status
    if [ -s "$t1" ]; then
        echo "$t1 exists and is non-empty."
    fi

done 