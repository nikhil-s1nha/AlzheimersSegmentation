#!/bin/bash

RAW_ROOT="/Volumes/SEAGATE_NIKHIL/OAS2_RAW_PART1"
OUT_ROOT="/Volumes/SEAGATE_NIKHIL/OASIS_Processed"

mkdir -p "$OUT_ROOT"

echo "Starting fast OASIS-2 reorganization and averaging..."

# Function to process a single session
process_session() {
    local session_path="$1"
    local session_name=$(basename "$session_path")
    local subj_id=$(echo "$session_name" | cut -d'_' -f1-2)
    local session_idx=$(echo "$session_name" | grep -o 'MR[0-9]*' | sed 's/MR//')
    
    local out_dir="$OUT_ROOT/$subj_id/session_$session_idx"
    mkdir -p "$out_dir"
    
    cd "$session_path/RAW" || return 1
    
    # Convert all mpr-*.nifti.img to mpr-*.nii.gz
    local nii_list=()
    for img in mpr-*.nifti.img; do
        local base=${img%.nifti.img}
        local hdr="$base.nifti.hdr"
        local nii="$base.nii.gz"
        local abs_nii="$session_path/RAW/$base.nii.gz"
        
        if [ -f "$img" ] && [ -f "$hdr" ]; then
            if [ ! -f "$nii" ]; then
                mri_convert "$img" "$nii" > /dev/null 2>&1
            fi
            nii_list+=("$abs_nii")
        fi
    done
    
    # Only proceed if at least one nii.gz was created
    if [ ${#nii_list[@]} -eq 0 ]; then
        echo "  No NIfTI files found for $session_name, skipping."
        return 1
    fi
    
    # Use faster averaging method
    cd "$out_dir"
    echo "  Fast averaging for $session_name..."
    
    if [ ${#nii_list[@]} -eq 1 ]; then
        # Single file - just copy
        cp "${nii_list[0]}" T1_avg.mgz
    elif [ ${#nii_list[@]} -eq 2 ]; then
        # Two files - simple average
        mri_average "${nii_list[@]}" T1_avg.mgz
    else
        # Multiple files - use robust template with fewer iterations
        mri_robust_template --mov "${nii_list[@]}" --average 1 --template T1_avg.mgz --satit --maxit 1 --epsit 0.1
    fi
    
    echo "  Completed $session_name"
}

export -f process_session
export RAW_ROOT OUT_ROOT

# Process all sessions in parallel (limit to 4 concurrent jobs to avoid overwhelming system)
find "$RAW_ROOT" -maxdepth 1 -type d -name "OAS2_*" | sort | xargs -I {} -P 4 bash -c 'process_session "$@"' _ {}

echo "Fast reorganization and averaging complete!" 