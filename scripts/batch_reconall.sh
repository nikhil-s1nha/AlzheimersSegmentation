#!/bin/bash

# Batch FreeSurfer recon-all for all skull-stripped OASIS-2 sessions
# Sets SUBJECTS_DIR to /Volumes/SEAGATE_NIKHIL/freesurfer_subjects (external drive)
# Processes each T1_stripped.mgz as a separate subject

export SUBJECTS_DIR="/Volumes/SEAGATE_NIKHIL/freesurfer_subjects"
mkdir -p "$SUBJECTS_DIR"

find /Volumes/SEAGATE_NIKHIL/OASIS_Processed -name T1_stripped.mgz | while read t1; do
    subj=$(echo $t1 | awk -F'/' '{print $(NF-2)"_"$(NF-1)}')
    subjdir="$SUBJECTS_DIR/$subj"
    
    echo "[INFO] Processing $subj..."
    
    # Create proper FreeSurfer directory structure
    mkdir -p "$subjdir/mri/orig"
    
    # Copy T1_stripped.mgz as 001.mgz (FreeSurfer expects this name)
    cp "$t1" "$subjdir/mri/orig/001.mgz"
    
    # Run recon-all -all (no need for import step since we set up the structure)
    echo "[INFO] Running recon-all -all for $subj..."
    recon-all -s "$subj" -all
    
    echo "[INFO] Finished $subj."
done

echo "[INFO] All recon-all jobs submitted. Check $SUBJECTS_DIR for results." 