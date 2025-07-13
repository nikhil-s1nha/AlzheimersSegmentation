#!/bin/bash

# Batch FreeSurfer recon-all for all skull-stripped OASIS-2 sessions
# Sets SUBJECTS_DIR to ~/freesurfer_subjects (local for speed)
# Processes each T1_stripped.mgz as a separate subject

export SUBJECTS_DIR="$HOME/freesurfer_subjects"
mkdir -p "$SUBJECTS_DIR"

find /Volumes/SEAGATE_NIKHIL/OASIS_Processed -name T1_stripped.mgz | while read t1; do
    subj=$(echo $t1 | awk -F'/' '{print $(NF-2)"_"$(NF-1)}')
    subjdir="$SUBJECTS_DIR/$subj"
    mkdir -p "$subjdir/mri"
    cp "$t1" "$subjdir/mri/orig.mgz"
    echo "[INFO] Running recon-all import for $subj..."
    recon-all -subject "$subj" -i "$subjdir/mri/orig.mgz"
    echo "[INFO] Running recon-all -all for $subj..."
    recon-all -subject "$subj" -all
    echo "[INFO] Finished $subj."
done

echo "[INFO] All recon-all jobs submitted. Check $SUBJECTS_DIR for results." 