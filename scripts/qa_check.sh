#!/bin/bash

# SET THESE
export FREESURFER_HOME=/Applications/freesurfer/8.0.0
source $FREESURFER_HOME/SetUpFreeSurfer.sh

export SUBJECTS_DIR="/Volumes/SEAGATE_NIKHIL/freesurfer_subjects/subjects/OAS2_0001_session_1"

# Loop through all subject directories
for SUBJECT in "$SUBJECTS_DIR"/*; do
  if [ -d "$SUBJECT" ]; then
    SUBJECT_ID=$(basename "$SUBJECT")
    echo "Launching Freeview QA for $SUBJECT_ID..."

    freeview \
      -v \
        "$SUBJECT/mri/T1.mgz" \
        "$SUBJECT/mri/brainmask.mgz" \
      -f \
        "$SUBJECT/surf/lh.pial:edgecolor=red" \
        "$SUBJECT/surf/rh.pial:edgecolor=blue" \
      --viewport axial &
    
    # Wait for user to close Freeview before moving on
    read -p "Press [Enter] to proceed to next subject..." dummy
  fi
done