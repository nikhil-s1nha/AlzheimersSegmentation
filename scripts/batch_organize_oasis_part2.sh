#!/bin/bash

RAW_ROOT="/Volumes/SEAGATE_NIKHIL/OAS2_RAW_PART2"
OUT_ROOT="/Volumes/SEAGATE_NIKHIL/OASIS_Processed"

mkdir -p "$OUT_ROOT"

echo "Starting OASIS-2 PART 2 reorganization and robust averaging..."

for session_path in "$RAW_ROOT"/*; do
  [ -d "$session_path" ] || continue
  session_name=$(basename "$session_path")
  subj_id=$(echo "$session_name" | cut -d'_' -f1-2)
  session_idx=$(echo "$session_name" | grep -o 'MR[0-9]*' | sed 's/MR//')

  # Prepare output directory
  out_dir="$OUT_ROOT/$subj_id/session_$session_idx"
  mkdir -p "$out_dir"

  cd "$session_path/RAW" || continue

  # Convert all mpr-*.nifti.img to mpr-*.nii.gz
  nii_list=()
  for img in mpr-*.nifti.img; do
    base=${img%.nifti.img}
    hdr="$base.nifti.hdr"
    nii="$base.nii.gz"
    abs_nii="$session_path/RAW/$base.nii.gz"
    if [ -f "$img" ] && [ -f "$hdr" ]; then
      if [ ! -f "$nii" ]; then
        echo "  Converting $img + $hdr to $nii"
        mri_convert "$img" "$nii" > /dev/null 2>&1
      fi
      nii_list+=("$abs_nii")
    fi
  done

  # Only proceed if at least one nii.gz was created
  if [ ${#nii_list[@]} -eq 0 ]; then
    echo "  No NIfTI files found for $session_name, skipping."
    continue
  fi

  # Run robust averaging
  cd "$out_dir"
  echo "  Averaging for $session_name..."
  mri_robust_template --mov "${nii_list[@]}" --average 1 --template T1_avg.mgz --satit

  # Optional: Skull-strip
  # echo "  Skull-stripping $out_dir/T1_avg.mgz..."
  # mri_synthstrip -i T1_avg.mgz -o T1_stripped.mgz > /dev/null 2>&1

done

echo "Reorganization and robust averaging complete!"
