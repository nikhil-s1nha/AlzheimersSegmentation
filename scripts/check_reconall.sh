for subj in $SUBJECTS_DIR/*; do
  if [ ! -f "$subj/scripts/recon-all.done" ]; then
    echo "$(basename "$subj") is missing recon-all.done"
  fi
done

