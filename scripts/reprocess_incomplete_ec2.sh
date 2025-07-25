#!/bin/bash

# Script to reprocess incomplete sessions identified by check_reconall_status_ec2.sh
# Optimized for EC2 instances with high CPU/memory

# Configuration for EC2
OASIS_PROCESSED_DIR="/home/ubuntu/data/OASIS_Processed"
SUBJECTS_DIR="/home/ubuntu/subjects"
INCOMPLETE_LIST="incomplete_sessions_ec2.txt"
N_JOBS=12  # Changed from 48 to 12
LOG_FILE="$HOME/reprocess_ec2.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

export OASIS_PROCESSED_DIR
export SUBJECTS_DIR
export LOG_FILE

echo "🔄 Reprocessing incomplete sessions on EC2..." | tee "$LOG_FILE"
echo "🖥️  Instance: $(hostname)" | tee -a "$LOG_FILE"
echo "⚡ Parallel jobs: $N_JOBS" | tee -a "$LOG_FILE"
echo "📁 OASIS_Processed: $OASIS_PROCESSED_DIR" | tee -a "$LOG_FILE"
echo "📁 Subjects: $SUBJECTS_DIR" | tee -a "$LOG_FILE"

# Check if incomplete sessions list exists
if [ ! -f "$INCOMPLETE_LIST" ]; then
    echo -e "${RED}❌ Incomplete sessions list not found: $INCOMPLETE_LIST${NC}"
    echo "Please run check_reconall_status_ec2.sh first to generate the list."
    exit 1
fi

# Count incomplete sessions
incomplete_count=$(wc -l < "$INCOMPLETE_LIST")
echo -e "${BLUE}📊 Found $incomplete_count incomplete sessions${NC}" | tee -a "$LOG_FILE"

if [ $incomplete_count -eq 0 ]; then
    echo -e "${GREEN}🎉 No incomplete sessions to reprocess!${NC}"
    exit 0
fi

# Function to reprocess a single session
reprocess_session() {
    local session_id="$1"
    local subject=$(echo "$session_id" | cut -d'/' -f1)
    local session=$(echo "$session_id" | cut -d'/' -f2)
    local session_path="$OASIS_PROCESSED_DIR/$subject/session_$session"
    local subj_id="${subject}_session_${session}"
    local subj_dir="$SUBJECTS_DIR/$subj_id"

    echo "DEBUG: session_id=$session_id, session_path=$session_path, subj_id=$subj_id" | tee -a "$LOG_FILE"
    echo "DEBUG: OASIS_PROCESSED_DIR=$OASIS_PROCESSED_DIR"

    if [ ! -d "$session_path" ]; then
        echo -e "${RED}❌ Session directory not found: $session_path${NC}" | tee -a "$LOG_FILE"
        return 1
    fi

    # Remove existing subject directory if it exists
    if [ -d "$subj_dir" ]; then
        echo "Removing existing FreeSurfer subject directory: $subj_dir" | tee -a "$LOG_FILE"
        rm -rf "$subj_dir"
    fi

    # Find input file
    local input_file=""
    if [ -f "$session_path/T1_avg.mgz" ]; then
        input_file="$session_path/T1_avg.mgz"
    elif [ -f "$session_path/T1.nii.gz" ]; then
        input_file="$session_path/T1.nii.gz"
    else
        echo "ERROR: No T1 image found for $subj_id at $session_path" | tee -a "$LOG_FILE"
        return 1
    fi

    echo "Running recon-all for $subj_id using $input_file" | tee -a "$LOG_FILE"
    recon-all -subject "$subj_id" -i "$input_file" -all -sd "$SUBJECTS_DIR" -openmp 8 -verbose 2>&1 | tee -a "$LOG_FILE"
}

export -f reprocess_session

# Check if GNU parallel is available
if ! command -v parallel &> /dev/null; then
    echo -e "${YELLOW}⚠️  GNU parallel not found. Installing...${NC}"
    sudo apt-get update
    sudo apt-get install -y parallel
fi

echo -e "${BLUE}⚡ Starting parallel reprocessing with $N_JOBS jobs...${NC}" | tee -a "$LOG_FILE"

# Process incomplete sessions in parallel
cat "$INCOMPLETE_LIST" | parallel -j "$N_JOBS" reprocess_session {}

echo -e "\n${GREEN}🎉 Reprocessing complete!${NC}" | tee -a "$LOG_FILE"

# Show final disk usage
echo -e "\n${BLUE}💾 Final Disk Usage:${NC}" | tee -a "$LOG_FILE"
df -h "$SUBJECTS_DIR" | tee -a "$LOG_FILE"

# Run status check again to verify completion
echo -e "\n${YELLOW}🔍 Running status check to verify completion...${NC}" | tee -a "$LOG_FILE"
./check_reconall_status_ec2.sh 