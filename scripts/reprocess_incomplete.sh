#!/bin/bash

# Script to reprocess incomplete sessions identified by check_reconall_status.sh

# Configuration
OASIS_PROCESSED_DIR="$HOME/data/OASIS_Processed"
SUBJECTS_DIR="$HOME/freesurfer_subjects"
INCOMPLETE_LIST="incomplete_sessions.txt"
LOG_DIR="logs"
N_JOBS=4  # Adjust based on your system

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create log directory
mkdir -p "$LOG_DIR"
export SUBJECTS_DIR

echo "🔄 Reprocessing incomplete sessions..." | tee "$LOG_DIR/reprocess.log"

# Check if incomplete sessions list exists
if [ ! -f "$INCOMPLETE_LIST" ]; then
    echo -e "${RED}❌ Incomplete sessions list not found: $INCOMPLETE_LIST${NC}"
    echo "Please run check_reconall_status.sh first to generate the list."
    exit 1
fi

# Count incomplete sessions
incomplete_count=$(wc -l < "$INCOMPLETE_LIST")
echo -e "${BLUE}📊 Found $incomplete_count incomplete sessions${NC}" | tee -a "$LOG_DIR/reprocess.log"

if [ $incomplete_count -eq 0 ]; then
    echo -e "${GREEN}🎉 No incomplete sessions to reprocess!${NC}"
    exit 0
fi

# Function to reprocess a single session
reprocess_session() {
    local session_id="$1"
    local log_file="$LOG_DIR/${session_id}_reprocess.log"
    local error_file="$LOG_DIR/${session_id}_reprocess_error.log"
    
    echo "🚀 Reprocessing $session_id..." | tee -a "$LOG_DIR/reprocess.log"
    
    # Find the corresponding session directory in OASIS_Processed
    local subject_id=$(echo "$session_id" | cut -d'_' -f1-2)
    local session_name=$(echo "$session_id" | cut -d'_' -f3-)
    local session_path="$OASIS_PROCESSED_DIR/$subject_id/$session_name"
    
    if [ ! -d "$session_path" ]; then
        echo -e "${RED}❌ Session directory not found: $session_path${NC}" | tee -a "$LOG_DIR/reprocess.log"
        return 1
    fi
    
    # Find input file
    local input_file=""
    if [ -f "$session_path/T1_avg.mgz" ]; then
        input_file="$session_path/T1_avg.mgz"
    elif [ -f "$session_path/T1.nii.gz" ]; then
        input_file="$session_path/T1.nii.gz"
    else
        echo -e "${RED}❌ No T1 file found in $session_path${NC}" | tee -a "$LOG_DIR/reprocess.log"
        return 1
    fi
    
    # Remove existing incomplete directory if it exists
    if [ -d "$SUBJECTS_DIR/$session_id" ]; then
        echo "🧹 Removing incomplete directory: $SUBJECTS_DIR/$session_id" | tee -a "$LOG_DIR/reprocess.log"
        rm -rf "$SUBJECTS_DIR/$session_id"
    fi
    
    # Start reprocessing
    local start_time=$(date +%s)
    echo "⏱️  Starting reprocess at $(date)" | tee -a "$LOG_DIR/reprocess.log"
    
    if recon-all -s "$session_id" -i "$input_file" -all -openmp "$N_JOBS" > "$log_file" 2> "$error_file"; then
        local end_time=$(date +%s)
        local duration=$(( (end_time - start_time) / 60 ))
        echo -e "${GREEN}✅ $session_id completed successfully in ${duration} minutes${NC}" | tee -a "$LOG_DIR/reprocess.log"
        return 0
    else
        echo -e "${RED}❌ $session_id failed - check $error_file for details${NC}" | tee -a "$LOG_DIR/reprocess.log"
        return 1
    fi
}

export -f reprocess_session

# Check if GNU parallel is available
if ! command -v parallel &> /dev/null; then
    echo -e "${YELLOW}⚠️  GNU parallel not found. Installing via Homebrew...${NC}"
    if command -v brew &> /dev/null; then
        brew install parallel
    else
        echo -e "${RED}❌ Homebrew not found. Please install GNU parallel manually:${NC}"
        echo "   brew install parallel"
        exit 1
    fi
fi

echo -e "${BLUE}⚡ Starting parallel reprocessing with $N_JOBS jobs...${NC}" | tee -a "$LOG_DIR/reprocess.log"

# Process incomplete sessions in parallel
cat "$INCOMPLETE_LIST" | parallel -j "$N_JOBS" reprocess_session {}

echo -e "\n${GREEN}🎉 Reprocessing complete!${NC}" | tee -a "$LOG_DIR/reprocess.log"
echo -e "${BLUE}📝 Check $LOG_DIR for detailed logs${NC}" | tee -a "$LOG_DIR/reprocess.log"

# Run status check again to verify completion
echo -e "\n${YELLOW}🔍 Running status check to verify completion...${NC}" | tee -a "$LOG_DIR/reprocess.log"
./scripts/check_reconall_status.sh 