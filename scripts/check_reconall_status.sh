#!/bin/bash

# Script to check recon-all processing status for OASIS subjects
# Compares OASIS_Processed directory with subjects directory

# Configuration
OASIS_PROCESSED_DIR="$HOME/data/OASIS_Processed"
SUBJECTS_DIR="$HOME/freesurfer_subjects"
LOG_FILE="reconall_status_check.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Initialize counters
total_subjects=0
total_sessions=0
processed_subjects=0
processed_sessions=0
failed_subjects=0
failed_sessions=0

echo "🔍 Checking recon-all processing status..." | tee "$LOG_FILE"
echo "📁 OASIS_Processed: $OASIS_PROCESSED_DIR" | tee -a "$LOG_FILE"
echo "📁 Subjects: $SUBJECTS_DIR" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# Check if directories exist
if [ ! -d "$OASIS_PROCESSED_DIR" ]; then
    echo -e "${RED}❌ OASIS_Processed directory not found: $OASIS_PROCESSED_DIR${NC}"
    exit 1
fi

if [ ! -d "$SUBJECTS_DIR" ]; then
    echo -e "${RED}❌ Subjects directory not found: $SUBJECTS_DIR${NC}"
    exit 1
fi

# Function to check if a subject session is processed
check_session_processed() {
    local subject_id="$1"
    local session_name="$2"
    local session_path="$3"
    
    # Check for key FreeSurfer output files
    local brain_mgz="$SUBJECTS_DIR/$session_name/mri/brain.mgz"
    local aseg_mgz="$SUBJECTS_DIR/$session_name/mri/aseg.mgz"
    local aparc_lh="$SUBJECTS_DIR/$session_name/stats/lh.aparc.stats"
    local aparc_rh="$SUBJECTS_DIR/$session_name/stats/rh.aparc.stats"
    
    if [ -f "$brain_mgz" ] && [ -f "$aseg_mgz" ] && [ -f "$aparc_lh" ] && [ -f "$aparc_rh" ]; then
        echo -e "${GREEN}✅ $session_name - Complete${NC}"
        return 0
    elif [ -d "$SUBJECTS_DIR/$session_name" ]; then
        echo -e "${YELLOW}⚠️  $session_name - Incomplete (directory exists but missing key files)${NC}"
        return 1
    else
        echo -e "${RED}❌ $session_name - Not processed${NC}"
        return 2
    fi
}

# Process each subject in OASIS_Processed
for subject_dir in "$OASIS_PROCESSED_DIR"/OAS2_*; do
    if [ ! -d "$subject_dir" ]; then
        continue
    fi
    
    subject_id=$(basename "$subject_dir")
    echo -e "\n${BLUE}📋 Subject: $subject_id${NC}" | tee -a "$LOG_FILE"
    
    total_subjects=$((total_subjects + 1))
    subject_processed=true
    subject_sessions=0
    
    # Check each session in the subject directory
    for session_dir in "$subject_dir"/session_*; do
        if [ ! -d "$session_dir" ]; then
            continue
        fi
        
        session_name=$(basename "$session_dir")
        session_id="${subject_id}_${session_name}"
        total_sessions=$((total_sessions + 1))
        subject_sessions=$((subject_sessions + 1))
        
        # Check if session is processed
        if check_session_processed "$subject_id" "$session_id" "$session_dir"; then
            processed_sessions=$((processed_sessions + 1))
        else
            failed_sessions=$((failed_sessions + 1))
            subject_processed=false
        fi
    done
    
    # Update subject status
    if [ "$subject_processed" = true ]; then
        processed_subjects=$((processed_subjects + 1))
    else
        failed_subjects=$((failed_subjects + 1))
    fi
    
    echo "   Sessions: $subject_sessions" | tee -a "$LOG_FILE"
done

# Summary
echo -e "\n${BLUE}==========================================" | tee -a "$LOG_FILE"
echo "📊 PROCESSING SUMMARY" | tee -a "$LOG_FILE"
echo "==========================================${NC}" | tee -a "$LOG_FILE"

echo -e "Total Subjects: ${BLUE}$total_subjects${NC}" | tee -a "$LOG_FILE"
echo -e "  ✅ Complete: ${GREEN}$processed_subjects${NC}" | tee -a "$LOG_FILE"
echo -e "  ❌ Incomplete: ${RED}$failed_subjects${NC}" | tee -a "$LOG_FILE"

echo -e "\nTotal Sessions: ${BLUE}$total_sessions${NC}" | tee -a "$LOG_FILE"
echo -e "  ✅ Complete: ${GREEN}$processed_sessions${NC}" | tee -a "$LOG_FILE"
echo -e "  ❌ Incomplete: ${RED}$failed_sessions${NC}" | tee -a "$LOG_FILE"

# Calculate percentages
if [ $total_subjects -gt 0 ]; then
    subject_percent=$((processed_subjects * 100 / total_subjects))
    echo -e "\nSubject Completion: ${BLUE}${subject_percent}%${NC}" | tee -a "$LOG_FILE"
fi

if [ $total_sessions -gt 0 ]; then
    session_percent=$((processed_sessions * 100 / total_sessions))
    echo -e "Session Completion: ${BLUE}${session_percent}%${NC}" | tee -a "$LOG_FILE"
fi

# Generate list of incomplete sessions for reprocessing
echo -e "\n${YELLOW}📝 Generating list of incomplete sessions...${NC}" | tee -a "$LOG_FILE"
incomplete_list="incomplete_sessions.txt"
> "$incomplete_list"

for subject_dir in "$OASIS_PROCESSED_DIR"/OAS2_*; do
    if [ ! -d "$subject_dir" ]; then
        continue
    fi
    
    subject_id=$(basename "$subject_dir")
    
    for session_dir in "$subject_dir"/session_*; do
        if [ ! -d "$session_dir" ]; then
            continue
        fi
        
        session_name=$(basename "$session_dir")
        session_id="${subject_id}_${session_name}"
        
        # Check if session is incomplete
        local brain_mgz="$SUBJECTS_DIR/$session_id/mri/brain.mgz"
        local aseg_mgz="$SUBJECTS_DIR/$session_id/mri/aseg.mgz"
        local aparc_lh="$SUBJECTS_DIR/$session_id/stats/lh.aparc.stats"
        local aparc_rh="$SUBJECTS_DIR/$session_id/stats/rh.aparc.stats"
        
        if [ ! -f "$brain_mgz" ] || [ ! -f "$aseg_mgz" ] || [ ! -f "$aparc_lh" ] || [ ! -f "$aparc_rh" ]; then
            echo "$session_id" >> "$incomplete_list"
        fi
    done
done

if [ -s "$incomplete_list" ]; then
    echo -e "${YELLOW}📄 Incomplete sessions saved to: $incomplete_list${NC}" | tee -a "$LOG_FILE"
    echo -e "${YELLOW}   Count: $(wc -l < "$incomplete_list") sessions${NC}" | tee -a "$LOG_FILE"
else
    echo -e "${GREEN}🎉 All sessions are complete!${NC}" | tee -a "$LOG_FILE"
fi

echo -e "\n${GREEN}✅ Status check complete! Log saved to: $LOG_FILE${NC}" 