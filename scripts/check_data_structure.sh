#!/bin/bash

# Script to check data directory structure for OASIS-2 processing

echo "🔍 Checking data directory structure..."

# Check if DATA_DIR exists
DATA_DIR="/Users/NikhilSinha/Downloads/ASDRP/AlzheimersSegmentation/AlzheimersSegmentation/data"

if [ ! -d "$DATA_DIR" ]; then
    echo "❌ DATA_DIR does not exist: $DATA_DIR"
    echo "Please create this directory or update the path in recon_all.sh"
    exit 1
fi

echo "✅ DATA_DIR exists: $DATA_DIR"

# Check for session directories
echo "🔍 Looking for session directories..."
session_dirs=$(find "$DATA_DIR" -type d -name "*session_*" 2>/dev/null)

if [ -z "$session_dirs" ]; then
    echo "❌ No session directories found in $DATA_DIR"
    echo "Expected pattern: *session_*"
    echo "Current contents of $DATA_DIR:"
    ls -la "$DATA_DIR"
    exit 1
fi

echo "✅ Found session directories:"
echo "$session_dirs" | head -5
if [ $(echo "$session_dirs" | wc -l) -gt 5 ]; then
    echo "... and $(($(echo "$session_dirs" | wc -l) - 5)) more"
fi

# Check for T1 files in first few session directories
echo "🔍 Checking for T1 files in session directories..."
count=0
for sess_dir in $session_dirs; do
    if [ $count -ge 3 ]; then
        break
    fi
    
    subj_id=$(basename "$sess_dir")
    echo "  Checking $subj_id..."
    
    if [ -f "$sess_dir/T1_avg.mgz" ]; then
        echo "    ✅ Found T1_avg.mgz"
    elif [ -f "$sess_dir/T1.nii.gz" ]; then
        echo "    ✅ Found T1.nii.gz"
    else
        echo "    ❌ No T1 file found"
        echo "    Contents: $(ls "$sess_dir" 2>/dev/null | head -3)"
    fi
    
    count=$((count + 1))
done

# Check FreeSurfer installation
echo "🔍 Checking FreeSurfer installation..."
if command -v recon-all &> /dev/null; then
    echo "✅ recon-all found"
    recon-all --version | head -1
else
    echo "❌ recon-all not found. Please install FreeSurfer."
fi

# Check GNU parallel
echo "🔍 Checking GNU parallel..."
if command -v parallel &> /dev/null; then
    echo "✅ GNU parallel found"
else
    echo "⚠️  GNU parallel not found. Will attempt to install via Homebrew."
fi

# Check available memory and CPU
echo "🔍 System resources:"
echo "  CPU cores: $(sysctl -n hw.ncpu 2>/dev/null || echo "Unknown")"
echo "  Memory: $(sysctl -n hw.memsize 2>/dev/null | awk '{print $0/1024/1024/1024 " GB"}' || echo "Unknown")"

echo "🎉 Data structure check complete!" 