#!/bin/bash

# ================= CONFIGURATION =================
# Remote Server Details
REMOTE_USER="jasonwu"
REMOTE_IP="172.17.45.79" # Or your server alias/hostname
REMOTE_PROJECT_DIR="/data/jason/Modern_UniM2AE/" # Full path on H100 server

# Local Directory (Current directory)
LOCAL_DIR="./"

# Exclusions (The "Clean" Part)
# We use an array for cleaner syntax
EXCLUDES=(
    # Version Control
    "--exclude=.git/"
    
    # Python Caches
    "--exclude=__pycache__/"
    "--exclude=*.pyc"
    "--exclude=.ipynb_checkpoints/"
    
    # DATASETS (Crucial for your symlink)
    # This prevents rsync from trying to traverse or mess with the symlink target
    "--exclude=mmdetection3d/data"
    
    # MMDetection3D / Training Artifacts
    "--exclude=mmdetection3d/work_dirs/"
    "--exclude=*.pth"
    "--exclude=*.pt"
    "--exclude=wandb/"
)

# Rsync Flags
# -a: Archive mode (preserves permissions, times, symlinks as links)
# -v: Verbose
# -z: Compress during transfer
# -P: Show progress bar
# --delete: Delete files on target if they were deleted on source (keeps it 1:1)
FLAGS="-avzP --delete"

# ================= LOGIC =================

# Function to run rsync
run_sync() {
    local direction=$1
    local dry_run=$2
    
    local CMD="rsync $FLAGS ${EXCLUDES[@]}"
    
    if [ "$dry_run" == "true" ]; then
        CMD="$CMD --dry-run"
        echo -e "\n🔍 \033[1;33mPERFORMING DRY RUN ($direction)...\033[0m"
    else
        echo -e "\n🚀 \033[1;32mSYNCING ($direction)...\033[0m"
    fi

    if [ "$direction" == "push" ]; then
        # Local -> Remote
        eval "$CMD $LOCAL_DIR $REMOTE_USER@$REMOTE_IP:$REMOTE_PROJECT_DIR"
    elif [ "$direction" == "pull" ]; then
        # Remote -> Local
        eval "$CMD $REMOTE_USER@$REMOTE_IP:$REMOTE_PROJECT_DIR $LOCAL_DIR"
    fi
}

# ================= INTERFACE =================

if [ "$1" == "push" ]; then
    run_sync "push" "$2"
elif [ "$1" == "pull" ]; then
    run_sync "pull" "$2"
else
    echo "Usage: ./sync.sh [push|pull] [true]"
    echo "  push : Sync Local (4090) -> Remote (H100)"
    echo "  pull : Sync Remote (H100) -> Local (4090)"
    echo "  true : Optional argument for DRY RUN (no actual changes)"
    exit 1
fi