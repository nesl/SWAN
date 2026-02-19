#!/bin/bash

# Define the paths
target_path='/workspace/mmdetection3d/data/multicorrupt/lidar_fog/3/samples/'
original_path='/workspace/mmdetection3d/data/nuscenes/samples/'

# Define the array of directories
dirs=("CAM_BACK" "CAM_BACK_LEFT" "CAM_BACK_RIGHT" "CAM_FRONT" "CAM_FRONT_LEFT" "CAM_FRONT_RIGHT")

# Loop through each directory and create the symlink
for dir in "${dirs[@]}"; do
    ln -s "${original_path}${dir}" "${target_path}${dir}"
done

echo "Symlinks created successfully."