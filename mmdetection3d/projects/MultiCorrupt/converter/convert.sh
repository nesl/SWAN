#!/bin/bash


cd /workspace/mmdetection3d/projects/MultiCorrupt/converter

python lidar_converter.py \
--corruption fog \
--root_folder /workspace/mmdetection3d/data/nuscenes \
--dst_folder /workspace/mmdetection3d/data/multicorrupt/fog/3 \
--severity 3 \
--n_cpus 31 \
--sweep True

python lidar_converter.py \
--corruption snow \
--root_folder /workspace/mmdetection3d/data/nuscenes \
--dst_folder /workspace/mmdetection3d/data/multicorrupt/snow/3 \
--severity 3 \
--n_cpus 31 \
--sweep True

python img_converter.py \
--corruption snow \
--root_folder /workspace/mmdetection3d/data/nuscenes \
--dst_folder /workspace/mmdetection3d/data/multicorrupt/snow/3 \
--severity 3 \
--n_cpus 31






