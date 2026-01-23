#!/bin/bash


cd /workspace/mmdetection3d/projects/MultiCorrupt/converter


python img_converter.py \
--corruption dark \
--root_folder /workspace/mmdetection3d/data/nuscenes \
--dst_folder /workspace/mmdetection3d/data/multicorrupt/dark/3 \
--severity 3 \
--n_cpus 8

python lidar_converter.py \
--corruption beamsreducing \
--root_folder /workspace/mmdetection3d/data/nuscenes \
--dst_folder /workspace/mmdetection3d/data/multicorrupt/test/3 \
--severity 3 \
--n_cpus 8 \
--sweep True


