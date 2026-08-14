#!/bin/bash
trap "kill 0" SIGINT
# Train ADMN baselines, we train single budget models with layer 1 active at all times
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/ADMN_controller.py \
    --cfg-options model.controller.layer_budget=4  --work-dir './work_dirs/ADMN/ADMN_model_4/' & 
CUDA_VISIBLE_DEVICES=1 python3 tools/train.py /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/ADMN_controller.py \
    --cfg-options model.controller.layer_budget=6  --work-dir './work_dirs/ADMN/ADMN_model_6/' &
CUDA_VISIBLE_DEVICES=2 python3 tools/train.py /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/ADMN_controller.py \
    --cfg-options model.controller.layer_budget=8  --work-dir './work_dirs/ADMN/ADMN_model_8/' &    
CUDA_VISIBLE_DEVICES=3 python3 tools/train.py /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/ADMN_controller.py \
    --cfg-options model.controller.layer_budget=12  --work-dir './work_dirs/ADMN/ADMN_model_12/'

wait   
CUDA_VISIBLE_DEVICES=2 python3 tools/train.py /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/ADMN_controller.py \
    --cfg-options model.controller.layer_budget=14  --work-dir './work_dirs/ADMN/ADMN_model_14/' &    
CUDA_VISIBLE_DEVICES=3 python3 tools/train.py /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/ADMN_controller.py \
    --cfg-options model.controller.layer_budget=16  --work-dir './work_dirs/ADMN/ADMN_model_16/'

wait