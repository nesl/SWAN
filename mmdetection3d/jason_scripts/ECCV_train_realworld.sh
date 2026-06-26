# bash tools/dist_train.sh /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs_Rain_Night/FT_EE_Universal_Controller.py 4 \
#     > "./work_dirs/FT_EE_Universal_Controller_train.txt"

# CUDA_VISIBLE_DEVICES=0,2,3 bash tools/dist_train.sh /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs_Rain_Night/FT_Soft_Pruning_EE_Universal_Controller.py 3 \
#     > "./work_dirs/FT_Soft_Pruning_EE_Universal_Controller.txt"

CUDA_VISIBLE_DEVICES=0,3 bash tools/dist_train.sh /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs_Rain_Night/FT_Hard_Pruning_EE_Universal_Controller.py 2 \
    > "./work_dirs/FT_Hard_Pruning_EE_Universal_Controller.txt"