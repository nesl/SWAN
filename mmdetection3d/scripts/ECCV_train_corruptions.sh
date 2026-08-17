# # Step 1: Finetune the multimodal network on the noisy data
# CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh projects/CMT/configs/cmt_voxel_015_flatformer_swin_multicorrupt.py 2 \
#     --cfg-options load_from='/workspace/mmdetection3d/work_dirs/cmt_voxel_015_flatformer_swin_both_pretrained_group_256_unfrozen_efficientvfe/epoch_12.pth' \
#     model.layerdrop_rate=0.2 \
#     --work-dir ./work_dirs/cmt_train_all_corruptions


# Step 2: Train the universal controller for all types of corruptions
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh /workspace/mmdetection3d/projects/CMT/configs/Universal_Controller_cmt_voxel_015_flatformer_swin_multicorrupt.py 2 \
        --work-dir "./work_dirs/ECCV_Controller_universal_model" \
        > "./work_dirs/ECCV_universal_controller_train.txt"

# Step 3: Train the early exit
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh /workspace/mmdetection3d/projects/CMT/configs/EE_Universal_Controller.py 2 > "./work_dirs/EE_Universal_Controller_train.txt"

# Step 4: Train the Token Pruning in two stages: soft pruning and then hard pruning
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh /workspace/mmdetection3d/projects/CMT/configs/Soft_Pruning_EE_Universal_Controller.py 2 \
        > "./work_dirs/Soft_Pruning_EE_Universal_Controller.txt"
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh /workspace/mmdetection3d/projects/CMT/configs/Hard_Pruning_EE_Universal_Controller.py 2 \
        > "./work_dirs/Hard_Pruning_EE_Universal_Controller.txt"


