# Train the ADMN model with pruning from the multimodal checkpoint

# # Step 1: Finetune the multimodal network on the noisy data, do this once 
# # only for all weather conditions
# bash tools/dist_train.sh projects/CMT/configs/Gsize_256_Configs/cmt_voxel_015_flatformer_swin_multicorrupt.py 4\
#     --cfg-options load_from='/workspace/mmdetection3d/work_dirs/cmt_voxel_015_flatformer_swin_both_pretrained_group_256_unfrozen_efficientvfe/epoch_12.pth' \
#     train_dataloader.dataset.corruptions=${CORRUPTIONS} \
#     val_dataloader.dataset.corruptions=${CORRUPTIONS} \
#     model.layerdrop_rate=0.2 \
#     --work-dir ./work_dirs/cmt_train_all_corruptions

# Universal controller
# CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist_train.sh /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/Universal_Controller_cmt_voxel_015_flatformer_swin_multicorrupt.py 3 \
#         --work-dir "./work_dirs/ECCV_Controller_universal_model" \
#         > "./work_dirs/ECCV_universal_controller_train.txt"

# bash tools/dist_train.sh /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/EE_Universal_Controller.py 4 \
#         > "./work_dirs/EE_Universal_Controller_train.txt"



CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/Soft_Pruning_EE_Universal_Controller.py 4 \
        > "./work_dirs/Soft_Pruning_EE_Universal_Controller.txt"

CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/Hard_Pruning_EE_Universal_Controller.py 4 \
        > "./work_dirs/Hard_Pruning_EE_Universal_Controller.txt"


        
# for LAYER_BUDGET in "${budgets[@]}"; do
#     #Step 2: Train controller from that checkpoint
#     bash tools/dist_train.sh /workspace/mmdetection3d/projects/CMT/configs/Gsize_256_Configs/ADMN_cmt_voxel_015_flatformer_swin_multicorrupt.py 4 \
#         --cfg-options load_from='./work_dirs/cmt_train_all_corruptions/epoch_8.pth' \
#         train_dataloader.dataset.corruptions=${CORRUPTIONS}\
#         val_dataloader.dataset.corruptions=${CORRUPTIONS} \
#         model.controller.layer_budget=${LAYER_BUDGET} \
#         --work-dir "./work_dirs/ADMN_model_${LAYER_BUDGET}" \
#         > "./work_dirs/ADMN_model_${LAYER_BUDGET}_training_log.txt"


        

#     # # # Step 3: Train with soft token masking for all noise(?)
#     # bash tools/dist_train.sh /workspace/mmdetection3d/projects/CMT/configs/Gsize_256_Configs/ADMN_cmt_voxel_015_flatformer_swin_multicorrupt_soft_pruner.py 4 \
#     #     --cfg-options load_from="./work_dirs/ADMN_model_${LAYER_BUDGET}/epoch_3.pth" \
#     #     train_dataloader.dataset.corruptions=${CORRUPTIONS} \
#     #     val_dataloader.dataset.corruptions=${CORRUPTIONS} \
#     #     model.controller.layer_budget=${LAYER_BUDGET} \
#     #     --work-dir "./work_dirs/ADMN_model_soft_pruner_${LAYER_BUDGET}"

#     # # # # Step 4: Train with hard token masking
#     # bash tools/dist_train.sh /workspace/mmdetection3d/projects/CMT/configs/Gsize_256_Configs/ADMN_cmt_voxel_015_flatformer_swin_multicorrupt_hard_pruner.py 4 \
#     #     --cfg-options load_from="./work_dirs/ADMN_model_soft_pruner_${LAYER_BUDGET}/epoch_8.pth" \
#     #     train_dataloader.dataset.corruptions=${CORRUPTIONS} \
#     #     val_dataloader.dataset.corruptions=${CORRUPTIONS} \
#     #     model.controller.layer_budget=${LAYER_BUDGET} \
#     #     --work-dir "./work_dirs/ADMN_model_hard_pruner_${LAYER_BUDGET}"
# done




