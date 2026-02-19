# Train the ADMN model with pruning from the multimodal checkpoint
CORRUPTIONS="['beamsreducing','dark','lidar_fog','camera_fog','lidar_motionblur','camera_motionblur','none']"
#CORRUPTIONS="['lidar_fog']"
budgets=(4 6 8 10)

# CORRUPTIONS="['beamsreducing','dark']"
# budgets=(6)


# # Step 1: Finetune the multimodal network on the noisy data, do this once 
# # only for all weather conditions
# bash tools/dist_train.sh projects/CMT/configs/Gsize_256_Configs/cmt_voxel_015_flatformer_swin_multicorrupt.py 4\
#     --cfg-options load_from='/workspace/mmdetection3d/work_dirs/cmt_voxel_015_flatformer_swin_both_pretrained_group_256_unfrozen_efficientvfe/epoch_12.pth' \
#     train_dataloader.dataset.corruptions=${CORRUPTIONS} \
#     val_dataloader.dataset.corruptions=${CORRUPTIONS} \
#     model.layerdrop_rate=0.2 \
#     --work-dir ./work_dirs/cmt_train_all_corruptions

# Universal controller
bash tools/dist_train.sh /workspace/mmdetection3d/projects/CMT/configs/Gsize_256_Configs/ADMN_universal_cmt_voxel_015_flatformer_swin_multicorrupt.py 4 \
        --cfg-options load_from='./work_dirs/cmt_train_all_corruptions/epoch_8.pth' \
        train_dataloader.dataset.corruptions=${CORRUPTIONS}\
        val_dataloader.dataset.corruptions=${CORRUPTIONS} \
        model.controller.layer_budgets="[4,6,8,10,12,15,18]" \
        --work-dir "./work_dirs/ADMN_universal_model" \
        > "./work_dirs/ADMN_universal_model_training_log.txt"

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




