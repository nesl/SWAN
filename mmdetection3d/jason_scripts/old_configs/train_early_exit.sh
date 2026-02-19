bash setup_env.sh
CORRUPTIONS="['beamsreducing','dark','lidar_fog','camera_fog','lidar_motionblur','camera_motionblur','none']"
#CORRUPTIONS="['camera_fog','beamsreducing']"
LAYER_BUDGET=20
bash tools/dist_train.sh /workspace/mmdetection3d/projects/CMT/configs/Gsize_256_Configs/ADMN_cmt_voxel_015_flatformer_swin_multicorrupt_early_exit.py 4\
        --cfg-options load_from='./work_dirs/cmt_train_all_corruptions/epoch_8.pth' \
        train_dataloader.dataset.corruptions=${CORRUPTIONS}\
        val_dataloader.dataset.corruptions=${CORRUPTIONS} \
        model.controller.layer_budget=${LAYER_BUDGET} \
        --work-dir "./work_dirs/ADMN_${LAYER_BUDGET}_early_exit/" > "./work_dirs/ADMN_${LAYER_BUDGET}_early_exit.txt"
