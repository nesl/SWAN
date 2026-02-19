if [ $# -gt 0 ]; then
    BUDGETS=("$@")
else
    BUDGETS=(4 6 8 10 20)
fi
#CORRUPTIONS=('lidar_fog' 'lidar_motionblur' 'camera_fog' 'camera_motionblur' 'beamsreducing' 'dark' 'none')
# CORRUPTIONS=('lidar_fog')
CORRUPTIONS=('none')
# Naive layer allocations
LIDAR_LAYERS[4]="[1,0,0,0,0,0,0,1]"
IMG_LAYERS[4]="[1,0,0,0,0,0,0,0,0,0,0,1]"

LIDAR_LAYERS[6]="[1,0,0,1,0,0,0,1]"
IMG_LAYERS[6]="[1,0,0,0,0,0,1,0,0,0,0,1]"

LIDAR_LAYERS[8]="[1,0,0,1,0,1,0,1]"
IMG_LAYERS[8]="[1,0,0,1,0,0,0,1,0,0,0,1]"

LIDAR_LAYERS[10]="[1,1,0,1,0,1,0,1]"
IMG_LAYERS[10]="[1,0,1,0,0,0,1,0,0,1,0,1]"

LIDAR_LAYERS[20]="[1,1,1,1,1,1,1,1]"
IMG_LAYERS[20]="[1,1,1,1,1,1,1,1,1,1,1,1]"

split='night'

for budget in "${BUDGETS[@]}"; do
    # Fetch the specific layer strings for this budget
    current_lidar="${LIDAR_LAYERS[$budget]}"
    current_img="${IMG_LAYERS[$budget]}"

    for corruption in "${CORRUPTIONS[@]}"; do
        echo "Running: Budget $budget | Corruption $corruption"

        # # Test Naive Alloc
        python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/test_configs/Gsize_256/Standard_Multicorrupt_test.py \
            ./work_dirs/cmt_train_all_corruptions/epoch_8.pth \
            --cfg-options model.test_lidar_retained_layers="$current_lidar" \
            model.test_img_retained_layers="$current_img" \
            test_dataloader.dataset.lidar_corruption=$corruption \
            test_dataloader.dataset.camera_corruption=$corruption \
            test_dataloader.dataset.split=$split\
            test_evaluator.split=$split\
            --work-dir ./work_dirs/Standard_Multicorrupt_test_$corruption/Naive_Alloc_${budget}_split_$split

        # # # # Test ADMN model
        # python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/test_configs/Gsize_256/ADMN_multicorrupt_test.py \
        #     ./work_dirs/ADMN_model_${budget}/epoch_3.pth \
        #     --cfg-options model.controller.layer_budget=${budget} \
        #     test_dataloader.dataset.lidar_corruption=$corruption \
        #     test_dataloader.dataset.camera_corruption=$corruption \
        #     --work-dir ./work_dirs/ADMN_multicorrupt_test_$corruption/ADMN_Controller_$budget \
        #     > ./work_dirs/ADMN_Controller_test_${budget}_${corruption}.txt

        # python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/Gsize_256_Configs/ADMN_cmt_voxel_015_flatformer_swin_multicorrupt_early_exit.py \
        #     ./work_dirs/ADMN_${budget}_early_exit/epoch_3.pth \
        #     --cfg-options model.controller.layer_budget=${budget} \
        #     test_dataloader.dataset.corruptions="[$corruption"] \
        #     --work-dir ./work_dirs/ADMN_multicorrupt_test_$corruption/ADMN_Controller_$budget \
        #     > ./work_dirs/ADMN_Early_Exit_test_${budget}_${corruption}.txt


        # # Test ADMN model with soft token pruning
        # python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/test_configs/Gsize_256/ADMN_multicorrupt_pruner_test.py \
        #     ./work_dirs//ADMN_model_soft_pruner_${budget}/epoch_8.pth \
        #     --cfg-options model.controller.layer_budget=${budget} \
        #     test_dataloader.dataset.lidar_corruption=$corruption \
        #     test_dataloader.dataset.camera_corruption=$corruption \
        #      --work-dir ./work_dirs/ADMN_multicorrupt_test_$corruption/ADMN_Soft_Pruning_$budget
  
    done
done






# # Test ADMN Model with 6 layers of Budget
# python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/test_configs/Gsize_256/ADMN_multicorrupt_test_lidar_3_camera_0.py \
#     work_dirs/ADMN_Multicorrupt_Budget_6/epoch_1.pth \
#     --cfg-options model.controller.layer_budget=6 \
#     --work-dir ./work_dirs/ADMN_multicorrupt_test_lidar_3_camera_0/Budget_6

# python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/test_configs/Gsize_256/ADMN_multicorrupt_test_lidar_0_camera_3.py \
#     work_dirs/ADMN_Multicorrupt_Budget_6/epoch_1.pth \
#     --cfg-options model.controller.layer_budget=6 \
#     --work-dir ./work_dirs/ADMN_multicorrupt_test_lidar_0_camera_3/Budget_6

# # Test ADMN Model with 10 layers of Budget
# python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/test_configs/Gsize_256/ADMN_multicorrupt_test_lidar_3_camera_0.py \
#     work_dirs/ADMN_Multicorrupt_Budget_10/epoch_1.pth \
#     --cfg-options model.controller.layer_budget=10 \
#     --work-dir ./work_dirs/ADMN_multicorrupt_test_lidar_3_camera_0/Budget_10

# python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/test_configs/Gsize_256/ADMN_multicorrupt_test_lidar_0_camera_3.py \
#     work_dirs/ADMN_Multicorrupt_Budget_10/epoch_1.pth \
#     --cfg-options model.controller.layer_budget=10 \
#     --work-dir ./work_dirs/ADMN_multicorrupt_test_lidar_0_camera_3/Budget_10






# # Naive Alloc 5:5

# python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/test_configs/Gsize_256/Standard_multicorrupt_test_lidar_0_camera_3.py \
#     work_dirs/cmt_voxel_015_flatformer_swin_multicorrupt/epoch_8.pth \
#     --cfg-options model.test_lidar_retained_layers="[1,1,0,1,0,1,0,1]" \
#     model.test_img_retained_layers="[1,0,1,0,0,0,1,0,0,1,0,1]" \
#     --work-dir ./work_dirs/Standard_multicorrupt_test_lidar_0_camera_3/Naive_10

# python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/test_configs/Gsize_256/Standard_multicorrupt_test_lidar_3_camera_0.py \
#     work_dirs/cmt_voxel_015_flatformer_swin_multicorrupt/epoch_8.pth \
#     --cfg-options model.test_lidar_retained_layers="[1,1,0,1,0,1,0,1]" \
#     model.test_img_retained_layers="[1,0,1,0,0,0,1,0,0,1,0,1]" \
#     --work-dir ./work_dirs/Standard_multicorrupt_test_lidar_3_camera_0/Naive_10

# # Naive Alloc 3:3
# python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/test_configs/Gsize_256/Standard_multicorrupt_test_lidar_0_camera_3.py \
#     work_dirs/cmt_voxel_015_flatformer_swin_multicorrupt/epoch_8.pth \
#     --cfg-options model.test_lidar_retained_layers="[1,0,0,1,0,0,0,1]" \
#     model.test_img_retained_layers="[1,0,0,0,0,0,1,0,0,0,0,1]" \
#     --work-dir ./work_dirs/Standard_multicorrupt_test_lidar_0_camera_3/Naive_6

# python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/test_configs/Gsize_256/Standard_multicorrupt_test_lidar_3_camera_0.py \
#     work_dirs/cmt_voxel_015_flatformer_swin_multicorrupt/epoch_8.pth \
#     --cfg-options model.test_lidar_retained_layers="[1,0,0,1,0,0,0,1]" \
#     model.test_img_retained_layers="[1,0,0,0,0,0,1,0,0,0,0,1]" \
#     --work-dir ./work_dirs/Standard_multicorrupt_test_lidar_3_camera_0/Naive_6


# # # Test the base model, upper bound
# python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/test_configs/Gsize_256/Standard_multicorrupt_test_lidar_0_camera_3.py \
#     work_dirs/cmt_voxel_015_flatformer_swin_multicorrupt/epoch_8.pth \
#     --work-dir ./work_dirs/Standard_multicorrupt_test_lidar_0_camera_3/Upper_Bound

# python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/test_configs/Gsize_256/Standard_multicorrupt_test_lidar_3_camera_0.py \
#     work_dirs/cmt_voxel_015_flatformer_swin_multicorrupt/epoch_8.pth \
#     --work-dir ./work_dirs/Standard_multicorrupt_test_lidar_3_camera_0/Upper_Bound







# budget=8
# corruption='camera_fog'
# python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/test_configs/Gsize_256/ADMN_multicorrupt_test.py \
#             ./work_dirs/ADMN_model_${budget}/epoch_3.pth \
#             --cfg-options model.controller.layer_budget=${budget} \
#             test_dataloader.dataset.lidar_corruption=$corruption \
#             test_dataloader.dataset.camera_corruption=$corruption \
#             --work-dir ./work_dirs/temp
# budget=10
# corruption='camera_fog'
# python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/test_configs/Gsize_256/ADMN_multicorrupt_pruner_test.py \
#             ./work_dirs/ADMN_model_hard_pruner_${budget}/epoch_8.pth \
#             --cfg-options model.controller.layer_budget=${budget} \
#             test_dataloader.dataset.lidar_corruption=$corruption \
#             test_dataloader.dataset.camera_corruption=$corruption \
#             --work-dir ./work_dirs/temp