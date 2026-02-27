#!/bin/bash

if [ $# -gt 0 ]; then
    BUDGETS=("$@")
else
    BUDGETS=(16 8 6 4)
fi

CORRUPTIONS=('lidar_motionblur' 'camera_fog' 'beamsreducing' 'camera_motionblur' 'dark')
split='day_dry'

LIDAR_LAYERS[4]="[1,0,0,0,0,0,0,1]"
IMG_LAYERS[4]="[1,0,0,0,0,0,0,0,0,0,0,1]"

LIDAR_LAYERS[6]="[1,0,0,1,0,0,0,1]"
IMG_LAYERS[6]="[1,0,0,0,0,0,1,0,0,0,0,1]"

LIDAR_LAYERS[8]="[1,0,0,1,0,1,0,1]"
IMG_LAYERS[8]="[1,0,0,1,0,0,0,1,0,0,0,1]"

LIDAR_LAYERS[10]="[1,1,0,1,0,1,0,1]"
IMG_LAYERS[10]="[1,0,1,0,0,0,1,0,0,1,0,1]"

LIDAR_LAYERS[12]="[1,1,0,1,0,1,1,1]"
IMG_LAYERS[12]="[1,0,1,0,1,0,1,0,0,1,0,1]"

LIDAR_LAYERS[14]="[1,1,1,1,0,1,1,1]"
IMG_LAYERS[14]="[1,0,1,0,1,0,1,0,1,1,0,1]"

LIDAR_LAYERS[16]="[1,1,1,1,1,1,1,1]"
IMG_LAYERS[16]="[1,1,1,0,1,0,1,0,1,1,0,1]"

LIDAR_LAYERS[20]="[1,1,1,1,1,1,1,1]"
IMG_LAYERS[20]="[1,1,1,1,1,1,1,1,1,1,1,1]"



for budget in "${BUDGETS[@]}"; do
    # Fetch the specific layer strings for this budget
    current_lidar="${LIDAR_LAYERS[$budget]}"
    current_img="${IMG_LAYERS[$budget]}"

    for corruption in "${CORRUPTIONS[@]}"; do
        echo "Running: Budget $budget | Corruption $corruption"

        # # Test the Naive Allocation Approach
        # python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/cmt_voxel_015_flatformer_swin_multicorrupt.py \
        #     ./work_dirs/cmt_train_all_corruptions/epoch_8.pth \
        #     --cfg-options model.test_lidar_retained_layers="$current_lidar" \
        #     model.test_img_retained_layers="$current_img" \
        #     test_dataloader.dataset.corruptions="[$corruption]" \
        #     --work-dir ./work_dirs/Standard_Multicorrupt_test_$corruption/Naive_Alloc_${budget}

        # # Test the Universal Controller
        # python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/Universal_Controller_cmt_voxel_015_flatformer_swin_multicorrupt.py \
        #     /workspace/mmdetection3d/work_dirs/ECCV_Controller_universal_model/epoch_16.pth \
        #     --cfg-options model.controller.layer_budgets="[$budget]" \
        #     test_dataloader.dataset.corruptions="[$corruption]" \
        #     --work-dir ./work_dirs/ECCV_multicorrupt_universal_test_${corruption}/ECCV_Controller_$budget \
        #     > ./work_dirs/ECCV_Controller_Universal_test_${budget}_${corruption}.txt

        # # # Test the Early-Exit Variant
        python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/EE_Universal_Controller.py \
            /workspace/mmdetection3d/work_dirs/EE_Universal_Controller/epoch_16.pth \
            --cfg-options model.controller.layer_budgets="[$budget]" \
            test_dataloader.dataset.corruptions="[$corruption]" \
            --work-dir ./work_dirs/EE_universal_test_${corruption}/EE_Controller_$budget \
            > ./work_dirs/EE_Universal_test_${budget}_${corruption}.txt

        # # Test the Hard Token Pruning
        # python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/Hard_Pruning_EE_Universal_Controller.py \
        #     /workspace/mmdetection3d/work_dirs/Hard_Pruning_EE_Universal_Controller/epoch_8.pth \
        #     --cfg-options model.controller.layer_budgets="[$budget]" \
        #     test_dataloader.dataset.corruptions="[$corruption]" \
        #     --work-dir ./work_dirs/Pruner_EE_universal_test_${corruption}/Pruner_EE_Controller_$budget \
        #     > ./work_dirs/Pruner_EE_Universal_test_${budget}_${corruption}.txt

        # python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/ADMN_controller.py \
        #     /workspace/mmdetection3d/work_dirs/ECCV_Main_Table_Results/ADMN_Results/ADMN/ADMN_model_${budget}/epoch_4.pth \
        #     --cfg-options model.controller.layer_budget=${budget} \
        #     test_dataloader.dataset.corruptions="[$corruption]" \
        #      --work-dir ./work_dirs/ADMN_test_${corruption}/ADMN_$budget \
        #     > ./work_dirs/ADMN_test_${budget}_${corruption}.txt
    done
done