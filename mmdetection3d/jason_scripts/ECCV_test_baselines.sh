# ADMN Baselines

#!/bin/bash

if [ $# -gt 0 ]; then
    BUDGETS=("$@")
else
    BUDGETS=(4 6 8 10 12 14 16)
fi

CORRUPTIONS=('lidar_fog' 'camera_fog' 'beamsreducing' 'lidar_motionblur' 'camera_motionblur' 'dark' 'none')
split='day_dry'



for budget in "${BUDGETS[@]}"; do
    # Fetch the specific layer strings for this budget
    current_lidar="${LIDAR_LAYERS[$budget]}"
    current_img="${IMG_LAYERS[$budget]}"

    for corruption in "${CORRUPTIONS[@]}"; do
        echo "Running: Budget $budget | Corruption $corruption"

        python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/ADMN_controller.py \
            /workspace/mmdetection3d/work_dirs/ADMN/ADMN_model_${budget}/epoch_4.pth \
            --cfg-options model.controller.layer_budget=${budget} \
            test_dataloader.dataset.corruptions="[$corruption]" \
            --work-dir ./work_dirs/ADMN_multicorrupt_${corruption}/ECCV_Controller_$budget \
            > ./work_dirs/ADMN_multicorrupt_test_${budget}_${corruption}.txt
    done
done