# ADMN Baselines

#!/bin/bash


CORRUPTIONS=('camera_fog' 'beamsreducing' 'lidar_motionblur' 'camera_motionblur' 'dark')
split='day_dry'


for corruption in "${CORRUPTIONS[@]}"; do
    python3 tools/test.py /workspace/mmdetection3d/projects/PETR/configs/petr_vovnet_gridmask_p4_800x320_ft.py \
        /workspace/mmdetection3d/work_dirs/petr_vovnet_gridmask_p4_800x320_ft/epoch_8.pth \
        --cfg-options test_dataloader.dataset.corruptions="[$corruption]" \
        --work-dir ./work_dirs/PETR_${corruption} > Latency_PETR_${corruption}.txt

    python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/cmt_sst_ft.py\
        /workspace/mmdetection3d/work_dirs/cmt_sst_ft/epoch_8.pth\
        --cfg-options test_dataloader.dataset.corruptions="[$corruption]" \
        --work-dir ./work_dirs/SST_${corruption} > Latency_SST_${corruption}.txt

    python3 tools/test.py /workspace/mmdetection3d/projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_ft.py\
        /workspace/mmdetection3d/work_dirs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d_ft/epoch_8.pth\
        --cfg-options test_dataloader.dataset.corruptions="[$corruption]" \
        --work-dir ./work_dirs/SST_${corruption} > Latency_BEV_${corruption}.txt

done


# for budget in "${BUDGETS[@]}"; do
#     # Fetch the specific layer strings for this budget
#     current_lidar="${LIDAR_LAYERS[$budget]}"
#     current_img="${IMG_LAYERS[$budget]}"

#     for corruption in "${CORRUPTIONS[@]}"; do
#         echo "Running: Budget $budget | Corruption $corruption"

#         python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/ADMN_controller.py \
#             /workspace/mmdetection3d/work_dirs/ADMN/ADMN_model_${budget}/epoch_4.pth \
#             --cfg-options model.controller.layer_budget=${budget} \
#             test_dataloader.dataset.corruptions="[$corruption]" \
#             --work-dir ./work_dirs/ADMN_multicorrupt_${corruption}/ECCV_Controller_$budget \
#             > ./work_dirs/ADMN_multicorrupt_test_${budget}_${corruption}.txt
#     done
# done