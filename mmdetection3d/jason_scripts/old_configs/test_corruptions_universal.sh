bash setup_env.sh
budget=$1
CORRUPTIONS=('lidar_fog' 'lidar_motionblur' 'camera_fog' 'camera_motionblur' 'beamsreducing' 'dark' 'none')
# Naive layer allocations


for corruption in "${CORRUPTIONS[@]}"; do
    echo "Test with ${corruption} | ${budget}"
    python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/test_configs/Gsize_256/ADMN_multicorrupt_test_universal.py \
            /workspace/mmdetection3d/work_dirs/ADMN_universal_model/epoch_16.pth \
            --cfg-options model.controller.layer_budgets="[$budget]" \
            test_dataloader.dataset.lidar_corruption=$corruption \
            test_dataloader.dataset.camera_corruption=$corruption \
            --work-dir ./work_dirs/ADMN_multicorrupt_universal_test_${corruption}/ADMN_Controller_$budget \
            > ./work_dirs/ADMN_Controller_Universal_test_${budget}_${corruption}.txt
done
