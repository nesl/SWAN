bash setup_env.sh
if [ $# -gt 0 ]; then
    BUDGETS=("$@")
else
    BUDGETS=(4 6 8 10 20)
fi
CORRUPTIONS=('lidar_fog' 'lidar_motionblur' 'camera_fog' 'camera_motionblur' 'beamsreducing' 'dark' 'none')
# Naive layer allocations

for budget in "${BUDGETS[@]}"; do
    for corruption in "${CORRUPTIONS[@]}"; do
        echo "Test with ${corruption} | ${budget}"
        python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/test_configs/Gsize_256/ADMN_multicorrupt_test_early_exit.py \
                /workspace/mmdetection3d/work_dirs/ADMN_20_early_exit/epoch_8.pth \
                --cfg-options model.controller.layer_budget=${budget} \
                test_dataloader.dataset.lidar_corruption=$corruption \
                test_dataloader.dataset.camera_corruption=$corruption \
                --work-dir ./work_dirs/ADMN_multicorrupt_EE_test_${corruption}/ADMN_EE_Controller_$budget \
                > ./work_dirs/ADMN_EE_Controller_test_${budget}_${corruption}.txt
        done
done