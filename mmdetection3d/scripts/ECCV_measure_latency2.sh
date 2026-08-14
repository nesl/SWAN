
BUDGETS=(6 4)

CORRUPTIONS=('camera_fog' 'beamsreducing' 'lidar_motionblur' 'camera_motionblur' 'dark')
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

        # Test the Naive Allocation Approach
        python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/cmt_voxel_015_flatformer_swin_multicorrupt.py \
            ./work_dirs/cmt_train_all_corruptions/epoch_8.pth \
            --cfg-options model.test_lidar_retained_layers="$current_lidar" \
            model.test_img_retained_layers="$current_img" \
            test_dataloader.dataset.corruptions="[$corruption]" \
            --work-dir ./work_dirs/temp > Latency_naive_${corruption}_${budget}.txt
        # Test the Universal Controller
        python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/Universal_Controller_cmt_voxel_015_flatformer_swin_multicorrupt.py \
            /workspace/mmdetection3d/work_dirs/ECCV_Controller_universal_model/epoch_16.pth \
            --cfg-options model.controller.layer_budgets="[$budget]" \
            test_dataloader.dataset.corruptions="[$corruption]" \
            --work-dir ./work_dirs/temp > Latency_controller_${corruption}_${budget}.txt
        # Test the Early-Exit Variant
        python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/EE_Universal_Controller.py \
            /workspace/mmdetection3d/work_dirs/EE_Universal_Controller/epoch_8.pth \
            --cfg-options model.controller.layer_budgets="[$budget]" \
            test_dataloader.dataset.corruptions="[$corruption]" \
            --work-dir ./work_dirs/temp > Latency_EE_${corruption}_${budget}.txt
        # Test the ADMN Variant
        python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/ADMN_controller.py \
            /workspace/mmdetection3d/work_dirs/ECCV_Main_Table_Results/ADMN_Results/ADMN/ADMN_model_${budget}/epoch_4.pth \
            --cfg-options model.controller.layer_budget=${budget} \
            test_dataloader.dataset.corruptions="[$corruption]" \
            --work-dir ./work_dirs/temp > Latency_ADMN_${corruption}_${budget}.txt
        # Test the Hard Token Pruning
    done
done