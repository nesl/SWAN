bash tools/dist_train.sh /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/Latency_Universal_Controller.py 4 \
        --work-dir "./work_dirs/ECCV_Controller_Latency" \
        > "./work_dirs/ECCV_latency_controller_train.txt"

if [ $# -gt 0 ]; then
    BUDGETS=("$@")
else
    BUDGETS=(5 10 14 19)
fi

LIDAR_LAYERS[5]="[1,0,0,0,0,0,0,0]"
IMG_LAYERS[5]="[1,0,0,0,0,0,0,0,0,0,0,0]"

LIDAR_LAYERS[10]="[1,0,0,0,0,0,0,1]"
IMG_LAYERS[10]="[1,0,0,0,0,0,0,0,0,0,0,1]"

LIDAR_LAYERS[14]="[1,0,0,0,1,0,0,1]"
IMG_LAYERS[14]="[1,0,0,0,0,1,0,0,0,0,0,1]"

LIDAR_LAYERS[19]="[1,0,1,0,0,1,0,1]"
IMG_LAYERS[19]="[1,0,0,1,0,0,0,1,0,0,0,1]"

CORRUPTIONS=('lidar_motionblur')
split='day_dry'
for budget in "${BUDGETS[@]}"; do
        current_lidar="${LIDAR_LAYERS[$budget]}"
        current_img="${IMG_LAYERS[$budget]}"
        for corruption in "${CORRUPTIONS[@]}"; do
                # python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/cmt_voxel_015_flatformer_swin_multicorrupt.py \
                #         ./work_dirs/cmt_train_all_corruptions/epoch_8.pth \
                #         --cfg-options model.test_lidar_retained_layers="$current_lidar" \
                #         model.test_img_retained_layers="$current_img" \
                #         test_dataloader.dataset.corruptions="[$corruption]" \
                #         --work-dir ./work_dirs/Standard_Multicorrupt_test_$corruption/Naive_Alloc_${budget}
                python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/Latency_Universal_Controller.py \
                        /workspace/mmdetection3d/work_dirs/ECCV_Controller_Latency/epoch_16.pth \
                        --cfg-options model.controller.layer_budgets="[$budget]" \
                        test_dataloader.dataset.corruptions="[$corruption]" \
                        --work-dir ./work_dirs/ECCV_Latency_Controller_test_${corruption}/ECCV_Controller_$budget \
                        > ./work_dirs/ECCV_Latency_Controller_test_${budget}_${corruption}.txt
        done
done