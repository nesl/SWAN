budget=4
LIDAR_LAYERS[4]="[1,0,0,0,0,0,0,1]"
IMG_LAYERS[4]="[1,0,0,0,0,0,0,0,0,0,0,1]"
current_lidar="${LIDAR_LAYERS[$budget]}"
current_img="${IMG_LAYERS[$budget]}"
corruption='camera_fog'
# python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/Universal_Controller_cmt_voxel_015_flatformer_swin_multicorrupt.py \
#             /workspace/mmdetection3d/work_dirs/ECCV_Controller_universal_model/epoch_16.pth \
#             --cfg-options model.controller.layer_budgets="[$budget]" \
#             test_dataloader.dataset.corruptions="[$corruption]" > "controller_same_as_naive.txt"

python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs/cmt_voxel_015_flatformer_swin_multicorrupt.py \
            ./work_dirs/cmt_train_all_corruptions/epoch_8.pth \
            --cfg-options model.test_lidar_retained_layers="$current_lidar" \
            model.test_img_retained_layers="$current_img" \
            test_dataloader.dataset.corruptions="[$corruption]" > "naive_same_as_controller.txt"