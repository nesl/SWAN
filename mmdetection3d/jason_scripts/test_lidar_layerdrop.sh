CUDA_VISIBLE_DEVICES=2 python3 tools/test.py projects/CMT/configs/cmt_voxel_015_flatformer_layerdrop.py \
    /workspace/mmdetection3d/work_dirs/cmt_voxel_015_flatformer/epoch_20.pth \
    --work-dir ./work_dirs/cmt_test_nold_full_layers \
    --cfg-options model.test_lidar_retained_layers="[1,1,1,1,1,1,1,1]"


# CUDA_VISIBLE_DEVICES=1 python3 tools/test.py projects/CMT/configs/cmt_voxel_015_flatformer_layerdrop.py \
#     /workspace/mmdetection3d/work_dirs/cmt_voxel_015_flatformer/epoch_20.pth \
#     --work-dir ./work_dirs/cmt_test_nold_7_layers \
#     --cfg-options model.test_lidar_retained_layers="[1,1,1,1,0,1,1,1]"

# CUDA_VISIBLE_DEVICES=1 python3 tools/test.py projects/CMT/configs/cmt_voxel_015_flatformer_layerdrop.py \
#     /workspace/mmdetection3d/work_dirs/cmt_voxel_015_flatformer/epoch_20.pth \
#     --work-dir ./work_dirs/cmt_test_nold_6_layers \
#     --cfg-options model.test_lidar_retained_layers="[1,1,0,1,0,1,1,1]"

# CUDA_VISIBLE_DEVICES=1 python3 tools/test.py projects/CMT/configs/cmt_voxel_015_flatformer_layerdrop.py \
#     /workspace/mmdetection3d/work_dirs/cmt_voxel_015_flatformer/epoch_20.pth \
#     --work-dir ./work_dirs/cmt_test_nold_5_layers \
#     --cfg-options model.test_lidar_retained_layers="[1,1,0,1,0,1,0,1]"

# CUDA_VISIBLE_DEVICES=1 python3 tools/test.py projects/CMT/configs/cmt_voxel_015_flatformer_layerdrop.py \
#     /workspace/mmdetection3d/work_dirs/cmt_voxel_015_flatformer/epoch_20.pth \
#     --work-dir ./work_dirs/cmt_test_nold_4_layers \
#     --cfg-options model.test_lidar_retained_layers="[1,0,0,1,0,1,0,1]"

CUDA_VISIBLE_DEVICES=3 python3 tools/test.py projects/CMT/configs/cmt_voxel_015_flatformer_layerdrop.py \
    /workspace/mmdetection3d/work_dirs/cmt_voxel_015_flatformer/epoch_20.pth \
    --work-dir ./work_dirs/cmt_test_nold_3_layers \
    --cfg-options model.test_lidar_retained_layers="[1,0,0,1,0,0,0,1]"

CUDA_VISIBLE_DEVICES=3 python3 tools/test.py projects/CMT/configs/cmt_voxel_015_flatformer_layerdrop.py \
    /workspace/mmdetection3d/work_dirs/cmt_voxel_015_flatformer/epoch_20.pth \
    --work-dir ./work_dirs/cmt_test_nold_2_layers \
    --cfg-options model.test_lidar_retained_layers="[1,0,0,0,0,0,0,1]"


CUDA_VISIBLE_DEVICES=3 python3 tools/test.py projects/CMT/configs/cmt_voxel_015_flatformer_layerdrop.py \
    /workspace/mmdetection3d/work_dirs/cmt_voxel_015_flatformer/epoch_20.pth \
    --work-dir ./work_dirs/cmt_test_nold_1_layers \
    --cfg-options model.test_lidar_retained_layers="[1,0,0,0,0,0,0,0]"

CUDA_VISIBLE_DEVICES=3 python3 tools/test.py projects/CMT/configs/cmt_voxel_015_flatformer_layerdrop.py \
    /workspace/mmdetection3d/work_dirs/cmt_voxel_015_flatformer/epoch_20.pth \
    --work-dir ./work_dirs/cmt_test_nold_0_layers \
    --cfg-options model.test_lidar_retained_layers="[0,0,0,0,0,0,0,0]"