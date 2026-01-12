# CUDA_VISIBLE_DEVICES=2 python3 tools/test.py projects/CMT/configs/cmt_voxel_015_flatformer_swin_both_pretrained.py \
#     ./work_dirs/cmt_voxel_015_flatformer_swin_both_pretrained/epoch_12.pth \
#     --work-dir ./work_dirs/full_cmt_all_lidar \
#     --cfg-options model.test_img_retained_layers="[0,0,0,0,0,0,0,0,0,0,0,0]" \
#     model.test_lidar_retained_layers="[1,1,1,1,1,1,1,1]"

# CUDA_VISIBLE_DEVICES=2 python3 tools/test.py projects/CMT/configs/cmt_voxel_015_flatformer_swin_both_pretrained.py \
#     ./work_dirs/cmt_voxel_015_flatformer_swin_both_pretrained/epoch_12.pth \
#     --work-dir ./work_dirs/full_cmt_all_img \
#     --cfg-options model.test_img_retained_layers="[1,1,1,1,1,1,1,1,1,1,1,1]" \
#     model.test_lidar_retained_layers="[0,0,0,0,0,0,0,0]"

# CUDA_VISIBLE_DEVICES=2 python3 tools/test.py projects/CMT/configs/cmt_voxel_015_flatformer_swin_both_pretrained.py \
#     ./work_dirs/cmt_voxel_015_flatformer_swin_both_pretrained/epoch_12.pth \
#     --work-dir ./work_dirs/full_cmt_l4_i6 \
#     --cfg-options model.test_img_retained_layers="[1,0,1,0,1,0,1,0,0,1,0,1]" \
#     model.test_lidar_retained_layers="[1,0,1,0,1,0,0,1]"

# CUDA_VISIBLE_DEVICES=2 python3 tools/test.py projects/CMT/configs/cmt_voxel_015_flatformer_swin_both_pretrained.py \
#     ./work_dirs/cmt_voxel_015_flatformer_swin_both_pretrained/epoch_12.pth \
#     --work-dir ./work_dirs/full_cmt_l3_i9 \
#     --cfg-options model.test_img_retained_layers="[1,1,1,0,1,0,1,0,1,1,1,1]" \
#     model.test_lidar_retained_layers="[1,0,0,0,1,0,0,1]"

CUDA_VISIBLE_DEVICES=2 python3 tools/test.py projects/CMT/configs/cmt_voxel_015_flatformer_swin_both_pretrained.py \
    ./work_dirs/cmt_voxel_015_flatformer_swin_both_pretrained/epoch_12.pth \
    --work-dir ./work_dirs/full_cmt_l2_i6 \
    --cfg-options model.test_img_retained_layers="[1,0,1,0,1,0,1,0,0,1,0,1]" \
    model.test_lidar_retained_layers="[1,0,0,0,0,0,0,1]"

CUDA_VISIBLE_DEVICES=2 python3 tools/test.py projects/CMT/configs/cmt_voxel_015_flatformer_swin_both_pretrained.py \
    ./work_dirs/cmt_voxel_015_flatformer_swin_both_pretrained/epoch_12.pth \
    --work-dir ./work_dirs/full_cmt_l2_i12 \
    --cfg-options model.test_img_retained_layers="[1,1,1,1,1,1,1,1,1,1,1,1]" \
    model.test_lidar_retained_layers="[1,0,0,0,0,0,0,1]"

CUDA_VISIBLE_DEVICES=2 python3 tools/test.py projects/CMT/configs/cmt_voxel_015_flatformer_swin_both_pretrained.py \
    ./work_dirs/cmt_voxel_015_flatformer_swin_both_pretrained/epoch_12.pth \
    --work-dir ./work_dirs/full_cmt_l6_i4 \
    --cfg-options model.test_img_retained_layers="[1,0,1,0,0,0,1,0,0,0,0,1]" \
    model.test_lidar_retained_layers="[1,1,0,1,0,1,1,1]"

CUDA_VISIBLE_DEVICES=2 python3 tools/test.py projects/CMT/configs/cmt_voxel_015_flatformer_swin_both_pretrained.py \
    ./work_dirs/cmt_voxel_015_flatformer_swin_both_pretrained/epoch_12.pth \
    --work-dir ./work_dirs/full_cmt_l1_i12 \
    --cfg-options model.test_img_retained_layers="[1,1,1,1,1,1,1,1,1,1,1,1]" \
    model.test_lidar_retained_layers="[1,0,0,0,0,0,0,0]"

