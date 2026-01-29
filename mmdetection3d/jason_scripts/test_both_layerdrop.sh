CUDA_VISIBLE_DEVICES=3 python3 tools/test.py projects/CMT/configs/cmt_voxel_015_flatformer_swin_both_pretrained.py work_dirs/cmt_voxel_015_flatformer_swin_both_pretrained/epoch_12.pth\
    --work-dir ./work_dirs/temp\
    --cfg-options model.test_img_retained_layers="[1,0,0,0,0,0,0,0,0,0,0,0]"\
    model.test_lidar_retained_layers="[1,1,1,1,1,1,1,1]"
