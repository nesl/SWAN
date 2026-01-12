# CUDA_VISIBLE_DEVICES=3 python3 tools/test.py projects/CMT/configs/cmt_swin_layerdrop.py \
#     ./work_dirs/cmt_swin_layerdrop/epoch_50.pth \
#     --work-dir ./work_dirs/cmt_test_full_layers \
#     --cfg-options model.test_img_retained_layers="[1,1,1,1,1,1,1,1,1,1,1,1]"


CUDA_VISIBLE_DEVICES=3 python3 tools/test.py projects/CMT/configs/cmt_swin_layerdrop.py \
    ./work_dirs/cmt_swin_layerdrop/epoch_50.pth \
    --work-dir ./work_dirs/cmt_test_11_layers \
    --cfg-options model.test_img_retained_layers="[1,1,1,1,1,0,1,1,1,1,1,1]"

CUDA_VISIBLE_DEVICES=3 python3 tools/test.py projects/CMT/configs/cmt_swin_layerdrop.py \
    ./work_dirs/cmt_swin_layerdrop/epoch_50.pth \
    --work-dir ./work_dirs/cmt_test_10_layers \
    --cfg-options model.test_img_retained_layers="[1,1,1,1,1,0,1,0,1,1,1,1]"

CUDA_VISIBLE_DEVICES=3 python3 tools/test.py projects/CMT/configs/cmt_swin_layerdrop.py \
    ./work_dirs/cmt_swin_layerdrop/epoch_50.pth \
    --work-dir ./work_dirs/cmt_test_9_layers \
    --cfg-options model.test_img_retained_layers="[1,1,1,1,1,0,1,0,1,0,1,1]"

CUDA_VISIBLE_DEVICES=3 python3 tools/test.py projects/CMT/configs/cmt_swin_layerdrop.py \
    ./work_dirs/cmt_swin_layerdrop/epoch_50.pth \
    --work-dir ./work_dirs/cmt_test_8_layers \
    --cfg-options model.test_img_retained_layers="[1,1,1,0,1,0,1,0,1,0,1,1]"

CUDA_VISIBLE_DEVICES=3 python3 tools/test.py projects/CMT/configs/cmt_swin_layerdrop.py \
    ./work_dirs/cmt_swin_layerdrop/epoch_50.pth \
    --work-dir ./work_dirs/cmt_test_7_layers \
    --cfg-options model.test_img_retained_layers="[1,0,1,0,1,0,1,0,1,0,1,1]"

CUDA_VISIBLE_DEVICES=3 python3 tools/test.py projects/CMT/configs/cmt_swin_layerdrop.py \
    ./work_dirs/cmt_swin_layerdrop/epoch_50.pth \
    --work-dir ./work_dirs/cmt_test_6_layers \
    --cfg-options model.test_img_retained_layers="[1,0,1,0,1,0,1,0,1,0,0,1]"

CUDA_VISIBLE_DEVICES=3 python3 tools/test.py projects/CMT/configs/cmt_swin_layerdrop.py \
    ./work_dirs/cmt_swin_layerdrop/epoch_50.pth \
    --work-dir ./work_dirs/cmt_test_5_layers \
    --cfg-options model.test_img_retained_layers="[1,0,1,0,0,0,1,0,1,0,0,1]"

CUDA_VISIBLE_DEVICES=3 python3 tools/test.py projects/CMT/configs/cmt_swin_layerdrop.py \
    ./work_dirs/cmt_swin_layerdrop/epoch_50.pth \
    --work-dir ./work_dirs/cmt_test_4_layers \
    --cfg-options model.test_img_retained_layers="[1,0,1,0,0,0,1,0,0,0,0,1]"
