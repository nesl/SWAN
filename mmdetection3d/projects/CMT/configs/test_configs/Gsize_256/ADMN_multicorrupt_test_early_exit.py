

_base_ = ['/workspace/mmdetection3d/projects/CMT/configs/test_configs/Gsize_256/ADMN_multicorrupt_test.py']

model = dict(
    lidar_early_exit_model=dict(
        type='Early_Exit_Lidar'
    ),
    camera_early_exit_model=dict(
        type='Early_Exit_Camera'
    ),
)