

_base_ = ['/workspace/mmdetection3d/projects/CMT/configs/Gsize_256_Configs/ADMN_universal_cmt_voxel_015_flatformer_swin_multicorrupt.py']
model=dict(
    lidar_early_exit_model=dict(
        type='Early_Exit_Lidar'
    ),
    camera_early_exit_model=dict(
        type='Early_Exit_Camera'
    )
)
