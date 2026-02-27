_base_=['./Universal_Controller_cmt_voxel_015_flatformer_swin_multicorrupt.py']

model=dict(
    lidar_early_exit_model=dict(
        type='Early_Exit_Lidar_Mean'
    ),
    camera_early_exit_model=dict(
        type='Early_Exit_Camera_Mean'
    )
)

custom_hooks = [
    dict(type='FreezeLayersHook', train_module_names=['early_exit_lidar', 'early_exit_camera'])
]


# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=16, val_interval=17)
val_cfg = dict()
test_cfg = dict()

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0,     # Start at 100% of the current LR
        end_factor=0.01,       # Gradually decay to 10% of the LR
        by_epoch=True,
        begin=0,             # Decay starts at the beginning of Epoch 12
        end=8,
        convert_to_iter_based=True
    )
]

find_unused_parameters=True

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2)
)

resume=False
load_from='/workspace/mmdetection3d/work_dirs/ECCV_Controller_universal_model/epoch_16.pth'