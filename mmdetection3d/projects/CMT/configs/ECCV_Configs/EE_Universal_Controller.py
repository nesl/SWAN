_base_=['./Universal_Controller_cmt_voxel_015_flatformer_swin_multicorrupt.py']

model=dict(
    lidar_early_exit_model=dict(
        type='Early_Exit_Lidar'
    ),
    camera_early_exit_model=dict(
        type='Early_Exit_Camera'
    )
)

custom_hooks = [
    dict(type='FreezeLayersHook', train_module_names=['early_exit_lidar', 'early_exit_camera'])
]


# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=8, val_interval=17) # USed to be 16
val_cfg = dict()
test_cfg = dict()

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0,     # Start at 100% of the current LR
        end_factor=0.1,       # Used ot be 0.01 in workign version
        by_epoch=True,
        begin=0,             # Decay starts at the beginning of Epoch 12
        end=8,
        convert_to_iter_based=True
    )
]

find_unused_parameters=True

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-5, weight_decay=0.01), #Was 1e-4
    clip_grad=dict(max_norm=35, norm_type=2)
)

resume=False
load_from='/workspace/mmdetection3d/work_dirs/ECCV_Controller_universal_model/epoch_16.pth'