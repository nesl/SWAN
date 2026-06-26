_base_=['./FT_Soft_Pruning_EE_Universal_Controller.py']
model = dict(
    use_hard_pruning=True,
)
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-5, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'lidar_pruner': dict(lr_mult=1),
            'img_pruner': dict(lr_mult=1)
        }
    )
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0,     # Start at 100% of the current LR
        end_factor=0.1,       # Gradually decay to 10% of the LR
        by_epoch=True,
        begin=0,             # Decay starts at the beginning of Epoch 12
        end=8,
        convert_to_iter_based=True
    )
]

custom_hooks = [
    dict(type='FreezeLayersHook', train_module_names=['pts_bbox_head'])
]
train_cfg = dict(by_epoch=True, max_epochs=8, val_interval=17)
resume=False
load_from='/workspace/mmdetection3d/work_dirs/FT_Soft_Pruning_EE_Universal_Controller/epoch_16.pth'