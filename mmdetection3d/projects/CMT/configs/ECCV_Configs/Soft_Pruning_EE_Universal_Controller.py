

base=['./EE_Universal_Controller.py']
model = dict(
    type='CmtDetector',
    enable_pruning=True,
    use_hard_pruning=False,
)

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

train_dataloader = dict(
    batch_size=8,
    num_workers=12,
    persistent_workers=True
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

custom_hooks = [
    dict(type='FreezeLayersHook', train_module_names=['lidar_pruner', 'img_pruner'])
]


log_processor = dict(window_size=50)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50)
)


randomness = dict(
    seed=100,
    diff_rank_seed=True,
    # deterministic=True
)

resume=False
load_from='/workspace/mmdetection3d/work_dirs/ECCV_EE_Controller_universal_model/epoch_16.pth'