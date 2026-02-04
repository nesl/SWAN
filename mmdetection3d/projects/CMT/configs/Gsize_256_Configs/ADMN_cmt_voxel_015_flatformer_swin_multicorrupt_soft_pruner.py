

_base_ = ['./ADMN_cmt_voxel_015_flatformer_swin_multicorrupt.py']
custom_imports = dict(
    imports=['projects.BEVFusion.bevfusion', 'projects.UniM2AE', 'projects.CMT'], allow_failed_imports=False)

model = dict(
    type='CmtDetector',
    use_grid_mask=True,
    enable_pruning=True,
    use_hard_pruning=False,
)

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=8, val_interval=4)
val_cfg = dict()
test_cfg = dict()

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0,     # Start at 100% of the current LR
        end_factor=0.01,       # Gradually decay to 10% of the LR
        by_epoch=True,
        begin=0,             # Decay starts at the beginning of Epoch 12
        end=6,
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
load_from='DUMMY_PATH'
