

_base_=['./EE_Universal_Controller.py']

model = dict(
    enable_pruning=True,
    use_hard_pruning=False,
)


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
    num_workers=12
)

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', 
        lr=1e-3, 
        weight_decay=0.01
    )
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
load_from='/workspace/mmdetection3d/work_dirs/EE_Universal_Controller/epoch_16.pth'