base=['./Soft_Pruning_EE_Universal_Controller.py']
model = dict(
    use_hard_pruning=True,
)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-6, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)

custom_hooks = [
    dict(type='FreezeLayersHook', train_module_names=['pts_bbox_head'])
]