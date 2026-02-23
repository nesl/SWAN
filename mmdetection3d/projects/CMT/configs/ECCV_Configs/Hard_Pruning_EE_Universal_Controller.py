_base_=['./Soft_Pruning_EE_Universal_Controller.py']
model = dict(
    use_hard_pruning=True,
)
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-6, weight_decay=0.01),
)

custom_hooks = [
    dict(type='FreezeLayersHook', train_module_names=['pts_bbox_head'])
]
train_cfg = dict(by_epoch=True, max_epochs=8, val_interval=17)
resume=False
load_from='/workspace/mmdetection3d/work_dirs/Soft_Pruning_EE_Universal_Controller/epoch_16.pth'