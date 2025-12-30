_base_ = [
    'nuscenes.py',
    'cosine_2x.py',
    '../../../configs/_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['projects.UniM2AE', 'projects.Camera_pretrain'], 
    allow_failed_imports=False
)

# -------------------model--------------------
img_size = (256, 704)
masking_ratio_img = 0.75

model = dict(
    type='CAMERA_PRETRAIN',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=False, 
    ),
    
    # ----------------- Encoder -----------------
    img_backbone=dict(
        type='MAESwinEncoder',
        img_size=img_size,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.1,
        drop_path_rate=0.2,     # admn use 0.2 too for vit
        ape=False,
        patch_norm=True,
        mask_ratio=masking_ratio_img
    ),
    
    # ----------------- Decoder (Head) -----------------
    img_head=dict(
        type='MAESwinDecoder',
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        mlp_ratio=4,
        num_patches=(8, 22), # Derived from img_size (256/32, 704/32) -> (8, 22)
        decoder_embed_dim=512,
        decoder_depth=3,
        decoder_num_heads=16,
        norm_pix_loss=False,
    ),
)
auto_scale_lr = dict(enable=True, base_batch_size=32)
# ----------------- Optimizer -----------------
lr = 1e-5
optim_wrapper = dict(
    type='OptimWrapper', 
    optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=35.0, norm_type=2),
)



param_scheduler = [
    # learning rate scheduler
    # During the first 8 epochs, learning rate increases from 0 to lr * 10
    # during the next 12 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
     dict(
        type='LinearLR',
        start_factor=1e-8,
        begin=0,
        end=2000,
        by_epoch=False
        ),
    dict(
        type='CosineAnnealingLR',
        T_max=8,
        eta_min=lr * 10,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=12,
        eta_min=lr * 1e-4,
        begin=8,
        end=50,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=8,
        eta_min=0.85 / 0.95,
        begin=0,
        end=8,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=50,
        eta_min=1,
        begin=8,
        end=50,
        by_epoch=True,
        convert_to_iter_based=True)
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_interval=51 # No validation needed for pure pretrain usually
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=3,
    )
)

custom_hooks = [
    dict(
        type='UpdateEpochHook',
        priority='NORMAL'
    )
]