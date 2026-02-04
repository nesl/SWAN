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
img_size = (320, 800)
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
        window_size=8,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
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
        depths=[2, 2, 6, 2],
        mlp_ratio=4,
        num_patches=(10, 25),
        decoder_embed_dim=256,
        decoder_depth=2,
        decoder_num_heads=4,
        norm_pix_loss=True,
    ),
)
auto_scale_lr = dict(enable=True, base_batch_size=32)
# ----------------- Optimizer -----------------
lr = 2.5e-4 
optim_wrapper = dict(
    type='OptimWrapper', 
    optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=35.0, norm_type=2),
)


param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1/10.0,
        by_epoch=False,
        begin=0,
        end=1000),
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