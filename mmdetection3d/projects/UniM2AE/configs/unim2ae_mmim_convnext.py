# Convnext Version
_base_ = [
    'nuscenes.py',
    'cosine_2x.py',
    '../../../configs/_base_/default_runtime.py', # Why does including this help resolve the mmengine registry problem??
]
custom_imports = dict(
    imports=['projects.UniM2AE', 'projects.UniM2AE.utils'], allow_failed_imports=False)
# -------------------model--------------------
voxel_size = (0.5, 0.5, 4)
sparse_shape = (200, 200, 2)
point_cloud_range = [-50, -50, -5, 50, 50, 3]
encoder_blocks = 8
img_size = (256, 704)
grid_size = [468, 468, 1]

use_chamfer, use_num_points, use_fake_voxels = True, True, True
relative_error = False
masking_ratio = 0.7
masking_ratio_img = 0.75
fake_voxels_ratio = 0.1
loss_weights = dict(
    loss_occupied=1.,
    loss_num_points_masked=1.,
    loss_chamfer_src_masked=1.,
    loss_chamfer_dst_masked=1.,
    loss_num_points_unmasked=0.,
    loss_chamfer_src_unmasked=0.,
    loss_chamfer_dst_unmasked=0.
)
window_shape = (16, 16, 1) # 12 * 0.32m
shifts_list = [(0, 0), (window_shape[0]//2, window_shape[1]//2)]
drop_info_training = {
    0: {'max_tokens': 30, 'drop_range': (0, 30)},
    1: {'max_tokens': 60, 'drop_range': (30, 60)},
    2: {'max_tokens': 100, 'drop_range': (60, 100)},
    3: {'max_tokens': 200, 'drop_range': (100, 200)},
    4: {'max_tokens': 256, 'drop_range': (200, 100000)},
}
drop_info_test = {
    0: {'max_tokens': 30, 'drop_range': (0, 30)},
    1: {'max_tokens': 60, 'drop_range': (30, 60)},
    2: {'max_tokens': 100, 'drop_range': (60, 100)},
    3: {'max_tokens': 200, 'drop_range': (100, 200)},
    4: {'max_tokens': 256, 'drop_range': (200, 100000)},
}
drop_info = (drop_info_training, drop_info_test)

model = dict(
    type='UniM2AE',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='dynamic',  # <--- here
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=[-50, -50, -5, 50, 50, 3],
            voxel_size=voxel_size,
            max_voxels=(-1, -1)
        ),
    ),
    fusion_module=dict(
        type='MMIM',
        volume_h=sparse_shape[0],
        volume_w=sparse_shape[1],
        volume_z=sparse_shape[2],
        embed_dims=192,
        strides=[1, 1],
        positional_encoding=dict(
            type='SinePositionalEncoding3D',
            num_feats=192 // 3,
            normalize=True
        ),
        encoder=dict(
            type='TransformerLayerSequence',
            num_layers=3,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention3D',
                    embed_dims=192,
                    num_heads=8,
                    num_levels=2,
                    num_points=4,
                    im2col_step=64, # it seems like setting this don't necessary work, the the value have to fulfill (batch_size *6 )% im2col_step ==0
                    dropout=0.0,
                    batch_first=False,
                    norm_cfg=None,
                    init_cfg=None),
                ffn_cfgs=dict(
                    embed_dims=192),
                feedforward_channels=192 * 4,
                ffn_dropout=0.0,
                operation_order=('self_attn', 'norm', 'ffn', 'norm')),
            init_cfg=None
        ),
    ),

    voxel_encoder=dict(
        type='DynamicVFE_New',
        in_channels=5,
        feat_channels=[64, 128],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        return_gt_points=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)
    ),

    middle_encoder=dict(
        type='SSTInputLayerV2Masked',
        window_shape=window_shape,
        sparse_shape=sparse_shape,
        voxel_size=voxel_size,
        shuffle_voxels=True,
        debug=True,
        drop_info=drop_info,
        pos_temperature=10000,
        normalize_pos=False,
        mute=True,
        masking_ratio=masking_ratio,
        drop_points_th=100,
        pred_dims=3,  # x, y, z
        use_chamfer=use_chamfer,
        use_num_points=use_num_points,
        use_fake_voxels=use_fake_voxels,
        fake_voxels_ratio=fake_voxels_ratio
    ),
    
    backbone=dict(
        type='SSTv2',
        d_model=[128, ] * encoder_blocks,
        nhead=[8, ] * encoder_blocks,
        num_blocks=encoder_blocks,
        dim_feedforward=[256, ] * encoder_blocks,
        output_shape=[200, 200],  # tot_point_cloud_range / voxel_size (50+50)/0.5
        num_attached_conv=0,
        # checkpoint_blocks=[0, 1, 2, 3, 4, 5, 6, 7], # Save the gpu memory but get slower
        conv_kwargs=[
            dict(kernel_size=3, dilation=1, padding=1, stride=1),
            dict(kernel_size=3, dilation=1, padding=1, stride=1),
            dict(kernel_size=3, dilation=2, padding=2, stride=1),
        ],
        conv_in_channel=128,
        conv_out_channel=128,
        debug=True,
        masked=True,
    ),

    neck=dict(
        type='SSTv2Decoder',
        d_model=[128, ] * 6,
        nhead=[8, ] * 6,
        num_blocks=6,
        dim_feedforward=[256, ] * 6,
        output_shape=sparse_shape,
        num_attached_conv=-1,
        debug=True,
        use_fake_voxels=use_fake_voxels,
    ),

    bbox_head=dict(
        type='ReconstructionHead',
        in_channels=192,
        feat_channels=192,
        num_chamfer_points=10,
        pred_dims=3,
        only_masked=True,
        relative_error=relative_error,
        loss_weights=loss_weights,
        use_chamfer=use_chamfer,
        use_num_points=use_num_points,
        use_fake_voxels=use_fake_voxels,
        train_cfg=dict(
            assigner=dict(
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
            allowed_border=0,
            code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
                use_rotate_nms=True,
                nms_across_levels=False,
                nms_pre=1000,   # Changed from initial 4096 
                nms_thr=0.2,    # Changed from initial 0.25
                score_thr=0.05, # Changed from initial 0.1
                min_bbox_size=0,
                max_num=500
        )
    ),
    
    camera_backbone=dict(
        type='MAEConvNeXtEncoder',  # The new class name
        arch='tiny',                # Channels: 768 (matches Swin Tiny output)
        img_size=img_size,
        in_chans=3,
        mask_ratio=masking_ratio_img,
        use_grn=True,               # ConvNeXt V2
        sparse = True,               # Sparse Attention
        drop_path_rate=0.2,         # Standard for Tiny
    ),
    
    camera_vtransform=dict(
        type='VolumeTransform',
        volume_h=sparse_shape[0]//2,
        volume_w=sparse_shape[1]//2,
        volume_z=sparse_shape[2]//2,
        embed_dims=256,
        in_channels=768,
        mask_ratio=masking_ratio_img,
        volume_encoder=dict(
            type='volumeEncoder',
            num_layers=6,
            pc_range=point_cloud_range,
            return_intermediate=False,
            transformerlayers=dict(
                type='volumeLayer',
                attn_cfgs=[
                    dict(
                        type='SpatialCrossAttention',
                        pc_range=point_cloud_range,
                        deformable_attention=dict(
                            type='MSDeformableAttention3D',
                            embed_dims=256,
                            num_points=4,
                            num_levels=1),
                        embed_dims=256,
                    )
                ],
                feedforward_channels=256*2,
                ffn_dropout=0.1,
                embed_dims=256,
                conv_num=2,
                operation_order=('cross_attn', 'norm',
                                    'ffn', 'norm', 'conv')
            ),
        ),
    ),
    
    camera_decoder=dict(
        type='MAESwinDecoder',
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 18, 2],
        mlp_ratio=4,
        # decoder settings
        num_patches=(8, 22),
        decoder_embed_dim=512,
        decoder_depth=1,
        decoder_num_heads=16,
        norm_pix_loss=True,
    ),

)


# optimizer
lr = 1e-4  # max learning rate
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
    # automatic_mixed_precision=dict(
    #     enabled=True,        # ← enables FP16
    #     loss_scale='dynamic' # can be 'dynamic' or a fixed float
    # ))
)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-7)

momentum_config = None

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=200,
    val_interval=201
)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,          # Save every 5 epochs
        max_keep_ckpts=3,    # (Optional) keep only 3 recent checkpoints
    )
)
custom_hooks = [
    dict(
        type='UpdateEpochHook',
        priority='NORMAL'
    )
]

# runtime settings
# workflow = [("train", 1)]
# checkpoint_config = dict(
#     interval=5,
#     max_keep_ckpts=5,
# )

# fp16 = dict(loss_scale=32.0)

