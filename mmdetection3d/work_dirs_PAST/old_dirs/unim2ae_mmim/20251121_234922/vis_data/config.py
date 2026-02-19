class_names = [
    'car',
    'truck',
    'trailer',
    'bus',
    'construction_vehicle',
    'bicycle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'barrier',
]
custom_hooks = [
    dict(priority='NORMAL', type='UpdateEpochHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'projects.UniM2AE',
        'projects.UniM2AE.utils',
    ])
data_prefix = dict(
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    pts='samples/LIDAR_TOP',
    sweeps='sweeps/LIDAR_TOP')
data_root = './data/nuscenes/'
dataset_type = 'NuScenesDataset'
default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
drop_info = (
    dict({
        0: dict(drop_range=(
            0,
            30,
        ), max_tokens=30),
        1: dict(drop_range=(
            30,
            60,
        ), max_tokens=60),
        2: dict(drop_range=(
            60,
            100,
        ), max_tokens=100),
        3: dict(drop_range=(
            100,
            200,
        ), max_tokens=200),
        4: dict(drop_range=(
            200,
            100000,
        ), max_tokens=256)
    }),
    dict({
        0: dict(drop_range=(
            0,
            30,
        ), max_tokens=30),
        1: dict(drop_range=(
            30,
            60,
        ), max_tokens=60),
        2: dict(drop_range=(
            60,
            100,
        ), max_tokens=100),
        3: dict(drop_range=(
            100,
            200,
        ), max_tokens=200),
        4: dict(drop_range=(
            200,
            100000,
        ), max_tokens=256)
    }),
)
drop_info_test = dict({
    0: dict(drop_range=(
        0,
        30,
    ), max_tokens=30),
    1: dict(drop_range=(
        30,
        60,
    ), max_tokens=60),
    2: dict(drop_range=(
        60,
        100,
    ), max_tokens=100),
    3: dict(drop_range=(
        100,
        200,
    ), max_tokens=200),
    4: dict(drop_range=(
        200,
        100000,
    ), max_tokens=256)
})
drop_info_training = dict({
    0: dict(drop_range=(
        0,
        30,
    ), max_tokens=30),
    1: dict(drop_range=(
        30,
        60,
    ), max_tokens=60),
    2: dict(drop_range=(
        60,
        100,
    ), max_tokens=100),
    3: dict(drop_range=(
        100,
        200,
    ), max_tokens=200),
    4: dict(drop_range=(
        200,
        100000,
    ), max_tokens=256)
})
encoder_blocks = 8
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
fake_voxels_ratio = 0.1
grid_size = [
    468,
    468,
    1,
]
img_size = (
    256,
    704,
)
input_modality = dict(
    use_camera=True,
    use_external=False,
    use_lidar=True,
    use_map=False,
    use_radar=False)
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
loss_weights = dict(
    loss_chamfer_dst_masked=1.0,
    loss_chamfer_dst_unmasked=0.0,
    loss_chamfer_src_masked=1.0,
    loss_chamfer_src_unmasked=0.0,
    loss_num_points_masked=1.0,
    loss_num_points_unmasked=0.0,
    loss_occupied=1.0)
lr = 0.00025
lr_config = dict(
    cyclic_times=1,
    min_lr_ratio=1e-07,
    policy='CosineAnnealing',
    step_ratio_up=0.1,
    target_ratio=(
        100,
        0.001,
    ),
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1)
masking_ratio = 0.7
masking_ratio_img = 0.75
metainfo = dict(classes=[
    'car',
    'truck',
    'trailer',
    'bus',
    'construction_vehicle',
    'bicycle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'barrier',
])
model = dict(
    backbone=dict(
        conv_in_channel=128,
        conv_kwargs=[
            dict(dilation=1, kernel_size=3, padding=1, stride=1),
            dict(dilation=1, kernel_size=3, padding=1, stride=1),
            dict(dilation=2, kernel_size=3, padding=2, stride=1),
        ],
        conv_out_channel=128,
        d_model=[
            128,
            128,
            128,
            128,
            128,
            128,
            128,
            128,
        ],
        debug=True,
        dim_feedforward=[
            256,
            256,
            256,
            256,
            256,
            256,
            256,
            256,
        ],
        masked=True,
        nhead=[
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
        ],
        num_attached_conv=0,
        num_blocks=8,
        output_shape=[
            200,
            200,
        ],
        type='SSTv2'),
    bbox_head=dict(
        feat_channels=192,
        in_channels=192,
        loss_weights=dict(
            loss_chamfer_dst_masked=1.0,
            loss_chamfer_dst_unmasked=0.0,
            loss_chamfer_src_masked=1.0,
            loss_chamfer_src_unmasked=0.0,
            loss_num_points_masked=1.0,
            loss_num_points_unmasked=0.0,
            loss_occupied=1.0),
        num_chamfer_points=10,
        only_masked=True,
        pred_dims=3,
        relative_error=False,
        test_cfg=dict(
            max_num=500,
            min_bbox_size=0,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.2,
            score_thr=0.05,
            use_rotate_nms=True),
        train_cfg=dict(
            allowed_border=0,
            assigner=dict(
                ignore_iof_thr=-1,
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.6,
                type='MaxIoUAssigner'),
            code_weight=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.2,
                0.2,
            ],
            debug=False,
            pos_weight=-1),
        type='ReconstructionHead',
        use_chamfer=True,
        use_fake_voxels=True,
        use_num_points=True),
    camera_backbone=dict(
        ape=False,
        depths=[
            2,
            2,
            6,
            2,
        ],
        drop_path_rate=0.0,
        drop_rate=0.0,
        embed_dim=96,
        img_size=(
            256,
            704,
        ),
        in_chans=3,
        mask_ratio=0.75,
        mlp_ratio=4,
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        patch_norm=True,
        patch_size=4,
        qk_scale=None,
        qkv_bias=True,
        type='MAESwinEncoder',
        window_size=7),
    camera_decoder=dict(
        decoder_depth=1,
        decoder_embed_dim=512,
        decoder_num_heads=16,
        depths=[
            2,
            2,
            18,
            2,
        ],
        embed_dim=96,
        in_chans=3,
        mlp_ratio=4,
        norm_pix_loss=True,
        num_patches=(
            8,
            22,
        ),
        patch_size=4,
        type='MAESwinDecoder'),
    camera_vtransform=dict(
        embed_dims=256,
        in_channels=768,
        mask_ratio=0.75,
        type='VolumeTransform',
        volume_encoder=dict(
            num_layers=6,
            pc_range=[
                -50,
                -50,
                -5,
                50,
                50,
                3,
            ],
            return_intermediate=False,
            transformerlayers=dict(
                attn_cfgs=[
                    dict(
                        deformable_attention=dict(
                            embed_dims=256,
                            num_levels=1,
                            num_points=4,
                            type='MSDeformableAttention3D'),
                        embed_dims=256,
                        pc_range=[
                            -50,
                            -50,
                            -5,
                            50,
                            50,
                            3,
                        ],
                        type='SpatialCrossAttention'),
                ],
                conv_num=2,
                embed_dims=256,
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=(
                    'cross_attn',
                    'norm',
                    'ffn',
                    'norm',
                    'conv',
                ),
                type='volumeLayer'),
            type='volumeEncoder'),
        volume_h=100,
        volume_w=100,
        volume_z=1),
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=-1,
            max_voxels=(
                -1,
                -1,
            ),
            point_cloud_range=[
                -50,
                -50,
                -5,
                50,
                50,
                3,
            ],
            voxel_size=(
                0.5,
                0.5,
                4,
            )),
        voxel_type='dynamic'),
    fusion_module=dict(
        embed_dims=192,
        encoder=dict(
            init_cfg=None,
            num_layers=3,
            transformerlayers=dict(
                attn_cfgs=dict(
                    batch_first=False,
                    dropout=0.0,
                    embed_dims=192,
                    im2col_step=64,
                    init_cfg=None,
                    norm_cfg=None,
                    num_heads=8,
                    num_levels=2,
                    num_points=4,
                    type='MultiScaleDeformableAttention3D'),
                feedforward_channels=768,
                ffn_cfgs=dict(embed_dims=192),
                ffn_dropout=0.0,
                operation_order=(
                    'self_attn',
                    'norm',
                    'ffn',
                    'norm',
                ),
                type='BaseTransformerLayer'),
            type='TransformerLayerSequence'),
        positional_encoding=dict(
            normalize=True, num_feats=64, type='SinePositionalEncoding3D'),
        strides=[
            1,
            1,
        ],
        type='MMIM',
        volume_h=200,
        volume_w=200,
        volume_z=2),
    middle_encoder=dict(
        debug=True,
        drop_info=(
            dict({
                0: dict(drop_range=(
                    0,
                    30,
                ), max_tokens=30),
                1: dict(drop_range=(
                    30,
                    60,
                ), max_tokens=60),
                2: dict(drop_range=(
                    60,
                    100,
                ), max_tokens=100),
                3: dict(drop_range=(
                    100,
                    200,
                ), max_tokens=200),
                4: dict(drop_range=(
                    200,
                    100000,
                ), max_tokens=256)
            }),
            dict({
                0: dict(drop_range=(
                    0,
                    30,
                ), max_tokens=30),
                1: dict(drop_range=(
                    30,
                    60,
                ), max_tokens=60),
                2: dict(drop_range=(
                    60,
                    100,
                ), max_tokens=100),
                3: dict(drop_range=(
                    100,
                    200,
                ), max_tokens=200),
                4: dict(drop_range=(
                    200,
                    100000,
                ), max_tokens=256)
            }),
        ),
        drop_points_th=100,
        fake_voxels_ratio=0.1,
        masking_ratio=0.7,
        mute=True,
        normalize_pos=False,
        pos_temperature=10000,
        pred_dims=3,
        shuffle_voxels=True,
        sparse_shape=(
            200,
            200,
            2,
        ),
        type='SSTInputLayerV2Masked',
        use_chamfer=True,
        use_fake_voxels=True,
        use_num_points=True,
        voxel_size=(
            0.5,
            0.5,
            4,
        ),
        window_shape=(
            16,
            16,
            1,
        )),
    neck=dict(
        d_model=[
            128,
            128,
            128,
            128,
            128,
            128,
        ],
        debug=True,
        dim_feedforward=[
            256,
            256,
            256,
            256,
            256,
            256,
        ],
        nhead=[
            8,
            8,
            8,
            8,
            8,
            8,
        ],
        num_attached_conv=-1,
        num_blocks=6,
        output_shape=(
            200,
            200,
            2,
        ),
        type='SSTv2Decoder',
        use_fake_voxels=True),
    type='UniM2AE',
    voxel_encoder=dict(
        feat_channels=[
            64,
            128,
        ],
        in_channels=5,
        norm_cfg=dict(eps=0.001, momentum=0.01, type='naiveSyncBN1d'),
        point_cloud_range=[
            -50,
            -50,
            -5,
            50,
            50,
            3,
        ],
        return_gt_points=True,
        type='DynamicVFE_New',
        voxel_size=(
            0.5,
            0.5,
            4,
        ),
        with_cluster_center=True,
        with_distance=False,
        with_voxel_center=True))
momentum_config = None
number_of_sweeps = 9
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(
        betas=(
            0.95,
            0.99,
        ), lr=0.00025, type='AdamW', weight_decay=0.01),
    type='AmpOptimWrapper')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ),
    lr=1e-05,
    paramwise_cfg=dict(custom_keys=dict(norm=dict(decay_mult=0.0))),
    type='AdamW',
    weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
point_cloud_range = [
    -50,
    -50,
    -5,
    50,
    50,
    3,
]
relative_error = False
resume = False
runner = dict(max_epochs=24, type='EpochBasedRunner')
shifts_list = [
    (
        0,
        0,
    ),
    (
        8,
        8,
    ),
]
sparse_shape = (
    200,
    200,
    2,
)
test_cfg = dict()
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_infos_val.pkl',
        box_type_3d='LiDAR',
        data_prefix=dict(
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            pts='samples/LIDAR_TOP',
            sweeps='sweeps/LIDAR_TOP'),
        data_root='./data/nuscenes/',
        modality=dict(
            use_camera=True,
            use_external=False,
            use_lidar=True,
            use_map=False,
            use_radar=False),
        pipeline=[
            dict(to_float32=True, type='LoadMultiViewImageFromFiles'),
            dict(
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                pad_empty_sweeps=True,
                remove_close=True,
                sweeps_num=9,
                test_mode=True,
                type='LoadPointsFromMultiSweeps',
                use_dim=[
                    0,
                    1,
                    2,
                    3,
                    4,
                ]),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                bot_pct_lim=[
                    0.0,
                    0.0,
                ],
                final_dim=[
                    256,
                    704,
                ],
                is_train=False,
                rand_flip=False,
                resize_lim=[
                    0.48,
                    0.48,
                ],
                type='ImageAug3D'),
            dict(
                point_cloud_range=[
                    -49.95,
                    -49.95,
                    -4.95,
                    49.95,
                    49.95,
                    2.95,
                ],
                type='PointsRangeFilter'),
            dict(
                point_cloud_range=[
                    -49.95,
                    -49.95,
                    -4.95,
                    49.95,
                    49.95,
                    2.95,
                ],
                type='ObjectRangeFilter'),
            dict(
                classes=[
                    'car',
                    'truck',
                    'trailer',
                    'bus',
                    'construction_vehicle',
                    'bicycle',
                    'motorcycle',
                    'pedestrian',
                    'traffic_cone',
                    'barrier',
                ],
                type='ObjectNameFilter'),
            dict(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
                type='ImageNormalize'),
            dict(
                keys=[
                    'points',
                    'img',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                    'gt_bboxes',
                    'gt_labels',
                ],
                meta_keys=[
                    'cam2img',
                    'ori_cam2img',
                    'lidar2cam',
                    'lidar2img',
                    'cam2lidar',
                    'ori_lidar2img',
                    'img_aug_matrix',
                    'box_type_3d',
                    'sample_idx',
                    'lidar_path',
                    'img_path',
                    'transformation_3d_flow',
                    'pcd_rotation',
                    'pcd_scale_factor',
                    'pcd_trans',
                    'img_aug_matrix',
                    'lidar_aug_matrix',
                    'img_shape',
                    'pad_shape',
                    'imgs_aug',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=False,
        type='NuScenesDataset',
        use_valid_flag=True),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='./data/nuscenes/nuscenes_infos_val.pkl',
    data_root='./data/nuscenes/',
    metric='bbox',
    type='NuScenesMetric')
train_cfg = dict(max_epochs=200, type='EpochBasedTrainLoop', val_interval=201)
train_dataloader = dict(
    batch_size=5,
    dataset=dict(
        ann_file='nuscenes_infos_train.pkl',
        box_type_3d='LiDAR',
        data_prefix=dict(
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            pts='samples/LIDAR_TOP',
            sweeps='sweeps/LIDAR_TOP'),
        data_root='./data/nuscenes/',
        metainfo=dict(classes=[
            'car',
            'truck',
            'trailer',
            'bus',
            'construction_vehicle',
            'bicycle',
            'motorcycle',
            'pedestrian',
            'traffic_cone',
            'barrier',
        ]),
        modality=dict(
            use_camera=True,
            use_external=False,
            use_lidar=True,
            use_map=False,
            use_radar=False),
        pipeline=[
            dict(to_float32=True, type='LoadMultiViewImageFromFiles'),
            dict(
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                pad_empty_sweeps=True,
                remove_close=True,
                sweeps_num=9,
                test_mode=True,
                type='LoadPointsFromMultiSweeps',
                use_dim=[
                    0,
                    1,
                    2,
                    3,
                    4,
                ]),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                bot_pct_lim=[
                    0.0,
                    0.0,
                ],
                final_dim=[
                    256,
                    704,
                ],
                is_train=True,
                rand_flip=True,
                resize_lim=[
                    0.44,
                    0.61,
                ],
                type='ImageAug3D'),
            dict(
                rot_range=[
                    -0.3925,
                    0.3925,
                ],
                scale_ratio_range=[
                    0.95,
                    1.05,
                ],
                translation_std=[
                    0,
                    0,
                    0,
                ],
                type='GlobalRotScaleTrans'),
            dict(
                flip_ratio_bev_horizontal=0.5,
                sync_2d=False,
                type='RandomFlip3D'),
            dict(
                point_cloud_range=[
                    -49.95,
                    -49.95,
                    -4.95,
                    49.95,
                    49.95,
                    2.95,
                ],
                type='PointsRangeFilter'),
            dict(
                point_cloud_range=[
                    -49.95,
                    -49.95,
                    -4.95,
                    49.95,
                    49.95,
                    2.95,
                ],
                type='ObjectRangeFilter'),
            dict(
                classes=[
                    'car',
                    'truck',
                    'trailer',
                    'bus',
                    'construction_vehicle',
                    'bicycle',
                    'motorcycle',
                    'pedestrian',
                    'traffic_cone',
                    'barrier',
                ],
                type='ObjectNameFilter'),
            dict(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
                type='ImageNormalize'),
            dict(type='SaveImgAug'),
            dict(type='PointShuffle'),
            dict(
                keys=[
                    'points',
                    'img',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                    'gt_bboxes',
                    'gt_labels',
                ],
                meta_keys=[
                    'cam2img',
                    'ori_cam2img',
                    'lidar2cam',
                    'lidar2img',
                    'cam2lidar',
                    'ori_lidar2img',
                    'img_aug_matrix',
                    'box_type_3d',
                    'sample_idx',
                    'lidar_path',
                    'img_path',
                    'transformation_3d_flow',
                    'pcd_rotation',
                    'pcd_scale_factor',
                    'pcd_trans',
                    'img_aug_matrix',
                    'lidar_aug_matrix',
                    'img_shape',
                    'pad_shape',
                    'imgs_aug',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=False,
        type='NuScenesDataset',
        use_valid_flag=True),
    num_workers=12,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(to_float32=True, type='LoadMultiViewImageFromFiles'),
    dict(coord_type='LIDAR', load_dim=5, type='LoadPointsFromFile', use_dim=5),
    dict(
        pad_empty_sweeps=True,
        remove_close=True,
        sweeps_num=9,
        test_mode=True,
        type='LoadPointsFromMultiSweeps',
        use_dim=[
            0,
            1,
            2,
            3,
            4,
        ]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        bot_pct_lim=[
            0.0,
            0.0,
        ],
        final_dim=[
            256,
            704,
        ],
        is_train=True,
        rand_flip=True,
        resize_lim=[
            0.44,
            0.61,
        ],
        type='ImageAug3D'),
    dict(
        rot_range=[
            -0.3925,
            0.3925,
        ],
        scale_ratio_range=[
            0.95,
            1.05,
        ],
        translation_std=[
            0,
            0,
            0,
        ],
        type='GlobalRotScaleTrans'),
    dict(flip_ratio_bev_horizontal=0.5, sync_2d=False, type='RandomFlip3D'),
    dict(
        point_cloud_range=[
            -49.95,
            -49.95,
            -4.95,
            49.95,
            49.95,
            2.95,
        ],
        type='PointsRangeFilter'),
    dict(
        point_cloud_range=[
            -49.95,
            -49.95,
            -4.95,
            49.95,
            49.95,
            2.95,
        ],
        type='ObjectRangeFilter'),
    dict(
        classes=[
            'car',
            'truck',
            'trailer',
            'bus',
            'construction_vehicle',
            'bicycle',
            'motorcycle',
            'pedestrian',
            'traffic_cone',
            'barrier',
        ],
        type='ObjectNameFilter'),
    dict(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
        type='ImageNormalize'),
    dict(type='SaveImgAug'),
    dict(type='PointShuffle'),
    dict(
        keys=[
            'points',
            'img',
            'gt_bboxes_3d',
            'gt_labels_3d',
            'gt_bboxes',
            'gt_labels',
        ],
        meta_keys=[
            'cam2img',
            'ori_cam2img',
            'lidar2cam',
            'lidar2img',
            'cam2lidar',
            'ori_lidar2img',
            'img_aug_matrix',
            'box_type_3d',
            'sample_idx',
            'lidar_path',
            'img_path',
            'transformation_3d_flow',
            'pcd_rotation',
            'pcd_scale_factor',
            'pcd_trans',
            'img_aug_matrix',
            'lidar_aug_matrix',
            'img_shape',
            'pad_shape',
            'imgs_aug',
        ],
        type='Pack3DDetInputs'),
]
use_chamfer = True
use_fake_voxels = True
use_num_points = True
val_cfg = dict()
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='nuscenes_infos_val.pkl',
        box_type_3d='LiDAR',
        data_prefix=dict(
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            pts='samples/LIDAR_TOP',
            sweeps='sweeps/LIDAR_TOP'),
        data_root='./data/nuscenes/',
        modality=dict(
            use_camera=True,
            use_external=False,
            use_lidar=True,
            use_map=False,
            use_radar=False),
        pipeline=[
            dict(to_float32=True, type='LoadMultiViewImageFromFiles'),
            dict(
                coord_type='LIDAR',
                load_dim=5,
                type='LoadPointsFromFile',
                use_dim=5),
            dict(
                pad_empty_sweeps=True,
                remove_close=True,
                sweeps_num=9,
                test_mode=True,
                type='LoadPointsFromMultiSweeps',
                use_dim=[
                    0,
                    1,
                    2,
                    3,
                    4,
                ]),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True),
            dict(
                bot_pct_lim=[
                    0.0,
                    0.0,
                ],
                final_dim=[
                    256,
                    704,
                ],
                is_train=False,
                rand_flip=False,
                resize_lim=[
                    0.48,
                    0.48,
                ],
                type='ImageAug3D'),
            dict(
                point_cloud_range=[
                    -49.95,
                    -49.95,
                    -4.95,
                    49.95,
                    49.95,
                    2.95,
                ],
                type='PointsRangeFilter'),
            dict(
                point_cloud_range=[
                    -49.95,
                    -49.95,
                    -4.95,
                    49.95,
                    49.95,
                    2.95,
                ],
                type='ObjectRangeFilter'),
            dict(
                classes=[
                    'car',
                    'truck',
                    'trailer',
                    'bus',
                    'construction_vehicle',
                    'bicycle',
                    'motorcycle',
                    'pedestrian',
                    'traffic_cone',
                    'barrier',
                ],
                type='ObjectNameFilter'),
            dict(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
                type='ImageNormalize'),
            dict(
                keys=[
                    'points',
                    'img',
                    'gt_bboxes_3d',
                    'gt_labels_3d',
                    'gt_bboxes',
                    'gt_labels',
                ],
                meta_keys=[
                    'cam2img',
                    'ori_cam2img',
                    'lidar2cam',
                    'lidar2img',
                    'cam2lidar',
                    'ori_lidar2img',
                    'img_aug_matrix',
                    'box_type_3d',
                    'sample_idx',
                    'lidar_path',
                    'img_path',
                    'transformation_3d_flow',
                    'pcd_rotation',
                    'pcd_scale_factor',
                    'pcd_trans',
                    'img_aug_matrix',
                    'lidar_aug_matrix',
                    'img_shape',
                    'pad_shape',
                    'imgs_aug',
                ],
                type='Pack3DDetInputs'),
        ],
        test_mode=False,
        type='NuScenesDataset',
        use_valid_flag=True),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='./data/nuscenes/nuscenes_infos_val.pkl',
    data_root='./data/nuscenes/',
    metric='bbox',
    type='NuScenesMetric')
val_pipeline = [
    dict(to_float32=True, type='LoadMultiViewImageFromFiles'),
    dict(coord_type='LIDAR', load_dim=5, type='LoadPointsFromFile', use_dim=5),
    dict(
        pad_empty_sweeps=True,
        remove_close=True,
        sweeps_num=9,
        test_mode=True,
        type='LoadPointsFromMultiSweeps',
        use_dim=[
            0,
            1,
            2,
            3,
            4,
        ]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        bot_pct_lim=[
            0.0,
            0.0,
        ],
        final_dim=[
            256,
            704,
        ],
        is_train=False,
        rand_flip=False,
        resize_lim=[
            0.48,
            0.48,
        ],
        type='ImageAug3D'),
    dict(
        point_cloud_range=[
            -49.95,
            -49.95,
            -4.95,
            49.95,
            49.95,
            2.95,
        ],
        type='PointsRangeFilter'),
    dict(
        point_cloud_range=[
            -49.95,
            -49.95,
            -4.95,
            49.95,
            49.95,
            2.95,
        ],
        type='ObjectRangeFilter'),
    dict(
        classes=[
            'car',
            'truck',
            'trailer',
            'bus',
            'construction_vehicle',
            'bicycle',
            'motorcycle',
            'pedestrian',
            'traffic_cone',
            'barrier',
        ],
        type='ObjectNameFilter'),
    dict(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
        type='ImageNormalize'),
    dict(
        keys=[
            'points',
            'img',
            'gt_bboxes_3d',
            'gt_labels_3d',
            'gt_bboxes',
            'gt_labels',
        ],
        meta_keys=[
            'cam2img',
            'ori_cam2img',
            'lidar2cam',
            'lidar2img',
            'cam2lidar',
            'ori_lidar2img',
            'img_aug_matrix',
            'box_type_3d',
            'sample_idx',
            'lidar_path',
            'img_path',
            'transformation_3d_flow',
            'pcd_rotation',
            'pcd_scale_factor',
            'pcd_trans',
            'img_aug_matrix',
            'lidar_aug_matrix',
            'img_shape',
            'pad_shape',
            'imgs_aug',
        ],
        type='Pack3DDetInputs'),
]
voxel_size = (
    0.5,
    0.5,
    4,
)
window_shape = (
    16,
    16,
    1,
)
work_dir = './work_dirs/unim2ae_mmim'
