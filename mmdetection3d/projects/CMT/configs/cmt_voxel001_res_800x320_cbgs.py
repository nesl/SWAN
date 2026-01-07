_base_ = ['../../../configs/_base_/default_runtime.py']
custom_imports = dict(
    imports=['projects.CMT'], allow_failed_imports=False)

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
metainfo = dict(classes=class_names)
voxel_size = [0.1, 0.1, 0.2]
grid_size = [1024, 1024, 40]
out_size_factor = 8
evaluation = dict(interval=20)
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

data_prefix = dict(
    pts='samples/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    sweeps='sweeps/LIDAR_TOP')


img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)
    
ida_aug_conf = {
        "resize_lim": (0.47, 0.625),
        "final_dim": (320, 800),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": True,
    }

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='UnifiedObjectSample',
        sample_2d=False,
        mixup_rate=0.5,
        db_sampler=dict(
            type='UnifiedDataBaseSampler',
            data_root='data/nuscenes/',
            info_path=data_root + 'nuscenes_dbinfos_train.new.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(
                    car=5,
                    truck=5,
                    bus=5,
                    trailer=5,
                    construction_vehicle=5,
                    traffic_cone=5,
                    barrier=5,
                    motorcycle=5,
                    bicycle=5,
                    pedestrian=5)),
            classes=class_names,
            sample_groups=dict(
                car=2,
                truck=3,
                construction_vehicle=7,
                bus=4,
                trailer=6,
                barrier=2,
                motorcycle=6,
                bicycle=6,
                pedestrian=2,
                traffic_cone=2),
            points_loader=dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=[0, 1, 2, 3, 4],
            ))),
    dict(type='ModalMask3D', mode='train'),
    dict(
        type='GlobalRotScaleTransAll',
        rot_range=[-0.3925 * 2, 0.3925 * 2],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5]),
    dict(
        type='CustomRandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    #dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats', 'gt_bboxes_3d', 'gt_labels_3d'
        ])
]


test_pipeline = [
    # 1. Load Points (Same as before)
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    # 2. Load Images (Same as before)
    dict(type='LoadMultiViewImageFromFiles'),
    
    # 3. Standard Transforms (Flattened - removed MultiScaleFlipAug3D wrapper)
    # The 'GlobalRotScaleTrans' with 0/1 does nothing, so we can verify weights 
    # more safely by removing it to avoid any rounding errors.
    
    # dict(type='RandomFlip3D'), # Removed because flip=False in your config
    
    dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=False),
    
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    
    dict(type='PadMultiViewImage', size_divisor=32),
    
    # 4. Packaging (Replaces DefaultFormatBundle3D & Collect3D)
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points'], # No GT keys for testing
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar', 
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx', 
            'lidar_path', 'img_path', 'num_pts_feats'
        ]
    )

]


train_dataloader = dict(
    batch_size=2,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='nuscenes_infos_train.new.pkl',
            pipeline=train_pipeline,
            metainfo=metainfo,
            modality=input_modality,
            test_mode=False,
            data_prefix=data_prefix,
            use_valid_flag=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_val.new.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.new.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=21)
val_cfg = dict()
test_cfg = dict()

model = dict(
    type='CmtDetector',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,          # Enable voxelization
        voxel_type='hard',   # <--- CRITICAL: Explicitly set to 'hard' (default)
        voxel_layer=dict(
            max_num_points=10,            # Match your old config
            voxel_size=voxel_size,        # Match your old config
            max_voxels=(120000, 160000),  # Match your old config
            point_cloud_range=point_cloud_range
        )
    ),
    use_grid_mask=True,
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CPFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=2),
    # pts_voxel_encoder=dict(
    #     type='DynamicSimpleVFE',
    #     voxel_size=voxel_size,
    #     point_cloud_range=point_cloud_range,
    # ),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE',
        num_features=5,
    ),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1024, 1024],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CmtHead',
        in_channels=512,
        hidden_dim=256,
        downsample_scale=8,
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        tasks=[
            dict(num_class=10, class_names=[
                'car', 'truck', 'construction_vehicle',
                'bus', 'trailer', 'barrier',
                'motorcycle', 'bicycle',
                'pedestrian', 'traffic_cone'
            ]),
        ],
        bbox_coder=dict(
            type='MultiTaskBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10), 
        separate_head=dict(
            type='SeparateTaskHead', init_bias=-2.19, final_kernel=1),
        transformer=dict(
            type='CmtTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadFlashAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),

                    feedforward_channels=1024, #unused
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        loss_cls=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    ),
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D_CMT',
                # cls_cost=dict(type='ClassificationCost', weight=2.0),
                cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
                pc_range=point_cloud_range,
                code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=grid_size,  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=grid_size,
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            nms_type=None,
            nms_thr=0.2,
            use_rotate_nms=True,
            max_num=200
        )))

optim_wrapper = dict(
    type='AmpOptimWrapper',  # Native Mixed Precision wrapper
    dtype='bfloat16',        # Use 'bfloat16' for your H100. Use 'float16' otherwise.
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.01
    ),
    # Parameter-specific learning rates move here
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.01, decay_mult=5),
            'img_neck': dict(lr_mult=0.1),
        }
    ),
    # Gradient clipping moves here
    clip_grad=dict(max_norm=35, norm_type=2)
)

# NOTE: The old 'custom_fp16=dict(pts_voxel_encoder=False...)' is removed.
# You have already handled this by ensuring your Voxel/Middle encoders
# use correct types or context managers in the code.

# ---------------------------------------------------------
# 2. Learning Rate & Momentum (Replaces 'lr_config' & 'momentum_config')
# ---------------------------------------------------------
# The old config used a 'cyclic' policy with a 40% step up. 
# The closest modern equivalent that handles both LR and Momentum is OneCycleLR.
param_scheduler = [
    dict(
        type='OneCycleLR',
        total_steps=20,            # Total epochs
        by_epoch=True,             # It's an epoch-based scheduler
        eta_max=0.0001,            # Max LR
        pct_start=0.4,             # Matches step_ratio_up=0.4
        div_factor=8.0,            # Matches target_ratio=(8, ...)
        final_div_factor=1e4,      # Standard decay
        convert_to_iter_based=True # Update every iteration, not just epoch end
    )
]

# ---------------------------------------------------------
# 3. Training Loop (Replaces 'total_epochs' & 'workflow')
# ---------------------------------------------------------
train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=20, 
    val_interval=1
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ---------------------------------------------------------
# 4. Hooks & Logging (Replaces 'checkpoint_config' & 'log_config')
# ---------------------------------------------------------
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'),
)
auto_scale_lr = dict(enable=True, base_batch_size=16)
# Tensorboard is now defined in the visualizer, not the hooks
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'), 
        dict(type='TensorboardVisBackend')
    ],
    name='visualizer'
)

# ---------------------------------------------------------
# 5. Runtime Settings
# ---------------------------------------------------------
# 'dist_params' is handled by the launcher arguments now, usually safe to omit
# 'gpu_ids' is deprecated; use CUDA_VISIBLE_DEVICES
log_level = 'INFO'
work_dir = None
load_from = 'paper_checkpoints/nuim_r50.pth'
resume = False
# load_from = '/workspace/mmdetection3d/work_dirs/cmt_voxel0075_vov_1600x640_cbgs/train_original_cmt_9epoch/epoch_9.pth'
# resume=True
# load_from='paper_checkpoints/cmt_converted_spconv.pth'
# resume=False

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01),
#     clip_grad=dict(max_norm=35, norm_type=2))

# optimizer_config = dict(
#     type='CustomFp16OptimizerHook',
#     loss_scale='dynamic',
#     grad_clip=dict(max_norm=35, norm_type=2),
#     custom_fp16=dict(pts_voxel_encoder=False, pts_middle_encoder=False, pts_bbox_head=False))
# param_scheduler = [
#     # 1. Linear Warmup
#     dict(
#         type='LinearLR',
#         start_factor=0.001,    # Initial LR = lr * start_factor
#         by_epoch=False,        # Count warmup in iterations (not epochs)
#         begin=0,
#         end=5000                # Warmup for the first 5000 iterations
#     )
# ]
# momentum_config = dict(
#     policy='cyclic',
#     target_ratio=(0.8947368421052632, 1),
#     cyclic_times=1,
#     step_ratio_up=0.4)
# checkpoint_config = dict(interval=1)
# log_config = dict(
#     interval=50,
#     hooks=[dict(type='TextLoggerHook'),
#            dict(type='TensorboardLoggerHook')])
# dist_params = dict(backend='nccl')
# log_level = 'INFO'
# work_dir = None
# load_from='paper_checkpoints/fcos3d_vovnet_imgbackbone-remapped.pth'
# resume_from = None
# workflow = [('train', 1)]
# gpu_ids = range(0, 8)
