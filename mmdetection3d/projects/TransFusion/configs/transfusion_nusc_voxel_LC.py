# Phase 2 Fusion: finetune with camera on top of voxel_L checkpoint

_base_ = [
    '../../../configs/_base_/default_runtime.py',
]
custom_imports = dict(
    imports=['projects.TransFusion', 'projects.BEVFusion.bevfusion'], allow_failed_imports=False)

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
voxel_size = [0.075, 0.075, 0.2]
grid_size = [1440, 1440, 40]  # (54+54)/0.075=1440, (5+3)/0.2=40
out_size_factor = 8
dataset_type = 'NuScenesDataset'
data_root = '/data/nuscenes/'

lr = 0.0001

backend_args = None
metainfo = dict(classes=class_names)
# Added camera paths for fusion
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    sweeps='sweeps/LIDAR_TOP')

# use_camera=True for fusion
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

# ================= Image Configs =================
img_scale = (800, 448)
num_views = 6
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# ================= Pipelines =================
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args,
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args,
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # Added for camera fusion
    dict(type='LoadMultiViewImageFromFiles', backend_args=backend_args),
 
 
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.3925 * 2, 0.3925 * 2],
    #     scale_ratio_range=[0.9, 1.1],
    #     translation_std=[0.5, 0.5, 0.5]),
    # dict(
    #     type='RandomFlip3D',
    #     sync_2d=True,
    #     flip_ratio_bev_horizontal=0.5,
    #     flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    # Image preprocessing
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    # Added 'img' key for camera fusion
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args,
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args,
    ),
    dict(type='LoadMultiViewImageFromFiles', backend_args=backend_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # Image preprocessing
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img'],
        meta_keys=[
            'box_type_3d', 'sample_idx', 'lidar_path',
            'num_pts_feats', 'lidar2img',  # lidar2img needed for fusion
        ]),
]

# ================= Data Loaders =================
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            metainfo=metainfo,
            modality=input_modality,
            test_mode=False,
            data_prefix=data_prefix,
            use_valid_flag=True,
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
        ann_file='nuscenes_infos_val.pkl',
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
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

model = dict(
    type='TransFusionDetector',
    freeze_img=True,  # freeze image backbone during fusion finetuning
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        pad_size_divisor=32,
        voxel=True,
        voxel_type='hard',
        mean=img_norm_cfg['mean'],
        std=img_norm_cfg['std'],
        voxel_layer=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(120000, 160000),
        )),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE',
        num_features=5,
    ),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1440, 1440],
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
        type='TransFusion_TransFusionHead',
        num_proposals=200,
        auxiliary=True,
        in_channels=256 * 2,
        hidden_channel=128,
        num_classes=len(class_names),
        num_decoder_layers=1,
        fuse_img=True,
        num_views=num_views,
        in_channels_img=256,        # matches FPN out_channels
        out_size_factor_img=4,      # default for ResNet+FPN
        initialize_by_heatmap=True,
        num_heads=8,
        ffn_channel=256,
        dropout=0.1,
        activation='relu',
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2]),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            out_size_factor=out_size_factor,
            voxel_size=voxel_size[:2],
            code_size=10),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        # loss_iou=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=0.0),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25)),
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            point_cloud_range=point_cloud_range,
            grid_size=grid_size,
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='mmdet.FocalLossCost',
                    gamma=2.0,
                    alpha=0.25,
                    weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)))),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=grid_size,
            out_size_factor=out_size_factor,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            nms_type=None)),
)

# Load phase 1 voxel_L checkpoint
load_from = 'work_dirs/transfusion_lidar/epoch_20.pth'

auto_scale_lr = dict(enable=False, base_batch_size=8)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=0.1, norm_type=2))

# 6 epochs for fusion finetuning (shorter than phase 1)
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=2,
        eta_min=lr * 10,
        begin=0,
        end=2,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=4,
        eta_min=lr * 1e-4,
        begin=2,
        end=6,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=2,
        eta_min=0.85 / 0.95,
        begin=0,
        end=2,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=4,
        eta_min=1,
        begin=2,
        end=6,
        by_epoch=True,
        convert_to_iter_based=True),
]

train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)
val_cfg = dict()
test_cfg = dict()
log_processor = dict(window_size=50)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1))
