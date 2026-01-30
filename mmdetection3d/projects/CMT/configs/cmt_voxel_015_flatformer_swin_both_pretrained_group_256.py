'''
We train from the pretrained lidar checkpoint that does about 0.62 mAP
Unfreeze lidar but make sure the learning rate is low
Train image with modal masking to make sure that the image backbone actually learns

Final result about 0.66 mAP, which is good w the limited image backbone resolution. 
TODO test the image only performance, may be able to push more performance from this

'''




_base_ = ['../../../configs/_base_/default_runtime.py']
custom_imports = dict(
    imports=['projects.BEVFusion.bevfusion', 'projects.UniM2AE', 'projects.CMT'], allow_failed_imports=False)

voxel_size = (0.15, 0.15, 8) 
grid_size = (720, 720, 1)
window_shape=(16, 16, 1)
sparse_shape = (720, 720, 1)
out_size_factor = 4
point_cloud_range = [-54.0, -54.0, -5.0, 53.95, 53.95, 2.95]
# point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
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

metainfo = dict(classes=class_names)
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    sweeps='sweeps/LIDAR_TOP')
input_modality = dict(use_lidar=True, use_camera=True)

backend_args = None


model = dict(
    type='CmtDetector',
    enable_modal_mask=True,
    layerdrop_rate=0.2,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='dynamic',  # <--- here
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(-1, -1)
        ),
    ),
    pts_voxel_encoder=dict(
        type='DynamicVFE_Efficient',
        in_channels=5,
        feat_channels=[64, 128],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01)
    ),
    pts_middle_encoder=dict(
        type='FlatFormer',
        in_channels=128,
        num_heads=8,
        num_blocks=2,
        activation="gelu",
        window_shape=window_shape,
        sparse_shape=sparse_shape,
        output_shape=[sparse_shape[0], sparse_shape[1]],
        pos_temperature=10000,
        normalize_pos=False,
        group_size=256,
    ),
    
    pts_backbone=dict(
        type='SECOND',
        in_channels=128,
        out_channels=[64, 128],
        layer_nums=[3, 3],
        layer_strides=[2, 2],
        conv_cfg=dict(type='Conv2d', bias=False),
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
    ),
    pts_neck=dict(
        type='SECONDFPN',
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        in_channels=[64, 128],
        upsample_strides=[0.5, 1],
        out_channels=[256, 256]
    ),
   

    # BEGIN CAMERA COMPONENTS

    # Hopefully I can override w my saved model parameters...
    img_backbone = dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[2, 3],
        with_cp=False,
        convert_weights=True,
    ),

    img_neck=dict(
        type='CPFPN',
        in_channels=[384, 768],
        out_channels=256,
        num_outs=2),

    pts_bbox_head=dict(
        type='CmtHead',
        in_channels=512,
        hidden_dim=256,
        downsample_scale=out_size_factor,
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
                iou_cost=dict(type='IoU3DCost_CMT', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
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
        )
    )
)



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
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    
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
    batch_size=12,
    num_workers=16,
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



param_scheduler = [
    dict(
        type='OneCycleLR',
        total_steps=12,            # Total epochs
        by_epoch=True,             # It's an epoch-based scheduler
        eta_max=0.0001,            # Max LR
        pct_start=0.1,             # Max at epoch 2
        div_factor=8.0,            # Matches target_ratio=(8, ...)
        final_div_factor=1e2,      # Standard decay
        convert_to_iter_based=True # Update every iteration, not just epoch end
    )
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=12, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# For the lidar components, set a smaller LR so they dont get overwritten at the start
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2),
)


log_processor = dict(window_size=50)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1))

custom_hooks = [
    dict(type='FreezeSpecificModuleHook', freeze_names=['img_backbone', 'pts_middle_encoder'])
]

randomness = dict(
    seed=100,
    diff_rank_seed=False,
    # deterministic=True
)

find_unused_parameters=True # If we do modal dropping find_unused_parameters must be true or else it throws error

resume = False
load_from='/workspace/mmdetection3d/unified_unimodal_layerdrop_256.pth' # Load the weights from the flatformer
