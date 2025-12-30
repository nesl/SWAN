_base_ = ['../../../configs/_base_/default_runtime.py']
custom_imports = dict(
    imports=[ 'projects.LIDAR_Pretrain', 'projects.UniM2AE'], allow_failed_imports=False)

# model settings
# Voxel size for voxel encoder
# Usually voxel size is changed consistently with the point cloud range
# If point cloud range is modified, do remember to change all related
# keys in the config.
# voxel_size = [0.075, 0.075, 0.2]
voxel_size = (0.3, 0.3, 4)
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
grid_size = (360, 360, 2) 
sparse_shape = (360,360,2) 
window_shape = (16, 16, 1)
encoder_blocks = 8
out_size_factor = 4


class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

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
drop_info_training = {
    0: {'max_tokens': 30, 'drop_range': (0, 30)},
    1: {'max_tokens': 60, 'drop_range': (30, 60)},
    2: {'max_tokens': 100, 'drop_range': (60, 100)},
    3: {'max_tokens': 200, 'drop_range': (100, 200)},
    4: {'max_tokens': 256, 'drop_range': (200, 100000)},
}
drop_info_test = drop_info_training

drop_info = (drop_info_training, drop_info_test)
# Loss Settings
use_chamfer = True
use_num_points = True
use_fake_voxels = True
loss_weights = dict(
    loss_occupied=1.0,
    loss_num_points_masked=1.0,
    loss_chamfer_src_masked=1.0,
    loss_chamfer_dst_masked=1.0,
    loss_num_points_unmasked=0.0, # Usually don't compute loss on visible parts
    loss_chamfer_src_unmasked=0.0,
    loss_chamfer_dst_unmasked=0.0
)
backend_args = None

model = dict(
    type='LIDAR_PRETRAIN',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='dynamic',
        voxel_layer=dict(
            max_num_points=-1,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(-1, -1)
        ),
    ),
    pts_voxel_encoder=dict(
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
    pts_middle_encoder=dict(
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
        masking_ratio=0.7,
        drop_points_th=100,
        pred_dims=3,
        use_chamfer=use_chamfer,
        use_num_points=use_num_points,
        use_fake_voxels=use_fake_voxels,
        fake_voxels_ratio=0.1
    ),
    pts_backbone=dict(
        type='SSTv2',
        d_model=[128] * 8,
        nhead=[8] * 8,
        num_blocks=8,
        dim_feedforward=[256] * 8,
        output_shape=[200, 200], 
        num_attached_conv=0, # Keep sparse!
        conv_in_channel=128,
        conv_out_channel=128,
        debug=True,
        masked=True, #
    ),
    pts_neck = dict(
        type='SSTv2Decoder',
        d_model=[128] * 6,
        nhead=[8] * 6,
        num_blocks=6,
        dim_feedforward=[256] * 6,
        output_shape=sparse_shape,
        num_attached_conv=0,
        debug=True,
        use_fake_voxels=use_fake_voxels,
        in_channel=128, 
    ),
    bbox_head=dict(
        type='ReconstructionHead',
        in_channels=128, # Must match decoder d_model
        feat_channels=128,
        num_chamfer_points=20,
        pred_dims=3,
        only_masked=True,
        relative_error=False,
        loss_weights=loss_weights,
        use_chamfer=use_chamfer,
        use_num_points=use_num_points,
        use_fake_voxels=use_fake_voxels,
        train_cfg=None,
        test_cfg=None 
        
    )
)

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
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
        backend_args=backend_args))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),

    dict(
        type='ObjectSample',
        db_sampler= db_sampler
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    # Actually, 'GridMask' is not used here
    dict(type='PointShuffle'),
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
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats', 'gt_bboxes_3d', 'gt_labels_3d'
        ])
]

train_dataloader = dict(
    batch_size=2,
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

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

optim_wrapper = dict(
    type='OptimWrapper',  # Native Mixed Precision wrapper
    
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.01
    ),
    # Parameter-specific learning rates move here
    paramwise_cfg=dict(
        custom_keys={
            'pts_backbone': dict(lr_mult=0.1),
        }
    ),
    # Gradient clipping moves here
    clip_grad=dict(max_norm=35, norm_type=2)
)

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
# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=16)
log_processor = dict(window_size=50)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1))
custom_hooks = [dict(type='DisableObjectSampleHook', disable_after_epoch=15)]
