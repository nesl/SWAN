_base_ = ['/workspace/mmdetection3d/projects/CMT/configs/Gsize_256_Configs/ADMN_cmt_voxel_015_flatformer_swin_multicorrupt_hard_pruner.py']

voxel_size = (0.15, 0.15, 8) 
grid_size = (720, 720, 1)
window_shape=(16, 16, 1)
sparse_shape = (720, 720, 1)
out_size_factor = 4
point_cloud_range = [-54.0, -54.0, -5.0, 53.95, 53.95, 2.95]
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

corruption_root = '/workspace/mmdetection3d/data/multicorrupt'  # Root directory where corrupted data is stored
camera_corruption = None  # 'fog', 'snow', 'temporalmisalignment', 'brightness', 'dark', 'missingcamera', 'motionblur', None
lidar_corruption = None  # 'pointsreducing', 'beamsreducing', 'snow', 'fog', 'spatialmisalignment', 'temporalmisalignment', 'motionblur', None
severity_distribution = {3:1}  # Only sample severity 3


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
            'lidar_path', 'img_path', 'num_pts_feats', 'corruption_info'
        ]
    )

]

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NuScenesCorruptDataset',
        data_root='/workspace/mmdetection3d/data/nuscenes/',
        ann_file='nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=True,
        data_prefix=data_prefix,
        lidar_corruption=lidar_corruption,
        camera_corruption=camera_corruption,
        corruption_root=corruption_root,
        use_valid_flag=True,
        severity_distribution = severity_distribution,
        return_corruption_info = True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'))

train_dataloader = val_dataloader
test_dataloader = val_dataloader