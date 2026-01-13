# If point cloud range is changed, the models should also change their point
# cloud range accordingly

point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
number_of_sweeps = 9  # Extra sweeps to be merged. Max is 9 extra.
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
metainfo = dict(classes=class_names)
dataset_type = 'NuScenesDataset'
data_root = './data/nuscenes/'
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    sweeps='sweeps/LIDAR_TOP')
# Uncomment the following if use ceph or other file clients.
# See https://mmcv.readthedocs.io/en/latest/api.html#mmcv.fileio.FileClient
# for more details.
# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/nuscenes/': 's3://nuscenes/nuscenes/',
#         'data/nuscenes/': 's3://nuscenes/nuscenes/'
#     }))

train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
    ),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=number_of_sweeps,
        pad_empty_sweeps=True,
        remove_close=True,
        test_mode=True,
        use_dim=[0, 1, 2, 3, 4]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ImageAug3D_UniM2AE', 
         final_dim=[320, 800],
         resize_lim=[0.44, 0.61],
         bot_pct_lim=[0.0, 0.0],
         rand_flip=True,
         is_train=True,
    ),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5, sync_2d=False), # TODO: Changed from RandomFlip3Dv2
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ImageNormalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
    dict(type='SaveImgAug'),
    # dict(type='ImageNormalize_mae', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
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
            'lidar_aug_matrix', 'img_shape', 'pad_shape', 'imgs_aug'
        ]
    )
]

val_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
    ),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=number_of_sweeps,
        pad_empty_sweeps=True,
        remove_close=True,
        test_mode=True,
        use_dim=[0, 1, 2, 3, 4]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # Originally it is ImageAug_UniM2AE, but it doesnt exist 
    dict(type='ImageAug3D_UniM2AE', 
         final_dim=[320, 800],
         resize_lim=[0.48, 0.48],
         bot_pct_lim=[0.0, 0.0],
         rand_flip=False,
         is_train=False,
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ImageNormalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],),
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
            'pcd_scale_factor', 'pcd_trans', 
            'lidar_aug_matrix', 'img_shape', 'pad_shape', 'imgs_aug'
        ]
    )
]


train_dataloader = dict(
    batch_size=5,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
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
        box_type_3d='LiDAR'))
val_dataloader=dict(
    batch_size=1, 
    num_workers=1, 
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset = dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='nuscenes_infos_val.pkl',
        pipeline=val_pipeline,
        modality=input_modality,
        test_mode=False,
        data_prefix=data_prefix,
        use_valid_flag=True,
        box_type_3d='LiDAR'
    )
)
test_dataloader = val_dataloader
val_cfg = dict()
test_cfg = dict()
val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + 'nuscenes_infos_val.pkl',
    metric='bbox')
test_evaluator = val_evaluator
