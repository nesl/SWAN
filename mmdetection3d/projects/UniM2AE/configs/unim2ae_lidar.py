_base_ = ['./unim2ae_base.py']

model = dict(
    # --- Disable Camera & Fusion ---
    camera_backbone=None,
    camera_vtransform=None,
    camera_decoder=None,
    fusion_module=None,
    voxel_encoder=dict(type='DynamicVFE'), # Ensure matches your VFE class name
)



train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    
    # Augmentations (LiDAR Only)
    dict(type='GlobalRotScaleTrans',
         rot_range=[-0.78539816, 0.78539816],
         scale_ratio_range=[0.95, 1.05],
         translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=[-50, -50, -5, 50, 50, 3]),
    
    # without 'imgs'
    dict(type='Pack3DDetInputs', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
