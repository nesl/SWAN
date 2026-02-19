# from os import path as osp
# from typing import Optional, Dict, Union, List
# import random

# from mmdet3d.registry import DATASETS
# from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset


# @DATASETS.register_module()
# class NuScenesCorruptDataset(NuScenesDataset):
#     r"""NuScenes Dataset with Dynamic Corruption Support.

#     This class extends NuScenesDataset to support dynamic corruption
#     selection for both LiDAR and camera data with configurable severity
#     probability distributions.

#     Additional Args (compared to NuScenesDataset):
#         lidar_corruption (str or None): Type of LiDAR corruption to apply.
#             Options: 'pointsreducing', 'beamsreducing', 'snow', 'fog', 
#             'spatialmisalignment', 'temporalmisalignment', 'motionblur', None.
#             Defaults to None.
#         camera_corruption (str or None): Type of camera corruption to apply.
#             Options: 'snow', 'fog', 'temporalmisalignment', 'brightness', 
#             'dark', 'missingcamera', 'motionblur', None.
#             Defaults to None.
#         corruption_root (str): Root directory where corrupted data is stored.
#             Defaults to '/data/multicorrupt'.
#         severity_distribution (dict or None): Probability distribution for severities.
#             Example: {1: 0.33, 2: 0.33, 3: 0.34} for uniform distribution.
#             If None, uses uniform distribution across [0, 1, 2, 3].
#             Defaults to None.
#         fixed_severity (int or None): If set, always use this severity level.
#             Overrides severity_distribution. Defaults to None.
#         return_corruption_info (bool): Whether to return corruption metadata
#             in the data_info dict. Defaults to True.
#     """

#     LIDAR_CORRUPTIONS = [
#         'pointsreducing', 'beamsreducing', 'snow', 'fog', 
#         'spatialmisalignment', 'temporalmisalignment', 'motionblur'
#     ]
    
#     CAMERA_CORRUPTIONS = [
#         'snow', 'fog', 'temporalmisalignment', 'brightness', 
#         'dark', 'missingcamera', 'motionblur'
#     ]

#     def __init__(self,
#                  lidar_corruption: Optional[str] = None,
#                  camera_corruption: Optional[str] = None,
#                  corruption_root: str = '/data/multicorrupt',
#                  severity_distribution: Optional[Dict[int, float]] = None,
#                  fixed_severity: Optional[int] = None,
#                  return_corruption_info: bool = True,
#                  **kwargs) -> None:
        
#         self.lidar_corruption = lidar_corruption
#         self.camera_corruption = camera_corruption
#         self.corruption_root = corruption_root
#         self.fixed_severity = fixed_severity
#         self.return_corruption_info = return_corruption_info

#         # # Validate corruption types
#         # if lidar_corruption is not None:
#         #     assert lidar_corruption in self.LIDAR_CORRUPTIONS, \
#         #         f"Invalid LiDAR corruption: {lidar_corruption}. " \
#         #         f"Must be one of {self.LIDAR_CORRUPTIONS}"
        
#         # if camera_corruption is not None:
#         #     assert camera_corruption in self.CAMERA_CORRUPTIONS, \
#         #         f"Invalid camera corruption: {camera_corruption}. " \
#         #         f"Must be one of {self.CAMERA_CORRUPTIONS}"

#         # Setup severity distribution
#         if fixed_severity is not None:
#             assert fixed_severity in [0, 1, 2, 3], \
#                 f"fixed_severity must be 0, 1, 2, or 3, got {fixed_severity}"
#             self.severity_probs = {fixed_severity: 1.0}
#         elif severity_distribution is not None:
#             self._validate_severity_distribution(severity_distribution)
#             self.severity_probs = severity_distribution
#         else:
#             # Default: uniform distribution
#             self.severity_probs = {0 : 1/4, 1: 1/4, 2: 1/4, 3: 1/4}

#         # Call parent constructor
#         super().__init__(**kwargs)

#     def _validate_severity_distribution(self, dist: Dict[int, float]) -> None:
#         """Validate severity probability distribution."""
#         assert all(k in [0, 1, 2, 3] for k in dist.keys()), \
#             "Severity levels must be 0, 1, 2, or 3"
#         assert all(0 <= v <= 1 for v in dist.values()), \
#             "Probabilities must be between 0 and 1"
#         assert abs(sum(dist.values()) - 1.0) < 1e-6, \
#             f"Probabilities must sum to 1, got {sum(dist.values())}"

#     def sample_severity(self) -> int:
#         """Sample a severity level based on the configured distribution.
        
#         Returns:
#             int: Severity level (1, 2, or 3).
#         """
#         severities = list(self.severity_probs.keys())
#         probs = list(self.severity_probs.values())
#         return random.choices(severities, weights=probs, k=1)[0]

#     def get_corruption_path(self, 
#                            original_path: str, 
#                            corruption_type: str, 
#                            severity: int) -> str:
#         """Get the path to corrupted data.
        
#         Args:
#             original_path (str): Original data path relative to data_root.
#             corruption_type (str): Type of corruption.
#             severity (int): Severity level.
            
#         Returns:
#             str: Path to corrupted data.
#         """
#         # Construct corruption path: corruption_root/corruption_type/severity/original_path
#         corrupted_path = osp.join(
#             self.corruption_root,
#             corruption_type,
#             str(severity),
#             original_path
#         )
#         return corrupted_path

#     def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
#         """Process the raw data info with corruption support.

#         Args:
#             info (dict): Raw info dict.

#         Returns:
#             List[dict] or dict: Has `ann_info` in training stage. And
#             all path has been converted to absolute path or corrupted path.
#         """
#         # Sample severities for this sample
#         lidar_severity = self.sample_severity() if self.lidar_corruption else None
#         camera_severity = self.sample_severity() if self.camera_corruption else None

#         if self.load_type == 'mv_image_based':
#             data_list = []
            
#             # Process LiDAR data
#             if self.modality['use_lidar']:
#                 original_lidar_path = info['lidar_points']['lidar_path']
                
#                 if self.lidar_corruption and self.lidar_corruption != 'none':
#                      # path for corruption
#                     full_original_path = osp.join('samples', 'LIDAR_TOP', original_lidar_path)
#                     info['lidar_points']['lidar_path'] = self.get_corruption_path(
#                         full_original_path,
#                         self.lidar_corruption,
#                         lidar_severity
#                     )
                    
#                     if 'lidar_sweeps' in info:
#                         for i in range(len(info['lidar_sweeps'])):
#                             full_orig_path = info['lidar_sweeps'][i]['lidar_points']['lidar_path']
#                             specific_sample_path = full_orig_path.split('nuscenes/')[-1]
#                             info['lidar_sweeps'][i]['lidar_points']['lidar_path'] = self.get_corruption_path(
#                                 specific_sample_path,
#                                 self.lidar_corruption,
#                                 lidar_severity
#                             )

#                 else:
#                     # Use original LiDAR data
#                     info['lidar_points']['lidar_path'] = osp.join(
#                         self.data_prefix.get('pts', ''),
#                         original_lidar_path
#                     )

                       
#                         # Process camera data
#             if self.modality['use_camera']:
#                 for cam_id, img_info in info['images'].items():
#                     if 'img_path' in img_info:
#                         original_img_path = img_info['img_path']
                        
#                         if self.camera_corruption and self.camera_corruption != 'none':
#                             # Use corrupted camera data
#                             # Need to prepend 'samples/CAM_ID/' to the path for corruption
#                             full_original_path = osp.join('samples', cam_id, original_img_path)
#                             img_info['img_path'] = self.get_corruption_path(
#                                 full_original_path,
#                                 self.camera_corruption,
#                                 camera_severity
#                             )
#                         else:
#                             # Use original camera data
#                             if cam_id in self.data_prefix:
#                                 cam_prefix = self.data_prefix[cam_id]
#                             else:
#                                 cam_prefix = self.data_prefix.get('img', '')
#                             img_info['img_path'] = osp.join(
#                                 cam_prefix, original_img_path
#                             )
#             for idx, (cam_id, img_info) in enumerate(info['images'].items()):
#                 camera_info = dict()
#                 camera_info['images'] = dict()
#                 camera_info['images'][cam_id] = img_info
#                 if 'cam_instances' in info and cam_id in info['cam_instances']:
#                     camera_info['instances'] = info['cam_instances'][cam_id]
#                 else:
#                     camera_info['instances'] = []
#                 camera_info['sample_idx'] = info['sample_idx'] * 6 + idx
#                 camera_info['token'] = info['token']
#                 camera_info['ego2global'] = info['ego2global']

#                 # Add corruption metadata
#                 if self.return_corruption_info:
#                     camera_info['corruption_info'] = {
#                         'lidar_corruption': self.lidar_corruption,
#                         'lidar_severity': lidar_severity,
#                         'camera_corruption': self.camera_corruption,
#                         'camera_severity': camera_severity,
#                     }

#                 if not self.test_mode:
#                     camera_info['ann_info'] = self.parse_ann_info(camera_info)
#                 if self.test_mode and self.load_eval_anns:
#                     camera_info['eval_ann_info'] = \
#                         self.parse_ann_info(camera_info)
#                 data_list.append(camera_info)
#             return data_list
#         else:
#             # Frame-based or FOV-based loading
#             # Process LiDAR data
#             if self.modality['use_lidar']:
#                 original_lidar_path = info['lidar_points']['lidar_path']
                
#                 if self.lidar_corruption and self.lidar_corruption != 'none':
#                      # path for corruption
#                     full_original_path = osp.join('samples', 'LIDAR_TOP', original_lidar_path)
#                     info['lidar_points']['lidar_path'] = self.get_corruption_path(
#                         full_original_path,
#                         self.lidar_corruption,
#                         lidar_severity)
                    
#                     if 'lidar_sweeps' in info:
#                         for i in range(len(info['lidar_sweeps'])):
#                             full_orig_path = info['lidar_sweeps'][i]['lidar_points']['lidar_path']
#                             specific_sample_path = full_orig_path.split('nuscenes/')[-1]
#                             info['lidar_sweeps'][i]['lidar_points']['lidar_path'] = self.get_corruption_path(
#                                 specific_sample_path,
#                                 self.lidar_corruption,
#                                 lidar_severity
#                             )
#                 else:
#                     info['lidar_points']['lidar_path'] = osp.join(
#                         self.data_prefix.get('pts', ''),
#                         original_lidar_path
#                     )

#             # Process camera data
#             # Process camera data
#             if self.modality['use_camera']:
#                 for cam_id, img_info in info['images'].items():
#                     if 'img_path' in img_info:
#                         original_img_path = img_info['img_path']
                        
#                         if self.camera_corruption and self.camera_corruption != 'none':
#                             # Use corrupted camera data
#                             # Need to prepend 'samples/CAM_ID/' to the path for corruption
#                             full_original_path = osp.join('samples', cam_id, original_img_path)
#                             img_info['img_path'] = self.get_corruption_path(
#                                 full_original_path,
#                                 self.camera_corruption,
#                                 camera_severity
#                             )
#                         else:
#                             if cam_id in self.data_prefix:
#                                 cam_prefix = self.data_prefix[cam_id]
#                             else:
#                                 cam_prefix = self.data_prefix.get('img', '')
#                             img_info['img_path'] = osp.join(
#                                 cam_prefix, original_img_path
#                             )

#             # Call parent's parse_data_info to handle the rest
#             # Note: We need to call the grandparent's method to avoid recursion
#             # and properly process the info dict with corruption paths
#             from mmdet3d.datasets.det3d_dataset import Det3DDataset
#             data_info = Det3DDataset.parse_data_info(self, info)
#             # Add corruption metadata to the processed data_info
#             if self.return_corruption_info:
#                 data_info['corruption_info'] = {
#                     'lidar_corruption': self.lidar_corruption,
#                     'lidar_severity': lidar_severity if lidar_severity is not None else 0,
#                     'camera_corruption': self.camera_corruption,
#                     'camera_severity': camera_severity if camera_severity is not None else 0,
#                 }
#             return data_info




# @DATASETS.register_module()
# class NuScenesDiverseCorruptDataset(NuScenesDataset):
#     r"""NuScenes Dataset with Dynamic Corruption Support.

#     This class extends NuScenesDataset to support dynamic corruption
#     selection for both LiDAR and camera data with configurable severity
#     probability distributions.

#     Additional Args (compared to NuScenesDataset):
#         lidar_corruption (str or None): Type of LiDAR corruption to apply.
#             Options: 'pointsreducing', 'beamsreducing', 'snow', 'fog', 
#             'spatialmisalignment', 'temporalmisalignment', 'motionblur', None.
#             Defaults to None.
#         camera_corruption (str or None): Type of camera corruption to apply.
#             Options: 'snow', 'fog', 'temporalmisalignment', 'brightness', 
#             'dark', 'missingcamera', 'motionblur', None.
#             Defaults to None.
#         corruption_root (str): Root directory where corrupted data is stored.
#             Defaults to '/data/multicorrupt'.
#         severity_distribution (dict or None): Probability distribution for severities.
#             Example: {1: 0.33, 2: 0.33, 3: 0.34} for uniform distribution.
#             If None, uses uniform distribution across [0, 1, 2, 3].
#             Defaults to None.
#         fixed_severity (int or None): If set, always use this severity level.
#             Overrides severity_distribution. Defaults to None.
#         return_corruption_info (bool): Whether to return corruption metadata
#             in the data_info dict. Defaults to True.
#     """
#     # Assume that corruptions apply to both modalities, change the dataset to have clean data
#     # if camera/lidar only. Symlink none of the original nuScenes directory
#     CORRUPTIONS = [
#         'pointsreducing', 'beamsreducing', 'snow', 'fog', 
#         'spatialmisalignment', 'temporalmisalignment', 'motionblur', 
#         'brightness', 'dark', 'missingcamera','none'
#     ]

    
#     def __init__(self,
#                  corruptions = None,
#                  corruption_root: str = '/data/multicorrupt',
#                  return_corruption_info: bool = True,
#                  **kwargs) -> None:
#         # Assume all severities are 3?
#         self.corruptions = corruptions
#         self.corruption_root = corruption_root
#         self.return_corruption_info = return_corruption_info

#         # # Validate corruption types
#         # if corruptions is not None:
#         #     for corruption in corruptions:
#         #         assert corruption in self.CORRUPTIONS, \
#         #             f"Invalid LiDAR corruption: {corruption}. " \
#         #             f"Must be one of {self.CORRUPTIONS}"

#         # We assume that all the severities are 3

#         # Call parent constructor
#         super().__init__(**kwargs)


#     def get_corruption_path(self, 
#                            original_path: str, 
#                            corruption_type: str, 
#                            severity: int) -> str:
#         """Get the path to corrupted data.
        
#         Args:
#             original_path (str): Original data path relative to data_root.
#             corruption_type (str): Type of corruption.
#             severity (int): Severity level.
            
#         Returns:
#             str: Path to corrupted data.
#         """
#         # Construct corruption path: corruption_root/corruption_type/severity/original_path
#         corrupted_path = osp.join(
#             self.corruption_root,
#             corruption_type,
#             str(severity),
#             original_path
#         )
#         return corrupted_path

#     def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
#         """Process the raw data info with corruption support.

#         Args:
#             info (dict): Raw info dict.

#         Returns:
#             List[dict] or dict: Has `ann_info` in training stage. And
#             all path has been converted to absolute path or corrupted path.
#         """
#         # Sample severities for this sample
#         selected_corruption = None
#             # Sample a corruption type
#         if self.corruptions is not None:
#             selected_corruption = self.corruptions[int(random.random() * len(self.corruptions))]
        
#         if self.load_type == 'mv_image_based':
#             data_list = []            
#             # Process LiDAR data
#             if self.modality['use_lidar']:
#                 original_lidar_path = info['lidar_points']['lidar_path']
                
#                 if selected_corruption and selected_corruption != 'none':
#                      # path for corruption
#                     full_original_path = osp.join('samples', 'LIDAR_TOP', original_lidar_path)
#                     # Always pass 3 for the corruption
#                     info['lidar_points']['lidar_path'] = self.get_corruption_path(
#                         full_original_path,
#                         selected_corruption,
#                         3
#                     )
                    
#                     if 'lidar_sweeps' in info:
#                         for i in range(len(info['lidar_sweeps'])):
#                             full_orig_path = info['lidar_sweeps'][i]['lidar_points']['lidar_path']
#                             specific_sample_path = full_orig_path.split('nuscenes/')[-1]
#                             info['lidar_sweeps'][i]['lidar_points']['lidar_path'] = self.get_corruption_path(
#                                 specific_sample_path,
#                                 selected_corruption,
#                                 3
#                             )

#                 else:
#                     # Use original LiDAR data
#                     info['lidar_points']['lidar_path'] = osp.join(
#                         self.data_prefix.get('pts', ''),
#                         original_lidar_path
#                     )

                       
#                         # Process camera data
#             if self.modality['use_camera']:
#                 for cam_id, img_info in info['images'].items():
#                     if 'img_path' in img_info:
#                         original_img_path = img_info['img_path']
                        
#                         if selected_corruption and selected_corruption != 'none':
#                             # Use corrupted camera data
#                             # Need to prepend 'samples/CAM_ID/' to the path for corruption
#                             full_original_path = osp.join('samples', cam_id, original_img_path)
#                             img_info['img_path'] = self.get_corruption_path(
#                                 full_original_path,
#                                 selected_corruption,
#                                 3
#                             )
#                         else:
#                             # Use original camera data
#                             if cam_id in self.data_prefix:
#                                 cam_prefix = self.data_prefix[cam_id]
#                             else:
#                                 cam_prefix = self.data_prefix.get('img', '')
#                             img_info['img_path'] = osp.join(
#                                 cam_prefix, original_img_path
#                             )
#             for idx, (cam_id, img_info) in enumerate(info['images'].items()):
#                 camera_info = dict()
#                 camera_info['images'] = dict()
#                 camera_info['images'][cam_id] = img_info
#                 if 'cam_instances' in info and cam_id in info['cam_instances']:
#                     camera_info['instances'] = info['cam_instances'][cam_id]
#                 else:
#                     camera_info['instances'] = []
#                 camera_info['sample_idx'] = info['sample_idx'] * 6 + idx
#                 camera_info['token'] = info['token']
#                 camera_info['ego2global'] = info['ego2global']

#                 # Add corruption metadata
#                 if self.return_corruption_info:
#                     camera_info['corruption_info'] = {
#                         'lidar_corruption': selected_corruption,
#                         'lidar_severity': 3,
#                         'camera_corruption': selected_corruption,
#                         'camera_severity': 3,
#                     }

#                 if not self.test_mode:
#                     camera_info['ann_info'] = self.parse_ann_info(camera_info)
#                 if self.test_mode and self.load_eval_anns:
#                     camera_info['eval_ann_info'] = \
#                         self.parse_ann_info(camera_info)
#                 data_list.append(camera_info)
#             return data_list
#         else:
#             # Frame-based or FOV-based loading
#             # Process LiDAR data
#             if self.modality['use_lidar']:
#                 original_lidar_path = info['lidar_points']['lidar_path']
                
#                 if selected_corruption and selected_corruption != 'none':
#                      # path for corruption
#                     full_original_path = osp.join('samples', 'LIDAR_TOP', original_lidar_path)
#                     info['lidar_points']['lidar_path'] = self.get_corruption_path(
#                         full_original_path,
#                         selected_corruption,
#                         3)
                    
#                     if 'lidar_sweeps' in info:
#                         for i in range(len(info['lidar_sweeps'])):
#                             full_orig_path = info['lidar_sweeps'][i]['lidar_points']['lidar_path']
#                             specific_sample_path = full_orig_path.split('nuscenes/')[-1]
#                             info['lidar_sweeps'][i]['lidar_points']['lidar_path'] = self.get_corruption_path(
#                                 specific_sample_path,
#                                 selected_corruption,
#                                 3
#                             )
#                 else:
#                     info['lidar_points']['lidar_path'] = osp.join(
#                         self.data_prefix.get('pts', ''),
#                         original_lidar_path
#                     )

#             # Process camera data
#             # Process camera data
#             if self.modality['use_camera']:
#                 for cam_id, img_info in info['images'].items():
#                     if 'img_path' in img_info:
#                         original_img_path = img_info['img_path']
                        
#                         if selected_corruption and selected_corruption != 'none':
#                             # Use corrupted camera data
#                             # Need to prepend 'samples/CAM_ID/' to the path for corruption
#                             full_original_path = osp.join('samples', cam_id, original_img_path)
#                             img_info['img_path'] = self.get_corruption_path(
#                                 full_original_path,
#                                 selected_corruption,
#                                 3
#                             )
#                         else:
#                             if cam_id in self.data_prefix:
#                                 cam_prefix = self.data_prefix[cam_id]
#                             else:
#                                 cam_prefix = self.data_prefix.get('img', '')
#                             img_info['img_path'] = osp.join(
#                                 cam_prefix, original_img_path
#                             )

#             # Call parent's parse_data_info to handle the rest
#             # Note: We need to call the grandparent's method to avoid recursion
#             # and properly process the info dict with corruption paths
#             from mmdet3d.datasets.det3d_dataset import Det3DDataset
#             data_info = Det3DDataset.parse_data_info(self, info)
#             # Add corruption metadata to the processed data_info
#             if self.return_corruption_info:
#                 data_info['corruption_info'] = {
#                     'lidar_corruption': selected_corruption,
#                     'lidar_severity': 3,
#                     'camera_corruption': selected_corruption,
#                     'camera_severity': 3,
#                 }
#             return data_info