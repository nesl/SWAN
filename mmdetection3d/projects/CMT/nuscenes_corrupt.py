from os import path as osp
from typing import Optional, Dict, Union, List
import random

from mmdet3d.registry import DATASETS, METRICS
from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset

# added
from mmdet3d.evaluation.metrics.nuscenes_metric import NuScenesMetric
from scene_split import train_day, train_night, train_rain, train_dry, val_day, val_night, val_rain, val_dry, train, val


from nuscenes.nuscenes import NuScenes


@DATASETS.register_module()
class NuScenesCorruptDataset(NuScenesDataset):
    r"""NuScenes Dataset with Dynamic Corruption Support.

    This class extends NuScenesDataset to support dynamic corruption
    selection for both LiDAR and camera data with configurable severity
    probability distributions.

    Additional Args (compared to NuScenesDataset):
        lidar_corruption (str or None): Type of LiDAR corruption to apply.
            Options: 'pointsreducing', 'beamsreducing', 'snow', 'fog', 
            'spatialmisalignment', 'temporalmisalignment', 'motionblur', None.
            Defaults to None.
        camera_corruption (str or None): Type of camera corruption to apply.
            Options: 'snow', 'fog', 'temporalmisalignment', 'brightness', 
            'dark', 'missingcamera', 'motionblur', None.
            Defaults to None.
        corruption_root (str): Root directory where corrupted data is stored.
            Defaults to '/data/multicorrupt'.
        severity_distribution (dict or None): Probability distribution for severities.
            Example: {1: 0.33, 2: 0.33, 3: 0.34} for uniform distribution.
            If None, uses uniform distribution across [0, 1, 2, 3].
            Defaults to None.
        fixed_severity (int or None): If set, always use this severity level.
            Overrides severity_distribution. Defaults to None.
        return_corruption_info (bool): Whether to return corruption metadata
            in the data_info dict. Defaults to True.
    """

    LIDAR_CORRUPTIONS = [
        'pointsreducing', 'beamsreducing', 'snow', 'fog', 
        'spatialmisalignment', 'temporalmisalignment', 'motionblur'
    ]
    
    CAMERA_CORRUPTIONS = [
        'snow', 'fog', 'temporalmisalignment', 'brightness', 
        'dark', 'missingcamera', 'motionblur'
    ]

    def __init__(self,
                 lidar_corruption: Optional[str] = None,
                 camera_corruption: Optional[str] = None,
                 corruption_root: str = '/data/multicorrupt',
                 severity_distribution: Optional[Dict[int, float]] = None,
                 fixed_severity: Optional[int] = None,
                 return_corruption_info: bool = True,
                 **kwargs) -> None:
        
        self.lidar_corruption = lidar_corruption
        self.camera_corruption = camera_corruption
        self.corruption_root = corruption_root
        self.fixed_severity = fixed_severity
        self.return_corruption_info = return_corruption_info

        # Validate corruption types
        if lidar_corruption is not None:
            assert lidar_corruption in self.LIDAR_CORRUPTIONS, \
                f"Invalid LiDAR corruption: {lidar_corruption}. " \
                f"Must be one of {self.LIDAR_CORRUPTIONS}"
        
        if camera_corruption is not None:
            assert camera_corruption in self.CAMERA_CORRUPTIONS, \
                f"Invalid camera corruption: {camera_corruption}. " \
                f"Must be one of {self.CAMERA_CORRUPTIONS}"

        # Setup severity distribution
        if fixed_severity is not None:
            assert fixed_severity in [0, 1, 2, 3], \
                f"fixed_severity must be 0, 1, 2, or 3, got {fixed_severity}"
            self.severity_probs = {fixed_severity: 1.0}
        elif severity_distribution is not None:
            self._validate_severity_distribution(severity_distribution)
            self.severity_probs = severity_distribution
        else:
            # Default: uniform distribution
            self.severity_probs = {0 : 1/4, 1: 1/4, 2: 1/4, 3: 1/4}

        # Call parent constructor
        super().__init__(**kwargs)

    def _validate_severity_distribution(self, dist: Dict[int, float]) -> None:
        """Validate severity probability distribution."""
        assert all(k in [0, 1, 2, 3] for k in dist.keys()), \
            "Severity levels must be 0, 1, 2, or 3"
        assert all(0 <= v <= 1 for v in dist.values()), \
            "Probabilities must be between 0 and 1"
        assert abs(sum(dist.values()) - 1.0) < 1e-6, \
            f"Probabilities must sum to 1, got {sum(dist.values())}"

    def sample_severity(self) -> int:
        """Sample a severity level based on the configured distribution.
        
        Returns:
            int: Severity level (1, 2, or 3).
        """
        severities = list(self.severity_probs.keys())
        probs = list(self.severity_probs.values())
        return random.choices(severities, weights=probs, k=1)[0]

    def get_corruption_path(self, 
                           original_path: str, 
                           corruption_type: str, 
                           severity: int) -> str:
        """Get the path to corrupted data.
        
        Args:
            original_path (str): Original data path relative to data_root.
            corruption_type (str): Type of corruption.
            severity (int): Severity level.
            
        Returns:
            str: Path to corrupted data.
        """
        # Construct corruption path: corruption_root/corruption_type/severity/original_path
        corrupted_path = osp.join(
            self.corruption_root,
            corruption_type,
            str(severity),
            original_path
        )
        return corrupted_path

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        """Process the raw data info with corruption support.

        Args:
            info (dict): Raw info dict.

        Returns:
            List[dict] or dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path or corrupted path.
        """
        # Sample severities for this sample
        lidar_severity = self.sample_severity() if self.lidar_corruption else None
        camera_severity = self.sample_severity() if self.camera_corruption else None

        if self.load_type == 'mv_image_based':
            data_list = []
            
            # Process LiDAR data
            if self.modality['use_lidar']:
                original_lidar_path = info['lidar_points']['lidar_path']
                
                if self.lidar_corruption:
                     # path for corruption
                    full_original_path = osp.join('samples', 'LIDAR_TOP', original_lidar_path)
                    info['lidar_points']['lidar_path'] = self.get_corruption_path(
                        full_original_path,
                        self.lidar_corruption,
                        lidar_severity
                    )
                    
                    if 'lidar_sweeps' in info:
                        for i in range(len(info['lidar_sweeps'])):
                            full_orig_path = info['lidar_sweeps'][i]['lidar_points']['lidar_path']
                            specific_sample_path = full_orig_path.split('nuscenes/')[-1]
                            info['lidar_sweeps'][i]['lidar_points']['lidar_path'] = self.get_corruption_path(
                                specific_sample_path,
                                self.lidar_corruption,
                                lidar_severity
                            )

                else:
                    # Use original LiDAR data
                    info['lidar_points']['lidar_path'] = osp.join(
                        self.data_prefix.get('pts', ''),
                        original_lidar_path
                    )

                       
                        # Process camera data
            if self.modality['use_camera']:
                for cam_id, img_info in info['images'].items():
                    if 'img_path' in img_info:
                        original_img_path = img_info['img_path']
                        
                        if self.camera_corruption:
                            # Use corrupted camera data
                            # Need to prepend 'samples/CAM_ID/' to the path for corruption
                            full_original_path = osp.join('samples', cam_id, original_img_path)
                            img_info['img_path'] = self.get_corruption_path(
                                full_original_path,
                                self.camera_corruption,
                                camera_severity
                            )
                        else:
                            # Use original camera data
                            if cam_id in self.data_prefix:
                                cam_prefix = self.data_prefix[cam_id]
                            else:
                                cam_prefix = self.data_prefix.get('img', '')
                            img_info['img_path'] = osp.join(
                                cam_prefix, original_img_path
                            )
            for idx, (cam_id, img_info) in enumerate(info['images'].items()):
                camera_info = dict()
                camera_info['images'] = dict()
                camera_info['images'][cam_id] = img_info
                if 'cam_instances' in info and cam_id in info['cam_instances']:
                    camera_info['instances'] = info['cam_instances'][cam_id]
                else:
                    camera_info['instances'] = []
                camera_info['sample_idx'] = info['sample_idx'] * 6 + idx
                camera_info['token'] = info['token']
                camera_info['ego2global'] = info['ego2global']

                # Add corruption metadata
                if self.return_corruption_info:
                    camera_info['corruption_info'] = {
                        'lidar_corruption': self.lidar_corruption,
                        'lidar_severity': lidar_severity,
                        'camera_corruption': self.camera_corruption,
                        'camera_severity': camera_severity,
                    }

                if not self.test_mode:
                    camera_info['ann_info'] = self.parse_ann_info(camera_info)
                if self.test_mode and self.load_eval_anns:
                    camera_info['eval_ann_info'] = \
                        self.parse_ann_info(camera_info)
                data_list.append(camera_info)
            return data_list
        else:
            # Frame-based or FOV-based loading
            # Process LiDAR data
            if self.modality['use_lidar']:
                original_lidar_path = info['lidar_points']['lidar_path']
                
                if self.lidar_corruption:
                     # path for corruption
                    full_original_path = osp.join('samples', 'LIDAR_TOP', original_lidar_path)
                    info['lidar_points']['lidar_path'] = self.get_corruption_path(
                        full_original_path,
                        self.lidar_corruption,
                        lidar_severity)
                    
                    if 'lidar_sweeps' in info:
                        for i in range(len(info['lidar_sweeps'])):
                            full_orig_path = info['lidar_sweeps'][i]['lidar_points']['lidar_path']
                            specific_sample_path = full_orig_path.split('nuscenes/')[-1]
                            info['lidar_sweeps'][i]['lidar_points']['lidar_path'] = self.get_corruption_path(
                                specific_sample_path,
                                self.lidar_corruption,
                                lidar_severity
                            )
                else:
                    info['lidar_points']['lidar_path'] = osp.join(
                        self.data_prefix.get('pts', ''),
                        original_lidar_path
                    )

            # Process camera data
            # Process camera data
            if self.modality['use_camera']:
                for cam_id, img_info in info['images'].items():
                    if 'img_path' in img_info:
                        original_img_path = img_info['img_path']
                        
                        if self.camera_corruption:
                            # Use corrupted camera data
                            # Need to prepend 'samples/CAM_ID/' to the path for corruption
                            full_original_path = osp.join('samples', cam_id, original_img_path)
                            img_info['img_path'] = self.get_corruption_path(
                                full_original_path,
                                self.camera_corruption,
                                camera_severity
                            )
                        else:
                            if cam_id in self.data_prefix:
                                cam_prefix = self.data_prefix[cam_id]
                            else:
                                cam_prefix = self.data_prefix.get('img', '')
                            img_info['img_path'] = osp.join(
                                cam_prefix, original_img_path
                            )

            # Call parent's parse_data_info to handle the rest
            # Note: We need to call the grandparent's method to avoid recursion
            # and properly process the info dict with corruption paths
            from mmdet3d.datasets.det3d_dataset import Det3DDataset
            data_info = Det3DDataset.parse_data_info(self, info)
            # Add corruption metadata to the processed data_info
            if self.return_corruption_info:
                data_info['corruption_info'] = {
                    'lidar_corruption': self.lidar_corruption,
                    'lidar_severity': lidar_severity if lidar_severity is not None else 0,
                    'camera_corruption': self.camera_corruption,
                    'camera_severity': camera_severity if camera_severity is not None else 0,
                }
            return data_info




@DATASETS.register_module()
class NuScenesDiverseCorruptDataset(NuScenesDataset):
    r"""NuScenes Dataset with Dynamic Corruption Support.

    This class extends NuScenesDataset to support dynamic corruption
    selection for both LiDAR and camera data with configurable severity
    probability distributions.

    Additional Args (compared to NuScenesDataset):
        lidar_corruption (str or None): Type of LiDAR corruption to apply.
            Options: 'pointsreducing', 'beamsreducing', 'snow', 'fog', 
            'spatialmisalignment', 'temporalmisalignment', 'motionblur', None.
            Defaults to None.
        camera_corruption (str or None): Type of camera corruption to apply.
            Options: 'snow', 'fog', 'temporalmisalignment', 'brightness', 
            'dark', 'missingcamera', 'motionblur', None.
            Defaults to None.
        corruption_root (str): Root directory where corrupted data is stored.
            Defaults to '/data/multicorrupt'.
        severity_distribution (dict or None): Probability distribution for severities.
            Example: {1: 0.33, 2: 0.33, 3: 0.34} for uniform distribution.
            If None, uses uniform distribution across [0, 1, 2, 3].
            Defaults to None.
        fixed_severity (int or None): If set, always use this severity level.
            Overrides severity_distribution. Defaults to None.
        return_corruption_info (bool): Whether to return corruption metadata
            in the data_info dict. Defaults to True.
    """
    # Assume that corruptions apply to both modalities, change the dataset to have clean data
    # if camera/lidar only. Symlink none of the original nuScenes directory
    CORRUPTIONS = [
        'pointsreducing', 'beamsreducing', 'snow', 'fog', 
        'spatialmisalignment', 'temporalmisalignment', 'motionblur', 
        'brightness', 'dark', 'missingcamera','none'
    ]

    
    def __init__(self,
                 corruptions = None,
                 corruption_root: str = '/data/multicorrupt',
                 return_corruption_info: bool = True,
                 **kwargs) -> None:
        # Assume all severities are 3?
        self.corruptions = corruptions
        self.corruption_root = corruption_root
        self.return_corruption_info = return_corruption_info

        # Validate corruption types
        if corruptions is not None:
            for corruption in corruptions:
                assert corruption in self.CORRUPTIONS, \
                    f"Invalid LiDAR corruption: {corruption}. " \
                    f"Must be one of {self.CORRUPTIONS}"

        # We assume that all the severities are 3

        # Call parent constructor
        super().__init__(**kwargs)


    def get_corruption_path(self, 
                           original_path: str, 
                           corruption_type: str, 
                           severity: int) -> str:
        """Get the path to corrupted data.
        
        Args:
            original_path (str): Original data path relative to data_root.
            corruption_type (str): Type of corruption.
            severity (int): Severity level.
            
        Returns:
            str: Path to corrupted data.
        """
        # Construct corruption path: corruption_root/corruption_type/severity/original_path
        corrupted_path = osp.join(
            self.corruption_root,
            corruption_type,
            str(severity),
            original_path
        )
        return corrupted_path

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        """Process the raw data info with corruption support.

        Args:
            info (dict): Raw info dict.

        Returns:
            List[dict] or dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path or corrupted path.
        """
        # Sample severities for this sample
        selected_corruption = None
            # Sample a corruption type
        if self.corruptions is not None:
            selected_corruption = self.corruptions[int(random.random() * len(self.corruptions))]
        
        if self.load_type == 'mv_image_based':
            data_list = []            
            # Process LiDAR data
            if self.modality['use_lidar']:
                original_lidar_path = info['lidar_points']['lidar_path']
                
                if selected_corruption and selected_corruption != 'none':
                     # path for corruption
                    full_original_path = osp.join('samples', 'LIDAR_TOP', original_lidar_path)
                    # Always pass 3 for the corruption
                    info['lidar_points']['lidar_path'] = self.get_corruption_path(
                        full_original_path,
                        selected_corruption,
                        3
                    )
                    
                    if 'lidar_sweeps' in info:
                        for i in range(len(info['lidar_sweeps'])):
                            full_orig_path = info['lidar_sweeps'][i]['lidar_points']['lidar_path']
                            specific_sample_path = full_orig_path.split('nuscenes/')[-1]
                            info['lidar_sweeps'][i]['lidar_points']['lidar_path'] = self.get_corruption_path(
                                specific_sample_path,
                                selected_corruption,
                                3
                            )

                else:
                    # Use original LiDAR data
                    info['lidar_points']['lidar_path'] = osp.join(
                        self.data_prefix.get('pts', ''),
                        original_lidar_path
                    )

                       
                        # Process camera data
            if self.modality['use_camera']:
                for cam_id, img_info in info['images'].items():
                    if 'img_path' in img_info:
                        original_img_path = img_info['img_path']
                        
                        if selected_corruption and selected_corruption != 'none':
                            # Use corrupted camera data
                            # Need to prepend 'samples/CAM_ID/' to the path for corruption
                            full_original_path = osp.join('samples', cam_id, original_img_path)
                            img_info['img_path'] = self.get_corruption_path(
                                full_original_path,
                                selected_corruption,
                                3
                            )
                        else:
                            # Use original camera data
                            if cam_id in self.data_prefix:
                                cam_prefix = self.data_prefix[cam_id]
                            else:
                                cam_prefix = self.data_prefix.get('img', '')
                            img_info['img_path'] = osp.join(
                                cam_prefix, original_img_path
                            )
            for idx, (cam_id, img_info) in enumerate(info['images'].items()):
                camera_info = dict()
                camera_info['images'] = dict()
                camera_info['images'][cam_id] = img_info
                if 'cam_instances' in info and cam_id in info['cam_instances']:
                    camera_info['instances'] = info['cam_instances'][cam_id]
                else:
                    camera_info['instances'] = []
                camera_info['sample_idx'] = info['sample_idx'] * 6 + idx
                camera_info['token'] = info['token']
                camera_info['ego2global'] = info['ego2global']

                # Add corruption metadata
                if self.return_corruption_info:
                    camera_info['corruption_info'] = {
                        'lidar_corruption': selected_corruption,
                        'lidar_severity': 3,
                        'camera_corruption': selected_corruption,
                        'camera_severity': 3,
                    }

                if not self.test_mode:
                    camera_info['ann_info'] = self.parse_ann_info(camera_info)
                if self.test_mode and self.load_eval_anns:
                    camera_info['eval_ann_info'] = \
                        self.parse_ann_info(camera_info)
                data_list.append(camera_info)
            return data_list
        else:
            # Frame-based or FOV-based loading
            # Process LiDAR data
            if self.modality['use_lidar']:
                original_lidar_path = info['lidar_points']['lidar_path']
                
                if selected_corruption and selected_corruption != 'none':
                     # path for corruption
                    full_original_path = osp.join('samples', 'LIDAR_TOP', original_lidar_path)
                    info['lidar_points']['lidar_path'] = self.get_corruption_path(
                        full_original_path,
                        selected_corruption,
                        3)
                    
                    if 'lidar_sweeps' in info:
                        for i in range(len(info['lidar_sweeps'])):
                            full_orig_path = info['lidar_sweeps'][i]['lidar_points']['lidar_path']
                            specific_sample_path = full_orig_path.split('nuscenes/')[-1]
                            info['lidar_sweeps'][i]['lidar_points']['lidar_path'] = self.get_corruption_path(
                                specific_sample_path,
                                selected_corruption,
                                3
                            )
                else:
                    info['lidar_points']['lidar_path'] = osp.join(
                        self.data_prefix.get('pts', ''),
                        original_lidar_path
                    )

            # Process camera data
            # Process camera data
            if self.modality['use_camera']:
                for cam_id, img_info in info['images'].items():
                    if 'img_path' in img_info:
                        original_img_path = img_info['img_path']
                        
                        if selected_corruption and selected_corruption != 'none':
                            # Use corrupted camera data
                            # Need to prepend 'samples/CAM_ID/' to the path for corruption
                            full_original_path = osp.join('samples', cam_id, original_img_path)
                            img_info['img_path'] = self.get_corruption_path(
                                full_original_path,
                                selected_corruption,
                                3
                            )
                        else:
                            if cam_id in self.data_prefix:
                                cam_prefix = self.data_prefix[cam_id]
                            else:
                                cam_prefix = self.data_prefix.get('img', '')
                            img_info['img_path'] = osp.join(
                                cam_prefix, original_img_path
                            )

            # Call parent's parse_data_info to handle the rest
            # Note: We need to call the grandparent's method to avoid recursion
            # and properly process the info dict with corruption paths
            from mmdet3d.datasets.det3d_dataset import Det3DDataset
            data_info = Det3DDataset.parse_data_info(self, info)
            # Add corruption metadata to the processed data_info
            if self.return_corruption_info:
                data_info['corruption_info'] = {
                    'lidar_corruption': selected_corruption,
                    'lidar_severity': 3,
                    'camera_corruption': selected_corruption,
                    'camera_severity': 3,
                }
            return data_info

# ---------------------------------------------------------------------------
# Scene-split variants of the corrupt datasets and a matching metric.
# ---------------------------------------------------------------------------
nusc = None

@DATASETS.register_module()
class NuScenesCorruptSplitDataset(NuScenesCorruptDataset):
    """
    NuScenes Dataset with Corruption Support and Custom Scene Splits.
    add a split argument to filter the data.
    """
    def __init__(self,
                 split = None,
                 **kwargs) -> None:
        global nusc
        self.split = split
        assert self.split in ['day', 'night', 'rain', 'dry',None], \
            f"Invalid split: {split}. Must be one of ['day', 'night', 'rain', 'dry', None]"
        self.nusc = None
        self.data_root = kwargs.get('data_root', None)
        if self.split is not None:
            print('initializing NuScenes devkit...')
            # If a split is specified, filter scenes accordingly
            if nusc is None:
                nusc = NuScenes(version="v1.0-trainval", dataroot=self.data_root, verbose=False)
            self.nusc = nusc

        # Call parent constructor
        super().__init__(**kwargs)
        
    def filter_data(self) -> List[dict]:
        # Find the scene token in the annotation use it to get their scene name and filter based on the split
        # currently we dont care whether it is train or val to make config simpler
        print('custom filter data for split:', self.split)
        print('original data list length:', len(self.data_list))
    
        if self.split:
            if getattr(self, 'nusc', None) is None:
                raise ValueError("NuScenes instance not initialized. Ensure that the parent constructor is called with a valid data_root.")
            new_data_list = []
            for data_info in self.data_list:
                sample_token = data_info['token']
                sample = self.nusc.get('sample', sample_token)
                scene_token = sample['scene_token']
                scene = self.nusc.get('scene', scene_token)
                scene_name = scene['name']
                if self.split == 'day':
                    if scene_name in train_day or scene_name in val_day:
                        new_data_list.append(data_info)
                elif self.split == 'night':
                    if scene_name in train_night or scene_name in val_night:
                        new_data_list.append(data_info)
                elif self.split == 'rain':
                    if scene_name in train_rain or scene_name in val_rain:
                        new_data_list.append(data_info)
                elif self.split == 'dry':
                    if scene_name in train_dry or scene_name in val_dry:
                        new_data_list.append(data_info)
                else:
                    # 
                    continue
            self.data_list = new_data_list


        print('filtered data list length:', len(self.data_list))
        return self.data_list



@DATASETS.register_module()
class NuScenesCorruptDiverseSplitDataset(NuScenesDiverseCorruptDataset):
    """
    NuScenes Dataset with Corruption Support and Custom Scene Splits.
    add a split argument to filter the data.
    """
    def __init__(self,
                 split = None,
                 **kwargs) -> None:
        global nusc
        self.split = split
        assert self.split in ['day', 'night', 'rain', 'dry',None], \
            f"Invalid split: {split}. Must be one of ['day', 'night', 'rain', 'dry', None]"
        self.nusc = None
        self.data_root = kwargs.get('data_root', None)
        if self.split is not None:
            print('initializing NuScenes devkit...')
            # If a split is specified, filter scenes accordingly
            if nusc is None:
                nusc = NuScenes(version="v1.0-trainval", dataroot=self.data_root, verbose=False)
            self.nusc = nusc

        # Call parent constructor
        super().__init__(**kwargs)
        
    def filter_data(self) -> List[dict]:
        # Find the scene token in the annotation use it to get their scene name and filter based on the split
        # currently we dont care whether it is train or val to make config simpler
        print('custom filter data for split:', self.split)
        print('original data list length:', len(self.data_list))
    
        if self.split:
            if getattr(self, 'nusc', None) is None:
                raise ValueError("NuScenes instance not initialized. Ensure that the parent constructor is called with a valid data_root.")
            new_data_list = []
            for data_info in self.data_list:
                sample_token = data_info['token']
                sample = self.nusc.get('sample', sample_token)
                scene_token = sample['scene_token']
                scene = self.nusc.get('scene', scene_token)
                scene_name = scene['name']
                if self.split == 'day':
                    if scene_name in train_day or scene_name in val_day:
                        new_data_list.append(data_info)
                elif self.split == 'night':
                    if scene_name in train_night or scene_name in val_night:
                        new_data_list.append(data_info)
                elif self.split == 'rain':
                    if scene_name in train_rain or scene_name in val_rain:
                        new_data_list.append(data_info)
                elif self.split == 'dry':
                    if scene_name in train_dry or scene_name in val_dry:
                        new_data_list.append(data_info)
                else:
                    # 
                    continue
            self.data_list = new_data_list


        print('filtered data list length:', len(self.data_list))
        return self.data_list

    
@METRICS.register_module()
class NuScenesPartialMetric(NuScenesMetric):
    """NuScenes evaluation metric that evaluates on a specific scene split.

    This metric extends NuScenesMetric to support evaluating only on subsets
    of the validation set defined by scene splits (day/night/rain/dry) from
    scene_split.py. It monkey-patches nuscenes.utils.splits.create_splits_scenes
    to inject the custom split, then uses it as the eval_set.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'bbox'.
        split (str, optional): Scene split to evaluate on.
            Options: 'day', 'night', 'rain', 'dry', None.
            If None, evaluates on all val scenes (default nuscenes behavior).
        **kwargs: Other arguments passed to NuScenesMetric.
    """

    VALID_SPLITS = ['day', 'night', 'rain', 'dry']

    SPLIT_SCENES = {
        'day': val_day,
        'night': val_night,
        'rain': val_rain,
        'dry': val_dry,
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metric: str = 'bbox',
                 split: Optional[str] = None,
                 **kwargs) -> None:
        global nusc
        self.nusc = None
        self.split = split
        if split is not None:
            assert split in self.VALID_SPLITS, \
                f"Invalid split: {split}. Must be one of {self.VALID_SPLITS}"
        if nusc is None:
            nusc = NuScenes(version="v1.0-trainval", dataroot=data_root, verbose=False)
        self.nusc = nusc  # Will be initialized in the dataset and used in evaluation

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metric=metric,
            **kwargs)

    def _evaluate_single(self,
                        result_path: str,
                        classes: Optional[List[str]] = None,
                        result_name: str = 'pred_instances_3d') -> Dict[str, float]:
        """Evaluation for a single model in nuScenes protocol.

        If a split is set, patches create_splits_scenes to include the
        split's scene list and uses it as eval_set. Otherwise falls back
        to the standard 'val' eval_set.
        """
        
        from nuscenes.eval.detection.evaluate import NuScenesEval
        import mmengine

        output_dir = osp.join(*osp.split(result_path)[:-1])

        if self.split is not None:
            eval_set = f'val_{self.split}'
            scene_list = self.SPLIT_SCENES[self.split]

            # DetectionEval uses is_predefined_split() to decide the code
            # path.  'val_day' etc. are NOT predefined, so it takes the
            # custom-split branch which calls get_samples_of_custom_split()
            # and correctly filters BOTH predictions and GT to the same
            # sample tokens.  We monkey-patch that function (in the
            # evaluate module namespace where it is looked up) to return
            # the sample tokens for our scene list.
            from nuscenes.eval.detection import evaluate as _eval_mod
            from nuscenes.eval.common.loaders import get_samples_of_scenes
            original_get_samples = _eval_mod.get_samples_of_custom_split

            def patched_get_samples(split_name, nusc):
                if split_name == eval_set:
                    return get_samples_of_scenes(
                        scene_names=scene_list, nusc=nusc)
                return original_get_samples(split_name, nusc)

            _eval_mod.get_samples_of_custom_split = patched_get_samples
        else:
            eval_set = 'val'

        try:
            nusc_eval = NuScenesEval(
                self.nusc,
                config=self.eval_detection_configs,
                result_path=result_path,
                eval_set=eval_set,
                output_dir=output_dir,
                verbose=False)

            nusc_eval.main(render_curves=False)
        finally:
            # Restore original function if we patched it
            if self.split is not None:
                _eval_mod.get_samples_of_custom_split = original_get_samples

        # Record metrics
        metrics = mmengine.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'

        for name in classes:
            for k, v in metrics['label_aps'][name].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{name}_AP_dist_{k}'] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{name}_{k}'] = val

        for k, v in metrics['tp_errors'].items():
            val = float(f'{v:.4f}')
            detail[f'{metric_prefix}/{self.ErrNameMapping[k]}'] = val

        detail[f'{metric_prefix}/NDS'] = metrics['nd_score']
        detail[f'{metric_prefix}/mAP'] = metrics['mean_ap']

        return detail