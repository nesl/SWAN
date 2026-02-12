from typing import Callable, List, Union, Optional

from mmdet3d.datasets import NuScenesDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class NuSimDataset(NuScenesDataset):
    """Simulation dataset in NuScenes format.
    
    This is a simple wrapper around NuScenesDataset for simulation data.
    It uses the same format and processing as NuScenes but allows for
    custom metadata and easy identification in configs.
    
    Args:
        Same as NuScenesDataset
    
    Example:
        >>> dataset_type = 'NuSimDataset'
        >>> train_dataloader = dict(
        >>>     dataset=dict(
        >>>         type=dataset_type,
        >>>         data_root='data/nuscenes/',
        >>>         ann_file='nuscenes_infos_train.pkl',
        >>>         pipeline=train_pipeline))
    """
    
    METAINFO = {
        # Note that all classes are needed because I still set car as 0 and pedestrains as 7
        'classes': ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                   'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'),
        'version': 'v1.0-trainval',  
        'palette': [
            (255, 158, 0),   # Orange - car
            (255, 99, 71),   # Tomato - truck
            (255, 140, 0),   # Darkorange - trailer
            (255, 127, 80),  # Coral - bus
            (233, 150, 70),  # Darksalmon - construction_vehicle
            (220, 20, 60),   # Crimson - bicycle
            (255, 61, 99),   # Red - motorcycle
            (0, 0, 230),     # Blue - pedestrian
            (47, 79, 79),    # Darkslategrey - traffic_cone
            (112, 128, 144), # Slategrey - barrier
        ]
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 box_type_3d: str = 'LiDAR',
                 load_type: str = 'frame_based',
                 modality: dict = dict(use_camera=False, use_lidar=True),
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 with_velocity: bool = True,
                 use_valid_flag: bool = False,
                 **kwargs) -> None:
        
        # Call parent NuScenesDataset constructor
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            box_type_3d=box_type_3d,
            load_type=load_type,
            modality=modality,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            with_velocity=with_velocity,
            use_valid_flag=use_valid_flag,
            **kwargs)