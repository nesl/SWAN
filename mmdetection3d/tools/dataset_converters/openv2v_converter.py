"""
OPV2V dataset converter for MMDetection3D.

Try to mimic the nuScenes converter

Assume datastructer:
OPENV2V/
    train/
        date/
            vehicle_id/
                
            
    test/
    validate
"""

import os
from os import path as osp
import pickle
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import warnings
from concurrent import futures as futures

from mmengine import print_log


def create_opv2v_infos(root_path: str,
                       info_prefix: str,
                       version: str = 'v1.0',
                       max_sweeps: int = 10) -> None:
    """Create info file of OPV2V dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the dataset. Default: 'v1.0'.
        max_sweeps (int): Max number of sweeps. Default: 10.
    """
    converter = OPV2VConverter(root_path, info_prefix, version, max_sweeps)
    converter.convert()


class OPV2VConverter:
    """OPV2V dataset converter.
    
    This class converts OPV2V dataset to the format compatible with
    MMDetection3D, similar to nuScenes.
    
    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file.
        version (str): Version of the dataset.
        max_sweeps (int): Max number of sweeps.
    """
    
    # OPV2V class mapping
    CLASS_MAP = {
        'car': 'car',
        'van': 'car',
        'truck': 'truck',
        'bus': 'bus',
        'motorcycle': 'motorcycle',
        'bicycle': 'bicycle',
        'pedestrian': 'pedestrian',
        'trailer': 'trailer',
    }
    
    def __init__(self,
                 root_path: str,
                 info_prefix: str,
                 version: str = 'v1.0',
                 max_sweeps: int = 10):
        self.root_path = Path(root_path)
        self.info_prefix = info_prefix
        self.version = version
        self.max_sweeps = max_sweeps
        
    def convert(self) -> None:
        """Convert OPV2V dataset to pickle format."""
        print_log('Start converting OPV2V dataset...')
        
        # Process each split
        splits = ['train', 'validate', 'test']
        for split in splits:
            split_path = self.root_path / split
            if not split_path.exists():
                print_log(f'Split {split} not found, skipping...')
                continue
            
            print_log(f'Processing {split} split...')
            infos = self._process_split(split)
            
            # Save to pickle
            split_name = 'val' if split == 'validate' else split
            filename = osp.join(self.root_path, 
                              f'{self.info_prefix}_infos_{split_name}.pkl')
            
            print_log(f'Samples: {len(infos)}')
            with open(filename, 'wb') as f:
                pickle.dump(infos, f)
            print_log(f'OPV2V info {split_name} file is saved to {filename}')
            
            # Print statistics
            self._print_statistics(infos, split)
    
    def _process_split(self, split: str) -> List[Dict[str, Any]]:
        """Process a single split.
        
        Args:
            split (str): Split name.
            
        Returns:
            list[dict]: List of info dicts.
        """
        split_path = self.root_path / split
        infos = []
        
        # Get all scenarios
        scenarios = sorted([d for d in split_path.iterdir() if d.is_dir()])
        
        print_log(f'Found {len(scenarios)} scenarios in {split}')
        
        for scenario in tqdm(scenarios, desc=f'Processing {split}'):
            # Get all vehicles/agents in scenario
            vehicles = sorted([d for d in scenario.iterdir() if d.is_dir()])
            
            for vehicle in vehicles:
                # Get all timestamps
                yaml_files = sorted(vehicle.glob('*.yaml'))
                
                for yaml_file in yaml_files:
                    timestamp = yaml_file.stem
                    pcd_file = vehicle / f'{timestamp}.pcd'
                    
                    if not pcd_file.exists():
                        warnings.warn(f'PCD file not found: {pcd_file}')
                        continue
                    
                    try:
                        info = self._process_sample(
                            yaml_file=yaml_file,
                            pcd_file=pcd_file,
                            scenario_name=scenario.name,
                            vehicle_id=vehicle.name,
                            timestamp=timestamp,
                            split=split
                        )
                        if info is not None:
                            infos.append(info)
                    except Exception as e:
                        warnings.warn(f'Error processing {yaml_file}: {e}')
                        continue
        
        return infos
    
    def _process_sample(self,
                       yaml_file: Path,
                       pcd_file: Path,
                       scenario_name: str,
                       vehicle_id: str,
                       timestamp: str,
                       split: str) -> Optional[Dict[str, Any]]:
        """Process a single sample.
        
        Args:
            yaml_file (Path): Path to YAML annotation.
            pcd_file (Path): Path to PCD file.
            scenario_name (str): Scenario name.
            vehicle_id (str): Vehicle ID.
            timestamp (str): Timestamp.
            split (str): Split name.
            
        Returns:
            dict: Info dict for the sample.
        """
        # Load YAML annotation
        with open(yaml_file, 'r') as f:
            anno_data = yaml.safe_load(f)
        
        # Create relative path
        lidar_path = str(pcd_file.relative_to(self.root_path))
        
        # Create unique token
        token = f'{scenario_name}_{vehicle_id}_{timestamp}'
        
        # Initialize info dict (following nuScenes format)
        info = {
            'lidar_path': lidar_path,
            'token': token,
            'sweeps': [],
            'cams': {},
            'lidar2ego_translation': [0.0, 0.0, 0.0],
            'lidar2ego_rotation': [1.0, 0.0, 0.0, 0.0],
            'ego2global_translation': [0.0, 0.0, 0.0],
            'ego2global_rotation': [1.0, 0.0, 0.0, 0.0],
            'timestamp': int(timestamp) if timestamp.isdigit() else 0,
        }
        
        # Extract ego pose if available
        if 'lidar_pose' in anno_data:
            ego_pose = self._extract_pose(anno_data['lidar_pose'])
            info['ego2global_translation'] = ego_pose[:3, 3].tolist()
            # Convert rotation matrix to quaternion for nuScenes compatibility
            info['ego2global_rotation'] = self._matrix_to_quaternion(
                ego_pose[:3, :3]).tolist()
        
        # Process camera data if available
        if 'camera_data' in anno_data or 'cameras' in anno_data:
            cam_data = anno_data.get('camera_data', anno_data.get('cameras', {}))
            info['cams'] = self._process_cameras(cam_data, pcd_file.parent)
        
        # Extract ground truth annotations
        gt_info = self._extract_annotations(anno_data)
        if gt_info is not None:
            info.update(gt_info)
        else:
            # Empty annotations
            info['gt_boxes'] = np.zeros((0, 7), dtype=np.float32)
            info['gt_names'] = np.array([], dtype='<U20')
            info['gt_velocity'] = np.zeros((0, 2), dtype=np.float32)
            info['num_lidar_pts'] = np.array([], dtype=np.int32)
            info['num_radar_pts'] = np.array([], dtype=np.int32)
            info['valid_flag'] = np.array([], dtype=bool)
        
        return info
    
    def _extract_pose(self, pose_data: Any) -> np.ndarray:
        """Extract 4x4 transformation matrix from pose data.
        
        Args:
            pose_data: Pose data from YAML.
            
        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        if isinstance(pose_data, dict):
            if 'transformation_matrix' in pose_data:
                pose = np.array(pose_data['transformation_matrix'])
            elif 'translation' in pose_data and 'rotation' in pose_data:
                pose = np.eye(4)
                pose[:3, 3] = pose_data['translation']
                rot = np.array(pose_data['rotation'])
                if rot.shape == (3, 3):
                    pose[:3, :3] = rot
            else:
                pose = np.eye(4)
        elif isinstance(pose_data, (list, np.ndarray)):
            pose = np.array(pose_data).reshape(4, 4)
        else:
            pose = np.eye(4)
        
        return pose
    
    def _matrix_to_quaternion(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion (w, x, y, z).
        
        Args:
            rotation_matrix (np.ndarray): 3x3 rotation matrix.
            
        Returns:
            np.ndarray: Quaternion [w, x, y, z].
        """
        # Using Shepperd's method
        trace = np.trace(rotation_matrix)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
            y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
            z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
        else:
            if rotation_matrix[0, 0] > rotation_matrix[1, 1] and \
               rotation_matrix[0, 0] > rotation_matrix[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - 
                                 rotation_matrix[1, 1] - rotation_matrix[2, 2])
                w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
                x = 0.25 * s
                y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - 
                                 rotation_matrix[0, 0] - rotation_matrix[2, 2])
                w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
                x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                y = 0.25 * s
                z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - 
                                 rotation_matrix[0, 0] - rotation_matrix[1, 1])
                w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
                x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                z = 0.25 * s
        
        return np.array([w, x, y, z])
    
    def _process_cameras(self,
                        cam_data: Dict,
                        sample_dir: Path) -> Dict[str, Any]:
        """Process camera information.
        
        Args:
            cam_data (dict): Camera data from YAML.
            sample_dir (Path): Sample directory.
            
        Returns:
            dict: Camera information dict.
        """
        cam_infos = {}
        
        for cam_name, cam_params in cam_data.items():
            # Find camera image
            cam_file = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_file = sample_dir / f'{cam_name}{ext}'
                if potential_file.exists():
                    cam_file = potential_file
                    break
            
            if cam_file is None:
                continue
            
            # Extract extrinsic
            if 'extrinsic' in cam_params:
                extrinsic = np.array(cam_params['extrinsic']).reshape(4, 4)
            else:
                extrinsic = np.eye(4)
            
            # Extract intrinsic
            if 'intrinsic' in cam_params:
                intrinsic = np.array(cam_params['intrinsic']).reshape(3, 3)
            else:
                intrinsic = np.eye(3)
            
            cam_info = {
                'data_path': str(cam_file.relative_to(self.root_path)),
                'type': cam_name,
                'sensor2ego_translation': extrinsic[:3, 3].tolist(),
                'sensor2ego_rotation': self._matrix_to_quaternion(
                    extrinsic[:3, :3]).tolist(),
                'sensor2lidar_translation': extrinsic[:3, 3].tolist(),
                'sensor2lidar_rotation': self._matrix_to_quaternion(
                    extrinsic[:3, :3]).tolist(),
                'cam_intrinsic': intrinsic.tolist(),
            }
            
            cam_infos[cam_name] = cam_info
        
        return cam_infos
    
    def _extract_annotations(self, anno_data: Dict) -> Optional[Dict[str, Any]]:
        """Extract ground truth annotations.
        
        Args:
            anno_data (dict): Annotation data from YAML.
            
        Returns:
            dict: Annotation info dict.
        """
        # Find objects in annotation
        objects = None
        for key in ['vehicles', 'objects', 'annotations', 'labels']:
            if key in anno_data:
                objects = anno_data[key]
                break
        
        if objects is None or len(objects) == 0:
            return None
        
        gt_boxes = []
        gt_names = []
        gt_velocity = []
        num_pts = []
        
        for obj in objects:
            # Extract center
            if 'location' in obj:
                center = obj['location']
            elif 'center' in obj:
                center = obj['center']
            else:
                continue
            
            # Extract dimensions
            if 'extent' in obj:
                extent = obj['extent']
                dimensions = [extent[0] * 2, extent[1] * 2, extent[2] * 2]
            elif 'dimensions' in obj:
                dimensions = obj['dimensions']
            elif 'size' in obj:
                dimensions = obj['size']
            else:
                continue
            
            # Extract rotation (yaw)
            if 'angle' in obj:
                angle = obj['angle']
                yaw = angle[2] if isinstance(angle, (list, np.ndarray)) else angle
            elif 'rotation' in obj:
                rotation = obj['rotation']
                yaw = rotation[2] if isinstance(rotation, (list, np.ndarray)) else rotation
            elif 'yaw' in obj:
                yaw = obj['yaw']
            else:
                yaw = 0.0
            
            # Box format: [x, y, z, l, w, h, yaw]
            box = [
                center[0], center[1], center[2],
                dimensions[0], dimensions[1], dimensions[2],
                yaw
            ]
            
            # Extract class name
            class_name = obj.get('type', obj.get('class', 'car')).lower()
            class_name = self.CLASS_MAP.get(class_name, 'car')
            
            # Extract velocity
            velocity = obj.get('velocity', [0.0, 0.0])
            vel = [velocity[0], velocity[1]] if len(velocity) >= 2 else [0.0, 0.0]
            
            # Number of points
            num_pt = obj.get('num_points', 0)
            
            gt_boxes.append(box)
            gt_names.append(class_name)
            gt_velocity.append(vel)
            num_pts.append(num_pt)
        
        return {
            'gt_boxes': np.array(gt_boxes, dtype=np.float32),
            'gt_names': np.array(gt_names, dtype='<U20'),
            'gt_velocity': np.array(gt_velocity, dtype=np.float32),
            'num_lidar_pts': np.array(num_pts, dtype=np.int32),
            'num_radar_pts': np.zeros(len(gt_boxes), dtype=np.int32),
            'valid_flag': np.ones(len(gt_boxes), dtype=bool),
        }
    
    def _print_statistics(self, infos: List[Dict], split: str) -> None:
        """Print dataset statistics.
        
        Args:
            infos (list[dict]): List of info dicts.
            split (str): Split name.
        """
        print_log(f'\n{split.upper()} Statistics:')
        print_log(f'Total samples: {len(infos)}')
        
        # Count objects per class
        class_counts = {}
        total_objects = 0
        for info in infos:
            for name in info['gt_names']:
                class_counts[name] = class_counts.get(name, 0) + 1
                total_objects += 1
        
        print_log(f'Total objects: {total_objects}')
        print_log('Class distribution:')
        for class_name, count in sorted(class_counts.items()):
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            print_log(f'  {class_name:15s}: {count:6d} ({percentage:5.2f}%)')
        
        avg_objects = total_objects / len(infos) if len(infos) > 0 else 0
        print_log(f'Average objects per frame: {avg_objects:.2f}')