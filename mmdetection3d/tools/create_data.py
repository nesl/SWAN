# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp

from mmengine import print_log

from tools.dataset_converters import indoor_converter as indoor
from tools.dataset_converters import kitti_converter as kitti
from tools.dataset_converters import lyft_converter as lyft_converter
from tools.dataset_converters import nuscenes_converter as nuscenes_converter
from tools.dataset_converters import semantickitti_converter
from tools.dataset_converters.create_gt_database import (
    GTDatabaseCreater, create_groundtruth_database)
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos


def kitti_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    with_plane=False):
    """Prepare data related to Kitti dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
    """
    kitti.create_kitti_info_file(root_path, info_prefix, with_plane)
    kitti.create_reduced_point_cloud(root_path, info_prefix)

    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    info_trainval_path = osp.join(out_dir, f'{info_prefix}_infos_trainval.pkl')
    info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_val_path)
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_trainval_path)
    update_pkl_infos('kitti', out_dir=out_dir, pkl_path=info_test_path)
    create_groundtruth_database(
        'KittiDataset',
        root_path,
        info_prefix,
        f'{info_prefix}_infos_train.pkl',
        relative_path=False,
        mask_anno_path='instances_train.json',
        with_mask=(version == 'mask'))


def nuscenes_data_prep(root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    if version == 'v1.0-test':
        info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
        update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_test_path)
        return

    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_val_path)
    create_groundtruth_database(dataset_name, root_path, info_prefix,
                                f'{info_prefix}_infos_train.pkl')


def lyft_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to Lyft dataset.

    Related data consists of '.pkl' files recording basic infos.
    Although the ground truth database and 2D annotations are not used in
    Lyft, it can also be generated like nuScenes.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Defaults to 10.
    """
    lyft_converter.create_lyft_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)
    if version == 'v1.01-test':
        info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
        update_pkl_infos('lyft', out_dir=root_path, pkl_path=info_test_path)
    elif version == 'v1.01-train':
        info_train_path = osp.join(root_path, f'{info_prefix}_infos_train.pkl')
        info_val_path = osp.join(root_path, f'{info_prefix}_infos_val.pkl')
        update_pkl_infos('lyft', out_dir=root_path, pkl_path=info_train_path)
        update_pkl_infos('lyft', out_dir=root_path, pkl_path=info_val_path)


def scannet_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for scannet dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    info_test_path = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
    update_pkl_infos('scannet', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('scannet', out_dir=out_dir, pkl_path=info_val_path)
    update_pkl_infos('scannet', out_dir=out_dir, pkl_path=info_test_path)


def s3dis_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for s3dis dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    splits = [f'Area_{i}' for i in [1, 2, 3, 4, 5, 6]]
    for split in splits:
        filename = osp.join(out_dir, f'{info_prefix}_infos_{split}.pkl')
        update_pkl_infos('s3dis', out_dir=out_dir, pkl_path=filename)


def sunrgbd_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for sunrgbd dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    indoor.create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    update_pkl_infos('sunrgbd', out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos('sunrgbd', out_dir=out_dir, pkl_path=info_val_path)


def waymo_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    workers,
                    max_sweeps=10,
                    only_gt_database=False,
                    save_senor_data=False,
                    skip_cam_instances_infos=False):
    """Prepare waymo dataset. There are 3 steps as follows:

    Step 1. Extract camera images and lidar point clouds from waymo raw
        data in '*.tfreord' and save as kitti format.
    Step 2. Generate waymo train/val/test infos and save as pickle file.
    Step 3. Generate waymo ground truth database (point clouds within
        each 3D bounding box) for data augmentation in training.
    Steps 1 and 2 will be done in Waymo2KITTI, and step 3 will be done in
    GTDatabaseCreater.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default to 10. Here we store ego2global information of these
            frames for later use.
        only_gt_database (bool, optional): Whether to only generate ground
            truth database. Default to False.
        save_senor_data (bool, optional): Whether to skip saving
            image and lidar. Default to False.
        skip_cam_instances_infos (bool, optional): Whether to skip
            gathering cam_instances infos in Step 2. Default to False.
    """
    from tools.dataset_converters import waymo_converter as waymo

    if version == 'v1.4':
        splits = [
            'training', 'validation', 'testing',
            'testing_3d_camera_only_detection'
        ]
    elif version == 'v1.4-mini':
        splits = ['training', 'validation']
    else:
        raise NotImplementedError(f'Unsupported Waymo version {version}!')
    out_dir = osp.join(out_dir, 'kitti_format')

    if not only_gt_database:
        for i, split in enumerate(splits):
            load_dir = osp.join(root_path, 'waymo_format', split)
            if split == 'validation':
                save_dir = osp.join(out_dir, 'training')
            else:
                save_dir = osp.join(out_dir, split)
            converter = waymo.Waymo2KITTI(
                load_dir,
                save_dir,
                prefix=str(i),
                workers=workers,
                test_mode=(split
                           in ['testing', 'testing_3d_camera_only_detection']),
                info_prefix=info_prefix,
                max_sweeps=max_sweeps,
                split=split,
                save_senor_data=save_senor_data,
                save_cam_instances=not skip_cam_instances_infos)
            converter.convert()
            if split == 'validation':
                converter.merge_trainval_infos()

        from tools.dataset_converters.waymo_converter import \
            create_ImageSets_img_ids
        create_ImageSets_img_ids(out_dir, splits)

    GTDatabaseCreater(
        'WaymoDataset',
        out_dir,
        info_prefix,
        f'{info_prefix}_infos_train.pkl',
        relative_path=False,
        with_mask=False,
        num_worker=workers).create()

    print_log('Successfully preparing Waymo Open Dataset')


def semantickitti_data_prep(info_prefix, out_dir):
    """Prepare the info file for SemanticKITTI dataset.

    Args:
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
    """
    semantickitti_converter.create_semantickitti_info_file(
        info_prefix, out_dir)


# ============================================================================
# NEW: Custom NuScenes dataset support
# ============================================================================
def custom_nuscenes_data_prep(root_path,
                               info_prefix,
                               dataset_name,
                               out_dir,
                               max_sweeps=10,
                               custom_splits_file=None):
    """Prepare data for custom NuScenes-format dataset with custom splits.
    
    This is a wrapper around nuscenes_data_prep that handles custom splits
    by patching them into the nuscenes.utils.splits module temporarily.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int, optional): Number of input consecutive frames.
        custom_splits_file (str): Path to custom splits.py file.
    """
    import importlib.util
    from nuscenes.utils import splits as nusc_splits
    
    # Load custom splits
    spec = importlib.util.spec_from_file_location("custom_splits", custom_splits_file)
    custom_splits_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_splits_module)
    
    # Find train and val splits
    train_split = None
    val_split = None
    available_splits = {}
    
    for attr_name in dir(custom_splits_module):
        if not attr_name.startswith('_'):
            attr_val = getattr(custom_splits_module, attr_name)
            if isinstance(attr_val, list):
                available_splits[attr_name] = attr_val
    
    # Prioritize non-mini splits
    for attr_name, attr_val in available_splits.items():
        attr_lower = attr_name.lower()
        if 'train' in attr_lower and 'mini' not in attr_lower and train_split is None:
            train_split = attr_val
            train_split_name = attr_name
        if 'val' in attr_lower and 'mini' not in attr_lower and val_split is None:
            val_split = attr_val
            val_split_name = attr_name
    
    # Fallback to mini splits if regular ones not found
    if train_split is None or val_split is None:
        for attr_name, attr_val in available_splits.items():
            attr_lower = attr_name.lower()
            if 'train' in attr_lower and train_split is None:
                train_split = attr_val
                train_split_name = attr_name
            if 'val' in attr_lower and val_split is None:
                val_split = attr_val
                val_split_name = attr_name
    
    if train_split is None or val_split is None:
        raise ValueError(
            f"Could not find train/val splits in {custom_splits_file}\n"
            f"Available splits: {list(available_splits.keys())}\n"
            f"Found train: {train_split is not None}, Found val: {val_split is not None}"
        )
    
    print(f"Using custom splits from {custom_splits_file}")
    print(f"Using '{train_split_name}' for training: {len(train_split)} scenes")
    print(f"Using '{val_split_name}' for validation: {len(val_split)} scenes")
    
    # Detect which version directory actually exists
    import os
    version_to_use = None
    for possible_version in ['v1.0-trainval', 'v1.0-mini', 'v1.0-test']:
        version_path = osp.join(root_path, possible_version)
        if os.path.exists(version_path):
            version_to_use = possible_version
            print(f"Detected version directory: {version_to_use}")
            break
    
    if version_to_use is None:
        raise ValueError(
            f"Could not find NuScenes version directory in {root_path}\n"
            f"Expected one of: v1.0-trainval, v1.0-mini, v1.0-test"
        )
    
    # Temporarily patch nuscenes splits
    original_train = getattr(nusc_splits, 'train', None)
    original_val = getattr(nusc_splits, 'val', None)
    
    try:
        nusc_splits.train = train_split
        nusc_splits.val = val_split
        
        # Use standard nuscenes converter to create the pkl files
        nuscenes_converter.create_nuscenes_infos(
            root_path, info_prefix, version=version_to_use, max_sweeps=max_sweeps)
        
        # Update pkl files to v2 format (required for GT database creation)
        # Work around hardcoded paths in update_infos_to_v2.py by temporarily
        # changing to the parent directory
        import os
        original_cwd = os.getcwd()
        
        # The update script expects to find data at ./data/nuscenes/
        # So we need to be in the directory where ./data/ would resolve correctly
        # Check if we need to change directory
        expected_relative_path = './data/nuscenes'
        if not osp.exists(expected_relative_path):
            # Try to figure out the right directory
            # If root_path is /data/HangQiu/.../data/nuscenes
            # We need to be at /data/HangQiu/.../
            parent_of_data = osp.dirname(osp.dirname(root_path))
            if osp.basename(osp.dirname(root_path)) == 'data':
                print(f"Changing to {parent_of_data} to work around hardcoded paths in update_infos_to_v2.py")
                os.chdir(parent_of_data)
        
        try:
            info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
            info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
            
            update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_train_path)
            update_pkl_infos('nuscenes', out_dir=out_dir, pkl_path=info_val_path)
            
            print(f"Updated info files to v2 format:")
            print(f"  - {info_train_path}")
            print(f"  - {info_val_path}")
        finally:
            # Restore original directory
            os.chdir(original_cwd)
        
        # Create GT database
        create_groundtruth_database(dataset_name, root_path, info_prefix,
                                    f'{info_prefix}_infos_train.pkl')
    finally:
        # Restore original splits
        if original_train is not None:
            nusc_splits.train = original_train
        if original_val is not None:
            nusc_splits.val = original_val
# ============================================================================


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--with-plane',
    action='store_true',
    help='Whether to use plane information for kitti.')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
parser.add_argument(
    '--only-gt-database',
    action='store_true',
    help='''Whether to only generate ground truth database.
        Only used when dataset is NuScenes or Waymo!''')
parser.add_argument(
    '--skip-cam_instances-infos',
    action='store_true',
    help='''Whether to skip gathering cam_instances infos.
        Only used when dataset is Waymo!''')
parser.add_argument(
    '--skip-saving-sensor-data',
    action='store_true',
    help='''Whether to skip saving image and lidar.
        Only used when dataset is Waymo!''')
# NEW: Add custom splits argument
parser.add_argument(
    '--custom-splits',
    type=str,
    default=None,
    help='''Path to custom splits.py file for NuScenes-format datasets.
        Example: data/sim_nuscenes/splits.py''')

args = parser.parse_args()

if __name__ == '__main__':
    from mmengine.registry import init_default_scope
    init_default_scope('mmdet3d')

    if args.dataset == 'kitti':
        if args.only_gt_database:
            create_groundtruth_database(
                'KittiDataset',
                args.root_path,
                args.extra_tag,
                f'{args.extra_tag}_infos_train.pkl',
                relative_path=False,
                mask_anno_path='instances_train.json',
                with_mask=(args.version == 'mask'))
        else:
            kitti_data_prep(
                root_path=args.root_path,
                info_prefix=args.extra_tag,
                version=args.version,
                out_dir=args.out_dir,
                with_plane=args.with_plane)
    elif args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        if args.only_gt_database:
            create_groundtruth_database('NuScenesDataset', args.root_path,
                                        args.extra_tag,
                                        f'{args.extra_tag}_infos_train.pkl')
        else:
            # NEW: Handle custom splits if provided
            if args.custom_splits:
                custom_nuscenes_data_prep(
                    root_path=args.root_path,
                    info_prefix=args.extra_tag,
                    dataset_name='NuScenesDataset',
                    out_dir=args.out_dir,
                    max_sweeps=args.max_sweeps,
                    custom_splits_file=args.custom_splits)
            else:
                # Standard NuScenes processing (unchanged)
                train_version = f'{args.version}-trainval'
                nuscenes_data_prep(
                    root_path=args.root_path,
                    info_prefix=args.extra_tag,
                    version=train_version,
                    dataset_name='NuScenesDataset',
                    out_dir=args.out_dir,
                    max_sweeps=args.max_sweeps)
                test_version = f'{args.version}-test'
                nuscenes_data_prep(
                    root_path=args.root_path,
                    info_prefix=args.extra_tag,
                    version=test_version,
                    dataset_name='NuScenesDataset',
                    out_dir=args.out_dir,
                    max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        if args.only_gt_database:
            create_groundtruth_database('NuScenesDataset', args.root_path,
                                        args.extra_tag,
                                        f'{args.extra_tag}_infos_train.pkl')
        else:
            # NEW: Handle custom splits if provided
            if args.custom_splits:
                custom_nuscenes_data_prep(
                    root_path=args.root_path,
                    info_prefix=args.extra_tag,
                    dataset_name='NuScenesDataset',
                    out_dir=args.out_dir,
                    max_sweeps=args.max_sweeps,
                    custom_splits_file=args.custom_splits)
            else:
                # Standard NuScenes mini processing (unchanged)
                train_version = f'{args.version}'
                nuscenes_data_prep(
                    root_path=args.root_path,
                    info_prefix=args.extra_tag,
                    version=train_version,
                    dataset_name='NuScenesDataset',
                    out_dir=args.out_dir,
                    max_sweeps=args.max_sweeps)
    elif args.dataset == 'waymo':
        waymo_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            workers=args.workers,
            max_sweeps=args.max_sweeps,
            only_gt_database=args.only_gt_database,
            save_senor_data=not args.skip_saving_sensor_data,
            skip_cam_instances_infos=args.skip_cam_instances_infos)
    elif args.dataset == 'lyft':
        train_version = f'{args.version}-train'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            max_sweeps=args.max_sweeps)
        test_version = f'{args.version}-test'
        lyft_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=test_version,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'scannet':
        scannet_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 's3dis':
        s3dis_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'sunrgbd':
        sunrgbd_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
    elif args.dataset == 'semantickitti':
        semantickitti_data_prep(
            info_prefix=args.extra_tag, out_dir=args.out_dir)
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')