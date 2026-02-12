# Custom dataset converter for our simulated nuscenes dataset generated with Carla.

import os.path as osp
import importlib.util


def load_custom_splits(splits_file_path):
    """
    Load custom splits from a Python file.
    
    Args:
        splits_file_path (str): Path to custom splits.py file
        
    Returns:
        module: The loaded splits module
    """
    if not osp.exists(splits_file_path):
        raise FileNotFoundError(f"Custom splits file not found: {splits_file_path}")
    
    spec = importlib.util.spec_from_file_location("custom_splits", splits_file_path)
    splits_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(splits_module)
    
    return splits_module


def create_nuscenes_infos_with_custom_splits(root_path,
                                             info_prefix,
                                             custom_splits_file,
                                             train_split_name='sim_train',
                                             val_split_name='sim_val',
                                             version='v1.0-trainval',
                                             max_sweeps=10):
    """
    Create NuScenes info files using custom splits by temporarily patching
    the nuscenes splits module.
    
    This is a lightweight wrapper that reuses ALL the standard nuscenes converter
    logic by just injecting custom splits into the nuscenes.utils.splits namespace.
    
    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        custom_splits_file (str): Path to custom splits.py file.
        train_split_name (str): Name of train split variable in splits file.
        val_split_name (str): Name of val split variable in splits file.
        version (str): Version string for the dataset.
        max_sweeps (int): Max number of sweeps.
    """
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits as nusc_splits
    
    # Load custom splits
    custom_splits = load_custom_splits(custom_splits_file)
    
    # Get the split lists
    train_scenes = getattr(custom_splits, train_split_name)
    val_scenes = getattr(custom_splits, val_split_name)
    
    print(f"Loaded custom splits from {custom_splits_file}")
    print(f"Train split '{train_split_name}': {len(train_scenes)} scenes")
    print(f"Val split '{val_split_name}': {len(val_scenes)} scenes")
    
    # Temporarily inject into nuscenes splits module
    # This allows us to reuse ALL of the standard converter logic
    original_train = getattr(nusc_splits, 'train', None)
    original_val = getattr(nusc_splits, 'val', None)
    
    try:
        # Patch the splits
        nusc_splits.train = train_scenes
        nusc_splits.val = val_scenes
        
        # Now just call the standard converter!
        # It will use our patched splits
        from tools.dataset_converters import nuscenes_converter
        nuscenes_converter.create_nuscenes_infos(
            root_path=root_path,
            info_prefix=info_prefix,
            version='v1.0-trainval',  # Use standard version, splits are already patched
            max_sweeps=max_sweeps
        )
        
    finally:
        # Restore original splits (good practice)
        if original_train is not None:
            nusc_splits.train = original_train
        if original_val is not None:
            nusc_splits.val = original_val


def create_nuscenes_infos_auto(root_path,
                               info_prefix,
                               version='v1.0-trainval',
                               max_sweeps=10,
                               custom_splits_file=None):
    """
    Smart wrapper that auto-detects whether to use custom or standard splits.
    
    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
        max_sweeps (int): Max number of sweeps.
        custom_splits_file (str, optional): Path to custom splits file.
            If None and version is not standard, will look for splits.py in root_path.
    """
    standard_versions = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    
    # If standard version and no custom splits specified, use standard converter
    if version in standard_versions and custom_splits_file is None:
        print(f"Using standard NuScenes converter for version {version}")
        from tools.dataset_converters import nuscenes_converter
        nuscenes_converter.create_nuscenes_infos(
            root_path=root_path,
            info_prefix=info_prefix,
            version=version,
            max_sweeps=max_sweeps
        )
        return
    
    # Otherwise, use custom splits
    if custom_splits_file is None:
        # Auto-detect splits.py in root path
        custom_splits_file = osp.join(root_path, 'splits.py')
        if not osp.exists(custom_splits_file):
            raise FileNotFoundError(
                f"No custom splits file found at {custom_splits_file}. "
                f"Please provide --custom-splits argument or place splits.py in dataset root."
            )
        print(f"Auto-detected custom splits file: {custom_splits_file}")
    
    # Determine split names from version string
    if 'mini' in version.lower():
        train_split = 'sim_mini_train'
        val_split = 'sim_mini_val'
    elif 'test' in version.lower():
        train_split = 'sim_test'
        val_split = None
    else:
        train_split = 'sim_train'
        val_split = 'sim_val'
    
    # Try common naming patterns
    custom_splits = load_custom_splits(custom_splits_file)
    
    # Find actual split names
    available_splits = [s for s in dir(custom_splits) if not s.startswith('_') and isinstance(getattr(custom_splits, s), list)]
    
    # Smart matching for train split
    if not hasattr(custom_splits, train_split):
        for split_name in available_splits:
            if 'train' in split_name.lower() and ('mini' in version.lower()) == ('mini' in split_name.lower()):
                train_split = split_name
                break
    
    # Smart matching for val split
    if val_split and not hasattr(custom_splits, val_split):
        for split_name in available_splits:
            if 'val' in split_name.lower() and ('mini' in version.lower()) == ('mini' in split_name.lower()):
                val_split = split_name
                break
    
    if 'test' in version.lower():
        # For test, reuse train split loader but mark as test
        create_nuscenes_infos_with_custom_splits(
            root_path=root_path,
            info_prefix=info_prefix,
            custom_splits_file=custom_splits_file,
            train_split_name=train_split,
            val_split_name=train_split,  # Dummy, won't be used
            version='v1.0-test',
            max_sweeps=max_sweeps
        )
    else:
        create_nuscenes_infos_with_custom_splits(
            root_path=root_path,
            info_prefix=info_prefix,
            custom_splits_file=custom_splits_file,
            train_split_name=train_split,
            val_split_name=val_split,
            version='v1.0-trainval',
            max_sweeps=max_sweeps
        )


# Backward compatibility - keep the standard function available
def create_nuscenes_infos(root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10,
                          custom_splits_file=None):
    """
    Wrapper that maintains backward compatibility with standard nuscenes_converter
    while adding custom splits support.
    """
    return create_nuscenes_infos_auto(
        root_path=root_path,
        info_prefix=info_prefix,
        version=version,
        max_sweeps=max_sweeps,
        custom_splits_file=custom_splits_file
    )