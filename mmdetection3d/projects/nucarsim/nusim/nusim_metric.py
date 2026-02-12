# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from typing import Dict, List, Optional
import tempfile

from mmdet3d.evaluation.metrics import NuScenesMetric
from mmdet3d.registry import METRICS
from mmengine.logging import MMLogger


@METRICS.register_module()
class NuSimMetric(NuScenesMetric):
    """NuScenes evaluation metric for simulation datasets with custom splits.
    
    This metric extends NuScenesMetric to support custom simulation datasets
    by working around the hardcoded split paths in nuscenes-devkit.
    
    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'bbox'.
        custom_splits_file (str, optional): Path to custom splits.py file.
            If provided, will patch nuscenes splits before evaluation.
        **kwargs: Other arguments passed to NuScenesMetric.
    
    Example:
        >>> val_evaluator = dict(
        >>>     type='SimNuScenesMetric',
        >>>     data_root='data/nuscenes/',
        >>>     ann_file='data/nuscenes/nuscenes_infos_val.pkl',
        >>>     metric='bbox',
        >>>     custom_splits_file='data/nuscenes/sim_nuscenes.py')
    """

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metric: str = 'bbox',
                 custom_splits_file: Optional[str] = None,
                 **kwargs) -> None:
        
        self.custom_splits_file = custom_splits_file
        
        # Patch nuscenes splits if custom splits provided
        if custom_splits_file and osp.exists(custom_splits_file):
            self._patch_nuscenes_splits(custom_splits_file)
        
        # Initialize parent
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metric=metric,
            **kwargs)
    
    def _patch_nuscenes_splits(self, splits_file: str) -> None:
        """Patch nuscenes.utils.splits with custom splits."""
        import importlib.util
        from nuscenes.utils import splits as nusc_splits
        
        # Load custom splits
        spec = importlib.util.spec_from_file_location("custom_splits", splits_file)
        custom_splits_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_splits_module)
        
        # Find and inject custom splits
        for attr_name in dir(custom_splits_module):
            if not attr_name.startswith('_'):
                attr_val = getattr(custom_splits_module, attr_name)
                if isinstance(attr_val, list):
                    # Inject into nuscenes splits module
                    setattr(nusc_splits, attr_name, attr_val)
        
        print(f"Patched nuscenes splits with custom splits from {splits_file}")
    
    def _evaluate_single(self,
                        result_path: str,
                        classes: Optional[List[str]] = None,
                        result_name: str = 'pred_instances_3d') -> Dict[str, float]:
        """Evaluation for a single model in nuScenes protocol.
        
        Overrides parent to handle custom eval_set mapping.
        """
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval
        import mmengine
        
        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)
        
        # For simulation datasets, always use sim_val split
        # The custom splits file must be patched before this point
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'sim_val',
        }

        eval_set = eval_set_map.get(self.version, 'sim_val')
        
        # Try evaluation
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set,
            output_dir=output_dir,
            verbose=False)
        
        nusc_eval.main(render_curves=False)
        
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