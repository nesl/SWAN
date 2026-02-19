from typing import Dict, List, Optional

import torch
from torch import Tensor

from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample


@MODELS.register_module()
class TransFusionDetector(MVXTwoStageDetector):
    """TransFusion detector for LiDAR-Camera 3D object detection.

    Inherits from MVXTwoStageDetector which provides:
    - extract_feat (voxelize -> voxel_encoder -> middle_encoder -> backbone -> neck)
    - loss (pts_bbox_head.loss)

    This subclass adds:
    - Image freezing logic for the two-stage training scheme
    - predict() override to pass batch_input_metas (list of dicts) instead
      of batch_data_samples to TransFusionHead.predict(), which expects dicts
    """

    def __init__(self, freeze_img: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.freeze_img = freeze_img
        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        img_feats, pts_feats = self.extract_feat(batch_inputs_dict,
                                                 batch_input_metas)
        #for input_meta in batch_input_metas:
         #   print(f"input_meta keys: {input_meta.keys()}")
            
        if img_feats is None:
            results_list_3d = self.pts_bbox_head.predict(
                (pts_feats, None), batch_input_metas, **kwargs)
        else:
            results_list_3d = self.pts_bbox_head.predict(
                (pts_feats, img_feats), batch_input_metas, **kwargs)

        detsamples = self.add_pred_to_datasample(batch_data_samples,
                                                 results_list_3d,
                                                 None)
        return detsamples
    

    def loss(self, batch_inputs_dict: Dict[List, torch.Tensor],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        """
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' and `imgs` keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Tensor of batch images, has shape
                  (B, C, H ,W)
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, .

        Returns:
            dict[str, Tensor]: A dictionary of loss components.

        """

        batch_input_metas = [item.metainfo for item in batch_data_samples]
        #for input_meta in batch_input_metas:
        #   print(f"input_meta keys: {input_meta.keys()}")
        img_feats, pts_feats = self.extract_feat(batch_inputs_dict,
                                                 batch_input_metas)
        losses = dict()
        
        if img_feats is None:
            loss = self.pts_bbox_head.loss((pts_feats, None), batch_data_samples, **kwargs)
        else:
            loss = self.pts_bbox_head.loss((pts_feats, img_feats), batch_data_samples, **kwargs)
        losses.update(loss )
        return losses
