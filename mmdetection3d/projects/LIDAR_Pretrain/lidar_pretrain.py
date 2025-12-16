# 

from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList


from mmdet3d.structures import bbox3d2result

from ..UniM2AE.sst_models import DynamicVFE_New


import torch.nn as nn


from mmengine.structures import InstanceData

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import (ForwardResults,
                                                  OptSampleList, SampleList)
from mmdet3d.utils.typing_utils import (OptConfigType, OptInstanceList,
                                        OptMultiConfig)


@MODELS.register_module()
class LIDAR_PRETRAIN(Base3DDetector):

    def __init__(self,
                pts_voxel_encoder : Optional[dict] = None, 
                pts_middle_encoder: Optional[dict] = None,
                pts_backbone: Optional[dict] = None,
                pts_neck: Optional[dict] = None,
                bbox_head: Optional[dict] = None,
                Mask_config: Optional[dict] = None,
                Drop_config: Optional[dict] = None,
     
                 **kwargs):
        super(LIDAR_PRETRAIN, self).__init__(**kwargs)
        # Build Modules
        if pts_voxel_encoder:
            self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = MODELS.build(pts_middle_encoder)
        if pts_backbone:
            self.pts_backbone = MODELS.build(pts_backbone)
        if pts_neck:
            self.pts_neck = MODELS.build(pts_neck)
        if bbox_head:
            self.bbox_head = MODELS.build(bbox_head)

        self.init_weights()

    def init_weights(self) -> None:
        super().init_weights()

    def extract_feat(self, batch_inputs_dict: dict) -> dict:
        """
        # due to this complexity, we will most likely have to re-implement the whole training pipeline given a differnet backbone
        # As there is not common sparse encoder/decoder transformer interface in mmdet3d, we also can't assume they are the same
        VFE -> Masking -> Sparse Encoder -> Sparse Decoder
        """
        # Voxelize
        voxels = batch_inputs_dict['voxels']['voxels']
        coors = batch_inputs_dict['voxels']['coors']
        vfe_outs = self.pts_voxel_encoder(voxels, coors)
        voxel_feats = vfe_outs[0]
        voxel_coors = vfe_outs[1]
        low_level_point_feature = vfe_outs[2] 
        point_coors = vfe_outs[3] 

        # Geberate mask and GT
        # Create masked voxels
        voxel_info_encoder = self.pts_middle_encoder(
            voxel_feats=voxel_feats, 
            voxel_coors=voxel_coors, 
            low_level_point_feature=low_level_point_feature,
            point_coors=point_coors
        )

        # process masked
        voxel_info_encoder = self.pts_backbone(voxel_info_encoder)

        # Maps unmasked features back to their original positions.
        _, voxel_info_decoder, _ = self.pts_neck(voxel_info_encoder)

        return voxel_info_decoder

    
    def predict(self, batch_inputs_dict: dict, batch_data_samples: List[Det3DDataSample], **kwargs):
        # Inference mode
        voxel_info_decoder = self.extract_feat(batch_inputs_dict)
        pred_dict = self.bbox_head(voxel_info_decoder)
        return self.add_pred_to_datasample(batch_data_samples, pred_dict)
    
    def loss(self, batch_inputs_dict: dict, batch_data_samples: List[Det3DDataSample], **kwargs) -> dict:
        # Train mode
        features = self.extract_feat(batch_inputs_dict)
        pred_dict = self.bbox_head(features)
        if isinstance(pred_dict, tuple):
            pred_dict = pred_dict[0]
        losses = self.bbox_head.loss(pred_dict)
        
        return losses
    
    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass
    


