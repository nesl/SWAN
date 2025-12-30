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
       # if(self.pts_backbone is not None):
            #torch.compile(self.pts_backbone)
        #if(self.pts_neck is not None):
           # torch.compile(self.pts_neck)

    def init_weights(self) -> None:
        super().init_weights()

    def extract_feat(self, batch_inputs_dict: dict) -> dict:
        """
        # due to this complexity, we will most likely have to re-implement the whole training pipeline given a differnet backbone
        # As there is not common sparse encoder/decoder transformer interface in mmdet3d, we also can't assume they are the same
        VFE -> Masking -> Sparse Encoder -> Sparse Decoder
        """
        batch_size = len(batch_inputs_dict['points'])
        # Voxelize
        voxels = batch_inputs_dict['voxels']['voxels']
        coors = batch_inputs_dict['voxels']['coors']
        vfe_outs = self.pts_voxel_encoder(voxels, coors, batch_size)
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
            point_coors=point_coors,
            batch_size=batch_size
        )

        # process masked
        voxel_info_encoder = self.pts_backbone(voxel_info_encoder)

        # Maps unmasked features back to their original positions.
        _, voxel_info_decoder, _ = self.pts_neck(voxel_info_encoder)

        return voxel_info_decoder

    
    def predict(self,
                batch_inputs_dict: dict,
                batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict function for pretraining visualization/evaluation."""
        features = self.extract_feat(batch_inputs_dict)

        # Forward through head with show=True to get detailed pretrain outputs
        outs = self.bbox_head(features, show=True)
        pred_dict = outs[0]  # As in UniM2AE

        batch_size = len(batch_data_samples)
        vx, vy, vz = self.pts_middle_encoder.sparse_shape

        voxel_coors = pred_dict["voxel_coors"]
        masked_voxel_coors = pred_dict["masked_voxel_coors"]
        unmasked_voxel_coors = pred_dict["unmasked_voxel_coors"]

        # === Build occupied_bev ===
        occupied = None
        if "pred_occupied" in pred_dict:
            device = pred_dict["pred_occupied"].device
            occupied = -torch.ones((batch_size, vx, vy), dtype=torch.long, device=device)
            index = (voxel_coors[:, 0], voxel_coors[:, 3], voxel_coors[:, 2])
            unmasked_index = (unmasked_voxel_coors[:, 0], unmasked_voxel_coors[:, 3], unmasked_voxel_coors[:, 2])

            gt_occupied = pred_dict["gt_occupied"].long() + 1
            occupied[index] = 2 * gt_occupied
            occupied[unmasked_index] -= 2
            occupied[index] += (torch.sigmoid(pred_dict["pred_occupied"]) + 0.5).long()

        # === Build gt_num_points_bev & diff ===
        gt_num_points = None
        diff_num_points = None
        if "pred_num_points_masked" in pred_dict or "pred_num_points_unmasked" in pred_dict:
            device = next(v.device for v in pred_dict.values() if hasattr(v, 'device'))
            gt_num_points = torch.zeros((batch_size, vx, vy), dtype=torch.long, device=device)
            diff_num_points = torch.zeros((batch_size, vx, vy), dtype=torch.float, device=device)

            if "pred_num_points_masked" in pred_dict:
                index = (masked_voxel_coors[:, 0], masked_voxel_coors[:, 3], masked_voxel_coors[:, 2])
                gt_num_points[index] = pred_dict["gt_num_points_masked"].long()
                diff_num_points[index] = pred_dict["gt_num_points_masked"].float() - pred_dict["pred_num_points_masked"]

            if "pred_num_points_unmasked" in pred_dict:
                index = (unmasked_voxel_coors[:, 0], unmasked_voxel_coors[:, 3], unmasked_voxel_coors[:, 2])
                gt_num_points[index] = pred_dict["gt_num_points_unmasked"].long()
                diff_num_points[index] = pred_dict["gt_num_points_unmasked"].float() - pred_dict["pred_num_points_unmasked"]

        # === Reconstruct predicted points in world coordinates ===
        pred_points_list = []
        pred_points_batch_idx = []

        if "pred_points_masked" in pred_dict:
            pred_pts = pred_dict["pred_points_masked"].clone()  # (M, K, 3)
            M, K, _ = pred_pts.shape
            # Denormalize from [-1,1] to voxel size
            pred_pts[..., 0] *= self.pts_voxel_encoder.vx / 2
            pred_pts[..., 1] *= self.pts_voxel_encoder.vy / 2
            pred_pts[..., 2] *= self.pts_voxel_encoder.vz / 2
            # Shift to voxel center
            shift = torch.stack([
                masked_voxel_coors[:, 3] * self.pts_voxel_encoder.vx + self.pts_voxel_encoder.x_offset,
                masked_voxel_coors[:, 2] * self.pts_voxel_encoder.vy + self.pts_voxel_encoder.y_offset,
                masked_voxel_coors[:, 1] * self.pts_voxel_encoder.vz + self.pts_voxel_encoder.z_offset,
            ], dim=-1).unsqueeze(1)  # (M, 1, 3)
            pred_pts = pred_pts + shift
            pred_points_list.append(pred_pts.reshape(-1, 3))
            pred_points_batch_idx.append(masked_voxel_coors[:, 0].unsqueeze(1).repeat(1, K).reshape(-1))

        if "pred_points_unmasked" in pred_dict:
            pred_pts = pred_dict["pred_points_unmasked"]
            M, K, _ = pred_pts.shape
            pred_pts[..., 0] *= self.pts_voxel_encoder.vx / 2
            pred_pts[..., 1] *= self.pts_voxel_encoder.vy / 2
            pred_pts[..., 2] *= self.pts_voxel_encoder.vz / 2
            shift = torch.stack([
                unmasked_voxel_coors[:, 3] * self.pts_voxel_encoder.vx + self.pts_voxel_encoder.x_offset,
                unmasked_voxel_coors[:, 2] * self.pts_voxel_encoder.vy + self.pts_voxel_encoder.y_offset,
                unmasked_voxel_coors[:, 1] * self.pts_voxel_encoder.vz + self.pts_voxel_encoder.z_offset,
            ], dim=-1).unsqueeze(1)
            pred_pts = pred_pts + shift
            pred_points_list.append(pred_pts.reshape(-1, 3))
            pred_points_batch_idx.append(unmasked_voxel_coors[:, 0].unsqueeze(1).repeat(1, K).reshape(-1))

        pred_points_all = torch.cat(pred_points_list, dim=0) if pred_points_list else None
        pred_points_batch_all = torch.cat(pred_points_batch_idx, dim=0) if pred_points_batch_idx else None

        # === Attach everything to data_samples as custom fields ===
        for i, data_sample in enumerate(batch_data_samples):
            # Create a sub-dict for this sample
            data_sample.pred_pts = dict(
                occupied_bev=occupied[i] if occupied is not None else None,
                gt_num_points_bev=gt_num_points[i] if gt_num_points is not None else None,
                diff_num_points_bev=diff_num_points[i] if diff_num_points is not None else None,
                recon_points=pred_points_all[pred_points_batch_all == i] if pred_points_all is not None else None,
                point_cloud_range=self.pts_voxel_encoder.point_cloud_range,
                voxel_size=(self.pts_voxel_encoder.vx, self.pts_voxel_encoder.vy, self.pts_voxel_encoder.vz),
            )

        return batch_data_samples
        
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
    


