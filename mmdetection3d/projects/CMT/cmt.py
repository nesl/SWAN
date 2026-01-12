# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------


import torch

import torch.nn.functional as F
import numpy as np

from mmdet3d.models.voxel_encoders import DynamicVFE
from mmdet3d.structures import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from ..UniM2AE.sst_models import DynamicVFE_New

from .cmt_utils.grid_mask import GridMask
from mmdet3d.registry import MODELS
import torch.nn as nn

'''
Top level CMT model

'''

@MODELS.register_module()
class CmtDetector(MVXTwoStageDetector):

    def __init__(self,
                 use_grid_mask=False,
                 enable_modal_mask=False,
                 layerdrop_rate = 0.0,
                 test_img_retained_layers = None,
                 test_lidar_retained_layers = None,
                 **kwargs):
        super(CmtDetector, self).__init__(**kwargs)
        self.enable_modal_mask = enable_modal_mask
        self.use_grid_mask = use_grid_mask
        # GridMask will mask the input image to reduce overfitting to the images
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.layerdrop_rate = layerdrop_rate
        if self.with_img_backbone:
            self.test_img_retained_layers = test_img_retained_layers if test_img_retained_layers is not None else torch.ones(self.img_backbone.total_depth)
        if self.with_pts_backbone:
            self.test_lidar_retained_layers = test_lidar_retained_layers if test_lidar_retained_layers is not None else torch.ones(len(self.pts_middle_encoder.block_list) * 4)

    def init_weights(self):
        """Initialize model weights."""
        super(CmtDetector, self).init_weights()

    def extract_img_feat(self, img, img_metas, drop_all_img_layers=False):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1: # Batch size 1 we squeeze and remove the batch dimension since img backbone handles 4 dimensional input
                img.squeeze_(0)
            # Compress the batch dimension and the N dimension (6 cameras per sample)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            
            # LAYERDROPPING LOGIC: Generate random mask during training according to the layerdrop rate
            # 0 indicates a layer is DROPPED, 1 means the layer is RETAINED
            
            if self.training:
                retained_layers = (torch.rand(self.img_backbone.total_depth) > self.layerdrop_rate).int()
                if drop_all_img_layers:
                    retained_layers = torch.zeros_like(retained_layers)
            # Currently we just use all layers during inference
            # TODO Change this to accept a list of layers (or some instance variable) to test different layer allocations during inference
            else:
                retained_layers = self.test_img_retained_layers
                
            img_feats = self.img_backbone(img.float(), retained_layers)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None # Return none for img_feat if no img_backbone or img input
        
        # Pass through additional neck (usually a feature pyramid network that merges multiscale features)
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        return img_feats

    def extract_pts_feat(self, voxel_dict, points, img_feats, batch_input_metas, drop_all_lidar_layers=False):
        if self.with_pts_backbone:
            # Keep operations in float32 to preseve voxelization accuracy
            with torch.autocast('cuda', enabled=False):
                voxels = voxel_dict['voxels']
                coors = voxel_dict['coors']
                """Extract features of points."""

                # Depending on the type of voxel encoder, we pass it different parameters
                if isinstance(self.pts_voxel_encoder, DynamicVFE_New) or isinstance(self.pts_voxel_encoder, DynamicVFE):
                    voxel_features, coors, low_level_point_feature, indices = self.pts_voxel_encoder(voxels, coors)
                else:
                    voxel_features = self.pts_voxel_encoder(voxels, voxel_dict['num_points'], coors)
                batch_size = coors[-1, 0] + 1

                # LAYERDROP LOGIC: We have four layers per BasicBlock (hard coded by FlatFormer).
                if self.training:
                    retained_layers = (torch.rand(len(self.pts_middle_encoder.block_list) * 4) > self.layerdrop_rate).int()
                    if drop_all_lidar_layers:
                        retained_layers = torch.zeros_like(retained_layers)
                else:
                    retained_layers = self.test_lidar_retained_layers

                # Middle encoder is the FlatFormer model
                x = self.pts_middle_encoder(voxel_features, coors, batch_size, retained_layers)
                x = self.pts_backbone(x)
                if self.with_pts_neck:
                    x = self.pts_neck(x)

                # The calling functione expects a list
                if not isinstance(x, list):
                    x = [x]
                return x
        return [None]

    # This is the main feature extract function that calls the subfunctions for pts and images
    def extract_feat(self, batch_inputs_dict,
                     batch_input_metas):
        """Extract features from images and points.

        Args:
            batch_inputs_dict (dict): Dict of batch inputs. It
                contains

                - points (List[tensor]):  Point cloud of multiple inputs.
                - imgs (tensor): Image tensor with shape (B, C, H, W).
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.

        Returns:
             tuple: Two elements in tuple arrange as
             image features and point cloud features.
        """
        voxel_dict = batch_inputs_dict.get('voxels', None)
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)

        drop_all_lidar_layers, drop_all_img_layers = False, False
        if self.training and self.enable_modal_mask:
            # Modal mask by removing lidar with 50% probability during training, do not do this during inference
            if torch.rand(1).item() < 0.3:
                drop_all_lidar_layers = True
            elif torch.rand(1).item() < 0.2:
                drop_all_img_layers = True
                
        img_feats = self.extract_img_feat(imgs, batch_input_metas, drop_all_img_layers=drop_all_img_layers)
        pts_feats = self.extract_pts_feat(
            voxel_dict,
            points=points,
            img_feats=img_feats,
            batch_input_metas=batch_input_metas,
            drop_all_lidar_layers=drop_all_lidar_layers)
        
        # TODO: Drop the pts_feats to force the model to actually focus on image features

        
        # Remove lidar during testing, see img mAP
        # if not self.training:
        #     pts_feats = [item * 0.0 for item in pts_feats]

        return (img_feats, pts_feats)


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        losses = dict()
        if pts_feats or img_feats:
            losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        with torch.autocast('cuda', enabled=False):
            outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        if points is None:
            points = [None]
        if img is None:
            img = [None]
        for var, name in [(points, 'points'), (img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        return self.simple_test(points[0], img_metas[0], img[0], **kwargs)
    
    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        img_feats, pts_feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
        losses = dict()
        if pts_feats or img_feats:
            gt_bboxes_3d = [sample.gt_instances_3d.bboxes_3d for sample in batch_data_samples]
            
            # Extract GT Labels (List[Tensor])
            gt_labels_3d = [sample.gt_instances_3d.labels_3d for sample in batch_data_samples]
            # Extract Ignore Masks (Optional - usually None for standard 3D detection)
            # Old code usually defaults this to None if not present
            gt_bboxes_ignore = None
            losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, batch_input_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        return losses


    def predict(self, batch_inputs_dict,
                batch_data_samples,
                **kwargs):
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        img_feats, pts_feats = self.extract_feat(batch_inputs_dict,
                                                 batch_input_metas)
        
        outs = self.pts_bbox_head(pts_feats, img_feats, batch_input_metas)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, batch_input_metas, rescale=False)
        
        # bbox_results = [
        #     bbox3d2result(bboxes, scores, labels)
        #     for bboxes, scores, labels in bbox_list
        # ] 
        return self.add_pred_to_datasample(batch_data_samples, bbox_list)
        # Instead of previous, call the   res = self.add_pred_to_datasample(batch_data_samples, outputs)
        # to get a data3dsample thingy
        return bbox_results


    

    def simple_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, x_img, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ] 
        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False):
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        
        bbox_list = [dict() for i in range(len(img_metas))]
        if (pts_feats or img_feats) and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list
