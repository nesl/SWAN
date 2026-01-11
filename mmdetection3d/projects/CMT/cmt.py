# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------


import torch

import torch.nn.functional as F
import numpy as np

from mmdet3d.structures import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from ..UniM2AE.sst_models import DynamicVFE_New

from .cmt_utils.grid_mask import GridMask
from mmdet3d.registry import MODELS
import torch.nn as nn


@MODELS.register_module()
class CmtDetector(MVXTwoStageDetector):

    def __init__(self,
                 use_grid_mask=False,
                 enable_sst_swin=False,
                 **kwargs):
        super(CmtDetector, self).__init__(**kwargs)
        self.enable_sst_swin = enable_sst_swin
        if self.enable_sst_swin:
            self.deblock_lidar = nn.Sequential(
                nn.Conv3d(in_channels=128, out_channels=512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(num_features=512),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(num_features=512),
                nn.ReLU(inplace=True),
            )
            self.pts_backbone.load_state_dict(torch.load('model_checkpoints/sst_epoch50_pretrain.pth'))
            # self.img_backbone.load_state_dict(torch.load('model_checkpoints/fixed_swin.pth'))
        self.use_grid_mask = use_grid_mask
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.corruptset_debug = False # For debug purpose, remove it after shipping
        # if pts_voxel_cfg:
        #     self.pts_voxel_layer = SPConvVoxelization(**pts_voxel_cfg)

    def init_weights(self):
        """Initialize model weights."""
        super(CmtDetector, self).init_weights()

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img.float())
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        return img_feats

    def extract_pts_feat(self, voxel_dict, points, img_feats, batch_input_metas):
        if not self.with_pts_bbox:
            return None
        voxels = voxel_dict['voxels']
        coors = voxel_dict['coors']
        """Extract features of points."""
        # Handle empty voxels case (can happen with DDP data distribution)
        if len(coors) == 0:
            return None
        if isinstance(self.pts_voxel_encoder, DynamicVFE_New):
            voxel_features, coors, low_level_point_feature, indices = self.pts_voxel_encoder(voxels, coors)
        else:
            voxel_features = self.pts_voxel_encoder(voxels, voxel_dict['num_points'], coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        if self.enable_sst_swin:
            x = self.deblock_lidar(x)
            x = [x[..., 0]]
        return x if isinstance(x, list) else [x]


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
        if pts_feats is not None or img_feats is not None:
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

        if self.corruptset_debug is False:
            # Normal training
            # check whether we have corruption severity info in the metas
            if 'corruption_info' in batch_input_metas[0]:
                # extract corruption severity info and add them to the batch_inputs_dict
               
                corrupotion_info_list = [item['corruption_info'] for item in batch_input_metas]
                print("Corruption info found in metas, passing to model:", corrupotion_info_list)
            self.corruptset_debug = True



        img_feats, pts_feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
        losses = dict()
        # Always run forward pass to ensure all parameters receive gradients for DDP
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
