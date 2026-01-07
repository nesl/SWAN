# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import time
import torch
import torch.nn as nn

import torchvision

import torch.nn.functional as F
import numpy as np

from mmdet3d.structures import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from .cmt_utils.grid_mask import GridMask
from mmdet3d.registry import MODELS
from torchvision.utils import save_image, make_grid


def save_masked_grid(imgs, msk, output_path="masked_grid.png", alpha=0.5):
    """
    Saves a grid of the 6 images from the first batch item, masking out 
    patches where the mask value is 0.
    
    Args:
        images: Tensor of shape (B, 6, 3, H, W). Assumed to be normalized in [0, 1].
        masks: Tensor of shape (B, 6, 20, 50). Binary mask (0 or 1).
        output_path: Filename for the saved image.
    """
    # 1. Select the first item in the batch
    # Shapes: imgs -> (6, 3, H, W), msk -> (6, 20, 50)
    # imgs = images[0].clone()
    # msk = masks[0].clone()

    # # 2. Resize the mask to match the image resolution
    # # We need to add a channel dim for interpolation: (6, 20, 50) -> (6, 1, 20, 50)
    # msk = msk.unsqueeze(1).float()
    
    # Use 'nearest' interpolation to keep the patch blocks sharp and distinct
    # Output shape: (6, 1, H, W)
    H, W = imgs.shape[-2:]
    msk_resized = F.interpolate(msk, size=(H, W), mode='nearest')


    overlay_color = torch.tensor([1.0, 0.0, 0.0], device=imgs.device).view(1, 3, 1, 1)

    # 4. Apply Translucent Blending
    # Logic: The final pixel is a weighted sum of the original pixel and the 
    # overlay color, based on the alpha value.
    # Formula: Blended = Original * (1 - alpha) + Overlay * alpha
    
    # Create the full overlay image (solid red everywhere)
    overlay_layer = torch.ones_like(imgs) * overlay_color
    
    # Calculate the blended version of the whole image
    blended_imgs = imgs * (1 - alpha) + overlay_layer * alpha

    # Final composition:
    # Where mask is 1, use original image.
    # Where mask is 0, use the blended image.
    final_imgs = imgs * msk_resized + blended_imgs * (1 - msk_resized)

    # 4. Create a grid
    # nrow=3 creates a standard 2x3 layout for the 6 images
    grid = make_grid(final_imgs, nrow=3, padding=2, normalize=False)

    # 5. Save the image
    save_image(grid, output_path)


@MODELS.register_module()
class CmtDetector(MVXTwoStageDetector):

    def __init__(self,
                 use_grid_mask=False,
                 enable_pruning=False,
                 **kwargs):
        super(CmtDetector, self).__init__(**kwargs)
        
        self.use_grid_mask = use_grid_mask
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.enable_pruning = enable_pruning
        self.img_pruner = torch.nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.lidar_pruner = torch.nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.mask_bias_value=0.2
        self.cutoff_ratio = 0.1
        # if pts_voxel_cfg:
        #     self.pts_voxel_layer = SPConvVoxelization(**pts_voxel_cfg)
        self.total_latency = 0
        self.img_extract_latency = 0
        self.lidar_extract_latency = 0
        self.output_bbox_latency = 0
        self.sample_count = 0

        self.num_lidar_tokens = 0
        self.num_image_tokens = 0


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
            start = time.time() 
            img_feats = self.img_backbone(img.float())
            self.img_extract_latency += time.time() - start
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, voxel_dict, points, img_feats, batch_input_metas):
        with torch.autocast('cuda', enabled=False):
            voxels = voxel_dict['voxels']
            coors = voxel_dict['coors']
            """Extract features of points."""
            voxel_features = self.pts_voxel_encoder(voxels, voxel_dict['num_points'], coors)
            batch_size = coors[-1, 0] + 1
            start = time.time()
            x = self.pts_middle_encoder(voxel_features, coors, batch_size)
            x = self.pts_backbone(x)
            self.lidar_extract_latency += time.time() - start
            if self.with_pts_neck:
                x = self.pts_neck(x)
            return x


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
                          gt_bboxes_ignore=None,
                          img_mask = None,
                          pts_mask = None):
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
            outs = self.pts_bbox_head(pts_feats, img_feats, img_metas, pts_mask=pts_mask, img_mask=img_mask)
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
        voxels, coors = batch_inputs_dict['voxels']['voxels'], batch_inputs_dict['voxels']['coors']

        start = time.time()
        img_feats, pts_feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
        import pdb; pdb.set_trace()
        losses = dict()
        if pts_feats or img_feats:
            points_mask, img_mask = None, None
            if self.enable_pruning:
                points_mask = self.lidar_pruner(pts_feats[0])
                img_mask = self.img_pruner(img_feats[0])

                points_mask_truncated = (torch.clone(points_mask) + self.mask_bias_value).round()
                img_mask_truncated = (torch.clone(img_mask) + self.mask_bias_value).round()

                points_mask = points_mask + (points_mask_truncated - points_mask).detach()
                img_mask = img_mask + (img_mask_truncated - img_mask).detach()

                pts_feats[0] = points_mask * pts_feats[0]
                img_feats = [img_mask * img_feats[0]]
            gt_bboxes_3d = [sample.gt_instances_3d.bboxes_3d for sample in batch_data_samples]
            
            # Extract GT Labels (List[Tensor])
            gt_labels_3d = [sample.gt_instances_3d.labels_3d for sample in batch_data_samples]
            # Extract Ignore Masks (Optional - usually None for standard 3D detection)
            # Old code usually defaults this to None if not present
            gt_bboxes_ignore = None
            losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, batch_input_metas,
                                                gt_bboxes_ignore, pts_mask=points_mask, img_mask=img_mask)
            # Binarization + L1 sparsity
            losses_pts['img_mask_loss'] = torch.mean(img_mask)
            losses_pts['pts_mask_loss'] = torch.mean(points_mask)
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
        predict_start = time.time()
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        img_feats, pts_feats = self.extract_feat(batch_inputs_dict,
                                                 batch_input_metas)

        if self.enable_pruning:
            points_mask = (self.lidar_pruner(pts_feats[0]) + self.mask_bias_value).round()
            img_mask = (self.img_pruner(img_feats[0]) + self.mask_bias_value).round()
            pts_feats[0] = points_mask * pts_feats[0]
            img_feats = [img_mask * img_feats[0]] # why did i do this?
            self.num_lidar_tokens += torch.sum(points_mask)
            self.num_image_tokens += torch.sum(img_mask)

            # img_mask is shape (B, 6, 20, 50)
            # print('------------------------------------------------')
            # print(f'Total Lidar Tokens: {self.num_lidar_tokens / self.sample_count}')
            # print(f'Total Image Tokens: {self.num_image_tokens / self.sample_count}\n')
            # print(f'Current Lidar Tokens: {torch.sum(points_mask)}')
            # print(f'Current Image Tokens: {torch.sum(img_mask)}')
            # print('------------------------------------------------')
            # if torch.sum(img_mask) > 1000:
            #     mean=torch.tensor([[103.530, 116.280, 123.675]]).cuda()
            #     std=torch.tensor([[57.375, 57.120, 58.395]]).cuda()
            #     sample_imgs = batch_inputs_dict['imgs'] * std[..., None, None] + mean[..., None, None]
            #     sample_imgs = sample_imgs[:, [2, 1, 0]] / 255 # Shuffle the channels since we load in BGR I think
            #     save_masked_grid(sample_imgs, img_mask)
            

            # torchvision.utils.save_image(batch_inputs_dict['imgs'][:, [2, 1, 0]], f'cmt_images/test_{self.sample_count}_{torch.sum(points_mask)}_{torch.sum(img_mask)}.png', nrow=3)
        
        self.sample_count += 1
        start = time.time()
        outs = self.pts_bbox_head(pts_feats, img_feats, batch_input_metas, img_mask=img_mask, pts_mask=points_mask)
        self.output_bbox_latency += time.time() - start
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, batch_input_metas, rescale=False)
        
        # bbox_results = [
        #     bbox3d2result(bboxes, scores, labels)
        #     for bboxes, scores, labels in bbox_list
        # ] 
        self.total_latency += time.time() - predict_start
        # print('------------------------------------------------')
        # print(f'Avg Overall Time: {self.total_latency / self.sample_count}')
        # print(f'\tAvg Img Backbone Time: {self.img_extract_latency / self.sample_count}')
        # print(f'\tAvg LiDAR Backbone Time: {self.lidar_extract_latency / self.sample_count}')
        # print(f'\tAvg Feat Fusion + PE Time: {self.pts_bbox_head.feat_fusion_latency / self.sample_count}')
        # # print(f'\tAvg Bbox Pred Time: {self.pts_bbox_head.bbox_pred_latency / self.sample_count}')
        # # print(f'\tAvg Bbox Format Time: {self.output_bbox_latency / self.sample_count}')
        # print(f'Avg RV_PE: {self.pts_bbox_head.rv_pe_latency / self.sample_count}')
        # print(f'Avg BEV Embed: {self.pts_bbox_head.bev_embedding_latency / self.sample_count}')
        # print(f'Avg Decoder Fusion: {self.pts_bbox_head.decoder_fusion_latency / self.sample_count}')
        # print('------------------------------------------------')
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
