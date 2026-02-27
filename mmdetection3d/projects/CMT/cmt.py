# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import time
import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np

from mmdet3d.structures import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from .cmt_utils.grid_mask import GridMask
from mmdet3d.registry import MODELS
from torchvision.utils import save_image, make_grid

from mmdet3d.models.voxel_encoders import DynamicVFE
from .cmt_utils import DynamicVFE_Linear
from mmdet3d.structures import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from ..UniM2AE.sst_models import DynamicVFE_New

from .cmt_utils.grid_mask import GridMask
from mmdet3d.registry import MODELS
import sys 
from mmengine.logging import MessageHub
from mmengine.dist import get_dist_info
import torchvision
import torch._inductor.config as config


# Define timing events
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

label_to_idx = {
    'none':0,
    'beamsreducing':1,
    'dark':2,
    'snow':3,
    'camera_fog':4,
    'lidar_fog':5,
    'camera_motionblur':6,
    'lidar_motionblur':7
}

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



'''
Top level CMT model

'''

@MODELS.register_module()
class CmtDetector(MVXTwoStageDetector):

    def __init__(self,
                 use_grid_mask=False,
                 enable_pruning=False,
                 use_hard_pruning=False,
                 enable_modal_mask=False,
                 enable_lidar_masking=False,
                 layerdrop_rate = 0.0,
                 test_img_retained_layers = None,
                 test_lidar_retained_layers = None,
                 controller = None,
                 lidar_early_exit_model = None,
                 camera_early_exit_model = None,
                 **kwargs):
        super(CmtDetector, self).__init__(**kwargs)

        self.use_grid_mask = use_grid_mask
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        # Modal mask and layerdrop are enabled together during training
        self.enable_modal_mask = enable_modal_mask
        self.enable_lidar_masking = enable_lidar_masking
        self.layerdrop_rate = layerdrop_rate

        self.enable_pruning = enable_pruning
        self.use_hard_pruning = use_hard_pruning
        assert (enable_pruning or not use_hard_pruning), "Hard pruning should only be enabled when we are pruning"

        # If enable pruning, define image and lidar mask generators
        if self.enable_pruning:
            self.img_pruner = torch.nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, padding=1),
                nn.InstanceNorm2d(num_features=16, affine=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
            self.lidar_pruner = torch.nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=16, kernel_size=3, padding=1),
                nn.InstanceNorm2d(num_features=16, affine=True),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )

            self.mask_bias_value=0.35
            self.num_lidar_tokens = 0
            self.num_image_tokens = 0

        if controller:
            self.controller = MODELS.build(controller)
        else:
            self.controller = None
        
        if lidar_early_exit_model:
            self.early_exit_lidar = MODELS.build(lidar_early_exit_model)
            
        else:
            self.early_exit_lidar = None

        if camera_early_exit_model:
            self.early_exit_camera= MODELS.build(camera_early_exit_model)
        else:
            self.early_exit_camera = None

        self.timing_stats = {
            'voxel_encoder': 0.0,
            'middle_encoder': 0.0,
            'backbone': 0.0,
            'neck': 0.0,
            'full': 0.0,
            'count': 0
        }
        torch.set_float32_matmul_precision('high')
        if self.with_img_backbone:
            self.test_img_retained_layers = torch.tensor(test_img_retained_layers) if test_img_retained_layers is not None else torch.ones(self.img_backbone.total_depth)
        if self.with_pts_backbone:
            self.test_lidar_retained_layers = torch.tensor(test_lidar_retained_layers) if test_lidar_retained_layers is not None else torch.ones(len(self.pts_middle_encoder.block_list) * 4)

    def init_weights(self):
        """Initialize model weights."""
        super(CmtDetector, self).init_weights()

    def extract_img_feat(self, img, img_metas, controller_selected_layers=None, drop_all_img_layers=False, losses=None, noise_embed=None, lidar_alloc=None):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
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

            retained_layers=None
            # If we specify the controller layers, then we use then immediately
            if controller_selected_layers is not None:
                retained_layers = controller_selected_layers
            # LAYERDROPPING LOGIC: 0 indicates a layer is DROPPED, 1 means the layer is RETAINED
            elif self.training:
                retained_layers = (torch.rand(self.img_backbone.total_depth) > self.layerdrop_rate).int()
                if drop_all_img_layers: # full modal layerdrop
                    retained_layers = torch.zeros_like(retained_layers)
                    # retained_layers[0] = 1
            # Accept a tensor of specified layers to evaluate during inference
            else:
                retained_layers = self.test_img_retained_layers

            # We want to enforce that the shape is always 2 for identical behavior between train and test
            if len(retained_layers.shape) == 1:
                retained_layers = torch.unsqueeze(retained_layers, dim=0)

            # If controller is training, we still compute the layer but multiply by 0 for gradient prop
            controller_training = self.controller is not None and self.controller.training
            img_feats = self.img_backbone(img.float(), retained_layer_list=retained_layers, controller_training=controller_training, early_exit_camera=self.early_exit_camera, losses=losses,
                                          noise_embed=noise_embed, 
                                            lidar_alloc=lidar_alloc)
 
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return [None] # Return none for img_feat if no img_backbone or img input
        
        # Pass through additional neck (usually a feature pyramid network that merges multiscale features)
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        return img_feats

    
    def extract_pts_feat(self, voxel_features, coors, controller_layers=None, drop_all_lidar_layers=False, losses=None, noise_embed=None, camera_alloc=None):
        if self.with_pts_backbone:
            # Keep operations in float32 to preseve voxelization accuracy
            with torch.autocast('cuda', enabled=False):
                batch_size = coors[-1, 0] + 1
                retained_layers = None
                
                # We have four layers per BasicBlock (hard coded by FlatFormer).
                if controller_layers is not None:
                    retained_layers = controller_layers
                elif self.training:
                    retained_layers = (torch.rand(len(self.pts_middle_encoder.block_list) * 4) > self.layerdrop_rate).int()
                    if drop_all_lidar_layers:
                        retained_layers = torch.zeros_like(retained_layers)
                        # retained_layers[0] = 1
                else:
                    retained_layers = self.test_lidar_retained_layers

                # We want to enforce that the shape is always 2 for identical behavior between train and test
                if len(retained_layers.shape) == 1:
                    retained_layers = torch.unsqueeze(retained_layers, dim=0)

                controller_training = self.controller is not None and self.controller.training
                # Executes FlatFormer
                x = self.pts_middle_encoder(voxel_features, coors, batch_size, 
                                            retained_layer_list=retained_layers, 
                                            controller_training=controller_training, 
                                            early_exit_lidar=self.early_exit_lidar, 
                                            losses=losses, 
                                            noise_embed=noise_embed, 
                                            camera_alloc=camera_alloc)
                
                # Pts Backbone
                x = self.pts_backbone(x)
                if self.with_pts_neck:
                    x = self.pts_neck(x)
                
                # The calling functione expects a list
                if not isinstance(x, list):
                    x = [x]
                return x
        return [None]
    
    def manual_scatter(self, voxel_features, coors, batch_size, grid_shape, feat_channels):
        H, W = grid_shape
        C = feat_channels
        canvas = torch.zeros((batch_size, C, H, W), device=voxel_features.device, dtype=voxel_features.dtype)
        canvas[coors[:, 0], :, coors[:, 2], coors[:, 3]] = voxel_features
        return canvas

    # This is the main feature extract function that calls the subfunctions for pts and images
    def extract_feat(self, batch_inputs_dict,
                     batch_input_metas, losses=None):
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
        message_hub = MessageHub.get_current_instance()
        current_epoch = message_hub.get_info('epoch')

        voxel_dict = batch_inputs_dict.get('voxels', None)
        imgs = batch_inputs_dict.get('imgs', None)

        voxels = voxel_dict['voxels']
        coors = voxel_dict['coors']
        with torch.autocast('cuda', enabled=False):
            voxel_features, coors = self.pts_voxel_encoder(voxels, coors)
        
        if 'corruption_info' in batch_input_metas[0]:
            gt_corruption_labels = torch.tensor([label_to_idx[m['corruption_info']['lidar_corruption']] for m in batch_input_metas])

        drop_all_lidar_layers, drop_all_img_layers = False, False
        modal_mask = None # This will potentially be used to zero out the LiDAR features
    
        if self.enable_modal_mask and self.training:
            # we only do this when training multimodal from lidar+img checkpoints
            if current_epoch <= 8 and self.enable_lidar_masking:
                modal_mask = (torch.rand(imgs.shape[0]) > 0.5).int()
                modal_mask = modal_mask.to(imgs.device)
            # We do this in two conditions: normal layerdrop training past epoch 8, and corruption fine-tuning always
            else:
                # Modal mask by removing lidar with 30% probability during training, do not do this during inference
                if torch.rand(1).item() < 0.3:
                    drop_all_lidar_layers = True
                elif torch.rand(1).item() < 0.2:
                    drop_all_img_layers = True
               
        
        if self.controller is not None:
            flatformer_layers = len(self.pts_middle_encoder.block_list) * 4
            layer_allocations, predicted_categories, noise_embed = self.controller(voxel_features, coors, imgs, flatformer_layers)
            
            retained_layers_lidar = layer_allocations[:, :flatformer_layers]
            retained_layers_img = layer_allocations[:, flatformer_layers:]

            camera_alloc = None
            lidar_alloc = None
            if self.early_exit_camera is not None:
                lidar_alloc = torch.sum(retained_layers_lidar.detach(), dim=-1)
            if self.early_exit_lidar is not None:
                camera_alloc = torch.sum(retained_layers_img.detach(), dim=-1)

            pts_feats = self.extract_pts_feat(
                voxel_features,
                coors,
                controller_layers=retained_layers_lidar,
                losses=losses,
                noise_embed=noise_embed,
                camera_alloc=camera_alloc
            )
           
            img_feats = self.extract_img_feat(
                imgs, 
                batch_input_metas, 
                controller_selected_layers=retained_layers_img,
                losses=losses,
                noise_embed=noise_embed,
                lidar_alloc=lidar_alloc
            )
            # Add the cross_entropy corruption prediction loss
            if losses is not None and self.controller.training:
                losses['noise_pred_loss'] = nn.functional.cross_entropy(predicted_categories, gt_corruption_labels.cuda())
            if get_dist_info()[0] == 0:
                print("Gt_corruption_labels", gt_corruption_labels[0])
                print('Predicted Noise', predicted_categories[0])
        else: # If we are not doing controller training
            pts_feats = self.extract_pts_feat(
                voxel_features=voxel_features,
                coors=coors,
                drop_all_lidar_layers=drop_all_lidar_layers)
            img_feats = self.extract_img_feat(imgs, batch_input_metas, drop_all_img_layers=drop_all_img_layers)
            # We drop out lidar during the first training of multimodal network to ensure it learns from img only
            if modal_mask is not None:
                pts_feats = [item * modal_mask[:, None, None, None] for item in pts_feats]
        
        return (img_feats, pts_feats)

    # Performs the DETR transformer prediction into Bboxes
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

    # This is the "forward" of train
    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        message_hub = MessageHub.get_current_instance()
        current_step = message_hub.get_info('iter')
        # Set the controller to eval mode
        if self.early_exit_lidar is not None or self.early_exit_camera is not None:
            self.controller.eval()
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        losses = dict()
        img_feats, pts_feats = self.extract_feat(batch_inputs_dict, batch_input_metas, losses)
        # First 200 steps of controller training are just to train the corruption perception
        # Set to 50 bc we are doing multigpu training
        if self.controller is not None and self.controller.training and current_step < 50 and not self.enable_pruning: 
            if get_dist_info()[0] == 0:
                print(current_step)
            if 'early_exit_loss' in losses:
                del losses['early_exit_loss']
            return losses
        
        # By default, we do not enable pruning
        if self.enable_pruning:
            points_mask = self.lidar_pruner(pts_feats[0])
            img_mask = self.img_pruner(img_feats[0])

            points_mask_truncated = (torch.clone(points_mask) + self.mask_bias_value).round()
            img_mask_truncated = (torch.clone(img_mask) + self.mask_bias_value).round()

            points_mask = points_mask + (points_mask_truncated - points_mask).detach()
            img_mask = img_mask + (img_mask_truncated - img_mask).detach()
            # Both of these are wrapped in a one elem list
            pts_feats[0] = points_mask * pts_feats[0]
            img_feats = [img_mask * img_feats[0]]

        gt_bboxes_3d = [sample.gt_instances_3d.bboxes_3d for sample in batch_data_samples]
        # Extract GT Labels (List[Tensor])
        gt_labels_3d = [sample.gt_instances_3d.labels_3d for sample in batch_data_samples]
        # Extract Ignore Masks (Optional - usually None for standard 3D detection)
        # Old code usually defaults this to None if not present
        gt_bboxes_ignore = None
        
        # Passing in the masks into forward_pts_train means that are we using hard pruning
        if self.use_hard_pruning:
            losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, batch_input_metas,
                                                gt_bboxes_ignore, pts_mask=points_mask, img_mask=img_mask)
        else:
            losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, batch_input_metas,
                                                gt_bboxes_ignore)
        # Add additional loss for the mask
        if self.enable_pruning:
            #L1 sparsity, masks already binary
            # We see a large-ish drop in performance, change to divide by 3 to encourage keeping accuracy
            losses_pts['img_mask_loss'] = torch.mean(img_mask) / 1.5
            losses_pts['pts_mask_loss'] = torch.mean(points_mask) / 1.5
        losses.update(losses_pts)
        return losses

    # This is the forward of test
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
        if self.timing_stats['count'] == 1:
            if self.early_exit_lidar is not None:
                self.early_exit_lidar = torch.compile(self.early_exit_lidar, mode="reduce-overhead").eval()
            if self.early_exit_camera is not None:
                self.early_exit_camera = torch.compile(self.early_exit_camera, mode="reduce-overhead").eval()

        # torch.cuda.synchronize()
        # start = time.perf_counter()

        batch_input_metas = [item.metainfo for item in batch_data_samples]

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # start.record()
        img_feats, pts_feats = self.extract_feat(batch_inputs_dict,
                                                 batch_input_metas)
        if self.enable_pruning:
            points_mask = (self.lidar_pruner(pts_feats[0]) + self.mask_bias_value).round()
            img_mask = (self.img_pruner(img_feats[0]) + self.mask_bias_value).round()
            # print("Points_Mask", torch.mean(points_mask))
            # print("Img_Mask", torch.mean(img_mask))
            pts_feats[0] = points_mask * pts_feats[0]
            img_feats = [img_mask * img_feats[0]]

        if self.use_hard_pruning:
            outs = self.pts_bbox_head(pts_feats, img_feats, batch_input_metas, img_mask=img_mask, pts_mask=points_mask)
        else:
            outs = self.pts_bbox_head(pts_feats, img_feats, batch_input_metas)

        bbox_list = self.pts_bbox_head.get_bboxes(outs, batch_input_metas, rescale=False)
        # torch.cuda.synchronize()
        # elapsed = time.perf_counter() - start
        # print("Elapsed:", elapsed)
        self.timing_stats['count'] += 1
        
        # if self.timing_stats['count'] == 4449:
        #     sys.exit(0)

        return self.add_pred_to_datasample(batch_data_samples, bbox_list)



    def _debug_token_pruning(self, points_mask, img_mask, batch_inputs_dict):
        self.num_lidar_tokens += torch.sum(points_mask)
        self.num_image_tokens += torch.sum(img_mask)

        # img_mask is shape (B, 6, 20, 50)
        print('------------------------------------------------')
        print(f'Total Lidar Tokens: {self.num_lidar_tokens / self.sample_count}')
        print(f'Total Image Tokens: {self.num_image_tokens / self.sample_count}\n')
        print(f'Current Lidar Tokens: {torch.sum(points_mask)}')
        print(f'Current Image Tokens: {torch.sum(img_mask)}')
        print('------------------------------------------------')
        if torch.sum(img_mask) > 1000:
            mean=torch.tensor([[103.530, 116.280, 123.675]]).cuda()
            std=torch.tensor([[57.375, 57.120, 58.395]]).cuda()
            sample_imgs = batch_inputs_dict['imgs'] * std[..., None, None] + mean[..., None, None]
            sample_imgs = sample_imgs[:, [2, 1, 0]] / 255 # Shuffle the channels since we load in BGR I think
            save_masked_grid(sample_imgs, img_mask)
        

        torchvision.utils.save_image(batch_inputs_dict['imgs'][:, [2, 1, 0]], f'cmt_images/test_{self.sample_count}_{torch.sum(points_mask)}_{torch.sum(img_mask)}.png', nrow=3)
    

# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

# import time
# import torch
# import torch.nn as nn

# import torch.nn.functional as F
# import numpy as np

# from mmdet3d.structures import bbox3d2result
# from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

# from .cmt_utils.grid_mask import GridMask
# from mmdet3d.registry import MODELS
# from torchvision.utils import save_image, make_grid

# from mmdet3d.models.voxel_encoders import DynamicVFE
# from .cmt_utils import DynamicVFE_Linear
# from mmdet3d.structures import bbox3d2result
# from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
# from ..UniM2AE.sst_models import DynamicVFE_New

# from .cmt_utils.grid_mask import GridMask
# from mmdet3d.registry import MODELS
# import sys 
# from mmengine.logging import MessageHub
# import torchvision

# # Define timing events
# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)

# file_handle = open('temp.txt', 'w')

# def save_masked_grid(imgs, msk, output_path="masked_grid.png", alpha=0.5):
#     """
#     Saves a grid of the 6 images from the first batch item, masking out 
#     patches where the mask value is 0.
    
#     Args:
#         images: Tensor of shape (B, 6, 3, H, W). Assumed to be normalized in [0, 1].
#         masks: Tensor of shape (B, 6, 20, 50). Binary mask (0 or 1).
#         output_path: Filename for the saved image.
#     """
#     # 1. Select the first item in the batch
#     # Shapes: imgs -> (6, 3, H, W), msk -> (6, 20, 50)
#     # imgs = images[0].clone()
#     # msk = masks[0].clone()

#     # # 2. Resize the mask to match the image resolution
#     # # We need to add a channel dim for interpolation: (6, 20, 50) -> (6, 1, 20, 50)
#     # msk = msk.unsqueeze(1).float()
    
#     # Use 'nearest' interpolation to keep the patch blocks sharp and distinct
#     # Output shape: (6, 1, H, W)
#     H, W = imgs.shape[-2:]
#     msk_resized = F.interpolate(msk, size=(H, W), mode='nearest')


#     overlay_color = torch.tensor([1.0, 0.0, 0.0], device=imgs.device).view(1, 3, 1, 1)

#     # 4. Apply Translucent Blending
#     # Logic: The final pixel is a weighted sum of the original pixel and the 
#     # overlay color, based on the alpha value.
#     # Formula: Blended = Original * (1 - alpha) + Overlay * alpha
    
#     # Create the full overlay image (solid red everywhere)
#     overlay_layer = torch.ones_like(imgs) * overlay_color
    
#     # Calculate the blended version of the whole image
#     blended_imgs = imgs * (1 - alpha) + overlay_layer * alpha

#     # Final composition:
#     # Where mask is 1, use original image.
#     # Where mask is 0, use the blended image.
#     final_imgs = imgs * msk_resized + blended_imgs * (1 - msk_resized)

#     # 4. Create a grid
#     # nrow=3 creates a standard 2x3 layout for the 6 images
#     grid = make_grid(final_imgs, nrow=3, padding=2, normalize=False)

#     # 5. Save the image
#     save_image(grid, output_path)



# '''
# Top level CMT model

# '''

# @MODELS.register_module()
# class CmtDetector(MVXTwoStageDetector):

#     def __init__(self,
#                  use_grid_mask=False,
#                  enable_pruning=False,
#                  use_hard_pruning=False,
#                  enable_modal_mask=False,
#                  layerdrop_rate = 0.0,
#                  test_img_retained_layers = None,
#                  test_lidar_retained_layers = None,
#                  controller = None,
#                  **kwargs):
#         super(CmtDetector, self).__init__(**kwargs)
#         self.use_grid_mask = use_grid_mask
#         self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
#         # Modal mask and layerdrop are enabled together during training
#         self.enable_modal_mask = enable_modal_mask
#         self.layerdrop_rate = layerdrop_rate

#         self.enable_pruning = enable_pruning
#         self.use_hard_pruning = use_hard_pruning
#         assert (enable_pruning or not use_hard_pruning), "Hard pruning should only be enabled when we are pruning"

#         # If enable pruning, define image and lidar mask generators
#         if self.enable_pruning:
#             self.img_pruner = torch.nn.Sequential(
#                 nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, padding=1),
#                 nn.InstanceNorm2d(num_features=16, affine=True),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
#                 nn.Sigmoid()
#             )
#             self.lidar_pruner = torch.nn.Sequential(
#                 nn.Conv2d(in_channels=512, out_channels=16, kernel_size=3, padding=1),
#                 nn.InstanceNorm2d(num_features=16, affine=True),
#                 nn.ReLU(),
#                 nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
#                 nn.Sigmoid()
#             )

#             self.mask_bias_value=0.35
#             self.num_lidar_tokens = 0
#             self.num_image_tokens = 0

#         if controller:
#             self.controller = MODELS.build(controller)
#         else:
#             self.controller = None

#         self.timing_stats = {
#             'voxel_encoder': 0.0,
#             'middle_encoder': 0.0,
#             'backbone': 0.0,
#             'neck': 0.0,
#             'img_backbone': 0.0,
#             'img_neck': 0.0,
#             'full': 0.0,
#             'count': 0
#         }

#         if self.with_img_backbone:
#             self.test_img_retained_layers = torch.tensor(test_img_retained_layers) if test_img_retained_layers is not None else torch.ones(self.img_backbone.total_depth)
#         if self.with_pts_backbone:
#             self.test_lidar_retained_layers = torch.tensor(test_lidar_retained_layers) if test_lidar_retained_layers is not None else torch.ones(len(self.pts_middle_encoder.block_list) * 4)

#     def init_weights(self):
#         """Initialize model weights."""
#         super(CmtDetector, self).init_weights()

#     def extract_img_feat(self, img, img_metas, controller_selected_layers=None, drop_all_img_layers=False):
#         """Extract features of images."""
#         if self.with_img_backbone and img is not None:
#             input_shape = img.shape[-2:]
#             for img_meta in img_metas:
#                 img_meta.update(input_shape=input_shape)

#             if img.dim() == 5 and img.size(0) == 1: # Batch size 1 we squeeze and remove the batch dimension since img backbone handles 4 dimensional input
#                 img.squeeze_(0)
#             # Compress the batch dimension and the N dimension (6 cameras per sample)
#             elif img.dim() == 5 and img.size(0) > 1:
#                 B, N, C, H, W = img.size()
#                 img = img.view(B * N, C, H, W)
#             if self.use_grid_mask:
#                 img = self.grid_mask(img)

#             retained_layers=None
#             # If we specify the controller layers, then we use then immediately
#             if controller_selected_layers is not None:
#                 retained_layers = controller_selected_layers
#             # LAYERDROPPING LOGIC: 0 indicates a layer is DROPPED, 1 means the layer is RETAINED
#             elif self.training:
#                 retained_layers = (torch.rand(self.img_backbone.total_depth) > self.layerdrop_rate).int()
#                 if drop_all_img_layers: # full modal layerdrop
#                     retained_layers = torch.zeros_like(retained_layers)
#             # Accept a tensor of specified layers to evaluate during inference
#             else:
#                 retained_layers = self.test_img_retained_layers

#             # We want to enforce that the shape is always 2 for identical behavior between train and test
#             if len(retained_layers.shape) == 1:
#                 retained_layers = torch.unsqueeze(retained_layers, dim=0)

#             # If controller is training, we still compute the layer but multiply by 0 for gradient prop
#             controller_training = self.controller is not None and self.training
            
#             # --- Timing img_backbone ---
#             start_event.record()
#             img_feats = self.img_backbone(img.float(), retained_layer_list=retained_layers, controller_training=controller_training)
#             end_event.record()
#             torch.cuda.synchronize()
#             self.timing_stats['img_backbone'] = start_event.elapsed_time(end_event)
#             # ---------------------------
 
#             if isinstance(img_feats, dict):
#                 img_feats = list(img_feats.values())
#         else:
#             return None # Return none for img_feat if no img_backbone or img input
        
#         # Pass through additional neck (usually a feature pyramid network that merges multiscale features)
#         if self.with_img_neck:
#             # --- Timing img_neck ---
#             start_event.record()
#             img_feats = self.img_neck(img_feats)
#             end_event.record()
#             torch.cuda.synchronize()
#             self.timing_stats['img_neck'] = start_event.elapsed_time(end_event)
#             # -----------------------

#         return img_feats

    
#     def extract_pts_feat(self, voxel_dict, controller_layers=None, drop_all_lidar_layers=False):
#         if self.with_pts_backbone:
#             # Keep operations in float32 to preseve voxelization accuracy
#             with torch.autocast('cuda', enabled=False):
#                 voxels = voxel_dict['voxels']
#                 coors = voxel_dict['coors']
#                 # No longer accounting for different voxel encoder options

#                 # --- Timing pts_voxel_encoder ---
#                 start_event.record()
#                 voxel_features, coors = self.pts_voxel_encoder(voxels, coors)
#                 end_event.record()
#                 torch.cuda.synchronize()
#                 self.timing_stats['pts_voxel_encoder'] = start_event.elapsed_time(end_event)
#                 # --------------------------------
    
#                 batch_size = coors[-1, 0] + 1
#                 retained_layers = None
                
#                 # We have four layers per BasicBlock (hard coded by FlatFormer).
#                 if controller_layers is not None:
#                     retained_layers = controller_layers
#                 elif self.training:
#                     retained_layers = (torch.rand(len(self.pts_middle_encoder.block_list) * 4) > self.layerdrop_rate).int()
#                     if drop_all_lidar_layers:
#                         retained_layers = torch.zeros_like(retained_layers)
#                 else:
#                     retained_layers = self.test_lidar_retained_layers

#                 # We want to enforce that the shape is always 2 for identical behavior between train and test
#                 if len(retained_layers.shape) == 1:
#                     retained_layers = torch.unsqueeze(retained_layers, dim=0)

#                 controller_training = self.controller is not None and self.training
#                 # Executes FlatFormer
#                 # --- Timing pts_middle_encoder ---
#                 start_event.record()
#                 x = self.pts_middle_encoder(voxel_features, coors, batch_size, retained_layer_list=retained_layers, controller_training=controller_training)
#                 end_event.record()
#                 torch.cuda.synchronize()
#                 self.timing_stats['pts_middle_encoder'] = start_event.elapsed_time(end_event)
#                 # ---------------------------------
                
#                 # Pts Backbone
#                 # --- Timing pts_backbone ---
#                 start_event.record()
#                 x = self.pts_backbone(x)
#                 end_event.record()
#                 torch.cuda.synchronize()
#                 self.timing_stats['pts_backbone'] = start_event.elapsed_time(end_event)
#                 # ---------------------------

#                 if self.with_pts_neck:
#                     # --- Timing pts_neck ---
#                     start_event.record()
#                     x = self.pts_neck(x)
#                     end_event.record()
#                     torch.cuda.synchronize()
#                     self.timing_stats['pts_neck'] = start_event.elapsed_time(end_event)
#                     # -----------------------
                
#                 # The calling functione expects a list
#                 if not isinstance(x, list):
#                     x = [x]
#                 return x
#         return [None]
    

#     # This is the main feature extract function that calls the subfunctions for pts and images
#     def extract_feat(self, batch_inputs_dict,
#                      batch_input_metas, losses=None):
#         """Extract features from images and points.

#         Args:
#             batch_inputs_dict (dict): Dict of batch inputs. It
#                 contains

#                 - points (List[tensor]):  Point cloud of multiple inputs.
#                 - imgs (tensor): Image tensor with shape (B, C, H, W).
#             batch_input_metas (list[dict]): Meta information of multiple inputs
#                 in a batch.

#         Returns:
#              tuple: Two elements in tuple arrange as
#              image features and point cloud features.
#         """
#         message_hub = MessageHub.get_current_instance()
#         current_epoch = message_hub.get_info('epoch')

#         voxel_dict = batch_inputs_dict.get('voxels', None)
#         imgs = batch_inputs_dict.get('imgs', None)

#         points = batch_inputs_dict.get('points', None)
        
#         if 'corruption_info' in batch_input_metas[0]:
#             lidar_sev = torch.tensor([m['corruption_info']['lidar_severity'] > 0 for m in batch_input_metas])
#             cam_sev = torch.tensor([m['corruption_info']['camera_severity'] > 0 for m in batch_input_metas])

#             # 0: Both Clean, 1: Camera Only, 2: LiDAR Only, 3: Both Corrupted 
#             gt_corruption_labels = (lidar_sev.long() * 2) + cam_sev.long()

#         drop_all_lidar_layers, drop_all_img_layers = False, False
#         modal_mask = None # This will potentially be used to zero out the LiDAR features
    
#         if self.enable_modal_mask and self.training:
#             # We do this in two conditions: normal layerdrop training past epoch 8, and corruption fine-tuning always
#             if current_epoch > 8 or self.layerdrop_rate == 0.0 :
#                 # Modal mask by removing lidar with 30% probability during training, do not do this during inference
#                 if torch.rand(1).item() < 0.3:
#                     drop_all_lidar_layers = True
#                 elif torch.rand(1).item() < 0.2:
#                     drop_all_img_layers = True
#             # This is needed when we train the multimodal modal from LiDAR only weights
#             # Even if we drop out the FlatFormer layers, the backbone still functions
#             else:
#                 modal_mask = (torch.rand(pts_feats[0].shape[0]) > 0.5).int()
#                 modal_mask = modal_mask.to(pts_feats[0].device)
        
#         if self.controller is not None:
#             flatformer_layers = len(self.pts_middle_encoder.block_list) * 4
#             layer_allocations, predicted_categories = self.controller(voxel_dict, imgs, flatformer_layers)
#             retained_layers_lidar = layer_allocations[:, :flatformer_layers]
#             retained_layers_img = layer_allocations[:, flatformer_layers:]
        
#             pts_feats = self.extract_pts_feat(
#                 voxel_dict,
#                 controller_layers=retained_layers_lidar,
#             )
           
#             img_feats = self.extract_img_feat(
#                 imgs, 
#                 batch_input_metas, 
#                 controller_selected_layers=retained_layers_img
#             )
#             # Add the cross_entropy corruption prediction loss
#             if losses is not None:
#                 # print('Predicted Noise', predicted_categories[0])
#                 # print("Gt_corruption_labels", gt_corruption_labels[0])
#                 losses['noise_pred_loss'] = nn.functional.cross_entropy(predicted_categories, gt_corruption_labels.cuda())
#         else: # If we are not doing controller training
#             pts_feats = self.extract_pts_feat(
#                 voxel_dict,
#                 drop_all_lidar_layers=drop_all_lidar_layers)
#             img_feats = self.extract_img_feat(imgs, batch_input_metas, drop_all_img_layers=drop_all_img_layers)
#             # We drop out lidar during the first training of multimodal network to ensure it learns from img only
#             if modal_mask is not None:
#                 pts_feats = [item * modal_mask[:, None, None, None] for item in pts_feats]
        
#         return (img_feats, pts_feats)

#     # Performs the DETR transformer prediction into Bboxes
#     def forward_pts_train(self,
#                           pts_feats,
#                           img_feats,
#                           gt_bboxes_3d,
#                           gt_labels_3d,
#                           img_metas,
#                           gt_bboxes_ignore=None,
#                           img_mask = None,
#                           pts_mask = None):
#         """Forward function for point cloud branch.

#         Args:
#             pts_feats (list[torch.Tensor]): Features of point cloud branch
#             gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
#                 boxes for each sample.
#             gt_labels_3d (list[torch.Tensor]): Ground truth labels for
#                 boxes of each sampole
#             img_metas (list[dict]): Meta information of samples.
#             gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
#                 boxes to be ignored. Defaults to None. 

#         Returns:
#             dict: Losses of each branch.
#         """
#         if pts_feats is None:
#             pts_feats = [None]
#         if img_feats is None:
#             img_feats = [None]
#         with torch.autocast('cuda', enabled=False):
#             outs = self.pts_bbox_head(pts_feats, img_feats, img_metas, pts_mask=pts_mask, img_mask=img_mask)
#             loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
#             losses = self.pts_bbox_head.loss(*loss_inputs)
#         return losses

#     # This is the "forward" of train
#     def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
#         message_hub = MessageHub.get_current_instance()
#         current_step = message_hub.get_info('iter')

#         batch_input_metas = [item.metainfo for item in batch_data_samples]
#         losses = dict()
#         img_feats, pts_feats = self.extract_feat(batch_inputs_dict, batch_input_metas, losses)
#         # First 500 epochs of controller training are just to train the corruption perception
#         if self.controller is not None and current_step < 500 and not self.enable_pruning:
#             print(current_step)
#             return losses
        
#         # By default, we do not enable pruning
#         if self.enable_pruning:
#             points_mask = self.lidar_pruner(pts_feats[0])
#             img_mask = self.img_pruner(img_feats[0])

#             points_mask_truncated = (torch.clone(points_mask) + self.mask_bias_value).round()
#             img_mask_truncated = (torch.clone(img_mask) + self.mask_bias_value).round()

#             points_mask = points_mask + (points_mask_truncated - points_mask).detach()
#             img_mask = img_mask + (img_mask_truncated - img_mask).detach()
            
#             # Both of these are wrapped in a one elem list
#             pts_feats[0] = points_mask * pts_feats[0]
#             img_feats = [img_mask * img_feats[0]]

#         gt_bboxes_3d = [sample.gt_instances_3d.bboxes_3d for sample in batch_data_samples]
#         # Extract GT Labels (List[Tensor])
#         gt_labels_3d = [sample.gt_instances_3d.labels_3d for sample in batch_data_samples]
#         # Extract Ignore Masks (Optional - usually None for standard 3D detection)
#         # Old code usually defaults this to None if not present
#         gt_bboxes_ignore = None
        
#         # Passing in the masks into forward_pts_train means that are we using hard pruning
#         if self.use_hard_pruning:
#             losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
#                                                 gt_labels_3d, batch_input_metas,
#                                                 gt_bboxes_ignore, pts_mask=points_mask, img_mask=img_mask)
#         else:
#             losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
#                                                 gt_labels_3d, batch_input_metas,
#                                                 gt_bboxes_ignore)
#         # Add additional loss for the mask
#         if self.enable_pruning:
#             #L1 sparsity, masks already binary
#             losses_pts['img_mask_loss'] = torch.mean(img_mask)
#             losses_pts['pts_mask_loss'] = torch.mean(points_mask)
#         losses.update(losses_pts)
#         return losses

#     # This is the forward of test
#     def predict(self, batch_inputs_dict,
#                 batch_data_samples,
#                 **kwargs):
#         """Forward of testing.

#         Args:
#             batch_inputs_dict (dict): The model input dict which include
#                 'points' keys.

#                 - points (list[torch.Tensor]): Point cloud of each sample.
#             batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
#                 Samples. It usually includes information such as
#                 `gt_instance_3d`.

#         Returns:
#             list[:obj:`Det3DDataSample`]: Detection results of the
#             input sample. Each Det3DDataSample usually contain
#             'pred_instances_3d'. And the ``pred_instances_3d`` usually
#             contains following keys.

#             - scores_3d (Tensor): Classification scores, has a shape
#                 (num_instances, )
#             - labels_3d (Tensor): Labels of bboxes, has a shape
#                 (num_instances, ).
#             - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
#                 contains a tensor with shape (num_instances, 7).
#         """

#         batch_input_metas = [item.metainfo for item in batch_data_samples]

#         # start = torch.cuda.Event(enable_timing=True)
#         # end = torch.cuda.Event(enable_timing=True)

#         # start.record()
#         # Reset timings just in case
#         for k in self.timing_stats:
#             if k != 'count':
#                 self.timing_stats[k] = 0.0

#         img_feats, pts_feats = self.extract_feat(batch_inputs_dict,
#                                                  batch_input_metas)
#         if self.enable_pruning:
#             points_mask = (self.lidar_pruner(pts_feats[0]) + self.mask_bias_value).round()
#             img_mask = (self.img_pruner(img_feats[0]) + self.mask_bias_value).round()
#             pts_feats[0] = points_mask * pts_feats[0]
#             img_feats = [img_mask * img_feats[0]]

#         # --- Timing pts_bbox_head ---
#         start_event.record()
#         if self.use_hard_pruning:
#             outs = self.pts_bbox_head(pts_feats, img_feats, batch_input_metas, img_mask=img_mask, pts_mask=points_mask)
#         else:
#             outs = self.pts_bbox_head(pts_feats, img_feats, batch_input_metas)
#         end_event.record()
#         torch.cuda.synchronize()
#         self.timing_stats['pts_bbox_head'] = start_event.elapsed_time(end_event)
#         # ----------------------------
        

#         bbox_list = self.pts_bbox_head.get_bboxes(outs, batch_input_metas, rescale=False)
#         # end.record()
#         # torch.cuda.synchronize()
#         # with open('full_model_latency.txt', 'a') as handle:
#         #     print("Elapsed:", start.elapsed_time(end), file=handle)
        
#         # Write component timings to file
#         with open('component_timings.txt', 'a') as f:
#             f.write(f"img_backbone: {self.timing_stats.get('img_backbone', 0):.2f}, "
#                     f"img_neck: {self.timing_stats.get('img_neck', 0):.2f}, "
#                     f"pts_voxel_encoder: {self.timing_stats.get('pts_voxel_encoder', 0):.2f}, "
#                     f"pts_middle_encoder: {self.timing_stats.get('pts_middle_encoder', 0):.2f}, "
#                     f"pts_backbone: {self.timing_stats.get('pts_backbone', 0):.2f}, "
#                     f"pts_neck: {self.timing_stats.get('pts_neck', 0):.2f}, "
#                     f"pts_bbox_head: {self.timing_stats.get('pts_bbox_head', 0):.2f}\n")

#         return self.add_pred_to_datasample(batch_data_samples, bbox_list)



#     def _debug_token_pruning(self, points_mask, img_mask, batch_inputs_dict):
#         self.num_lidar_tokens += torch.sum(points_mask)
#         self.num_image_tokens += torch.sum(img_mask)

#         # img_mask is shape (B, 6, 20, 50)
#         print('------------------------------------------------')
#         print(f'Total Lidar Tokens: {self.num_lidar_tokens / self.sample_count}')
#         print(f'Total Image Tokens: {self.num_image_tokens / self.sample_count}\n')
#         print(f'Current Lidar Tokens: {torch.sum(points_mask)}')
#         print(f'Current Image Tokens: {torch.sum(img_mask)}')
#         print('------------------------------------------------')
#         if torch.sum(img_mask) > 1000:
#             mean=torch.tensor([[103.530, 116.280, 123.675]]).cuda()
#             std=torch.tensor([[57.375, 57.120, 58.395]]).cuda()
#             sample_imgs = batch_inputs_dict['imgs'] * std[..., None, None] + mean[..., None, None]
#             sample_imgs = sample_imgs[:, [2, 1, 0]] / 255 # Shuffle the channels since we load in BGR I think
#             save_masked_grid(sample_imgs, img_mask)
        

#         torchvision.utils.save_image(batch_inputs_dict['imgs'][:, [2, 1, 0]], f'cmt_images/test_{self.sample_count}_{torch.sum(points_mask)}_{torch.sum(img_mask)}.png', nrow=3)