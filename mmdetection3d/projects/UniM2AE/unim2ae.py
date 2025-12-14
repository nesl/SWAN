import torch.nn as nn
import torch
from mmdet3d.registry import MODELS
from mmdet3d.models.detectors.dynamic_voxelnet import DynamicVoxelNet
import numpy as np
import matplotlib.pyplot as plt
from mmcv.cnn import build_conv_layer, build_norm_layer
import time
from mmengine.dist import get_rank

def show_image(image, title=''):
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def vis_image(ori_img, pred_img, mask, model, out_dir, sample_idx):
    from pathlib import Path
    Path(f'{out_dir}/cam').mkdir(parents=True, exist_ok=True)
    
    ori_img = model.camera_decoder.patchify(ori_img)
    mean = ori_img.mean(dim=-1, keepdim=True)
    var = ori_img.var(dim=-1, keepdim=True)
    ori_img = model.camera_decoder.unpatchify(ori_img)
    x = torch.einsum('nchw->nhwc', ori_img).detach().cpu()

    pred_img = pred_img * (var + 1.e-6)**.5 + mean
    y = model.camera_decoder.unpatchify(pred_img)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.camera_decoder.final_patch_size**2 *3)
    mask = model.camera_decoder.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    im_masked = x * (1 - mask)
    im_paste = x * (1 - mask) + y * mask

    plt.rcParams['figure.figsize'] = [24, 12]
    B = min(x.shape[0], 6)
    for i in range(B):
        plt.subplot(B, 4, i*4+1)
        show_image(x[i], "original")
        plt.subplot(B, 4, i*4+2)
        show_image(im_masked[i], "masked")
        plt.subplot(B, 4, i*4+3)
        show_image(y[i], "reconstruction")
        plt.subplot(B, 4, i*4+4)
        show_image(im_paste[i], "reconstruction + visible")

    plt.savefig(f"{out_dir}/cam/{sample_idx}")
    plt.close()




@MODELS.register_module()
class UniM2AE(DynamicVoxelNet):
    def __init__(self, 
                 voxel_encoder, 
                 middle_encoder, 
                 backbone,
                 fusion_module,
                 camera_backbone,
                 camera_vtransform,
                 data_preprocessor,
                 camera_decoder=None,
                 neck=None, 
                 bbox_head=None, 
                 train_cfg=None, 
                 test_cfg=None, 
                 init_cfg=None, 
                 ):
        super(UniM2AE, self).__init__(
            voxel_encoder=voxel_encoder, 
            middle_encoder=middle_encoder, 
            backbone=backbone, 
            neck=neck, 
            bbox_head=bbox_head, 
            train_cfg=train_cfg, 
            test_cfg=test_cfg, 
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor
            )
        self.init_weights()

        self.camera_encoder = nn.ModuleDict(
            {
                "backbone": MODELS.build(camera_backbone),
                "vtransform": MODELS.build(camera_vtransform),
            }
        )
        self.relu = nn.ReLU(inplace=True)
        self.camera_decoder = MODELS.build(camera_decoder)
        
        self.fusion_module = MODELS.build(fusion_module)
        
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
        self.deblock_lidar = nn.Sequential(
                build_conv_layer(
                dict(type='Conv3d', bias=False),
                in_channels=128,
                out_channels=192,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            build_norm_layer(norm_cfg, 192)[1],
            nn.ReLU(inplace=True),
        )
        
        self.sample_idx = 0
        self.epoch = 0

    
    def extract_feat(self, batch_inputs_dict):
        with torch.cuda.amp.autocast(enabled=False):
            voxels, coors = batch_inputs_dict['voxels']['voxels'], batch_inputs_dict['voxels']['coors']
            batch_size = coors[-1, 0].item() + 1
            voxel_features, feature_coors, low_level_point_feature, indices = self.voxel_encoder(voxels, coors)
        x = self.middle_encoder(voxel_features, feature_coors, low_level_point_feature, indices, batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        lidar_x = x[1]
        lidar_volume_embed = self.deblock_lidar(lidar_x[0]['output'])
        return lidar_x, lidar_volume_embed
    
    @torch.no_grad()
    def test_pretrain(self,  batch_inputs_dict,
                batch_samples_dict,
                      out_dir, sample_idx=0):
        """Test function without augmentaiton."""
        batch_size = len(batch_inputs_dict['points'])
        vx, vy, vz = self.middle_encoder.sparse_shape

        batched_input_metas = [item.metainfo for item in batch_samples_dict]

        # Standard forward pass 
        lidar_x, lidar_volume_embed = self.extract_feat(batch_inputs_dict)

        imgs = batch_inputs_dict.get('imgs', None)
        imgs = torch.transpose(torch.transpose(imgs, 2, 3), 3, 4)
        B, N, C, H, W = imgs.size() # CARE SUS???
        imgs = imgs.view(B * N, C, H, W)

        camera_x, camera_mask, camera_ids_restore = self.camera_encoder["backbone"](imgs, camera_only=True)

        camera_volume_embed, camera_x = self.camera_encoder["vtransform"](
            camera_x, 
            (B, N, C, H, W),
            camera_ids_restore, 
            batched_input_metas
        )

        cam_proj_feat, lidar_proj_feat = self.fusion_module(
            [lidar_volume_embed, camera_volume_embed], 
            lidar_x,
            batched_input_metas,
        )
        cam_pred = self.relu(cam_proj_feat + camera_x.view(B*N, -1, H//32, W//32))
        cam_pred = cam_pred.flatten(2).permute(0, 2, 1)
        cam_pred = self.camera_decoder(cam_pred, camera_ids_restore)

        
        
        vis_image(imgs, cam_pred, camera_mask, self, out_dir, sample_idx=sample_idx)
        
        lidar_x[0]['output'] = lidar_proj_feat

        outs = self.bbox_head(lidar_x[0], show=True)
        pred_dict = outs[0] # unsure
        voxel_coors = pred_dict["voxel_coors"]
        masked_voxel_coors = pred_dict["masked_voxel_coors"]
        unmasked_voxel_coors = pred_dict["unmasked_voxel_coors"]

        occupied = None
        if "pred_occupied" in pred_dict:
            occupied = -torch.ones((batch_size, vx, vy), dtype=torch.long, device=pred_dict["pred_occupied"].device)
            index = (voxel_coors[:, 0], voxel_coors[:, 3], voxel_coors[:, 2])  # b ,x, y
            unmasked_index = (
                unmasked_voxel_coors[:, 0], unmasked_voxel_coors[:, 3], unmasked_voxel_coors[:, 2])
            gt_occupied = pred_dict["gt_occupied"].long()+1  # 1 -> real voxel, 2 -> fake voxel
            occupied[index] = 2 * gt_occupied  # 2 -> real voxel, 4 -> fake voxel
            occupied[unmasked_index] -= 2  # 0 -> unmasked voxels 2 -> masked voxel, 4 -> fake voxel
            occupied[index] += (torch.sigmoid(pred_dict["pred_occupied"]) + 0.5).long()
            # 0 -> unmasked voxel predicted as real,
            # 1 -> unmasked voxel predicted as fake,
            # 2 -> masked voxel predicted as real,
            # 3 -> masked voxel predicted as fake,
            # 4 -> fake voxel predicted as real,
            # 5 -> fake voxel predicted as fake

        gt_num_points = None
        diff_num_points = None
        if "pred_num_points_masked" in pred_dict:
            device = pred_dict["pred_num_points_masked"].device
            gt_num_points = torch.zeros((batch_size, vx, vy), dtype=torch.long, device=device)
            diff_num_points = torch.zeros((batch_size, vx, vy), dtype=torch.float, device=device)
            index = (masked_voxel_coors[:, 0], masked_voxel_coors[:, 3], masked_voxel_coors[:, 2])  # b ,x, y
            pred_num_points_masked = pred_dict["pred_num_points_masked"]
            gt_num_points_masked = pred_dict["gt_num_points_masked"]
            gt_num_points[index] = gt_num_points_masked.long()
            diff_num_points[index] = gt_num_points_masked.float()-pred_num_points_masked
        if "pred_num_points_unmasked" in pred_dict:
            device = pred_dict["pred_num_points_unmasked"].device
            gt_num_points = torch.zeros((batch_size, vx, vy), dtype=torch.long, device=device) if gt_num_points is None else gt_num_points
            diff_num_points = torch.zeros((batch_size, vx, vy), dtype=torch.float, device=device) if diff_num_points is None else diff_num_points
            index = (unmasked_voxel_coors[:, 0], unmasked_voxel_coors[:, 3], unmasked_voxel_coors[:, 2])  # b ,x, y
            pred_num_points_unmasked = pred_dict["pred_num_points_unmasked"]
            gt_num_points_unmasked = pred_dict["gt_num_points_unmasked"]
            gt_num_points[index] = gt_num_points_unmasked.long()
            diff_num_points[index] = gt_num_points_unmasked.float() - pred_num_points_unmasked

        points = []
        batch = []
        if "pred_points_masked" in pred_dict:
            pred_points_masked = pred_dict["pred_points_masked"].clone()  # M, num_chamfer_points, 3
            M, n, C = pred_points_masked.shape
            x_shift = (masked_voxel_coors[:, 3].type_as(pred_points_masked) * self.voxel_encoder.vx + self.voxel_encoder.x_offset)  # M
            y_shift = (masked_voxel_coors[:, 2].type_as(pred_points_masked) * self.voxel_encoder.vy + self.voxel_encoder.y_offset)  # M
            z_shift = (masked_voxel_coors[:, 1].type_as(pred_points_masked) * self.voxel_encoder.vz + self.voxel_encoder.z_offset)  # M
            shift = torch.cat([x_shift.unsqueeze(-1), y_shift.unsqueeze(-1), z_shift.unsqueeze(-1)], dim=1).view(-1, 1, 3)
            pred_points_masked[..., 0] = pred_points_masked[..., 0] * self.voxel_encoder.vx / 2  # [-1, 1] -> [voxel_encoder.vx/2, voxel_encoder.vx/2]
            pred_points_masked[..., 1] = pred_points_masked[..., 1] * self.voxel_encoder.vy / 2  # [-1, 1] -> [voxel_encoder.vy/2, voxel_encoder.vy/2]
            pred_points_masked[..., 2] = pred_points_masked[..., 2] * self.voxel_encoder.vz / 2  # [-1, 1] -> [voxel_encoder.vz/2, voxel_encoder.vz/2]
            batch.append(masked_voxel_coors[:, 0].view(-1, 1).repeat(1, n).view(-1))
            
            gt_points_unmasked = pred_dict["gt_points_unmasked"].clone()
            gt_points_unmasked[..., 0] = gt_points_unmasked[..., 0] * self.voxel_encoder.vx / 2  # [-1, 1] -> [voxel_encoder.vx/2, voxel_encoder.vx/2]
            gt_points_unmasked[..., 1] = gt_points_unmasked[..., 1] * self.voxel_encoder.vy / 2  # [-1, 1] -> [voxel_encoder.vy/2, voxel_encoder.vy/2]
            gt_points_unmasked[..., 2] = gt_points_unmasked[..., 2] * self.voxel_encoder.vz / 2  # [-1, 1] -> [voxel_encoder.vz/2, voxel_encoder.vz/2]
            x_shift_m = (unmasked_voxel_coors[:, 3].type_as(pred_points_masked) * self.voxel_encoder.vx + self.voxel_encoder.x_offset)  # M
            y_shift_m = (unmasked_voxel_coors[:, 2].type_as(pred_points_masked) * self.voxel_encoder.vy + self.voxel_encoder.y_offset)  # M
            z_shift_m = (unmasked_voxel_coors[:, 1].type_as(pred_points_masked) * self.voxel_encoder.vz + self.voxel_encoder.z_offset)  # M
            shift_m = torch.cat([x_shift_m.unsqueeze(-1), y_shift_m.unsqueeze(-1), z_shift_m.unsqueeze(-1)], dim=1).view(-1, 1, 3)
            gt_points_unmasked = gt_points_unmasked + shift_m
            
            points.append((pred_points_masked + shift).reshape(-1, 3))
            points.append(gt_points_unmasked.reshape(-1, 3))

        if "pred_points_unmasked" in pred_dict:
            pred_points_unmasked = pred_dict["pred_points_unmasked"]  # N-M, num_chamfer_points, 3
            M, n, C = pred_points_unmasked.shape
            x_shift = unmasked_voxel_coors[:, 3].type_as(pred_points_unmasked) * self.voxel_encoder.vx + self.voxel_encoder.x_offset  # M
            y_shift = unmasked_voxel_coors[:, 2].type_as(pred_points_unmasked) * self.voxel_encoder.vy + self.voxel_encoder.y_offset  # M
            z_shift = unmasked_voxel_coors[:, 1].type_as(pred_points_unmasked) * self.voxel_encoder.vz + self.voxel_encoder.z_offset  # M
            shift = torch.cat([x_shift.unsqueeze(-1), y_shift.unsqueeze(-1), z_shift.unsqueeze(-1)], dim=1).view(-1, 1, 3)
            batch.append(unmasked_voxel_coors[:, 0].view(-1, 1).repeat(1, n).view(-1))
            points.append((pred_points_unmasked + shift).reshape(-1, 3))
        points = torch.cat(points, dim=0) if points else None
        batch = torch.cat(batch, dim=0) if batch else None

        return {
            "occupied_bev": occupied,
            "gt_num_points_bev": gt_num_points,
            "diff_num_points_bev": diff_num_points,
            "points": points,
            "points_batch": batch,
            "gt_points": pred_dict["gt_points"],
            "gt_points_unmasked": gt_points_unmasked,
            "gt_points_batch":  pred_dict["gt_point_coors"][:, 0],
            "point_cloud_range": self.voxel_encoder.point_cloud_range,
            "voxel_shape": (self.voxel_encoder.vx, self.voxel_encoder.vy, self.voxel_encoder.vz)
        }
    
    def loss(self, batch_inputs_dict, batch_data_samples,
             **kwargs):
        loss = self._forward(batch_inputs_dict, batch_data_samples)
        return loss
    
    def _forward(self,
                batch_inputs_dict,
                batch_samples_dict,
                return_loss=True, 
                pretrain=False,
                out_dir=None,
                **kwargs,
                ):
        
        batched_input_metas = [item.metainfo for item in batch_samples_dict]
        if return_loss:
            lidar_x, lidar_volume_embed = self.extract_feat(batch_inputs_dict)

            imgs = batch_inputs_dict.get('imgs', None)
            
            imgs = torch.transpose(torch.transpose(imgs, 2, 3), 3, 4)
            B, N, C, H, W = imgs.size() # CARE SUS???
            imgs = imgs.view(B * N, C, H, W)
            camera_x, camera_mask, camera_ids_restore = self.camera_encoder["backbone"](imgs, camera_only=True)
    
            camera_volume_embed, camera_x = self.camera_encoder["vtransform"](
                camera_x, 
                (B, N, C, H, W),
                camera_ids_restore, 
                batched_input_metas
            )

            cam_proj_feat, lidar_proj_feat = self.fusion_module(
                [lidar_volume_embed, camera_volume_embed], 
                lidar_x,
                batched_input_metas,
            )
            cam_pred = self.relu(cam_proj_feat + camera_x.view(B*N, -1, H//32, W//32))
            cam_pred = cam_pred.flatten(2).permute(0, 2, 1)
            cam_pred = self.camera_decoder(cam_pred, camera_ids_restore)

            if get_rank() == 0 and self.sample_idx % 500 == 0:
                vis_image(imgs[0:6], cam_pred[0:6], camera_mask[0:6], self, 'viz', sample_idx=f'{self.epoch}_{self.sample_idx}.png')
            if get_rank() == 0:
                self.sample_idx += 1

            lidar_x[0]['output'] = lidar_proj_feat

            lidar_pred = self.bbox_head(lidar_x[0])
            losses = self.bbox_head.loss(*lidar_pred)
            losses['camera_loss'] = self.camera_decoder.forward_loss(imgs, cam_pred, camera_mask)
            
            return losses
        
        elif pretrain:
            for var, name in [(points, 'points'), (img_metas, 'img_metas')]:
                if not isinstance(var, list):
                    raise TypeError('{} must be a list, but got {}'.format(
                        name, type(var)))
                    
            img = [img] if img is None else img
            return self.test_pretrain(img[0], points[0], img_metas[0], out_dir)
        
        else: # i think this is prediction logic, ignore for now since we are doing pretraining
            for var, name in [(points, 'points'), (img_metas, 'img_metas')]:
                if not isinstance(var, list):
                    raise TypeError('{} must be a list, but got {}'.format(
                        name, type(var)))

            num_augs = len(points)
            if num_augs != len(img_metas):
                raise ValueError(
                    'num of augmentations ({}) != num of image meta ({})'.format(
                        len(points), len(img_metas)))
                
            if num_augs == 1:
                img = [img] if img is None else img
                return self.simple_test(points[0], img_metas[0], img[0])
            else:
                return self.aug_test(points, img_metas, img)

@MODELS.register_module()
class DummyModule(nn.Module):
    """Placeholder module to satisfy parent class requirements."""
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, x, *args, **kwargs):
        return x
    def init_weights(self):
        pass

# --- Visualization Utils ---
def show_image(image, title=''):
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def vis_image(ori_img, pred_img, mask, model, out_dir, sample_idx):
    from pathlib import Path
    Path(f'{out_dir}/cam').mkdir(parents=True, exist_ok=True)
    
    # Handle patchify/unpatchify
    ori_img = model.camera_decoder.patchify(ori_img)
    mean = ori_img.mean(dim=-1, keepdim=True)
    var = ori_img.var(dim=-1, keepdim=True)
    ori_img = model.camera_decoder.unpatchify(ori_img)
    x = torch.einsum('nchw->nhwc', ori_img).detach().cpu()

    pred_img = pred_img * (var + 1.e-6)**.5 + mean
    y = model.camera_decoder.unpatchify(pred_img)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.camera_decoder.final_patch_size**2 *3)
    mask = model.camera_decoder.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    im_masked = x * (1 - mask)
    im_paste = x * (1 - mask) + y * mask

    plt.rcParams['figure.figsize'] = [24, 12]
    B = min(x.shape[0], 6)
    for i in range(B):
        plt.subplot(B, 4, i*4+1)
        show_image(x[i], "original")
        plt.subplot(B, 4, i*4+2)
        show_image(im_masked[i], "masked")
        plt.subplot(B, 4, i*4+3)
        show_image(y[i], "reconstruction")
        plt.subplot(B, 4, i*4+4)
        show_image(im_paste[i], "reconstruction + visible")

    plt.savefig(f"{out_dir}/cam/{sample_idx}")
    plt.close()

@MODELS.register_module()
class DummyHead(nn.Module):
    """Placeholder module for bbox_head."""
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, x):
        return x
    def loss(self, *args, **kwargs):
        return dict()
    def predict(self, *args, **kwargs):
        return []
    def init_weights(self):
        pass




@MODELS.register_module()
class UniM2AE_modular(DynamicVoxelNet):
    def __init__(self, 
                 data_preprocessor, # Shared
                 fusion_module = None, #Shared, not needed when there is only one 
                 voxel_encoder = None, # Lidar
                 middle_encoder = None, # Lidar
                 backbone = None, # Lidar
                 camera_backbone = None, # Camera
                 camera_vtransform = None, # Camera
                 camera_decoder=None, # Camera
                 neck=None, 
                 bbox_head=None, 
                 train_cfg=None, 
                 test_cfg=None, 
                 init_cfg=None, 
                 ):

        # Workaround for dynamic voxelnet init
        _voxel_enc = voxel_encoder if voxel_encoder else dict(type='DummyModule') 
        _middle_enc = middle_encoder if middle_encoder else dict(type='DummyModule')
        _backbone = backbone if backbone else dict(type='DummyModule')
        _bbox_head = bbox_head if bbox_head else dict(type='DummyHead')


        super(UniM2AE_modular, self).__init__(
            voxel_encoder=_voxel_enc, 
            middle_encoder=_middle_enc, 
            backbone=_backbone, 
            neck=neck, 
            bbox_head=_bbox_head, 
            train_cfg=train_cfg, 
            test_cfg=test_cfg, 
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor
            )
        
        # use_lidar
        self.with_lidar = voxel_encoder is not None
        # use_camera
        self.with_camera = camera_backbone is not None
        # both
        self.with_fusion = fusion_module is not None
       # build camera
        if self.with_camera:
            self.camera_encoder = nn.ModuleDict({
                "backbone": MODELS.build(camera_backbone),
                "vtransform": MODELS.build(camera_vtransform),
            })
            self.camera_decoder = MODELS.build(camera_decoder)
        else:
            self.camera_encoder = None
            self.camera_decoder = None

        self.relu = nn.ReLU(inplace=True)
        
        # build lidar
        if self.with_lidar:
             norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
             self.deblock_lidar = nn.Sequential(
                    build_conv_layer(dict(type='Conv3d', bias=False), 128, 192, 3, 1, 1),
                    build_norm_layer(norm_cfg, 192)[1],
                    nn.ReLU(inplace=True),
            )
            
        # Need fusion module only when both camera and lidar are used
        if fusion_module is not None:
            self.fusion_module = MODELS.build(fusion_module)
            self.with_fusion = True
            
         
            if hasattr(self.fusion_module, 'proj_cam_downsample'):
                if isinstance(self.fusion_module.proj_cam_downsample, (nn.Conv2d, nn.Sequential)):
                     # Wrap single layer in ModuleList so the loop 'for layer in ...' works
                     self.fusion_module.proj_cam_downsample = nn.ModuleList([self.fusion_module.proj_cam_downsample])
            # --------------------------
        else:
            self.fusion_module = None
            self.with_fusion = False
        
        if not self.with_lidar:
            self.voxel_encoder  = None
            self.middle_encoder = None
            self.backbone = None

        self.sample_idx = 0
        self.epoch = 0

    def init_weights(self):
        # Prevent parent from crashing on None modules
        if self.with_lidar:
            if self.voxel_encoder: self.voxel_encoder.init_weights()
            if self.middle_encoder: self.middle_encoder.init_weights()
            if self.backbone: self.backbone.init_weights()
            if self.neck: self.neck.init_weights()
            if self.bbox_head: self.bbox_head.init_weights()
        if self.with_lidar and self.with_camera:
            self.fusion_module.init_weights()

    # This function only extracts lidar features
    def extract_feat(self, batch_inputs_dict):
        if not self.with_lidar:
            return None, None
        with torch.cuda.amp.autocast(enabled=False):
            voxels, coors = batch_inputs_dict['voxels']['voxels'], batch_inputs_dict['voxels']['coors']
            batch_size = coors[-1, 0].item() + 1
            voxel_features, feature_coors, low_level_point_feature, indices = self.voxel_encoder(voxels, coors)
        x = self.middle_encoder(voxel_features, feature_coors, low_level_point_feature, indices, batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        lidar_x = x[1]
        lidar_volume_embed = self.deblock_lidar(lidar_x[0]['output'])
        return lidar_x, lidar_volume_embed
    
    @torch.no_grad()
    def test_pretrain(self,  batch_inputs_dict,
                batch_samples_dict,
                      out_dir, sample_idx=0):
        """Test function without augmentaiton."""
        batch_size = len(batch_inputs_dict['points'])
        vx, vy, vz = self.middle_encoder.sparse_shape

        batched_input_metas = [item.metainfo for item in batch_samples_dict]

        # Standard forward pass 
        lidar_x, lidar_volume_embed = self.extract_feat(batch_inputs_dict)

        imgs = batch_inputs_dict.get('imgs', None)
        imgs = torch.transpose(torch.transpose(imgs, 2, 3), 3, 4)
        B, N, C, H, W = imgs.size() # CARE SUS???
        imgs = imgs.view(B * N, C, H, W)

        camera_x, camera_mask, camera_ids_restore = self.camera_encoder["backbone"](imgs, camera_only=True)

        camera_volume_embed, camera_x = self.camera_encoder["vtransform"](
            camera_x, 
            (B, N, C, H, W),
            camera_ids_restore, 
            batched_input_metas
        )

        cam_proj_feat, lidar_proj_feat = self.fusion_module(
            [lidar_volume_embed, camera_volume_embed], 
            lidar_x,
            batched_input_metas,
        )
        cam_pred = self.relu(cam_proj_feat + camera_x.view(B*N, -1, H//32, W//32))
        cam_pred = cam_pred.flatten(2).permute(0, 2, 1)
        cam_pred = self.camera_decoder(cam_pred, camera_ids_restore)

        
        
        vis_image(imgs, cam_pred, camera_mask, self, out_dir, sample_idx=sample_idx)
        
        lidar_x[0]['output'] = lidar_proj_feat

        outs = self.bbox_head(lidar_x[0], show=True)
        pred_dict = outs[0] # unsure
        voxel_coors = pred_dict["voxel_coors"]
        masked_voxel_coors = pred_dict["masked_voxel_coors"]
        unmasked_voxel_coors = pred_dict["unmasked_voxel_coors"]

        occupied = None
        if "pred_occupied" in pred_dict:
            occupied = -torch.ones((batch_size, vx, vy), dtype=torch.long, device=pred_dict["pred_occupied"].device)
            index = (voxel_coors[:, 0], voxel_coors[:, 3], voxel_coors[:, 2])  # b ,x, y
            unmasked_index = (
                unmasked_voxel_coors[:, 0], unmasked_voxel_coors[:, 3], unmasked_voxel_coors[:, 2])
            gt_occupied = pred_dict["gt_occupied"].long()+1  # 1 -> real voxel, 2 -> fake voxel
            occupied[index] = 2 * gt_occupied  # 2 -> real voxel, 4 -> fake voxel
            occupied[unmasked_index] -= 2  # 0 -> unmasked voxels 2 -> masked voxel, 4 -> fake voxel
            occupied[index] += (torch.sigmoid(pred_dict["pred_occupied"]) + 0.5).long()
            # 0 -> unmasked voxel predicted as real,
            # 1 -> unmasked voxel predicted as fake,
            # 2 -> masked voxel predicted as real,
            # 3 -> masked voxel predicted as fake,
            # 4 -> fake voxel predicted as real,
            # 5 -> fake voxel predicted as fake

        gt_num_points = None
        diff_num_points = None
        if "pred_num_points_masked" in pred_dict:
            device = pred_dict["pred_num_points_masked"].device
            gt_num_points = torch.zeros((batch_size, vx, vy), dtype=torch.long, device=device)
            diff_num_points = torch.zeros((batch_size, vx, vy), dtype=torch.float, device=device)
            index = (masked_voxel_coors[:, 0], masked_voxel_coors[:, 3], masked_voxel_coors[:, 2])  # b ,x, y
            pred_num_points_masked = pred_dict["pred_num_points_masked"]
            gt_num_points_masked = pred_dict["gt_num_points_masked"]
            gt_num_points[index] = gt_num_points_masked.long()
            diff_num_points[index] = gt_num_points_masked.float()-pred_num_points_masked
        if "pred_num_points_unmasked" in pred_dict:
            device = pred_dict["pred_num_points_unmasked"].device
            gt_num_points = torch.zeros((batch_size, vx, vy), dtype=torch.long, device=device) if gt_num_points is None else gt_num_points
            diff_num_points = torch.zeros((batch_size, vx, vy), dtype=torch.float, device=device) if diff_num_points is None else diff_num_points
            index = (unmasked_voxel_coors[:, 0], unmasked_voxel_coors[:, 3], unmasked_voxel_coors[:, 2])  # b ,x, y
            pred_num_points_unmasked = pred_dict["pred_num_points_unmasked"]
            gt_num_points_unmasked = pred_dict["gt_num_points_unmasked"]
            gt_num_points[index] = gt_num_points_unmasked.long()
            diff_num_points[index] = gt_num_points_unmasked.float() - pred_num_points_unmasked

        points = []
        batch = []
        if "pred_points_masked" in pred_dict:
            pred_points_masked = pred_dict["pred_points_masked"].clone()  # M, num_chamfer_points, 3
            M, n, C = pred_points_masked.shape
            x_shift = (masked_voxel_coors[:, 3].type_as(pred_points_masked) * self.voxel_encoder.vx + self.voxel_encoder.x_offset)  # M
            y_shift = (masked_voxel_coors[:, 2].type_as(pred_points_masked) * self.voxel_encoder.vy + self.voxel_encoder.y_offset)  # M
            z_shift = (masked_voxel_coors[:, 1].type_as(pred_points_masked) * self.voxel_encoder.vz + self.voxel_encoder.z_offset)  # M
            shift = torch.cat([x_shift.unsqueeze(-1), y_shift.unsqueeze(-1), z_shift.unsqueeze(-1)], dim=1).view(-1, 1, 3)
            pred_points_masked[..., 0] = pred_points_masked[..., 0] * self.voxel_encoder.vx / 2  # [-1, 1] -> [voxel_encoder.vx/2, voxel_encoder.vx/2]
            pred_points_masked[..., 1] = pred_points_masked[..., 1] * self.voxel_encoder.vy / 2  # [-1, 1] -> [voxel_encoder.vy/2, voxel_encoder.vy/2]
            pred_points_masked[..., 2] = pred_points_masked[..., 2] * self.voxel_encoder.vz / 2  # [-1, 1] -> [voxel_encoder.vz/2, voxel_encoder.vz/2]
            batch.append(masked_voxel_coors[:, 0].view(-1, 1).repeat(1, n).view(-1))
            
            gt_points_unmasked = pred_dict["gt_points_unmasked"].clone()
            gt_points_unmasked[..., 0] = gt_points_unmasked[..., 0] * self.voxel_encoder.vx / 2  # [-1, 1] -> [voxel_encoder.vx/2, voxel_encoder.vx/2]
            gt_points_unmasked[..., 1] = gt_points_unmasked[..., 1] * self.voxel_encoder.vy / 2  # [-1, 1] -> [voxel_encoder.vy/2, voxel_encoder.vy/2]
            gt_points_unmasked[..., 2] = gt_points_unmasked[..., 2] * self.voxel_encoder.vz / 2  # [-1, 1] -> [voxel_encoder.vz/2, voxel_encoder.vz/2]
            x_shift_m = (unmasked_voxel_coors[:, 3].type_as(pred_points_masked) * self.voxel_encoder.vx + self.voxel_encoder.x_offset)  # M
            y_shift_m = (unmasked_voxel_coors[:, 2].type_as(pred_points_masked) * self.voxel_encoder.vy + self.voxel_encoder.y_offset)  # M
            z_shift_m = (unmasked_voxel_coors[:, 1].type_as(pred_points_masked) * self.voxel_encoder.vz + self.voxel_encoder.z_offset)  # M
            shift_m = torch.cat([x_shift_m.unsqueeze(-1), y_shift_m.unsqueeze(-1), z_shift_m.unsqueeze(-1)], dim=1).view(-1, 1, 3)
            gt_points_unmasked = gt_points_unmasked + shift_m
            
            points.append((pred_points_masked + shift).reshape(-1, 3))
            points.append(gt_points_unmasked.reshape(-1, 3))

        if "pred_points_unmasked" in pred_dict:
            pred_points_unmasked = pred_dict["pred_points_unmasked"]  # N-M, num_chamfer_points, 3
            M, n, C = pred_points_unmasked.shape
            x_shift = unmasked_voxel_coors[:, 3].type_as(pred_points_unmasked) * self.voxel_encoder.vx + self.voxel_encoder.x_offset  # M
            y_shift = unmasked_voxel_coors[:, 2].type_as(pred_points_unmasked) * self.voxel_encoder.vy + self.voxel_encoder.y_offset  # M
            z_shift = unmasked_voxel_coors[:, 1].type_as(pred_points_unmasked) * self.voxel_encoder.vz + self.voxel_encoder.z_offset  # M
            shift = torch.cat([x_shift.unsqueeze(-1), y_shift.unsqueeze(-1), z_shift.unsqueeze(-1)], dim=1).view(-1, 1, 3)
            batch.append(unmasked_voxel_coors[:, 0].view(-1, 1).repeat(1, n).view(-1))
            points.append((pred_points_unmasked + shift).reshape(-1, 3))
        points = torch.cat(points, dim=0) if points else None
        batch = torch.cat(batch, dim=0) if batch else None

        return {
            "occupied_bev": occupied,
            "gt_num_points_bev": gt_num_points,
            "diff_num_points_bev": diff_num_points,
            "points": points,
            "points_batch": batch,
            "gt_points": pred_dict["gt_points"],
            "gt_points_unmasked": gt_points_unmasked,
            "gt_points_batch":  pred_dict["gt_point_coors"][:, 0],
            "point_cloud_range": self.voxel_encoder.point_cloud_range,
            "voxel_shape": (self.voxel_encoder.vx, self.voxel_encoder.vy, self.voxel_encoder.vz)
        }
    
    def loss(self, batch_inputs_dict, batch_data_samples,
             **kwargs):
        loss = self._forward(batch_inputs_dict, batch_data_samples)
        return loss
    

    def _forward(self,
                batch_inputs_dict,
                batch_samples_dict,
                return_loss=True, 
                pretrain=False,
                out_dir=None,
                **kwargs,
                ):
        
        batched_input_metas = [item.metainfo for item in batch_samples_dict]
     
        if return_loss:
            losses = dict()

            # 1. LiDAR
            lidar_x = None
            lidar_volume_embed = None
            if self.with_lidar: 
                lidar_x, lidar_volume_embed = self.extract_feat(batch_inputs_dict)

            # 2. Camera
            camera_volume_embed = None
            camera_x = None
            camera_mask = None
            camera_ids_restore = None
            cam_pred = None
            
            imgs = batch_inputs_dict.get('imgs', None)
            run_camera = self.with_camera and (imgs is not None)
            
            if run_camera:
                if imgs.ndim == 5 and imgs.shape[2] == 704 and imgs.shape[3] == 3 and imgs.shape[4] == 256:
                    imgs = imgs.permute(0, 1, 3, 4, 2).contiguous()
                elif imgs.ndim == 5 and imgs.shape[-1] == 3:
                     imgs = imgs.permute(0, 1, 4, 2, 3).contiguous()

                B, N, C, H, W = imgs.size()
                imgs = imgs.view(B * N, C, H, W)
                
                camera_x, camera_mask, camera_ids_restore = self.camera_encoder["backbone"](imgs, camera_only=True)
                camera_volume_embed, camera_x = self.camera_encoder["vtransform"](
                    camera_x, 
                    (B, N, C, H, W),
                    camera_ids_restore, 
                    batched_input_metas
                )

            # 3. Fusion
            cam_proj_feat, lidar_proj_feat = None, None
            is_camera_only = (lidar_volume_embed is None) and (camera_volume_embed is not None)

            if self.with_fusion:
                # --- Dummy Inputs for MMIM ---
                if is_camera_only:
                    # Camera Only: create dummy LiDAR
                    lidar_volume_embed = torch.zeros_like(camera_volume_embed)
                    B_val = camera_volume_embed.shape[0]
                    dummy_gather = [torch.zeros(1, dtype=torch.long, device=camera_volume_embed.device) for _ in range(B_val)]
                    dummy_scatter = [torch.zeros(1, dtype=torch.long, device=camera_volume_embed.device) for _ in range(B_val)]
                    
                    # Create dummy voxel_coors to satisfy MMIM (shape [B, 4])
                    dummy_voxel_coors = torch.zeros((B_val, 4), dtype=torch.int32, device=camera_volume_embed.device)

                    lidar_x = [
                        {
                            'output': camera_volume_embed,
                            'voxel_coors': dummy_voxel_coors # Required by cam_transform
                        }, 
                        dummy_gather, 
                        dummy_scatter, 
                        B_val 
                    ]
                
                elif camera_volume_embed is None and lidar_volume_embed is not None:
                    camera_volume_embed = torch.zeros_like(lidar_volume_embed)
                # -----------------------------

                cam_proj_feat, lidar_proj_feat = self.fusion_module(
                    [lidar_volume_embed, camera_volume_embed], 
                    lidar_x,
                    batched_input_metas,
                )

            # 4. Camera Loss
            if run_camera:
                H_feat = H // 32
                W_feat = W // 32
                cam_x_feat = camera_x.view(B*N, -1, H_feat, W_feat)

                # Use fusion features only if valid (not camera-only mode)
                if cam_proj_feat is not None and not is_camera_only:
                    try:
                        reshaped_cam_proj = cam_proj_feat.reshape(B*N, -1, H_feat, W_feat)
                        if reshaped_cam_proj.shape == cam_x_feat.shape:
                            cam_x_feat = cam_x_feat + reshaped_cam_proj
                    except:
                        pass
                
                cam_pred = self.relu(cam_x_feat)
                cam_pred = cam_pred.flatten(2).permute(0, 2, 1)
                cam_pred = self.camera_decoder(cam_pred, camera_ids_restore)

                if get_rank() == 0 and self.sample_idx % 500 == 0:
                    vis_image(imgs[0:6], cam_pred[0:6], camera_mask[0:6], self, 'viz', sample_idx=f'{self.epoch}_{self.sample_idx}.png')
                if get_rank() == 0:
                    self.sample_idx += 1

                losses['camera_loss'] = self.camera_decoder.forward_loss(imgs, cam_pred, camera_mask)

                # --- DDP Fix: Connect unused fusion params ---
                if self.with_fusion and cam_proj_feat is not None:
                    dummy_loss = 0.0 * cam_proj_feat.sum()
                    losses['camera_loss'] += dummy_loss

            # 5. LiDAR Loss
            if self.with_lidar and (self.bbox_head is not None):
                if lidar_proj_feat is not None:
                    lidar_x[0]['output'] = lidar_proj_feat
                lidar_pred = self.bbox_head(lidar_x[0])
                losses.update(self.bbox_head.loss(*lidar_pred))
            
            return losses
        
        elif pretrain:
            return self.test_pretrain(batch_inputs_dict, batch_samples_dict, out_dir)
        
        else: 
            points = batch_inputs_dict.get('points', None)
            imgs = batch_inputs_dict.get('imgs', None)
            img_metas = batched_input_metas 
            if points is None and imgs is None: raise ValueError("No input data provided")
            if isinstance(points, list):
                if len(points) == 1: return self.simple_test(points[0], img_metas[0], imgs[0] if imgs else None)
                else: return self.aug_test(points, img_metas, imgs)
            else: return self.simple_test(points, img_metas, imgs)

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        return self._forward(batch_inputs_dict, batch_data_samples, return_loss=True, **kwargs)

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        return self._forward(batch_inputs_dict, batch_data_samples, return_loss=False, **kwargs)
