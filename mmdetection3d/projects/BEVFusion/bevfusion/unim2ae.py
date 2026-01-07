import torch.nn as nn
import torch
from mmdet3d.registry import MODELS
from mmdet3d.models.detectors.dynamic_voxelnet import DynamicVoxelNet
import numpy as np
import matplotlib.pyplot as plt
from mmcv.cnn import build_conv_layer, build_norm_layer
import time
from mmengine.dist import get_rank



@MODELS.register_module()
class UniM2AE_BEVFusion(DynamicVoxelNet):
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
                 camera_neck=None,
                 bbox_head=None, 
                 train_cfg=None, 
                 test_cfg=None, 
                 init_cfg=None, 
                 bev_backbone=None,
                 bev_neck=None
                 ):
        super(UniM2AE_BEVFusion, self).__init__(
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
                "neck":MODELS.build(camera_neck)
            }
        )
        self.cam_vtransform = MODELS.build(camera_vtransform)

        self.relu = nn.ReLU(inplace=True)
        
        self.fusion_module = MODELS.build(fusion_module)
        
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
        self.deblock_lidar = nn.Sequential(
                build_conv_layer(
                dict(type='Conv3d', bias=False),
                in_channels=128,
                out_channels=192,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            build_norm_layer(norm_cfg, 192)[1],
            nn.ReLU(inplace=True),
        )
        self.deblock_camera = nn.Sequential(
            build_conv_layer(
                dict(type='Conv3d', bias=False),
                in_channels=80,
                out_channels=192,
                kernel_size=3,
                stride=2,
                padding=1),
            build_norm_layer(norm_cfg, 192)[1],
            nn.ReLU(inplace=True),
        )

        self.deblock_fusion = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=1, stride=1, bias=False),
            build_norm_layer(dict(type='BN', eps=1.0e-3, momentum=0.01), 192)[1],
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(192, 192, kernel_size=2, stride=2, bias=False),
            build_norm_layer(dict(type='BN', eps=1.0e-3, momentum=0.01), 192)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=1, stride=1, bias=False),
            build_norm_layer(dict(type='BN', eps=1.0e-3, momentum=0.01), 128)[1],
            nn.ReLU(inplace=True),
        )

        self.bev_backbone = MODELS.build(bev_backbone)
        self.bev_neck = MODELS.build(bev_neck)

        self.sample_idx = 0
        self.epoch = 0

    def init_weights(self) -> None:
        pass 

    def extract_feat(self, batch_inputs_dict):
        with torch.cuda.amp.autocast(enabled=False):
            voxels, coors = batch_inputs_dict['voxels']['voxels'], batch_inputs_dict['voxels']['coors']
            batch_size = coors[-1, 0].item() + 1
            voxel_features, feature_coors, low_level_point_feature, indices = self.voxel_encoder(voxels, coors)
        x = self.middle_encoder(voxel_features, feature_coors, batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        lidar_volume_embed = self.deblock_lidar(x)
        return lidar_volume_embed
    
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
        loss = self._forward(batch_inputs_dict, batch_data_samples, pretrain=False, finetune=True)
        return loss
    
    def _forward(self,
                batch_inputs_dict,
                batch_samples_dict,
                pretrain=True, 
                finetune=False,
                out_dir=None,
                **kwargs,
                ):
        
        batched_input_metas = [item.metainfo for item in batch_samples_dict]
        lidar_volume_embed = self.extract_feat(batch_inputs_dict)

        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        # imgs = torch.transpose(torch.transpose(imgs, 2, 3), 3, 4)
        B, N, C, H, W = imgs.size() # CARE SUS???
        imgs = imgs.view(B * N, C, H, W)
        camera_x = self.camera_encoder["backbone"](imgs)
        camera_x = self.camera_encoder["neck"](camera_x)
        if not isinstance(camera_x, torch.Tensor):
            camera_x = camera_x[0]

        BN, C, H, W = camera_x.size()
        camera_x = camera_x.view(B, int(BN / B), C, H, W)
        lidar2image, camera_intrinsics, camera2lidar = [], [], []
        img_aug_matrix, lidar_aug_matrix = [], []
        for i, meta in enumerate(batched_input_metas):
            lidar2image.append(meta['lidar2img'])
            camera_intrinsics.append(meta['cam2img'])
            camera2lidar.append(meta['cam2lidar'])
            img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
            lidar_aug_matrix.append(
                meta.get('lidar_aug_matrix', np.eye(4)))

        lidar2image = imgs.new_tensor(np.asarray(lidar2image))
        camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
        camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
        img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
        lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            camera_volume_embed = self.cam_vtransform(
                camera_x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                batched_input_metas,
            )
        camera_volume_embed = self.deblock_camera(camera_volume_embed)
        cam_proj_feat, lidar_proj_feat = self.fusion_module.fuse(
            [lidar_volume_embed, camera_volume_embed]
        )
        x = torch.cat((cam_proj_feat, lidar_proj_feat), dim=1)
        x = torch.cat(x.unbind(dim=-1), 1) # unbind to create BEV sapce
        x = self.deblock_fusion(x)

        x = self.bev_backbone(x)
        x = self.bev_neck(x)
        losses = dict()
        bbox_loss = self.bbox_head.loss(x, batch_samples_dict)

        losses.update(bbox_loss)

        return losses
        
       

if __name__ == '__main__':
    import pdb; pdb.set_trace()