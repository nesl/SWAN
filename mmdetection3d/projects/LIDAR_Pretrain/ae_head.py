import torch
from torch.nn.functional import smooth_l1_loss, binary_cross_entropy_with_logits, l1_loss, mse_loss
from mmengine.model import BaseModule
from torch import nn as nn
import numpy as np
from mmdet3d.registry import MODELS


@MODELS.register_module()
class LIDARReconstructionHead(BaseModule):


    def __init__(self,
                 in_channels=128, # CHECK pts feat assuming z is not flatten, e.g. 1, 128, 720, 720, 2
                 grid_size=(720, 720, 2),
                 point_cloud_range=[-54, -54, -5, 54, 54, 3],
                 voxel_size=[0.15, 0.15, 4],  
                 num_chamfer_points=20,           # Number of points to predict per voxel
                 loss_chamfer=dict(type='ChamferLoss', loss_weight=1.0), # Placeholder name
                 loss_occupancy=dict(type='mmdet.FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
                 init_cfg=None):
        """LIDAR Reconstruction Head."""
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_chamfer_points = num_chamfer_points
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.grid_size = grid_size
        self.pred_dims = 3  # x, y, z
        self.loss_chamfer = MODELS.build(loss_chamfer) if loss_chamfer is not None else None
        self.loss_occupancy = MODELS.build(loss_occupancy) if loss_occupancy is not None else None

        # predict voxel is empty
        self.conv_occupied = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm3d(in_channels // 2),
            nn.ReLU(),
            nn.Conv3d(in_channels // 2, 1, kernel_size=1)
        )
        # predict num_chamfer_points per voxel in 3d
        self.conv_chamfer = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm3d(in_channels // 2),
            nn.ReLU(),
            nn.Conv3d(in_channels // 2, num_chamfer_points * 3, kernel_size=1)
        )


    def _apply_1dconv(self, conv, x):
        if conv is not None:
            x = x.unsqueeze(0).transpose(1, 2)
            x = conv(x).transpose(1, 2).squeeze(0)
            return x

    def forward(self, x, show=None):
        """Forward pass.

        Args:r
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple[list[torch.Tensor]]: Multi-level class score, bbox \
                and direction predictions.
        """
        voxel_info_decoder = x
        dec_out = voxel_info_decoder["output"]  # [N, C]
        gt_dict = voxel_info_decoder["gt_dict"]
        pred_dict = {}

        masked_predictions = dec_out[voxel_info_decoder["dec2masked_idx"]]
        unmasked_predictions = dec_out[voxel_info_decoder["dec2unmasked_idx"]]

        # Occupied loss
        if self.use_fake_voxels:
            pred_occupied = self._apply_1dconv(self.conv_occupied, dec_out).view(-1)
            gt_occupied = gt_dict["fake_voxel_mask"]
            gt_occupied = gt_occupied[voxel_info_decoder["original_index"]]  # Maps the input to decoder output index
            pred_dict["pred_occupied"] = pred_occupied
            pred_dict["gt_occupied"] = gt_occupied

        # Predict number of points loss
        if self.use_num_points:
            gt_num_points = gt_dict["num_points_per_voxel"]

            pred_num_points_masked = self._apply_1dconv(self.conv_num_points, masked_predictions).view(-1)
            gt_num_points_masked = gt_num_points[voxel_info_decoder["masked_idx"]]
            pred_dict["pred_num_points_masked"] = pred_num_points_masked
            pred_dict["gt_num_points_masked"] = gt_num_points_masked

            if not self.only_masked:
                pred_num_points_unmasked = self._apply_1dconv(self.conv_num_points, unmasked_predictions).view(-1)
                gt_num_points_unmasked = gt_num_points[voxel_info_decoder["unmasked_idx"]]
                pred_dict["pred_num_points_unmasked"] = pred_num_points_unmasked
                pred_dict["gt_num_points_unmasked"] = gt_num_points_unmasked

        # Chamfer loss
        if self.use_chamfer:
            gt_points_per_voxel = gt_dict["points_per_voxel"]
            gt_points_padding = gt_dict["points_per_voxel_padding"]

            pred_points_masked = self._apply_1dconv(self.conv_chamfer, masked_predictions).view(
                len(masked_predictions), self.num_chamfer_points, self.pred_dims)
            pred_points_masked = torch.tanh(pred_points_masked)  # map to [-1, 1]
            gt_points_masked = gt_points_per_voxel[voxel_info_decoder["masked_idx"]]
            gt_point_padding_masked = gt_points_padding[voxel_info_decoder["masked_idx"]]
            pred_dict["pred_points_masked"] = pred_points_masked
            pred_dict["gt_points_masked"] = gt_points_masked
            pred_dict["gt_point_padding_masked"] = gt_point_padding_masked.bool()

            if not self.only_masked:
                pred_points_unmasked = self._apply_1dconv(self.conv_chamfer, unmasked_predictions).view(
                    len(unmasked_predictions), self.num_chamfer_points, self.pred_dims)
                pred_points_unmasked = torch.tanh(pred_points_unmasked)  # map to [-1, 1]
                gt_points_unmasked = gt_points_per_voxel[voxel_info_decoder["unmasked_idx"]]
                gt_point_padding_unmasked = gt_points_padding[voxel_info_decoder["unmasked_idx"]]
                pred_dict["pred_points_unmasked"] = pred_points_unmasked
                pred_dict["gt_points_unmasked"] = gt_points_unmasked
                pred_dict["gt_point_padding_unmasked"] = gt_point_padding_unmasked.bool()

        if show is not None:
            pred_dict["voxel_coors"] = voxel_info_decoder["voxel_coors"]
            pred_dict["masked_voxel_coors"] = voxel_info_decoder["voxel_coors"][
                voxel_info_decoder["dec2masked_idx"]]  # b, z, y, x
            pred_dict["unmasked_voxel_coors"] = voxel_info_decoder["voxel_coors"][
                voxel_info_decoder["dec2unmasked_idx"]]  # b, z, y, x
            pred_dict["gt_points"] = gt_dict["gt_points"]
            pred_dict["gt_point_coors"] = gt_dict["gt_point_coors"]
            pred_dict["gt_points_unmasked"] = gt_points_per_voxel[voxel_info_decoder["unmasked_idx"]]

        return pred_dict,  # Output needs to be tuple which the ',' achieves

    @staticmethod
    def rel_error_loss(src, trg):
        loss = l1_loss(src, trg, reduction="none")
        loss = torch.sqrt(loss/trg).mean()
        return loss

    @staticmethod
    def rel_error_loss_2(src, trg):
        loss_l1 = l1_loss(src, trg, reduction="none")
        loss_l2 = mse_loss(src, trg, reduction="none")

        beta = torch.clip(trg*0.1, min=0.5)
        threshold_mask = loss_l1 < beta

        loss = loss_l1 - beta/2
        loss[threshold_mask] = loss_l2[threshold_mask]/(2*beta[threshold_mask])

        loss.mean()
        return loss

    @staticmethod
    def chamfer_distance_loss(src, trg, trg_padding, criterion_mode="l2"):
        """

        :param src: predicted point positions (B, N, C)
        :param trg: gt point positions (B, M, C)
        :param trg_padding: Which points are padded (B, M)
        :type trg_padding: torch.Tensor([torch.bool])
        :param criterion_mode: way of calculating distance, l1, l2, or smooth_l1
        :return:
        """
        if criterion_mode == 'smooth_l1':
            criterion = smooth_l1_loss
        elif criterion_mode == 'l1':
            criterion = l1_loss
        elif criterion_mode == 'l2':
            criterion = mse_loss
        else:
            raise NotImplementedError
        # src->(B,N,C) dst->(B,M,C)
        src_expand = src.unsqueeze(2).repeat(1, 1, trg.shape[1], 1)  # (B,N M,C)
        trg_expand = trg.unsqueeze(1).repeat(1, src.shape[1], 1, 1)  # (B,N M,C)
        trg_padding_expand = trg_padding.unsqueeze(1).repeat(1, src.shape[1], 1)  # (B,N M)

        distance = criterion(src_expand, trg_expand, reduction='none').sum(-1)  # (B,N M)
        distance[trg_padding_expand] = np.inf

        src2trg_distance, indices1 = torch.min(distance, dim=2)  # (B,N)
        trg2src_distance, indices2 = torch.min(distance, dim=1)  # (B,M)
        trg2src_distance[trg_padding] = 0

        loss_src = torch.mean(src2trg_distance)
        # Since there is different number of points in each voxel we want to have each voxel matter equally much
        # and to not have voxels with more points be more important to mimic
        loss_trg = trg2src_distance.sum(1) / (~trg_padding).sum(1)  # B
        loss_trg = loss_trg.mean()

        return loss_src, loss_trg

    def loss(self,
             pred_dict,
             gt_bboxes_ignore=None,
             ):
        """Calculate losses.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            gt_bboxes (list[:obj:`BaseInstance3DBoxes`]): Gt bboxes
                of each sample.
            gt_labels (list[torch.Tensor]): Gt labels of each sample.
            input_metas (list[dict]): Contain pcd and img's meta info.
            gt_bboxes_ignore (None | list[torch.Tensor]): Specify
                which bounding.

        Returns:
            dict[str, list[torch.Tensor]]: Classification, bbox, and \
                direction losses of each level.

                - loss_cls (list[torch.Tensor]): Classification losses.
                - loss_bbox (list[torch.Tensor]): Box regression losses.
                - loss_dir (list[torch.Tensor]): Direction classification \
                    losses.
        """
        loss_dict = {}

        if self.use_fake_voxels:
            pred_occupied = pred_dict["pred_occupied"]
            gt_occupied = pred_dict["gt_occupied"]
            loss_occupied = binary_cross_entropy_with_logits(pred_occupied, gt_occupied)
            loss_dict["loss_occupied"] = loss_occupied

        if self.use_num_points:
            pred_num_points_masked = pred_dict["pred_num_points_masked"]
            gt_num_points_masked = pred_dict["gt_num_points_masked"]
            loss_num_points_masked = self.num_points_loss(pred_num_points_masked, gt_num_points_masked.float())
            loss_dict["loss_num_points_masked"] = loss_num_points_masked
            if not self.only_masked:
                pred_num_points_unmasked = pred_dict["pred_num_points_unmasked"]
                gt_num_points_unmasked = pred_dict["gt_num_points_unmasked"]
                loss_num_points_unmasked = self.num_points_loss(
                    pred_num_points_unmasked, gt_num_points_unmasked.float())
                loss_dict["loss_num_points_unmasked"] = loss_num_points_unmasked

        if self.use_chamfer:
            pred_points_masked = pred_dict["pred_points_masked"]
            gt_points_masked = pred_dict["gt_points_masked"]
            gt_point_padding_masked = pred_dict["gt_point_padding_masked"]
            loss_chamfer_src_masked, loss_chamfer_dst_masked = self.chamfer_loss(
                pred_points_masked, gt_points_masked, trg_padding=gt_point_padding_masked)
            loss_dict["loss_chamfer_src_masked"] = loss_chamfer_src_masked
            loss_dict["loss_chamfer_dst_masked"] = loss_chamfer_dst_masked
            if not self.only_masked:
                pred_points_unmasked = pred_dict["pred_points_unmasked"]
                gt_points_unmasked = pred_dict["gt_points_unmasked"]
                gt_point_padding_unmasked = pred_dict["gt_point_padding_unmasked"]
                loss_chamfer_src_unmasked, loss_chamfer_dst_unmasked = self.chamfer_loss(
                    pred_points_unmasked, gt_points_unmasked, trg_padding=gt_point_padding_unmasked)
                loss_dict["loss_chamfer_src_unmasked"] = loss_chamfer_src_unmasked
                loss_dict["loss_chamfer_dst_unmasked"] = loss_chamfer_dst_unmasked

        if self.loss_weights:
            for key in loss_dict:
                loss_dict[key] *= self.loss_weights.get(key, 0)  # ignore loss if not mentioned

        return loss_dict

if __name__ == '__main__':
    import pdb; pdb.set_trace()