import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmcv.ops import DynamicScatter
from torch.nn import functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer

@MODELS.register_module()
class DynamicVFE_Linear(nn.Module):
    """Dynamic Voxel feature encoder used in DV-SECOND.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.
    The number of points inside the voxel varies.

    Args:
        in_channels (int): Input channels of VFE. Defaults to 4.
        feat_channels (list(int)): Channels of features in VFE.
        with_distance (bool): Whether to use the L2 distance of points to the
            origin point. Default False.
        with_cluster_center (bool): Whether to use the distance to cluster
            center of points inside a voxel. Default to False.
        with_voxel_center (bool): Whether to use the distance to center of
            voxel for each points inside a voxel. Default to False.
        voxel_size (tuple[float]): Size of a single voxel. Default to
            (0.2, 0.2, 4).
        point_cloud_range (tuple[float]): The range of points or voxels.
            Default to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict): Config dict of normalization layers.
        mode (str): The mode when pooling features of points inside a voxel.
            Available options include 'max' and 'avg'. Default to 'max'.
        fusion_layer (dict | None): The config dict of fusion layer used in
            multi-modal detectors. Default to None.
        return_point_feats (bool): Whether to return the features of each
            points. Default to False.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 mode='max',
                 return_point_feats=False,
                 return_gt_points=False
                 ):
        super(DynamicVFE_Linear, self).__init__()
        assert mode in ['avg', 'max']
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 3
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats
        self.return_gt_points = return_gt_points
        self.fp16_enabled = False

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2

            vfe_layers.append(
                nn.Linear(in_filters, out_filters)
            )
        self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)
        self.vfe_scatter = DynamicScatter(voxel_size, point_cloud_range,
                                          (mode != 'max'))
        self.cluster_scatter = DynamicScatter(
            voxel_size, point_cloud_range, average_points=True)
        self.fusion_layer = None

    
    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        """Map voxel features to its corresponding points.

        Args:
            pts_coors (torch.Tensor): Voxel coordinate of each point.
            voxel_mean (torch.Tensor): Voxel features to be mapped.
            voxel_coors (torch.Tensor): Coordinates of valid voxels

        Returns:
            torch.Tensor: Features or centers of each point.
        """
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_z = round(
            (self.point_cloud_range[5] - self.point_cloud_range[2]) / self.vz)
        canvas_y = round(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = round(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
        # canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0].int() + 1
        canvas_len = canvas_z * canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_len, dtype=torch.long)
        # Only include non-empty pillars
        indices = (
            voxel_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            voxel_coors[:, 1] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[indices.long()] = torch.arange(
            start=0, end=voxel_mean.size(0), device=voxel_mean.device)

        # Step 2: get voxel mean for each point
        voxel_index = (
            pts_coors[:, 0] * canvas_z * canvas_y * canvas_x +
            pts_coors[:, 1] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        voxel_inds = canvas[voxel_index.long()]
        center_per_point = voxel_mean[voxel_inds, ...]
        return center_per_point

    # if out_fp16=True, the large numbers of points 
    # lead to overflow error in following layers
    def forward(self,
                features,
                coors,
                points=None,
                img_feats=None,
                img_metas=None):
        """Forward functions.

        Args:
            features (torch.Tensor): Features of voxels, shape is NxC.
            coors (torch.Tensor): Coordinates of voxels, shape is  Nx(1+NDim).[Nx4] [b,z,y,x]
            points (list[torch.Tensor], optional): Raw points used to guide the
                multi-modality fusion. Defaults to None.
            img_feats (list[torch.Tensor], optional): Image fetures used for
                multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels.
        """
        features_ls = [features]
        origin_point_coors = features[:, :3]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            voxel_mean, mean_coors = self.cluster_scatter(features, coors)
            points_mean = self.map_voxel_center_to_point(
                coors, voxel_mean, mean_coors)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)


        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)

        # features.requires_grad = True # for cam vis
        # features.register_hook(append_grad(-1)) # for cam vis
        # point_feat_dict[-1] = features.detach() # for cam vis

        low_level_point_feature = features
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)

            # point_feats.register_hook(append_grad(i)) # for cam vis
            # point_feat_dict[i] = point_feats.detach() # for cam vis

            if (i == len(self.vfe_layers) - 1 and self.fusion_layer is not None
                    and img_feats is not None):
                point_feats = self.fusion_layer(img_feats, points, point_feats,
                                                img_metas)
            voxel_feats, voxel_coors = self.vfe_scatter(point_feats, coors)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)
        if self.return_point_feats:
            return point_feats
        if self.return_gt_points:
            return voxel_feats, voxel_coors, low_level_point_feature, coors
        return voxel_feats, voxel_coors
    

def DynamicScatterMean(features, coors, voxel_size, point_cloud_range, return_inverse=False):
    """
    DynamicScatterMean
    Matches logic of DynamicScatter(mode='avg') but prevents CUDA crashes.
    Associates points to voxels based on coordinates and averages features within each voxel.
    """
    device = features.device
    vx, vy, vz = voxel_size

    # Grid dimensions
    grid_x = round((point_cloud_range[3] - point_cloud_range[0]) / vx)
    grid_y = round((point_cloud_range[4] - point_cloud_range[1]) / vy)
    grid_z = round((point_cloud_range[5] - point_cloud_range[2]) / vz)

    # coordinates are [B, Z, Y, X]
    # In the C kernel: "T *reduced_feats_offset = reduced_feats + reduce_to * num_feats;"
    b_idx = coors[:, 0].long()
    z_idx = torch.clamp(coors[:, 1].long(), 0, grid_z - 1)
    y_idx = torch.clamp(coors[:, 2].long(), 0, grid_y - 1)
    x_idx = torch.clamp(coors[:, 3].long(), 0, grid_x - 1)

    # Flatten for unique grouping
    # Prevent using the full grid
    indices = (b_idx * grid_z * grid_y * grid_x +
               z_idx * grid_y * grid_x +
               y_idx * grid_x +
               x_idx)

    # find unqiue voxels
    unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
    num_voxels = unique_indices.shape[0]
    
    # Aggregate
    # create zero tensor to hold sums
    sum_feats = torch.zeros(num_voxels, features.shape[1], device=device, dtype=features.dtype)
    # sum up features into their respective voxel slots
    sum_feats.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, features.shape[1]), features)
    # count number of points per voxel
    counts = torch.zeros(num_voxels, device=device, dtype=features.dtype)
    counts.scatter_add_(0, inverse_indices, torch.ones_like(b_idx, dtype=features.dtype))
    
    voxel_feats = sum_feats / counts.clamp(min=1).unsqueeze(1)

    # Back to [B, Z, Y, X]
    voxel_coors = torch.zeros(num_voxels, 4, device=device, dtype=torch.long)
    voxel_coors[:, 0] = unique_indices // (grid_z * grid_y * grid_x)
    rem = unique_indices % (grid_z * grid_y * grid_x)
    voxel_coors[:, 1] = rem // (grid_y * grid_x)
    rem = rem % (grid_y * grid_x)
    voxel_coors[:, 2] = rem // grid_x
    voxel_coors[:, 3] = rem % grid_x

    if return_inverse:
        return voxel_feats, voxel_coors, inverse_indices
    return voxel_feats, voxel_coors


class DynamicVFELayer(nn.Module):
    """Replace the Voxel Feature Encoder layer in VFE layers.

    This layer has the same utility as VFELayer above

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        norm_cfg (dict): Config dict of normalization layers
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)
                 ):
        super(DynamicVFELayer, self).__init__()
        self.fp16_enabled = False
        # self.units = int(out_channels / 2)
        self.norm = build_norm_layer(norm_cfg, out_channels)[1]
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (torch.Tensor): Voxels features of shape (M, C).
                M is the number of points, C is the number of channels of point features.

        Returns:
            torch.Tensor: point features in shape (M, C).
        """
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        x = self.linear(inputs)
        x = self.norm(x)
        pointwise = F.relu(x)
        return pointwise


@MODELS.register_module()
class DynamicVFE_Efficient(nn.Module):
    """Dynamic Voxel feature encoder used in DV-SECOND.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.
    The number of points inside the voxel varies.

    The main difference is that we calculate distance feature with 3d rather than 1d.

    Args:
        in_channels (int): Input channels of VFE. Defaults to 4.
        feat_channels (list(int)): Channels of features in VFE.
        with_distance (bool): Whether to use the L2 distance of points to the
            origin point. Default False.
        with_cluster_center (bool): Whether to use the distance to cluster
            center of points inside a voxel. Default to False.
        with_voxel_center (bool): Whether to use the distance to center of
            voxel for each points inside a voxel. Default to False.
        voxel_size (tuple[float]): Size of a single voxel. Default to
            (0.2, 0.2, 4).
        point_cloud_range (tuple[float]): The range of points or voxels.
            Default to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict): Config dict of normalization layers.
        mode (str): The mode when pooling features of points inside a voxel.
            Available options include 'max' and 'avg'. Default to 'max'.
        fusion_layer (dict | None): The config dict of fusion layer used in
            multi-modal detectors. Default to None.
        return_point_feats (bool): Whether to return the features of each
            points. Default to False.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 return_point_feats=False,
                 return_gt_points=False
                 ):
        super(DynamicVFE_Efficient, self).__init__()
        assert mode in ['avg', 'max']
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        # here we use 3d distance instead of 1d distance
        if with_distance:
            in_channels += 3
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.return_point_feats = return_point_feats
        self.return_gt_points = return_gt_points
        self.fp16_enabled = False

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range
        # This is never used?
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2

            vfe_layers.append(
                DynamicVFELayer(
                    in_filters,
                    out_filters,
                    norm_cfg))
        self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)
        self.vfe_scatter = DynamicScatter(voxel_size, point_cloud_range,
                                          (mode != 'max'))
        self.cluster_scatter = DynamicScatter(
            voxel_size, point_cloud_range, average_points=True)
        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = MODELS.build(fusion_layer)

    
    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors, batch_size=None):
        """Map voxel features to its corresponding points.

        Args:
            pts_coors (torch.Tensor): Voxel coordinate of each point. [N, 4] in [b, z, y, x] order
            voxel_mean (torch.Tensor): Voxel features to be mapped. [M, C]
            voxel_coors (torch.Tensor): Coordinates of valid voxels [M, 4] in [b, z, y, x] order

        Returns:
            torch.Tensor: Features or centers of each point. [N, C]
        """
        # Calculate grid dimensions
        canvas_z = round((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.vz)
        canvas_y = round((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = round((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)

        if batch_size is None:
            batch_size = int(pts_coors[:, 0].max().item()) + 1

        # Handle empty inputs
        if voxel_mean.shape[0] == 0 or pts_coors.shape[0] == 0:
            return voxel_mean.new_zeros((pts_coors.shape[0], voxel_mean.shape[1]))

        canvas_len = canvas_z * canvas_y * canvas_x * batch_size

        # Initialize canvas with -1 to indicate empty voxels
        canvas = voxel_mean.new_full((canvas_len,), -1, dtype=torch.long)

        # Clamp voxel coordinates to valid range before computing indices
        v_batch = voxel_coors[:, 0].long()
        v_z = torch.clamp(voxel_coors[:, 1].long(), 0, canvas_z - 1)
        v_y = torch.clamp(voxel_coors[:, 2].long(), 0, canvas_y - 1)
        v_x = torch.clamp(voxel_coors[:, 3].long(), 0, canvas_x - 1)

        # Only include voxels with valid batch indices
        valid_vox_mask = (v_batch >= 0) & (v_batch < batch_size)

        if valid_vox_mask.any():
            v_indices = (v_batch[valid_vox_mask] * canvas_z * canvas_y * canvas_x +
                        v_z[valid_vox_mask] * canvas_y * canvas_x +
                        v_y[valid_vox_mask] * canvas_x +
                        v_x[valid_vox_mask])

            # Clamp indices as final safety
            v_indices = torch.clamp(v_indices, 0, canvas_len - 1)

            # Create index mapping
            voxel_indices = torch.arange(voxel_mean.size(0), device=voxel_mean.device)
            canvas[v_indices] = voxel_indices[valid_vox_mask]

        # Clamp point coordinates to valid range
        p_batch = pts_coors[:, 0].long()
        p_z = torch.clamp(pts_coors[:, 1].long(), 0, canvas_z - 1)
        p_y = torch.clamp(pts_coors[:, 2].long(), 0, canvas_y - 1)
        p_x = torch.clamp(pts_coors[:, 3].long(), 0, canvas_x - 1)

        # Compute point indices (already clamped, so always valid)
        point_indices = (p_batch * canvas_z * canvas_y * canvas_x +
                        p_z * canvas_y * canvas_x +
                        p_y * canvas_x + p_x)

        # Clamp as final safety
        point_indices = torch.clamp(point_indices, 0, canvas_len - 1)

        # Gather voxel indices from canvas
        voxel_inds = canvas[point_indices]

        # Points are valid if they map to an existing voxel and have valid batch
        valid_pts_mask = (voxel_inds >= 0) & (p_batch >= 0) & (p_batch < batch_size)

        # Prepare output
        center_per_point = voxel_mean.new_zeros((pts_coors.shape[0], voxel_mean.shape[1]))

        if valid_pts_mask.any():
            # Clamp voxel_inds to valid range for safety
            safe_voxel_inds = torch.clamp(voxel_inds[valid_pts_mask], 0, voxel_mean.size(0) - 1)
            center_per_point[valid_pts_mask] = voxel_mean[safe_voxel_inds]

        return center_per_point

    # if out_fp16=True, the large numbers of points 
    # lead to overflow error in following layers
    def forward(self,
                features,
                coors,
                points=None,
                img_feats=None,
                img_metas=None,
                batch_size=None):
        """Forward functions.

        Args:
            features (torch.Tensor): Features of voxels, shape is NxC.
            coors (torch.Tensor): Coordinates of voxels, shape is  Nx(1+NDim).[Nx4] [b,z,y,x]
            points (list[torch.Tensor], optional): Raw points used to guide the
                multi-modality fusion. Defaults to None.
            img_feats (list[torch.Tensor], optional): Image fetures used for
                multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels.
        """
        if batch_size is None:
            batch_size = int(coors[:, 0].max().item()) + 1
        else:
            batch_size = int(batch_size)

        # # Calculate grid dimensions
        # canvas_z = round((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.vz)
        # canvas_y = round((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        # canvas_x = round((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)

        # # Clamp coordinates instead of filtering to preserve tensor shapes for backward pass
        # coors_clamped = coors.clone()
        # coors_clamped[:, 0] = torch.clamp(coors[:, 0], 0, batch_size - 1)
        # coors_clamped[:, 1] = torch.clamp(coors[:, 1], 0, canvas_z - 1)  # z
        # coors_clamped[:, 2] = torch.clamp(coors[:, 2], 0, canvas_y - 1)  # y
        # coors_clamped[:, 3] = torch.clamp(coors[:, 3], 0, canvas_x - 1)  # x

        # # Create spatial_coors in MMCV order [batch_idx, x, y, z] with clamped values
        # spatial_coors = coors_clamped.contiguous().long()

        features_ls = [features]

        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            # Use DynamicScatterMean with return_inverse for efficient point mapping
            voxel_mean, mean_coors, cluster_inverse = DynamicScatterMean(
                features[:, :3], coors,
                (self.vx, self.vy, self.vz),
                self.point_cloud_range,
                return_inverse=True
            )
            # Map voxel mean back to points using inverse indices (much faster than canvas lookup)
            points_mean = voxel_mean[cluster_inverse]
            f_cluster = features[:, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center (use clamped coords)
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)

        voxel_feats, voxel_coors_out = None, None
        low_level_point_feature = features
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)

            if (i == len(self.vfe_layers) - 1 and self.fusion_layer is not None
                    and img_feats is not None):
                point_feats = self.fusion_layer(img_feats, points, point_feats, img_metas)

            # Use DynamicScatterMean instead of MMCV DynamicScatter to avoid CUDA backward crash
            voxel_feats, voxel_coors_out, vfe_inverse = DynamicScatterMean(
                point_feats, coors,
                (self.vx, self.vy, self.vz),
                self.point_cloud_range,
                return_inverse=True
            )

            if i != len(self.vfe_layers) - 1:
                # Use inverse indices for efficient point mapping instead of canvas lookup
                feat_per_point = voxel_feats[vfe_inverse]
                features = torch.cat([point_feats, feat_per_point], dim=1)

        if self.return_point_feats:
            return point_feats

        if self.return_gt_points:
            return voxel_feats, voxel_coors_out, low_level_point_feature, coors

        return voxel_feats, voxel_coors_out