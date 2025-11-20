import torch
from torch import nn

from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
import copy
import warnings
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
import numpy as np
import torch
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils import digit_version
from mmcv.utils import ext_loader
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from .spatial_cross_attention import MSDeformableAttention3D
from mmdet3d.registry import MODELS, TRANSFORMS

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])
__all__ = ["VolumeTransform"]


@MODELS.register_module()
class volumeEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, pc_range=None, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):

        super(volumeEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.pc_range = pc_range
        self.fp16_enabled = False
        
        self.init_weights()
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()

    @staticmethod
    def get_reference_points(H, W, Z, bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W, Z: spatial shape of volume.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        
        zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype,
                            device=device).view(Z, 1, 1).expand(Z, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                            device=device).view(1, 1, W).expand(Z, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                            device=device).view(1, H, 1).expand(Z, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(3, 0, 1, 2).flatten(1).permute(1, 0)
        ref_3d = ref_3d[None, None].repeat(bs, 1, 1, 1)
        return ref_3d



    def point_sampling(self, reference_points, pc_range,  img_metas):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            lidar2img = []
            for img_meta in img_metas:
                lidar2cam = torch.as_tensor(img_meta["lidar2cam"], dtype=torch.float32)
                # (2) Load cam2img intrinsics: [N, 3, 3]
                K = torch.as_tensor(img_meta["cam2img"], dtype=torch.float32)
                # (3) Convert cam2img (K) to 3×4 projection matrix: [N, 3, 4]
                pad_col = torch.zeros(K.size(0), 3, 1, dtype=torch.float32, device=K.device)
                cam2img = torch.cat([K, pad_col], dim=-1)  # [N, 3, 4]
                # (4) lidar2img = cam2img @ lidar2cam  -> [N, 3, 4]
                lidar2img_3x4 = torch.bmm(cam2img, lidar2cam)
                # (5) Make it 4×4 by appending a final row [0,0,0,1]
                bottom_row = torch.tensor(
                    [[[0.0, 0.0, 0.0, 1.0]]],
                    dtype=torch.float32,
                    device=lidar2img_3x4.device
                ).repeat(lidar2img_3x4.size(0), 1, 1)  # [N, 1, 1]
                lidar2img_4x4 = torch.cat([lidar2img_3x4, bottom_row], dim=1)  # [N, 4, 4]
                lidar2img.append(lidar2img_4x4)

            lidar2img = torch.stack(lidar2img, dim=0).to(reference_points.device)

            #lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
            reference_points = reference_points.clone()

            reference_points[..., 0:1] = reference_points[..., 0:1] * \
                (pc_range[3] - pc_range[0]) + pc_range[0]
            reference_points[..., 1:2] = reference_points[..., 1:2] * \
                (pc_range[4] - pc_range[1]) + pc_range[1]
            reference_points[..., 2:3] = reference_points[..., 2:3] * \
                (pc_range[5] - pc_range[2]) + pc_range[2]

            reference_points = torch.cat(
                (reference_points, torch.ones_like(reference_points[..., :1])), -1)

            reference_points = reference_points.permute(1, 0, 2, 3)
            D, B, num_query = reference_points.size()[:3]
            
            num_cam = lidar2img.size(1)

            reference_points = reference_points.view(
                D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

            lidar2img = lidar2img.view(
                1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

            reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                                reference_points.to(torch.float32)).squeeze(-1)
            eps = 1e-5

            volume_mask = (reference_points_cam[..., 2:3] > eps)
            reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
                reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
            reference_points_cam[..., 0] /= img_metas[0]['img_shape'][1] # TODO HERE STOP
            reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0]

            volume_mask = (volume_mask & (reference_points_cam[..., 1:2] > 0.0)
                        & (reference_points_cam[..., 1:2] < 1.0)
                        & (reference_points_cam[..., 0:1] < 1.0)
                        & (reference_points_cam[..., 0:1] > 0.0))
            if digit_version(TORCH_VERSION) >= digit_version('1.8'):
                volume_mask = torch.nan_to_num(volume_mask)
            else:
                volume_mask = volume_mask.new_tensor(
                    np.nan_to_num(volume_mask.cpu().numpy()))

            reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4) #num_cam, B, num_query, D, 3
            volume_mask = volume_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

            return reference_points_cam, volume_mask

    def forward(self,
                volume_query,
                key,
                value,
                *args,
                volume_h=None,
                volume_w=None,
                volume_z=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            volume_query (Tensor): Input 3D volume query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = volume_query
        intermediate = []

        ref_3d = self.get_reference_points(
                    volume_h, volume_w, volume_z, bs=volume_query.size(1),  device=volume_query.device, dtype=volume_query.dtype)

        reference_points_cam, volume_mask = self.point_sampling(
            ref_3d, self.pc_range, kwargs['img_metas'])


        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        volume_query = volume_query.permute(1, 0, 2)

        for lid, layer in enumerate(self.layers):
            output = layer(
                volume_query,
                key,
                value,
                *args,
                ref_3d=ref_3d,
                volume_h=volume_h,
                volume_w=volume_w,
                volume_z=volume_z,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=volume_mask,
                **kwargs)

            volume_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@MODELS.register_module()
class volumeLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 embed_dims,
                 ffn_dropout=0.0,
                 operation_order=None,
                 conv_num=1,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(volumeLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            embed_dims=embed_dims,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False

        self.deblock = nn.ModuleList()
        conv_cfg=dict(type='Conv3d', bias=False)
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
        for i in range(conv_num):
            conv_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=embed_dims,
                    out_channels=embed_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1)
            deblock = nn.Sequential(conv_layer,
                                    build_norm_layer(norm_cfg, embed_dims)[1],
                                    nn.ReLU(inplace=True))
            self.deblock.append(deblock)
        #assert len(operation_order) == 6
        #assert set(operation_order) == set(
        #    ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_3d=None,
                volume_h=None,
                volume_w=None,
                volume_z=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'conv':
                bs = query.shape[0]
                identity = query
                query = query.reshape(bs, volume_z, volume_h, volume_w, -1).permute(0, 4, 3, 2, 1)
                for i in range(len(self.deblock)):
                    query = self.deblock[i](query)
                query = query.permute(0, 4, 3, 2, 1).reshape(bs, volume_z*volume_h*volume_w, -1)
                query = query + identity
    
            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1
            

        return query



@MODELS.register_module()
class VolumeTransform(nn.Module):
    def __init__(
        self,
        volume_h,
        volume_w,
        volume_z,
        embed_dims,
        in_channels,
        volume_encoder,
        mask_ratio,
    ) -> None:
        super(VolumeTransform, self).__init__()
        
        self.mask_token = None
        if mask_ratio > 0:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, in_channels))
        
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z
        self.volume_embedding = nn.Embedding(
            self.volume_h* self.volume_w* self.volume_z, 
            embed_dims
        )
        
        self.transfer_conv = nn.Sequential(
            build_conv_layer(
                dict(type='Conv2d', bias=True),
                in_channels=in_channels,
                out_channels=embed_dims,
                kernel_size=1,
                stride=1
            ), 
            nn.ReLU(inplace=True)
        )
        self.level_embeds = nn.Parameter(torch.Tensor(1, embed_dims))
        self.cams_embeds = nn.Parameter(torch.Tensor(6, embed_dims))
        
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
        upsample_cfg=dict(type='deconv3d', bias=False)

        self.deblock_camera = nn.Sequential(
            build_upsample_layer(
                upsample_cfg,
                in_channels=256,
                out_channels=256,
                kernel_size=2,
                stride=2
            ),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(inplace=True),
            build_conv_layer(
                dict(type='Conv3d', bias=False),
                in_channels=256,
                out_channels=192,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            build_norm_layer(norm_cfg, 192)[1],
            nn.ReLU(inplace=True)
        )
        
        self.volume_encoder = build_transformer_layer_sequence(volume_encoder)

        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.level_embeds)
        nn.init.normal_(self.cams_embeds)
        
        if self.mask_token is not None:
            nn.init.normal_(self.mask_token, std=.02)
        
    def forward(
        self, 
        camera_x,
        img_shape,
        camera_ids_restore,
        img_metas
    ):
        B, N, C, H, W = img_shape
        if self.mask_token is not None:
            mask_tokens = self.mask_token.repeat(camera_x.shape[0], camera_ids_restore.shape[1] - camera_x.shape[1], 1)
            camera_x = torch.cat([camera_x, mask_tokens], dim=1)

        camera_x = torch.gather(camera_x, dim=1, index=camera_ids_restore.unsqueeze(-1).repeat(1, 1, camera_x.shape[2]))  # unshuffle
        camera_x = camera_x.permute(0, 2, 1).view(B, N, -1, H//32, W//32)
        
        B, N, C, H, W = camera_x.shape
        dtype = camera_x.dtype
        volume_queries = self.volume_embedding.weight.to(dtype)
        
        volume_queries = volume_queries.unsqueeze(1).repeat(1, B, 1)
        view_features = self.transfer_conv(camera_x.view(B*N, C, H, W))
        view_features = view_features.view(B, N, -1, H, W).flatten(3).permute(1, 0, 3, 2)
        view_features = view_features + self.cams_embeds[:, None, None, :].to(view_features.dtype)
        view_features = view_features + self.level_embeds[None, None, 0:1, :].to(view_features.dtype)
        spatial_shapes = torch.as_tensor([[H, W]], dtype=torch.long, device=view_features.device)
        level_start_index = spatial_shapes.new_zeros((1,))
        view_features = view_features.permute(0, 2, 1, 3)
        
        volume_embed = self.volume_encoder(
            volume_queries,
            view_features,
            view_features,
            volume_h=self.volume_h,
            volume_w=self.volume_w,
            volume_z=self.volume_z,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            img_metas=img_metas
        ).reshape(B, self.volume_z, self.volume_h, self.volume_w, -1).permute(0, 4, 3, 2, 1)
        
        volume_embed = self.deblock_camera(volume_embed)

        return volume_embed, camera_x
 