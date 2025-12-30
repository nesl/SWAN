# Copyright (c) OpenMMLab. All rights reserved.
# modified Shir-Kang Scott Jin
from functools import partial
from itertools import chain
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import numpy as np

from mmengine.model import BaseModule, ModuleList, Sequential
from mmcv.cnn.bricks import DropPath
from mmdet3d.registry import MODELS
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

# Attempt to import spconv
try:
    import spconv.pytorch as spconv
    from spconv.pytorch import SparseConvTensor
except ImportError:
    spconv = None
    SparseConvTensor = None


# ============================================================================
#  Provided Modules (GRN, LayerNorm2d, build_norm_layer)
# ============================================================================

@MODELS.register_module()
class GRN(nn.Module):
    """Global Response Normalization Module.

    Come from `ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
    Autoencoders <http://arxiv.org/abs/2301.00808>`_

    Args:
        in_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-6.
    """

    def __init__(self, in_channels, eps=1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.zeros(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor, data_format='channel_first'):
        """Forward method.

        Args:
            x (torch.Tensor): The input tensor.
            data_format (str): The format of the input tensor. If
                ``"channel_first"``, the shape of the input tensor should be
                (B, C, H, W). If ``"channel_last"``, the shape of the input
                tensor should be (B, H, W, C). Defaults to "channel_first".
        """
        if data_format == 'channel_last':
            gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
            nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
            x = self.gamma * (x * nx) + self.beta + x
        elif data_format == 'channel_first':
            gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
            x = self.gamma.view(1, -1, 1, 1) * (x * nx) + self.beta.view(
                1, -1, 1, 1) + x
        return x


@MODELS.register_module('LN2d')
class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.

    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x, data_format='channel_first'):
        """Forward method.

        Args:
            x (torch.Tensor): The input tensor.
            data_format (str): The format of the input tensor. If
                ``"channel_first"``, the shape of the input tensor should be
                (B, C, H, W). If ``"channel_last"``, the shape of the input
                tensor should be (B, H, W, C). Defaults to "channel_first".
        """
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got tensor with shape {x.shape}'
        if data_format == 'channel_last':
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
        elif data_format == 'channel_first':
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias,
                             self.eps)
            # If the output is discontiguous, it may cause some unexpected
            # problem in the downstream tasks
            x = x.permute(0, 3, 1, 2).contiguous()
        return x


def build_norm_layer(cfg: dict, num_features: int) -> nn.Module:
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.

        num_features (int): Number of input channels.

    Returns:
        nn.Module: The created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    norm_layer = MODELS.get(layer_type)
    if norm_layer is None:
        raise KeyError(f'Cannot find {layer_type} in registry under scope '
                       f'name {MODELS.scope}')

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)

    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
    else:
        layer = norm_layer(num_channels=num_features, **cfg_)

    if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
        layer._specify_ddp_gpu_num(1)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return layer


# ============================================================================
#  Sparse Helper Modules
# ============================================================================

class SparseLayerNorm(nn.Module):
    """LayerNorm that operates on spconv.SparseConvTensor.features."""
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        # x: SparseConvTensor
        # Apply LayerNorm to the feature dimension (N_active, C)
        x = x.replace_feature(self.ln(x.features))
        return x

class SparseGRN(nn.Module):
    """Global Response Normalization for SparseConvTensor.
    Replicates the logic of GRN for 'channel_first' but handling sparse features.
    """
    def __init__(self, in_channels, eps=1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.zeros(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
        self.eps = eps

    def forward(self, x):
        # x: SparseConvTensor
        # features: (N_points, C)
        # indices: (N_points, 4) -> [batch_idx, z, y, x] (3d) or [batch_idx, y, x] (2d)
        
        batch_idx = x.indices[:, 0]
        batch_size = x.batch_size
        
        # 1. Compute L2 norm across spatial dimensions for each channel, per batch sample.
        #    gx_i = ||X_i||_2
        #    We need to aggregate (x.features^2) by batch_idx.
        
        sq_feat = x.features.pow(2) # (N_points, C)
        
        # Prepare container for summed squares: (Batch_Size, C)
        gx_sq = torch.zeros((batch_size, x.features.shape[1]), 
                            device=x.features.device, dtype=x.features.dtype)
        
        # Sum squares based on batch index
        # indices are int32, need long for indexing
        gx_sq.index_add_(0, batch_idx.long(), sq_feat)
        
        gx = torch.sqrt(gx_sq) # (B, C) represent global spatial norm per channel
        
        # 2. Normalize gx
        #    nx = gx / (mean(gx) + eps)
        #    mean taken over channels (dim=1)
        mx = gx.mean(dim=1, keepdim=True) # (B, 1)
        nx = gx / (mx + self.eps) # (B, C)
        
        # 3. Broadcast nx back to each point based on batch_idx
        nx_expanded = nx[batch_idx.long()] # (N_points, C)
        
        # 4. Apply affine transformation
        #    result = gamma * (x * nx) + beta + x
        out_features = self.gamma * (x.features * nx_expanded) + self.beta + x.features
        
        x = x.replace_feature(out_features)
        return x


# ============================================================================
#  Modified ConvNeXt Classes with Sparse Support
# ============================================================================

class ConvNeXtBlock(BaseModule):
    """ConvNeXt Block.

    Args:
        in_channels (int): The number of input channels.
        dw_conv_cfg (dict): Config of depthwise convolution.
        norm_cfg (dict): The config dict for norm layers.
        act_cfg (dict): The config dict for activation.
        mlp_ratio (float): The expansion ratio.
        linear_pw_conv (bool): Whether to use linear layer for pointwise conv.
        drop_path_rate (float): Stochastic depth rate.
        layer_scale_init_value (float): Init value for Layer Scale.
        use_grn (bool): Whether to use Global Response Normalization.
        with_cp (bool): Whether to use checkpointing.
        sparse (bool): Whether to use sparse convolutions.
    """

    def __init__(self,
                 in_channels,
                 dw_conv_cfg=dict(kernel_size=7, padding=3),
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=0,
                 use_grn=True, 
                 with_cp=False,
                 sparse=False):
        super().__init__()
        self.with_cp = with_cp
        self.sparse = sparse

        if self.sparse:
            assert spconv is not None, "spconv is not installed"
            # Depthwise convolution in Sparse: SubMConv2d with groups=in_channels
            # Submanifold conv keeps sparsity pattern same as input
            self.depthwise_conv = spconv.SubMConv2d(
                in_channels, 
                in_channels, 
                kernel_size=dw_conv_cfg['kernel_size'], 
                padding=dw_conv_cfg['padding'],
                groups=in_channels,
                bias=True, # ConvNeXt V2 typically uses bias
                indice_key=f'dw_{in_channels}' # Optional: share indices if helpful, usually unique per stage/block preferred?
            )
            # Sparse LayerNorm
            self.norm = SparseLayerNorm(in_channels, eps=norm_cfg.get('eps', 1e-6))
        else:
            self.depthwise_conv = nn.Conv2d(
                in_channels, in_channels, groups=in_channels, **dw_conv_cfg)
            self.norm = build_norm_layer(norm_cfg, in_channels)

        self.linear_pw_conv = linear_pw_conv
        mid_channels = int(mlp_ratio * in_channels)

        if self.sparse:
            # Pointwise convs are 1x1 SubMConvs
            self.pointwise_conv1 = spconv.SubMConv2d(in_channels, mid_channels, kernel_size=1, bias=True)
            self.pointwise_conv2 = spconv.SubMConv2d(mid_channels, in_channels, kernel_size=1, bias=True)
        else:
            if self.linear_pw_conv:
                pw_conv = nn.Linear
            else:
                pw_conv = partial(nn.Conv2d, kernel_size=1)
            self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
            self.pointwise_conv2 = pw_conv(mid_channels, in_channels)

        self.act = MODELS.build(act_cfg)

        if use_grn:
            if self.sparse:
                self.grn = SparseGRN(mid_channels)
            else:
                self.grn = GRN(mid_channels)
        else:
            self.grn = None

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, mask=None):
        """
        mask: (b, 1, 1, 1) whether to mask, if True return the input
        """
        
        # --- Sparse Path ---
        if self.sparse:
            def _inner_forward_sparse(x):
                identity = x
                # 1. Depthwise (SubM)
                x = self.depthwise_conv(x)
                
                # 2. Norm
                x = self.norm(x)
                
                # 3. PW Conv 1
                x = self.pointwise_conv1(x)
                
                # 4. Act (Apply to features)
                x = x.replace_feature(self.act(x.features))
                
                # 5. GRN
                if self.grn is not None:
                    x = self.grn(x)
                
                # 6. PW Conv 2
                x = self.pointwise_conv2(x)
                
                # 7. Gamma / Scale
                if self.gamma is not None:
                    x = x.replace_feature(x.features.mul(self.gamma))
                
                # 8. Drop Path & Residual
                # We add features. indices are guaranteed same for SubMConv.
                if self.drop_path.drop_prob > 0.:
                    x = x.replace_feature(self.drop_path(x.features))
                
                out = x.replace_feature(identity.features + x.features)
                return out

            if self.with_cp and x.requires_grad:
                x = cp.checkpoint(_inner_forward_sparse, x)
            else:
                x = _inner_forward_sparse(x)
            return x

        # --- Dense Path ---
        if mask is not None and float(mask.max()) == 0.:
            return x
        
        def _inner_forward(x):
            shortcut = x
            x = self.depthwise_conv(x)

            if self.linear_pw_conv:
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x = self.norm(x, data_format='channel_last')
                x = self.pointwise_conv1(x)
                x = self.act(x)
                if self.grn is not None:
                    x = self.grn(x, data_format='channel_last')
                x = self.pointwise_conv2(x)
                x = x.permute(0, 3, 1, 2).contiguous()   # (N, H, W, C) -> (N, C, H, W)
            else:
                x = self.norm(x, data_format='channel_first')
                x = self.pointwise_conv1(x)
                x = self.act(x)

                if self.grn is not None:
                    x = self.grn(x, data_format='channel_first') 
                x = self.pointwise_conv2(x)

            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1))

            res = self.drop_path(x)

            if mask is not None:
                res = res * mask 
            return shortcut + res

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


@MODELS.register_module()
class ConvNeXt(BaseModule):
    """ConvNeXt v1&v2 backbone.
    ...
    Args:
        ...
        sparse (bool): Whether to use sparse convolutions. Defaults to False.
    """  
    arch_settings = {
        'atto': {'depths': [2, 2, 6, 2], 'channels': [40, 80, 160, 320]},
        'femto': {'depths': [2, 2, 6, 2], 'channels': [48, 96, 192, 384]},
        'pico': {'depths': [2, 2, 6, 2], 'channels': [64, 128, 256, 512]},
        'nano': {'depths': [2, 2, 8, 2], 'channels': [80, 160, 320, 640]},
        'tiny': {'depths': [3, 3, 9, 3], 'channels': [96, 192, 384, 768]},
        'small': {'depths': [3, 3, 27, 3], 'channels': [96, 192, 384, 768]},
        'base': {'depths': [3, 3, 27, 3], 'channels': [128, 256, 512, 1024]},
        'large': {'depths': [3, 3, 27, 3], 'channels': [192, 384, 768, 1536]},
        'xlarge': {'depths': [3, 3, 27, 3], 'channels': [256, 512, 1024, 2048]},
        'huge': {'depths': [3, 3, 27, 3], 'channels': [352, 704, 1408, 2816]}
    }
    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 sparse=False,
                 init_cfg=[
                     dict(type='TruncNormal', layer=['Conv2d', 'Linear'], std=.02, bias=0.),
                     dict(type='Constant', layer=['LayerNorm'], val=1., bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)
        self.sparse = sparse
        
        if self.sparse:
            assert spconv is not None

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)
        self.total_blocks = sum(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        self.downsample_layers = ModuleList()
        
        # Stem Layer
        if self.sparse:
            # Sparse Stem
            stem = nn.Sequential(
                spconv.SparseConv2d(
                    in_channels,
                    self.channels[0],
                    kernel_size=stem_patch_size,
                    stride=stem_patch_size,
                    bias=True
                ),
                SparseLayerNorm(self.channels[0], eps=norm_cfg.get('eps', 1e-6)),
            )
        else:
            # Dense Stem
            stem = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.channels[0],
                    kernel_size=stem_patch_size,
                    stride=stem_patch_size),
                build_norm_layer(norm_cfg, self.channels[0]),
            )
        self.downsample_layers.append(stem)

        self.stages = ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                # Downsample Layer
                if self.sparse:
                    downsample_layer = nn.Sequential(
                        SparseLayerNorm(self.channels[i - 1], eps=norm_cfg.get('eps', 1e-6)),
                        spconv.SparseConv2d(
                            self.channels[i - 1],
                            channels,
                            kernel_size=2,
                            stride=2,
                            bias=True),
                    )
                else:
                    downsample_layer = nn.Sequential(
                        build_norm_layer(norm_cfg, self.channels[i - 1]),
                        nn.Conv2d(
                            self.channels[i - 1],
                            channels,
                            kernel_size=2,
                            stride=2),
                    )
                self.downsample_layers.append(downsample_layer)

            stage_blocks = ModuleList()
            for j in range(depth):
                stage_blocks.append(ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn, 
                    with_cp=with_cp,
                    sparse=self.sparse))
            
            self.stages.append(stage_blocks)
            block_idx += depth

            if i in self.out_indices:
                if self.sparse:
                    norm_layer = SparseLayerNorm(channels, eps=norm_cfg.get('eps', 1e-6))
                else:
                    norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt, self).train(mode)
        self._freeze_stages()

    def get_total_blocks(self):
        return self.total_blocks

    def forward(self, x, block_mask=None):
        outs = []
        global_block_idx = 0 

        for i, stage_blocks in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            
            for block in stage_blocks:
                current_mask = None
                if block_mask is not None and not self.sparse:
                    # In dense mode, we use the mask to zero out blocks
                    current_mask = block_mask[:, global_block_idx].view(-1, 1, 1, 1)
                
                # In sparse mode, block_mask is less standard to apply per block unless
                # we explicitly drop indices. Usually kept implicit.
                x = block(x, mask=current_mask)
                global_block_idx += 1

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.sparse:
                    # x is SparseConvTensor
                    # If global average pool is needed for classification, we would need to 
                    # aggregate. However, typical usage in detectors is returning features.
                    # We return the sparse tensor (or its features) here.
                    outs.append(norm_layer(x)) 
                else:
                    if self.gap_before_final_norm:
                        gap = x.mean([-2, -1], keepdim=True)
                        outs.append(norm_layer(gap).flatten(1))
                    else:
                        outs.append(norm_layer(x))

        return tuple(outs)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """ Generate 2D sin-cos positional embedding. """
    grid_h = np.arange(grid_size[0], dtype=float)
    grid_w = np.arange(grid_size[1], dtype=float)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


@MODELS.register_module()
class MAEConvNeXtEncoder(BaseModule):
    """
    ConvNeXt V2 Encoder Wrapper with Sparse Support.
    """
    def __init__(self, img_size=224, in_chans=3, arch='tiny', 
                 mask_ratio=0.75, use_grn=True, drop_path_rate=0.2, 
                 sparse=False,
                 init_cfg=None, **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.mask_ratio = mask_ratio
        self.sparse = sparse
        
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = img_size

        # ConvNeXt stage 4 usually has stride 32
        self.output_stride = 32 
        self.grid_size = (self.img_size[0] // self.output_stride, 
                          self.img_size[1] // self.output_stride)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.encoder = ConvNeXt(
            arch=arch,
            in_channels=in_chans,
            use_grn=use_grn,
            drop_path_rate=drop_path_rate,
            out_indices=[-1], 
            gap_before_final_norm=False, 
            sparse=self.sparse,
            **kwargs
        )
        
        if isinstance(arch, str):
            self.embed_dim = ConvNeXt.arch_settings[arch]['channels'][-1]
        else:
            self.embed_dim = arch['channels'][-1]
        
        if init_cfg is None:
            self.init_weights()

    def init_weights(self):
        self.encoder.init_weights()
        super().init_weights()

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        """
        N, L = x.shape[0], self.num_patches 
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask.scatter_add_(1, ids_keep, torch.full([N, len_keep], fill_value=-1, dtype=mask.dtype, device=x.device))
        
        return mask, ids_restore, ids_shuffle

    def dense_to_sparse(self, x, mask):
        """
        Convert (B, C, H, W) image and (B, L) mask into a SparseConvTensor containing only kept pixels.
        """
        B, C, H, W = x.shape
        
        # 1. Upsample mask to (B, H, W)
        mask_img = mask.reshape(B, 1, self.grid_size[0], self.grid_size[1])
        mask_img = F.interpolate(mask_img, size=(H, W), mode='nearest').squeeze(1) # (B, H, W)
        
        # 2. Get indices where mask == 0 (kept)
        # indices: (N_kept, 3) -> [batch_idx, y, x]
        indices = torch.nonzero(mask_img == 0).int()
        
        # 3. Get features
        x_perm = x.permute(0, 2, 3, 1) # (B, H, W, C)
        features = x_perm[mask_img == 0] # (N_kept, C)
        
        # 4. Construct SparseConvTensor
        sp_x = spconv.SparseConvTensor(
            features=features,
            indices=indices,
            spatial_shape=[H, W],
            batch_size=B
        )
        return sp_x

    def sparse_to_dense_tokens(self, sp_x, ids_restore):
        """
        Convert sparse output back to dense tokens (B, L, C).
        Since sparse output has holes, we assume the dense tensor should be filled with zeros 
        where data is missing, and then reshuffled.
        """
        B = sp_x.batch_size
        C = sp_x.features.shape[1]
        H_feat, W_feat = self.grid_size
        
        # Prepare dense map (B, H, W, C)
        dense_map = torch.zeros((B, H_feat, W_feat, C), device=sp_x.features.device, dtype=sp_x.features.dtype)
        
        # Fill kept features
        b_idx = sp_x.indices[:, 0].long()
        y_idx = sp_x.indices[:, 1].long()
        x_idx = sp_x.indices[:, 2].long()
        
        # Guard against indices out of bound if stride calculation is off (though it shouldn't be)
        valid = (y_idx < H_feat) & (x_idx < W_feat)
        dense_map[b_idx[valid], y_idx[valid], x_idx[valid]] = sp_x.features[valid]
        
        # Flatten (B, L, C)
        dense_flat = dense_map.reshape(B, -1, C)
        
        return dense_flat

    def forward(self, x, camera_only=False, block_mask=None):
        B, C, H, W = x.shape
        
        # 1. Generate Mask
        mask, ids_restore, ids_shuffle = self.random_masking(x, self.mask_ratio)
        
        # --- Sparse Logic ---
        if self.sparse:
            # Create sparse tensor of kept tokens
            x_sparse = self.dense_to_sparse(x, mask)
            
            # Forward Backbone
            latent_out = self.encoder(x_sparse, block_mask=block_mask)
            latent_sp = latent_out[-1] # SparseConvTensor
            
            # Convert back to dense for alignment with standard MAE Decoder
            latent_full_zeros = self.sparse_to_dense_tokens(latent_sp, ids_restore)
            
            # Gather valid tokens:
            # We shuffle the dense representation (which has 0s at masked spots)
            # The 'ids_shuffle' puts kept tokens first.
            ids_shuffle_expanded = ids_shuffle.unsqueeze(-1).repeat(1, 1, latent_full_zeros.shape[-1])
            latent_shuffled = torch.gather(latent_full_zeros, dim=1, index=ids_shuffle_expanded)
            
            len_keep = int(self.num_patches * (1 - self.mask_ratio))
            latent_masked = latent_shuffled[:, :len_keep, :]
        
        # --- Dense Logic ---
        else:
            mask_img = mask.reshape(B, 1, self.grid_size[0], self.grid_size[1])
            mask_img = F.interpolate(mask_img, size=(H, W), mode='nearest')
            x_masked = x * (1 - mask_img)

            latent_out = self.encoder(x_masked, block_mask=block_mask)
            if isinstance(latent_out, (tuple, list)):
                latent = latent_out[-1]
            else:
                latent = latent_out
            
            latent = latent.flatten(2).transpose(1, 2)

            len_keep = int(self.num_patches * (1 - self.mask_ratio))
            ids_shuffle_expanded = ids_shuffle.unsqueeze(-1).repeat(1, 1, latent.shape[-1])
            latent_shuffled = torch.gather(latent, dim=1, index=ids_shuffle_expanded)
            latent_masked = latent_shuffled[:, :len_keep, :]

        if camera_only:
            return latent_masked, mask, ids_restore

        # Forward Full Image
        # For sparse mode, full image means inputting a sparse tensor with NO mask (all pixels active)
        if self.sparse:
            mask_zeros = torch.zeros_like(mask)
            x_full_sparse = self.dense_to_sparse(x, mask_zeros)
            latent_full_out = self.encoder(x_full_sparse, block_mask=block_mask)
            latent_full_sp = latent_full_out[-1]
            latent_full = self.sparse_to_dense_tokens(latent_full_sp, ids_restore)
        else:
            latent_full_out = self.encoder(x, block_mask=block_mask)
            if isinstance(latent_full_out, (tuple, list)):
                latent_full = latent_full_out[-1]
            else:
                latent_full = latent_full_out
            latent_full = latent_full.flatten(2).transpose(1, 2)
        
        return latent_masked, latent_full, mask, ids_restore


@MODELS.register_module()
class MAEConvNeXtDecoder(BaseModule):
    """
    ConvNeXt V2 Decoder wrapper. 
    (Standard Dense Decoder, as reconstruction usually happens in dense space)
    """
    def __init__(self, num_patches, patch_size=32, in_chans=3,
                 embed_dim=768, decoder_embed_dim=512, decoder_depth=1, 
                 decoder_num_heads=16, use_grn=True, norm_pix_loss=False, 
                 decoder=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.grid_size = num_patches
        self.in_chans = in_chans

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim) if embed_dim != decoder_embed_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches[0] * num_patches[1], decoder_embed_dim), requires_grad=False
        )

        # Decoder uses dense ConvNeXtBlocks always
        self.decoder_blocks = nn.ModuleList([
            ConvNeXtBlock(
                in_channels=decoder_embed_dim, 
                dw_conv_cfg=dict(kernel_size=7, padding=3), 
                use_grn=use_grn,
                with_cp=False,
                sparse=False # Decoder is dense
            ) for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, (patch_size**2) * in_chans)

        self.cross_modality_module = None 
        if decoder is not None:
            self.cross_modality_module = build_transformer_layer_sequence(decoder)
            self.level_start_index = nn.Parameter(torch.as_tensor((0), dtype=torch.long), requires_grad=False)
            self.valid_ratios = nn.Parameter(torch.tensor([[[1., 1.]]], dtype=torch.float), requires_grad=False)
            self.cross_pos_embed = nn.Parameter(torch.randn(num_patches[0] * num_patches[1], 1, decoder_embed_dim))
            self.reference_camera = nn.Linear(decoder_embed_dim, 2)
            self.lidar2token = nn.Conv2d(128, decoder_embed_dim, kernel_size=1)

        self.norm_pix_loss = norm_pix_loss
        
        if init_cfg is None:
            self.init_weights()

    def init_weights(self):
        if self.cross_modality_module is None:
            pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches, cls_token=False)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights_impl)
        super().init_weights()

    def _init_weights_impl(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        p = self.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        p = self.patch_size
        h = self.grid_size[0]
        w = self.grid_size[1]
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def forward(self, x, ids_restore, lidar_x=None):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        if lidar_x is not None:
            _, _, H, W = lidar_x.shape
            lidar_x = self.lidar2token(lidar_x).flatten(-2).permute(0, 2, 1)
            lidar_x = torch.cat([i.repeat(6, 1, 1) for i in lidar_x]) 
            spatial_shapes = torch.as_tensor([(H, W)], dtype=torch.long, device=lidar_x.device)
            valid_ratios = self.valid_ratios.repeat(x.shape[0], 1, 1)
            x_cross = x_ + self.cross_pos_embed.transpose(0, 1)
            reference_point = self.reference_camera(x_cross).sigmoid()
            x_c, _ = self.cross_modality_module(
                query=x_cross.permute(1, 0, 2),
                value=lidar_x.permute(1, 0, 2),
                spatial_shapes=spatial_shapes,
                reference_points=reference_point,
                level_start_index=self.level_start_index,
                valid_ratios=valid_ratios
            )
            x_ = x_c.permute(1, 0, 2)
        else:
            x_ = x_ + self.decoder_pos_embed

        B, L, C = x_.shape
        H_grid, W_grid = self.grid_size
        x_2d = x_.transpose(1, 2).reshape(B, C, H_grid, W_grid)
        
        for blk in self.decoder_blocks:
            x_2d = blk(x_2d)
            
        x_ = x_2d.flatten(2).transpose(1, 2)
        x_ = self.decoder_norm(x_)
        x_ = self.decoder_pred(x_)
        return x_

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss