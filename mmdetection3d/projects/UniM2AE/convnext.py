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
from .norm import GRN, LayerNorm2d, build_norm_layer

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
from .norm import GRN, LayerNorm2d, build_norm_layer



class ConvNeXtBlock(BaseModule):
    """ConvNeXt Block.

    Args:
        in_channels (int): The number of input channels.
        dw_conv_cfg (dict): Config of depthwise convolution.
            Defaults to ``dict(kernel_size=7, padding=3)``.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        mlp_ratio (float): The expansion ratio in both pointwise convolution.
            Defaults to 4.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. More details can be found in the note.
            Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
        use_grn (bool): Whether to use Global Response Normalization (V2 config).
        with_cp (bool): Whether to use checkpointing to save memory.
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
                 use_grn=True, # V2 Flag
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, groups=in_channels, **dw_conv_cfg)

        self.linear_pw_conv = linear_pw_conv
        self.norm = build_norm_layer(norm_cfg, in_channels)

        mid_channels = int(mlp_ratio * in_channels)
        if self.linear_pw_conv:
            # Use linear layer to do pointwise conv.
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)

        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = MODELS.build(act_cfg)
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)

        if use_grn:
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
        mask: (b, 1, 1, 1)wether to mask, if True return the input
        * This is temp implementation of the drop behavior
        """

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

    A PyTorch implementation of `A ConvNet for the 2020s
    <https://arxiv.org/abs/2201.03545>`_ and
    `ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
    <http://arxiv.org/abs/2301.00808>`_

    Modified from the `official repo
    <https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py>`_
    and `timm
    <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py>`_.

    To use ConvNeXt v2, please set ``use_grn=True`` and ``layer_scale_init_value=0.``.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``ConvNeXt.arch_settings``. And if dict, it
            should include the following two keys:

            - depths (list[int]): Number of blocks at each stage.
            - channels (list[int]): The number of channels at each stage.

            Defaults to 'tiny'.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_patch_size (int): The size of one patch in the stem layer.
            Defaults to 4.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. Defaults to True.
        use_grn (bool): Whether to add Global Response Normalization in the
            blocks. Defaults to False.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        gap_before_final_norm (bool): Whether to globally average the feature
            map before the final norm layer. In the official repo, it's only
            used in classification task. Defaults to True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): Initialization config dict
    """  # noqa: E501
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
    }
    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=True, # [v2]
                 drop_path_rate=0.,
                 layer_scale_init_value=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

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
        self.total_blocks = sum(self.depths) # This is for future dropping behavior to keep track of total blocks

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

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            #  Use ModuleList explicitly to iterate easily with masks
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
                    with_cp=with_cp))
            
            self.stages.append(stage_blocks)
            block_idx += depth

            if i in self.out_indices:
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
        # Return the total number of blocks across all stages"
        return self.total_blocks

    def forward(self, x, block_mask=None):
        """
        use get_total_blocks to check how many blocks we have 
        Added 'block_mask' for dropping.
        block_mask (Tensor, optional): (Batch, Total_Blocks) of 0s and 1s.
        """
        outs = []
        global_block_idx = 0 # [ADDED] To track position in flattened mask

        for i, stage_blocks in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            
            # Iterate through blocks and apply specific mask bit
            for block in stage_blocks:
                current_mask = None
                if block_mask is not None:
                    # Extract (B, 1) -> Broadcast to (B, 1, 1, 1)
                    current_mask = block_mask[:, global_block_idx].view(-1, 1, 1, 1)
                
                x = block(x, mask=current_mask)
                global_block_idx += 1
            # 3. Output features
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
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
    ConvNeXt V2 Encoder Wrapper
    """
    def __init__(self, img_size=224, in_chans=3, arch='tiny', 
                 mask_ratio=0.75, use_grn=True, drop_path_rate=0.2, 
                 init_cfg=None, **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.mask_ratio = mask_ratio
        
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = img_size

        # ConvNeXt stage 4 usually has stride 32
        self.output_stride = 32 
        self.grid_size = (self.img_size[0] // self.output_stride, 
                          self.img_size[1] // self.output_stride)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Backbone
        self.encoder = ConvNeXt(
            arch=arch,
            in_channels=in_chans,
            use_grn=use_grn,
            drop_path_rate=drop_path_rate,
            out_indices=[-1], # FCMAE uses the final feature map
            gap_before_final_norm=False, 
            **kwargs
        )
        
        if isinstance(arch, str):
            self.embed_dim = ConvNeXt.arch_settings[arch]['channels'][-1]
        else:
            self.embed_dim = arch['channels'][-1]
        
        # Initialize weights if no config provided
        if init_cfg is None:
            self.init_weights()

    def init_weights(self):
        self.encoder.init_weights()
        super().init_weights()

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
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

    def forward(self, x, camera_only=False, block_mask=None):
        """
        Args:
            x: Input images (B, C, H, W)
            block_mask: ADMN mask tensor (B, Total_Blocks)
        """
        B, C, H, W = x.shape
        
        # 1. Generate Mask
        mask, ids_restore, ids_shuffle = self.random_masking(x, self.mask_ratio)
        
        # Upsample mask to image size to mask pixels
        mask_img = mask.reshape(B, 1, self.grid_size[0], self.grid_size[1])
        mask_img = F.interpolate(mask_img, size=(H, W), mode='nearest')
        
        # Apply mask to input image (0 out masked pixels)
        x_masked = x * (1 - mask_img)

        latent_out = self.encoder(x_masked, block_mask=block_mask)
        
        if isinstance(latent_out, (tuple, list)):
            latent = latent_out[-1]
        else:
            latent = latent_out
        
        # Flatten: (B, C, H, W) -> (B, L, C)
        latent = latent.flatten(2).transpose(1, 2)

        # ilter out masked tokens
        len_keep = int(self.num_patches * (1 - self.mask_ratio))
        ids_shuffle_expanded = ids_shuffle.unsqueeze(-1).repeat(1, 1, latent.shape[-1])
        latent_shuffled = torch.gather(latent, dim=1, index=ids_shuffle_expanded)
        latent_masked = latent_shuffled[:, :len_keep, :]

        if camera_only:
            return latent_masked, mask, ids_restore

        #  Forward Full Image 
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
        
        # Fixed Sin-Cos Pos Embed
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches[0] * num_patches[1], decoder_embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList([
            ConvNeXtBlock(
                in_channels=decoder_embed_dim, 
                dw_conv_cfg=dict(kernel_size=7, padding=3), 
                use_grn=use_grn,
                with_cp=False
            ) for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, (patch_size**2) * in_chans)

        # Cross Modality Module (Transformer-based)
        self.cross_modality_module = None 
        if decoder is not None:
            self.cross_modality_module = build_transformer_layer_sequence(decoder)
            self.level_start_index = nn.Parameter(torch.as_tensor((0), dtype=torch.long), requires_grad=False)
            self.valid_ratios = nn.Parameter(torch.tensor([[[1., 1.]]], dtype=torch.float), requires_grad=False)
            self.cross_pos_embed = nn.Parameter(torch.randn(num_patches[0] * num_patches[1], 1, decoder_embed_dim))
            self.reference_camera = nn.Linear(decoder_embed_dim, 2)
            self.lidar2token = nn.Conv2d(128, decoder_embed_dim, kernel_size=1)

        self.norm_pix_loss = norm_pix_loss
        
        # Manually trigger init if no init_cfg provided
        if init_cfg is None:
            self.init_weights()

    def init_weights(self):
        """Initialize weights for decoder."""
        # Initialize Positional Embeddings
        if self.cross_modality_module is None:
            pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches, cls_token=False)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Mask Token
        torch.nn.init.normal_(self.mask_token, std=.02)

        # Apply standard init
        self.apply(self._init_weights_impl)
        
        # Parent init (handles init_cfg)
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
        """
        imgs: (N, 3, H, W) -> x: (N, L, patch_size**2 * 3)
        """
        p = self.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * 3) -> imgs: (N, 3, H, W)
        """
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