# Modified to support MAE-style masking while maintaining weight compatibility
# with MMDetection's SwinTransformer implementation.
from functools import partial
from collections import OrderedDict
from copy import deepcopy
import warnings
from typing import Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmdet.models.layers import PatchEmbed
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import constant_init, trunc_normal_, trunc_normal_init
from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.logging import MMLogger
from mmengine.utils import to_2tuple


from mmdet3d.registry import MODELS
from timm.layers import DropPath, drop_path
from timm.models.vision_transformer import Block
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

"""
reference https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/backbones/swin.py\
### Potential solutions###
It might be beneficial to just overload the function?
### Changes Summary Note###
Mlp class is replaced by FFN from mmcv to maintain compatibility.
get_coordinates is merged into WindowMSA class as a static method.
SwinTransformerBlock is renamed to SwinBlock
BasicLayer is renamed to SwinBlockSequence
PatchEmbed is removed in favor for mmdetection version however we lost the flops
flops are dangerous
"""

# Mlp is replaced

# get_coordinates is merged into WindowMSA with the function double_step_seq

class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias. Originally called WindowAttention.

    Supports both standard attention and MAE-style masked attention.
    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Relative position index
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, pos_idx=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: attention mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
            pos_idx: position index for MAE masking (optional)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Handle relative position bias
        if pos_idx is not None and mask is not None:
            # MAE-style masking with custom position indices
            assert pos_idx.dim() == 3
            rel_pos_mask = torch.masked_fill(torch.ones_like(mask), mask=mask.bool(), value=0.0)
            pos_idx_m = torch.masked_fill(pos_idx, mask.bool(), value=0).view(-1)
            relative_position_bias = self.relative_position_bias_table[pos_idx_m].view(
                -1, N, N, self.num_heads)
            relative_position_bias = relative_position_bias * rel_pos_mask.view(-1, N, N, 1)
            nW = relative_position_bias.shape[0]
            relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous()
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + relative_position_bias.unsqueeze(0)
            attn = attn + mask.view(1, nW, 1, N, N)
            attn = attn.view(B, self.num_heads, N, N)
        else:
            # Standard attention
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1],
                    self.window_size[0] * self.window_size[1], 
                    -1)
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

            # attention mask
            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)

        # normalization
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        # aggregation
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        # Repalce get_coordinates function
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)

    def extra_repr(self) -> str:
        return f'dim={self.embed_dims}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.embed_dims * 3 * self.embed_dims
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.embed_dims // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.embed_dims // self.num_heads)
        # x = self.proj(x)
        flops += N * self.embed_dims * self.embed_dims
        return flops



class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """
    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        if self.shift_size > 0:
            shifted_query = torch.roll(query, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        query_windows = self.window_partition(shifted_query)
        query_windows = query_windows.view(-1, self.window_size**2, C)

        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows



class SwinBlock(BaseModule):
    """
    Previously SwinTransformerBlock.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super(SwinBlock, self).__init__()
        self.init_cfg = init_cfg
        self.with_cp = with_cp
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0
        self.embed_dims = embed_dims
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=self.shift_size,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop_rate=attn_drop_rate,
                    proj_drop_rate=drop_rate,
                    dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                    init_cfg=None)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)
        
    def forward(self, x, hw_shape):
        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)
            x = self.drop_path(x) + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x
    
    def forward_masked(self, x, attn_mask, rel_pos_idx, group_block):
        """Forward pass with MAE-style masking."""
        identity = x
        x = self.norm1(x)
        x = group_block.group(x)
        # Use WindowMSA directly with custom mask/pos_idx
        x = self.attn.w_msa(x, mask=attn_mask, pos_idx=rel_pos_idx)
        x = self.attn.drop(x)   # attention drop handles drop on its own
        x = group_block.merge(x)
        x = x + identity

        identity = x
        x = self.norm2(x)
        x = self.ffn(x, identity=identity) #FFN also handles drop on its own
        return x


    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.embed_dims * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.embed_dims * self.embed_dims * self.mlp_ratio
        # norm2
        flops += self.embed_dims * H * W
        return flops

def get_coordinates(h, w, device='cpu'):
    """Generate coordinate grid."""
    coords_h = torch.arange(h, device=device)
    coords_w = torch.arange(w, device=device)
    #coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'), dim=-1)
    return coords


class PatchMerging(BaseModule):
    """ 
    This is not very matched with mmdetection3d due to the differnece in masking
     Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride for downsampling. Default: 2.
        norm_cfg (dict): Config dict for normalization layer. Default: LayerNorm.
        init_cfg (dict, optional): Initialization config. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=2,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.reduction = nn.Linear(4 * in_channels, out_channels, bias=False)
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, 4 * in_channels)[1]
        else:
            self.norm = None

    def forward(self, x,  input_size: Tuple[int]) -> Tuple[Tensor, Tuple[int]]:
        """Standard forward for detection."""
        H, W = input_size
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        assert H % 2 == 0 and W % 2 == 0, f'x size ({H}*{W}) are not even.'

        x = x.view(B, H, W, C)

        # Merge 2x2 patches
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        if self.norm is not None:
            x = self.norm(x)
        x = self.reduction(x)

        return x, (H // 2, W // 2)

    def forward_masked(self, x, mask_prev, input_size):
        """Forward with MAE-style masking.
        This is needed due to the multi scale encoder , rather than the single scale before, 
        the new mask will be needed in the decoder to know which patces are omitted 
        Args:
            x: Input features [B, L, C] (visible tokens only)
            mask_prev: Boolean mask [1, L] where True/1 = masked, False/0 = visible
            input_size: (H, W) tuple

        Returns:
            x: Output features
            coords_new: Coordinates at new resolution (visible tokens only)
            mask_new: Mask at new resolution (1 = masked, 0 = visible)
        """
        H, W = input_size
        B, L, C = x.shape
        assert H % 2 == 0 and W % 2 == 0

        # Gather patches in 2x2 local windows
        # Reshape mask to match local window structure
        mask_vis = mask_prev.reshape(H, W).float()
        #import matplotlib.pyplot as plt
        #plt.imsave(f'/data/HangQiu/proj/mlsys/DynMMOT_scripts/pretrain/visualize/{H}x{W}_mask_prev.png', mask_vis.cpu().numpy(), cmap='viridis')
        mask = mask_prev.reshape(H//2, 2, W//2, 2).permute(0, 2, 1, 3).reshape(-1)
        #print(mask_prev.shape)
        # consider that 
        #print(f"Mask sum before merging: {mask.sum().item()} out of {mask.numel()} shape: {mask.shape}")
        #print(mask)
        # plot mask
        # repalced coords = get_coordinates(H, W, device=x.device).reshape(2, -1).permute(1, 0)
        coords = get_coordinates(H, W, device=x.device).reshape(1, -1, 2)
        
        coords = coords.reshape(H//2, 2, W//2, 2, 2).permute(0, 2, 1, 3, 4).reshape(-1, 2)
        # Select visible coordinates (where mask is False/0, meaning not masked)
        coords_vis_local = coords[~mask].reshape(-1, 2)
        coords_vis_local = coords_vis_local[:, 0] * W + coords_vis_local[:, 1]
        idx_shuffle = torch.argsort(torch.argsort(coords_vis_local))

        x = torch.index_select(x, 1, index=idx_shuffle)
        x = x.reshape(B, L//4, 4, C)
        x = torch.cat([x[:, :, 0], x[:, :, 2], x[:, :, 1], x[:, :, 3]], dim=-1)

        if self.norm is not None:
            x = self.norm(x)
        x = self.reduction(x)

        # Compute mask at new resolution: a 2x2 region is masked only if all 4 patches were masked
        # Sum counts masked patches (0-4), mask_new = True if all 4 masked
        mask_sum = mask_prev.view(1, H//2, 2, W//2, 2).sum(dim=(2, 4))
        mask_new = (mask_sum == 4).reshape(1, -1)

        #coords_new = get_coordinates(H//2, W//2, x.device).reshape(1, 2, -1)
        # Select visible coordinates (where mask_new is False, meaning not masked)
        #coords_new = coords_new.transpose(2, 1)[~mask_new].reshape(1, -1, 2)
        # SHAPE Change need debug
        coords_new = get_coordinates(H//2, W//2, x.device).reshape(1, -1, 2)
        # Select visible coordinates (where mask_new is False, meaning not masked)
        coords_new = coords_new[~mask_new].reshape(1, -1, 2)

        return x, coords_new, mask_new



# this is for 
def knapsack(W, wt):
    '''Args:
        W (int): capacity
        wt (tuple[int]): the numbers of elements within each window
    '''
    val = wt
    n = len(val)
    K = [[0 for w in range(W + 1)]
            for i in range(n + 1)]
            
    # Build table K[][] in bottom up manner
    for i in range(n + 1):
        for w in range(W + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif wt[i - 1] <= w:
                K[i][w] = max(val[i - 1]
                + K[i - 1][w - wt[i - 1]],
                            K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]

    # stores the result of Knapsack
    res = res_ret = K[n][W]

    # stores the selected indexes
    w = W
    idx = []
    for i in range(n, 0, -1):
        if res <= 0:
            break
        # Either the result comes from the top (K[i-1][w]) 
        # or from (val[i-1] + K[i-1] [w-wt[i-1]]) as in Knapsack table.
        # If it comes from the latter one, it means the item is included.
        if res == K[i - 1][w]:
            continue
        else:
            # This item is included.
            idx.append(i - 1)
            # Since this weight is included, its value is deducted
            res = res - val[i - 1]
            w = w - wt[i - 1]
    
    return res_ret, idx[::-1]   # make the idx in an increasing order

def group_windows(group_size, num_ele_win):
    '''Greedily apply the DP algorithm to group the elements.
    Args:
        group_size (int): maximal size of the group
        num_ele_win (list[int]): number of visible elements of each window
    Outputs:
        num_ele_group (list[int]): number of elements of each group
        grouped_idx (list[list[int]]): the seleted indeices of each group
    '''
    wt = num_ele_win.copy()
    ori_idx = list(range(len(wt)))
    grouped_idx = []
    num_ele_group = []

    while len(wt) > 0:
        res, idx = knapsack(group_size, wt)
        num_ele_group.append(res)
        selected_ori_idx = [ori_idx[i] for i in idx]
        grouped_idx.append(selected_ori_idx)
        wt = [wt[i] for i in range(len(ori_idx)) if i not in idx]
        ori_idx = [ori_idx[i] for i in range(len(ori_idx)) if i not in idx]

    return num_ele_group, grouped_idx


class GroupingModule:
    """Handles window grouping for masked attention in MAE-style training."""

    def __init__(self, window_size, shift_size, group_size=None):
        self.window_size = window_size
        self.shift_size = shift_size
        assert shift_size >= 0 and shift_size < window_size
        self.group_size = group_size or self.window_size**2
        self.attn_mask = None
        self.rel_pos_idx = None

    def _get_group_id(self, coords):
        group_id = coords.clone()
        group_id += (self.window_size - self.shift_size) % self.window_size
        group_id = group_id // self.window_size
        group_id = group_id[0, :, 0] * group_id.shape[1] + group_id[0, :, 1]
        return group_id

    def _get_attn_mask(self, group_id):
        pos_mask = (group_id == -1)
        pos_mask = torch.logical_and(pos_mask[:, :, None], pos_mask[:, None, :])
        gid = group_id.float()
        attn_mask_float = gid.unsqueeze(2) - gid.unsqueeze(1)
        attn_mask = torch.logical_or(attn_mask_float != 0, pos_mask)
        attn_mask_float.masked_fill_(attn_mask, -100.)
        return attn_mask_float

    def _get_rel_pos_idx(self, coords):
        rel_pos_idx = coords[:, :, None, :] - coords[:, None, :, :]
        rel_pos_idx += self.window_size - 1
        rel_pos_idx[..., 0] *= 2 * self.window_size - 1
        rel_pos_idx = rel_pos_idx.sum(dim=-1)
        return rel_pos_idx

    def _prepare_masking(self, coords):
        group_id = self._get_group_id(coords)
        attn_mask = self._get_attn_mask(group_id.unsqueeze(0))
        rel_pos_idx = self._get_rel_pos_idx(coords[:1])
        self.idx_shuffle = None
        self.idx_unshuffle = None
        return attn_mask, rel_pos_idx

    def _prepare_grouping(self, coords):
        group_id = self._get_group_id(coords)
        idx_merge = torch.argsort(group_id)
        group_id = group_id[idx_merge].contiguous()
        exact_win_sz = torch.unique_consecutive(group_id, return_counts=True)[1].tolist()

        self.group_size = min(self.window_size**2, max(exact_win_sz))
        num_ele_group, grouped_idx = group_windows(self.group_size, exact_win_sz)

        idx_merge_spl = idx_merge.split(exact_win_sz)
        group_id_spl = group_id.split(exact_win_sz)
        shuffled_idx, attn_mask = [], []
        for num_ele, gidx in zip(num_ele_group, grouped_idx):
            pad_r = self.group_size - num_ele
            sidx = torch.cat([idx_merge_spl[i] for i in gidx], dim=0)
            shuffled_idx.append(F.pad(sidx, (0, pad_r), value=-1))
            amask = torch.cat([group_id_spl[i] for i in gidx], dim=0)
            attn_mask.append(F.pad(amask, (0, pad_r), value=-1))

        self.idx_shuffle = torch.cat(shuffled_idx, dim=0)
        self.idx_unshuffle = torch.argsort(self.idx_shuffle)[-sum(num_ele_group):]
        self.idx_shuffle[self.idx_shuffle==-1] = 0

        attn_mask = torch.stack(attn_mask, dim=0)
        attn_mask = self._get_attn_mask(attn_mask)

        coords_shuffled = coords[0][self.idx_shuffle].reshape(-1, self.group_size, 2)
        rel_pos_idx = self._get_rel_pos_idx(coords_shuffled)
        rel_pos_mask = torch.ones_like(rel_pos_idx).masked_fill_(attn_mask.bool(), 0)
        rel_pos_idx = rel_pos_idx * rel_pos_mask

        return attn_mask, rel_pos_idx

    def prepare(self, coords, mode):
        self._mode = mode
        if mode == 'masking':
            return self._prepare_masking(coords)
        elif mode == 'grouping':
            return self._prepare_grouping(coords)
        else:
            raise KeyError(f"Unknown mode: {mode}")

    def group(self, x):
        if self._mode == 'grouping':
            self.ori_shape = x.shape
            x = torch.index_select(x, 1, self.idx_shuffle)
            x = x.reshape(-1, self.group_size, x.shape[-1])
        return x

    def merge(self, x):
        if self._mode == 'grouping':
            B, N, C = self.ori_shape
            x = x.reshape(B, -1, C)
            x = torch.index_select(x, 1, self.idx_unshuffle)
        return x


class SwinBlockSequence(BaseModule):
    """Implements one stage in Swin Transformer.
       renames from BasicLayer.
       Mask generation is moved to encoder.
       necessary because the forward masking
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        self.dim = embed_dims
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.blocks.append(block)

        self.downsample = downsample
        self.window_size = window_size

    def forward(self, x, hw_shape, keep_layer_mask=None):
        """Standard forward for detection."""
        # Is this the correct way to use the keep_layer_mask?
        for i, block in enumerate(self.blocks):
            if keep_layer_mask is not None and keep_layer_mask[i]:
                continue
            x = block(x, hw_shape)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape

    def forward_masked(self, x, coords, mask, hw_shape,return_x_before_down=True, keep_layer_mask=None):
        """Forward with MAE-style masking.

        Args:
            x: Input features [B, L, C] (visible tokens only)
            coords: Coordinates of visible tokens
            mask: Boolean mask [1, L] where True/1 = masked, False/0 = visible
            hw_shape: (H, W) tuple

        Returns:
            x_before_down: Features before downsampling
            x_down: Features after downsampling
            coords: Updated coordinates
            mask: Updated mask (1 = masked)
        """
        mode = "masking" if x.shape[1] <= 2 * self.window_size**2 else "grouping"
        shift_size = self.window_size // 2

        group_block = GroupingModule(self.window_size, 0)
        attn_mask, pos_idx = group_block.prepare(coords, mode)

        if self.window_size < min(hw_shape):
            group_block_shift = GroupingModule(self.window_size, shift_size)
            attn_mask_shift, pos_idx_shift = group_block_shift.prepare(coords, mode)
        else:
            group_block_shift = group_block
            attn_mask_shift, pos_idx_shift = attn_mask, pos_idx

        for i, blk in enumerate(self.blocks):
            # Do not run the block if this is 0
            if keep_layer_mask is not None and keep_layer_mask[i]:
                continue
            gblk = group_block if i % 2 == 0 else group_block_shift
            cur_attn_mask = attn_mask if i % 2 == 0 else attn_mask_shift
            rel_pos_idx = pos_idx if i % 2 == 0 else pos_idx_shift
            x = blk.forward_masked(x, cur_attn_mask, rel_pos_idx, gblk)

        x_before_down = x
        if self.downsample is not None:
            x_down, coords, mask = self.downsample.forward_masked(x, mask, hw_shape)
        else:
            x_down = x

        if return_x_before_down:
            return x_before_down, x_down, coords, mask
        else: 
            return x_down, coords, mask

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, window_size={self.window_size},"\
                f"shift_size={self.shift_size}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

@MODELS.register_module()
class SwinTransformer(BaseModule):
    """ Swin Transformer
    A PyTorch implement of : `Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows`  -
        https://arxiv.org/abs/2103.14030

    Inspiration from
    https://github.com/microsoft/Swin-Transformer

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): The num of input channels.
            Defaults: 3.
        embed_dims (int): The feature dimension. Default: 96.
        patch_size (int | tuple[int]): Patch size. Default: 4.
        window_size (int): Window size. Default: 7.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        depths (tuple[int]): Depths of each Swin Transformer stage.
            Default: (2, 2, 6, 2).
        num_heads (tuple[int]): Parallel attention heads of each Swin
            Transformer stage. Default: (3, 6, 12, 24).
        strides (tuple[int]): The patch merging or patch embedding stride of
            each Swin Transformer stage. (In swin, we set kernel size equal to
            stride.) Default: (4, 2, 2, 2).
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
            value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        patch_norm (bool): If add a norm layer for patch embed and patch
            merging. Default: True.
        drop_rate (float): Dropout rate. Defaults: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: False.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer at
            output of backone. Defaults: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            Default: -1 (-1 means not freezing any parameters).
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 init_cfg=None):

        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages

        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super(SwinTransformer, self).__init__(init_cfg=self.init_cfg)

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed
        self.num_layers = num_layers
        self.depths = depths
        self.embed_dims = embed_dims
        self.window_size = window_size

        assert strides[0] == patch_size

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=int(mlp_ratio * in_channels),
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]

        # Add norm layer for each output (MMDetection style)
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):
            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False
            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for {self.__class__.__name__}')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            # Not sure
            #if self.convert_weights:
            #   _state_dict = swin_converter(_state_dict)

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v

            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # Handle position embedding interpolation
            relative_position_bias_table_keys = [
                k for k in state_dict.keys() if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2), mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0).contiguous()

            self.load_state_dict(state_dict, strict=False)

    def forward(self, x, Mask=None, keep_layer_mask=None):
        if Mask!=None:
            return self.forward_masked(x, Mask, keep_layer_mask)
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)
        
        if keep_layer_mask:
            # We run torch.split to group it according to the depth blocks. 
            keep_layer_mask = torch.split(keep_layer_mask, self.depths)
        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape, keep_layer_mask[i] if keep_layer_mask else None)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs

    def forward_masked(self, x, mask, keep_layer_mask= None):
        """Forward with MAE masking .

        Args:
            x: Input images [B, C, H, W]
            mask: Boolean mask [B, L] where 1/True = masked, 0/False = visible
            keep_layer_mask:
        Returns:
            outs: List of 4 feature tensors from each stage (visible tokens only)
        """
        x, hw_shape = self.patch_embed(x)
        H, W = hw_shape

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        B, N, C = x.shape
        mask = mask[:1].clone()

        # Handle mask size mismatch
        up_ratio = N // mask.shape[1]
        assert up_ratio * mask.shape[1] == N
        num_repeats = int(up_ratio**0.5)
        if up_ratio > 1:
            Mh, Mw = H // num_repeats, W // num_repeats
            mask = mask.reshape(1, Mh, 1, Mw, 1)
            mask = mask.expand(-1, -1, num_repeats, -1, num_repeats)
            mask = mask.reshape(1, -1)

        # Get coordinates
        coords_h = torch.arange(H, device=x.device)
        coords_w = torch.arange(W, device=x.device)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]), dim=-1).reshape(1, H*W, 2)

        # Select visible patches (mask=1 means masked, so use ~mask to select visible)
        #print(f"x shape before masking: {x.shape}")
        x_vis = x[(~mask).expand(B, -1)].reshape(B, -1, C)
        #print(f"x_vis shape after masking: {x_vis.shape}")
        coords = coords[~mask].reshape(1, -1, 2)

        # Forward through stages
        # Note: mask convention is 1 = masked throughout
        outs = []
        input_resolution = (H, W)

        if keep_layer_mask:
            # We run torch.split to group it according to the depth blocks. 
            keep_layer_mask = torch.split(keep_layer_mask, self.depths)


        for i, stage in enumerate(self.stages):
            x_before, x_vis, coords, mask = stage.forward_masked(
                x_vis, coords, mask, input_resolution, True, keep_layer_mask[i] if keep_layer_mask else None)

            # Normalize and store output
            norm_layer = getattr(self, f'norm{i}')
            out = norm_layer(x_before)
            outs.append(out)

            # Update resolution for next stage
            if stage.downsample is not None:
                input_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)

        return outs



@MODELS.register_module()
class MAESwinEncoder(nn.Module):
    """Swin Transformer Encoder for Masked Autoencoders.

    Wraps SwinTransformer and adds MAE-specific functionality.
    Returns multi-scale features from all 4 stages.
    """

    def __init__(self,
                 img_size=(224, 224),
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 ape=False,
                 patch_norm=True,
                 mask_ratio=0.75,
                 **kwargs):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.depths = depths
        self.embed_dim = embed_dim

        img_size = to_2tuple(img_size)

        # Initialize the Swin Backbone (MMDetection compatible)
        self.encoder = SwinTransformer(
            pretrain_img_size=img_size,
            in_channels=in_chans,
            embed_dims=embed_dim,
            patch_size=patch_size,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            depths=depths,
            num_heads=num_heads,
            strides=(patch_size, 2, 2, 2),
            out_indices=(0, 1, 2, 3),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            patch_norm=patch_norm,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            use_abs_pos_embed=ape,
            norm_cfg=dict(type='LN', eps=1e-6),
            **kwargs
        )

        # Calculate patches at final resolution
        H, W = img_size[0] // patch_size, img_size[1] // patch_size
        for _ in range(len(depths) - 1):
            H, W = H // 2, W // 2
        self.num_patches = H * W
        self.final_resolution = (H, W)

        # Stage dimensions for decoder
        self.stage_dims = [embed_dim * (2 ** i) for i in range(len(depths))]

        self.initialize_weights()

    def initialize_weights(self):
        w = self.encoder.patch_embed.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def random_masking(self, x, mask_ratio):
        """Generate random mask for MAE training.

        Returns:
            mask: [N, L] where 1 = masked, 0 = visible
            ids_restore: indices to restore original order
            ids_shuffle: indices used for shuffling
        """
        N, L = x.shape[0], self.num_patches
        len_keep = int(L * (1 - mask_ratio))

        # Use same random mask for all samples in batch (consistent with encoder using mask[:1])
        noise = torch.rand(1, L, device=x.device).expand(N, -1)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Create mask: 1 = masked, 0 = visible
        mask = torch.ones(N, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask, ids_restore, ids_shuffle

    def forward(self, x, camera_only=False):
        """
        Forward pass.

        Returns:
            outs: List of 4 feature tensors from each stage
            mask: [B, L] mask tensor where 1 = masked, 0 = visible
            ids_restore: [B, L] restoration indices
        """
        mask, ids_restore, ids_shuffle = self.random_masking(x, self.mask_ratio)

        # Get multi-scale features with masking
        outs = self.encoder.forward_masked(x, mask.bool())

        return outs, mask, ids_restore

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Generate 2D sinusoidal positional embeddings."""
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
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


@MODELS.register_module()
class MAESwinDecoder(nn.Module):
    """MAE Decoder with progressive upsampling and multi-scale skip connections."""

    def __init__(self,
                 num_patches,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 mlp_ratio=4.,
                 decoder_embed_dim=512,
                 decoder_depth=2,
                 decoder_num_heads=16,
                 norm_pix_loss=True,
                 decoder=None):
        super().__init__()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.grid_size = num_patches  # (H, W) at stage 3
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_stages = len(depths)
        self.stage_dims = [embed_dim * (2 ** i) for i in range(self.num_stages)]

        # Stage 3: 10x25, Stage 2: 20x50, Stage 1: 40x100, Stage 0: 80x200
        self.stage_resolutions = [
            (num_patches[0] * (2 ** (self.num_stages - 1 - i)),
             num_patches[1] * (2 ** (self.num_stages - 1 - i)))
            for i in range(self.num_stages)
        ]

        L = self.grid_size[0] * self.grid_size[1]
        enc_dim = self.stage_dims[-1]  # 768

        # Initial projection and mask token at stage 3 resolution
        self.decoder_embed = nn.Linear(enc_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, L, decoder_embed_dim), requires_grad=False)

        # Initial transformer blocks at stage 3 resolution
        self.initial_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])

        # Lateral projections for skip connections (project encoder features to decoder dim)
        self.lateral_projs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, decoder_embed_dim),
            )
            for dim in self.stage_dims
        ])

        # Progressive upsampling: stage3 -> stage2 -> stage1 -> stage0
        # Each upsample stage: upsample + fuse skip + refine
        self.upsample_layers = nn.ModuleList()
        self.refine_blocks = nn.ModuleList()
        self.upsample_norms = nn.ModuleList()

        for _ in range(self.num_stages - 1):
            self.upsample_layers.append(
                nn.ConvTranspose2d(decoder_embed_dim, decoder_embed_dim, kernel_size=2, stride=2)
            )
            self.refine_blocks.append(
                Block(decoder_embed_dim, decoder_num_heads // 2, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            )
            self.upsample_norms.append(norm_layer(decoder_embed_dim))

        # Final prediction at stage 0 resolution (80x200) with 4x4 patches
        self.final_norm = norm_layer(decoder_embed_dim)
        self.final_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        # Cross-modality (optional)
        self.cross_modality_module = None
        if decoder is not None:
            self.cross_modality_module = build_transformer_layer_sequence(decoder)
            self.level_start_index = nn.Parameter(torch.as_tensor((0), dtype=torch.long), requires_grad=False)
            self.valid_ratios = nn.Parameter(torch.tensor([[[1., 1.]]], dtype=torch.float), requires_grad=False)
            self.decoder_pos_embed = nn.Parameter(torch.randn(1, L, decoder_embed_dim))
            self.reference_camera = nn.Linear(decoder_embed_dim, 2)
            self.lidar2token = nn.Conv2d(128, decoder_embed_dim, kernel_size=1)

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        if self.cross_modality_module is None:
            pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.grid_size)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, patch_size=None, grid_size=None):
        p = patch_size or self.patch_size
        h, w = grid_size or self.stage_resolutions[0]
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x, patch_size=None, grid_size=None):
        p = patch_size or self.patch_size
        h, w = grid_size or self.stage_resolutions[0]
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def _seq_to_2d(self, x, h, w):
        return x.transpose(1, 2).reshape(x.shape[0], -1, h, w)

    def _2d_to_seq(self, x):
        return x.flatten(2).transpose(1, 2)

    def forward(self, encoder_outs, ids_restore, lidar_x=None):
        B = encoder_outs[-1].shape[0]
        L = ids_restore.shape[1]
        H3, W3 = self.grid_size

        # Stage 3: embed + mask tokens + unshuffle
        x = self.decoder_embed(encoder_outs[-1])
        mask_tokens = self.mask_token.repeat(B, L - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))

        # Add stage 3 skip connection
        skip3 = self.lateral_projs[3](encoder_outs[3])
        skip3_aligned = F.interpolate(skip3.transpose(1,2), size=L, mode='linear', align_corners=False).transpose(1,2)
        x = x + skip3_aligned

        # Initial transformer blocks at stage 3
        x = x + self.decoder_pos_embed
        for blk in self.initial_blocks:
            x = blk(x)

        # Handle cross-modality if present
        if lidar_x is not None and self.cross_modality_module is not None:
            B_cam = x.shape[0]
            _, _, H, W = lidar_x.shape
            lidar_f = self.lidar2token(lidar_x).flatten(-2)
            num_cams = B_cam // lidar_x.shape[0]
            lidar_f = lidar_f.repeat_interleave(num_cams, dim=0).permute(0, 2, 1)
            spatial_shapes = torch.as_tensor([(H, W)], dtype=torch.long, device=x.device)
            valid_ratios = self.valid_ratios.expand(B_cam, -1, -1)
            ref_pts = self.reference_camera(x).sigmoid()
            x_c, _ = self.cross_modality_module(
                query=x.permute(1, 0, 2), key=None, value=lidar_f.permute(1, 0, 2),
                query_pos=self.decoder_pos_embed.permute(1, 0, 2), reference_points=ref_pts,
                spatial_shapes=spatial_shapes, level_start_index=self.level_start_index,
                valid_ratios=valid_ratios
            )
            x = x_c.permute(1, 0, 2)

        # Progressive upsampling: stage3 -> stage2 -> stage1 -> stage0
        current_h, current_w = H3, W3
        for i in range(self.num_stages - 1):
            stage_idx = self.num_stages - 2 - i  # 2, 1, 0
            target_h, target_w = self.stage_resolutions[stage_idx]

            # Convert to 2D, upsample, back to sequence
            x_2d = self._seq_to_2d(x, current_h, current_w)
            x_2d = self.upsample_layers[i](x_2d)
            x = self._2d_to_seq(x_2d)

            # Add skip connection from encoder
            skip = self.lateral_projs[stage_idx](encoder_outs[stage_idx])
            skip_aligned = F.interpolate(
                skip.transpose(1,2), size=target_h * target_w, mode='linear', align_corners=False
            ).transpose(1,2)
            x = x + skip_aligned

            # Refine
            x = self.upsample_norms[i](x)
            x = self.refine_blocks[i](x)

            current_h, current_w = target_h, target_w

        # Final prediction at stage 0 resolution with 4x4 patches
        x = self.final_norm(x)
        x = self.final_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask):
        """Compute reconstruction loss on masked patches.

        Args:
            imgs: Original images
            pred: Predicted patch values
            mask: Mask where 1 = masked, 0 = visible. Loss is computed only on masked patches.
        """
        # Prediction is at stage 0 resolution (80x200 with 4x4 patches)
        target = self.patchify(imgs)

        # Upsample mask from stage 3 to stage 0 resolution
        H3, W3 = self.grid_size
        H0, W0 = self.stage_resolutions[0]
        mask_2d = mask.reshape(mask.shape[0], H3, W3)
        mask_up = F.interpolate(mask_2d.unsqueeze(1).float(), size=(H0, W0), mode='nearest')
        mask_up = mask_up.squeeze(1).reshape(mask.shape[0], -1)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        # Compute loss only on masked patches (where mask = 1)
        assert mask_up.sum() > 0
        loss = (loss * mask_up).sum() / (mask_up.sum() + 1e-6)
        return loss


if __name__ == '__main__':
    # Test the implementation
    import pdb; pdb.set_trace()
