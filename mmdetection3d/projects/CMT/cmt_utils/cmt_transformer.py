# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

'''
Contains our CMT Transformer which handles the DETR encoder/decoder processing of our modality features
We have three variants that have the same weights: CMTTransformer, CMTImageTransformer, CMTLidarTransformer
We specify the type of decoder and encoder in the config file

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from typing import Sequence
from einops import rearrange
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import (
    build_transformer_layer_sequence
)
from mmengine.model import xavier_init, BaseModule
from mmdet3d.registry import MODELS



@MODELS.register_module()
class CmtTransformer(BaseModule):
    """Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None, cross=False):
        super(CmtTransformer, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.cross = cross

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    # If we are doing hard selection, we have different selections per batch
    def compact_and_pad(self, data, mask):
        """
        Args:
            data: Tensor of shape (N, B, dim)
            mask: Tensor of shape (N, B) containing 0s and 1s
        Returns:
            Tensor of shape (K, B, dim) where K is max selections in the batch.
        """
        # 1. Sort the mask descending to push 1s to the top.
        # stable=True is CRITICAL to preserve the original sequence order of the data.
        # sorted_mask will have 1s at the top, 0s at the bottom.
        # indices tells us how to rearrange the data rows to match this.
        sorted_mask, indices = torch.sort(mask, dim=0, descending=True, stable=True)

        # 2. Determine K (the maximum number of 1s in any batch element)
        # We can just sum the sorted mask along dim 0 and take the max.
        K = sorted_mask.sum(dim=0).max().long().item()

        # 3. Gather the data
        # We need to expand indices to match data's shape: (N, B) -> (N, B, dim)
        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, data.size(-1))
        
        # Gather data along dimension 0 according to the sort indices
        # The top rows now contain the "selected" data, the bottom rows contain "rejected" data.
        gathered_data = torch.gather(data, 0, expanded_indices)

        # 4. Slice and Zero-out Padding
        # Slice to K (discard the rows that definitely contain no valid data for any batch)
        final_data = gathered_data[:K]
        
        # The "padding" areas currently contain the data we wanted to discard. 
        # We must multiply by the sorted mask to turn them into strict zeros.
        final_mask = sorted_mask[:K].unsqueeze(-1) # Shape (K, B, 1)
        
        return final_data * final_mask

    def forward(self, x, x_img, query_embed, bev_pos_embed, rv_pos_embed, attn_masks=None, 
                reg_branch=None, pts_mask=None, img_mask=None):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, c, h, w = x.shape
        bev_memory = rearrange(x, "bs c h w -> (h w) bs c") # [bs, n, c, h, w] -> [n*h*w, bs, c]
        rv_memory = rearrange(x_img, "(bs v) c h w -> (v h w) bs c", bs=bs)
        bev_pos_embed = bev_pos_embed.unsqueeze(1).repeat(1, bs, 1) # [bs, n, c, h, w] -> [n*h*w, bs, c]
        rv_pos_embed = rearrange(rv_pos_embed, "(bs v) h w c -> (v h w) bs c", bs=bs)
        

        
        # If we provide either pts_mask or img_mask, it is hard masking
        if pts_mask is not None:
            pts_mask = rearrange(pts_mask,  "bs 1 h w -> (h w) bs")
            if self.training:
                bev_memory = self.compact_and_pad(bev_memory, pts_mask)
                bev_pos_embed = self.compact_and_pad(bev_pos_embed, pts_mask)
            else:
                assert bs == 1
                bev_memory = torch.unsqueeze(bev_memory[pts_mask.bool()], dim=1)
                bev_pos_embed = torch.unsqueeze(bev_pos_embed[pts_mask.bool()], dim=1)
        if img_mask is not None:
            img_mask = rearrange(img_mask, "(bs v) 1 h w -> (v h w) bs", bs=bs)
            if self.training:
                rv_memory = self.compact_and_pad(rv_memory, img_mask)
                rv_pos_embed = self.compact_and_pad(rv_pos_embed, img_mask)
            else:
                assert bs == 1
                rv_memory = torch.unsqueeze(rv_memory[img_mask.bool()], dim=1)
                rv_pos_embed = torch.unsqueeze(rv_pos_embed[img_mask.bool()], dim=1)


            
        memory, pos_embed = torch.cat([bev_memory, rv_memory], dim=0), torch.cat([bev_pos_embed, rv_pos_embed], dim=0)
        query_embed = query_embed.transpose(0, 1)  # [num_query, dim] -> [num_query, bs, dim]
        mask =  memory.new_zeros(bs, memory.shape[0]) # [bs, n, h, w] -> [bs, n*h*w]
        target = torch.zeros_like(query_embed)
        # out_dec: [num_layers, num_query, bs, dim]
        if self.encoder:
            memory = self.encoder(query=memory,
                                  key=memory,
                                  value=memory,
                                  query_pos=pos_embed, key_pos=pos_embed)
            if len(memory.shape) == 4: # Somehow it adds an extra dimension at dim 0 during the self attention process
                memory = memory[0]

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
            attn_masks=[attn_masks, None],
            reg_branch=reg_branch,
            )
        # end.record()
        # torch.cuda.synchronize()
        # with open('detr_attention.txt', 'a') as handle:
        #     print('Elapsed:', start.elapsed_time(end), file=handle)
        out_dec = out_dec.transpose(1, 2)
        return out_dec, memory


@MODELS.register_module()
class CmtLidarTransformer(BaseModule):
    """Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None, cross=False):
        super(CmtLidarTransformer, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.cross = cross

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, x, mask, query_embed, pos_embed, attn_masks=None, reg_branch=None):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        bs, c, h, w = x.shape
        memory = rearrange(x, "bs c h w -> (h w) bs c") # [bs, n, c, h, w] -> [n*h*w, bs, c]
        pos_embed = pos_embed.unsqueeze(1).repeat(1, bs, 1) # [bs, n, c, h, w] -> [n*h*w, bs, c]
        query_embed = query_embed.transpose(0, 1)  # [num_query, dim] -> [num_query, bs, dim]
        mask = mask.view(bs, -1)  # [bs, n, h, w] -> [bs, n*h*w]
        target = torch.zeros_like(query_embed)
        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
            attn_masks=[attn_masks, None],
            reg_branch=reg_branch,
            )
        out_dec = out_dec.transpose(1, 2)
        return  out_dec, memory


@MODELS.register_module()
class CmtImageTransformer(BaseModule):
    """Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None, cross=False):
        super(CmtImageTransformer, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.cross = cross

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, x_img, query_embed, rv_pos_embed, attn_masks=None, reg_branch=None, bs=2):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        memory = rearrange(x_img.float(), "(bs v) c h w -> (v h w) bs c", bs=bs)
        pos_embed = rearrange(rv_pos_embed, "(bs v) h w c -> (v h w) bs c", bs=bs)
        
        query_embed = query_embed.transpose(0, 1)  # [num_query, dim] -> [num_query, bs, dim]
        mask =  memory.new_zeros(bs, memory.shape[0]) # [bs, n, h, w] -> [bs, n*h*w]

        target = torch.zeros_like(query_embed)
        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
            attn_masks=[attn_masks, None],
            reg_branch=reg_branch,
            )
        out_dec = out_dec.transpose(1, 2)
        return  out_dec, memory
