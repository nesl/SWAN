'''
Docstring for projects.CMT.cmt_utils.attention

This holds the FlashAttention functions that we call inside our FlatFormer attention 
and our decoder attention
'''


# Copyright (c) 2023 megvii-model. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn.init import (
    xavier_uniform_,
    constant_,
)
from torch.nn.functional import linear
from einops import rearrange

# --- NEW IMPORTS for FlashAttention v2 ---
try:
    # Attempt to import for FlashAttention v2.x
    from flash_attn import flash_attn_varlen_kvpacked_func
    # The location of unpad_input changed in newer versions
    try:
        from flash_attn.bert_padding import unpad_input
    except ImportError:
        # Fallback for some v2.x versions where it might be in utils
        from flash_attn.utils.bert_padding import unpad_input
except ImportError:
    print("Error: flash_attn v2 not found. Please install flash-attn>=2.0.0")


def _in_projection_packed(q, k, v, w, b=None):
    w_q, w_k, w_v = w.chunk(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax using FlashAttention v2."""
    
    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.fp16_enabled = True

    def forward(self, q, kv, causal=False, key_padding_mask=None):
        """
        Arguments:
            q: (B, Sq, H, D) 
            kv: (B, Sk, 2, H, D) 
            key_padding_mask: bool tensor (B, Sk)
        """
        # 1. Cast to bf16 (preferred for H100) or fp16
        #target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        target_dtype = torch.float16
        if q.dtype == torch.float32:
            q = q.to(dtype=target_dtype)
        if kv.dtype == torch.float32:
            kv = kv.to(dtype=target_dtype)

        assert q.dtype in [torch.float16, torch.bfloat16] and kv.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        
        batch_size = q.shape[0]
        seqlen_q = q.shape[1]
        seqlen_k = kv.shape[1]
        
        # FlashAttention v2 expects "total_q" flattened, so we generally unpad/flatten first.
        
        if key_padding_mask is None:
            # Case 1: No padding mask (Simple flatten)
            
            # Flatten batch and seqlen dimensions: (B, S, ...) -> (B*S, ...)
            q = rearrange(q, 'b s ... -> (b s) ...')
            kv = rearrange(kv, 'b s ... -> (b s) ...')
            
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device)
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=kv.device)
            
            max_sq = seqlen_q
            max_sk = seqlen_k

            output = flash_attn_varlen_kvpacked_func(
                q, kv, 
                cu_seqlens_q, cu_seqlens_k, 
                max_sq, max_sk,
                dropout_p=self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, 
                causal=causal
            )
            
            # Reshape back to (B, S, ...)
            output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            
        else:
            # Case 2: With padding mask (Need unpad_input)
            nheads = kv.shape[-2]
            head_dim = kv.shape[-1]
            
            # FlashAttn v2 varlen expects Q to be packed too if we use varlen func.
            # However, usually Q is unpadded in this specific PETR architecture (it's the query features).
            # But KV (image features) might be padded.
            
            # 2a. Handle Query (Assume Q is not padded or we just flatten it)
            # If Q also has padding, we'd need to unpad it. But typically in DETR/PETR, Q is fixed size.
            q_flat = rearrange(q, 'b s ... -> (b s) ...')
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q.device)
            max_sq = seqlen_q

            # 2b. Handle KV (Unpad based on mask)
            # kv shape: (B, S, 2, H, D) -> flatten to (B, S, 2*H*D) for unpad_input
            x = rearrange(kv, 'b s two h d -> b s (two h d)')
            
            # unpad_input returns flattened valid tokens
            x_unpad, indices, cu_seqlens_k, max_sk = unpad_input(x, key_padding_mask)
            
            # Reshape back to (Total_Valid_Tokens, 2, H, D)
            # Note: unpad_input returns indices for reconstruction if needed, but here we only need KV.
            x_unpad = rearrange(x_unpad, 'nnz (two h d) -> nnz two h d', two=2, h=nheads)
            
            output_unpad = flash_attn_varlen_kvpacked_func(
                q_flat, x_unpad, 
                cu_seqlens_q, cu_seqlens_k, 
                max_sq, max_sk,
                dropout_p=self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, 
                causal=causal
            )
            
            output = rearrange(output_unpad, '(b s) ... -> b s ...', b=batch_size)

        return output, None


class FlashMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                 causal=False, device=None, dtype=None, **kwargs) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.bias = bias

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        
        # FlashAttention requirement: Head dim must be multiple of 8 (usually <= 256 for v2)
        assert self.head_dim % 8 == 0, "Head dim must be divisible by 8"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
            
        self.inner_attn = FlashAttention(attention_dropout=attention_dropout, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        
    def forward(self, q, k, v, key_padding_mask=None):
        """
        q: (batch, seqlen_q, embed_dim)
        k: (batch, seqlen_k, embed_dim)
        v: (batch, seqlen_k, embed_dim)
        key_padding_mask: bool tensor (batch, seqlen_k) (True = Ignore/Pad)
        """
        # Project Q, K, V
        q, k, v = _in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)
        
        # Reshape for FlashAttention: (Batch, Seq, Heads, Dim)
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads)
        
        # Stack KV for kvpacked implementation
        kv = torch.stack([k, v], dim=2)
        
        # Invert mask logic if necessary
        # FlashAttention unpad_input expects a mask where 0 (False) is KEEP and 1 (True) is PAD?
        # Actually: FlashAttn unpad_input expects:
        #   mask: (batch, seqlen). 0 means masked out (remove), 1 means keep? 
        #   NO. unpad_input logic: indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        #   So it keeps elements where mask is True (or non-zero).
        
        # CHECK YOUR DATASET/MODEL: 
        # In standard PyTorch MHA, key_padding_mask=True usually means "PAD/IGNORE".
        # If so, we must invert it for unpad_input (which treats True as "KEEP").
        if key_padding_mask is not None:
             # Assuming input mask is: True=PAD, False=Valid
             # We need: True=Valid, False=PAD for unpad_input
             key_padding_mask = ~key_padding_mask

        context, attn_weights = self.inner_attn(q, kv, key_padding_mask=key_padding_mask, causal=self.causal)
        if context.dtype != self.out_proj.weight.dtype:
            context = context.to(self.out_proj.weight.dtype)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights