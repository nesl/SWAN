import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
# Import the "varlen" (variable length) version for v2
from flash_attn import flash_attn_varlen_qkvpacked_func
from torch.nn import functional as F
from mmdet3d.registry import MODELS
from mmengine.dist import get_dist_info
from mmengine.logging import MessageHub

__all__ = ["FlatFormer"]


def _create_cu_seqlens(batch_size: int, num_tokens: int, device: torch.device) -> torch.Tensor:
    return torch.arange(
        0,
        num_tokens * (batch_size + 1),
        step=num_tokens,
        dtype=torch.int32,
        device=device,
    )

class OrdinaryMultiHeadAttn(nn.MultiheadAttention):
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 1. Setup dimensions
        batch_size, num_tokens, embed_dim = q.shape
        num_heads = self.num_heads
        head_dim = embed_dim // num_heads
        origin_dtype = q.dtype
        # 2. Linear Projections (Replicating the baddbmm logic)
        # FlashAttention uses internal weights (in_proj_weight/bias)
        # We apply them to q, k, v to get the combined QKV tensor
        x = torch.stack([q, k, v]) # [3, batch_size, num_tokens, embed_dim]
        x = x.view(3, -1, embed_dim) # Flatten for baddbmm
        
        # Replicate: x = torch.baddbmm(self.ib(), x, self.iw())
        # Note: Using the weights from the provided flash_attn_module
        x = torch.baddbmm(
            self.in_proj_bias.view(3, 1, -1).to(origin_dtype), 
            x, 
            self.in_proj_weight.view(3, embed_dim, embed_dim).transpose(1, 2).to(origin_dtype)
        )

        # 3. Reshape and Split Heads
        # Reshape back to [3, batch, tokens, heads, head_dim]
        qkv = x.view(3, batch_size, num_tokens, num_heads, head_dim)
        
        # Split into individual Q, K, V and transpose to [batch, heads, tokens, head_dim]
        # This is the standard format for PyTorch attention
        q_out = qkv[0].transpose(1, 2)
        k_out = qkv[1].transpose(1, 2)
        v_out = qkv[2].transpose(1, 2)

        scaling = head_dim ** -0.5
        attn_scores = torch.matmul(q_out, k_out.transpose(-2, -1)) * scaling
        
        # 3. Apply Softmax
        # This is the point where standard attention usually slows down 
        # because it has to read/write the huge N x N matrix to memory
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 4. Multiply by Values
        # [batch, heads, tokens, tokens] @ [batch, heads, tokens, head_dim]
        attn_output = torch.matmul(attn_weights, v_out)

        # 5. Reshape and Final Projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_tokens, embed_dim)
        
        # Final output projection
        output = F.linear(
            attn_output, 
            self.out_proj.weight, 
            self.out_proj.bias
        )

        return output, None

class FlashAttention(nn.MultiheadAttention):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert self._qkv_same_embed_dim

        batch_size, num_tokens, embed_dim = q.shape
        head_dim = embed_dim // self.num_heads

        origin_dtype = q.dtype
        # Determine target dtype for FlashAttention (must be float16 or bfloat16)
        target_dtype = torch.float16 if origin_dtype == torch.float32 else origin_dtype
        x = torch.stack([q, k, v])
        x = x.to(target_dtype)
        x = x.view(3, -1, x.shape[-1])

        x = torch.baddbmm(
            self.ib().to(target_dtype), 
            x, 
            self.iw().to(target_dtype)
        )
        qkv = x.view(3, -1, self.num_heads, head_dim).transpose(0, 1)

        cu_seqlens = _create_cu_seqlens(batch_size, num_tokens, qkv.device)
        x = flash_attn_varlen_qkvpacked_func(
            qkv, 
            cu_seqlens, 
            max_seqlen=num_tokens, 
            dropout_p=0.0
        )
        x = x.view(batch_size, num_tokens, -1)
        x = x.to(origin_dtype)
        x = F.linear(x, self.out_proj.weight, self.out_proj.bias)
        return x, None

    def iw(self) -> torch.Tensor:
        tensor = self.in_proj_weight
        tensor = tensor.view(3, -1, tensor.shape[-1])
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor

    def ib(self) -> torch.Tensor:
        tensor = self.in_proj_bias
        tensor = tensor.view(3, 1, -1)
        return tensor


class GroupAttention(nn.Module):
    def __init__(self, in_channels: int, num_heads: int, group_size: int) -> None:
        super().__init__()
        self.group_size = group_size
        self.attn = FlashAttention(in_channels, num_heads)
        # self.attn = OrdinaryMultiHeadAttn(in_channels, num_heads)

    def forward(self, x, pe):
        size = x.shape[0]
        num_groups = int(math.ceil(size / self.group_size))

        x = x.view(num_groups, self.group_size, -1)
        pe = pe.view(num_groups, self.group_size, -1)

        q = k = x + pe
        v = x

        x, _ = self.attn(q, k, v)

        x = x.view(num_groups * self.group_size, -1)

        return x


class BasicLayer(nn.Module):
    def __init__(self, in_channels, num_heads, activation, group_size) -> None:
        super().__init__()
        self.attn = GroupAttention(in_channels, num_heads, group_size)

        self.fc1 = nn.Linear(in_channels, 2 * in_channels)
        self.fc2 = nn.Linear(2 * in_channels, in_channels)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        self.act = _get_activation_fn(activation)

        self.fp16_enabled = False


    def forward(self, src, pe):
        src = self.norm1(src + self.attn(src, pe))
        src = self.norm2(src + self.fc2(self.act(self.fc1(src))))

        return src

def _gumbel_sigmoid(
    logits, tau=1, hard=False, eps=1e-10, training = True, threshold = 0.5
):
    if training :
        # ~Gumbel(0,1)`
        gumbels1 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels2 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        # Difference of two` gumbels because we apply a sigmoid
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
    else :
        y_soft = logits.sigmoid()

    if hard:
        # Straight through.
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).masked_fill(y_soft > threshold, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

# Each basicblock contains 4 basic layers for x, x_shift, y, and y_shift
# Add base layer dropping logic here
class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        num_heads,
        activation,
        group_size,
    ) -> None:
        super().__init__()
        self.block = nn.ModuleList()
        for _ in range(4):
            layer = BasicLayer(
                in_channels,
                num_heads,
                activation,
                group_size=group_size,
            )
            self.block.append(layer)
    # TODO: We may have to update this structure slightly when moving towards TensorRT and edge devices 
    def forward(self, x: torch.Tensor, pe: torch.Tensor, mappings: Dict[str, Any], 
                retained_layer_list=None, controller_training=False, early_exit_lidar=None,
                  losses=None, block_layer_start=0, remaining_layer_count=None,
                  predicted_noise=None, camera_alloc=None) -> torch.Tensor:
        # Run through the four basic layers while performing shift and sort operations
        decision_list = []
        logits_list = []
        early_exit_training = early_exit_lidar is not None and early_exit_lidar.training
        for k, name in enumerate(["x", "x_shift", "y", "y_shift"]):

            indices = mappings[name]
            # If we are training early exit, or if we are testing with early_exit_enabled and controller specifically activates this layer (bsize 1)
            if early_exit_training or (early_exit_lidar is not None and not self.training and retained_layer_list[0][k]):
                lidar_feature = x[indices]
                lengths = [mappings['batch_start_indices'][i+1] - mappings['batch_start_indices'][i] for i in range(len(mappings['batch_start_indices']) - 1)]
                voxel_batches = torch.split(lidar_feature, lengths)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                raw_logits = early_exit_lidar(voxel_batches, block_layer_start + k, 
                                              torch.tensor(remaining_layer_count).to(x.device), 
                                              retained_layer_list[:, k], predicted_noise=predicted_noise,
                                              camera_alloc=camera_alloc)
                end.record()
                torch.cuda.synchronize()
                print(start.elapsed_time(end))
                # Subtract the mask to tell the EE module how many controller layers it has left
                # We have to keep as list or else pytorch freaks out about grad computation
                for list_idx in range(len(remaining_layer_count)):
                    remaining_layer_count[list_idx] -= torch.round(retained_layer_list[:, k])[list_idx].item()
                logits_list.append(raw_logits[0][0].item())

                current_epoch = 1
                if early_exit_lidar.training:
                    message_hub = MessageHub.get_current_instance()
                    current_epoch = message_hub.get_info('epoch') + 1
                decision = torch.squeeze(_gumbel_sigmoid(raw_logits, training=early_exit_lidar.training, tau=max(0.25/current_epoch, 0.05), hard=not early_exit_lidar.training)) # b_size x 1
                # Ignore this layer during inference
                if not early_exit_lidar.training:
                    decision_list.append(decision.item())
                    if decision.item() == 0:
                        continue    
                else:
                    if decision.ndim != 0:
                        decision_list.append(decision[0].item())

            # If retained_layer_list does not exist OR is 1 OR controller training is active, run that particular layer
            # During normal layerdrop training and all inference retained_layer_list has tensor shape [1, 8]
            if controller_training or early_exit_training or retained_layer_list is None or retained_layer_list[0][k]:

                x_new = self.block[k](x[indices][mappings["flat2win"]], pe[indices][mappings["flat2win"]])[
                    mappings["win2flat"]
                ]

                # If we are doing controller training, then retained_layer_list is going to be B x 12 as it is lidar
                if controller_training or early_exit_training:
                    across_batch_mask = retained_layer_list[:, k]
                    if early_exit_lidar is not None:
                        # I am 80% sure that when the controller picks a 0, it will kill the grads
                        # This means it will naturally get minimized by the hinge loss
                        # This means the EE should ignore the ones that the controller doesnt pick
                        across_batch_mask = across_batch_mask * decision
                        if 'early_exit_loss' not in losses:
                            losses['early_exit_loss'] = torch.mean(torch.relu(raw_logits + 2)) / 30
                        else:
                            losses['early_exit_loss'] += torch.mean(torch.relu(raw_logits + 2)) / 30
                    repeats = mappings['batch_start_indices'][1:] - mappings['batch_start_indices'][:-1]
                    expanded_mask = torch.unsqueeze(torch.repeat_interleave(across_batch_mask, repeats), dim=-1) # Match dims with x[indices]
                    x[indices] = x[indices] * (1 - expanded_mask) + x_new * expanded_mask
                else:
                    x[indices] = x_new
        # if get_dist_info()[0] == 0 and early_exit_lidar is not None:
        #     print("Lidar Logits List", logits_list)
        #     print("Lidar Decision List", decision_list)
        return x


def _get_activation_fn(activation):
    if activation == "relu":
        return torch.nn.functional.relu
    if activation == "gelu":
        return torch.nn.functional.gelu
    if activation == "glu":
        return torch.nn.functional.glu


@torch.inference_mode()
def get_window_coors_shift(coords, sparse_shape, window_shape, shifted):
    n, m, _ = sparse_shape
    n2, m2, _ = window_shape

    n1 = int(np.ceil(n / n2) + 1)  # plus one here to meet the needs of shift.
    m1 = int(np.ceil(m / m2) + 1)  # plus one here to meet the needs of shift.

    if shifted:
        shift_x, shift_y = (n2 // 2, m2 // 2)
        x = coords[:, 3] + shift_x
        y = coords[:, 2] + shift_y
    else:
        x = coords[:, 3]
        y = coords[:, 2]

    x1 = x // n2
    y1 = y // m2
    x2 = x % n2
    y2 = y % m2

    return 2 * n2, 2 * m2, 2 * n1, 2 * m1, x1, y1, x2, y2


class FlattenedWindowMapping(nn.Module):
    def __init__(
        self,
        window_shape,
        sparse_shape,
        group_size,
    ) -> None:
        super().__init__()
        self.sparse_shape = sparse_shape
        self.window_shape = window_shape
        self.group_size = group_size

    def forward(self, coords: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        coords = coords.long()

        _, num_per_batch = torch.unique(coords[:, 0], sorted=False, return_counts=True)

        batch_start_indices = F.pad(torch.cumsum(num_per_batch, dim=0), (1, 0))
        num_per_batch_p = (
            torch.div(
                batch_start_indices[1:] - batch_start_indices[:-1] + self.group_size - 1,
                self.group_size,
                rounding_mode="trunc",
            )
            * self.group_size
        )
        batch_start_indices_p = F.pad(torch.cumsum(num_per_batch_p, dim=0), (1, 0))
        needed1 = int(batch_start_indices_p[-1])
        needed2 = int(batch_start_indices[-1])
        max_needed = max(needed1, needed2)

        if not hasattr(self, "_arange_cache") or self._arange_cache.numel() < max_needed:
            self._arange_cache = torch.arange(max_needed, device=coords.device)

        flat2win = self._arange_cache[:needed1].clone()
        win2flat = self._arange_cache[:needed2].clone()

        # flat2win = torch.arange(batch_start_indices_p[-1]).to(coords.device)
        # win2flat = torch.arange(batch_start_indices[-1]).to(coords.device)

        for i in range(batch_size):
            win2flat[batch_start_indices[i] : batch_start_indices[i + 1]] += (
                batch_start_indices_p[i] - batch_start_indices[i]
            )
            if num_per_batch[i] != num_per_batch_p[i]:
                flat2win[
                    batch_start_indices_p[i + 1]
                    - self.group_size
                    + (num_per_batch[i] % self.group_size) : batch_start_indices_p[i + 1]
                ] = flat2win[
                    batch_start_indices_p[i + 1]
                    - 2 * self.group_size
                    + (num_per_batch[i] % self.group_size) : batch_start_indices_p[i + 1]
                    - self.group_size
                ]
            flat2win[batch_start_indices_p[i] : batch_start_indices_p[i + 1]] -= (
                batch_start_indices_p[i] - batch_start_indices[i]
            )
        

        mappings = {"flat2win": flat2win, "win2flat": win2flat, 'batch_start_indices':batch_start_indices}

        for shifted in [False, True]:
            (
                n2,
                m2,
                n1,
                m1,
                x1,
                y1,
                x2,
                y2,
            ) = get_window_coors_shift(coords, self.sparse_shape, self.window_shape, shifted=shifted)
            vx = (n1 * y1 + (-1) ** y1 * x1) * n2 * m2 + (-1) ** y1 * (m2 * x2 + (-1) ** x2 * y2)
            vx += coords[:, 0] * self.sparse_shape[0] * self.sparse_shape[1] * 10
            vy = (m1 * x1 + (-1) ** x1 * y1) * m2 * n2 + (-1) ** x1 * (n2 * y2 + (-1) ** y2 * x2)
            vy += coords[:, 0] * self.sparse_shape[0] * self.sparse_shape[1] * 10
            _, mappings["x" + ("_shift" if shifted else "")] = torch.sort(vx)
            _, mappings["y" + ("_shift" if shifted else "")] = torch.sort(vy)

        return mappings


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        feat_dim,
        sparse_shape,
        normalize_pos,
        pos_temperature,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.sparse_shape = sparse_shape
        self.normalize_pos = normalize_pos
        self.pos_temperature = pos_temperature

    def forward(self, coors, dtype):
        size_x, size_y, size_z = self.sparse_shape

        x, y = coors[:, 3], coors[:, 2]

        if self.normalize_pos:
            x = x / size_x * 2 * 3.1415  # [-pi, pi]
            y = y / size_y * 2 * 3.1415  # [-pi, pi]

        inv_freq = self.inv_freq

        # [num_tokens, pos_length]
        pex = x[:, None] / inv_freq()[None, :]
        pey = y[:, None] / inv_freq()[None, :]

        # [num_tokens, pos_length]
        pex = torch.stack([pex[:, ::2].sin(), pex[:, 1::2].cos()], dim=-1).flatten(1)
        pey = torch.stack([pey[:, ::2].sin(), pey[:, 1::2].cos()], dim=-1).flatten(1)
        pe = torch.cat([pex, pey], dim=-1).to(dtype)

        gap = self.feat_dim - pe.size(1)
        if gap > 0:
            pe_p = torch.zeros((pe.size(0), gap), dtype=dtype, device=coors.device)
            pe = torch.cat([pe, pe_p], dim=1)

        return pe

    def inv_freq(self):
        ndim = 2
        pos_length = (self.feat_dim // (ndim * 2)) * 2

        # [pos_length]
        inv_freq = torch.arange(pos_length, dtype=torch.float32, device="cuda")
        inv_freq = self.pos_temperature ** (2 * (inv_freq // 2) / pos_length)
        return inv_freq


@MODELS.register_module()
class FlatFormer(nn.Module):
    def __init__(
        self,
        in_channels=128,
        num_heads=8,
        num_blocks=2,
        activation="gelu",
        window_shape=(9, 9, 1),
        sparse_shape=(468, 468, 1),
        output_shape=(468, 468),
        pos_temperature=10000,
        normalize_pos=False,
        group_size=69,
    ) -> None:
        super().__init__()
        self.group_size = group_size

        self.embedding = PositionalEmbedding(in_channels, sparse_shape, normalize_pos, pos_temperature)
        self.mapping = FlattenedWindowMapping(
            window_shape=window_shape,
            sparse_shape=sparse_shape,
            group_size=group_size,
        )

        self.block_list = nn.ModuleList()
        for _ in range(num_blocks):
            self.block_list.append(BasicBlock(in_channels, num_heads, activation, group_size))

        self._reset_parameters()

        self.output_shape = output_shape

    # Added LayerDropping Logic
    def forward(self, x, coords, batch_size, retained_layer_list=None, controller_training=False, early_exit_lidar=None, losses=None, predicted_noise=None, camera_alloc=None):

        pe = self.embedding(coords, x.dtype)
        mappings = self.mapping(coords, batch_size)
        # This is to pass to the early-exit to make it aware of how many layers it has left to execute
        remaining_layer_count=[8] * batch_size
        
        if early_exit_lidar is not None:
            remaining_layer_count = torch.round(torch.sum(retained_layer_list, dim=-1)).int().detach().tolist()

        for i, block in enumerate(self.block_list):
            # Chunk the list four at a time for each block
            stage_layer_list = None
            if retained_layer_list is not None:
                stage_layer_list = retained_layer_list[:, i*4:(i+1) * 4]
            x = block(x, pe, mappings, stage_layer_list, controller_training, early_exit_lidar, 
                      losses=losses, block_layer_start=i * 4, remaining_layer_count = remaining_layer_count,
                      predicted_noise=predicted_noise, camera_alloc=camera_alloc)
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        x = self.recover_bev(x, coords, batch_size)
        # end.record()
        # torch.cuda.synchronize()
        # print("BEV Proj", start.elapsed_time(end))

        return x

    def _reset_parameters(self):
        for _, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def recover_bev(self, voxel_feat, coors, batch_size):
        ny, nx = self.output_shape
        feat_dim = voxel_feat.shape[-1]

        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(feat_dim, nx * ny, dtype=voxel_feat.dtype, device=voxel_feat.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_feat[batch_mask, :]  # [n, c]
            voxels = voxels.t()  # [c, n]
            canvas[:, indices] = voxels
            batch_canvas.append(canvas)

        batch_canvas = torch.stack(batch_canvas, 0)

        batch_canvas = batch_canvas.view(batch_size, feat_dim, ny, nx)

        return batch_canvas


