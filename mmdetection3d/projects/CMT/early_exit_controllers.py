import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from torch.nn.utils.rnn import pad_sequence

import torch.nn.functional as F

def pad_or_truncate(tensor, target_size=5000):
    n = tensor.shape[0]
    if n >= target_size:
        return tensor[:target_size]
    else:
        # pad(left, right, top, bottom) for 2D, but we only pad the first dim
        # The syntax for 2D padding: (dim_d_pad_left, dim_d_pad_right, dim_n_pad_left, dim_n_pad_right)
        padding_size = target_size - n
        return F.pad(tensor, (0, 0, 0, padding_size), "constant", 0)

import torch
import math

def get_sinusoidal_embeddings(budget_tensor, dim, max_period=24):
    """
    Args:
        budget_tensor: Tensor of shape (batch_size,) or (seq_len,) containing budget ints.
        dim: The dimension of the output embedding (must be even).
        max_period: The maximum period for the frequencies (standard is 10000).
        
    Returns:
        Tensor of shape (..., dim) with sinusoidal embeddings.
    """
    if dim % 2 != 0:
        raise ValueError("Embedding dimension must be even for sin/cos pairs.")

    device = budget_tensor.device
    half_dim = dim // 2
    
    # Compute the frequency scale: exp(log(10000) * i / half_dim)
    # This is more numerically stable than power operations.
    weights = torch.arange(half_dim, dtype=torch.float32, device=device)
    weights = torch.exp(weights * -math.log(max_period) / half_dim)
    
    # Calculate angles: shape (batch_size, half_dim)
    # We use unsqueeze to allow broadcasting against the weights
    args = budget_tensor.unsqueeze(-1).float() * weights
    
    # Combine sin and cos
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    return embedding


@MODELS.register_module()
# Use conv layers with maxpooling
class Early_Exit_Camera(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim

        self.predicted_noise = None
        self.noise_embed = nn.Parameter(torch.randn(10, embed_dim) * 2)
        self.lidar_alloc = None

        self.processing_dict = nn.ModuleDict()
        self.processing_dict['96'] = nn.Sequential(
            nn.Conv1d(in_channels=96, out_channels=16, kernel_size=10, stride=16),
            # nn.MaxPool1d(kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=40, stride=16),
            # nn.MaxPool1d(kernel_size=4, stride=4),# output shape is 372
            nn.ReLU(),
            nn.Linear(373, embed_dim)
        ) 
        self.processing_dict['192'] = nn.Sequential(
            nn.Conv1d(in_channels=192, out_channels=8, kernel_size=10, stride=10),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=40, stride=10),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Linear(237, embed_dim)
        )
        self.processing_dict['384'] = nn.Sequential(
            nn.Conv1d(in_channels=384, out_channels=4, kernel_size=10, stride=4),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=4, out_channels=1, kernel_size=40, stride=4),
            # nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(), # embed_dim 365
            nn.Linear(365, embed_dim)
        )
        self.processing_dict['768'] = nn.Sequential(
            nn.Conv1d(in_channels=768, out_channels=2, kernel_size=10, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=40, stride=2),
            nn.ReLU(), # embed_dim 365
            nn.Linear(354, embed_dim)
        )
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.procs = nn.ModuleList([
            self.processing_dict['96'], 
            self.processing_dict['192'], 
            self.processing_dict['384'], 
            self.processing_dict['768']
        ])
 
        embeds = get_sinusoidal_embeddings(torch.arange(0,13), dim=self.embed_dim//2)
        self.register_buffer('sinusoidal_embeds', embeds)
    # We assume 320 x 800 images
    def forward(self, swin_feature, layer_count, remaining_layer_count, retained_layer_list, predicted_noise, lidar_alloc):
        _, N, D = swin_feature.shape
        swin_feature = torch.reshape(swin_feature, (-1, 6 * N, D))
        B_size = swin_feature.shape[0]
        swin_feature = torch.transpose(swin_feature, 1, 2)
        
        idx = 0 if D == 96 else 1 if D == 192 else 2 if D == 384 else 3
        condensed_embed = self.procs[idx](swin_feature)[:, 0] # get rid of the 1 dimension
        
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        noise_embed = self.noise_embed[predicted_noise] # What was the predicted noise?
        # end.record()
        # torch.cuda.synchronize()
        # print(start.elapsed_time(end))
        lidar_alloc = self.sinusoidal_embeds[lidar_alloc.int()]
        
        #lidar_alloc = get_sinusoidal_embeddings(self.lidar_alloc.int(), dim=self.embed_dim//2) # What did the other mod get?
        controller_decision = (retained_layer_list.detach() * 2 - 1).unsqueeze(-1).expand(-1, self.embed_dim//2)
        layer_cls = self.sinusoidal_embeds[torch.full((B_size,), layer_count, dtype=torch.int)]
        remaining_embed = self.sinusoidal_embeds[remaining_layer_count.int()]
        # layer_cls = get_sinusoidal_embeddings(torch.full((B_size,), layer_count, dtype=torch.int), dim=self.embed_dim//2).to(swin_feature.device)
        # remaining_embed = get_sinusoidal_embeddings(remaining_layer_count.int(), dim=self.embed_dim//2).to(swin_feature.device)
        
        condensed_embed = torch.cat((condensed_embed, lidar_alloc, noise_embed, layer_cls, controller_decision, remaining_embed), dim=-1) # incorporate layer info
        output = 3 * torch.tanh(self.output_head(condensed_embed))
       
        return output

@MODELS.register_module()
class Early_Exit_Lidar(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim=embed_dim
        self.noise_embed = nn.Parameter(torch.randn(10, embed_dim) * 2)
        self.process_lidar = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=8, kernel_size=10, stride=5),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=10, stride=2),
            nn.ReLU(),
            nn.Linear(495, embed_dim)
        )
        embeds = get_sinusoidal_embeddings(torch.arange(0,13), dim=self.embed_dim//2)
        self.register_buffer('sinusoidal_embeds', embeds)
        # self.layer_cls = nn.Parameter(torch.randn(8, embed_dim))
        # self.remaining_embed = nn.Parameter(torch.randn(8, embed_dim))
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, voxel_batches, layer_count, remaining_layer_count, retained_layer_list, predicted_noise, camera_alloc):
        voxel_batches = [pad_or_truncate(v) for v in voxel_batches]
        aggregated_tensor = torch.stack(voxel_batches, dim=0)
        condensed_embed = self.process_lidar(torch.transpose(aggregated_tensor, 1, 2))[:, 0]

        noise_embed = self.noise_embed[predicted_noise] # What was the predicted noise?
        camera_alloc = self.sinusoidal_embeds[camera_alloc.int()] # What did the other mod pick?

        controller_decision = (retained_layer_list.detach() * 2 - 1).unsqueeze(-1).expand(-1, self.embed_dim//2)
        layer_cls = self.sinusoidal_embeds[torch.full((len(voxel_batches),), layer_count, dtype=torch.int)]
        remaining_embed = self.sinusoidal_embeds[remaining_layer_count.int()]
        condensed_embed = torch.cat((condensed_embed, camera_alloc, noise_embed, layer_cls, controller_decision, remaining_embed), dim=-1) # incorporate layer info
        return 3 * torch.tanh(self.output_head(condensed_embed))

        

if __name__=='__main__':
    test = Early_Exit_Camera().cuda()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    elapsed_time = 0
    for i in range(1000):
        test_in = torch.randn(6 * 12, 4000, 192).cuda()
        # start.record()
        res = test(test_in)
        print(res.shape)
    #     stop.record()
    #     torch.cuda.synchronize()
    #     if i > 50:
    #         elapsed_time += start.elapsed_time(stop)
    # print("Average:", elapsed_time / 9950)

    # import pdb; pdb.set_trace()



        