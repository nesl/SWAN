# turn off logging for now to prevent spam
import torch
import torch.nn as nn
import math
from einops import rearrange
import random

def get_top_k(x, k=8, zero_value=0):
    top_k_indices = torch.topk(x, k, dim=1).indices
    result = torch.full(x.shape, zero_value).cuda()
    return result.scatter_(1, top_k_indices, 1)

def pad_or_truncate(tensor, target_size=5000):
    n = tensor.shape[0]
    if n >= target_size:
        return tensor[:target_size]
    else:
        # pad(left, right, top, bottom) for 2D, but we only pad the first dim
        # The syntax for 2D padding: (dim_d_pad_left, dim_d_pad_right, dim_n_pad_left, dim_n_pad_right)
        padding_size = target_size - n
        return F.pad(tensor, (0, 0, 0, padding_size), "constant", 0)
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
            nn.Conv1d(in_channels=96, out_channels=16, kernel_size=10, stride=6),
            # nn.MaxPool1d(kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=1, kernel_size=10, stride=6),
            # nn.MaxPool1d(kernel_size=4, stride=4),# output shape is 372
            nn.ReLU(),
            nn.Linear(265, embed_dim)
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
            nn.Conv1d(in_channels=768, out_channels=2, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=2, out_channels=1, kernel_size=40, stride=2),
            nn.ReLU(), # embed_dim 365
            nn.Linear(231, embed_dim)
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
        self.register_buffer('index_lookup', torch.arange(13, dtype=torch.int32))
        embeds = get_sinusoidal_embeddings(torch.arange(0,13), dim=self.embed_dim//2)
        self.register_buffer('sinusoidal_embeds', embeds)
    # We assume 320 x 800 images
    def forward(self, swin_feature, layer_count, remaining_layer_count, retained_layer_list, predicted_noise, lidar_alloc):
        _, N, D = swin_feature.shape
        swin_feature = torch.reshape(swin_feature, (-1, 6 * N, D))
        #swin_feature = swin_feature[:, 10]
        B_size = swin_feature.shape[0]
        swin_feature = torch.transpose(swin_feature, 1, 2)
        
        idx = 0 if D == 96 else 1 if D == 192 else 2 if D == 384 else 3
        condensed_embed = self.procs[idx](swin_feature)[:, 0] # get rid of the 1 dimension
        
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        noise_embed = self.noise_embed[predicted_noise] # What was the predicted noise?
        #condensed_embed = torch.ones_like(noise_embed)
        # end.record()
        # torch.cuda.synchronize()
        # print(start.elapsed_time(end))
        lidar_alloc = self.sinusoidal_embeds[lidar_alloc.int()]
        
        #lidar_alloc = get_sinusoidal_embeddings(self.lidar_alloc.int(), dim=self.embed_dim//2) # What did the other mod get?
        controller_decision = (retained_layer_list.detach() * 2 - 1).unsqueeze(-1).expand(-1, self.embed_dim//2)
        l_count_tensor = self.index_lookup[layer_count]
        l_idx = l_count_tensor.view(1).expand(B_size)
        layer_cls = self.sinusoidal_embeds[l_idx]
        remaining_embed = self.sinusoidal_embeds[remaining_layer_count.int()]
        # layer_cls = get_sinusoidal_embeddings(torch.full((B_size,), layer_count, dtype=torch.int), dim=self.embed_dim//2).to(swin_feature.device)
        # remaining_embed = get_sinusoidal_embeddings(remaining_layer_count.int(), dim=self.embed_dim//2).to(swin_feature.device)
        
        condensed_embed = torch.cat((condensed_embed, lidar_alloc, noise_embed, layer_cls, controller_decision, remaining_embed), dim=-1) # incorporate layer info
        output = 3 * torch.tanh(self.output_head(condensed_embed))
       
        return output

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
        self.register_buffer('index_lookup', torch.arange(13, dtype=torch.int32))
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
        l_count_tensor = self.index_lookup[layer_count]
        l_idx = l_count_tensor.view(1).expand(condensed_embed.shape[0])
        layer_cls = self.sinusoidal_embeds[l_idx]
        remaining_embed = self.sinusoidal_embeds[remaining_layer_count.int()]
        condensed_embed = torch.cat((condensed_embed, camera_alloc, noise_embed, layer_cls, controller_decision, remaining_embed), dim=-1) # incorporate layer info
        return 3 * torch.tanh(self.output_head(condensed_embed))

        


class UniversalConvLayerControllerCompile(nn.Module):
    
    # Total layer refers to the total available compute budget
    # TODO CHANGE THIS back to 256
    def __init__(self, embed_dim=128, num_classes=8, total_layers_img=12, total_layers_lidar=8, layer_budgets=[4, 6, 8, 10]):
        super(UniversalConvLayerControllerCompile, self).__init__()

        self.voxel_extractor = nn.Sequential(
            nn.Conv2d(128, out_channels=64, kernel_size=3, stride=3),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, out_channels=1, kernel_size=3, stride=3),
            nn.BatchNorm2d(num_features=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
        )
        self.voxel_adapter = nn.Linear(400, embed_dim//2)

        self.img_extractor = nn.Sequential(
            nn.Conv2d(3, out_channels=64, kernel_size=3, stride=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            nn.Conv2d(64, out_channels=64, kernel_size=3, stride=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),


            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(8, 16), stride=(8, 16)),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU()
        )
        self.img_adapter = nn.Linear(360, embed_dim//2)
        self.cls = nn.Parameter(torch.randn(1, embed_dim))

        self.layer_budgets = layer_budgets
        self.layer_token_dict = nn.ParameterDict({
                str(budget):nn.Parameter(torch.randn(1, embed_dim)) for budget in layer_budgets
            }
        )
        # Add this for backwards compatability
        module_list = []
        self.idx_mapping = {}
        for i,key in enumerate(self.layer_token_dict):
            module_list.append(self.layer_token_dict[key])
            self.idx_mapping[int(key)] = i

        self.layer_token_tensor = nn.Parameter(torch.stack(module_list, dim=0))

        self.output_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, total_layers_img + total_layers_lidar, bias=False) # 12 layers in each ViT, we want to generate a one-hot at the end
        )
        self.noise_output = nn.Sequential(
            nn.Linear(embed_dim*2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes) # We softmax over these three logits (clean, noisy lidar, noisy image)
        )

    def manual_scatter(self, voxel_features, coors, batch_size, grid_shape, feat_channels):
        H, W = grid_shape
        C = feat_channels
        canvas = torch.zeros((batch_size, C, H, W), device=voxel_features.device, dtype=voxel_features.dtype)
        canvas[coors[:, 0], :, coors[:, 2], coors[:, 3]] = voxel_features
        return canvas
    
    # Temperature define peakiness of the gumbel softmax
    def forward(self, bev_grid, raw_image, flatformer_layers):

        B = raw_image.shape[0]
        current_epoch = 1
        current_budget = self.layer_budgets[0]
        if self.training:
            current_budget = self.layer_budgets[int(random.random() * len(self.layer_budgets))]
            message_hub = MessageHub.get_current_instance()
            current_epoch = message_hub.get_info('epoch') + 1
            
        budget_token = self.layer_token_tensor[self.idx_mapping[current_budget]]
        budget_token = budget_token.repeat(B, 1) # expand dim
    
        voxel_noise_features = torch.reshape(self.voxel_extractor(bev_grid), (B, -1))
        voxel_noise_features = self.voxel_adapter(voxel_noise_features)
        
        raw_image = rearrange(raw_image, 'b n c h w -> (b n) c h w')
        img_noise_features = self.img_extractor(raw_image)
        img_noise_features = torch.reshape(img_noise_features, (B, -1))
        img_noise_features = self.img_adapter(img_noise_features)
        joint_embed = torch.cat([voxel_noise_features, img_noise_features, budget_token], dim=-1)

        logits = 3 * torch.tanh(self.output_head(joint_embed)) # logits are of shape B_size x 24 \
        
        predicted_noise = self.noise_output(joint_embed) # b_size x 2 (img and depth)
        
        if self.training:
            sampler = NeuralSort(tau=0.5/current_epoch, hard=False) # used to be 0.5
            weights = sampler(logits) # B_size, N, N
            weights = torch.sum(weights[:, :current_budget], dim=1) # B_size, N
        else:
            weights = get_top_k(logits, k=current_budget, zero_value=0)
        return weights, predicted_noise




foo3 = Early_Exit_Camera().cuda().eval()
opt_foo3 = torch.compile(foo3, mode="reduce-overhead")


# Returns the result of running `fn()` and the time it took for `fn()` to run,
# in seconds. We use CUDA events and synchronization for the most accurate
# measurements.
def timed(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


in1 = torch.randn((1, 128, 720, 720)).cuda()
in2 = torch.randn((1, 6, 3, 320, 800)).cuda()
in3 = 8

torch._logging.set_logs(graph_code=False)
torch.set_float32_matmul_precision('high')
eager_times = []

in_shape = (6, 16000, 96)
for i in range(10):
    inp = torch.randn(in_shape).cuda()
    inp = inp[:, ::10]
    one_tensor=torch.tensor([random.randint(1, 8)]).cuda()
    _, eager_time = timed(lambda: foo3(inp, 1, one_tensor, one_tensor, one_tensor, one_tensor))
    eager_times.append(eager_time)
    print(f"eager time {i}: {eager_time}")
print("~" * 10)

compile_times = []
for i in range(100):
    input_buffer = torch.randn(in_shape).cuda()
    input_buffer = input_buffer[:, ::10]
    one_tensor=torch.tensor([random.randint(1, 8)]).cuda()
    _, compile_time = timed(lambda: opt_foo3(input_buffer, 1, one_tensor, one_tensor, one_tensor, one_tensor))
    compile_times.append(compile_time)
    print(f"compile time {i}: {compile_time}")
print("~" * 10)

import numpy as np

eager_med = np.median(eager_times)
compile_med = np.median(compile_times)
speedup = eager_med / compile_med
print(
    f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x"
)
print("~" * 10)