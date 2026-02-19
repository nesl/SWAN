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

@MODELS.register_module()
# Use conv layers with maxpooling
class Early_Exit_Camera(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.processing_dict = nn.ModuleDict()
        self.layer_cls = nn.Parameter(torch.randn(12, embed_dim))
        self.remaining_embed = nn.Parameter(torch.randn(12, embed_dim))
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
            nn.Linear(embed_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    # We assume 320 x 800 images
    def forward(self, swin_feature, layer_count, remaining_layer_count):
        _, N, D = swin_feature.shape
        swin_feature = torch.reshape(swin_feature, (-1, 6 * N, D))
        B_size = swin_feature.shape[0]
        swin_feature = torch.transpose(swin_feature, 1, 2)
        condensed_embed = self.processing_dict[str(D)](swin_feature)[:, 0] # get rid of the 1 dimension
        layer_cls = self.layer_cls[layer_count].unsqueeze(dim=0).repeat(B_size,1)
        remaining_embed = self.remaining_embed[remaining_layer_count.int() - 1]
        condensed_embed = torch.cat((condensed_embed, layer_cls, remaining_embed), dim=-1) # incorporate layer info
        return self.output_head(condensed_embed)

@MODELS.register_module()
class Early_Exit_Lidar(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.process_lidar = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=8, kernel_size=10, stride=5),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=10, stride=2),
            nn.ReLU(),
            nn.Linear(495, embed_dim)
        )
        self.layer_cls = nn.Parameter(torch.randn(8, embed_dim))
        self.remaining_embed = nn.Parameter(torch.randn(8, embed_dim))
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, voxel_batches, layer_count, remaining_layer_count):
        voxel_batches = [pad_or_truncate(v) for v in voxel_batches]
        aggregated_tensor = torch.stack(voxel_batches, dim=0)
            
        condensed_embed = self.process_lidar(torch.transpose(aggregated_tensor, 1, 2))[:, 0]
        layer_cls = self.layer_cls[layer_count].unsqueeze(dim=0).repeat(len(voxel_batches),1)
        remaining_embed = self.remaining_embed[remaining_layer_count.int() - 1]
        condensed_embed = torch.cat((condensed_embed, layer_cls, remaining_embed), dim=-1) # incorporate layer info
        return self.output_head(condensed_embed)

        

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



        