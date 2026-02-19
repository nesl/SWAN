import torch
import torch.nn as nn
from .cmt_utils import PETRTransformerDecoder
from torchvision import transforms
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
import math
from mmengine.dist import get_dist_info
from mmengine.logging import MessageHub
import random

EPSILON = np.finfo(np.float32).tiny

# This does gumbel softmax sampling without using straight through estimator
class SubsetOperator(torch.nn.Module):
    def __init__(self, k, tau=1.0, hard=False):
        super(SubsetOperator, self).__init__()
        self.k = k
        self.hard = hard
        self.tau = tau

    def forward(self, scores):
        m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
        g = m.sample()
        scores = scores + g

        # continuous top k
        khot = torch.zeros_like(scores)
        onehot_approx = torch.zeros_like(scores)
        for i in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, torch.tensor([EPSILON]).cuda())
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores / self.tau, dim=1)
            khot = khot + onehot_approx

        if self.hard:
            # straight through
            khot_hard = torch.zeros_like(khot)
            val, ind = torch.topk(khot, self.k, dim=1)
            khot_hard = khot_hard.scatter_(1, ind, 1)
            res = khot_hard - khot.detach() + khot
        else:
            res = khot

        return res
    
class NeuralSort (torch.nn.Module):
    def __init__(self, tau=1.0, hard=False):
        super(NeuralSort, self).__init__()
        self.hard = hard
        self.tau = tau

    def forward(self, scores):
        """
        scores: elements to be sorted. Typical shape: batch_size x n x 1
        """
        if not self.hard:
            scores = scores + sample_gumbel(scores.shape, 1)
        scores = scores.unsqueeze(-1)
        bsize = scores.size()[0]
        dim = scores.size()[1]
        one = torch.cuda.FloatTensor(dim, 1).fill_(1)

        A_scores = torch.abs(scores - scores.permute(0, 2, 1))
        B = torch.matmul(A_scores, torch.matmul(
            one, torch.transpose(one, 0, 1)))
        scaling = (dim + 1 - 2 * (torch.arange(dim) + 1)
                   ).type(torch.cuda.FloatTensor)
        C = torch.matmul(scores, scaling.unsqueeze(0))

        P_max = (C-B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / self.tau)

        if self.hard:
            P = torch.zeros_like(P_hat, device='cuda')
            b_idx = torch.arange(bsize).repeat([1, dim]).view(dim, bsize).transpose(
                dim0=1, dim1=0).flatten().type(torch.cuda.LongTensor)
            r_idx = torch.arange(dim).repeat(
                [bsize, 1]).flatten().type(torch.cuda.LongTensor)
            c_idx = torch.argmax(P_hat, dim=-1).flatten()  # this is on cuda
            brc_idx = torch.stack((b_idx, r_idx, c_idx))

            P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
            P_hat = (P-P_hat).detach() + P_hat
        return P_hat

# Used to sample top k, returning 1's in the top k indices and 0's otherwise
def get_top_k(x, k=8, zero_value=0):
    top_k_indices = torch.topk(x, k, dim=1).indices
    result = torch.full(x.shape, zero_value).cuda()
    return result.scatter_(1, top_k_indices, 1)

# x is b_size x 24, assume k is the number of audio layers, vision is 3x audio
# If not valid
def get_top_k_unequal(x, k=8, zero_value=0):
    budget_max = k
    k = min(k, 24)
    top_k_indices = torch.topk(x, k, dim=1).indices # Returns top_k (b_size x k) 
    for batch_idx in range(top_k_indices.shape[0]):
        budget_count = 0
        for i in range(k):
            if budget_count <= budget_max - 3:
                budget_count += 3 if top_k_indices[batch_idx][i] < 12 else 1 # vision contributes 3
            elif budget_count < budget_max and top_k_indices[batch_idx][i] >= 12:
                budget_count += 1 # Greater than k-3 but less than k, fill with ones
            else:
                top_k_indices[batch_idx][i] = 0 # Set to index zero, index 0 is always chosen anyways
    result = torch.full(x.shape, zero_value).cuda()
    return result.scatter_(1, top_k_indices, 1) # hopefully scatter works w repeats




def sample_gumbel(shape, scale):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + 1e-12) + 1e-12) * scale

# Perform gumbel softmax sampling with specified temperature and scale
def gumbel_softmax_sample(logits, temperature, scale=1):
    y = logits + sample_gumbel(logits.size(), scale)
    return nn.functional.softmax(y / temperature, dim=-1)

def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(-0.005, 0.005)

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe.cuda()


    
# This is the Controller that allocates layers among modalities in accordance to modality quality
@MODELS.register_module()
class ConvLayerController(nn.Module):
    
    # Total layer refers to the total available compute budget
    # TODO CHANGE THIS back to 256
    def __init__(self, embed_dim=256, num_classes=8, total_layers_img=12, total_layers_lidar=8, layer_budget=8):
        super(ConvLayerController, self).__init__()

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

        self.additional_layers = layer_budget # how many layers we are allocating, first layer is always 1
        self.layer_budget = layer_budget
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, 200, bias=False),
            nn.ReLU(),
            nn.Linear(200, total_layers_img + total_layers_lidar, bias=False) # 12 layers in each ViT, we want to generate a one-hot at the end
        )
        self.noise_output = nn.Sequential(
            nn.Linear(embed_dim, 250),
            nn.ReLU(),
            nn.Linear(250, num_classes) # We softmax over these three logits (clean, noisy lidar, noisy image)
        )
        self.output_head.apply(init_weights) # init output head weights to be very small, this forces it to be responsive to the noise embedding value
        self.logits_memory = [] # Used during inference only
        self.grad_accum = None

    def manual_scatter(self, voxel_features, coors, batch_size, grid_shape, feat_channels):
        H, W = grid_shape
        C = feat_channels
        canvas = torch.zeros((batch_size, C, H, W), device=voxel_features.device, dtype=voxel_features.dtype)
        canvas[coors[:, 0], :, coors[:, 2], coors[:, 3]] = voxel_features
        return canvas
    
    # Temperature define peakiness of the gumbel softmax
    def forward(self, voxel_features, controller_coors, raw_image, flatformer_layers, temp=1, discretization_method = 'neuralsort'):

        current_epoch = 1
        if self.training:
            message_hub = MessageHub.get_current_instance()
            current_epoch = message_hub.get_info('epoch') + 1

        B = raw_image.shape[0]

        bev_grid = self.manual_scatter(voxel_features, controller_coors, B, (720, 720), 128)
    
        voxel_noise_features = torch.reshape(self.voxel_extractor(bev_grid), (B, -1))
        voxel_noise_features = self.voxel_adapter(voxel_noise_features)
        
        raw_image = rearrange(raw_image, 'b n c h w -> (b n) c h w')
        img_noise_features = self.img_extractor(raw_image)
        img_noise_features = torch.reshape(img_noise_features, (B, -1))
        img_noise_features = self.img_adapter(img_noise_features)

        joint_embed = torch.cat([voxel_noise_features, img_noise_features], dim=-1)

        logits = self.output_head(joint_embed) # logits are of shape B_size x 24 \
        
        # # First logit for each modality will ALWAYS be chosen, set it to -99 to avoid influencing the softmax too heavily
        # logits[:, 0] = -99
        # logits[:, flatformer_layers] = -99

        # Get the predicted noise for each of the modalities
        predicted_noise = self.noise_output(joint_embed) # b_size x 2 (img and depth)
        if discretization_method == 'admn':
            if self.training:
                gumbel_samples = gumbel_softmax_sample(logits, temperature=temp, scale=0.01)
            else: # If this is during inference, we don't do any gumbel softmax sampling
                gumbel_samples = logits
            discretized = get_top_k(gumbel_samples, k=self.additional_layers, zero_value=0)
            discretized = torch.reshape(discretized, (B, -1))
            discretized[:, 0] = 1 # Set the first layer to always chosen
            discretized[:, flatformer_layers] = 1
            gumbel_samples = torch.reshape(gumbel_samples, (B, -1))
            logits = torch.reshape(logits, (B, -1))
            if get_dist_info()[0] == 0:
                print('\nLogits LiDAR:', logits[0][:flatformer_layers])
                print('Logits Image:', logits[0][flatformer_layers:])
                print('LiDAR:', discretized[0][:flatformer_layers])
                print('Image:',  discretized[0][flatformer_layers:])
            return gumbel_samples + (discretized - gumbel_samples).detach(), predicted_noise
        elif discretization_method == 'straight_through': # No softmax sampling used
            gumbel_samples = logits
            discretized = get_top_k(gumbel_samples, k=self.additional_layers, zero_value=0)
            discretized = torch.reshape(discretized, (B, -1, 12))
            discretized[:, :, 0] = 1
            gumbel_samples = torch.reshape(gumbel_samples, (B, -1, 12))
            logits = torch.reshape(logits, (B, -1, 12))
            # print('Image:', logits[0][0])
            # print('Depth:', logits[0][1])
            print('Image:', discretized[0][0])
            print('Audio:',  discretized[0][1])
            return gumbel_samples + (discretized - gumbel_samples).detach(), predicted_noise
        elif discretization_method == 'progressive': # In theory this would require us to progressively adjust the temperature w gumbel softmax only
            if self.training:
                sampler = SubsetOperator(self.additional_layers, tau=temp, hard=False)
                discretized = sampler(logits)
            else:
                discretized = get_top_k(logits, k=self.additional_layers, zero_value=0)
            discretized = torch.reshape(discretized, (B, -1))
            # discretized[:, 0] = 1 # Set the first layer to always chosen
            # discretized[:, flatformer_layers] = 1
            logits = torch.reshape(logits, (B, -1))
            if get_dist_info()[0] == 0:
                print('\nLiDAR Logits:', logits[0][:flatformer_layers])
                print('Image Logits:', logits[0][flatformer_layers:])
                print('LiDAR:', discretized[0][:flatformer_layers])
                print('Image:',  discretized[0][flatformer_layers:])
            return discretized, predicted_noise
        elif discretization_method == 'sigmoid':
            if self.training:
                weights = torch.sigmoid(logits)
            else: # If this is during inference, we don't do any gumbel softmax sampling
                weights = get_top_k(logits, k=self.additional_layers, zero_value=0)
            weights = torch.reshape(weights, (B, -1))
            if get_dist_info()[0] == 0:
                print('\nLiDAR Logits:', logits[0][:flatformer_layers])
                print('Image Logits:', logits[0][flatformer_layers:])
                print('LiDAR', weights[0][:flatformer_layers])
                print('Image', weights[0][flatformer_layers:])
            return weights, predicted_noise
        elif discretization_method == 'neuralsort':
            if self.training:
                sampler = NeuralSort(tau=0.5/current_epoch, hard=False) # used to be 0.5
                weights = sampler(logits) # B_size, N, N
                weights = torch.sum(weights[:, :self.additional_layers], dim=1) # B_size, N
            else:
                weights = get_top_k(logits, k=self.additional_layers, zero_value=0)
            # weights[:, 0] = 1 # Set the first layer to always chosen
            # weights[:, flatformer_layers] = 1
            # if get_dist_info()[0] == 0:
            #     print('\nController LiDAR Logits:', logits[0][:flatformer_layers])
            #     print('Controller Image Logits:', logits[0][flatformer_layers:])
            #     print('Controller LiDAR', weights[0][:flatformer_layers])
            #     print('Controller Image', weights[0][flatformer_layers:])
            return weights, predicted_noise
        else:
            raise Exception('Invalid discretization')
        

  
# This is the Controller that allocates layers among modalities in accordance to modality quality
@MODELS.register_module()
class UniversalConvLayerController(nn.Module):
    
    # Total layer refers to the total available compute budget
    # TODO CHANGE THIS back to 256
    def __init__(self, embed_dim=128, num_classes=8, total_layers_img=12, total_layers_lidar=8, layer_budgets=[4, 6, 8, 10]):
        super(UniversalConvLayerController, self).__init__()

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
        self.output_head.apply(init_weights) # init output head weights to be very small, this forces it to be responsive to the noise embedding value
        self.logits_memory = [] # Used during inference only
        self.grad_accum = None

    def manual_scatter(self, voxel_features, coors, batch_size, grid_shape, feat_channels):
        H, W = grid_shape
        C = feat_channels
        canvas = torch.zeros((batch_size, C, H, W), device=voxel_features.device, dtype=voxel_features.dtype)
        canvas[coors[:, 0], :, coors[:, 2], coors[:, 3]] = voxel_features
        return canvas
    
    # Temperature define peakiness of the gumbel softmax
    def forward(self, voxel_features, controller_coors, raw_image, flatformer_layers, temp=1, discretization_method = 'neuralsort'):

        B = raw_image.shape[0]
        current_epoch = 1
        current_budget = self.layer_budgets[int(random.random() * len(self.layer_budgets))]
        
        if self.training:
            message_hub = MessageHub.get_current_instance()
            current_epoch = message_hub.get_info('epoch') + 1
            

        budget_token = self.layer_token_dict[str(current_budget)]
        budget_token = budget_token.repeat(B, 1) # expand dim

        bev_grid = self.manual_scatter(voxel_features, controller_coors, B, (720, 720), 128)
    
        voxel_noise_features = torch.reshape(self.voxel_extractor(bev_grid), (B, -1))
        voxel_noise_features = self.voxel_adapter(voxel_noise_features)
        
        raw_image = rearrange(raw_image, 'b n c h w -> (b n) c h w')
        img_noise_features = self.img_extractor(raw_image)
        img_noise_features = torch.reshape(img_noise_features, (B, -1))
        img_noise_features = self.img_adapter(img_noise_features)

        joint_embed = torch.cat([voxel_noise_features, img_noise_features, budget_token], dim=-1)

        logits = self.output_head(joint_embed) # logits are of shape B_size x 24 \
        
        # # First logit for each modality will ALWAYS be chosen, set it to -99 to avoid influencing the softmax too heavily
        # logits[:, 0] = -99
        # logits[:, flatformer_layers] = -99

        # Get the predicted noise for each of the modalities
        predicted_noise = self.noise_output(joint_embed) # b_size x 2 (img and depth)
        if discretization_method == 'admn':
            if self.training:
                gumbel_samples = gumbel_softmax_sample(logits, temperature=temp, scale=0.01)
            else: # If this is during inference, we don't do any gumbel softmax sampling
                gumbel_samples = logits
            discretized = get_top_k(gumbel_samples, k=self.additional_layers, zero_value=0)
            discretized = torch.reshape(discretized, (B, -1))
            discretized[:, 0] = 1 # Set the first layer to always chosen
            discretized[:, flatformer_layers] = 1
            gumbel_samples = torch.reshape(gumbel_samples, (B, -1))
            logits = torch.reshape(logits, (B, -1))
            if get_dist_info()[0] == 0:
                print('\nLogits LiDAR:', logits[0][:flatformer_layers])
                print('Logits Image:', logits[0][flatformer_layers:])
                print('LiDAR:', discretized[0][:flatformer_layers])
                print('Image:',  discretized[0][flatformer_layers:])
            return gumbel_samples + (discretized - gumbel_samples).detach(), predicted_noise
        elif discretization_method == 'straight_through': # No softmax sampling used
            gumbel_samples = logits
            discretized = get_top_k(gumbel_samples, k=self.additional_layers, zero_value=0)
            discretized = torch.reshape(discretized, (B, -1, 12))
            discretized[:, :, 0] = 1
            gumbel_samples = torch.reshape(gumbel_samples, (B, -1, 12))
            logits = torch.reshape(logits, (B, -1, 12))
            # print('Image:', logits[0][0])
            # print('Depth:', logits[0][1])
            print('Image:', discretized[0][0])
            print('Audio:',  discretized[0][1])
            return gumbel_samples + (discretized - gumbel_samples).detach(), predicted_noise
        elif discretization_method == 'progressive': # In theory this would require us to progressively adjust the temperature w gumbel softmax only
            if self.training:
                sampler = SubsetOperator(self.additional_layers, tau=temp, hard=False)
                discretized = sampler(logits)
            else:
                discretized = get_top_k(logits, k=self.additional_layers, zero_value=0)
            discretized = torch.reshape(discretized, (B, -1))
            # discretized[:, 0] = 1 # Set the first layer to always chosen
            # discretized[:, flatformer_layers] = 1
            logits = torch.reshape(logits, (B, -1))
            if get_dist_info()[0] == 0:
                print('\nLiDAR Logits:', logits[0][:flatformer_layers])
                print('Image Logits:', logits[0][flatformer_layers:])
                print('LiDAR:', discretized[0][:flatformer_layers])
                print('Image:',  discretized[0][flatformer_layers:])
            return discretized, predicted_noise
        elif discretization_method == 'sigmoid':
            if self.training:
                weights = torch.sigmoid(logits)
            else: # If this is during inference, we don't do any gumbel softmax sampling
                weights = get_top_k(logits, k=self.additional_layers, zero_value=0)
            weights = torch.reshape(weights, (B, -1))
            if get_dist_info()[0] == 0:
                print('\nLiDAR Logits:', logits[0][:flatformer_layers])
                print('Image Logits:', logits[0][flatformer_layers:])
                print('LiDAR', weights[0][:flatformer_layers])
                print('Image', weights[0][flatformer_layers:])
            return weights, predicted_noise
        elif discretization_method == 'neuralsort':
            if self.training:
                sampler = NeuralSort(tau=0.5/current_epoch, hard=False) # used to be 0.5
                weights = sampler(logits) # B_size, N, N
                weights = torch.sum(weights[:, :current_budget], dim=1) # B_size, N
            else:
                weights = get_top_k(logits, k=current_budget, zero_value=0)
            # weights[:, 0] = 1 # Set the first layer to always chosen
            # weights[:, flatformer_layers] = 1
            if get_dist_info()[0] == 0:
                print('\nController LiDAR Logits:', logits[0][:flatformer_layers])
                print('Controller Image Logits:', logits[0][flatformer_layers:])
                print('Controller LiDAR', weights[0][:flatformer_layers])
                print('Controller Image', weights[0][flatformer_layers:])
            return weights, predicted_noise
        else:
            raise Exception('Invalid discretization')
        




'''
LiDAR: tensor([-99.0000,   0.5488,   0.5771,   0.5854,   0.6016,   0.6001,   0.5825,
          0.5596], device='cuda:0', dtype=torch.float16,
       grad_fn=<SliceBackward0>)
Image: tensor([-99.0000,  -4.9492,   0.6382,   0.6567,   0.1958,   0.6025,   0.5840,
          0.6528,  -3.5117,   0.5576,   0.6221,  -4.2305]
'''



# This is the Controller that allocates layers among modalities in accordance to modality quality
class ConvLayerControllerUnequal(nn.Module):
    
    # Total layer refers to the total available compute budget
    # TODO CHANGE THIS back to 256
    def __init__(self, embed_dim=512, depth=4, num_heads=4, mlp_ratio=4, num_modalities = 2, total_layers=6):
        super(ConvLayerControllerUnequal, self).__init__()

        self.encoder_dict = nn.ModuleDict({
            'image': nn.Sequential(
                transforms.Resize((100, 100)),
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.MaxPool2d((3, 3)),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(5, 5)),
                nn.BatchNorm2d(num_features=3),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(1587, embed_dim)
            ),
            'audio': nn.Sequential(
                transforms.Resize((128, 512)),
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.MaxPool2d((3, 6)),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(5, 5)),
                nn.BatchNorm2d(num_features=3),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(7488, embed_dim)
            )
        })
        # Fuses the information together to output joint layer config of all modalities
        self.combiner_encoder = TransformerEnc(embed_dim, depth, num_heads, dim_head=embed_dim//num_heads, mlp_dim=mlp_ratio * embed_dim)
        self.cls = nn.Parameter(torch.randn(1, embed_dim))
        self.additional_layers = total_layers - 4 # how many layers we are allocating, vision is 3 layers, audio is 1
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, 200, bias=False),
            nn.ReLU(),
            nn.Linear(200, 12 * num_modalities, bias=False) # 12 layers in each ViT, we want to generate a one-hot at the end
        )
        self.noise_output = nn.Sequential(
            nn.Linear(embed_dim, 250),
            nn.ReLU(),
            nn.Linear(250, 4),
        )
        self.output_head.apply(init_weights) # init output head weights to be very small, this forces it to be responsive to the noise embedding value
        self.logits_memory = [] # Used during inference only
        self.grad_accum = None

    # Used during backward hook to see the gradient values, will help if I want to implement gradient clipping to prevent weird behavior
    # def print_grad(self, grad):
    #     #print("THE GRADIENT IS", grad[0])
    #     if self.grad_accum is None:
    #         self.grad_accum = torch.mean(grad, dim=0)
    #     else:
    #         self.grad_accum += torch.mean(grad, dim=0)
    #     #return torch.clip(grad, -0.1, 0.1)
    #     return grad 

    # Temperature define peakiness of the gumbel softmax
    def forward(self, batched_data, valid_mods, temp=1, discretization_method = 'admn'):
        audio_data, img_data, labels = batched_data
        conv_embeds = []
        if 'image' in valid_mods:
            img_data = rearrange(img_data, 'b s c h w -> (b s) c h w')
            out = self.encoder_dict['image'](img_data)
            out = rearrange(out, '(b s) e-> b s e', b = audio_data.shape[0])
            conv_embeds.append(out)
        if 'audio' in valid_mods:
            out = self.encoder_dict['audio'](audio_data)
            if (len(out.shape) == 1):
                out = torch.unsqueeze(out, dim=0)
            out = torch.unsqueeze(out, dim=1)
            conv_embeds.append(out)

        conv_embeds = torch.cat(conv_embeds, dim=1)
        B = conv_embeds.shape[0]
        
        cls_tokens = self.cls.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, conv_embeds), dim=1)
        x += positionalencoding1d(self.cls.shape[-1], x.shape[1])
        x = self.combiner_encoder(x)[:, 0] # Get CLS output

        logits = self.output_head(x) # logits are of shape B_size x 24 \
        # First logit for each modality will ALWAYS be chosen, set it to -99 to avoid influencing the softmax too heavily
        
        logits[:, 0] = -99
        logits[:, 12] = -99

        # Get the predicted noise for each of the modalities
        predicted_noise = self.noise_output(x) # b_size x 2 (img and depth)

        if discretization_method == 'admn':
            if self.training:
                gumbel_samples = gumbel_softmax_sample(logits, temperature=temp, scale=0.1)
            else: # If this is during inference, we don't do any gumbel softmax sampling
                gumbel_samples = logits
            discretized = get_top_k_unequal(gumbel_samples, k=self.additional_layers, zero_value=0)
            discretized = torch.reshape(discretized, (B, -1, 12))
            discretized[:, :, 0] = 1 # Set the first layer to always chosen
            gumbel_samples = torch.reshape(gumbel_samples, (B, -1, 12))
            logits = torch.reshape(logits, (B, -1, 12))
            print('Image:', logits[0][0])
            print('Depth:', logits[0][1])
            print('Image:', discretized[0][0])
            print('Audio:',  discretized[0][1])
            return gumbel_samples + (discretized - gumbel_samples).detach(), predicted_noise
        elif discretization_method == 'straight_through': # No softmax sampling used
            gumbel_samples = logits
            discretized = get_top_k_unequal(gumbel_samples, k=self.additional_layers, zero_value=0)
            discretized = torch.reshape(discretized, (B, -1, 12))
            discretized[:, :, 0] = 1
            gumbel_samples = torch.reshape(gumbel_samples, (B, -1, 12))
            logits = torch.reshape(logits, (B, -1, 12))
            # print('Image:', logits[0][0])
            # print('Depth:', logits[0][1])
            print('Image:', discretized[0][0])
            print('Audio:',  discretized[0][1])
            return gumbel_samples + (discretized - gumbel_samples).detach(), predicted_noise
        elif discretization_method == 'progressive': # In theory this would require us to progressively adjust the temperature w gumbel softmax only
            if self.training:
                sampler = SubsetOperator(self.additional_layers, tau=temp, hard=False)
                discretized = sampler(logits)
            else:
                discretized = get_top_k_unequal(logits, k=self.additional_layers, zero_value=0)
            discretized = torch.reshape(discretized, (B, -1, 12))
            discretized[:, :, 0] = 1
            logits = torch.reshape(logits, (B, -1, 12))
            # print('Image:', logits[0][0])
            # print('Depth:', logits[0][1])
            print('Image:', discretized[0][0])
            print('Audio:',  discretized[0][1])
            return discretized, predicted_noise
        else:
            raise Exception('Invalid discretization')


class Conv_Controller_AE(nn.Module):
    # Total layer refers to the total available compute budget
    def __init__(self, embed_dim=512, depth=4, num_heads=4, mlp_ratio=4):
        super(Conv_Controller_AE, self).__init__()
        # Do the respective resizes before the AE function
        self.encoder_dict = nn.ModuleDict({
            'image': nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.MaxPool2d((3, 3)),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(5, 5)),
                nn.BatchNorm2d(num_features=3),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(1587, embed_dim)
            ),
            'audio': nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.MaxPool2d((3, 6)),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(5, 5)),
                nn.BatchNorm2d(num_features=3),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(7488, embed_dim)
            )
        })
        self.decoder_dict = nn.ModuleDict({
            'image': nn.Sequential(
                nn.ConvTranspose2d(in_channels=1, out_channels=32, kernel_size=(14, 14)),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(12, 8), stride=(3, 2)),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(5, 5)),
            ),
            'audio': nn.Sequential(
                nn.ConvTranspose2d(in_channels=1, out_channels=32, kernel_size=(10, 8), stride=(2, 3)),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(7, 8), stride=(3, 5)),
                nn.BatchNorm2d(num_features=32),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(5, 5)),
            )
        })
        # Fuses the information together to output joint layer config of all modalities
        self.combiner_encoder = TransformerEnc(embed_dim, depth, num_heads, dim_head=embed_dim//num_heads, mlp_dim=mlp_ratio * embed_dim)
        self.cls = nn.Parameter(torch.randn(1, embed_dim))
        
    def forward(self, batched_data, valid_mods):
        audio_data, img_data, labels = batched_data
        conv_embeds = []
        if 'image' in valid_mods:
            img_data = rearrange(img_data, 'b s c h w -> (b s) c h w')      
            out = self.encoder_dict['image'](img_data)
            out = rearrange(out, '(b s) e-> b s e', b = audio_data.shape[0])
            conv_embeds.append(out)
        if 'audio' in valid_mods:
            out = self.encoder_dict['audio'](audio_data)
            if (len(out.shape) == 1):
                out = torch.unsqueeze(out, dim=0)
            out = torch.unsqueeze(out, dim=1)
            conv_embeds.append(out)

        conv_embeds = torch.cat(conv_embeds, dim=1)
        B = conv_embeds.shape[0]
        
        cls_tokens = self.cls.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, conv_embeds), dim=1)
        x += positionalencoding1d(self.cls.shape[-1], x.shape[1])
        x = self.combiner_encoder(x)[:, 0] # Get CLS output
        x = torch.reshape(x, (-1, 1, 16, 32))
        
        img_recon = self.decoder_dict['image'](x)
        audio_recon = self.decoder_dict['audio'](x)
        return img_recon, audio_recon




# This is the Controller that allocates layers among modalities in accordance to modality quality
class AdaMML_Modality_Selector(nn.Module):
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4):
        super(AdaMML_Modality_Selector, self).__init__()
        self.encoder_dict = nn.ModuleDict({
            'image': nn.Sequential(
                transforms.Resize((100, 100)),
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.MaxPool2d((3, 3)),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(5, 5)),
                nn.BatchNorm2d(num_features=3),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(1587, embed_dim)
            ),
            'audio': nn.Sequential(
                transforms.Resize((128, 512)),
                nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.Conv2d(in_channels=6, out_channels=6, kernel_size=(10, 10)),
                nn.BatchNorm2d(num_features=6),
                nn.ReLU(),
                nn.MaxPool2d((3, 6)),
                nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(5, 5)),
                nn.BatchNorm2d(num_features=3),
                nn.ReLU(),
                nn.Flatten(1, -1),
                nn.Linear(7488, embed_dim)
            )
        })
        # Fuses the information together to output joint layer config of all modalities
        self.combiner_encoder = TransformerEnc(embed_dim, depth, num_heads, dim_head=embed_dim//num_heads, mlp_dim=mlp_ratio * embed_dim)
        self.cls = nn.Parameter(torch.randn(1, embed_dim))
        
          # output head for the 3-way modality selector
        self.mod_sel_head = nn.Sequential(
            nn.Linear(embed_dim, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 3, bias=False)  # 3 classes: Img only / Dep only / Both
        )

        self.temperature = 5.0

    # Temperature define peakiness of the gumbel softmax
    def forward(self, batched_data):
        audio_data, img_data, labels = batched_data
        conv_embeds = []
        img_data = rearrange(img_data, 'b s c h w -> (b s) c h w')
        out = self.encoder_dict['image'](img_data)
        out = rearrange(out, '(b s) e-> b s e', b = audio_data.shape[0])
        conv_embeds.append(out)
        out = self.encoder_dict['audio'](audio_data)
        if (len(out.shape) == 1):
            out = torch.unsqueeze(out, dim=0)
        out = torch.unsqueeze(out, dim=1)
        conv_embeds.append(out)

        conv_embeds = torch.cat(conv_embeds, dim=1)
        B = conv_embeds.shape[0]
        
        cls_tokens = self.cls.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, conv_embeds), dim=1)
        x += positionalencoding1d(self.cls.shape[-1], x.shape[1])
        x = self.combiner_encoder(x)[:, 0] # Get CLS output

        logits = self.mod_sel_head(x) # B * 3

        tau = self.temperature

        if self.training:
            soft = F.gumbel_softmax(logits, tau=tau, hard=False) # B * 3
            # construct had one-hot
            idx = torch.argmax(soft, dim=-1)
            hard = F.one_hot(idx, num_classes=3).float()

            # STE
            samp = soft + (hard - soft).detach()
        else:
            # use hard argmax for inference
            idx = torch.argmax(logits, dim=-1) # B
            samp = F.one_hot(idx, num_classes=3).float().to(logits.device) # B * 3

        # project the 3-way one-hot to binary decision
        mask = torch.zeros(B, 2, device=logits.device)
        mask[:, 0] = samp[:, 0] + samp[:, 1]
        mask[:, 1] = samp[:, 2] + samp[:, 1]

        return samp, mask, logits
    
