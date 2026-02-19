import torch

path = '/workspace/mmdetection3d/projects/BEVFusion/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth'
model = torch.load(path)

def transpose_weights(model, layer_names):
    for layer in layer_names:
        if layer in model['state_dict']:
            weight = model['state_dict'][layer]
            # Transpose weights from [N, C, D, H, W] to [C, D, H, W, N] (from torch.Size([16, 3, 3, 3, 5] to  torch.Size([3, 3, 3, 5, 16])
            weight = weight.permute(1, 2, 3, 4, 0) 
            model['state_dict'][layer] = weight 
   
# Fill with BEVFusion's layers, an example:
# mismatched layers
layer_names = [
    "pts_middle_encoder.conv_input.0.weight",
    "pts_middle_encoder.encoder_layers.encoder_layer1.0.conv1.weight",
    "pts_middle_encoder.encoder_layers.encoder_layer1.0.conv2.weight",
    "pts_middle_encoder.encoder_layers.encoder_layer1.1.conv1.weight",
    "pts_middle_encoder.encoder_layers.encoder_layer1.1.conv2.weight",
    "pts_middle_encoder.encoder_layers.encoder_layer1.2.0.weight",
    "pts_middle_encoder.encoder_layers.encoder_layer2.0.conv1.weight",
    "pts_middle_encoder.encoder_layers.encoder_layer2.0.conv2.weight",
    "pts_middle_encoder.encoder_layers.encoder_layer2.1.conv1.weight",
    "pts_middle_encoder.encoder_layers.encoder_layer2.1.conv2.weight",
    "pts_middle_encoder.encoder_layers.encoder_layer2.2.0.weight",
    "pts_middle_encoder.encoder_layers.encoder_layer3.0.conv1.weight",
    "pts_middle_encoder.encoder_layers.encoder_layer3.0.conv2.weight",
    "pts_middle_encoder.encoder_layers.encoder_layer3.1.conv1.weight",
    "pts_middle_encoder.encoder_layers.encoder_layer3.1.conv2.weight",
    "pts_middle_encoder.encoder_layers.encoder_layer3.2.0.weight",
    "pts_middle_encoder.encoder_layers.encoder_layer4.0.conv1.weight",
    "pts_middle_encoder.encoder_layers.encoder_layer4.0.conv2.weight",
    "pts_middle_encoder.encoder_layers.encoder_layer4.1.conv1.weight",
    "pts_middle_encoder.encoder_layers.encoder_layer4.1.conv2.weight",
    "pts_middle_encoder.conv_out.0.weight",
]


transpose_weights(model, layer_names)
torch.save(model, './bevfusion_fixed.pth')