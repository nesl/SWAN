import torch

lidar_checkpoint = torch.load('work_dirs/cmt_voxel_015_flatformer_layerdrop_group256_efficientvfe/epoch_40.pth')
img_checkpoint = torch.load('/workspace/mmdetection3d/work_dirs/cmt_swin_layerdrop/epoch_50.pth')
merged_checkpoint = dict(lidar_checkpoint)

for key in img_checkpoint['state_dict'].keys():
    if key not in lidar_checkpoint['state_dict'].keys():
        merged_checkpoint['state_dict'][key] = img_checkpoint['state_dict'][key]

torch.save(merged_checkpoint, 'unified_256_efficientvfe.pth')
print(merged_checkpoint['state_dict'].keys())