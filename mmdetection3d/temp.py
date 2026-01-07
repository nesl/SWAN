import torch

def convert_sparse_encoder_weights(src_ckpt, dst_ckpt):
    print(f"Loading checkpoint from {src_ckpt}...")
    checkpoint = torch.load(src_ckpt, map_location='cpu')
    
    # Handle if weights are wrapped in 'state_dict' key or are the root dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    print("Converting weights...")
    converted_count = 0
    
    for key, value in state_dict.items():
        # Filter for Sparse Encoder weights
        if 'pts_middle_encoder' in key and 'weight' in key:
            
            # We only want to permute the Convolution weights (which are 5D).
            # We must NOT touch BatchNorm weights (which are 1D).
            if value.dim() == 5:
                # Original Shape: [Out, Kd, Kh, Kw, In] -> [16, 3, 3, 3, 5]
                # Target Shape:   [Kd, Kh, Kw, In, Out] -> [3, 3, 3, 5, 16]
                # Permutation:    (1, 2, 3, 4, 0)
                
                original_shape = value.shape
                new_value = value.permute(1, 2, 3, 4, 0)
                
                # Update the dictionary in place
                state_dict[key] = new_value
                
                print(f"Permuted {key}: {original_shape} -> {new_value.shape}")
                converted_count += 1

    print(f"Finished! Converted {converted_count} layers.")
    
    # Save back to the structure
    if 'state_dict' in checkpoint:
        checkpoint['state_dict'] = state_dict
    else:
        checkpoint = state_dict
        
    torch.save(checkpoint, dst_ckpt)
    print(f"Saved converted checkpoint to {dst_ckpt}")

# --- USAGE ---
if __name__ == "__main__":
    # Path to your original CMT weights
    original_weights = '/workspace/mmdetection3d/paper_checkpoints/voxel0100_r50_800x320_epoch20.pth' # Or whatever your file is
    
    # Path where you want the new weights
    new_weights = 'paper_checkpoints/CONVERTED_voxel0100_r50_800x320_epoch20.pth'
    
    convert_sparse_encoder_weights(original_weights, new_weights)