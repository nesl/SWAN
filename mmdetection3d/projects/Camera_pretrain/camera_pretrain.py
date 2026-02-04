from typing import List, Optional, Dict
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmengine.dist import get_rank
from matplotlib import pyplot as plt
from mmdet3d.utils.typing_utils import OptSampleList
from mmdet3d.structures import bbox3d2result
import numpy as np


def calculate_psnr(original: torch.Tensor, reconstructed: torch.Tensor,
                   max_val: float = 1.0, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Calculate Peak Signal-to-Noise Ratio between original and reconstructed images.

    Args:
        original: Original images tensor
        reconstructed: Reconstructed images tensor of same shape as original
        max_val: Maximum possible pixel value (1.0 for normalized, 255 for uint8)
        mask: Optional mask tensor of same shape as original. If provided, PSNR is
              calculated only on regions where mask == 1.

    Returns:
        PSNR value in dB (scalar tensor)
    """
    if mask is not None:
        mask_bool = mask.bool()
        if mask_bool.sum() == 0:
            return torch.tensor(float('inf'))
        diff_sq = (original - reconstructed) ** 2
        mse = diff_sq[mask_bool].mean()
    else:
        mse = torch.mean((original - reconstructed) ** 2)

    if mse == 0:
        return torch.tensor(float('inf'))
    psnr = 20 * torch.log10(torch.tensor(max_val, device=mse.device)) - 10 * torch.log10(mse)
    return psnr


def show_image(img, title):
    """Helper function to display"""
    img = np.clip(img.numpy(), 0, 1)
    plt.imshow(img)
    plt.title(title, fontsize=16)
    plt.axis('off')

def vis_image(ori_img, pred_img, mask, model, out_dir, sample_idx):
    """visualize the original, masked, and reconstructed images."""
    from pathlib import Path
    Path(f'{out_dir}/cam').mkdir(parents=True, exist_ok=True)

    decoder = model.img_head

    ori_img_patches = decoder.patchify(ori_img)
    mean = ori_img_patches.mean(dim=-1, keepdim=True)
    var = ori_img_patches.var(dim=-1, keepdim=True)
    ori_img = decoder.unpatchify(ori_img_patches)
    x = torch.einsum('nchw->nhwc', ori_img).detach().cpu()

    pred_img = pred_img * (var + 1.e-6)**.5 + mean
    y = decoder.unpatchify(pred_img)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, decoder.patch_size**2 * 3)
    mask = decoder.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    im_masked = x * (1 - mask)
    im_paste = x * (1 - mask) + y * mask

    plt.rcParams['figure.figsize'] = [24, 12]
    B = min(x.shape[0], 6)
    for i in range(B):
        plt.subplot(B, 4, i*4+1)
        show_image(x[i], "original")
        plt.subplot(B, 4, i*4+2)
        show_image(im_masked[i], "masked")
        plt.subplot(B, 4, i*4+3)
        show_image(y[i], "reconstruction")
        plt.subplot(B, 4, i*4+4)
        show_image(im_paste[i], "reconstruction + visible")

    plt.savefig(f"{out_dir}/cam/{sample_idx}")
    plt.close()

@MODELS.register_module()
class CAMERA_PRETRAIN(Base3DDetector):
    """
    Camera-only Pretraining Model with MAE.
    Have to improve validation 
    """

    def __init__(self,
                 img_backbone: Optional[dict] = None,
                 img_head: Optional[dict] = None, # The Decoder acts as the head here
                 data_preprocessor: Optional[dict] = None,
                 **kwargs):
        super(CAMERA_PRETRAIN, self).__init__(
            data_preprocessor=data_preprocessor, 
            **kwargs)

        # Build Modules
        if img_backbone:
            self.img_backbone = MODELS.build(img_backbone)
        if img_head:
            self.img_head = MODELS.build(img_head)
            
        self.sample_idx = 0
        self.epoch = 0
        self.init_weights()

    def init_weights(self) -> None:
        super().init_weights()

    def extract_feat(self, batch_inputs_dict: dict,  batch_input_metas=Optional[List[Dict]],
        **kwargs,) -> dict:
        """
        Extract features from images. 
        Flow: Input -> Reshape -> Backbone (Encoder + Masking)
        """
        imgs = batch_inputs_dict.get('imgs', None)
        if imgs is None and 'img' in batch_inputs_dict:
            imgs = batch_inputs_dict['img']
            #print("number of batches: ", len(imgs))
            # e.g. 1
            #print("batch shape: ", imgs[0].shape)
            # e.g. number of images: torch.Size([6, 704, 3, 256])
            # Assemble into single tensor
            imgs = torch.stack(imgs, dim=0).cuda()
        if imgs is None:
            raise ValueError("imgs not found in batch_inputs_dict")

        #print("images shape before reshape: ", imgs.shape)
        
        B, N, W, C, H = imgs.shape
        imgs = imgs.permute(0, 1, 3, 4, 2).contiguous() 
        imgs = imgs.view(B * N, C, H, W)

        # Backbone (Encoder)
        # Expects: [B*N, C, H, W]
        # Returns: multi-scale features (list of 4 tensors), mask, and restoration ids
        encoder_outs, mask, ids_restore = self.img_backbone(imgs, camera_only=True)


        # Check:
        #total_patches = self.img_backbone.num_patches  # e.g., 250
        #output_tokens = encoder_outs[-1].shape[1]    # Should be 62 if 75% masking

        #print(f"Input: {total_patches} patches")
        #print(f"Output: {output_tokens} tokens")
        return {
            'encoder_outs': encoder_outs,  # List of 4 tensors from each stage
            'mask': mask,
            'ids_restore': ids_restore,
            'imgs_reshaped': imgs,
            'input_shape': (B, N, C, H, W)
        }

    def loss(self, batch_inputs_dict: dict, batch_data_samples: List[Det3DDataSample], **kwargs) -> dict:
        """
        Forward pass for training.
        """
        # Extract Features
        feat_dict = self.extract_feat(batch_inputs_dict)

        encoder_outs = feat_dict['encoder_outs']  # List of 4 multi-scale features
        ids_restore = feat_dict['ids_restore']
        imgs_reshaped = feat_dict['imgs_reshaped']
        mask = feat_dict['mask']

        # Decoder
        # Decoder takes multi-scale encoder features + ids to reconstruct
        # pass ids_restore so the decoder knows where to put the mask tokens
        pred_imgs = self.img_head(encoder_outs, ids_restore)

        # Calculate Loss
        losses = dict()
        losses['loss_mae_cam'] = self.img_head.forward_loss(imgs_reshaped, pred_imgs, mask)

        # Visualization (Optional, restricted to rank 0)
        if get_rank() == 0 and self.sample_idx % 500 == 0:
            # Assuming you have the vis_image helper available
            try:
                vis_image(imgs_reshaped[0:6], pred_imgs[0:6], mask[0:6], 
                          self, 'viz_cam', sample_idx=f'{self.sample_idx}.png')
            except Exception as e:
                pass # Don't crash training on viz error
            self.sample_idx += 1

        return losses

    def predict(self, batch_inputs_dict: dict, batch_data_samples: List[Det3DDataSample], **kwargs):
        """
        Inference pass. Returns:
            - original images (normalized, as input) 
            - reconstructed images (denormalized to original pixel distribution, full resolution)
        """
        feat_dict = self.extract_feat(batch_inputs_dict, batch_data_samples=batch_data_samples)

        encoder_outs = feat_dict['encoder_outs']  # List of 4 multi-scale features
        ids_restore = feat_dict['ids_restore']
        imgs_reshaped = feat_dict['imgs_reshaped']          # (B*N, C, H, W) normalized
        mask = feat_dict['mask']                            # (B*N, L), 0=keep, 1=masked
        #print("encoder_outs[0] shape:", encoder_outs[0].shape)
        #print("encoder_outs[1] shape:", encoder_outs[1].shape)
        #print("encoder_outs[2] shape:", encoder_outs[2].shape)
        #print("encoder_outs[3] shape:", encoder_outs[3].shape)
        # Forward decoder: predictions in patch space
        pred_patches = self.img_head(encoder_outs, ids_restore)   # (B*N, L, patch_size²*3)

        
        # Patchify original images to compute per-patch statistics
        ori_patches = self.img_head.patchify(imgs_reshaped)  # (B*N, L, patch_size²*3)
        mean = ori_patches.mean(dim=-1, keepdim=True)        # (B*N, L, 1)
        var = ori_patches.var(dim=-1, keepdim=True)         # (B*N, L, 1)

        # Denormalize predictions 
        pred_patches_denorm = pred_patches * (var + 1e-6)**0.5 + mean
       
        
        # Unpatchify to full resolution
        reconstructed_imgs = self.img_head.unpatchify(pred_patches_denorm)  # (B*N, C, H, W)

        # reshape back to (B, N, C, H, W) for easier downstream use
        B_N = reconstructed_imgs.shape[0]
        C, H, W = reconstructed_imgs.shape[1:]
        reconstructed_imgs = reconstructed_imgs.view(
            -1, feat_dict['input_shape'][1], C, H, W  # B, N, C, H, W
        )  

        # return original images in same shape for DEBUG
        original_imgs = imgs_reshaped.view(
            -1, feat_dict['input_shape'][1], C, H, W
        )
        # Reshape mask from (B*N, L) to (B, N, L) for return
        L = mask.shape[1]
        mask_patches = mask.view(-1, feat_dict['input_shape'][1], L)

        # Upsample mask from stage 3 (grid_size) to stage 0 resolution for PSNR
        H3, W3 = self.img_head.grid_size
        H0, W0 = self.img_head.stage_resolutions[0]
        patch_size = self.img_head.patch_size

        # mask shape: (B*N, L) where L = H3 * W3
        mask_2d = mask.reshape(mask.shape[0], H3, W3)
        mask_up = F.interpolate(mask_2d.unsqueeze(1).float(), size=(H0, W0), mode='nearest')
        mask_up = mask_up.squeeze(1)  # (B*N, H0, W0)

        # Expand mask to pixel space: each patch becomes patch_size x patch_size
        mask_pixels = mask_up.unsqueeze(-1).unsqueeze(-1)  # (B*N, H0, W0, 1, 1)
        mask_pixels = mask_pixels.expand(-1, -1, -1, patch_size, patch_size)  # (B*N, H0, W0, p, p)
        mask_pixels = mask_pixels.permute(0, 1, 3, 2, 4).reshape(
            mask.shape[0], H0 * patch_size, W0 * patch_size
        )  # (B*N, H, W)
        mask_pixels = mask_pixels.unsqueeze(1).expand(-1, C, -1, -1)  # (B*N, C, H, W)
        mask_pixels = mask_pixels.view(-1, feat_dict['input_shape'][1], C, H, W)  # (B, N, C, H, W)

        # 1. PSNR on full image
        psnr_full = calculate_psnr(original_imgs, reconstructed_imgs, max_val=1.0)
        #print(f"PSNR (full image): {psnr_full.item():.2f} dB")

        # 2. PSNR on visible/unmasked regions (mask == 0)
        visible_mask = 1 - mask_pixels
        psnr_visible = calculate_psnr(original_imgs, reconstructed_imgs, max_val=1.0, mask=visible_mask)
        #print(f"PSNR (visible regions): {psnr_visible.item():.2f} dB")

        # 3. PSNR on masked regions (mask == 1)
        psnr_masked = calculate_psnr(original_imgs, reconstructed_imgs, max_val=1.0, mask=mask_pixels)
        #print(f"PSNR (masked regions): {psnr_masked.item():.2f} dB")
        selfpsnr = calculate_psnr(original_imgs, original_imgs, max_val=1.0, mask=None)
        #print(f"PSNR (masked regions): {psnr_masked.item():.2f} dB")
        #print(f"PNSR self{selfpsnr.item():.2f} dB")
        return original_imgs, reconstructed_imgs, mask_patches, encoder_outs, psnr_full, psnr_visible, psnr_masked

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None):
        pass
