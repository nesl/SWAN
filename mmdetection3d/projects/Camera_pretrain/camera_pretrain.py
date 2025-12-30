from typing import List, Optional, Dict
import torch
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



def show_image(image, title=''):
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def vis_image(ori_img, pred_img, mask, model, out_dir, sample_idx):
    from pathlib import Path
    Path(f'{out_dir}/cam').mkdir(parents=True, exist_ok=True)
    
    ori_img = model.camera_decoder.patchify(ori_img)
    mean = ori_img.mean(dim=-1, keepdim=True)
    var = ori_img.var(dim=-1, keepdim=True)
    ori_img = model.camera_decoder.unpatchify(ori_img)
    x = torch.einsum('nchw->nhwc', ori_img).detach().cpu()

    pred_img = pred_img * (var + 1.e-6)**.5 + mean
    y = model.camera_decoder.unpatchify(pred_img)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.camera_decoder.final_patch_size**2 *3)
    mask = model.camera_decoder.unpatchify(mask)
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
    Camera-only Pretraining Model (MAE Style).
    Follows MMEngine Base3DDetector interface.
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
            # Dont know why in validation image is list
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
        # Returns: latent features, mask, and restoration ids
        latent, mask, ids_restore = self.img_backbone(imgs, camera_only=True)

        return {
            'latent': latent,
            'mask': mask,
            'ids_restore': ids_restore,
            'imgs_reshaped': imgs,
            'input_shape': (B, N, C, H, W)
        }

    def loss(self, batch_inputs_dict: dict, batch_data_samples: List[Det3DDataSample], **kwargs) -> dict:
        """
        Forward pass for training.
        """
        # 1. Extract Features (Encoder)
        feat_dict = self.extract_feat(batch_inputs_dict)
        
        latent = feat_dict['latent']
        ids_restore = feat_dict['ids_restore']
        imgs_reshaped = feat_dict['imgs_reshaped']
        mask = feat_dict['mask']

        # 2. Forward Head (Decoder)
        # In MAE, the decoder takes latent + ids to reconstruct
        # Note: We pass ids_restore so the decoder knows where to put the mask tokens
        pred_imgs = self.img_head(latent, ids_restore)

        # 3. Calculate Loss
        # The decoder usually contains the reconstruction loss logic (MSE on masked patches)
        losses = dict()
        losses['loss_mae_cam'] = self.img_head.forward_loss(imgs_reshaped, pred_imgs, mask)

        # 4. Visualization (Optional, restricted to rank 0)
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
        
        latent = feat_dict['latent']
        ids_restore = feat_dict['ids_restore']
        imgs_reshaped = feat_dict['imgs_reshaped']          # (B*N, C, H, W) normalized
        mask = feat_dict['mask']                            # (B*N, L), 0=keep, 1=masked

        # Forward decoder: predictions in patch space
        pred_patches = self.img_head(latent, ids_restore)   # (B*N, L, patch_size²*3)

        # === Correct reconstruction (same as vis_image) ===
        # 1. Patchify original images to compute per-patch statistics
        ori_patches = self.img_head.patchify(imgs_reshaped)  # (B*N, L, patch_size²*3)
        mean = ori_patches.mean(dim=-1, keepdim=True)        # (B*N, L, 1)
        var = ori_patches.var(dim=-1, keepdim=True)         # (B*N, L, 1)

        # 2. Denormalize predictions (even if norm_pix_loss=False, this is safe)
        pred_patches_denorm = pred_patches * (var + 1e-6)**0.5 + mean

        # 3. Unpatchify to full resolution
        reconstructed_imgs = self.img_head.unpatchify(pred_patches_denorm)  # (B*N, C, H, W)

        # Optional: reshape back to (B, N, C, H, W) for easier downstream use
        B_N = reconstructed_imgs.shape[0]
        C, H, W = reconstructed_imgs.shape[1:]
        reconstructed_imgs = reconstructed_imgs.view(
            -1, feat_dict['input_shape'][1], C, H, W  # B, N, C, H, W
        )  # Note: original input_shape is (B, N, C, H, W)

        # Also return original images in same shape for comparison
        original_imgs = imgs_reshaped.view(
            -1, feat_dict['input_shape'][1], C, H, W
        )

        return original_imgs, reconstructed_imgs

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None):
        pass
