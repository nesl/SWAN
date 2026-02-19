import copy

import numpy as np
import mmcv
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MyComputeMultiViewCalib(BaseTransform):
    """Compute multi-view calibration matrices from per-camera info in results['images'].
    Some how we don't have lidar2img so this is necessary
    The standard LoadMultiViewImageFromFiles and det3d_dataset only set lidar2img for the
    default_cam_key (single view). For LC fusion the head needs (num_views, 4, 4) projection
    matrices. This transform replicates BEVFusion's BEVLoadMultiViewImageFromFiles calibration
    logic (cam2img, lidar2cam, cam2lidar, lidar2img) without modifying BEVFusion code.

    Added keys:
        lidar2img  (num_views, 4, 4): cam2img @ lidar2cam per view.
        cam2img    (num_views, 4, 4): 4x4 intrinsic (3x3 padded to 4x4) per view.
        lidar2cam  (num_views, 4, 4): extrinsic per view.
        cam2lidar  (num_views, 4, 4): inverse extrinsic per view.
        ori_cam2img, ori_lidar2img: deep copies before any augmentation.
    """

    def transform(self, results: dict) -> dict:
        cam2img, lidar2cam, cam2lidar, lidar2img = [], [], [], []
        for _, cam_item in results['images'].items():
            l2c = np.array(cam_item['lidar2cam'], dtype=np.float32)
            c2l = np.eye(4, dtype=np.float32)
            c2l[:3, :3] = l2c[:3, :3].T
            c2l[:3, 3:4] = -l2c[:3, :3].T @ l2c[:3, 3:4]

            c2i = np.eye(4, dtype=np.float32)
            c2i[:3, :3] = np.array(cam_item['cam2img'], dtype=np.float32)

            lidar2cam.append(l2c)
            cam2img.append(c2i)
            cam2lidar.append(c2l)
            lidar2img.append(c2i @ l2c)

        results['cam2img'] = np.stack(cam2img, axis=0)
        results['lidar2cam'] = np.stack(lidar2cam, axis=0)
        results['cam2lidar'] = np.stack(cam2lidar, axis=0)
        results['lidar2img'] = np.stack(lidar2img, axis=0)
        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])
        results['ori_lidar2img'] = copy.deepcopy(results['lidar2img'])
        return results


@TRANSFORMS.register_module()
class MyImgToList(BaseTransform):
    """Convert results['img'] from ndarray to a list of per-view arrays."""

    def transform(self, results):
        img = results['img']
        if not isinstance(img, list):
            if img.ndim == 4:
                results['img'] = [img[i] for i in range(img.shape[0])]
            else:
                results['img'] = [img]
        return results


@TRANSFORMS.register_module()
class MyNormalize(BaseTransform):
    """Normalize each image in a multi-view image list."""

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def transform(self, results):
        results['img'] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results['img']
        ]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results


@TRANSFORMS.register_module()
class MyResize(BaseTransform):
    """Resize each image in a multi-view image list."""

    def __init__(self, scale, keep_ratio=False, interpolation='bilinear'):
        if isinstance(scale, int):
            self.scale = (scale, scale)
        else:
            self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation

    def transform(self, results):
        resized = []
        scale_factor = None
        for img in results['img']:
            if self.keep_ratio:
                img_out, scale_factor = mmcv.imrescale(
                    img, self.scale,
                    interpolation=self.interpolation,
                    return_scale=True)
            else:
                img_out, w_scale, h_scale = mmcv.imresize(
                    img, self.scale,
                    interpolation=self.interpolation,
                    return_scale=True)
                scale_factor = (w_scale, h_scale)
            resized.append(img_out)
        results['img'] = resized
        results['img_shape'] = resized[0].shape[:2]
        if scale_factor is not None:
            if isinstance(scale_factor, (int, float)):
                # imrescale returns a single uniform scale
                results['scale_factor'] = np.array([scale_factor, scale_factor], dtype=np.float32)
            else:
                results['scale_factor'] = np.array([scale_factor[0], scale_factor[1]], dtype=np.float32)
        return results


@TRANSFORMS.register_module()
class MyPad(BaseTransform):
    """Pad each image in a multi-view image list."""

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        assert size is not None or size_divisor is not None
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val

    def transform(self, results):
        padded = []
        for img in results['img']:
            if self.size_divisor is not None:
                h, w = img.shape[:2]
                pad_h = int(np.ceil(h / self.size_divisor)) * self.size_divisor
                pad_w = int(np.ceil(w / self.size_divisor)) * self.size_divisor
                size = (pad_h, pad_w)
            else:
                size = self.size[::-1]
            pad_val = self.pad_val
            if isinstance(pad_val, int) and img.ndim == 3:
                pad_val = tuple(pad_val for _ in range(img.shape[2]))
            padded.append(mmcv.impad(img, shape=size, pad_val=pad_val))
        results['img'] = padded
        results['pad_shape'] = padded[0].shape
        results['input_shape'] = padded[0].shape[:2]
        return results
