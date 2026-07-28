# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from compressai.registry import register_dataset

import numpy as np 



@register_dataset("FeatureFolder")
class FeatureFolder(Dataset):
    """Load an feature folder database. Training and testing feature samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train", model_type="sd3", layer="blk05", task="tti", trun_flag=False, trun_low=-20, trun_high=20, quant_type="uniform", qsamples=0, bit_depth=1, patch_size=(512, 512), gt_path=None, preload=True):
        splitdir = Path(root) / split / model_type / layer

        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')

        self.samples = sorted(f for f in splitdir.iterdir() if f.is_file())

        self.transform = transform

        self.gt = {}
        if gt_path is not None:
            with open(gt_path, "r") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    base, idx = ln.split()
                    self.gt[base] = int(idx)

        self.model_type = model_type
        self.task = task
        self.trun_flag = trun_flag
        self.trun_low = trun_low
        self.trun_high = trun_high
        self.quant_type = quant_type
        self.qsamples = qsamples
        self.bit_depth = bit_depth
        self.patch_size = patch_size    #(height, width), must be the multiple of 64

        self._cache = None
        if preload:
            print(f"  Preloading {len(self.samples)} features into RAM ...")
            self._cache = [np.load(f).astype(np.float32) for f in self.samples]
            mem_mb = sum(a.nbytes for a in self._cache) / 1024**2
            print(f"  Preloaded: {mem_mb:.0f} MB")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        if self._cache is not None:
            feat = self._cache[index].copy()
        else:
            feat = np.load(self.samples[index]).astype(np.float32)
        orig_shape = np.array(feat.shape, dtype=np.int64)
        if self.gt:
            org_feat = feat.copy()
        feat = FeatureFolder.truncation(feat, self.trun_low, self.trun_high)
        feat = FeatureFolder.uniform_quantization(feat, self.trun_low, self.trun_high, self.bit_depth)
        feat = FeatureFolder.packing(feat, self.model_type)
        if not self.gt and self.patch_size is not None:
            feat = FeatureFolder.random_crop(feat, self.patch_size)
        feat = np.expand_dims(feat, axis=0)
        if self.gt:
            base = self.samples[index].stem
            label = self.gt.get(base, -1)
            norm_params = np.array([self.trun_low, self.trun_high], dtype=np.float32)
            return feat, label, norm_params, orig_shape, org_feat
        return feat

    def __len__(self):
        return len(self.samples)
    
    @staticmethod
    def truncation(feat, trun_low, trun_high):
        trun_feat = np.zeros_like(feat).astype(np.float32)
        if isinstance(trun_low, list):
            for idx in range(len(trun_low)):
                trun_feat[:,idx,:,:] = np.clip(feat[:,idx,:,:], trun_low[idx], trun_high[idx])
        else:
            trun_feat = np.clip(feat, trun_low, trun_high)
        
        return trun_feat

    @staticmethod
    def uniform_quantization(feat, min_v, max_v, bit_depth):
        quant_feat = np.zeros_like(feat).astype(np.float32)
        if isinstance(min_v, list):
            for idx in range(len(min_v)):
                scale = ((2**bit_depth) -1) / (max_v[idx] - min_v[idx])
                quant_feat[:,idx,:,:] = ((feat[:,idx,:,:]-min_v[idx]) * scale)
        else:
            scale = ((2**bit_depth) -1) / (max_v - min_v)
            quant_feat = ((feat-min_v) * scale)

        return quant_feat

    @staticmethod
    def uniform_dequantization(feat, min_v, max_v, bit_depth):
        feat = feat.astype(np.float32)
        dequant_feat = np.zeros_like(feat).astype(np.float32)
        if isinstance(min_v, list):
            for idx in range(len(min_v)):
                scale = ((2**bit_depth) -1) / (max_v[idx] - min_v[idx])
                dequant_feat[:,idx,:,:] = feat[:,idx,:,:] / scale + min_v[idx]
        else:
            scale = ((2**bit_depth) -1) / (max_v - min_v)
            dequant_feat = feat / scale + min_v
        return dequant_feat

    @staticmethod
    def packing(feat, model_type):
        if len(feat.shape) != 4:
            return feat
        N, C, H, W = feat.shape
        if model_type == 'llama3':
            feat = feat[0,0,:,:]
        elif model_type == 'dinov2':
            feat = feat.transpose(0,2,1,3).reshape(N*H,C*W)
        elif model_type == 'sd3':
            feat = feat.reshape(int(C/4), int(C/4), H, W).transpose(0, 2, 1, 3).reshape(int(C/4*H), int(C/4*W)) 
        return feat

    @staticmethod
    def unpacking(feat, shape, model_type):
        if len(shape) != 4:
            return feat
        N, C, H, W = shape
        if model_type == 'llama3':
            feat = np.expand_dims(feat, axis=0); feat = np.expand_dims(feat, axis=0)
        elif model_type == 'dinov2':
            feat = feat.reshape(N,H,C,W).transpose(0, 2, 1, 3) 
        elif model_type == 'sd3':
            feat = feat.reshape(int(C/4), H, int(C/4), W).transpose(0,2,1,3).reshape(N,C,H,W)
        return feat

    @staticmethod
    def random_crop(feat, crop_shape): # (hight, width)
        max_row = feat.shape[0] - crop_shape[0]
        max_col = feat.shape[1] - crop_shape[1]
        
        if max_row < 0 or max_col < 0:
            print(feat.shape[0], crop_shape[0])
            print(feat.shape[1], crop_shape[1])
            raise ValueError("crop_shape exceeds the feature shape")

        start_row = np.random.randint(0, max_row + 1)
        start_col = np.random.randint(0, max_col + 1)
        
        end_row = start_row + crop_shape[0]
        end_col = start_col + crop_shape[1]
        
        return feat[start_row:end_row, start_col:end_col]


@register_dataset("SegFeatureFolder")
class SegFeatureFolder(Dataset):
    """
    分割任务特征数据集（与 FeatureFolder 处理方式一致）
    
    特征格式：[num_slides, 1+N, D]，如 [2, 1370, 1024]
    Packing：沿 D 维度拼接 → [1+N, num_slides*D]，如 [1370, 2048]
    后续预处理（截断、归一化、量化、random_crop）与 FeatureFolder 完全一致
    
    目录结构：root/{model_type}/{layer}/*.npy（无 split 子目录）
    """
    
    # slide inference 常量（与 wrapper.py / mmseg pipeline 一致）
    CROP_SIZE = (512, 512)
    STRIDE = (341, 341)
    PATCH_SIZE = 14
    IMG_SCALE = (2048, 512)  # mmseg Resize(keep_ratio=True) 的 img_scale
    
    @staticmethod
    def compute_resized_shape(orig_h, orig_w, img_scale=(2048, 512)):
        """
        复现 mmseg Resize(keep_ratio=True) 的缩放逻辑，返回 resize 后的尺寸。
        img_scale = (max_long_edge, max_short_edge)
        """
        max_long = max(img_scale)
        max_short = min(img_scale)
        scale_factor = min(max_long / max(orig_h, orig_w), max_short / min(orig_h, orig_w))
        new_h = int(round(orig_h * scale_factor))
        new_w = int(round(orig_w * scale_factor))
        return new_h, new_w
    
    def __init__(
        self, 
        root, 
        split="train",
        transform=None, 
        model_type="dinov2_vitl14", 
        layer="blk10", 
        task="seg",
        trun_flag=False, 
        trun_low=-5, 
        trun_high=5, 
        quant_type="uniform",
        qsamples=0,
        bit_depth=1, 
        patch_size=(512, 512), 
        gt_root=None
    ):
        # 分割特征目录结构：root/{model_type}/{layer}（无 split 子目录）
        splitdir = Path(root) / model_type / layer
        
        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')
        
        # 获取所有特征文件
        self.samples = sorted(f for f in splitdir.iterdir() if f.suffix == '.npy')
        
        self.transform = transform
        self.model_type = model_type
        self.layer = layer
        self.task = task
        self.trun_flag = trun_flag
        self.trun_low = trun_low
        self.trun_high = trun_high
        self.quant_type = quant_type
        self.qsamples = qsamples
        self.bit_depth = bit_depth
        self.patch_size = patch_size
        self.gt_root = gt_root  # VOC2012 SegmentationClass 目录（测试时需要）
        self.split = split
        self._crop_cache = {}  # 缓存 (padded_h, padded_w) → crops
        
    def __getitem__(self, index):
        """
        返回格式与 FeatureFolder 一致：
            训练时: feat [1, H, W]
            测试时(gt_root不为None): (feat, gt, norm_params, orig_shape)
        
        Packing: [num_slides, 1+N, D] → [1+N, num_slides*D]
        """
        feat_path = self.samples[index]
        image_name = feat_path.stem
        
        # 加载特征 [num_slides, 1+N, D]
        feat = np.load(feat_path).astype(np.float32)
        orig_shape = np.array(feat.shape, dtype=np.int64)  # [3]: (num_slides, 1+N, D)
        
        # Packing: [num_slides, 1+N, D] → [1+N, num_slides*D]
        # 沿最后一维拼接所有 slides
        num_slides = feat.shape[0]
        feat = np.concatenate([feat[s] for s in range(num_slides)], axis=-1)  # [1+N, num_slides*D]
        if self.gt_root is not None:
            org_feat = feat.copy()  # 仅测试时保留原始特征
        
        feat = FeatureFolder.truncation(feat, self.trun_low, self.trun_high)
        feat = FeatureFolder.uniform_quantization(feat, self.trun_low, self.trun_high, self.bit_depth)
        
        # 测试时不裁剪，返回完整特征和GT
        if self.gt_root is not None:
            feat = np.expand_dims(feat, axis=0)  # [1, 1+N, num_slides*D]
            
            # 加载GT分割图
            gt_path = Path(self.gt_root) / f"{image_name}.png"
            if gt_path.exists():
                gt = np.array(Image.open(gt_path))
            else:
                gt = np.zeros((1, 1), dtype=np.uint8)
            
            norm_params = np.array([self.trun_low, self.trun_high], dtype=np.float32)
            
            # 预计算并缓存滑窗坐标（基于 mmseg resize 后的图片尺寸）
            orig_h, orig_w = gt.shape[:2]
            h_img, w_img = self.compute_resized_shape(orig_h, orig_w, self.IMG_SCALE)
            cache_key = (h_img, w_img)
            if cache_key not in self._crop_cache:
                self._crop_cache[cache_key] = self.get_slide_crops(
                    h_img, w_img, self.CROP_SIZE, self.STRIDE
                )
            crops = np.array(self._crop_cache[cache_key], dtype=np.int64)  # [num_slides, 4]
            
            return feat, gt, norm_params, orig_shape, crops, org_feat
        else:
            # 训练模式：随机裁剪（与 FeatureFolder 一致）
            if self.patch_size is not None:
                feat = FeatureFolder.random_crop(feat, self.patch_size)
            feat = np.expand_dims(feat, axis=0)  # [1, H, W]
            return feat
    
    def __len__(self):
        return len(self.samples)
    
    @staticmethod
    def seg_unpacking(feat_2d, orig_shape):
        """
        逆 packing: [1+N, num_slides*D] → [num_slides, 1+N, D]
        
        Args:
            feat_2d: numpy array [1+N, num_slides*D]
            orig_shape: (num_slides, 1+N, D)
        Returns:
            feat_3d: numpy array [num_slides, 1+N, D]
        """
        if len(orig_shape) != 3:
            return feat_2d
        num_slides, seq_len, dim = orig_shape
        # feat_2d: [1+N, num_slides*D] → split along last dim
        slides = np.split(feat_2d, num_slides, axis=-1)  # list of [1+N, D]
        return np.stack(slides, axis=0)  # [num_slides, 1+N, D]
    
    @staticmethod
    def get_slide_crops(h_img, w_img, crop_size=(512, 512), stride=(341, 341)):
        """计算滑窗裁剪区域，返回 list of (y1, x1, y2, x2)"""
        h_crop, w_crop = crop_size
        h_stride, w_stride = stride
        crops = []
        for h_idx in range(0, max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1):
            for w_idx in range(0, max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crops.append((y1, x1, y2, x2))
        return crops