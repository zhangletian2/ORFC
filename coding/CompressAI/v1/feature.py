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
from compressai.ops import compute_padding

import os
import numpy as np

# zlt MultiLayerFeatureFolder
from typing import List, Tuple, Dict
import json
import re

import importlib.util
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
path = os.path.join(_PROJECT_ROOT, "coding", "preprocess", "dt_ufc", "nonlinear_transform_v2.py")
spec = importlib.util.spec_from_file_location("nonlinear_transform_v2", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

nonlinear_quantization = mod.nonlinear_quantization
nonlinear_dequantization = mod.nonlinear_dequantization


@register_dataset("MultiSourceFeatureFolder")
class MultiSourceFeatureFolder:
    """
    将多层目录合并成一个数据池，但 __getitem__ 每次只返回一个样本（不拼接）。
    目录结构：
      /features/{split}/{backbone_name}/{layer}/*.npy
    """

    def __init__(
            self,
            root: str,
            backbone_name: List[str],
            layers: List[str],
            split: str = "train",
            transform=None,
            model_type: str = "dinov2",
            task: str = "cls",
            trun_flag: bool = False,
            trun_low=None,
            trun_high=None,
            quant_type: str = "uniform",
            qsamples: int = 0,
            bit_depth: int = 1,
            patch_size: Tuple[int, int] = (512, 512),
            layer_stats_root: str = None,
            feat_transform: str = "p2b",  # version5: trunc / p2b / zscore
            zscore_stats_root: str = None,  # version5: zscore 统计根目录
            quantization_mapping_root: str = None,  # dt-ufc quantization mapping
    ):
        self.root = Path(root)
        self.backbone_name = backbone_name
        self.layers = layers
        self.split = split
        self.transform = transform
        self.model_type = model_type
        self.task = task
        self.trun_flag = trun_flag
        self.trun_low = trun_low
        self.trun_high = trun_high
        self.quant_type = quant_type
        self.qsamples = qsamples
        self.bit_depth = bit_depth
        self.patch_size = patch_size

        # 收集所有模型、层的文件，形成 (path, backbone, layer) 列表
        self.items = []
        for backbone in backbone_name:
            for l in layers:
                d = self.root / split / backbone / l
                if not d.is_dir():
                    raise RuntimeError(f"Missing directory: {d}")
                for p in sorted(d.glob("*.npy")):
                    self.items.append((p, backbone, l))
        if not self.items:
            raise RuntimeError("No feature files found in provided layers.")
        # === version5: 记录特征变换类型 ===
        self.split = split
        # 兼容 train_p2b / test_p2b / train_zscore 等
        self.orig_split = "_".join(split.split("_")[:-1])
        self.feat_transform = feat_transform

        # 仅在使用 zscore 时，预先读取 mean/std
        if feat_transform == "zscore":
            self.zscore_stats = {}
            for backbone in backbone_name:
                for l in layers:
                    stats_path = Path(zscore_stats_root) / f"zscore_{backbone}_{l}.npz"
                    if not stats_path.is_file():
                        raise FileNotFoundError(f"Missing zscore stats: {stats_path}")
                    data = np.load(stats_path)
                    mean = data["mean"].astype(np.float32)  # [C]
                    std = data["std"].astype(np.float32)  # [C]
                    self.zscore_stats[(backbone, l)] = (mean, std)

        if feat_transform == "trunc":
            self.layer_stats = {}
            # 收集所有模型、层的统计量
            if layer_stats_root is not None:
                for backbone in self.backbone_name:
                    model_stat_path = Path(layer_stats_root) / f"{backbone}.json"
                    with open(model_stat_path, "r") as f:
                        self.layer_stats[backbone] = json.load(f)

        else:  # 'kmeans' / 'density' / 'blend' / 'ekmeans'
            # 仅调试kmeans用：
            pass
            # self.kmeans_bits = int(re.search(r'_(\d+)bit$', feat_transform).group(1))
            # self.layer_stats = {}
            # # 收集所有模型、层的统计量
            # if layer_stats_root is not None:
            #     for backbone in self.backbone_name:
            #         model_stat_path = Path(layer_stats_root) / f"{backbone}.json"
            #         with open(model_stat_path, "r") as f:
            #             self.layer_stats[backbone] = json.load(f)

            # self.transform_mapping = {}
            # if quantization_mapping_root is not None:
            #     for backbone in self.backbone_name:
            #         for l in layers:
            #             layer_mapping_path = Path(quantization_mapping_root) / backbone / f"{l}.json"
            #             with open(layer_mapping_path, "r") as f:
            #                 self.transform_mapping[(backbone, l)] = np.array(json.load(f))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        """
        返回：
          x_packed : [1, H, W]，喂给 codec 的输入（目前仍然是 offline P2B 后的特征 / 或原始特征）
          orig_feat: [257, C]，原始中间层特征（未 packing、未 P2B），用于在原空间上算 MSE
          backbone : 对应的模型名字符串
          layer    : 对应的层名字符串（如 'blk05'）
        """
        path, backbone, layer = self.items[index]
        # 原始特征（features/{orig_split}/...），用于真实 token 空间 MSE / distill
        orig_root = self.root / self.orig_split
        orig_path = orig_root / backbone / layer / path.name
        if not orig_path.is_file():
            # fallback：如果没有对应原始 split，就用当前 path（兼容某些目录结构）
            orig_path = path
        orig_feat = np.load(orig_path).astype(np.float32)  # [257, C]
        
        # 输入给 codec 的特征
        if self.feat_transform == "p2b":
            # 当前 split 目录下的文件就是离线 p2b 后的特征
            feat = np.load(path).astype(np.float32)  # [257, C]
            # packing
            x = MultiSourceFeatureFolder.feat_forward_transform(feat)  # [H, W]

        elif self.feat_transform == "zscore":
            # 在线 zscore：基于原始 token 特征做归一化
            feat = orig_feat.copy()
            mean, std = self.zscore_stats[(backbone, layer)]  # [C], [C]
            std_safe = np.maximum(std, 1e-6)
            feat = (feat - mean[None, :]) / std_safe[None, :]
            # packing
            x = MultiSourceFeatureFolder.feat_forward_transform(feat)  # [H, W]

        elif self.feat_transform == "trunc" or self.feat_transform == "residual_trunc":
            # assert self.trun_flag
            # feat = orig_feat.copy()

            # global normalization
            # bounds = self.layer_stats[backbone].get(layer, None)
            # low, high = float(bounds["low"]), float(bounds["high"])
            # feat = FeatureFolder.truncation(feat, low, high)
            # feat = FeatureFolder.uniform_quantization(feat, low, high, self.bit_depth)

            # sample-wise normalization
            # min_val = np.min(feat)
            # max_val = np.max(feat)   
            # feat = (feat - min_val) / (max_val - min_val)
            # # packing
            # x = MultiSourceFeatureFolder.feat_forward_transform(feat)  # [H, W]
            # 仅调试split training用：当前 split 目录下的文件就是离线处理后的sample-wise normalization特征
            x = np.load(path).astype(np.float32)  # [1088, 256]

        # else:  # 'kmeans' / 'density' / 'blend' / 'ekmeans'
        #     assert self.trun_flag
        #     feat = orig_feat.copy()
        #     bounds = self.layer_stats[backbone].get(layer, None)
        #     low, high = float(bounds["low"]), float(bounds["high"])
        #     feat = FeatureFolder.truncation(feat, low, high)
        #     # packing
        #     x = MultiSourceFeatureFolder.feat_forward_transform(feat)  # [H, W]
        #     kmeans_points = self.transform_mapping[(backbone, layer)]
        #     x = nonlinear_quantization(x, kmeans_points, self.kmeans_bits)
        else:
            # 仅调试kmeans用：当前 split 目录下的文件就是离线处理后的特征
            x = np.load(path).astype(np.float32)  # [1088, 256]
        x = np.expand_dims(x, axis=0)  # [1, H, W]
        return x, orig_feat, backbone, layer


    # 第三版处理：257通过将CLS复制16次stack到17*16，再展开1024变成(64*17)*(16*16)
    @staticmethod
    def feat_forward_transform(feat, n_split=64):
        C = feat.shape[1]
        n = C // n_split
        cls = feat[0:1, :]  # (1, 1024)
        patches = feat[1:, :]  # (256, 1024)
        patch_grid = patches.reshape(16, 16, C)  # (16, 16, 1024)
        cls_row = np.tile(cls, (1, 16, 1))  # (1, 16, 1024)
        stacked = np.concatenate([cls_row, patch_grid], axis=0)  # (17, 16, 1024)
        arr4 = stacked.reshape(17, 16, n_split, n)  # (17, 16, 64, n)
        # 先交换成 (17, 64, 16, n)，再合并
        packed = arr4.transpose(0, 2, 1, 3).reshape(17 * n_split, 16 * n)
        return packed

    @staticmethod
    def feat_inverse_transform(packed, n_split=64):
        H, W = packed.shape
        h_blocks = H // n_split  # 应为 17
        n = W // 16  # 对 1024 情况下 n = 16
        C = n_split * n  # 应为 1024

        arr4 = packed.reshape(h_blocks, n_split, 16, n)  # (17, 64, 16, n)
        stacked = arr4.transpose(0, 2, 1, 3).reshape(h_blocks, 16, C)  # (17, 16, 1024)
        cls_row = stacked[0]  # (16, 1024)，16 个重复的 CLS
        patch_grid = stacked[1:]  # (16, 16, 1024)
        cls = cls_row[0:1, :]  # (1, 1024)
        patches = patch_grid.reshape(16 * 16, C)  # (256, 1024)
        feat_recon = np.vstack([cls, patches])  # (257, 1024)
        return feat_recon


# === End add ====================================================

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

    def __init__(self, root, transform=None, split="train", model_type="sd3", task="tti", trun_flag=False, trun_low=-20,
                 trun_high=20, quant_type="uniform", qsamples=0, bit_depth=1, patch_size=(512, 512)):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')

        self.samples = sorted(f for f in splitdir.iterdir() if f.is_file())
        # gcs
        # self.samples = self.samples[:100]

        self.transform = transform

        # gcs
        self.model_type = model_type
        self.task = task
        self.trun_flag = trun_flag
        self.trun_low = trun_low
        self.trun_high = trun_high
        self.quant_type = quant_type
        self.qsamples = qsamples
        self.bit_depth = bit_depth
        self.patch_size = patch_size  # (height, width), must be the multiple of 64

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        # Load feature, use float32 for training
        feat = np.load(self.samples[index]).astype(np.float32)
        # gcs, preprocessing
        if self.trun_flag == True: feat = FeatureFolder.truncation(feat, self.trun_low, self.trun_high)
        feat = FeatureFolder.uniform_quantization(feat, self.trun_low, self.trun_high, self.bit_depth)
        feat = FeatureFolder.packing(feat, self.model_type)
        feat = FeatureFolder.random_crop(feat, self.patch_size)  # (height, width), must be the multiple of 64
        feat = np.expand_dims(feat, axis=0)  # (C,H,W)
        # print(feat.shape)
        return feat

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def truncation(feat, trun_low, trun_high):
        trun_feat = np.zeros_like(feat).astype(np.float32)
        if isinstance(trun_low, list):
            for idx in range(len(trun_low)):
                trun_feat[:, idx, :, :] = np.clip(feat[:, idx, :, :], trun_low[idx], trun_high[idx])
        else:
            trun_feat = np.clip(feat, trun_low, trun_high)

        return trun_feat

    @staticmethod
    def uniform_quantization(feat, min_v, max_v, bit_depth):
        quant_feat = np.zeros_like(feat).astype(np.float32)
        if isinstance(min_v, list):
            for idx in range(len(min_v)):
                scale = ((2 ** bit_depth) - 1) / (max_v[idx] - min_v[idx])
                quant_feat[:, idx, :, :] = ((feat[:, idx, :, :] - min_v[idx]) * scale)
        else:
            scale = ((2 ** bit_depth) - 1) / (max_v - min_v)
            quant_feat = ((feat - min_v) * scale)

        return quant_feat

    @staticmethod
    def uniform_dequantization(feat, min_v, max_v, bit_depth):
        feat = feat.astype(np.float32)
        dequant_feat = np.zeros_like(feat).astype(np.float32)
        if isinstance(min_v, list):
            for idx in range(len(min_v)):
                scale = ((2 ** bit_depth) - 1) / (max_v[idx] - min_v[idx])
                dequant_feat[:, idx, :, :] = feat[:, idx, :, :] / scale + min_v[idx]
        else:
            scale = ((2 ** bit_depth) - 1) / (max_v - min_v)
            dequant_feat = feat / scale + min_v
        return dequant_feat

    @staticmethod
    def packing(feat, model_type):
        N, C, H, W = feat.shape
        if model_type == 'llama3':
            feat = feat[0, 0, :, :]
        elif model_type == 'dinov2':
            feat = feat.transpose(0, 2, 1, 3).reshape(N * H, C * W)
        elif model_type == 'sd3':
            feat = feat.reshape(int(C / 4), int(C / 4), H, W).transpose(0, 2, 1, 3).reshape(int(C / 4 * H),
                                                                                            int(C / 4 * W))
        # zlt
        elif model_type == 'clip':
            feat = feat.transpose(0, 2, 1, 3).reshape(N * H, C * W)
        return feat

    @staticmethod
    def unpacking(feat, shape, model_type):
        N, C, H, W = shape
        if model_type == 'llama3':
            feat = np.expand_dims(feat, axis=0);
            feat = np.expand_dims(feat, axis=0)
        elif model_type == 'dinov2':
            feat = feat.reshape(N, H, C, W).transpose(0, 2, 1, 3)
        elif model_type == 'sd3':
            feat = feat.reshape(int(C / 4), H, int(C / 4), W).transpose(0, 2, 1, 3).reshape(N, C, H, W)
        # zlt
        elif model_type == 'clip':
            feat = feat.reshape(N, H, C, W).transpose(0, 2, 1, 3)
        return feat

    @staticmethod
    def random_crop(feat, crop_shape):  # (hight, width)
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
