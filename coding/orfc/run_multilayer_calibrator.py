import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import csv
from tqdm import tqdm
import math
from datetime import datetime
import fcntl
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from typing import Tuple, List, Optional

ORFC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(ORFC_ROOT, "..", ".."))

from simple_fcvq import SimpleFCVQCompat, create_vq_codec
from backbone.wrapper import Dinov2Wrapper
from entropy_coding import (
    GroupConditionedEntropyModel, PositionConditionedEntropyModel,
    rvq_compress_grouped, rvq_compress_position, full_encode_decode_pipeline
)

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
import logging
try:
    from mmcv.utils import get_logger
    logger = get_logger('mmcv')
except ImportError:
    logger = logging.getLogger('mmcv')
logger.setLevel(logging.WARNING)

# ============== 随机种子 ==============

def set_seed(seed):
    """固定随机种子，确保实验可复现"""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============== 归一化 ==============
#
# 支持的归一化模式:
#   - per_token_ln: 每个 token 独立的 LayerNorm (mu + std), 32 bits/token
#   - per_image: 整张图片共享一组 mu + std, ~0.12 bits/token (32 bits / 257 tokens)
#
# 各模式的侧信息计算:
#   - per_token_ln: num_tokens * (mu_bits + std_bits)
#   - per_image: mu_bits + std_bits (整张图共享)
#

def per_token_layernorm(x: np.ndarray, eps: float = 1e-5):
    """Per-token LayerNorm (baseline): 每个 token 独立的 mu 和 std"""
    mu = x.mean(axis=1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
    std = np.sqrt(var + eps)
    y = (x - mu) / std
    return y.astype(np.float32), mu.astype(np.float32), std.astype(np.float32)


def inv_per_token_layernorm(y_hat, mu, std):
    """逆 per-token LayerNorm"""
    return (y_hat * std + mu).astype(np.float32)


def per_image_norm(x: np.ndarray, eps: float = 1e-5):
    """
    Per-image 归一化: 整张图片共享一组 mu 和 std
    
    mu = mean(x), std = sqrt(mean((x - mu)^2))
    y = (x - mu) / std
    
    侧信息: ~0.12 bits/token (32 bits / 257 tokens)
    
    Args:
        x: [T, C] 单张图的特征
    
    Returns:
        y: 归一化后的特征 [T, C]
        mu: 全局均值 [1, C]
        std: 全局标准差 [1, C]
    """
    # 全局统计量 (跨所有 token)
    mu = x.mean(axis=0, keepdims=True).mean(axis=1, keepdims=True)  # 标量近似
    mu_full = x.mean()  # 标量
    # 计算全局方差
    var = ((x - mu_full) ** 2).mean()
    std = np.sqrt(var + eps)
    
    # 归一化
    y = (x - mu_full) / std
    
    # 返回格式: mu 和 std 都是标量，但需要广播
    # 存储为 [1, 1] 形状，应用时广播
    mu_out = np.array([[mu_full]], dtype=np.float32)
    std_out = np.array([[std]], dtype=np.float32)
    
    return y.astype(np.float32), mu_out, std_out


def inv_per_image_norm(y_hat, mu, std):
    """
    逆 per-image 归一化
    
    x_hat = y_hat * std + mu
    """
    return (y_hat * std + mu).astype(np.float32)


# ============== 统一归一化接口 ==============

def normalize(x: np.ndarray, mode: str = 'per_token_ln', eps: float = 1e-5):
    """
    统一归一化接口
    
    Args:
        x: [T, C] 单张图的特征
        mode: 归一化模式
            - 'per_token_ln': 每 token LayerNorm (baseline)
            - 'per_image': 整张图共享
    
    Returns:
        y: 归一化后的特征
        mu: 均值 (格式因模式而异)
        std: 标准差 (格式因模式而异)
    """
    if mode == 'per_token_ln':
        return per_token_layernorm(x, eps)
    elif mode == 'per_image':
        return per_image_norm(x, eps)
    else:
        raise ValueError(f"Unknown norm_mode: {mode}")


def inv_normalize(y_hat, mu, std, mode: str = 'per_token_ln'):
    """
    统一逆归一化接口
    
    Args:
        y_hat: 归一化后的特征
        mu: 均值
        std: 标准差
        mode: 归一化模式
    
    Returns:
        x_hat: 重建特征
    """
    if mode == 'per_token_ln':
        return inv_per_token_layernorm(y_hat, mu, std)
    elif mode == 'per_image':
        return inv_per_image_norm(y_hat, mu, std)
    else:
        raise ValueError(f"Unknown norm_mode: {mode}")


# ============== BPFP 计算 ==============
#
# BPFP (Bits Per Feature Parameter) 统计说明:
#
# BPFP 由两部分组成:
#   1. VQ 码流字节 (vq_bytes): 通过 constriction 熵编码器实际压缩得到
#   2. 侧信息字节 (side_info_bytes): 归一化统计量, 采用【理论计费】
#
# 理论计费说明:
#   - 统计量 (mu/std) 并未通过实际熵编码器压缩并序列化保存
#   - 根据不同归一化模式计算理论计费:
#     - per_token_ln: 32 bits/token (mu + std, 各 16 bits)
#     - per_image: 32 bits/image (~0.12 bits/token for 257 tokens)
#

def compute_side_info_bits_per_token(norm_mode: str, num_tokens: int = 257,
                                      mu_bits: int = 16, std_bits: int = 16):
    """
    计算每个 token 的侧信息位数 (理论计费)
    
    Args:
        norm_mode: 归一化模式
        num_tokens: 每张图的 token 数 (default: 257 = 1 CLS + 256 patches)
        mu_bits: mu 的位数 (默认 16)
        std_bits: std 的位数 (默认 16)
    
    Returns:
        bits_per_token: 每个 token 的侧信息位数
    """
    if norm_mode == 'per_token_ln':
        # 每 token: mu (16) + std (16) = 32 bits
        return mu_bits + std_bits
    
    elif norm_mode == 'per_image':
        # 整张图: mu (16) + std (16) = 32 bits
        # 分摊到每 token: 32 / 257 ≈ 0.12 bits
        bits_per_image = mu_bits + std_bits
        return bits_per_image / num_tokens
    
    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode}")


def compute_bpfp_fixed(K_list, embedding_dim, feat_dim=1024, norm_mode='per_token_ln',
                       num_tokens=257, mu_bits=16, std_bits=16):
    """
    计算固定长度 BPFP (理论值)
    
    假设 VQ 索引不经熵编码，直接用 log2(K) bits 存储
    侧信息按理论计费 (未实际压缩)
    
    Args:
        K_list: 各层 VQ 的码本大小列表
        embedding_dim: VQ embedding 维度
        feat_dim: 特征维度 (default: 1024)
        norm_mode: 归一化模式
        num_tokens: 每张图的 token 数
        mu_bits: mu 的位数
        std_bits: std 的位数
    """
    vq_bits_per_group = sum(math.log2(K) for K in K_list if K > 0)
    num_groups = feat_dim / embedding_dim
    vq_bits_per_token = vq_bits_per_group * num_groups
    side_info_bits = compute_side_info_bits_per_token(norm_mode, num_tokens, mu_bits, std_bits)
    total_bits_per_token = vq_bits_per_token + side_info_bits
    bpfp = total_bits_per_token / feat_dim
    return bpfp


def compute_bpfp_from_bytes(vq_bytes, side_info_bytes, total_tokens, feat_dim=1024):
    """
    从实际压缩字节数计算 BPFP
    
    注意: vq_bytes 是实际熵编码字节, side_info_bytes 是理论计费字节
    
    Args:
        vq_bytes: VQ 熵编码的字节数 (实际压缩得到)
        side_info_bytes: 侧信息的字节数 (理论计费, 未实际压缩)
        total_tokens: 总 token 数（所有图片的 token 数之和）
        feat_dim: 特征维度
    
    Returns:
        bpfp: Bits Per Feature Parameter
    """
    total_bits = (vq_bytes + side_info_bytes) * 8
    total_features = total_tokens * feat_dim
    return total_bits / total_features


def compute_side_info_bytes(num_images, num_tokens=257, norm_mode='per_token_ln',
                             mu_bits=16, std_bits=16):
    """
    计算侧信息的理论计费字节数
    
    【理论计费】: 统计量并未实际序列化压缩
    
    Args:
        num_images: 图像数量
        num_tokens: 每张图像的 token 数
        norm_mode: 归一化模式
        mu_bits: mu 的位数 (默认 16)
        std_bits: std 的位数 (默认 16)
    
    Returns:
        side_info_bytes: 侧信息理论计费字节数
    """
    bits_per_token = compute_side_info_bits_per_token(norm_mode, num_tokens, mu_bits, std_bits)
    total_bits = num_images * num_tokens * bits_per_token
    return int(total_bits) // 8


# ============== 投影矩阵 (用于校准器) ==============

def get_next_block_weights(backbone, layer_idx, head=None, device='cuda'):
    """获取下一层的权重用于校准器"""
    next_layer_idx = layer_idx + 1
    if next_layer_idx < len(backbone.blocks):
        block = backbone.blocks[next_layer_idx]
        attn = block.attn
        mlp = block.mlp
        qkv_weight = attn.qkv.weight.data
        dim = qkv_weight.shape[1]
        return {
            'Wq': qkv_weight[:dim, :].T,
            'Wk': qkv_weight[dim:2*dim, :].T,
            'Wv': qkv_weight[2*dim:, :].T,
            'norm1_weight': block.norm1.weight.data,
            'norm1_bias': block.norm1.bias.data,
            'dim': dim,
            'is_last_layer': False
        }
    else:
        dim = backbone.norm.weight.shape[0]
        result = {
            'norm_weight': backbone.norm.weight.data,
            'norm_bias': backbone.norm.bias.data,
            'dim': dim,
            'is_last_layer': True,
            'use_head': head is not None
        }
        if head is not None:
            result['head_weight'] = head.weight.data
        return result


# ============== 特征处理 ==============

def _process_single_feature(x, embedding_dim, norm_mode='per_token_ln'):
    """处理单个特征（用于并行）"""
    y, _, _ = normalize(x, mode=norm_mode)
    T, C = y.shape
    num_chunks = C // embedding_dim
    y_reshaped = y.reshape(T, num_chunks, embedding_dim)
    return y_reshaped.reshape(-1, embedding_dim)


def _process_single_residual(r, embedding_dim):
    """处理单个残差（用于并行）"""
    T, C = r.shape
    num_chunks = C // embedding_dim
    r_reshaped = r.reshape(T, num_chunks, embedding_dim)
    return r_reshaped.reshape(-1, embedding_dim)


def flatten_features(features, embedding_dim, num_workers=1, norm_mode='per_token_ln'):
    """展平特征为向量（支持并行）"""
    if num_workers <= 1:
        all_vectors = []
        for x in features:
            all_vectors.append(_process_single_feature(x, embedding_dim, norm_mode))
        return np.concatenate(all_vectors, axis=0)
    
    # 并行处理
    process_fn = partial(_process_single_feature, embedding_dim=embedding_dim, norm_mode=norm_mode)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        all_vectors = list(executor.map(process_fn, features))
    return np.concatenate(all_vectors, axis=0)


def flatten_residuals(residuals, embedding_dim, num_workers=1):
    """展平残差为向量（支持并行）"""
    if num_workers <= 1:
        all_vectors = []
        for r in residuals:
            all_vectors.append(_process_single_residual(r, embedding_dim))
        return np.concatenate(all_vectors, axis=0)
    
    # 并行处理
    process_fn = partial(_process_single_residual, embedding_dim=embedding_dim)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        all_vectors = list(executor.map(process_fn, residuals))
    return np.concatenate(all_vectors, axis=0)


# ============== K-means ==============

def kmeans(vectors, K, max_iter=50, chunk_size=100000, device='cuda', verbose=True):
    """
    普通 K-means 聚类
    
    Args:
        vectors: 输入向量 [N, D]
        K: 聚类数
        max_iter: 最大迭代次数
        chunk_size: 批处理大小
        device: 设备
        verbose: 是否打印进度
    
    Returns:
        centroids: 聚类中心 [K, D]
    """
    N, D = vectors.shape
    X = torch.from_numpy(vectors).float().to(device)
    
    # 随机初始化
    indices = torch.randperm(N)[:K]
    centroids = X[indices].clone()
    
    for it in range(max_iter):
        all_labels = []
        
        for i in range(0, N, chunk_size):
            batch_X = X[i:i+chunk_size]
            dists = torch.cdist(batch_X, centroids)
            labels = dists.argmin(dim=1)
            all_labels.append(labels)
        
        labels = torch.cat(all_labels)
        
        # 更新 centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(K, device=device)
        ones = torch.ones(N, device=device)
        counts.scatter_add_(0, labels, ones)
        labels_expand = labels.unsqueeze(1).expand(-1, D)
        new_centroids.scatter_add_(0, labels_expand, X)
        
        mask = counts > 0
        new_centroids[mask] = new_centroids[mask] / counts[mask].unsqueeze(1)
        
        # 处理空簇
        empty_mask = counts == 0
        if empty_mask.any():
            num_empty = empty_mask.sum().item()
            new_centroids[empty_mask] = X[torch.randint(N, (num_empty,), device=device)]
        
        shift = (new_centroids - centroids).norm()
        centroids = new_centroids
        
        if verbose and (it + 1) % 10 == 0:
            print(f"    K-means iter {it+1}: shift={shift:.4f}")
        
        if shift < 1e-4:
            break
    
    return centroids.cpu().numpy().astype(np.float32)


# ============== VQ 编解码 ==============

def vq_forward_batch(codec, feat_batch, device, batch_size=16):
    """批量 VQ forward (纯最近邻量化)"""
    results = []
    for i in range(0, len(feat_batch), batch_size):
        batch = feat_batch[i:i+batch_size]
        batch_tensors = [torch.from_numpy(f.T).float() for f in batch]
        batch_t = torch.stack(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            outputs = codec(batch_t)
            feat_hat = outputs[0]
        for j in range(feat_hat.shape[0]):
            results.append(feat_hat[j].T.cpu().numpy())
    return results


def collect_vq_indices(codec, features, device, batch_size=32):
    """收集 VQ 索引"""
    all_indices = []
    for i in range(0, len(features), batch_size):
        batch = features[i:i+batch_size]
        batch_tensors = [torch.from_numpy(f.T).float() for f in batch]
        batch_t = torch.stack(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            outputs = codec(batch_t)
            encoding_inds = outputs[4]
        for chunk_inds in encoding_inds:
            all_indices.append(chunk_inds.cpu().numpy().flatten())
    return np.concatenate(all_indices)


def collect_residuals(codec_list, features, device, norm_mode='per_token_ln'):
    """收集残差"""
    y_list = [normalize(x, mode=norm_mode)[0] for x in features]
    y_hat_list = [np.zeros_like(y) for y in y_list]
    
    for codec in codec_list:
        if codec is None:
            continue
        r_list = [y - y_hat for y, y_hat in zip(y_list, y_hat_list)]
        r_hat_list = vq_forward_batch(codec, r_list, device)
        for i, r_hat in enumerate(r_hat_list):
            y_hat_list[i] = y_hat_list[i] + r_hat
    
    residuals = [y - y_hat for y, y_hat in zip(y_list, y_hat_list)]
    return residuals


# ============== RVQ 编解码 ==============

def rvq_encode_decode(codec_list, features, device, norm_mode='per_token_ln'):
    """
    RVQ 编解码 (用 forward，纯最近邻量化)
    
    Args:
        codec_list: VQ codec 列表
        features: 特征列表
        device: 设备
        norm_mode: 归一化模式
    
    Returns:
        xhat_list: 重建特征列表
    """
    y_list, mu_list, std_list = [], [], []
    for x in features:
        y, mu, std = normalize(x, mode=norm_mode)
        y_list.append(y)
        mu_list.append(mu)
        std_list.append(std)
    
    y_hat_list = [np.zeros_like(y) for y in y_list]
    
    for codec in codec_list:
        if codec is None:
            continue
        r_list = [y - y_hat for y, y_hat in zip(y_list, y_hat_list)]
        r_hat_list = vq_forward_batch(codec, r_list, device)
        for i, r_hat in enumerate(r_hat_list):
            y_hat_list[i] = y_hat_list[i] + r_hat
    
    xhat_list = [inv_normalize(y_hat, mu, std, mode=norm_mode)
                 for y_hat, mu, std in zip(y_hat_list, mu_list, std_list)]
    return xhat_list


def rvq_compress_all(codec_list, features, device, norm_mode='per_token_ln',
                      mu_bits=16, std_bits=16):
    """
    RVQ 真实熵编码压缩 (用 compress)
    
    关键步骤:
    1. 先用 forward 收集所有索引
    2. 调用 update_prior 更新熵模型 (基于实际索引频率)
    3. 再用 compress 进行真实熵编码
    
    返回:
        xhat_list: 重建特征列表
        vq_bytes: VQ 熵编码的字节数
        side_info_bytes: 侧信息的字节数
        encode_time: 编码时间 (秒)
    """
    import time
    
    y_list, mu_list, std_list = [], [], []
    for x in features:
        y, mu, std = normalize(x, mode=norm_mode)
        y_list.append(y)
        mu_list.append(mu)
        std_list.append(std)
    
    y_hat_list = [np.zeros_like(y) for y in y_list]
    vq_bytes = 0
    encode_time = 0.0
    
    for codec in codec_list:
        if codec is None:
            continue
        r_list = [y - y_hat for y, y_hat in zip(y_list, y_hat_list)]
        
        # Step 1: 先用 forward 收集所有索引
        all_indices = []
        for i, r in enumerate(r_list):
            r_t = torch.from_numpy(r.T).float().to(device)
            with torch.no_grad():
                outputs = codec(r_t.unsqueeze(0))  # forward 返回 [x_hat, mse, rd, rate, indices]
                encoding_inds = outputs[4]  # indices 是列表
            for chunk_inds in encoding_inds:
                all_indices.append(chunk_inds.cpu().numpy().flatten())
        
        # Step 2: 更新 prior (基于实际索引频率)
        all_indices_arr = np.concatenate(all_indices)
        codec.update_prior(all_indices_arr)
        
        # Step 3: 用 compress 进行真实熵编码 (计时)
        t0 = time.perf_counter()
        for i, r in enumerate(r_list):
            r_t = torch.from_numpy(r.T).float().to(device)
            with torch.no_grad():
                result = codec.compress(r_t)
            r_hat = result[0].T.cpu().numpy()
            y_hat_list[i] = y_hat_list[i] + r_hat
            vq_bytes += sum(len(s) for s in result[2] if isinstance(s, (bytes, bytearray)))
        encode_time += time.perf_counter() - t0
    
    xhat_list = [inv_normalize(y_hat, mu, std, mode=norm_mode)
                 for y_hat, mu, std in zip(y_hat_list, mu_list, std_list)]
    
    # 计算侧信息字节数
    num_images = len(features)
    num_tokens = features[0].shape[0] if len(features) > 0 else 257
    side_info_bytes = compute_side_info_bytes(num_images, num_tokens, norm_mode, mu_bits, std_bits)
    
    return xhat_list, vq_bytes, side_info_bytes, encode_time


# ============== 分组静态熵编码 ==============
# 熵编码模型已移至 entropy_coding.py:
#   - GroupConditionedEntropyModel: group 条件化熵编码
#   - PositionConditionedEntropyModel: 位置 + group 条件化熵编码

# ============== 校准器 ==============

def apply_layernorm(x_tensor, weight, bias, eps=1e-6):
    """应用 LayerNorm"""
    mean = x_tensor.mean(dim=-1, keepdim=True)
    var = ((x_tensor - mean) ** 2).mean(dim=-1, keepdim=True)
    x_norm = (x_tensor - mean) / torch.sqrt(var + eps)
    return x_norm * weight + bias


class ScaleBiasCalibrator(torch.nn.Module):
    """Scale-Bias 校准器 (原版)"""
    
    def __init__(self, dim, s_init=None, b_init=None, device='cuda'):
        super().__init__()
        if s_init is None:
            s_init = torch.ones(dim)
        if b_init is None:
            b_init = torch.zeros(dim)
        self.scale = torch.nn.Parameter(s_init.reshape(1, 1, dim).float().to(device))
        self.bias = torch.nn.Parameter(b_init.reshape(1, 1, dim).float().to(device))
    
    def forward(self, x):
        return x * self.scale + self.bias
    
class LoRACalibrator(torch.nn.Module):
    """
    LoRA 校准器: Scale-Bias + 低秩残差
    
    x_cal = x * scale + (x @ lora_A @ lora_B) * alpha + bias
    
    Args:
        dim: 特征维度
        rank: 低秩维度 (r)
        alpha: LoRA 缩放因子，实际使用 alpha/rank
        s_init: scale 初始化
        b_init: bias 初始化
        A_init: lora_A 初始化 (可选, [dim, rank])
        B_init: lora_B 初始化 (可选, [rank, dim])
    """
    
    def __init__(self, dim, rank=16, alpha=1.0, s_init=None, b_init=None, 
                 A_init=None, B_init=None, device='cuda'):
        super().__init__()
        if s_init is None:
            s_init = torch.ones(dim)
        if b_init is None:
            b_init = torch.zeros(dim)
        
        self.dim = dim
        self.rank = rank
        self.scaling = alpha / rank  # LoRA 标准缩放
        
        # Scale-Bias 部分
        self.scale = torch.nn.Parameter(s_init.reshape(1, 1, dim).float().to(device))
        self.bias = torch.nn.Parameter(b_init.reshape(1, 1, dim).float().to(device))
        
        # LoRA 低秩部分
        if A_init is not None:
            # 使用提供的初始化
            if isinstance(A_init, np.ndarray):
                A_init = torch.from_numpy(A_init)
            self.lora_A = torch.nn.Parameter(A_init.float().to(device))
        else:
            # 默认: A 用 Kaiming 初始化
            self.lora_A = torch.nn.Parameter(
                torch.randn(dim, rank, device=device) * (1.0 / math.sqrt(dim))
            )
        
        if B_init is not None:
            # 使用提供的初始化
            if isinstance(B_init, np.ndarray):
                B_init = torch.from_numpy(B_init)
            self.lora_B = torch.nn.Parameter(B_init.float().to(device))
        else:
            # 默认: B 初始化为 0
            self.lora_B = torch.nn.Parameter(
                torch.zeros(rank, dim, device=device)
            )
    
    def forward(self, x):
        # 对角变换
        out = x * self.scale
        # 低秩残差 (捕捉维度间交互)
        lora_out = (x @ self.lora_A) @ self.lora_B
        out = out + lora_out * self.scaling
        # 偏置
        out = out + self.bias
        return out
    
    def get_full_state(self):
        """返回完整状态用于精确推理"""
        return {
            'type': 'lora',
            'dim': self.dim,
            'rank': self.rank,
            'scaling': self.scaling,
            'scale': self.scale.detach().cpu(),
            'bias': self.bias.detach().cpu(),
            'lora_A': self.lora_A.detach().cpu(),
            'lora_B': self.lora_B.detach().cpu(),
        }


class MLPCalibrator(torch.nn.Module):
    """
    轻量 MLP 校准器: 残差连接 + 两层 MLP
    
    x_cal = x + MLP(x)
    
    其中 MLP(x) = Linear(GELU(Linear(x)))
    
    Args:
        dim: 特征维度
        hidden_dim: MLP 隐藏层维度
        s_init: scale 初始化 (用于初始化第一层 bias)
        b_init: bias 初始化 (用于初始化输出层 bias)
    """
    
    def __init__(self, dim, hidden_dim=64, s_init=None, b_init=None, device='cuda'):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # 两层 MLP: dim -> hidden_dim -> dim
        self.fc1 = torch.nn.Linear(dim, hidden_dim, device=device)
        self.fc2 = torch.nn.Linear(hidden_dim, dim, device=device)
        
        # 初始化: 使 MLP 输出接近于 (s-1)*x + b，即初始校准效果
        # fc1: Kaiming 初始化
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.fc1.bias)
        
        # fc2: 初始化为零，使初始 MLP(x) ≈ 0，即 x_cal ≈ x
        torch.nn.init.zeros_(self.fc2.weight)
        if b_init is not None:
            # 用 b_init 初始化输出偏置，使初始 x_cal = x + b
            self.fc2.bias = torch.nn.Parameter(
                torch.from_numpy(b_init).float().to(device)
            )
        else:
            torch.nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        # 残差连接: x + MLP(x)
        h = self.fc1(x)
        h = F.gelu(h)
        h = self.fc2(h)
        return x + h
    
    def get_full_state(self):
        """返回完整状态用于精确推理"""
        return {
            'type': 'mlp',
            'dim': self.dim,
            'hidden_dim': self.hidden_dim,
            'fc1_weight': self.fc1.weight.detach().cpu(),
            'fc1_bias': self.fc1.bias.detach().cpu(),
            'fc2_weight': self.fc2.weight.detach().cpu(),
            'fc2_bias': self.fc2.bias.detach().cpu(),
        }


def create_calibrator(calibrator_type, dim, s_init, b_init, rank=16, hidden_dim=64, alpha=1.0,
                      device='cuda'):
    """
    创建校准器
    
    Args:
        calibrator_type: 'scale_bias', 'lora', 或 'mlp'
        dim: 特征维度
        s_init: scale 初始化 (numpy array)
        b_init: bias 初始化 (numpy array)
        rank: LoRA 秩 (仅 lora 类型使用)
        hidden_dim: MLP 隐藏层维度 (仅 mlp 类型使用)
        alpha: LoRA alpha (仅 lora 类型使用)
    """
    s_init_t = torch.from_numpy(s_init)
    b_init_t = torch.from_numpy(b_init)
    
    if calibrator_type == 'scale_bias':
        return ScaleBiasCalibrator(dim, s_init_t, b_init_t, device)
    elif calibrator_type == 'lora':
        return LoRACalibrator(dim, rank, alpha, s_init_t, b_init_t, device=device)
    elif calibrator_type == 'mlp':
        return MLPCalibrator(dim, hidden_dim, s_init, b_init, device)
    else:
        raise ValueError(f"Unknown calibrator type: {calibrator_type}")


def fit_scale_bias_next_layer(x_list, xhat_list, block_weights, steps=1000, lr=1e-2,
                               batch_size=20, calibrator_type='scale_bias', rank=16, hidden_dim=64, device='cuda'):
    """
    v3 原版校准器: 只优化下一层投影
    
    Args:
        calibrator_type: 'scale_bias', 'lora', 或 'mlp'
        rank: LoRA 秩
        hidden_dim: MLP 隐藏层维度
    """
    is_last_layer = block_weights.get('is_last_layer', False)
    n_samples = len(x_list)
    C = x_list[0].shape[1]
    
    # 统计量初始化
    X = np.concatenate(x_list, axis=0)
    XH = np.concatenate(xhat_list, axis=0)
    mu_x, std_x = X.mean(axis=0), X.std(axis=0) + 1e-6
    mu_xh, std_xh = XH.mean(axis=0), XH.std(axis=0) + 1e-6
    s_init = std_x / std_xh
    b_init = mu_x - s_init * mu_xh
    
    # 创建校准器
    calibrator = create_calibrator(calibrator_type, C, s_init, b_init, rank=rank, hidden_dim=hidden_dim, device=device)
    opt = torch.optim.AdamW(calibrator.parameters(), lr=lr)
    
    if is_last_layer:
        norm_w = block_weights['norm_weight']
        norm_b = block_weights['norm_bias']
        use_head = block_weights.get('use_head', False)
        head_weight = block_weights.get('head_weight')
    else:
        Wq, Wk, Wv = block_weights['Wq'], block_weights['Wk'], block_weights['Wv']
        norm1_w, norm1_b = block_weights['norm1_weight'], block_weights['norm1_bias']
    
    for step in range(steps):
        indices = np.random.choice(n_samples, min(batch_size, n_samples), replace=False)
        x = torch.from_numpy(np.stack([x_list[i] for i in indices], axis=0)).float().to(device)
        x_hat = torch.from_numpy(np.stack([xhat_list[i] for i in indices], axis=0)).float().to(device)
        B, T, C_dim = x.shape
        
        # 使用校准器
        x_cal = calibrator(x_hat)
        
        if is_last_layer:
            x_normed = apply_layernorm(x.reshape(-1, C_dim), norm_w, norm_b).reshape(B, T, C_dim)
            x_cal_normed = apply_layernorm(x_cal.reshape(-1, C_dim), norm_w, norm_b).reshape(B, T, C_dim)
            
            if use_head:
                with torch.no_grad():
                    cls_target = x_normed[:, 0]
                    patch_target = x_normed[:, 1:].mean(dim=1)
                    logits_target = torch.cat([cls_target, patch_target], dim=1) @ head_weight.T
                
                cls_cal = x_cal_normed[:, 0]
                patch_cal = x_cal_normed[:, 1:].mean(dim=1)
                logits_cal = torch.cat([cls_cal, patch_cal], dim=1) @ head_weight.T
                loss = F.mse_loss(logits_cal, logits_target)
            else:
                loss = F.mse_loss(x_cal_normed, x_normed)
        else:
            x_normed = apply_layernorm(x.reshape(-1, C_dim), norm1_w, norm1_b).reshape(B, T, C_dim)
            x_cal_normed = apply_layernorm(x_cal.reshape(-1, C_dim), norm1_w, norm1_b).reshape(B, T, C_dim)
            
            with torch.no_grad():
                q_target = x_normed @ Wq
                k_target = x_normed @ Wk
                v_target = x_normed @ Wv
            
            q_hat = x_cal_normed @ Wq
            k_hat = x_cal_normed @ Wk
            v_hat = x_cal_normed @ Wv
            
            loss = (F.mse_loss(q_hat, q_target) + F.mse_loss(k_hat, k_target) +
                    F.mse_loss(v_hat, v_target))
        
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    # 返回校准器 (用于后续应用)
    return calibrator


def compute_layer_weights(num_layers, weight_mode='exp_inc'):
    """
    计算多层校准的层权重
    
    Args:
        num_layers: 层数
        weight_mode: 权重模式
            - 'exp_inc': 指数递增 2^i (越远权重越大，当前默认)
            - 'exp_dec': 指数递减 2^(-i) (越近权重越大)
            - 'uniform': 均匀权重
            - 'final_only': 只用最后一层
    
    Returns:
        layer_weights: 归一化的权重列表
        active_layers: 实际参与计算的层索引列表 (相对于 0~num_layers-1)
    """
    if weight_mode == 'exp_inc':
        # 指数递增: 2^0, 2^1, 2^2, ...
        weights = [2.0 ** i for i in range(num_layers)]
        active_layers = list(range(num_layers))
    
    elif weight_mode == 'exp_dec':
        # 指数递减: 2^(n-1), 2^(n-2), ..., 2^0
        weights = [2.0 ** (num_layers - 1 - i) for i in range(num_layers)]
        active_layers = list(range(num_layers))
    
    elif weight_mode == 'uniform':
        # 均匀权重
        weights = [1.0 for _ in range(num_layers)]
        active_layers = list(range(num_layers))
    
    elif weight_mode == 'final_only':
        # 只用最后一层 (但仍需传播)
        weights = [0.0] * (num_layers - 1) + [1.0]
        active_layers = [num_layers - 1]
    
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")
    
    # 归一化
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]
    
    return weights, active_layers


def fit_scale_bias_multilayer(x_list, xhat_list, backbone, layer_idx, num_layers=5,
                               steps=1000, lr=1e-2, batch_size=20, head=None,
                               weight_mode='exp_inc',
                               calibrator_type='scale_bias', rank=16, hidden_dim=64,
                               alpha=1.0, proj_mask='k', device='cuda'):
    """
    多层真实传播校准器 (v3.1 核心改进)
    
    让特征真正经过后续 block 的前向传播，在每层计算 QKV 投影误差。
    
    设计原理:
        ViT block 的输入 x 直接影响的是 LN1(x) → Q/K/V 投影
        - Q = LN1(x) @ Wq
        - K = LN1(x) @ Wk  
        - V = LN1(x) @ Wv
        
        MLP 分支的输入是 x' = x + attn_residual，已经不是原始 x
        因此只对齐 QKV（特别是 K）是理论上合理的设计选择
    
    真实传播路径:
        x_blk05 → [blk06] → x_blk06 → [blk07] → x_blk07 → ...
                     ↓                    ↓
                  loss6               loss7
    
    Args:
        x_list: 原始特征列表 (blk05 输出)
        xhat_list: 量化后特征列表 (blk05 输出的量化重建)
        backbone: DINOv2 backbone，用于真实前向传播
        layer_idx: 当前特征层索引 (如 blk05 → 5)
        num_layers: 要考虑的后续层数
        steps: 优化步数
        lr: 学习率
        batch_size: 批大小
        head: 分类头 (用于最后一层的校准)
        weight_mode: 层权重模式 (exp_inc, exp_dec, uniform, final_only)
        calibrator_type: 校准器类型 ('scale_bias', 'lora', 或 'mlp')
        rank: LoRA 秩
        hidden_dim: MLP 隐藏层维度
        proj_mask: 投影掩码，控制使用哪些 QKV 损失项
            - 'q': Q 投影
            - 'k': K 投影 (推荐，默认)
            - 'v': V 投影
            例如: 'k' (默认推荐), 'qk', 'qkv'
        device: 设备
    
    Returns:
        calibrator: 训练好的校准器
    """
    n_samples = len(x_list)
    C = x_list[0].shape[1]
    total_blocks = len(backbone.blocks)
    
    # 计算实际要传播的层数
    start_block = layer_idx + 1
    end_block = min(start_block + num_layers, total_blocks)
    actual_num_layers = end_block - start_block
    
    if actual_num_layers <= 0:
        # 如果没有后续层，退化为 next_layer 模式（使用最后的 norm + head）
        print(f"  警告: layer_idx={layer_idx} 没有后续 block，使用 next_layer 模式")
        block_weights = get_next_block_weights(backbone, layer_idx, head=head, device=device)
        return fit_scale_bias_next_layer(x_list, xhat_list, block_weights, steps, lr, batch_size,
                                         calibrator_type, rank, hidden_dim, device)
    
    print(f"  真实传播: block {start_block} → {end_block-1} (共 {actual_num_layers} 层)")
    type_info = calibrator_type
    if calibrator_type == 'lora':
        type_info += f", rank={rank}"
    elif calibrator_type == 'mlp':
        type_info += f", hidden_dim={hidden_dim}"
    print(f"  校准器类型: {type_info}")
    print(f"  投影掩码: {proj_mask}")
    
    # 统计量初始化
    X = np.concatenate(x_list, axis=0)
    XH = np.concatenate(xhat_list, axis=0)
    mu_x, std_x = X.mean(axis=0), X.std(axis=0) + 1e-6
    mu_xh, std_xh = XH.mean(axis=0), XH.std(axis=0) + 1e-6
    s_init = std_x / std_xh
    b_init = mu_x - s_init * mu_xh
    
    # 创建校准器
    calibrator = create_calibrator(calibrator_type, C, s_init, b_init, rank=rank, hidden_dim=hidden_dim,
                                   alpha=alpha, device=device)
    opt = torch.optim.AdamW(calibrator.parameters(), lr=lr)
    
    # 打印参数量
    num_params = sum(p.numel() for p in calibrator.parameters())
    print(f"  校准器参数量: {num_params:,} ({num_params/1e3:.1f}K)")
    
    # 获取每层的 QKV 投影权重 (用于计算投影误差)
    # 只保留 Q/K/V，因为这是 block 输入 x 的直接下游
    layer_proj_weights = []
    for blk_idx in range(start_block, end_block):
        block = backbone.blocks[blk_idx]
        attn = block.attn
        qkv_weight = attn.qkv.weight.data
        dim = qkv_weight.shape[1]
        proj_dict = {
            'Wq': qkv_weight[:dim, :].T.to(device),
            'Wk': qkv_weight[dim:2*dim, :].T.to(device),
            'Wv': qkv_weight[2*dim:, :].T.to(device),
            'norm1_weight': block.norm1.weight.data.to(device),
            'norm1_bias': block.norm1.bias.data.to(device),
        }
        layer_proj_weights.append(proj_dict)
    
    # 计算层权重
    layer_weights, active_layers = compute_layer_weights(actual_num_layers, weight_mode)
    print(f"  权重模式: {weight_mode}, 活跃层: {active_layers}, 权重: {[f'{w:.3f}' for w in layer_weights]}")
    
    for step in range(steps):
        indices = np.random.choice(n_samples, min(batch_size, n_samples), replace=False)
        x = torch.from_numpy(np.stack([x_list[i] for i in indices], axis=0)).float().to(device)
        x_hat = torch.from_numpy(np.stack([xhat_list[i] for i in indices], axis=0)).float().to(device)
        B, T, C_dim = x.shape
        
        # 使用校准器
        x_cal = calibrator(x_hat)
        
        total_loss = 0.0
        
        # 真实传播: 让原始特征和校准后特征经过后续 block
        x_prop = x.clone()       # 原始特征的传播状态
        x_cal_prop = x_cal       # 校准后特征的传播状态 (需要梯度)
        
        for layer_i, blk_idx in enumerate(range(start_block, end_block)):
            block = backbone.blocks[blk_idx]
            proj_w = layer_proj_weights[layer_i]
            weight = layer_weights[layer_i]
            
            # 在当前层计算 QKV 投影误差 (进入 block 之前)
            # 设计原理: block 输入 x 直接影响 LN1(x) → Q/K/V 投影
            norm1_w = proj_w['norm1_weight']
            norm1_b = proj_w['norm1_bias']
            Wq, Wk, Wv = proj_w['Wq'], proj_w['Wk'], proj_w['Wv']
            
            # 对传播状态应用该层的 norm1
            x_normed = apply_layernorm(x_prop.reshape(-1, C_dim), norm1_w, norm1_b).reshape(B, T, C_dim)
            x_cal_normed = apply_layernorm(x_cal_prop.reshape(-1, C_dim), norm1_w, norm1_b).reshape(B, T, C_dim)
            
            # 根据 proj_mask 计算损失
            # 各项损失按输出维度归一化，使不同项的尺度可比
            layer_loss = 0.0
            
            # Q 投影 (输出维度: dim)
            if 'q' in proj_mask:
                with torch.no_grad():
                    q_target = x_normed @ Wq
                q_hat = x_cal_normed @ Wq
                layer_loss = layer_loss + F.mse_loss(q_hat, q_target) / C_dim
            
            # K 投影 (输出维度: dim) - 推荐
            if 'k' in proj_mask:
                with torch.no_grad():
                    k_target = x_normed @ Wk
                k_hat = x_cal_normed @ Wk
                layer_loss = layer_loss + F.mse_loss(k_hat, k_target) / C_dim
            
            # V 投影 (输出维度: dim)
            if 'v' in proj_mask:
                with torch.no_grad():
                    v_target = x_normed @ Wv
                v_hat = x_cal_normed @ Wv
                layer_loss = layer_loss + F.mse_loss(v_hat, v_target) / C_dim
            
            # 真实前向传播到下一层
            with torch.no_grad():
                x_prop_next = block(x_prop)
            x_cal_prop_next = block(x_cal_prop)
            
            # X: block 输出对齐 (直接对齐 block 输出特征)
            if 'x' in proj_mask:
                with torch.no_grad():
                    x_out_target = x_prop_next
                layer_loss = layer_loss + F.mse_loss(x_cal_prop_next, x_out_target) / C_dim
            
            total_loss = total_loss + weight * layer_loss
            
            # 更新传播状态
            x_prop = x_prop_next
            x_cal_prop = x_cal_prop_next
        
        opt.zero_grad()
        total_loss.backward()
        opt.step()
    
    return calibrator


def fit_scale_bias(x_list, xhat_list, backbone, layer_idx, block_weights,
                   mode='next_layer', num_layers=5, steps=1000, lr=1e-2,
                   batch_size=20, head=None, weight_mode='exp_inc',
                   calibrator_type='scale_bias', rank=16, hidden_dim=64,
                   alpha=1.0, proj_mask='k', device='cuda'):
    """
    统一的校准器接口
    
    Args:
        mode: 校准模式
            - 'next_layer': v3 原版，只优化下一层
            - 'multi_layer': v3.1 真实传播，优化多层 QKV 投影误差
        num_layers: multi_layer 模式下考虑的层数
        head: 分类头 (用于最后一层的校准)
        weight_mode: 多层权重模式 (exp_inc, exp_dec, uniform, final_only)
        proj_mask: QKV 投影掩码 ('q'/'k'/'v'，可组合如 'qk', 'qkv')
        calibrator_type: 校准器类型 ('scale_bias', 'lora', 或 'mlp')
        rank: LoRA 秩 (仅 lora 类型使用)
        hidden_dim: MLP 隐藏层维度 (仅 mlp 类型使用)
        alpha: LoRA alpha 参数 (仅 lora 类型使用)
    
    Returns:
        calibrator: 训练好的校准器对象
    """
    if mode == 'next_layer':
        return fit_scale_bias_next_layer(x_list, xhat_list, block_weights, steps, lr, batch_size,
                                         calibrator_type, rank, hidden_dim, device)
    
    elif mode == 'multi_layer':
        return fit_scale_bias_multilayer(x_list, xhat_list, backbone, layer_idx, num_layers,
                                         steps, lr, batch_size, head=head,
                                         weight_mode=weight_mode,
                                         calibrator_type=calibrator_type, rank=rank,
                                         hidden_dim=hidden_dim, alpha=alpha,
                                         proj_mask=proj_mask,
                                         device=device)
    
    else:
        raise ValueError(f"Unknown calibration mode: {mode}")


def apply_scale_bias(xhat_list, s, b):
    """应用 scale-bias 校准 (兼容旧接口)"""
    return [(xh * s[np.newaxis, :] + b[np.newaxis, :]).astype(np.float32) for xh in xhat_list]


def apply_calibrator(xhat_list, calibrator, device='cuda'):
    """
    使用校准器对象应用校准
    
    Args:
        xhat_list: 量化后特征列表
        calibrator: 校准器对象 (ScaleBiasCalibrator 或 LoRACalibrator)
    
    Returns:
        校准后的特征列表
    """
    calibrator.eval()
    results = []
    with torch.no_grad():
        for xh in xhat_list:
            xh_t = torch.from_numpy(xh).float().unsqueeze(0).to(device)  # [1, T, C]
            xh_cal = calibrator(xh_t)
            results.append(xh_cal.squeeze(0).cpu().numpy().astype(np.float32))
    return results


# ============== 评估 ==============

def evaluate_accuracy(xhat_list, basenames, gt_dict, dino_wrapper, layer_idx, device):
    """评估分类准确率"""
    correct = 0
    total = 0
    for x_hat, basename in zip(xhat_list, basenames):
        if basename in gt_dict:
            label = gt_dict[basename]
            feat_tensor = torch.from_numpy(x_hat).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logits = dino_wrapper.forward_from_tokens(feat_tensor, layer_idx)
                pred = torch.argmax(logits, dim=1).item()
            if pred == label:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0


# ============== 数据加载 ==============

def _load_single_feature(f):
    """加载单个特征文件（用于并行）"""
    return np.load(f).astype(np.float32), f.stem


def preload_features(feat_files, num_workers=1, verbose=True):
    """预加载特征（支持多线程并行）"""
    if num_workers <= 1:
        features = []
        basenames = []
        for f in tqdm(feat_files, desc="Loading", disable=not verbose):
            features.append(np.load(f).astype(np.float32))
            basenames.append(f.stem)
        return features, basenames
    
    # 多线程并行加载（IO密集型用线程更高效）
    if verbose:
        print(f"  并行加载特征 (workers={num_workers})...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(_load_single_feature, feat_files),
            total=len(feat_files),
            desc="Loading",
            disable=not verbose
        ))
    
    features = [r[0] for r in results]
    basenames = [r[1] for r in results]
    return features, basenames


def load_gt(gt_path):
    """加载 ground truth"""
    gt_dict = {}
    with open(gt_path, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                parts = ln.split()
                if len(parts) >= 2:
                    gt_dict[parts[0]] = int(parts[1])
    return gt_dict


def compute_bpfp_entropy(codec_list, features, device, entropy_mode, norm_mode,
                         embedding_dim, verbose=False):
    """
    统一的 BPFP 熵编码计算函数
    
    Args:
        codec_list: VQ codec 列表
        features: 特征列表，每个元素 [T, D]
        device: 设备
        entropy_mode: 熵编码模式 ('global', 'grouped', 'position')
        norm_mode: 归一化模式
        embedding_dim: 向量维度
        verbose: 是否打印详细信息
    
    Returns:
        bpfp: 根据 entropy_mode 计算的 BPFP（含 side_info）
        vq_bytes: VQ 编码字节数
        side_info_bytes: 侧信息字节数
    """
    if entropy_mode == 'global':
        # 全局编码
        _, vq_bytes, side_info_bytes, _ = rvq_compress_all(
            codec_list, features, device, norm_mode=norm_mode
        )
    elif entropy_mode == 'grouped':
        # 分组编码
        _, vq_bytes, side_info_bytes, _, _ = rvq_compress_grouped(
            codec_list, features, device, norm_mode=norm_mode,
            embedding_dim=embedding_dim, verbose=verbose,
            normalize_fn=normalize, rvq_encode_decode_fn=rvq_encode_decode,
            compute_side_info_bytes_fn=compute_side_info_bytes
        )
    elif entropy_mode == 'position':
        # 位置条件化编码
        _, vq_bytes, side_info_bytes, _, _ = rvq_compress_position(
            codec_list, features, device, norm_mode=norm_mode,
            embedding_dim=embedding_dim, verbose=verbose,
            normalize_fn=normalize, rvq_encode_decode_fn=rvq_encode_decode,
            compute_side_info_bytes_fn=compute_side_info_bytes
        )
    else:
        raise ValueError(f"Unknown entropy_mode: {entropy_mode}")
    
    # 计算总 token 数
    total_tokens = sum(f.shape[0] for f in features)
    
    bpfp = compute_bpfp_from_bytes(vq_bytes, side_info_bytes, total_tokens)
    return bpfp, vq_bytes, side_info_bytes


# ============== 主流程 ==============

def run_experiment(args, device):
    """运行单次实验"""
    
    # 固定随机种子
    set_seed(args.seed)
    
    # 参数
    K1, K2 = args.K1, args.K2
    embedding_dim = args.embedding_dim
    calibrator_mode = args.calibrator_mode
    calibrator_layers = args.calibrator_layers
    weight_mode = args.weight_mode
    norm_mode = args.norm_mode
    
    K_list = [K1]
    if K2 > 0:
        K_list.append(K2)
    
    # BPFP (根据归一化模式计算侧信息)
    bpfp_fixed = compute_bpfp_fixed(K_list, embedding_dim, norm_mode=norm_mode)
    side_info_bits = compute_side_info_bits_per_token(norm_mode)
    
    print(f"\n{'='*70}")
    print(f"v3.1 多层真实传播校准器实验")
    print(f"{'='*70}")
    print(f"  层: {args.layer}")
    print(f"  K1={K1}, K2={K2}")
    print(f"  归一化模式: {norm_mode} (侧信息: {side_info_bits:.2f} bits/token)")
    print(f"  校准模式: {calibrator_mode}")
    print(f"  校准器类型: {args.calibrator_type}")
    if args.calibrator_type == 'lora':
        print(f"  LoRA rank: {args.rank}")
    elif args.calibrator_type == 'mlp':
        print(f"  MLP hidden_dim: {args.hidden_dim}")
    print(f"  校准层数: {calibrator_layers}")
    print(f"  权重模式: {weight_mode}")
    print(f"  熵编码模式: {args.entropy_mode}")
    print(f"  BPFP_fixed={bpfp_fixed:.4f}")
    print(f"{'='*70}\n")
    
    # 加载数据
    layer_idx = int(args.layer[-2:])
    train_dir = Path(args.feat_root) / args.train_subset / args.backbone / args.layer
    test_dir = Path(args.feat_root) / "test" / args.backbone / args.layer
    
    train_files = sorted(train_dir.glob("*.npy"))
    test_files = sorted(test_dir.glob("*.npy"))
    
    print(f"数据: train={len(train_files)}, test={len(test_files)}")
    
    # 设置并行worker数
    num_workers = args.num_workers
    if num_workers == 0:
        import os
        num_workers = os.cpu_count() or 1
    
    features_train, basenames_train = preload_features(train_files, num_workers=num_workers)
    features_test, basenames_test = preload_features(test_files, num_workers=num_workers)
    
    # 加载 GT
    gt_test = load_gt(args.gt_path)
    
    # 加载 DINOv2
    dino_wrapper = Dinov2Wrapper(head_layers=1, device=device)
    
    # 获取下一层权重 (用于 next_layer 校准)
    block_weights = get_next_block_weights(dino_wrapper.backbone, layer_idx,
                                           head=dino_wrapper.head, device=device)
    
    # =========== VQ1: K-means ===========
    print(f"\n[Phase 1] VQ1 K-means (K={K1})...")
    vectors = flatten_features(features_train, embedding_dim, num_workers=num_workers, norm_mode=norm_mode)
    print(f"  向量数: {vectors.shape[0]}")
    
    codebook1 = kmeans(vectors, K1, max_iter=args.kmeans_max_iter, device=device, verbose=True)
    codec1 = create_vq_codec(K1, embedding_dim, codebook1, device)
    
    # =========== VQ2: K-means (残差量化) ===========
    codec_list = [codec1]
    if K2 > 0:
        print(f"\n[Phase 2] VQ2 K-means (K={K2})...")
        residuals1 = collect_residuals([codec1], features_train, device, norm_mode=norm_mode)
        vectors2 = flatten_residuals(residuals1, embedding_dim, num_workers=num_workers)
        print(f"  向量数: {vectors2.shape[0]}")
        
        codebook2 = kmeans(vectors2, K2, max_iter=args.kmeans_max_iter, device=device, verbose=True)
        codec2 = create_vq_codec(K2, embedding_dim, codebook2, device)
        
        codec_list.append(codec2)
    
    # =========== 校准器训练 (v3.1 核心改进) ===========
    calibrator_type = args.calibrator_type
    rank = args.rank
    alpha = args.alpha
    hidden_dim = args.hidden_dim
    proj_mask = args.proj_mask
    print(f"\n[Phase 3] 校准器训练 (mode={calibrator_mode}, type={calibrator_type}, layers={calibrator_layers}, weight={weight_mode})...")
    if calibrator_type == 'lora':
        print(f"  LoRA rank={rank}, alpha={alpha}, scaling={alpha/rank:.4f}")
    elif calibrator_type == 'mlp':
        print(f"  MLP hidden_dim={hidden_dim}")
    print(f"  投影掩码: {proj_mask}")
    xhat_train = rvq_encode_decode(codec_list, features_train, device, norm_mode=norm_mode)
    
    calibrator = fit_scale_bias(
        features_train, xhat_train,
        backbone=dino_wrapper.backbone,
        layer_idx=layer_idx,
        block_weights=block_weights,
        mode=calibrator_mode,
        num_layers=calibrator_layers,
        steps=args.calibrator_steps,
        lr=args.calibrator_lr,
        batch_size=args.calibrator_batch_size,
        head=dino_wrapper.head,
        weight_mode=weight_mode,
        calibrator_type=calibrator_type,
        rank=rank,
        hidden_dim=hidden_dim,
        alpha=alpha,
        proj_mask=proj_mask,
        device=device
    )
    print(f"  校准器训练完成")
    
    entropy_mode = args.entropy_mode  # 'global', 'grouped', 'position'
    num_test_images = len(features_test)
    
    # =========== 分类评估 ===========
    print(f"\n[Phase 4] 分类评估...")
    
    # 使用完整编解码流程
    xhat_test_cal, total_bytes_pipeline, timing = full_encode_decode_pipeline(
        codec_list, features_test, device, 
        norm_mode=norm_mode,
        embedding_dim=embedding_dim,
        entropy_mode=entropy_mode,
        calibrator=calibrator,
        verbose=True,
        normalize_fn=normalize, inv_normalize_fn=inv_normalize,
        compute_side_info_bytes_fn=compute_side_info_bytes
    )
    
    # 计算准确率
    acc = evaluate_accuracy(xhat_test_cal, basenames_test, gt_test,
                           dino_wrapper, layer_idx, device)
    print(f"  分类 Acc: {acc:.4f}")
    
    # 计算分类 BPFP（含 side_info）
    cls_bpfp, _, _ = compute_bpfp_entropy(
        codec_list, features_test, device, entropy_mode, norm_mode,
        embedding_dim, verbose=False
    )
    print(f"  分类 BPFP ({entropy_mode}): {cls_bpfp:.4f}")
    print(f"  分类 BPFP_fixed: {bpfp_fixed:.4f}")
    
    # =========== 分割评估 (可选) ===========
    seg_miou = 0.0
    seg_bpfp = 0.0
    
    if args.seg_eval:
        print(f"\n[Phase 5] 分割评估...")
        
        # 加载分割特征（格式: [num_slides, 1+N, D]，按 slide 展平）
        seg_feat_dir = Path(args.seg_feat_root) / args.backbone / args.layer
        seg_feat_files = sorted(seg_feat_dir.glob("*.npy"))
        print(f"  分割特征目录: {seg_feat_dir}")
        print(f"  分割特征文件数: {len(seg_feat_files)}")
        
        if len(seg_feat_files) == 0:
            print(f"  [warn] 分割特征目录为空，跳过分割评估")
        else:
            # 按 slide 展平：[num_slides, 1+N, D] → num_slides 个 [1+N, D]
            features_seg_flat = []
            for f in seg_feat_files:
                arr = np.load(f).astype(np.float32)  # [num_slides, 1+N, D]
                for s in range(arr.shape[0]):
                    features_seg_flat.append(arr[s])  # [1+N, D]
            
            print(f"  展平后 slide 数: {len(features_seg_flat)} "
                  f"(来自 {len(seg_feat_files)} 张图片)")
            
            # 计算分割 BPFP（每个 slide 独立编码，和分类完全一致）
            seg_bpfp, _, _ = compute_bpfp_entropy(
                codec_list, features_seg_flat, device, entropy_mode, norm_mode,
                embedding_dim, verbose=False
            )
            print(f"  分割 BPFP ({entropy_mode}): {seg_bpfp:.4f}")
            
            # mIoU 评估（slide inference）
            try:
                from backbone.wrapper import SegmentationEvaluator
                
                print(f"\n  mIoU 评估 (slide inference)...")
                seg_evaluator = SegmentationEvaluator(
                    codec_list=codec_list,
                    calibrator=calibrator,
                    layer_idx=layer_idx,
                    voc_root=args.voc_root,
                    weights_root=dino_wrapper.weights_root,
                    device=device,
                    norm_mode=norm_mode,
                    feat_dim=1024
                )
                
                seg_results = seg_evaluator.evaluate(
                    seg_feat_dir=str(seg_feat_dir),
                    image_list=args.seg_image_list,
                    verbose=True
                )
                
                seg_miou = seg_results['miou']
                print(f"\n  分割 mIoU: {seg_miou*100:.2f}%")
                print(f"  分割 aAcc: {seg_results['acc']*100:.2f}%")
                
                # 打印每类 IoU
                if len(seg_results['class_iou']) > 0:
                    print("\n  Per-class IoU:")
                    for i, (name, iou) in enumerate(zip(seg_evaluator.VOC_CLASSES, seg_results['class_iou'])):
                        print(f"    {i:2d}. {name:15s}: {iou*100:.2f}%")
                        
            except ImportError as e:
                print(f"  [warn] mmseg 未安装，跳过 mIoU 评估: {e}")
            except Exception as e:
                print(f"  [error] mIoU 评估失败: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"结果:")
    print(f"  分类 BPFP_fixed: {bpfp_fixed:.4f}")
    print(f"  分类 BPFP ({entropy_mode}): {cls_bpfp:.4f}")
    print(f"  分类 Acc: {acc:.4f}")
    if args.seg_eval:
        print(f"  分割 BPFP ({entropy_mode}): {seg_bpfp:.4f}")
        print(f"  分割 mIoU: {seg_miou*100:.2f}%")
    print(f"{'='*70}\n")
    
    result = {
        'layer': args.layer,
        'K1': K1,
        'K2': K2,
        'embedding_dim': embedding_dim,
        'norm_mode': norm_mode,
        'calibrator_type': calibrator_type,
        'calibrator_layers': calibrator_layers,
        'entropy_mode': entropy_mode,
        'BPFP_fixed': bpfp_fixed,
        'cls_bpfp': cls_bpfp,
        'acc': acc,
        'seg_bpfp': seg_bpfp,
        'seg_miou': seg_miou,
    }
    
    return result


def save_result(result, output_csv):
    """保存结果到 CSV (带文件锁)"""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ['layer', 'K1', 'K2', 'embedding_dim', 'norm_mode',
                  'calibrator_type', 'calibrator_layers', 'entropy_mode',
                  'BPFP_fixed', 'cls_bpfp', 'acc', 'seg_bpfp', 'seg_miou']
    
    with open(output_path, 'a', newline='') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            file_empty = output_path.stat().st_size == 0
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if file_empty:
                writer.writeheader()
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in result.items()})
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    print(f"结果已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="v3.1 多层真实传播校准器实验")
    
    # VQ 参数
    parser.add_argument("--K1", type=int, default=64, help="VQ1 码本大小")
    parser.add_argument("--K2", type=int, default=0, help="VQ2 码本大小 (0=禁用)")
    parser.add_argument("--embedding_dim", type=int, default=32, help="向量维度")
    
    # 归一化模式
    parser.add_argument("--norm_mode", type=str, default="per_token_ln",
                        choices=["per_token_ln", "per_image"],
                        help="归一化模式: "
                             "per_token_ln (32 bits/token, baseline), "
                             "per_image (~0.12 bits/token)")
    
    # 校准器模式
    parser.add_argument("--calibrator_mode", type=str, default="multi_layer",
                        choices=["next_layer", "multi_layer"],
                        help="校准器模式: next_layer, multi_layer(真实传播)")
    parser.add_argument("--calibrator_layers", type=int, default=10,
                        help="multi_layer 模式下考虑的层数")
    parser.add_argument("--weight_mode", type=str, default="exp_inc",
                        choices=["exp_inc", "exp_dec", "uniform", "final_only"],
                        help="多层权重模式")
    
    # 校准器架构参数
    parser.add_argument("--calibrator_type", type=str, default="lora",
                        choices=["scale_bias", "lora", "mlp"],
                        help="校准器类型: scale_bias(对角变换), lora(对角+低秩), mlp(残差MLP)")
    parser.add_argument("--rank", type=int, default=32,
                        help="LoRA 秩 (仅 lora 类型使用)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="LoRA alpha 参数，scaling = alpha/rank (仅 lora 类型使用)")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="MLP 隐藏层维度 (仅 mlp 类型使用)")
    
    # 投影选择参数
    parser.add_argument("--proj_mask", type=str, default="x",
                        help="投影掩码: q(Q)/k(K)/v(V)/x(block输出), 如 'k'(推荐), 'x', 'kx', 'qkv'")
    
    # 熵编码模式
    parser.add_argument("--entropy_mode", type=str, default="global",
                        choices=["global", "grouped", "position"],
                        help="熵编码模式: "
                             "global (全局单一 prior), "
                             "grouped (group 条件化, 32 个表), "
                             "position (位置+group 条件化, 256×32 个表)")
    
    # 训练参数
    parser.add_argument("--kmeans_max_iter", type=int, default=100)
    parser.add_argument("--calibrator_steps", type=int, default=2000)
    parser.add_argument("--calibrator_lr", type=float, default=0.01)
    parser.add_argument("--calibrator_batch_size", type=int, default=20,
                        help="校准器训练批大小")
    
    # 并行参数
    parser.add_argument("--num_workers", type=int, default=32,
                        help="并行加载/处理的worker数 (0=使用CPU核心数)")
    
    # 随机种子
    parser.add_argument("--seed", type=int, default=42)
    
    # 路径
    parser.add_argument("--feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features"))
    parser.add_argument("--train_subset", type=str, default="train")
    parser.add_argument("--backbone", type=str, default="dinov2_vitl14")
    parser.add_argument("--layer", type=str, default="blk05")
    # 分类任务参数
    parser.add_argument("--gt_path", type=str,
                        default=os.path.join(PROJECT_ROOT, "utils", "imagenet_selected_label500.txt"))
    
    # 分割任务参数
    parser.add_argument("--seg_eval", action="store_true",
                        help="是否进行分割 mIoU 评估 (mmseg 官方 slide 模式)")
    parser.add_argument("--voc_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "VOCdevkit", "VOC2012"),
                        help="VOC2012 根目录")
    parser.add_argument("--seg_feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features", "voc2012_100"),
                        help="分割特征根目录（预提取的 VOC 特征）")
    parser.add_argument("--seg_image_list", type=str, default=os.path.join(PROJECT_ROOT, "utils", "voc2012_val_100.txt"),
                        help="分割评估图片列表文件（每行一个名称，不含扩展名），默认使用全部验证集")
    
    parser.add_argument("--output_csv", type=str, default=None)
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    result = run_experiment(args, device)
    
    if args.output_csv:
        save_result(result, args.output_csv)
    
    return result


if __name__ == "__main__":
    main()
