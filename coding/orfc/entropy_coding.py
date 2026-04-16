"""
分组静态熵编码 (Group-Conditioned Static rANS)

设计说明:
  支持多种条件化熵编码模式：
  - GroupConditioned: 每个 group 使用独立的频率表 (32 个表)
  - PositionConditioned: 每个 (位置, group) 使用独立的频率表 (256×32 个表)

分组策略:
  - CLS level1: 1 token × 32 groups = 32 个索引
  - CLS level2: 1 token × 32 groups = 32 个索引 (如果有)
  - Patch level1: 256 tokens × 32 groups = 8192 个索引
  - Patch level2: 256 tokens × 32 groups = 8192 个索引 (如果有)
"""

import math
import numpy as np
import torch
from typing import List, Tuple, Optional

# 尝试导入 compressai
try:
    from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
    from compressai import ans
    HAS_COMPRESSAI = True
except (ImportError, ModuleNotFoundError):
    _pmf_to_quantized_cdf = None
    ans = None
    HAS_COMPRESSAI = False


class GroupConditionedEntropyModel:
    """
    分组条件化熵编码模型
    
    每个 group 使用独立的频率表，支持 CLS/Patch 分离编码
    
    索引形状说明 (单张图):
        - codec 输出索引: flatten 后是 [257 * 32] = [8224]
        - 排列顺序: token 优先，即 [t0_g0, t0_g1, ..., t0_g31, t1_g0, ...]
        - CLS: indices[0:32]
        - Patches: indices[32:8224]
    """
    
    def __init__(self, K: int, num_groups: int = 32, num_tokens: int = 257,
                 precision: int = 16):
        """
        Args:
            K: 码本大小
            num_groups: group 数量 (特征维度 / embedding_dim)
            num_tokens: token 数量 (1 CLS + 256 patches)
            precision: rANS 精度
        """
        self.K = K
        self.num_groups = num_groups
        self.num_tokens = num_tokens
        self.precision = precision
        
        # 频率表: [num_groups, K] 分别用于 CLS 和 Patch
        self.cls_counts = np.zeros((num_groups, K), dtype=np.float64)
        self.patch_counts = np.zeros((num_groups, K), dtype=np.float64)
        
        # CDF 表 (延迟构建)
        self.cls_cdfs = None
        self.patch_cdfs = None
        
        # rANS 编解码器
        if HAS_COMPRESSAI:
            self._encoder = ans.RansEncoder()
            self._decoder = ans.RansDecoder()
        else:
            self._encoder = None
            self._decoder = None
    
    def _reshape_indices_per_image(self, indices_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将 flatten 的索引重塑为 [tokens, groups] 格式，并分离 CLS 和 Patch
        
        Args:
            indices_flat: flatten 后的索引 [num_tokens * num_groups]
        
        Returns:
            cls_indices: [num_groups] CLS 的索引
            patch_indices: [num_patches, num_groups] Patch 的索引
        """
        # 重塑为 [tokens, groups]
        indices_2d = indices_flat.reshape(self.num_tokens, self.num_groups)
        
        # 分离 CLS 和 Patch
        cls_indices = indices_2d[0, :]  # [num_groups]
        patch_indices = indices_2d[1:, :]  # [num_patches, num_groups]
        
        return cls_indices, patch_indices
    
    def update_counts(self, all_indices: List[np.ndarray]):
        """
        统计频率表
        
        Args:
            all_indices: 所有图像的索引列表，每个元素是 [num_tokens * num_groups]
        """
        self.cls_counts.fill(0)
        self.patch_counts.fill(0)
        
        for indices_flat in all_indices:
            cls_idx, patch_idx = self._reshape_indices_per_image(indices_flat)
            
            # 统计 CLS
            for g in range(self.num_groups):
                k = int(cls_idx[g])
                self.cls_counts[g, k] += 1
            
            # 统计 Patch
            for g in range(self.num_groups):
                for k in patch_idx[:, g]:
                    self.patch_counts[g, int(k)] += 1
        
        # 构建 CDF 表
        self._build_cdfs()
    
    def _build_cdfs(self, smooth_eps: float = 1e-6):
        """从频率表构建 CDF"""
        if not HAS_COMPRESSAI:
            return
        
        self.cls_cdfs = []
        self.patch_cdfs = []
        
        for g in range(self.num_groups):
            # CLS CDF
            counts = self.cls_counts[g] + smooth_eps
            pmf = counts / counts.sum()
            cdf = self._pmf_to_cdf(pmf)
            self.cls_cdfs.append(cdf)
            
            # Patch CDF
            counts = self.patch_counts[g] + smooth_eps
            pmf = counts / counts.sum()
            cdf = self._pmf_to_cdf(pmf)
            self.patch_cdfs.append(cdf)
    
    def _pmf_to_cdf(self, pmf: np.ndarray) -> List[int]:
        """将 PMF 转换为整数 CDF"""
        pmf_tensor = torch.from_numpy(pmf).float()
        overflow = (1.0 - pmf_tensor.sum()).clamp_min(0)
        pmf_tensor = torch.cat([pmf_tensor, overflow.unsqueeze(0)], dim=0)
        # _pmf_to_quantized_cdf 返回 list of int
        cdf = _pmf_to_quantized_cdf(pmf_tensor.tolist(), self.precision)
        return cdf
    
    def compute_entropy(self) -> dict:
        """
        计算各部分的经验熵 (理论下限)
        
        Returns:
            dict with entropy in bits for each part
        """
        def entropy_from_counts(counts):
            """从频率计算熵"""
            total = counts.sum()
            if total == 0:
                return 0.0
            probs = counts / total
            probs = probs[probs > 0]  # 避免 log(0)
            return -np.sum(probs * np.log2(probs))
        
        cls_entropy_per_group = []
        patch_entropy_per_group = []
        
        for g in range(self.num_groups):
            cls_entropy_per_group.append(entropy_from_counts(self.cls_counts[g]))
            patch_entropy_per_group.append(entropy_from_counts(self.patch_counts[g]))
        
        # 计算加权平均熵
        num_cls = self.cls_counts.sum()
        num_patch = self.patch_counts.sum()
        
        # 每个 group 的 CLS 数量相同，Patch 数量相同
        cls_avg_entropy = np.mean(cls_entropy_per_group)
        patch_avg_entropy = np.mean(patch_entropy_per_group)
        
        # 全局混合熵 (不分 group，用于对比)
        cls_global_counts = self.cls_counts.sum(axis=0)
        patch_global_counts = self.patch_counts.sum(axis=0)
        cls_global_entropy = entropy_from_counts(cls_global_counts)
        patch_global_entropy = entropy_from_counts(patch_global_counts)
        
        return {
            'cls_entropy_per_group': cls_entropy_per_group,
            'patch_entropy_per_group': patch_entropy_per_group,
            'cls_avg_entropy': cls_avg_entropy,
            'patch_avg_entropy': patch_avg_entropy,
            'cls_global_entropy': cls_global_entropy,
            'patch_global_entropy': patch_global_entropy,
            # 理论比特数
            'cls_bits_grouped': cls_avg_entropy * num_cls / self.num_groups,
            'patch_bits_grouped': patch_avg_entropy * num_patch / self.num_groups,
            'cls_bits_global': cls_global_entropy * num_cls / self.num_groups,
            'patch_bits_global': patch_global_entropy * num_patch / self.num_groups,
        }
    
    def compress(self, indices_flat: np.ndarray) -> Tuple[bytes, bytes]:
        """
        压缩单张图的索引
        
        Args:
            indices_flat: flatten 后的索引 [num_tokens * num_groups]
        
        Returns:
            cls_string: CLS 索引的压缩字节串
            patch_string: Patch 索引的压缩字节串
        """
        if not HAS_COMPRESSAI or self.cls_cdfs is None:
            raise RuntimeError("CDF tables not built or compressai not available")
        
        cls_idx, patch_idx = self._reshape_indices_per_image(indices_flat)
        
        # 编码 CLS (每个 group 使用对应的 CDF)
        cls_symbols = []
        cls_cdf_indices = []
        for g in range(self.num_groups):
            cls_symbols.append(int(cls_idx[g]))
            cls_cdf_indices.append(g)
        
        cls_string = self._encoder.encode_with_indexes(
            cls_symbols, cls_cdf_indices,
            self.cls_cdfs,
            [self.K + 2] * self.num_groups,
            [0] * self.num_groups
        )
        
        # 编码 Patch (按 group 分组)
        patch_symbols = []
        patch_cdf_indices = []
        num_patches = patch_idx.shape[0]
        for t in range(num_patches):
            for g in range(self.num_groups):
                patch_symbols.append(int(patch_idx[t, g]))
                patch_cdf_indices.append(g)
        
        patch_string = self._encoder.encode_with_indexes(
            patch_symbols, patch_cdf_indices,
            self.patch_cdfs,
            [self.K + 2] * self.num_groups,
            [0] * self.num_groups
        )
        
        return cls_string, patch_string
    
    def decompress(self, cls_string: bytes, patch_string: bytes) -> np.ndarray:
        """
        解压索引
        
        Returns:
            indices_flat: flatten 后的索引 [num_tokens * num_groups]
        """
        if not HAS_COMPRESSAI or self.cls_cdfs is None:
            raise RuntimeError("CDF tables not built or compressai not available")
        
        # 解码 CLS
        cls_cdf_indices = list(range(self.num_groups))
        cls_symbols = self._decoder.decode_with_indexes(
            cls_string, cls_cdf_indices,
            self.cls_cdfs,
            [self.K + 2] * self.num_groups,
            [0] * self.num_groups
        )
        
        # 解码 Patch
        num_patches = self.num_tokens - 1
        patch_cdf_indices = []
        for t in range(num_patches):
            for g in range(self.num_groups):
                patch_cdf_indices.append(g)
        
        patch_symbols = self._decoder.decode_with_indexes(
            patch_string, patch_cdf_indices,
            self.patch_cdfs,
            [self.K + 2] * self.num_groups,
            [0] * self.num_groups
        )
        
        # 重组索引
        cls_idx = np.array(cls_symbols, dtype=np.int64)
        patch_idx = np.array(patch_symbols, dtype=np.int64).reshape(num_patches, self.num_groups)
        
        indices_2d = np.vstack([cls_idx.reshape(1, -1), patch_idx])
        return indices_2d.flatten()


class PositionConditionedEntropyModel:
    """
    位置条件化熵编码模型
    
    为 16×16 的每个位置和每个 group 单独统计频率表 P(k|r,c,g)
    
    频率表数量:
        - CLS: 1 × 32 = 32 个表 (只有 group 条件)
        - Patch: 16 × 16 × 32 = 8192 个表 (位置 + group 条件)
    
    索引形状说明 (单张图):
        - 总索引: [257 * 32] = [8224]
        - 排列顺序: token 优先，即 [t0_g0, t0_g1, ..., t0_g31, t1_g0, ...]
        - CLS: indices[0:32]
        - Patches: indices[32:8224], reshape 为 [256, 32] 即 [位置, group]
        - 位置顺序: raster scan (行优先), 即 (0,0), (0,1), ..., (0,15), (1,0), ...
    """
    
    def __init__(self, K: int, num_groups: int = 32, num_tokens: int = 257,
                 grid_size: int = 16, precision: int = 16):
        """
        Args:
            K: 码本大小
            num_groups: group 数量 (特征维度 / embedding_dim)
            num_tokens: token 数量 (1 CLS + 256 patches)
            grid_size: patch 网格大小 (16 for 16×16)
            precision: rANS 精度
        """
        self.K = K
        self.num_groups = num_groups
        self.num_tokens = num_tokens
        self.grid_size = grid_size
        self.num_positions = grid_size * grid_size  # 256
        self.precision = precision
        
        # CLS 频率表: [num_groups, K] (与 GroupConditioned 相同)
        self.cls_counts = np.zeros((num_groups, K), dtype=np.float64)
        
        # Patch 频率表: [num_positions, num_groups, K] = [256, 32, K]
        # 即 P(k | position, group)
        self.patch_counts = np.zeros((self.num_positions, num_groups, K), dtype=np.float64)
        
        # CDF 表 (延迟构建)
        self.cls_cdfs = None
        self.patch_cdfs = None  # [num_positions * num_groups] 个 CDF
        
        # rANS 编解码器
        if HAS_COMPRESSAI:
            self._encoder = ans.RansEncoder()
            self._decoder = ans.RansDecoder()
        else:
            self._encoder = None
            self._decoder = None
    
    def _reshape_indices_per_image(self, indices_flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将 flatten 的索引重塑为结构化格式
        
        Args:
            indices_flat: flatten 后的索引 [num_tokens * num_groups]
        
        Returns:
            cls_indices: [num_groups] CLS 的索引
            patch_indices: [num_positions, num_groups] Patch 的索引
        """
        indices_2d = indices_flat.reshape(self.num_tokens, self.num_groups)
        cls_indices = indices_2d[0, :]  # [num_groups]
        patch_indices = indices_2d[1:, :]  # [num_positions, num_groups]
        return cls_indices, patch_indices
    
    def update_counts(self, all_indices: List[np.ndarray]):
        """
        统计频率表
        
        Args:
            all_indices: 所有图像的索引列表，每个元素是 [num_tokens * num_groups]
        """
        self.cls_counts.fill(0)
        self.patch_counts.fill(0)
        
        for indices_flat in all_indices:
            cls_idx, patch_idx = self._reshape_indices_per_image(indices_flat)
            
            # 统计 CLS (只有 group 条件)
            for g in range(self.num_groups):
                k = int(cls_idx[g])
                self.cls_counts[g, k] += 1
            
            # 统计 Patch (位置 + group 条件)
            for pos in range(self.num_positions):
                for g in range(self.num_groups):
                    k = int(patch_idx[pos, g])
                    self.patch_counts[pos, g, k] += 1
        
        # 构建 CDF 表
        self._build_cdfs()
    
    def _pmf_to_cdf(self, pmf: np.ndarray) -> List[int]:
        """将 PMF 转换为整数 CDF"""
        pmf_tensor = torch.from_numpy(pmf).float()
        overflow = (1.0 - pmf_tensor.sum()).clamp_min(0)
        pmf_tensor = torch.cat([pmf_tensor, overflow.unsqueeze(0)], dim=0)
        cdf = _pmf_to_quantized_cdf(pmf_tensor.tolist(), self.precision)
        return cdf
    
    def _build_cdfs(self, smooth_eps: float = 1e-6):
        """从频率表构建 CDF"""
        if not HAS_COMPRESSAI:
            return
        
        # CLS CDF: [num_groups] 个
        self.cls_cdfs = []
        for g in range(self.num_groups):
            counts = self.cls_counts[g] + smooth_eps
            pmf = counts / counts.sum()
            self.cls_cdfs.append(self._pmf_to_cdf(pmf))
        
        # Patch CDF: [num_positions * num_groups] 个
        # 索引方式: cdf_index = pos * num_groups + g
        self.patch_cdfs = []
        for pos in range(self.num_positions):
            for g in range(self.num_groups):
                counts = self.patch_counts[pos, g] + smooth_eps
                pmf = counts / counts.sum()
                self.patch_cdfs.append(self._pmf_to_cdf(pmf))
    
    def compute_entropy(self) -> dict:
        """
        计算各部分的经验熵 (理论下限)
        
        Returns:
            dict with entropy statistics
        """
        def entropy_from_counts(counts):
            """从频率计算熵"""
            total = counts.sum()
            if total == 0:
                return 0.0
            probs = counts / total
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs))
        
        # CLS 熵 (按 group)
        cls_entropy_per_group = [entropy_from_counts(self.cls_counts[g]) 
                                  for g in range(self.num_groups)]
        cls_avg_entropy = np.mean(cls_entropy_per_group)
        
        # Patch 熵 (按位置和 group)
        patch_entropy_per_pos_group = np.zeros((self.num_positions, self.num_groups))
        for pos in range(self.num_positions):
            for g in range(self.num_groups):
                patch_entropy_per_pos_group[pos, g] = entropy_from_counts(
                    self.patch_counts[pos, g]
                )
        
        # 位置条件化的平均熵
        patch_avg_entropy_position = np.mean(patch_entropy_per_pos_group)
        
        # 仅 group 条件化的熵 (用于对比)
        patch_counts_by_group = self.patch_counts.sum(axis=0)  # [num_groups, K]
        patch_entropy_per_group = [entropy_from_counts(patch_counts_by_group[g])
                                    for g in range(self.num_groups)]
        patch_avg_entropy_group = np.mean(patch_entropy_per_group)
        
        # 全局熵
        patch_global_counts = self.patch_counts.sum(axis=(0, 1))  # [K]
        patch_global_entropy = entropy_from_counts(patch_global_counts)
        
        # 计算收益
        position_gain = patch_avg_entropy_group - patch_avg_entropy_position
        
        return {
            'cls_avg_entropy': cls_avg_entropy,
            'patch_avg_entropy_position': patch_avg_entropy_position,  # 位置条件化
            'patch_avg_entropy_group': patch_avg_entropy_group,  # 仅 group 条件化
            'patch_global_entropy': patch_global_entropy,  # 全局
            'position_gain': position_gain,  # 位置条件化相比 group 条件化的收益
        }
    
    def compress(self, indices_flat: np.ndarray) -> Tuple[bytes, bytes]:
        """
        压缩单张图的索引
        
        Args:
            indices_flat: flatten 后的索引 [num_tokens * num_groups]
        
        Returns:
            cls_string: CLS 索引的压缩字节串
            patch_string: Patch 索引的压缩字节串
        """
        if not HAS_COMPRESSAI or self.cls_cdfs is None:
            raise RuntimeError("CDF tables not built or compressai not available")
        
        cls_idx, patch_idx = self._reshape_indices_per_image(indices_flat)
        
        # 编码 CLS (group 条件)
        cls_symbols = [int(cls_idx[g]) for g in range(self.num_groups)]
        cls_cdf_indices = list(range(self.num_groups))
        
        cls_string = self._encoder.encode_with_indexes(
            cls_symbols, cls_cdf_indices,
            self.cls_cdfs,
            [self.K + 2] * self.num_groups,
            [0] * self.num_groups
        )
        
        # 编码 Patch (位置 + group 条件)
        # cdf_index = pos * num_groups + g
        patch_symbols = []
        patch_cdf_indices = []
        num_patch_cdfs = self.num_positions * self.num_groups
        
        for pos in range(self.num_positions):
            for g in range(self.num_groups):
                patch_symbols.append(int(patch_idx[pos, g]))
                patch_cdf_indices.append(pos * self.num_groups + g)
        
        patch_string = self._encoder.encode_with_indexes(
            patch_symbols, patch_cdf_indices,
            self.patch_cdfs,
            [self.K + 2] * num_patch_cdfs,
            [0] * num_patch_cdfs
        )
        
        return cls_string, patch_string
    
    def decompress(self, cls_string: bytes, patch_string: bytes) -> np.ndarray:
        """
        解压索引
        
        Returns:
            indices_flat: flatten 后的索引 [num_tokens * num_groups]
        """
        if not HAS_COMPRESSAI or self.cls_cdfs is None:
            raise RuntimeError("CDF tables not built or compressai not available")
        
        # 解码 CLS
        cls_cdf_indices = list(range(self.num_groups))
        cls_symbols = self._decoder.decode_with_indexes(
            cls_string, cls_cdf_indices,
            self.cls_cdfs,
            [self.K + 2] * self.num_groups,
            [0] * self.num_groups
        )
        
        # 解码 Patch
        num_patch_cdfs = self.num_positions * self.num_groups
        patch_cdf_indices = []
        for pos in range(self.num_positions):
            for g in range(self.num_groups):
                patch_cdf_indices.append(pos * self.num_groups + g)
        
        patch_symbols = self._decoder.decode_with_indexes(
            patch_string, patch_cdf_indices,
            self.patch_cdfs,
            [self.K + 2] * num_patch_cdfs,
            [0] * num_patch_cdfs
        )
        
        # 重组索引
        cls_idx = np.array(cls_symbols, dtype=np.int64)
        patch_idx = np.array(patch_symbols, dtype=np.int64).reshape(
            self.num_positions, self.num_groups
        )
        
        indices_2d = np.vstack([cls_idx.reshape(1, -1), patch_idx])
        return indices_2d.flatten()


# ============== 压缩/解压函数 ==============

def collect_rvq_indices_structured(codec_list, features, device, norm_mode='per_token_ln',
                                    embedding_dim=32, normalize_fn=None):
    """
    收集 RVQ 索引，保持结构化格式
    
    Args:
        codec_list: VQ codec 列表 [codec1, codec2, ...]
        features: 特征列表 [N, T, C]
        device: 设备
        norm_mode: 归一化模式
        embedding_dim: embedding 维度
        normalize_fn: 归一化函数 (从主模块传入)
    
    Returns:
        level_indices: 每层的索引列表
    """
    if normalize_fn is None:
        raise ValueError("normalize_fn must be provided")
    
    y_list = [normalize_fn(x, mode=norm_mode)[0] for x in features]
    y_hat_list = [np.zeros_like(y) for y in y_list]
    
    num_tokens = features[0].shape[0]
    feat_dim = features[0].shape[1]
    num_groups = feat_dim // embedding_dim
    
    level_indices = []
    
    for level, codec in enumerate(codec_list):
        if codec is None:
            level_indices.append(None)
            continue
        
        r_list = [y - y_hat for y, y_hat in zip(y_list, y_hat_list)]
        this_level_indices = []
        
        for i, r in enumerate(r_list):
            r_t = torch.from_numpy(r.T).float().to(device)
            with torch.no_grad():
                outputs = codec(r_t.unsqueeze(0))
                encoding_inds = outputs[4]
            
            indices_flat = encoding_inds[0].cpu().numpy().flatten()
            this_level_indices.append(indices_flat)
            
            r_hat = outputs[0].squeeze(0).T.cpu().numpy()
            y_hat_list[i] = y_hat_list[i] + r_hat
        
        level_indices.append(this_level_indices)
    
    return level_indices


def rvq_compress_grouped(codec_list, features, device, norm_mode='per_token_ln',
                          mu_bits=16, std_bits=16, embedding_dim=32, verbose=True,
                          normalize_fn=None, rvq_encode_decode_fn=None, 
                          compute_side_info_bytes_fn=None):
    """
    RVQ 分组静态熵编码压缩
    """
    import time
    
    num_images = len(features)
    num_tokens = features[0].shape[0]
    feat_dim = features[0].shape[1]
    num_groups = feat_dim // embedding_dim
    
    if verbose:
        print(f"  [分组熵编码] num_images={num_images}, num_tokens={num_tokens}, "
              f"num_groups={num_groups}")
    
    level_indices = collect_rvq_indices_structured(
        codec_list, features, device, norm_mode, embedding_dim, normalize_fn
    )
    
    entropy_models = []
    all_entropy_stats = []
    total_vq_bytes = 0
    encode_time = 0.0
    
    for level, (codec, indices_list) in enumerate(zip(codec_list, level_indices)):
        if codec is None or indices_list is None:
            entropy_models.append(None)
            all_entropy_stats.append(None)
            continue
        
        K = codec.num_embeddings
        entropy_model = GroupConditionedEntropyModel(K=K, num_groups=num_groups, num_tokens=num_tokens)
        entropy_model.update_counts(indices_list)
        entropy_stats = entropy_model.compute_entropy()
        all_entropy_stats.append(entropy_stats)
        
        if verbose:
            print(f"  Level {level+1} (K={K}):")
            print(f"    CLS 熵: grouped={entropy_stats['cls_avg_entropy']:.4f}, "
                  f"global={entropy_stats['cls_global_entropy']:.4f} bits/index")
            print(f"    Patch 熵: grouped={entropy_stats['patch_avg_entropy']:.4f}, "
                  f"global={entropy_stats['patch_global_entropy']:.4f} bits/index")
            print(f"    log2(K)={math.log2(K):.4f} bits/index")
        
        t0 = time.perf_counter()
        level_bytes = 0
        for indices_flat in indices_list:
            cls_string, patch_string = entropy_model.compress(indices_flat)
            level_bytes += len(cls_string) + len(patch_string)
        encode_time += time.perf_counter() - t0
        
        total_vq_bytes += level_bytes
        entropy_models.append(entropy_model)
        
        if verbose:
            total_indices = num_images * num_tokens * num_groups
            actual_bpi = level_bytes * 8 / total_indices
            print(f"    实际码率: {actual_bpi:.4f} bits/index")
    
    xhat_list = rvq_encode_decode_fn(codec_list, features, device, norm_mode)
    side_info_bytes = compute_side_info_bytes_fn(num_images, num_tokens, norm_mode, mu_bits, std_bits)
    
    return xhat_list, total_vq_bytes, side_info_bytes, all_entropy_stats, encode_time


def rvq_compress_position(codec_list, features, device, norm_mode='per_token_ln',
                           mu_bits=16, std_bits=16, embedding_dim=32, verbose=True,
                           normalize_fn=None, rvq_encode_decode_fn=None,
                           compute_side_info_bytes_fn=None):
    """
    RVQ 位置条件化熵编码压缩
    """
    import time
    
    num_images = len(features)
    num_tokens = features[0].shape[0]
    feat_dim = features[0].shape[1]
    num_groups = feat_dim // embedding_dim
    grid_size = int(math.sqrt(num_tokens - 1))
    
    if verbose:
        print(f"  [位置条件化熵编码] num_images={num_images}, num_tokens={num_tokens}, "
              f"num_groups={num_groups}, grid={grid_size}×{grid_size}")
        print(f"    频率表数量: CLS={num_groups}, Patch={grid_size**2 * num_groups}")
    
    level_indices = collect_rvq_indices_structured(
        codec_list, features, device, norm_mode, embedding_dim, normalize_fn
    )
    
    entropy_models = []
    all_entropy_stats = []
    total_vq_bytes = 0
    encode_time = 0.0
    
    for level, (codec, indices_list) in enumerate(zip(codec_list, level_indices)):
        if codec is None or indices_list is None:
            entropy_models.append(None)
            all_entropy_stats.append(None)
            continue
        
        K = codec.num_embeddings
        entropy_model = PositionConditionedEntropyModel(
            K=K, num_groups=num_groups, num_tokens=num_tokens, grid_size=grid_size
        )
        entropy_model.update_counts(indices_list)
        entropy_stats = entropy_model.compute_entropy()
        all_entropy_stats.append(entropy_stats)
        
        if verbose:
            print(f"  Level {level+1} (K={K}):")
            print(f"    CLS 熵: {entropy_stats['cls_avg_entropy']:.4f} bits/index")
            print(f"    Patch 熵: position={entropy_stats['patch_avg_entropy_position']:.4f}, "
                  f"group={entropy_stats['patch_avg_entropy_group']:.4f}, "
                  f"global={entropy_stats['patch_global_entropy']:.4f} bits/index")
            print(f"    位置条件化收益: {entropy_stats['position_gain']:.4f} bits/index")
            print(f"    log2(K)={math.log2(K):.4f} bits/index")
        
        t0 = time.perf_counter()
        level_bytes = 0
        for indices_flat in indices_list:
            cls_string, patch_string = entropy_model.compress(indices_flat)
            level_bytes += len(cls_string) + len(patch_string)
        encode_time += time.perf_counter() - t0
        
        total_vq_bytes += level_bytes
        entropy_models.append(entropy_model)
        
        if verbose:
            total_indices = num_images * num_tokens * num_groups
            actual_bpi = level_bytes * 8 / total_indices
            print(f"    实际码率: {actual_bpi:.4f} bits/index")
    
    xhat_list = rvq_encode_decode_fn(codec_list, features, device, norm_mode)
    side_info_bytes = compute_side_info_bytes_fn(num_images, num_tokens, norm_mode, mu_bits, std_bits)
    
    return xhat_list, total_vq_bytes, side_info_bytes, all_entropy_stats, encode_time


def full_encode_decode_pipeline(codec_list, features, device, norm_mode='per_token_ln',
                                 embedding_dim=32, entropy_mode='global', calibrator=None,
                                 verbose=False, normalize_fn=None, inv_normalize_fn=None,
                                 compute_side_info_bytes_fn=None):
    """
    完整的端到端编解码流程，测量全流程时间
    """
    import time
    
    num_images = len(features)
    num_tokens = features[0].shape[0]
    feat_dim = features[0].shape[1]
    num_groups = feat_dim // embedding_dim
    grid_size = int(math.sqrt(num_tokens - 1))
    
    timing = {
        'encode_total': 0.0, 'decode_total': 0.0, 'normalize': 0.0,
        'vq_encode': 0.0, 'entropy_encode': 0.0, 'entropy_decode': 0.0,
        'vq_decode': 0.0, 'inv_normalize': 0.0, 'calibrator': 0.0,
    }
    
    # 编码端
    t_encode_start = time.perf_counter()
    
    t0 = time.perf_counter()
    y_list, mu_list, std_list = [], [], []
    for x in features:
        y, mu, std = normalize_fn(x, mode=norm_mode)
        y_list.append(y)
        mu_list.append(mu)
        std_list.append(std)
    timing['normalize'] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    y_hat_list = [np.zeros_like(y) for y in y_list]
    all_level_indices = []
    
    for level, codec in enumerate(codec_list):
        if codec is None:
            all_level_indices.append(None)
            continue
        
        r_list = [y - y_hat for y, y_hat in zip(y_list, y_hat_list)]
        level_indices = []
        
        for i, r in enumerate(r_list):
            r_t = torch.from_numpy(r.T).float().to(device)
            with torch.no_grad():
                outputs = codec(r_t.unsqueeze(0))
                r_hat_t = outputs[0].squeeze(0)
                encoding_inds = outputs[4]
            
            indices_flat = np.concatenate([ind.cpu().numpy().flatten() for ind in encoding_inds])
            level_indices.append(indices_flat)
            y_hat_list[i] = y_hat_list[i] + r_hat_t.T.cpu().numpy()
        
        all_level_indices.append(level_indices)
    timing['vq_encode'] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    all_bitstreams = []
    entropy_models = []
    
    for level, (codec, indices_list) in enumerate(zip(codec_list, all_level_indices)):
        if codec is None or indices_list is None:
            all_bitstreams.append(None)
            entropy_models.append(None)
            continue
        
        K = codec.num_embeddings
        if entropy_mode == 'position':
            entropy_model = PositionConditionedEntropyModel(K=K, num_groups=num_groups, num_tokens=num_tokens, grid_size=grid_size)
        else:
            entropy_model = GroupConditionedEntropyModel(K=K, num_groups=num_groups, num_tokens=num_tokens)
        
        entropy_model.update_counts(indices_list)
        entropy_models.append(entropy_model)
        
        level_bitstreams = []
        for indices_flat in indices_list:
            cls_string, patch_string = entropy_model.compress(indices_flat)
            level_bitstreams.append((cls_string, patch_string))
        all_bitstreams.append(level_bitstreams)
    
    timing['entropy_encode'] = time.perf_counter() - t0
    timing['encode_total'] = time.perf_counter() - t_encode_start
    
    # 解码端
    t_decode_start = time.perf_counter()
    
    t0 = time.perf_counter()
    decoded_indices = []
    for level, (bitstreams, entropy_model) in enumerate(zip(all_bitstreams, entropy_models)):
        if bitstreams is None or entropy_model is None:
            decoded_indices.append(None)
            continue
        level_decoded = []
        for cls_string, patch_string in bitstreams:
            indices_flat = entropy_model.decompress(cls_string, patch_string)
            level_decoded.append(indices_flat)
        decoded_indices.append(level_decoded)
    timing['entropy_decode'] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    y_hat_decoded = [np.zeros_like(y) for y in y_list]
    for level, (codec, indices_list) in enumerate(zip(codec_list, decoded_indices)):
        if codec is None or indices_list is None:
            continue
        codebook_np = codec.get_codebook()
        codebook_tensor = torch.from_numpy(codebook_np).float().to(device)
        for i, indices_flat in enumerate(indices_list):
            indices_2d = indices_flat.reshape(num_tokens, num_groups)
            with torch.no_grad():
                for g in range(num_groups):
                    group_indices = torch.from_numpy(indices_2d[:, g].astype(np.int64)).long().to(device)
                    quantized = codebook_tensor[group_indices]
                    y_hat_decoded[i][:, g*embedding_dim:(g+1)*embedding_dim] += quantized.cpu().numpy()
    timing['vq_decode'] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    xhat_list = [inv_normalize_fn(y_hat, mu, std, mode=norm_mode) for y_hat, mu, std in zip(y_hat_decoded, mu_list, std_list)]
    timing['inv_normalize'] = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    if calibrator is not None:
        calibrator.eval()
        xhat_calibrated = []
        with torch.no_grad():
            for xh in xhat_list:
                xh_t = torch.from_numpy(xh).float().unsqueeze(0).to(device)
                xh_cal = calibrator(xh_t)
                xhat_calibrated.append(xh_cal.squeeze(0).cpu().numpy().astype(np.float32))
        xhat_list = xhat_calibrated
    timing['calibrator'] = time.perf_counter() - t0
    
    timing['decode_total'] = time.perf_counter() - t_decode_start
    
    total_bytes = 0
    for level_bitstreams in all_bitstreams:
        if level_bitstreams is None:
            continue
        for cls_string, patch_string in level_bitstreams:
            total_bytes += len(cls_string) + len(patch_string)
    
    side_info_bytes = compute_side_info_bytes_fn(num_images, num_tokens, norm_mode, 16, 16)
    total_bytes += side_info_bytes
    
    if verbose:
        print(f"  [全流程时间统计] {num_images} 张图")
        print(f"    编码端: {timing['encode_total']*1000:.1f}ms "
              f"(归一化={timing['normalize']*1000:.1f}, VQ={timing['vq_encode']*1000:.1f}, "
              f"熵编码={timing['entropy_encode']*1000:.1f})")
        print(f"    解码端: {timing['decode_total']*1000:.1f}ms "
              f"(熵解码={timing['entropy_decode']*1000:.1f}, VQ={timing['vq_decode']*1000:.1f}, "
              f"反归一化={timing['inv_normalize']*1000:.1f}, 校准={timing['calibrator']*1000:.1f})")
        print(f"    平均每图: 编码={timing['encode_total']/num_images*1000:.2f}ms, "
              f"解码={timing['decode_total']/num_images*1000:.2f}ms")
    
    return xhat_list, total_bytes, timing
