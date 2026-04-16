"""
SimpleFCVQ: 量化与熵编码分离的 VQ 模型

设计原则：
1. 量化 (forward/quantize): 纯最近邻，argmin(MSE)
2. 熵编码 (compress): 基于 prior 的熵编码，不影响量化决策
3. prior 更新 (update_prior): 只影响熵编码效率，不影响量化结果

这样设计的好处：
- 量化最小化失真 → 最高下游任务准确率
- 熵编码最小化码率 → 最低 BPFP_entropy
- 两者独立，可分别优化

与原 FCVQ 的区别：
- 原 FCVQ: forward() 和 compress() 都使用 rate_bias = log2_pmf / lmbda
  - 量化决策受 lmbda 和 prior 影响
  - update_prior 会改变量化结果，可能降低准确率
- SimpleFCVQ: 量化不使用 rate_bias
  - 量化决策仅基于 MSE（纯最近邻）
  - update_prior 只影响熵编码，不影响量化结果

使用方法：
```python
# 方法 1: 直接替换 (使用兼容接口)
from simple_fcvq import create_vq_codec

# 替换: from coding.vq.v1.fcvq_model import FCVQ
# 原来: codec = FCVQ(K, D, num_chunks=1, lmbda=1.0).to(device)
# 现在:
codec = create_vq_codec(K, D, codebook, device)

# 使用方式不变
x_hat, mse, rd, rate, indices = codec(x)
x_hat, mse, strings, indices = codec.compress(x)

# 方法 2: 使用新接口
from simple_fcvq import create_simple_fcvq

model = create_simple_fcvq(K, D, codebook, device)
result = model.forward(x)  # 返回 dict
x_hat, indices = model.quantize(x)  # 简洁接口
model.update_prior(indices)  # 只影响熵编码
```
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

# 熵编码依赖
try:
    from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
    from compressai import ans
    HAS_COMPRESSAI = True
except (ImportError, ModuleNotFoundError):
    _pmf_to_quantized_cdf = None
    ans = None
    HAS_COMPRESSAI = False


def pmf_to_quantized_cdf(pmf: torch.Tensor, precision: int = 16) -> torch.Tensor:
    """将 PMF 转换为量化 CDF"""
    if _pmf_to_quantized_cdf is None:
        raise RuntimeError("compressai not installed")
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    return torch.IntTensor(cdf)


class SimpleEntropyModel(nn.Module):
    """
    简化的熵编码模型
    
    只负责：
    1. 管理 prior (logits → PMF)
    2. 熵编码/解码索引
    3. 计算熵 (bits)
    """
    
    def __init__(self, num_embeddings: int, range_coder_precision: int = 16):
        super().__init__()
        self.K = num_embeddings
        self.precision = range_coder_precision
        
        # Prior: 均匀分布初始化
        self.register_buffer("logits", torch.zeros(1, num_embeddings))
        
        # 熵编码器/解码器
        if HAS_COMPRESSAI:
            self._encoder = ans.RansEncoder()
            self._decoder = ans.RansDecoder()
        else:
            self._encoder = None
            self._decoder = None
        
        # CDF 表 (延迟初始化)
        self._cdf_ready = False
        self.cdf = None
        self.cdf_length = None
        self.cdf_offset = None
    
    def update_prior(self, indices: np.ndarray, smooth_eps: float = 1e-6):
        """
        根据实际使用频率更新 prior
        
        Args:
            indices: 码字索引 (任意形状，会被 flatten)
            smooth_eps: 平滑系数，防止零概率
        """
        indices = indices.flatten().astype(int)
        counts = np.bincount(indices, minlength=self.K)
        total = counts.sum()
        
        # 平滑概率
        probs = (counts + smooth_eps) / (total + smooth_eps * self.K)
        log_probs = np.log(probs + 1e-10)
        
        # 更新 logits
        self.logits.data = torch.from_numpy(log_probs).float().unsqueeze(0).to(self.logits.device)
        
        # 标记需要重新构建 CDF 表
        self._cdf_ready = False
        
        # 返回熵
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return entropy
    
    def get_pmf(self) -> torch.Tensor:
        """获取当前 PMF"""
        return F.softmax(self.logits, dim=-1)
    
    def get_log_pmf(self) -> torch.Tensor:
        """获取当前 log PMF"""
        return F.log_softmax(self.logits, dim=-1)
    
    def get_entropy(self) -> float:
        """计算当前 prior 的熵 (bits)"""
        pmf = self.get_pmf().squeeze()
        entropy = -torch.sum(pmf * torch.log2(pmf + 1e-10))
        return entropy.item()
    
    def compute_bits(self, indices: torch.Tensor) -> torch.Tensor:
        """
        计算给定索引的理论比特数
        
        Args:
            indices: 码字索引 [N]
        
        Returns:
            总比特数
        """
        log_pmf = self.get_log_pmf().squeeze()  # [K]
        log_probs = log_pmf[indices.flatten()]  # [N]
        bits = -torch.sum(log_probs) / math.log(2)
        return bits
    
    def _build_cdf_tables(self):
        """构建熵编码所需的 CDF 表"""
        if not HAS_COMPRESSAI:
            raise RuntimeError("compressai not installed, cannot compress")
        
        pmf = self.get_pmf()  # [1, K]
        
        # 构建 CDF
        p = pmf.squeeze().cpu()
        overflow = (1.0 - p.sum()).clamp_min(0)
        p = torch.cat([p, overflow.unsqueeze(0)], dim=0)
        cdf = pmf_to_quantized_cdf(p, self.precision)
        
        # compressai 需要的格式: cdf 是嵌套列表
        self.cdf = [cdf.int().tolist()]
        self.cdf_length = [self.K + 2]
        self.cdf_offset = [0]
        self._cdf_ready = True
    
    def compress(self, indices: torch.Tensor) -> bytes:
        """
        熵编码压缩索引
        
        Args:
            indices: 码字索引 [N, 1] 或 [N]
        
        Returns:
            压缩后的字节串
        """
        if not self._cdf_ready:
            self._build_cdf_tables()
        
        indices = indices.flatten().int().tolist()
        cdf_indices = [0] * len(indices)  # 所有索引使用同一个 CDF
        
        string = self._encoder.encode_with_indexes(
            indices, cdf_indices, self.cdf, self.cdf_length, self.cdf_offset
        )
        return string
    
    def decompress(self, string: bytes, num_symbols: int) -> torch.Tensor:
        """
        熵解码
        
        Args:
            string: 压缩的字节串
            num_symbols: 符号数量
        
        Returns:
            解码后的索引 [num_symbols]
        """
        if not self._cdf_ready:
            self._build_cdf_tables()
        
        cdf_indices = [0] * num_symbols
        values = self._decoder.decode_with_indexes(
            string, cdf_indices, self.cdf, self.cdf_length, self.cdf_offset
        )
        return torch.tensor(values, dtype=torch.int64)


class SimpleVQ(nn.Module):
    """
    简化的 Vector Quantizer
    
    只负责：
    1. 管理码本 (embedding)
    2. 纯最近邻量化 (无 rate_bias)
    3. 查表重建
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        
        # 码本
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.5)
    
    def set_codebook(self, codebook: np.ndarray):
        """设置码本"""
        cb_tensor = torch.from_numpy(codebook).float()
        self.embedding.weight.data.copy_(cb_tensor)
    
    def get_codebook(self) -> np.ndarray:
        """获取码本"""
        return self.embedding.weight.detach().cpu().numpy()
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        纯最近邻量化
        
        Args:
            x: 输入特征 [N, D] 或 [B, N, D]
        
        Returns:
            x_hat: 量化后的特征 [N, D] 或 [B, N, D]
            indices: 码字索引 [N] 或 [B, N]
        """
        original_shape = x.shape
        if len(x.shape) == 3:
            B, N, D = x.shape
            x = x.view(B * N, D)
        else:
            B = None
            N, D = x.shape
        
        # 计算距离矩阵 (纯 MSE，无 rate_bias)
        codebook = self.embedding.weight  # [K, D]
        
        # dist[i,j] = ||x[i] - c[j]||^2
        dist = (
            torch.sum(x ** 2, dim=1, keepdim=True)  # [N, 1]
            + torch.sum(codebook ** 2, dim=1)  # [K]
            - 2 * torch.matmul(x, codebook.t())  # [N, K]
        )
        
        # 最近邻
        indices = torch.argmin(dist, dim=1)  # [N]
        
        # 查表重建
        x_hat = self.embedding(indices)  # [N, D]
        
        # 恢复形状
        if B is not None:
            x_hat = x_hat.view(B, N // B if N % B == 0 else -1, D)
            indices = indices.view(B, -1)
        
        return x_hat, indices
    
    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """
        根据索引查表
        
        Args:
            indices: 码字索引 [N] 或 [B, N]
        
        Returns:
            x_hat: 量化后的特征 [N, D] 或 [B, N, D]
        """
        return self.embedding(indices)


class SimpleFCVQ(nn.Module):
    """
    简化的 Feature Compression VQ
    
    核心设计：量化与熵编码分离
    - quantize(): 纯最近邻量化
    - forward(): 量化 + 返回统计信息
    - compress(): 量化 + 熵编码
    - decompress(): 熵解码 + 查表重建
    - update_prior(): 更新熵编码 prior
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_chunks: int = 1,
    ):
        super().__init__()
        
        self.K = num_embeddings
        self.D = embedding_dim
        self.num_chunks = num_chunks
        
        # VQ 模块 (每个 chunk 一个)
        self.vq_modules = nn.ModuleList([
            SimpleVQ(num_embeddings, embedding_dim)
            for _ in range(num_chunks)
        ])
        
        # 熵模型 (每个 chunk 共享)
        self.entropy_model = SimpleEntropyModel(num_embeddings)
    
    def set_codebook(self, codebook: np.ndarray, chunk_idx: int = 0):
        """设置指定 chunk 的码本"""
        self.vq_modules[chunk_idx].set_codebook(codebook)
    
    def get_codebook(self, chunk_idx: int = 0) -> np.ndarray:
        """获取指定 chunk 的码本"""
        return self.vq_modules[chunk_idx].get_codebook()
    
    def update_prior(self, indices: np.ndarray, smooth_eps: float = 1e-6) -> float:
        """
        更新熵编码的 prior
        
        注意：这只影响熵编码效率，不影响量化结果
        
        Args:
            indices: 码字索引
            smooth_eps: 平滑系数
        
        Returns:
            更新后的 prior 熵 (bits)
        """
        return self.entropy_model.update_prior(indices, smooth_eps)
    
    def _reshape_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, tuple, bool]:
        """
        统一输入形状处理
        
        支持的输入格式：
        - [H, W]: 单个特征图
        - [N, H, W]: batch 特征图
        - [N, C, H, W]: batch 多通道特征图
        
        Returns:
            x: 重塑后的输入 [N, H, W]
            original_shape: 原始形状
            was_2d: 是否为 2D 输入
        """
        original_shape = x.shape
        was_2d = False
        
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            was_2d = True
        elif len(x.shape) == 4:
            N, C, H, W = x.shape
            x = x.view(N * C, H, W)
        
        return x, original_shape, was_2d
    
    def _reshape_output(self, x: torch.Tensor, original_shape: tuple, was_2d: bool) -> torch.Tensor:
        """恢复输出形状"""
        if was_2d:
            x = x.squeeze(0)
        elif len(original_shape) == 4:
            N, C, H, W = original_shape
            x = x.view(N, C, H, W)
        return x
    
    def _process_chunk(self, chunk: torch.Tensor, chunk_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理单个 chunk
        
        输入形状处理：
        - chunk: [N, H, W] 其中 H 是特征维度，W 是 token 数量
        - 需要 reshape 为 [N*W, H/D, D] 用于量化
        """
        N, H, W = chunk.shape
        
        # 处理 padding (如果 H 不是 D 的倍数)
        target_rows = H % self.D
        if target_rows != 0:
            pad_len = self.D - target_rows
            last_cols = chunk[:, -pad_len:, :]
            chunk = torch.cat([chunk, last_cols], dim=1)
        
        H_padded = chunk.shape[1]
        
        # Reshape: [N, H, W] -> [N*W*(H/D), D]
        # 先 permute: [N, H, W] -> [N, W, H]
        chunk = chunk.permute(0, 2, 1).contiguous()
        # 再 view: [N, W, H] -> [N, W, H/D, D] -> [N*W*H/D, D]
        chunk = chunk.view(N, W, H_padded // self.D, self.D)
        chunk = chunk.view(N * W * H_padded // self.D, self.D)
        
        # 量化
        vq = self.vq_modules[chunk_idx]
        x_hat, indices = vq.quantize(chunk)
        
        # 恢复形状
        x_hat = x_hat.view(N, W, H_padded // self.D, self.D)
        x_hat = x_hat.view(N, W, H_padded)
        x_hat = x_hat.permute(0, 2, 1).contiguous()
        
        # 去除 padding
        x_hat = x_hat[:, :H, :]
        
        return x_hat, indices
    
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        前向传播：纯最近邻量化
        
        Args:
            x: 输入特征 [H, W] 或 [N, H, W] 或 [N, C, H, W]
        
        Returns:
            dict with:
                - x_hat: 量化后的特征 (形状与输入相同)
                - indices: 码字索引列表 (每个 chunk 一个)
                - mse_loss: MSE 损失
                - bits: 理论比特数 (基于当前 prior)
        """
        x, original_shape, was_2d = self._reshape_input(x)
        
        # 分 chunk
        chunks = torch.chunk(x, self.num_chunks, dim=2)
        
        quantized_chunks = []
        all_indices = []
        total_mse = 0.0
        total_bits = 0.0
        
        for i, chunk in enumerate(chunks):
            x_hat, indices = self._process_chunk(chunk, i)
            quantized_chunks.append(x_hat)
            all_indices.append(indices)
            
            # 计算 MSE
            mse = F.mse_loss(x_hat, chunk)
            total_mse += mse.item()
            
            # 计算理论比特数
            bits = self.entropy_model.compute_bits(indices)
            total_bits += bits.item()
        
        # 合并 chunks
        x_hat = torch.cat(quantized_chunks, dim=2)
        x_hat = self._reshape_output(x_hat, original_shape, was_2d)
        
        return {
            "x_hat": x_hat,
            "indices": all_indices,
            "mse_loss": total_mse / self.num_chunks,
            "bits": total_bits,
        }
    
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        简化接口：只返回量化结果和索引
        
        Args:
            x: 输入特征
        
        Returns:
            x_hat: 量化后的特征
            indices: 码字索引列表
        """
        result = self.forward(x)
        return result["x_hat"], result["indices"]
    
    def compress(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        压缩：量化 + 熵编码
        
        Args:
            x: 输入特征
        
        Returns:
            dict with:
                - x_hat: 量化后的特征
                - strings: 压缩后的字节串列表
                - indices: 码字索引列表
                - num_bytes: 总字节数
                - bpp: bits per point (每个量化点的比特数)
        """
        result = self.forward(x)
        
        strings = []
        total_bytes = 0
        total_points = 0
        
        for indices in result["indices"]:
            string = self.entropy_model.compress(indices)
            strings.append(string)
            total_bytes += len(string)
            total_points += indices.numel()
        
        return {
            "x_hat": result["x_hat"],
            "strings": strings,
            "indices": result["indices"],
            "num_bytes": total_bytes,
            "bpp": total_bytes * 8 / total_points if total_points > 0 else 0,
        }
    
    def decompress(self, strings: List[bytes], shape: tuple) -> torch.Tensor:
        """
        解压：熵解码 + 查表重建
        
        Args:
            strings: 压缩的字节串列表 (每个 chunk 一个)
            shape: 输出特征的形状 [H, W] 或 [N, H, W] 或 [N, C, H, W]
        
        Returns:
            x_hat: 重建的特征
        """
        # 计算每个 chunk 的符号数
        if len(shape) == 2:
            H, W = shape
            N = 1
        elif len(shape) == 3:
            N, H, W = shape
        else:
            N, C, H, W = shape
            N = N * C
        
        # padding
        target_rows = H % self.D
        if target_rows != 0:
            pad_len = self.D - target_rows
            H_padded = H + pad_len
        else:
            H_padded = H
        
        W_per_chunk = W // self.num_chunks
        num_symbols_per_chunk = N * W_per_chunk * H_padded // self.D
        
        quantized_chunks = []
        device = next(self.parameters()).device
        
        for i, string in enumerate(strings):
            # 熵解码
            indices = self.entropy_model.decompress(string, num_symbols_per_chunk)
            indices = indices.to(device)
            
            # 查表重建
            vq = self.vq_modules[i]
            x_hat = vq.lookup(indices)  # [num_symbols, D]
            
            # 恢复形状
            x_hat = x_hat.view(N, W_per_chunk, H_padded // self.D, self.D)
            x_hat = x_hat.view(N, W_per_chunk, H_padded)
            x_hat = x_hat.permute(0, 2, 1).contiguous()
            
            # 去除 padding
            x_hat = x_hat[:, :H, :]
            
            quantized_chunks.append(x_hat)
        
        # 合并
        x_hat = torch.cat(quantized_chunks, dim=2)
        
        # 恢复原始形状
        if len(shape) == 2:
            x_hat = x_hat.squeeze(0)
        elif len(shape) == 4:
            x_hat = x_hat.view(*shape)
        
        return x_hat


# ============== 兼容旧 FCVQ 接口的包装类 ==============

class SimpleFCVQCompat(SimpleFCVQ):
    """
    兼容旧 FCVQ 接口的 SimpleFCVQ
    
    保持与原 FCVQ 相同的调用方式：
    - codec(batch) 返回 (x_hat, mse_loss, rd_loss, rate, encoding_inds)
    - codec.compress(feat) 返回 (x_hat, mse_loss, strings, encoding_inds)
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, num_chunks: int = 1, **kwargs):
        # 忽略 lmbda 参数，因为我们不使用 rate_bias
        super().__init__(num_embeddings, embedding_dim, num_chunks)
        # 添加兼容属性
        self.logits = self.entropy_model.logits
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor) -> List[Any]:
        """
        兼容 FCVQ 的 forward 接口
        
        Returns:
            [x_hat, mse_loss, rd_loss, rate, encoding_inds]
        """
        x, original_shape, was_2d = self._reshape_input(x)
        
        # 分 chunk
        chunks = torch.chunk(x, self.num_chunks, dim=2)
        
        quantized_chunks = []
        all_indices = []
        total_mse = 0.0
        total_bits = 0.0
        
        for i, chunk in enumerate(chunks):
            x_hat_chunk, indices = self._process_chunk(chunk, i)
            quantized_chunks.append(x_hat_chunk)
            all_indices.append(indices)
            
            # 计算 MSE
            mse = F.mse_loss(x_hat_chunk, chunk)
            total_mse += mse
            
            # 计算理论比特数
            bits = self.entropy_model.compute_bits(indices)
            total_bits += bits
        
        # 合并 chunks
        x_hat = torch.cat(quantized_chunks, dim=2)
        x_hat = self._reshape_output(x_hat, original_shape, was_2d)
        
        mse_loss = total_mse / self.num_chunks
        rate = total_bits / x.numel()
        rd_loss = rate + mse_loss  # 简化的 RD loss
        
        return [x_hat, mse_loss, rd_loss, rate, all_indices]
    
    def compress(self, x: torch.Tensor) -> Tuple[torch.Tensor, float, List[bytes], List[torch.Tensor]]:
        """
        兼容 FCVQ 的 compress 接口
        
        Returns:
            (x_hat, mse_loss, strings, encoding_inds)
        """
        x, original_shape, was_2d = self._reshape_input(x)
        
        # 分 chunk
        chunks = torch.chunk(x, self.num_chunks, dim=2)
        
        quantized_chunks = []
        all_indices = []
        all_strings = []
        total_mse = 0.0
        
        for i, chunk in enumerate(chunks):
            x_hat_chunk, indices = self._process_chunk(chunk, i)
            quantized_chunks.append(x_hat_chunk)
            all_indices.append(indices)
            
            # 计算 MSE
            mse = F.mse_loss(x_hat_chunk, chunk)
            total_mse += mse
            
            # 熵编码
            string = self.entropy_model.compress(indices)
            all_strings.append(string)
        
        # 合并 chunks
        x_hat = torch.cat(quantized_chunks, dim=2)
        x_hat = self._reshape_output(x_hat, original_shape, was_2d)
        
        mse_loss = total_mse / self.num_chunks
        
        return x_hat, mse_loss.item(), all_strings, all_indices


# ============== 工厂函数 ==============

def create_simple_fcvq(
    num_embeddings: int,
    embedding_dim: int,
    codebook: Optional[np.ndarray] = None,
    device: str = "cuda",
    compat_mode: bool = False,
) -> SimpleFCVQ:
    """
    创建 SimpleFCVQ 实例
    
    Args:
        num_embeddings: 码本大小 K
        embedding_dim: 向量维度 D
        codebook: 可选的预训练码本 [K, D]
        device: 设备
        compat_mode: 是否使用兼容模式 (兼容旧 FCVQ 接口)
    
    Returns:
        SimpleFCVQ 实例
    """
    if compat_mode:
        model = SimpleFCVQCompat(num_embeddings, embedding_dim, num_chunks=1).to(device)
    else:
        model = SimpleFCVQ(num_embeddings, embedding_dim, num_chunks=1).to(device)
    
    if codebook is not None:
        model.set_codebook(codebook)
    
    model.eval()
    return model


def create_vq_codec(K: int, embedding_dim: int, codebook: np.ndarray, 
                    device: str, lmbda: float = 1.0) -> SimpleFCVQCompat:
    """
    兼容旧 create_vq_codec 接口
    
    注意：lmbda 参数被忽略，因为 SimpleFCVQ 不使用 rate_bias
    
    Args:
        K: 码本大小
        embedding_dim: 向量维度
        codebook: 预训练码本 [K, D]
        device: 设备
        lmbda: 忽略 (保留用于接口兼容)
    
    Returns:
        SimpleFCVQCompat 实例
    """
    return create_simple_fcvq(K, embedding_dim, codebook, device, compat_mode=True)


# ============== 兼容接口 ==============

def simple_vq_forward_batch(
    codec: SimpleFCVQ,
    feat_batch: List[np.ndarray],
    device: str = "cuda",
) -> List[np.ndarray]:
    """
    兼容 run_minimal_ablation.py 的 vq_forward_batch 接口
    
    Args:
        codec: SimpleFCVQ 实例
        feat_batch: 特征列表，每个形状 [D, N_tokens]
        device: 设备
    
    Returns:
        量化后的特征列表
    """
    # 转换为 tensor: [B, D, N] -> [B, D, N] (不需要转置，因为输入已经是 [D, N])
    batch_tensors = [torch.from_numpy(f.T).float() for f in feat_batch]  # [N, D] each
    # stack: [B, N, D] -> permute -> [B, D, N]
    batch = torch.stack(batch_tensors, dim=0).permute(0, 2, 1).to(device)  # [B, D, N]
    
    with torch.no_grad():
        x_hat, _ = codec.quantize(batch)
    
    # [B, D, N] -> [B, N, D] -> list of [D, N]
    x_hat = x_hat.permute(0, 2, 1)  # [B, N, D]
    return [x_hat[i].T.cpu().numpy() for i in range(x_hat.shape[0])]


def simple_vq_compress_batch(
    codec: SimpleFCVQ,
    feat_batch: List[np.ndarray],
    device: str = "cuda",
) -> Tuple[List[np.ndarray], int]:
    """
    压缩特征并返回字节数
    
    Args:
        codec: SimpleFCVQ 实例
        feat_batch: 特征列表
        device: 设备
    
    Returns:
        (量化后的特征列表, 总字节数)
    """
    batch_tensors = [torch.from_numpy(f.T).float() for f in feat_batch]
    batch = torch.stack(batch_tensors, dim=0).permute(0, 2, 1).to(device)
    
    with torch.no_grad():
        result = codec.compress(batch)
    
    x_hat = result["x_hat"].permute(0, 2, 1)
    feat_hat_list = [x_hat[i].T.cpu().numpy() for i in range(x_hat.shape[0])]
    
    return feat_hat_list, result["num_bytes"]


if __name__ == "__main__":
    # 简单测试
    print("=" * 60)
    print("Testing SimpleFCVQ (新接口)")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建模型
    K, D = 1024, 32
    model = create_simple_fcvq(K, D, device=device)
    
    # 随机输入
    x = torch.randn(4, 256, 128).to(device)  # [B, H, W]
    
    # 量化
    result = model.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {result['x_hat'].shape}")
    print(f"MSE loss: {result['mse_loss']:.6f}")
    print(f"Bits: {result['bits']:.2f}")
    
    # 更新 prior
    all_indices = torch.cat([idx.flatten() for idx in result["indices"]]).cpu().numpy()
    entropy = model.update_prior(all_indices)
    print(f"Prior entropy after update: {entropy:.4f} bits")
    
    # 压缩
    if HAS_COMPRESSAI:
        compress_result = model.compress(x)
        print(f"Compressed bytes: {compress_result['num_bytes']}")
        print(f"BPP: {compress_result['bpp']:.4f}")
        
        # 解压
        x_rec = model.decompress(compress_result["strings"], x.shape)
        print(f"Reconstructed shape: {x_rec.shape}")
        
        # 验证无损
        assert torch.allclose(result["x_hat"], x_rec), "Compression/decompression mismatch!"
        print("Compression/decompression verified!")
    else:
        print("compressai not installed, skipping compression test")
    
    print("\n" + "=" * 60)
    print("Testing SimpleFCVQCompat (兼容接口)")
    print("=" * 60)
    
    # 测试兼容模式
    codebook = np.random.randn(K, D).astype(np.float32)
    codec = create_vq_codec(K, D, codebook, device)
    
    # 测试 forward (模拟 vq_forward_batch)
    batch_tensors = [torch.randn(D, 128).float() for _ in range(4)]
    batch = torch.stack(batch_tensors, dim=0).to(device)  # [B, D, W]
    
    with torch.no_grad():
        outputs = codec(batch)
        x_hat = outputs[0]
        encoding_inds = outputs[4]
    
    print(f"Compat forward - Input: {batch.shape}, Output: {x_hat.shape}")
    print(f"Compat forward - MSE: {outputs[1].item():.6f}")
    print(f"Compat forward - Indices: {len(encoding_inds)} chunks")
    
    # 测试 compress (模拟 vq_compress)
    feat = torch.randn(D, 128).float().to(device)  # [D, W]
    
    with torch.no_grad():
        quantized, mse_loss, strings, encoding_inds = codec.compress(feat)
    
    print(f"Compat compress - Input: {feat.shape}, Output: {quantized.shape}")
    print(f"Compat compress - MSE: {mse_loss:.6f}")
    print(f"Compat compress - Bytes: {sum(len(s) for s in strings)}")
    
    # 测试 update_prior
    all_idx = np.random.randint(0, K, size=10000)
    entropy = codec.update_prior(all_idx)
    print(f"Compat update_prior - Entropy: {entropy:.4f} bits")
    
    print("\n" + "=" * 60)
    print("验证量化结果不受 lmbda 影响（核心设计）")
    print("=" * 60)
    
    # 创建相同码本但不同 lmbda 的 codec
    fixed_codebook = np.random.randn(K, D).astype(np.float32)
    codec1 = create_vq_codec(K, D, fixed_codebook, device, lmbda=0.001)
    codec2 = create_vq_codec(K, D, fixed_codebook, device, lmbda=1000.0)
    
    test_input = torch.randn(D, 128).float().to(device)
    
    with torch.no_grad():
        out1 = codec1(test_input.unsqueeze(0))[0]
        out2 = codec2(test_input.unsqueeze(0))[0]
    
    # 量化结果应该完全相同（因为我们不使用 rate_bias）
    are_equal = torch.allclose(out1, out2)
    print(f"lmbda=0.001 vs lmbda=1000: {'相同 ✓' if are_equal else '不同 ✗'}")
    assert are_equal, "量化结果应该不受 lmbda 影响！"
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
