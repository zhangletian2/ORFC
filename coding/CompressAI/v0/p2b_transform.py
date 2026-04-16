#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn


def _to_numpy(x):
    """
    将输入统一转成 NumPy 数组（不复制时复用内存）。

    Args:
        x: np.ndarray 或 torch.Tensor 或 array-like

    Returns:
        arr: np.ndarray
    """
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return np.asarray(x)


# =========================
# 1. 统计阶段：fit_p2b
# =========================

def fit_p2b(train_feats,
           K: int = 31,
           max_samples_per_channel: int = 1_000_000,
           random_state=None):
    """
    离线统计每个通道的分位点表（per-channel quantile table）。

    Args:
        train_feats: np.ndarray 或 torch.Tensor，形状 [N, T, C]，float32/float64
            N: 样本数或 batch 大小
            T: token 数（如 257）
            C: 通道数（如 1024）
        K: int
            概率网格点数，p = linspace(0, 1, K)
        max_samples_per_channel: int
            计算分位数时，每个通道最多使用多少样本（从 N*T 中下采样）
        random_state: int or None
            随机种子，控制下采样一致性

    Returns:
        p: np.ndarray, 形状 [K]，dtype=float32
            概率网格 p[k] ∈ [0, 1]
        quantile_table: np.ndarray, 形状 [C, K]，dtype=float32
            第 c 行为该通道在 p 上的分位点 q_c[k]
    """
    feats = _to_numpy(train_feats).astype(np.float32, copy=False)
    if feats.ndim != 3:
        raise ValueError(f"train_feats must be 3D [N, T, C], got shape {feats.shape}")
    N, T, C = feats.shape

    # 展平 N, T 维度，后续在每个通道上取样
    flat = feats.reshape(-1, C)  # [N*T, C]
    num_samples_total = flat.shape[0]

    rng = np.random.default_rng(random_state)
    if num_samples_total > max_samples_per_channel:
        idx = rng.choice(num_samples_total,
                         size=max_samples_per_channel,
                         replace=False)
        flat_sampled = flat[idx]
    else:
        flat_sampled = flat

    # 全局共享的概率网格
    p = np.linspace(0.0, 1.0, K, dtype=np.float64)
    quantile_table = np.empty((C, K), dtype=np.float64)

    # 逐通道计算分位点
    for c in range(C):
        values = flat_sampled[:, c]
        # nanquantile 对 NaN 更鲁棒
        quantile_table[c] = np.nanquantile(values, p)

    return p.astype(np.float32), quantile_table.astype(np.float32)


# =========================
# 2. 前向变换：CDF + logit
# =========================

def _forward_channel(x_c, p, q_c, eps: float):
    """
    单个通道上的前向变换（经验 CDF + logit）。

    Args:
        x_c: np.ndarray, shape [...], 该通道的所有标量
        p: np.ndarray, shape [K], 概率网格
        q_c: np.ndarray, shape [K], 该通道的分位点
        eps: float, 防止 logit 溢出的裁剪常数

    Returns:
        z: np.ndarray, shape [...], logit(u_c)
    """
    K = q_c.shape[0]
    p = p.reshape(-1)
    q_c = q_c.reshape(-1)

    # 用分位点 q_c[k] 定义分段区间，找到每个 x_c 所在的区间 index k
    idx = np.searchsorted(q_c, x_c, side='right') - 1  # 可能在 [-1, K-1]

    # x_c 小于最小分位点、或大于最大分位点的 mask
    mask_low = x_c < q_c[0]
    mask_high = x_c > q_c[-1]

    # 将 idx 限制在 [0, K-2]，便于取 k 和 k+1
    idx = np.clip(idx, 0, K - 2)

    q_left = q_c[idx]
    q_right = q_c[idx + 1]
    denom = q_right - q_left
    # 避免除 0；当 denom=0 时，q_left==q_right，插值退化为常数
    denom_safe = np.where(denom != 0, denom, 1.0)

    # 线性插值系数 t ∈ [0, 1]
    t = (x_c - q_left) / denom_safe
    t = np.clip(t, 0.0, 1.0)

    p_left = p[idx]
    p_right = p[idx + 1]
    u = p_left + (p_right - p_left) * t

    # 尾部截断：小于 q_min 的直接映射到 p[0]，大于 q_max 的映射到 p[-1]
    u = np.where(mask_low, p[0], u)
    u = np.where(mask_high, p[-1], u)

    # 防止 logit 溢出
    u = np.clip(u, eps, 1.0 - eps)

    # logit 映射
    z = np.log(u / (1.0 - u))
    return z


def p2b_forward(x, p, quantile_table, eps: float = 1e-6):
    """
    前向 P2B 变换：x -> z

    Args:
        x: np.ndarray 或 torch.Tensor，shape [B, T, C]，float32/float64
        p: np.ndarray, shape [K], 概率网格
        quantile_table: np.ndarray, shape [C, K], 每个通道的分位点
        eps: float, 裁剪 u 的 epsilon，防止 logit 溢出

    Returns:
        z: 与 x 类型相同（np.ndarray 或 torch.Tensor），shape [B, T, C]
    """
    x_arr = _to_numpy(x)
    if x_arr.ndim != 3:
        raise ValueError(f"x must be 3D [B, T, C], got shape {x_arr.shape}")
    B, T, C = x_arr.shape

    p_np = np.asarray(p, dtype=np.float32)
    qt = np.asarray(quantile_table, dtype=np.float32)
    if qt.shape[0] != C:
        raise ValueError(f"quantile_table shape {qt.shape} incompatible with x last dim {C}")
    if qt.shape[1] != p_np.shape[0]:
        raise ValueError(f"quantile_table K={qt.shape[1]} incompatible with p length {p_np.shape[0]}")

    dtype = x_arr.dtype
    z_np = np.empty_like(x_arr, dtype=np.float32)

    # 逐通道处理（对 [B, T] 维度完全向量化）
    for c in range(C):
        z_np[..., c] = _forward_channel(x_arr[..., c], p_np, qt[c], eps)

    z_np = z_np.astype(dtype, copy=False)

    if isinstance(x, torch.Tensor):
        return torch.from_numpy(z_np).to(x.device).to(x.dtype)
    else:
        return z_np


# =========================
# 3. 逆变换：sigmoid + 反向分段插值
# =========================

def _inverse_channel(z_c, p, q_c):
    """
    单个通道上的逆变换（sigmoid + 反向分段线性插值）。

    Args:
        z_c: np.ndarray, shape [...], 前向输出 z_c
        p: np.ndarray, shape [K], 概率网格
        q_c: np.ndarray, shape [K], 该通道分位点表

    Returns:
        x_hat_c: np.ndarray, shape [...], 近似回到原特征坐标系
    """
    K = q_c.shape[0]
    p = p.reshape(-1)
    q_c = q_c.reshape(-1)

    # 先做 sigmoid 恢复 u_c
    u_c = 1.0 / (1.0 + np.exp(-z_c))

    # 用概率轴 p[k] 做分段线性插值的反变换
    idx = np.searchsorted(p, u_c, side='right') - 1  # 可能在 [-1, K-1]
    mask_low = u_c < p[0]
    mask_high = u_c > p[-1]

    idx = np.clip(idx, 0, K - 2)

    p_left = p[idx]
    p_right = p[idx + 1]
    denom = p_right - p_left
    denom_safe = np.where(denom != 0, denom, 1.0)

    t = (u_c - p_left) / denom_safe
    t = np.clip(t, 0.0, 1.0)

    q_left = q_c[idx]
    q_right = q_c[idx + 1]
    x_hat = q_left + (q_right - q_left) * t

    # 同样处理越界：小于 p_min 的映射到 q_min，大于 p_max 的映射到 q_max
    x_hat = np.where(mask_low, q_c[0], x_hat)
    x_hat = np.where(mask_high, q_c[-1], x_hat)

    return x_hat


def p2b_inverse(z, p, quantile_table):
    """
    逆向 P2B 变换：z -> x_hat

    Args:
        z: np.ndarray 或 torch.Tensor，shape [B, T, C]
        p: np.ndarray, shape [K]
        quantile_table: np.ndarray, shape [C, K]

    Returns:
        x_hat: 与 z 类型相同（np.ndarray 或 torch.Tensor），shape [B, T, C]
    """
    z_arr = _to_numpy(z)
    if z_arr.ndim != 3:
        raise ValueError(f"z must be 3D [B, T, C], got shape {z_arr.shape}")
    B, T, C = z_arr.shape

    p_np = np.asarray(p, dtype=np.float32)
    qt = np.asarray(quantile_table, dtype=np.float32)
    if qt.shape[0] != C:
        raise ValueError(f"quantile_table shape {qt.shape} incompatible with z last dim {C}")
    if qt.shape[1] != p_np.shape[0]:
        raise ValueError(f"quantile_table K={qt.shape[1]} incompatible with p length {p_np.shape[0]}")

    dtype = z_arr.dtype
    x_hat_np = np.empty_like(z_arr, dtype=np.float32)

    # 逐通道处理
    for c in range(C):
        x_hat_np[..., c] = _inverse_channel(z_arr[..., c], p_np, qt[c])

    x_hat_np = x_hat_np.astype(dtype, copy=False)

    if isinstance(z, torch.Tensor):
        return torch.from_numpy(x_hat_np).to(z.device).to(z.dtype)
    else:
        return x_hat_np


# =========================
# 4. PyTorch 模块封装
# =========================

class P2BTransform(nn.Module):
    """
    Per-channel Peaky-to-Balanced (P2B) 分布均衡模块。

    - fit(train_feats): 使用训练集 [N, T, C] 估计 p 和 quantile_table
    - forward(x): 前向变换 x -> z，形状 [B, T, C]
    - inverse(z): 逆变换 z -> x_hat，形状 [B, T, C]

    说明：
        * 内部数值计算全部基于 NumPy（searchsorted + 分段线性插值）。
        * 对于 torch.Tensor，会自动在 CPU 上转成 NumPy 处理，再转回原 device。
        * p 和 quantile_table 也会注册成 buffer，方便一起保存到 state_dict。
    """

    def __init__(self,
                 K: int = 31,
                 max_samples_per_channel: int = 1_000_000,
                 eps: float = 1e-6,
                 random_state=None):
        super().__init__()
        self.K = int(K)
        self.max_samples_per_channel = int(max_samples_per_channel)
        self.eps = float(eps)
        self.random_state = random_state

        # PyTorch buffer，用于保存到 state_dict（初始化为 None）
        self.register_buffer("p", None)
        self.register_buffer("quantile_table", None)

        # NumPy 版本参数，实际数值计算使用
        self._p_np = None
        self._qt_np = None

    # 统计阶段
    def fit(self, train_feats):
        """
        根据训练特征估计每个通道的分位点。

        Args:
            train_feats: np.ndarray 或 torch.Tensor，shape [N, T, C]
        """
        p_np, qt_np = fit_p2b(
            train_feats,
            K=self.K,
            max_samples_per_channel=self.max_samples_per_channel,
            random_state=self.random_state,
        )
        self._p_np = p_np.astype(np.float32, copy=False)
        self._qt_np = qt_np.astype(np.float32, copy=False)

        # 同步一份到 PyTorch buffer，方便保存/加载
        self.p = torch.from_numpy(self._p_np)
        self.quantile_table = torch.from_numpy(self._qt_np)

    # 前向变换
    def forward(self, x):
        """
        前向变换：x -> z

        Args:
            x: np.ndarray 或 torch.Tensor，shape [B, T, C]

        Returns:
            z: 与 x 类型相同，shape [B, T, C]
        """
        if self._p_np is None or self._qt_np is None:
            raise RuntimeError("P2BTransform has not been fitted yet. Call `fit(train_feats)` first.")
        return p2b_forward(x, self._p_np, self._qt_np, eps=self.eps)

    # 逆变换
    def inverse(self, z):
        """
        逆变换：z -> x_hat

        Args:
            z: np.ndarray 或 torch.Tensor，shape [B, T, C]

        Returns:
            x_hat: 与 z 类型相同，shape [B, T, C]
        """
        if self._p_np is None or self._qt_np is None:
            raise RuntimeError("P2BTransform has not been fitted yet. Call `fit(train_feats)` first.")
        return p2b_inverse(z, self._p_np, self._qt_np)
# =========================
# Torch 版本：用于训练图中反向传播
# =========================
def feat_inverse_transform_torch(packed: torch.Tensor, n_split: int = 64) -> torch.Tensor:
    """
    反 packing：从 [B, 1, H, W] 还原到 [B, 257, C]（C=1024）

    Args:
        packed: torch.Tensor, shape [B, 1, H, W] 或 [B, H, W]
        n_split: 通道拆分因子，默认 64

    Returns:
        feat: torch.Tensor, shape [B, 257, C]
    """
    if packed.dim() == 4:
        B, C, H, W = packed.shape
        assert C == 1, f"expected C=1, got {C}"
        x = packed[:, 0]  # [B, H, W]
    elif packed.dim() == 3:
        B, H, W = packed.shape
        x = packed
    else:
        raise ValueError(f"packed must be [B,1,H,W] or [B,H,W], got {packed.shape}")

    h_blocks = H // n_split  # 应为 17
    n = W // 16              # 对 1024 情况下 n = 16
    C_feat = n_split * n     # 应为 1024

    # [B, 17, 64, 16, n]
    arr4 = x.view(B, h_blocks, n_split, 16, n)
    # -> [B, 17, 16, 64, n] -> [B, 17, 16, C_feat]
    stacked = arr4.permute(0, 1, 3, 2, 4).reshape(B, h_blocks, 16, C_feat)
    cls_row = stacked[:, 0]          # [B, 16, C]
    patch_grid = stacked[:, 1:]      # [B, 16, 16, C]

    cls = cls_row[:, 0:1, :]         # [B, 1, C]
    patches = patch_grid.reshape(B, 16 * 16, C_feat)  # [B, 256, C]
    feat = torch.cat([cls, patches], dim=1)           # [B, 257, C]
    return feat


def p2b_inverse_torch(z: torch.Tensor,
                      p: torch.Tensor,
                      quantile_table: torch.Tensor) -> torch.Tensor:
    """
    向量化的 Torch 版 P2B 逆变换（支持 autograd，GPU 友好）.

    Args:
        z: [B, T, C]  来自 codec 的输出（P2B 空间）
        p: [K]        概率网格
        quantile_table: [C, K]  每个通道的分位点表

    Returns:
        x_hat: [B, T, C]  近似回到原始特征空间
    """
    if z.dim() != 3:
        raise ValueError(f"z must be [B, T, C], got {z.shape}")
    B, T, C = z.shape
    device = z.device

    # 保证在当前 device 且连续
    p = p.to(device).contiguous()              # [K]
    qt = quantile_table.to(device).contiguous()  # [C, K]
    K = p.numel()

    # 1) z -> u = sigmoid(z)
    u = torch.sigmoid(z).contiguous()          # [B, T, C]
    u_flat = u.view(-1)                        # [N], N = B*T*C

    # 2) 在概率轴 p 上做分段线性插值：找到区间 index k
    idx = torch.searchsorted(p, u_flat, right=True) - 1   # [N]
    idx = idx.clamp(0, K - 2)

    p_left = p[idx]                            # [N]
    p_right = p[idx + 1]
    denom = p_right - p_left
    denom_safe = torch.where(denom != 0, denom, torch.ones_like(denom))

    t = (u_flat - p_left) / denom_safe         # [N]
    t = t.clamp(0.0, 1.0)

    # 3) 在分位点表 qt 上取对应的 q_left / q_right
    # 为每个位置构造 channel index：0..C-1 重复 B*T 次
    c_idx = torch.arange(C, device=device).view(1, 1, C).expand(B, T, C).reshape(-1)  # [N]

    q_left = qt[c_idx, idx]                    # [N]
    q_right = qt[c_idx, idx + 1]

    # 4) 线性插值得到 x_flat，再 reshape 回 [B, T, C]
    x_flat = q_left + (q_right - q_left) * t   # [N]
    x_hat = x_flat.view(B, T, C)

    return x_hat

# =========================
# Z-score per-channel 标准化（Torch 版）
# =========================

class ZScoreTransform(nn.Module):
    """
    线性 Z-score 变换：逐通道 (x - mean) / std，可逆且对 tail 无饱和。

    用法：
        zt = ZScoreTransform()
        zt.fit(train_feats_np)           # offline，一次性调用
        z = zt(x_torch)                  # forward
        x_hat = zt.inverse(z_hat_torch)  # inverse
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("mean", None)  # [C]
        self.register_buffer("std", None)   # [C]

    def fit(self, train_feats):
        """离线拟合 mean/std，train_feats: np 或 torch，[N, T, C]"""
        mean_np, std_np = fit_zscore(train_feats, eps=self.eps)
        mean_t = torch.from_numpy(mean_np)
        std_t = torch.from_numpy(std_np)
        # 保存为 buffer，方便一起保存/加载
        self.mean = mean_t
        self.std = std_t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C]
        Returns:
            z: [B, T, C]
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("ZScoreTransform not fitted yet. Call fit() first.")
        m = self.mean.to(x.device)   # [C]
        s = self.std.to(x.device)    # [C]
        s_safe = torch.clamp(s, min=self.eps)
        return (x - m) / s_safe

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, T, C]
        Returns:
            x_hat: [B, T, C]
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("ZScoreTransform not fitted yet. Call fit() first.")
        m = self.mean.to(z.device)
        s = self.std.to(z.device)
        return z * s + m


# =========================
# 5. 简单单元测试示例
# =========================

def _run_basic_tests():
    np.random.seed(42)
    torch.manual_seed(42)

    # 假设中间层特征形状 [N, T, C]
    N, T, C = 64, 257, 1024
    train_feats = np.random.randn(N, T, C).astype(np.float32) * 2.0 + 0.5

    # 1) 先做 offline calibration
    p, qt = fit_p2b(train_feats, K=31, max_samples_per_channel=100_000)
    print("fit_p2b: p shape", p.shape, "quantile_table shape", qt.shape)

    # 2) NumPy 版 forward + inverse
    x = np.random.randn(4, T, C).astype(np.float32)
    z = p2b_forward(x, p, qt)
    x2 = p2b_inverse(z, p, qt)
    max_err_np = np.max(np.abs(x - x2))
    print("NumPy max |x - inverse(forward(x))|:", float(max_err_np))

    # 3) PyTorch + 模块版 forward + inverse
    module = P2BTransform(K=31, max_samples_per_channel=100_000, eps=1e-6)
    module.fit(torch.from_numpy(train_feats))

    x_t = torch.randn(4, T, C, dtype=torch.float32)
    z_t = module(x_t)
    x2_t = module.inverse(z_t)
    max_err_t = (x_t - x2_t).abs().max().item()
    print("Torch max |x - inverse(forward(x))|:", max_err_t)

    # 简单断言：最大误差在 1e-3 以内（数值误差 + 尾部截断）
    assert max_err_np < 1e-3, "NumPy path reconstruction error too large"
    assert max_err_t < 1e-3, "Torch path reconstruction error too large"

    print("All basic tests passed.")
    import pdb
    pdb.set_trace()
    return max_err_np, max_err_t


if __name__ == "__main__":
    _run_basic_tests()
