"""
OPQ (Optimized Product Quantization) 旋转预处理 + GPU 批量 k-means

全部核心计算在 GPU 上完成, 最小化 CPU↔GPU 数据搬运。

提供:
1. batch_normalize_gpu / batch_inv_normalize_gpu: GPU 批量归一化
2. batched_kmeans:        GPU 批量 k-means (多组并行)
3. batched_assign:        GPU 批量最近邻分配 + 重建
4. learn_pca_rotation:    PCA + interleave 分配 (GPU 加速)
5. learn_opq_rotation:    交替优化旋转矩阵 + PQ 码本 (全 GPU)
6. apply_rotation / apply_inv_rotation: 旋转变换 (支持 numpy/torch)

参考: Ge T. et al., "Optimized Product Quantization", TPAMI 2014
"""

import numpy as np
import torch


def _to_gpu(x, device):
    """numpy 或 torch tensor → 指定设备的 float32 tensor"""
    if isinstance(x, torch.Tensor):
        return x.float().to(device)
    return torch.from_numpy(np.ascontiguousarray(x)).float().to(device)


# ================================================================
#                    GPU 批量归一化
# ================================================================

def batch_normalize_gpu(X, mode='per_image', eps=1e-5, n_prefix=0):
    """
    GPU 批量归一化

    Args:
        X:    [N_img, T, C] GPU tensor
        mode: 'per_image' | 'per_token_ln' | 'split_cls_patch' | 'split_reg_cls_patch'
        n_prefix: CLS+register prefix length (split_* modes). DINOv3: 5 = 1 CLS + 4 reg.

    Returns:
        Y:   [N_img, T, C] 归一化后
        mu:  均值 (per_image: [N_img, 1, 1]; split_*/per_token_ln: [N_img, T, 1])
        std: 标准差 (同 mu 形状)
    """
    if mode == 'split_reg_cls_patch':
        # Register tokens [1, n_prefix) have their own μ/σ.
        # CLS (token 0) shares μ/σ with patch tokens [n_prefix:].
        N, T, C = X.shape
        mu = torch.empty(N, T, 1, device=X.device, dtype=X.dtype)
        std = torch.empty(N, T, 1, device=X.device, dtype=X.dtype)
        if n_prefix >= 2 and n_prefix < T:
            X_reg = X[:, 1:n_prefix, :]
            mu_reg = X_reg.mean(dim=(1, 2), keepdim=True)
            std_reg = (((X_reg - mu_reg) ** 2).mean(dim=(1, 2), keepdim=True) + eps).sqrt()
            mu[:, 1:n_prefix, :] = mu_reg
            std[:, 1:n_prefix, :] = std_reg

            X_cp = torch.cat([X[:, :1, :], X[:, n_prefix:, :]], dim=1)
            mu_cp = X_cp.mean(dim=(1, 2), keepdim=True)
            std_cp = (((X_cp - mu_cp) ** 2).mean(dim=(1, 2), keepdim=True) + eps).sqrt()
            mu[:, :1, :] = mu_cp
            mu[:, n_prefix:, :] = mu_cp
            std[:, :1, :] = std_cp
            std[:, n_prefix:, :] = std_cp
        else:
            mu_all = X.mean(dim=(1, 2), keepdim=True)
            std_all = (((X - mu_all) ** 2).mean(dim=(1, 2), keepdim=True) + eps).sqrt()
            mu[:] = mu_all
            std[:] = std_all
    elif mode == 'split_cls_patch':
        N, T, C = X.shape
        mu = torch.empty(N, T, 1, device=X.device, dtype=X.dtype)
        std = torch.empty(N, T, 1, device=X.device, dtype=X.dtype)
        if 0 < n_prefix < T:
            X_cr = X[:, :n_prefix, :]
            mu_cr = X_cr.mean(dim=(1, 2), keepdim=True)
            std_cr = (((X_cr - mu_cr) ** 2).mean(dim=(1, 2), keepdim=True) + eps).sqrt()
            mu[:, :n_prefix, :] = mu_cr
            std[:, :n_prefix, :] = std_cr

            X_p = X[:, n_prefix:, :]
            mu_p = X_p.mean(dim=(1, 2), keepdim=True)
            std_p = (((X_p - mu_p) ** 2).mean(dim=(1, 2), keepdim=True) + eps).sqrt()
            mu[:, n_prefix:, :] = mu_p
            std[:, n_prefix:, :] = std_p
        else:
            mu_all = X.mean(dim=(1, 2), keepdim=True)
            std_all = (((X - mu_all) ** 2).mean(dim=(1, 2), keepdim=True) + eps).sqrt()
            mu[:] = mu_all
            std[:] = std_all
    elif mode == 'per_image':
        mu = X.mean(dim=(1, 2), keepdim=True)       # [N, 1, 1]
        var = ((X - mu) ** 2).mean(dim=(1, 2), keepdim=True)
        std = (var + eps).sqrt()
    elif mode == 'per_token_ln':
        mu = X.mean(dim=2, keepdim=True)             # [N, T, 1]
        var = ((X - mu) ** 2).mean(dim=2, keepdim=True)
        std = (var + eps).sqrt()
    else:
        raise ValueError(f"Unknown norm mode: {mode}")
    Y = (X - mu) / std
    return Y, mu, std


def batch_inv_normalize_gpu(Y, mu, std):
    """GPU 批量逆归一化: X = Y * std + mu"""
    return Y * std + mu


# ================================================================
#                    GPU 批量 k-means
# ================================================================

def batched_kmeans(sub_vectors_3d, K, max_iter=100, device='cuda',
                   verbose=False, initial_centroids=None):
    """
    GPU 批量 k-means: 同时对 G 组子向量做 k-means

    输入支持 numpy 或 GPU tensor, **返回 GPU tensor**。

    Args:
        sub_vectors_3d: [G, N, dim] numpy 或 GPU tensor
        K:              码本大小
        max_iter:       最大迭代次数
        device:         GPU 设备
        verbose:        是否打印
        initial_centroids: 可选的 [G, K, dim] Lloyd 初始值

    Returns:
        centroids: [G, K, dim] **GPU tensor** float32
    """
    X = _to_gpu(sub_vectors_3d, device)
    G, N, dim = X.shape

    # 动态 chunk_size: 同时考虑 cdist 输出 [G,chunk,K] 和内部缓冲 [G,chunk,dim]
    max_mem_bytes = 1 * 1024**3
    mem_per_sample = G * max(K, dim) * 4
    chunk_size = max(1, min(N, max_mem_bytes // mem_per_sample))

    if initial_centroids is None:
        init_indices = torch.stack([
            torch.randperm(N, device=device)[:K] for _ in range(G)
        ])
        centroids = torch.gather(
            X, 1, init_indices.unsqueeze(-1).expand(-1, -1, dim)
        )
    else:
        centroids = _to_gpu(initial_centroids, device).clone()

    # flat scatter_add 的 group offset
    offset = torch.arange(G, device=device).unsqueeze(1) * K  # [G, 1]

    for it in range(max_iter):
        flat_sums = torch.zeros(G * K, dim, device=device)
        flat_counts = torch.zeros(G * K, device=device)

        for i in range(0, N, chunk_size):
            batch = X[:, i:i + chunk_size, :]
            bs = batch.shape[1]

            with torch.no_grad():
                dists = torch.cdist(batch, centroids)  # [G, chunk, K]
                labels = dists.argmin(dim=2)            # [G, chunk]

            flat_labels = (labels + offset).reshape(-1)
            flat_batch = batch.reshape(G * bs, dim)

            flat_sums.scatter_add_(
                0, flat_labels.unsqueeze(1).expand(-1, dim), flat_batch
            )
            flat_counts.scatter_add_(
                0, flat_labels, torch.ones(G * bs, device=device)
            )

        counts = flat_counts.reshape(G, K)
        counts_safe = counts.unsqueeze(-1).clamp(min=1)
        new_centroids = flat_sums.reshape(G, K, dim) / counts_safe

        # 空簇: 用同组随机样本替换
        empty = (counts == 0)
        if empty.any():
            eg = empty.nonzero(as_tuple=True)
            n_empty = eg[0].shape[0]
            random_n = torch.randint(N, (n_empty,), device=device)
            new_centroids[eg[0], eg[1]] = X[eg[0], random_n]

        shifts = (new_centroids - centroids).reshape(G, -1).norm(dim=1)
        max_shift = shifts.max().item()
        centroids = new_centroids

        if verbose and (it + 1) % 10 == 0:
            print(f"    batched_kmeans iter {it+1}/{max_iter}: "
                  f"max_shift={max_shift:.6f}")

        if max_shift < 1e-4:
            if verbose:
                print(f"    batched_kmeans converged at iter {it+1}")
            break

    return centroids  # GPU tensor


def batched_assign(sub_vectors_3d, centroids_3d, device='cuda',
                   chunk_size=None):
    """
    GPU 批量最近邻分配 + 重建

    输入支持 numpy 或 GPU tensor, **返回 GPU tensor**。

    Args:
        sub_vectors_3d: [G, N, dim] numpy 或 GPU tensor
        centroids_3d:   [G, K, dim] numpy 或 GPU tensor
        device:         GPU 设备
        chunk_size:     沿 N 维分块大小 (None → 自动)

    Returns:
        recon:  [G, N, dim] **GPU tensor** float32
        labels: [G, N] **GPU tensor** int64
    """
    X = _to_gpu(sub_vectors_3d, device)
    C = _to_gpu(centroids_3d, device)
    G, N, dim = X.shape
    K = C.shape[1]

    if chunk_size is None:
        max_mem = 1 * 1024**3
        chunk_size = max(1, min(N, max_mem // (G * K * 4)))

    all_recon = []
    all_labels = []

    for i in range(0, N, chunk_size):
        batch = X[:, i:i + chunk_size, :]

        with torch.no_grad():
            dists = torch.cdist(batch, C)                         # [G, chunk, K]
            labels = dists.argmin(dim=2)                           # [G, chunk]
            labels_exp = labels.unsqueeze(-1).expand(-1, -1, dim)  # [G, chunk, dim]
            recon = torch.gather(C, 1, labels_exp)                 # [G, chunk, dim]

        all_recon.append(recon)
        all_labels.append(labels)

    recon = torch.cat(all_recon, dim=1)    # [G, N, dim]
    labels = torch.cat(all_labels, dim=1)  # [G, N]
    return recon, labels


# ================================================================
#                    PCA / OPQ 旋转 (GPU)
# ================================================================

def learn_pca_rotation(vectors, num_groups, embedding_dim, device='cuda',
                       verbose=True):
    """
    PCA 旋转 + interleave 分配 (GPU 加速)

    Args:
        vectors:       [N, D] numpy 或 GPU tensor
        num_groups:    分组数
        embedding_dim: 每组维度
        device:        GPU 设备
        verbose:       是否打印

    Returns:
        R:               [D, D] numpy float32, 正交旋转矩阵 (z = y @ R)
        eigenvalues:     [D] numpy float32, PCA 特征值 (降序)
        group_variances: [num_groups] numpy float32, 每组总方差
    """
    X = _to_gpu(vectors, device)
    N, D = X.shape
    assert D == num_groups * embedding_dim, \
        f"D={D} != num_groups({num_groups}) * embedding_dim({embedding_dim})"

    if verbose:
        print(f"    PCA rotation: N={N:,}, D={D}")

    with torch.no_grad():
        # 中心化 + 协方差 (GPU matmul)
        mean = X.mean(dim=0, keepdim=True)
        X_centered = X - mean
        cov = (X_centered.T @ X_centered) / N  # [D, D]

        # 特征值分解 (GPU eigh: 返回升序)
        eigenvalues_t, eigenvectors_t = torch.linalg.eigh(cov)

        # 降序排列
        idx = torch.argsort(eigenvalues_t, descending=True)
        eigenvalues_t = eigenvalues_t[idx]
        eigenvectors_t = eigenvectors_t[:, idx]

        # Interleave 分配
        perm = torch.zeros(D, dtype=torch.long, device=device)
        for g in range(num_groups):
            for k in range(embedding_dim):
                perm[g * embedding_dim + k] = g + k * num_groups

        R = eigenvectors_t[:, perm].contiguous()  # [D, D]

    # 转回 numpy 输出
    eigenvalues_np = eigenvalues_t.cpu().numpy().astype(np.float32)
    R_np = R.cpu().numpy().astype(np.float32)

    group_variances = np.zeros(num_groups, dtype=np.float32)
    for g in range(num_groups):
        group_pca_indices = [g + k * num_groups for k in range(embedding_dim)]
        group_variances[g] = eigenvalues_np[group_pca_indices].sum()

    if verbose:
        print(f"    特征值范围: [{eigenvalues_np[-1]:.6f}, {eigenvalues_np[0]:.4f}]")
        print(f"    组方差范围: [{group_variances.min():.4f}, {group_variances.max():.4f}], "
              f"ratio={group_variances.max() / (group_variances.min() + 1e-10):.2f}")
        orth_err = torch.max(torch.abs(
            R.T @ R - torch.eye(D, device=device)
        )).item()
        print(f"    正交性误差: {orth_err:.2e}")

    return R_np, eigenvalues_np, group_variances


def _pq_train_and_recon(Z, num_groups, embedding_dim, K, max_iter, device):
    """
    PQ 训练 + 重建 (OPQ 内部使用, 全 GPU)

    Args:
        Z:              [N, D] **GPU tensor** (旋转后向量)
        num_groups:     分组数
        embedding_dim:  每组维度
        K:              码本大小
        max_iter:       k-means 最大迭代次数
        device:         设备

    Returns:
        Z_hat:      [N, D] **GPU tensor**, PQ 重建
        centroids:  [G, K, dim] **GPU tensor**, 码本
        total_mse:  float
    """
    N, D = Z.shape

    # [N, D] → [G, N, dim] — GPU reshape + permute (零拷贝)
    sub_3d = Z.reshape(N, num_groups, embedding_dim) \
              .permute(1, 0, 2).contiguous()  # [G, N, dim]

    # 批量 k-means → GPU tensor centroids
    centroids = batched_kmeans(sub_3d, K, max_iter=max_iter,
                               device=device, verbose=False)

    # 批量最近邻重建 → GPU tensor recon
    recon_3d, _ = batched_assign(sub_3d, centroids, device=device)

    # [G, N, dim] → [N, D]
    Z_hat = recon_3d.permute(1, 0, 2).reshape(N, D)

    total_mse = float(((Z - Z_hat) ** 2).mean().item())
    return Z_hat, centroids, total_mse


def learn_opq_rotation(vectors, num_groups, embedding_dim, K,
                       max_iter_opq=20, max_iter_kmeans=50,
                       device='cuda', verbose=True):
    """
    OPQ 交替优化 (全 GPU 加速版)

    交替:
    1. 固定 R, 对旋转后向量做 PQ (batched_kmeans)
    2. 固定码本, 用 Procrustes 优化 R (GPU SVD)

    整个循环中间零 CPU 操作, 仅最终结果转 numpy 输出。

    Args:
        vectors:          [N, D] numpy, 归一化后的全维度向量
        num_groups:       分组数
        embedding_dim:    每组维度
        K:                码本大小
        max_iter_opq:     OPQ 外层迭代次数
        max_iter_kmeans:  每轮 k-means 的迭代次数
        device:           设备
        verbose:          是否打印

    Returns:
        R:          [D, D] numpy float32, 最优旋转矩阵
        codebooks:  list of [K, dim] numpy, 最优每组码本
        history:    list of (mse, delta), 优化历史
    """
    # 一次性上传到 GPU
    X = _to_gpu(vectors, device)
    N, D = X.shape
    assert D == num_groups * embedding_dim

    # PCA 初始化 R (返回 numpy, 再上传 GPU)
    if verbose:
        print("  OPQ: PCA 初始化旋转矩阵...")
    R_np, eigenvalues, group_vars = learn_pca_rotation(
        X, num_groups, embedding_dim, device=device, verbose=verbose
    )
    R = torch.from_numpy(R_np).float().to(device)

    I = torch.eye(D, device=device)
    prev_mse = float('inf')
    history = []
    best_R = R.clone()
    best_centroids = None
    best_mse = float('inf')

    for it in range(max_iter_opq):
        if verbose:
            print(f"\n  OPQ iter {it + 1}/{max_iter_opq}:")

        # Step 1: 固定 R, 做 PQ (全 GPU)
        with torch.no_grad():
            Z = X @ R  # GPU matmul [N, D] @ [D, D]

        Z_hat, centroids, mse = _pq_train_and_recon(
            Z, num_groups, embedding_dim, K, max_iter_kmeans, device
        )

        delta = prev_mse - mse
        history.append((float(mse), float(delta)))

        if verbose:
            print(f"    PQ MSE = {mse:.8f}, delta = {delta:.2e}")

        if mse < best_mse:
            best_mse = mse
            best_R = R.clone()
            best_centroids = centroids.clone()

        # 收敛检查
        if abs(delta) < 1e-8 and it > 0:
            if verbose:
                print(f"    收敛于 iter {it + 1}")
            break
        prev_mse = mse

        # Step 2: Procrustes 优化 R (全 GPU)
        with torch.no_grad():
            A = X.T @ Z_hat                # GPU matmul [D, N] @ [N, D]
            U, S, Vh = torch.linalg.svd(A)  # GPU SVD
            R_new = U @ Vh

            # 确保 det(R) = +1 (proper rotation)
            if torch.det(R_new) < 0:
                U[:, -1] *= -1
                R_new = U @ Vh

            R = R_new

        if verbose:
            orth_err = torch.max(torch.abs(R @ R.T - I)).item()
            det_val = torch.det(R).item()
            print(f"    R 正交性误差: {orth_err:.2e}, det(R)={det_val:.6f}")

    if verbose:
        print(f"\n  OPQ 完成: best MSE = {best_mse:.8f} "
              f"({len(history)} iters)")

    # 转回 numpy 输出
    R_np = best_R.cpu().numpy().astype(np.float32)
    centroids_np = best_centroids.cpu().numpy().astype(np.float32)
    codebooks = [centroids_np[g] for g in range(num_groups)]

    return R_np, codebooks, history


# ================================================================
#                    旋转变换
# ================================================================

def apply_rotation(y, R):
    """
    应用旋转: z = y @ R  (支持 numpy / torch, 自动适配)

    Args:
        y: [T, D] numpy 或 torch tensor
        R: [D, D] numpy 或 torch tensor
    Returns:
        z: [T, D] (类型与 y 一致)
    """
    if isinstance(y, torch.Tensor):
        if isinstance(R, torch.Tensor):
            return y @ R
        R_t = torch.from_numpy(R).to(y.device, y.dtype)
        return y @ R_t
    return (y @ R).astype(np.float32)


def apply_inv_rotation(z_hat, R):
    """
    应用逆旋转: y_hat = z_hat @ R^T  (支持 numpy / torch, 自动适配)

    Args:
        z_hat: [T, D] numpy 或 torch tensor
        R:     [D, D] numpy 或 torch tensor
    Returns:
        y_hat: [T, D] (类型与 z_hat 一致)
    """
    if isinstance(z_hat, torch.Tensor):
        if isinstance(R, torch.Tensor):
            return z_hat @ R.T
        R_t = torch.from_numpy(R).to(z_hat.device, z_hat.dtype)
        return z_hat @ R_t.T
    return (z_hat @ R.T).astype(np.float32)
