"""
Task-metric estimation for Preconditioned OPQ (Pre-OPQ)

Estimates the per-token channel Gauss-Newton metric
    G_ch = E_{b,t}[ J_{b,t}^T J_{b,t} ]
in the raw (inference) feature space using Rademacher-Hutchinson probes.

Provides:
1. ProjectionF          — frozen ViT block wrapper (same as v3.2)
2. estimate_task_metric — Hutchinson diagonal / low-rank metric estimation
3. compute_precondition — A, A_inv from metric (row-vector right-multiply convention)
4. sanity_check_*       — three mandatory pre-flight checks
"""

import numpy as np
import torch
import torch.autograd


# ================================================================
#                    ProjectionF (frozen ViT blocks)
# ================================================================

class ProjectionF:
    """Wrap 1..N consecutive ViT blocks as projection F: (B,T,D) -> (B,T,D).

    All block parameters are frozen; gradients flow only through the input tensor.
    """

    def __init__(self, blocks, device='cuda'):
        if not isinstance(blocks, (list, tuple)):
            blocks = [blocks]
        self.blocks = list(blocks)
        self.n_blocks = len(self.blocks)
        for blk in self.blocks:
            blk.to(device)
            blk.eval()
            for p in blk.parameters():
                p.requires_grad_(False)
        self.device = device

    def __call__(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

    @torch.no_grad()
    def forward_nograd(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

    def to(self, device):
        for blk in self.blocks:
            blk.to(device)
        self.device = device
        return self


# ================================================================
#                    Metric estimation
# ================================================================

def estimate_task_metric(features_list, F_func, mode='diag',
                         n_probes=50, batch_size=8, lowrank_k=32,
                         device='cuda', verbose=True):
    """
    Estimate per-token channel metric G_ch in the raw feature space.

    Uses Rademacher-Hutchinson probes:
        r ~ Uniform({-1, +1}^{output_dim})
        grad_k = d(F(X) * r).sum() / dX          => grad_k = J^T r
        diag(G) ≈ (1/K) sum_k grad_k^2           (Hutchinson diagonal)

    Args:
        features_list:  list of [T, D] numpy arrays (raw features)
        F_func:         ProjectionF instance
        mode:           'diag' | 'lowrank'
        n_probes:       number of Rademacher probes
        batch_size:     images per forward pass
        lowrank_k:      rank for low-rank component (only used when mode='lowrank')
        device:         GPU device
        verbose:        print progress

    Returns:
        dict with keys:
            'g_diag':    (D,) tensor — diagonal of G_ch
            'U_k':       (D, k) tensor — top-k directions (lowrank only)
            'lambda_k':  (k,) tensor — corresponding eigenvalues (lowrank only)
    """
    D = features_list[0].shape[1]
    N_img = len(features_list)
    G_diag = torch.zeros(D, device=device, dtype=torch.float64)
    n_tokens_total = 0

    collect_grads = (mode == 'lowrank')
    grad_samples = [] if collect_grads else None

    for start in range(0, N_img, batch_size):
        end = min(start + batch_size, N_img)
        X = torch.from_numpy(
            np.stack(features_list[start:end])
        ).float().to(device)
        X.requires_grad_(True)

        F_out = F_func(X)
        B, T, _ = X.shape

        for k in range(n_probes):
            r = torch.sign(torch.randn_like(F_out))
            r[r == 0] = 1.0
            loss = (F_out * r).sum()
            grad = torch.autograd.grad(
                loss, X, retain_graph=(k < n_probes - 1)
            )[0]

            G_diag += (grad.detach().double() ** 2).sum(dim=(0, 1))

            if collect_grads:
                grad_samples.append(
                    grad.detach().reshape(-1, D).float()
                )

        n_tokens_total += B * T

        del X, F_out
        torch.cuda.empty_cache()

        if verbose and (start // batch_size) % 5 == 0:
            print(f"    metric estimation: {end}/{N_img} images")

    G_diag = (G_diag / (n_probes * n_tokens_total)).float()

    result = {'g_diag': G_diag}

    if collect_grads:
        S = torch.cat(grad_samples, dim=0)
        S_mean = S.mean(dim=0, keepdim=True)
        S_centered = S - S_mean
        actual_k = min(lowrank_k, min(S_centered.shape) - 1)

        if actual_k > 0:
            U, sigma, Vh = torch.svd_lowrank(S_centered, q=actual_k)
            lambda_k = sigma ** 2 / S_centered.shape[0]
            V_k = Vh[:, :actual_k]
            result['U_k'] = V_k
            result['lambda_k'] = lambda_k[:actual_k]
        else:
            result['U_k'] = torch.zeros(D, 0, device=device)
            result['lambda_k'] = torch.zeros(0, device=device)

        del S, S_centered, grad_samples
        torch.cuda.empty_cache()

    if verbose:
        g = G_diag
        print(f"    G_diag stats: min={g.min():.6f}, max={g.max():.6f}, "
              f"mean={g.mean():.6f}, kappa={g.max() / g.min().clamp(min=1e-10):.2f}")
        if 'lambda_k' in result and result['lambda_k'].numel() > 0:
            lk = result['lambda_k']
            total_var = (G_diag.sum() + lk.sum()).item()
            explained = lk.sum().item()
            print(f"    lowrank: k={lk.numel()}, top-k var={explained:.4f}, "
                  f"ratio={explained / (total_var + 1e-10):.4f}")

    return result


# ================================================================
#                    Preconditioning transform
# ================================================================

def compute_precondition(metric_info, mode='diag', eps=1e-6, device='cuda'):
    """
    Compute preconditioning matrices from task metric.

    Row-vector right-multiply convention:
        forward:  X_tilde = X @ A
        inverse:  X_hat   = X_tilde_hat @ A_inv

    Satisfies: delta^T G delta = ||delta @ A||^2  (i.e. G = A A^T for diag).

    Args:
        metric_info: dict from estimate_task_metric
        mode:        'diag' | 'lowrank'
        eps:         floor for eigenvalues
        device:      GPU device

    Returns:
        A:          (D,) for diag, (D, D) for lowrank
        A_inv:      (D,) for diag, (D, D) for lowrank
        diagnostics: dict
    """
    g_diag = metric_info['g_diag'].to(device)
    D = g_diag.shape[0]
    g_safe = g_diag.clamp(min=eps)

    diagnostics = {
        'g_min': g_safe.min().item(),
        'g_max': g_safe.max().item(),
        'g_mean': g_safe.mean().item(),
        'g_std': g_safe.std().item(),
        'kappa': (g_safe.max() / g_safe.min()).item(),
    }

    if mode == 'diag':
        sqrt_g = g_safe.sqrt()
        inv_sqrt_g = 1.0 / sqrt_g
        diagnostics['mode'] = 'diag'
        return sqrt_g, inv_sqrt_g, diagnostics

    # lowrank: G ≈ diag(g) + U_k Lambda_k U_k^T
    #
    # Strategy:
    #   1. In diag-scaled space: M = D^{-1/2} U Λ U^T D^{-1/2}  (rank-k)
    #      where D = diag(g).  Full metric there is I + M.
    #   2. Eigendecompose M via SVD of V Λ^{1/2} where V = D^{-1/2} U:
    #      V Λ^{1/2} = W Σ P^T  =>  M = W Σ^2 W^T
    #   3. (I + M)^{1/2}   = I + W diag(√(1+σ²)-1) W^T
    #      (I + M)^{-1/2}  = I + W diag(1/√(1+σ²)-1) W^T
    #   4. A     = diag(√g) @ S,       S = (I+M)^{1/2}
    #      A_inv = S_inv @ diag(1/√g), S_inv = (I+M)^{-1/2}

    U_k = metric_info.get('U_k')
    lambda_k = metric_info.get('lambda_k')

    if U_k is None or lambda_k is None or lambda_k.numel() == 0:
        sqrt_g = g_safe.sqrt()
        inv_sqrt_g = 1.0 / sqrt_g
        diagnostics['mode'] = 'diag_fallback'
        return sqrt_g, inv_sqrt_g, diagnostics

    U_k = U_k.to(device)
    lambda_k = lambda_k.to(device)

    sqrt_g = g_safe.sqrt()
    inv_sqrt_g = 1.0 / sqrt_g

    # V = D^{-1/2} U_k,  shape (D, k)
    V = U_k * inv_sqrt_g.unsqueeze(1)

    # SVD of V diag(√λ):  W Σ P^T,  then M = W Σ² W^T
    VL = V * lambda_k.sqrt().unsqueeze(0)       # (D, k)
    W, sigma, _Pt = torch.linalg.svd(VL, full_matrices=False)  # W: (D, k), sigma: (k,)
    sigma2 = sigma ** 2                          # eigenvalues of M

    # S = (I + M)^{1/2} = I + W diag(√(1+σ²) - 1) W^T
    s_fwd = (1.0 + sigma2).sqrt() - 1.0         # coefficients for forward
    s_inv = 1.0 / (1.0 + sigma2).sqrt() - 1.0   # coefficients for inverse

    S     = torch.eye(D, device=device) + W @ torch.diag(s_fwd) @ W.T
    S_inv = torch.eye(D, device=device) + W @ torch.diag(s_inv) @ W.T

    A     = torch.diag(sqrt_g) @ S
    A_inv = S_inv @ torch.diag(inv_sqrt_g)

    diagnostics['mode'] = 'lowrank'
    diagnostics['lowrank_k'] = lambda_k.numel()
    diagnostics['lambda_k_sum'] = lambda_k.sum().item()
    diagnostics['M_eigenvalues'] = sigma2.cpu().tolist()

    return A, A_inv, diagnostics


# ================================================================
#                    Sanity checks
# ================================================================

def sanity_check_inverse(A, A_inv, mode='diag', D=1024, device='cuda',
                         tol=1e-4):
    """Check 1: X @ A @ A_inv ≈ X."""
    X = torch.randn(16, D, device=device)
    if mode == 'diag':
        X_rec = (X * A) * A_inv
    else:
        X_rec = X @ A @ A_inv
    err = (X_rec - X).abs().max().item()
    ok = err < tol
    if not ok:
        raise RuntimeError(
            f"Sanity check FAILED (inverse): max|X @ A @ A_inv - X| = {err:.2e} "
            f"(tol={tol:.1e})"
        )
    return err


def sanity_check_metric(A, A_inv, metric_info, mode='diag', D=1024,
                        device='cuda', tol=1e-3):
    """Check 2: ||(X-X') @ A||^2 ≈ (X-X')^T G (X-X')."""
    g_diag = metric_info['g_diag'].to(device)
    X = torch.randn(64, D, device=device)
    Xp = torch.randn(64, D, device=device)
    delta = X - Xp

    if mode == 'diag':
        lhs = ((delta * A) ** 2).sum(dim=1)
        rhs = (delta ** 2 * g_diag.unsqueeze(0)).sum(dim=1)
    else:
        lhs = ((delta @ A) ** 2).sum(dim=1)
        G_full = A @ A.T
        rhs = (delta * (delta @ G_full)).sum(dim=1)

    rel_err = ((lhs - rhs).abs() / rhs.clamp(min=1e-10)).max().item()
    ok = rel_err < tol
    if not ok:
        raise RuntimeError(
            f"Sanity check FAILED (metric): max relative error = {rel_err:.2e} "
            f"(tol={tol:.1e})"
        )
    return rel_err


def sanity_check_pipeline_identity(A, A_inv, mode='diag', D=1024,
                                   norm_mode='per_image', device='cuda',
                                   tol=5e-4):
    """Check 3: full pipeline with PQ=identity returns X_hat ≈ X."""
    import sys, os
    v32_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'v3.2')
    if v32_root not in sys.path:
        sys.path.insert(0, v32_root)
    from opq import batch_normalize_gpu, batch_inv_normalize_gpu

    B, T = 4, 197
    X = torch.randn(B, T, D, device=device)

    if mode == 'diag':
        X_tilde = X * A.view(1, 1, D)
    else:
        X_tilde = X @ A

    Y, Mu, Std = batch_normalize_gpu(X_tilde, mode=norm_mode)
    R = torch.eye(D, device=device)
    Z = Y.reshape(-1, D) @ R
    Z_hat = Z
    Y_hat_flat = Z_hat @ R.T
    Y_hat = Y_hat_flat.reshape(B, T, D)
    X_tilde_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)

    if mode == 'diag':
        X_hat = X_tilde_hat * A_inv.view(1, 1, D)
    else:
        X_hat = X_tilde_hat @ A_inv

    err = (X_hat - X).abs().max().item()
    ok = err < tol
    if not ok:
        raise RuntimeError(
            f"Sanity check FAILED (pipeline identity): max|X_hat - X| = {err:.2e} "
            f"(tol={tol:.1e})"
        )
    return err


def run_all_sanity_checks(A, A_inv, metric_info, mode='diag', D=1024,
                          norm_mode='per_image', device='cuda', verbose=True):
    """Run all three mandatory pre-flight checks."""
    if verbose:
        print("  Running sanity checks...")

    e1 = sanity_check_inverse(A, A_inv, mode, D, device)
    if verbose:
        print(f"    Check 1 (inverse):  max_err={e1:.2e}  OK")

    e2 = sanity_check_metric(A, A_inv, metric_info, mode, D, device)
    if verbose:
        print(f"    Check 2 (metric):   rel_err={e2:.2e}  OK")

    e3 = sanity_check_pipeline_identity(A, A_inv, mode, D, norm_mode, device)
    if verbose:
        print(f"    Check 3 (pipeline): max_err={e3:.2e}  OK")

    return {'inverse_err': e1, 'metric_err': e2, 'pipeline_err': e3}
