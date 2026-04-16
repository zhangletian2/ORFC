"""
Tail distortion metrics for OPQ quantization experiments.

Computes label-free metrics by comparing two forward passes through the
frozen ViT tail (original H vs reconstructed H_hat):

  1. Head-wise attention KL / JS divergence
  2. CLS-to-patch attention divergence
  3. Attention output MSE  +  decomposition (deltaA)V vs A(deltaV)
  4. Attention rollout divergence (standard + Watt-style residual)
  5. Block output (residual stream) MSE  +  per-block/CLS/patch profile
  6. Quadratic sensitivity  delta^T G delta  +  dimensionless variants

Reference implementations:
  - v3.1/analyze_attention_drift.py  (compute_attention_values)
  - v3.3/metric_estimator.py         (estimate_task_metric, ProjectionF)
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd

ORFC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(ORFC_ROOT, "..", ".."))
if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)

from metric_estimator import ProjectionF, estimate_task_metric


# ================================================================
#                    Low-level helpers
# ================================================================

def _layernorm(x_flat, weight, bias, eps=1e-6):
    """Apply LayerNorm with given weight/bias.  x_flat: [*, C]"""
    mean = x_flat.mean(dim=-1, keepdim=True)
    var = ((x_flat - mean) ** 2).mean(dim=-1, keepdim=True)
    x_norm = (x_flat - mean) / torch.sqrt(var + eps)
    return x_norm * weight + bias


def _extract_block_attn(x, block):
    """Run one ViT block and return intermediate attention quantities.

    Args:
        x: [B, T, C] input to the block.
        block: a single ViT Block module.

    Returns:
        attn_weights: [B, H, T, T]  (after softmax)
        attn_output:  [B, H, T, d_head]  (A @ V per head)
        v:            [B, H, T, d_head]  (value vectors per head)
        block_output: [B, T, C]
    """
    B, T, C = x.shape
    attn = block.attn
    num_heads = attn.num_heads
    head_dim = C // num_heads
    scale = head_dim ** -0.5

    x_normed = _layernorm(
        x.reshape(-1, C), block.norm1.weight.data, block.norm1.bias.data
    ).reshape(B, T, C)

    qkv = F.linear(x_normed, attn.qkv.weight.data, attn.qkv.bias.data)
    qkv = qkv.reshape(B, T, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    aw = F.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
    ao = aw @ v                                       # [B, H, T, d_head]

    ao_cat = ao.transpose(1, 2).reshape(B, T, C)
    attn_proj = F.linear(
        ao_cat, attn.proj.weight.data, attn.proj.bias.data
    )
    x_after_attn = x + attn_proj

    x_normed2 = _layernorm(
        x_after_attn.reshape(-1, C),
        block.norm2.weight.data, block.norm2.bias.data,
    ).reshape(B, T, C)
    mlp = block.mlp
    mlp_out = mlp.fc2(mlp.act(mlp.fc1(x_normed2)))
    block_output = x_after_attn + mlp_out

    return aw, ao, v, block_output


def _kl_rows(p, q, eps=1e-10):
    """Row-wise KL(p || q).  p, q: [..., T]  (probability rows)."""
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def _js_rows(p, q, eps=1e-10):
    """Row-wise Jensen-Shannon divergence.  Symmetric, bounded."""
    m = 0.5 * (p + q)
    return 0.5 * _kl_rows(p, m, eps) + 0.5 * _kl_rows(q, m, eps)


# ================================================================
#                    TailMetricComputer
# ================================================================

class TailMetricComputer:
    """Compute label-free tail distortion metrics between original and
    reconstructed features at a given ViT layer."""

    def __init__(self, backbone, layer_idx, device='cuda'):
        self.backbone = backbone
        self.layer_idx = layer_idx
        self.device = device
        self.tail_blocks = list(backbone.blocks[layer_idx + 1:])
        self.norm = backbone.norm
        self.n_tail = len(self.tail_blocks)
        self._g_diag_cache = None

    # ----------------------------------------------------------
    #  Internal: run tail and collect per-block quantities
    # ----------------------------------------------------------

    @torch.no_grad()
    def _forward_tail(self, h):
        """Forward through tail, returning per-block intermediates.

        Args:
            h: [B, T, C]

        Returns:
            block_outputs: list of [B, T, C]
            attn_weights:  list of [B, H, T, T]
            attn_outputs:  list of [B, H, T, d_head]
            values:        list of [B, H, T, d_head]
        """
        block_outs, aws, aos, vs = [], [], [], []
        x = h
        for blk in self.tail_blocks:
            aw, ao, v, x = _extract_block_attn(x, blk)
            block_outs.append(x)
            aws.append(aw)
            aos.append(ao)
            vs.append(v)
        return block_outs, aws, aos, vs

    # ----------------------------------------------------------
    #  3.1  Head-wise attention KL / JS
    # ----------------------------------------------------------

    def _attn_divergence(self, aws_orig, aws_hat, mode='kl'):
        """Average row-level divergence across blocks, heads, tokens."""
        fn = _kl_rows if mode == 'kl' else _js_rows
        total = 0.0
        count = 0
        for aw_o, aw_h in zip(aws_orig, aws_hat):
            div = fn(aw_o, aw_h)            # [B, H, T]
            total += div.sum().item()
            count += div.numel()
        return total / max(count, 1)

    # ----------------------------------------------------------
    #  3.2  CLS attention divergence
    # ----------------------------------------------------------

    def _cls_attn_divergence(self, aws_orig, aws_hat, mode='kl'):
        """Divergence of the CLS-query attention row (row 0) only."""
        fn = _kl_rows if mode == 'kl' else _js_rows
        total = 0.0
        count = 0
        for aw_o, aw_h in zip(aws_orig, aws_hat):
            cls_o = aw_o[:, :, 0, :]
            cls_h = aw_h[:, :, 0, :]
            div = fn(cls_o, cls_h)
            total += div.sum().item()
            count += div.numel()
        return total / max(count, 1)

    def _cls_attn_l2(self, aws_orig, aws_hat):
        """L2 distance of CLS-query attention row."""
        total = 0.0
        count = 0
        for aw_o, aw_h in zip(aws_orig, aws_hat):
            cls_o = aw_o[:, :, 0, :]
            cls_h = aw_h[:, :, 0, :]
            diff = ((cls_o - cls_h) ** 2).sum(dim=-1)
            total += diff.sum().item()
            count += diff.numel()
        return total / max(count, 1)

    # ----------------------------------------------------------
    #  3.3  Attention output MSE
    # ----------------------------------------------------------

    def _attn_output_mse(self, aos_orig, aos_hat):
        """Per-head attention output MSE, averaged across blocks/heads/tokens."""
        total = 0.0
        count = 0
        for ao_o, ao_h in zip(aos_orig, aos_hat):
            diff = ((ao_o - ao_h) ** 2).sum(dim=-1)   # [B, H, T]
            total += diff.sum().item()
            count += diff.numel()
        return total / max(count, 1)

    # ----------------------------------------------------------
    #  C.  Attention output decomposition: (dA)V vs A(dV)
    # ----------------------------------------------------------

    def _attn_output_decomposition(self, aws_o, aws_h, vs_o, vs_h):
        """Decompose delta(AV) into (deltaA)V and A(deltaV).

        Returns:
            dA_V_mse:  float — MSE of (A_hat - A) @ V_orig
            A_dV_mse:  float — MSE of A_orig @ (V_hat - V_orig)
            per_block_dA_V: list of float
            per_block_A_dV: list of float
        """
        dA_V_total, A_dV_total = 0.0, 0.0
        per_block_dA_V, per_block_A_dV = [], []
        count = 0
        for aw_o, aw_h, v_o, v_h in zip(aws_o, aws_h, vs_o, vs_h):
            dA = aw_h - aw_o           # [B, H, T, T]
            dV = v_h - v_o             # [B, H, T, d]
            dA_V = dA @ v_o            # (deltaA)V
            A_dV = aw_o @ dV           # A(deltaV)
            dA_V_val = (dA_V ** 2).sum(dim=-1).mean().item()
            A_dV_val = (A_dV ** 2).sum(dim=-1).mean().item()
            dA_V_total += dA_V_val
            A_dV_total += A_dV_val
            per_block_dA_V.append(dA_V_val)
            per_block_A_dV.append(A_dV_val)
            count += 1
        return (dA_V_total / max(count, 1),
                A_dV_total / max(count, 1),
                per_block_dA_V, per_block_A_dV)

    # ----------------------------------------------------------
    #  4.  Attention rollout divergence
    # ----------------------------------------------------------

    def _compute_rollout(self, aws):
        """Standard attention rollout: blend = 0.5*I + 0.5*A_bar."""
        R = None
        for aw in aws:
            A_bar = aw.mean(dim=1)                     # [B, T, T]
            T_dim = A_bar.shape[-1]
            I = torch.eye(T_dim, device=A_bar.device).unsqueeze(0)
            blend = 0.5 * I + 0.5 * A_bar
            R = blend if R is None else R @ blend
        return R

    def _compute_rollout_watt(self, aws):
        """No-residual rollout: blend = A_bar (no identity blending).

        Tests the hypothesis that residual blending (0.5*I + 0.5*A_bar)
        smooths the signal in long tails. If blk05 partial rho recovers
        with this variant, the smoothing hypothesis is confirmed.
        """
        R = None
        for aw in aws:
            A_bar = aw.mean(dim=1)  # already row-stochastic
            R = A_bar if R is None else R @ A_bar
        return R

    def _rollout_cls_divergence(self, aws_orig, aws_hat, mode='l2'):
        """Divergence of CLS->patch rollout map."""
        R_o = self._compute_rollout(aws_orig)
        R_h = self._compute_rollout(aws_hat)
        cls_o = R_o[:, 0, 1:]
        cls_h = R_h[:, 0, 1:]
        if mode == 'l2':
            return ((cls_o - cls_h) ** 2).mean().item()
        p = cls_o / cls_o.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        q = cls_h / cls_h.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        return _kl_rows(p, q).mean().item()

    def _rollout_patch_frobenius(self, aws_orig, aws_hat):
        """Frobenius-norm difference of patch-to-patch rollout."""
        R_o = self._compute_rollout(aws_orig)
        R_h = self._compute_rollout(aws_hat)
        patch_o = R_o[:, 1:, 1:]
        patch_h = R_h[:, 1:, 1:]
        return ((patch_o - patch_h) ** 2).mean().item()

    def _rollout_watt_cls_divergence(self, aws_orig, aws_hat, mode='l2'):
        """Watt-style rollout CLS->patch divergence."""
        R_o = self._compute_rollout_watt(aws_orig)
        R_h = self._compute_rollout_watt(aws_hat)
        cls_o = R_o[:, 0, 1:]
        cls_h = R_h[:, 0, 1:]
        if mode == 'l2':
            return ((cls_o - cls_h) ** 2).mean().item()
        p = cls_o / cls_o.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        q = cls_h / cls_h.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        return _kl_rows(p, q).mean().item()

    def _rollout_watt_patch_frobenius(self, aws_orig, aws_hat):
        """Watt-style rollout patch-to-patch Frobenius difference."""
        R_o = self._compute_rollout_watt(aws_orig)
        R_h = self._compute_rollout_watt(aws_hat)
        return ((R_o[:, 1:, 1:] - R_h[:, 1:, 1:]) ** 2).mean().item()

    # ----------------------------------------------------------
    #  5.  Block output MSE  (residual stream)
    # ----------------------------------------------------------

    def _block_output_mse(self, bouts_orig, bouts_hat, weight_mode='uniform'):
        """Per-token L2 averaged over blocks with optional weighting."""
        n = len(bouts_orig)
        if weight_mode == 'exp_inc':
            raw = [2.0 ** i for i in range(n)]
            s = sum(raw)
            weights = [w / s for w in raw]
        else:
            weights = [1.0 / n] * n

        total = 0.0
        for w, bo_o, bo_h in zip(weights, bouts_orig, bouts_hat):
            mse = ((bo_o - bo_h) ** 2).sum(dim=-1).mean().item()
            total += w * mse
        return total

    # ----------------------------------------------------------
    #  B.  Per-block detail profiling
    # ----------------------------------------------------------

    def _compute_per_block_detail(self, bouts_o, bouts_h, aos_o, aos_h):
        """Compute per-block profiling metrics for error propagation analysis.

        Returns dict with per-block lists.
        """
        n = len(bouts_o)
        per_block_output_mse = []
        per_block_cls_mse = []
        per_block_patch_mse = []
        per_block_attn_output_mse = []
        per_block_per_head_attn_output_mse = []

        for i in range(n):
            bo_o, bo_h = bouts_o[i], bouts_h[i]
            # Full block output MSE
            per_block_output_mse.append(
                ((bo_o - bo_h) ** 2).sum(dim=-1).mean().item()
            )
            # CLS token MSE (token 0)
            per_block_cls_mse.append(
                ((bo_o[:, 0, :] - bo_h[:, 0, :]) ** 2).sum(dim=-1).mean().item()
            )
            # Patch tokens MSE (tokens 1:)
            per_block_patch_mse.append(
                ((bo_o[:, 1:, :] - bo_h[:, 1:, :]) ** 2).sum(dim=-1).mean().item()
            )
            # Attention output MSE for this block
            ao_o, ao_h = aos_o[i], aos_h[i]  # [B, H, T, d_head]
            ao_diff = ((ao_o - ao_h) ** 2).sum(dim=-1)  # [B, H, T]
            per_block_attn_output_mse.append(ao_diff.mean().item())
            # Per-head breakdown: mean over [B, T] for each head
            n_heads = ao_o.shape[1]
            head_mses = []
            for h in range(n_heads):
                head_mses.append(ao_diff[:, h, :].mean().item())
            per_block_per_head_attn_output_mse.append(head_mses)

        return {
            'per_block_output_mse': per_block_output_mse,
            'per_block_cls_mse': per_block_cls_mse,
            'per_block_patch_mse': per_block_patch_mse,
            'per_block_attn_output_mse': per_block_attn_output_mse,
            'per_block_per_head_attn_output_mse': per_block_per_head_attn_output_mse,
        }

    # ----------------------------------------------------------
    #  6.  Quadratic sensitivity  delta^T G delta
    # ----------------------------------------------------------

    def _ensure_g_diag(self, features_orig, n_probes=50, n_metric_images=100,
                       batch_size=8, cache_path=None):
        """Estimate or load cached G_diag."""
        if self._g_diag_cache is not None:
            return self._g_diag_cache

        if cache_path and os.path.exists(cache_path):
            g_np = np.load(cache_path)
            self._g_diag_cache = torch.from_numpy(g_np).float().to(self.device)
            return self._g_diag_cache

        subset = features_orig[:n_metric_images]
        F_func = ProjectionF(self.tail_blocks, device=self.device)
        info = estimate_task_metric(
            subset, F_func, mode='diag',
            n_probes=n_probes, batch_size=batch_size,
            device=self.device, verbose=True,
        )
        self._g_diag_cache = info['g_diag'].to(self.device)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, self._g_diag_cache.cpu().numpy())
        return self._g_diag_cache

    def _quad_sensitivity(self, features_orig, features_hat, g_diag,
                          batch_size=64):
        """D_quad = E[ delta^T diag(G) delta ] per token."""
        total = 0.0
        count = 0
        N = len(features_orig)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X = torch.from_numpy(
                np.stack(features_orig[start:end])
            ).float().to(self.device)
            Xh = torch.from_numpy(
                np.stack(features_hat[start:end])
            ).float().to(self.device)
            delta = Xh - X
            weighted = (delta ** 2) * g_diag.unsqueeze(0).unsqueeze(0)
            total += weighted.sum().item()
            B, T, _ = X.shape
            count += B * T
            del X, Xh, delta, weighted
        return total / max(count, 1)

    # ----------------------------------------------------------
    #  Unified API
    # ----------------------------------------------------------

    def compute_all_metrics(self, features_orig, features_hat,
                            batch_size=16, n_metric_images=200,
                            g_diag_cache_path=None, n_probes=50):
        """Compute all tail distortion metrics.

        Returns:
            dict with scalar metrics at top level, plus a 'detail' sub-dict
            containing per-block profiling data.
        """
        N = min(len(features_orig), len(features_hat))
        D = features_orig[0].shape[1]

        acc = {
            'attn_kl': 0.0, 'attn_js': 0.0,
            'cls_attn_kl': 0.0, 'cls_attn_js': 0.0, 'cls_attn_l2': 0.0,
            'attn_output_mse': 0.0,
            'rollout_cls_l2': 0.0, 'rollout_cls_kl': 0.0,
            'rollout_patch_fro': 0.0,
            'rollout_watt_cls_l2': 0.0, 'rollout_watt_cls_kl': 0.0,
            'rollout_watt_patch_fro': 0.0,
            'block_output_mse_uniform': 0.0, 'block_output_mse_exp': 0.0,
            'decomp_dA_V_mse': 0.0, 'decomp_A_dV_mse': 0.0,
        }
        # Per-block detail accumulators (will be averaged over batches)
        detail_acc = None
        decomp_detail_acc = None
        n_batches = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            H = torch.from_numpy(
                np.stack(features_orig[start:end])
            ).float().to(self.device)
            Hh = torch.from_numpy(
                np.stack(features_hat[start:end])
            ).float().to(self.device)

            bouts_o, aws_o, aos_o, vs_o = self._forward_tail(H)
            bouts_h, aws_h, aos_h, vs_h = self._forward_tail(Hh)

            # Scalar aggregates
            acc['attn_kl'] += self._attn_divergence(aws_o, aws_h, 'kl')
            acc['attn_js'] += self._attn_divergence(aws_o, aws_h, 'js')
            acc['cls_attn_kl'] += self._cls_attn_divergence(aws_o, aws_h, 'kl')
            acc['cls_attn_js'] += self._cls_attn_divergence(aws_o, aws_h, 'js')
            acc['cls_attn_l2'] += self._cls_attn_l2(aws_o, aws_h)
            acc['attn_output_mse'] += self._attn_output_mse(aos_o, aos_h)
            acc['rollout_cls_l2'] += self._rollout_cls_divergence(
                aws_o, aws_h, 'l2')
            acc['rollout_cls_kl'] += self._rollout_cls_divergence(
                aws_o, aws_h, 'kl')
            acc['rollout_patch_fro'] += self._rollout_patch_frobenius(
                aws_o, aws_h)
            acc['rollout_watt_cls_l2'] += self._rollout_watt_cls_divergence(
                aws_o, aws_h, 'l2')
            acc['rollout_watt_cls_kl'] += self._rollout_watt_cls_divergence(
                aws_o, aws_h, 'kl')
            acc['rollout_watt_patch_fro'] += self._rollout_watt_patch_frobenius(
                aws_o, aws_h)
            acc['block_output_mse_uniform'] += self._block_output_mse(
                bouts_o, bouts_h, 'uniform')
            acc['block_output_mse_exp'] += self._block_output_mse(
                bouts_o, bouts_h, 'exp_inc')

            # Decomposition
            dA_V, A_dV, pb_dA_V, pb_A_dV = self._attn_output_decomposition(
                aws_o, aws_h, vs_o, vs_h)
            acc['decomp_dA_V_mse'] += dA_V
            acc['decomp_A_dV_mse'] += A_dV

            # Per-block detail
            detail_batch = self._compute_per_block_detail(
                bouts_o, bouts_h, aos_o, aos_h)
            if detail_acc is None:
                detail_acc = {
                    k: ([0.0] * len(v) if isinstance(v[0], float)
                        else [[0.0] * len(v[0]) for _ in range(len(v))])
                    for k, v in detail_batch.items()
                }
                decomp_detail_acc = {
                    'per_block_dA_V': [0.0] * len(pb_dA_V),
                    'per_block_A_dV': [0.0] * len(pb_A_dV),
                }
            for k, v in detail_batch.items():
                if isinstance(v[0], float):
                    for i in range(len(v)):
                        detail_acc[k][i] += v[i]
                else:
                    for i in range(len(v)):
                        for j in range(len(v[i])):
                            detail_acc[k][i][j] += v[i][j]
            for i in range(len(pb_dA_V)):
                decomp_detail_acc['per_block_dA_V'][i] += pb_dA_V[i]
                decomp_detail_acc['per_block_A_dV'][i] += pb_A_dV[i]

            n_batches += 1
            del H, Hh, bouts_o, bouts_h, aws_o, aws_h, aos_o, aos_h
            del vs_o, vs_h
            torch.cuda.empty_cache()

            if (start // batch_size) % 10 == 0:
                print(f"    tail metrics: {end}/{N} images")

        results = {k: v / max(n_batches, 1) for k, v in acc.items()}

        # Average per-block detail
        detail = {}
        if detail_acc is not None:
            for k, v in detail_acc.items():
                if isinstance(v[0], float):
                    detail[k] = [x / max(n_batches, 1) for x in v]
                else:
                    detail[k] = [[x / max(n_batches, 1) for x in row]
                                 for row in v]
            for k, v in decomp_detail_acc.items():
                detail[k] = [x / max(n_batches, 1) for x in v]

        # Raw feature MSE
        mse_total = 0.0
        mse_count = 0
        for i in range(N):
            diff = features_hat[i] - features_orig[i]
            mse_total += (diff ** 2).sum()
            mse_count += diff.size
        results['feature_mse'] = float(mse_total / max(mse_count, 1))

        # Quadratic sensitivity + dimensionless variants + G_diag stats
        print("    computing quadratic sensitivity (G_diag)...")
        g_diag = self._ensure_g_diag(
            features_orig, n_probes=n_probes,
            n_metric_images=n_metric_images,
            batch_size=8, cache_path=g_diag_cache_path,
        )
        results['quad_sensitivity'] = self._quad_sensitivity(
            features_orig, features_hat, g_diag, batch_size=64,
        )

        # E: dimensionless quad metrics
        feature_mse_times_D = results['feature_mse'] * D
        trace_G = g_diag.sum().item()
        results['quad_rayleigh'] = (
            results['quad_sensitivity'] / max(feature_mse_times_D, 1e-10))
        results['quad_norm_trace'] = (
            results['quad_sensitivity'] / max(trace_G, 1e-10))

        # E: G_diag statistics
        results['g_diag_mean'] = g_diag.mean().item()
        results['g_diag_std'] = g_diag.std().item()
        results['g_diag_trace'] = trace_G
        results['g_diag_p5'] = torch.quantile(g_diag, 0.05).item()
        results['g_diag_p50'] = torch.quantile(g_diag, 0.50).item()
        results['g_diag_p95'] = torch.quantile(g_diag, 0.95).item()
        results['g_diag_kappa'] = (
            g_diag.max() / g_diag.min().clamp(min=1e-10)).item()

        results['detail'] = detail
        return results
