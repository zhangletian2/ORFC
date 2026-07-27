"""
ORFC-v1 training loop with elastic loss, gradient logging,
per-group diagnostics, and Phase-A / Phase-B / Phase-C protocol support.

Fixes the features_train lifecycle bug (soft_pq.py L554/L783).

Changes from code review (TODO §11):
  11.1  differentiable_groups — elastic loss backprops to U
  11.2  X_hat centre for elasticity probe
  11.3  only sampled groups allocated in GPU memory
  11.4  periodic validation + final test held-out evaluation
  11.6  separate g0 and g_el gradient diagnostics; clipping stats
  11.7  dual-space q_g (normalised + original)
  12    Phase B (joint) / Phase C (alternating) via step_mode param
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys, os
_ORFC_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'orfc'))
if _ORFC_DIR not in sys.path:
    sys.path.insert(0, _ORFC_DIR)

from opq import batch_normalize_gpu, batch_inv_normalize_gpu, batched_kmeans
from soft_pq import SoftPQ, OrthogonalTransform, FeatureTransform, FrozenTail

from codec_v1 import FeatureCodecV1
from elastic import (
    compute_elasticity, elastic_loss, compute_loss_v1,
    GroupSampler, GroupStatsAccumulator,
    compute_quantisation_difficulty, compute_qg_from_rg,
    compute_qg_per_image_from_rg,
    compute_curvature, compute_interaction,
    # V1.1
    pairwise_dispersion_loss,
    energy_match_residuals,
    compute_single_sided_probe,
    fixed_energy_response,
    operational_response,
    scale_nonlinearity,
    compute_response_loss,
    compute_loss_v1_1,
)


# ================================================================
#  Dataset (same as soft_pq.FeatureDataset)
# ================================================================

class FeatureDataset(Dataset):
    def __init__(self, features_array, teacher_cache=None):
        self.data = features_array
        self.teacher_cache = teacher_cache

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = torch.as_tensor(np.array(self.data[idx]), dtype=torch.float32)
        if self.teacher_cache is not None:
            y = torch.as_tensor(
                np.array(self.teacher_cache[idx]), dtype=torch.float32)
            return x, y
        return x


# ================================================================
#  Perplexity helper
# ================================================================

def compute_perplexity(usage):
    p = usage / usage.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    entropy = -(p * (p + 1e-10).log()).sum(dim=-1)
    return entropy.exp().mean().item()


# ================================================================
#  Gradient diagnostics (§11.6)
# ================================================================

def _grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def _collect_grad_vec(params):
    """Concatenate gradients into a single vector."""
    parts = []
    for p in params:
        if p.grad is not None:
            parts.append(p.grad.data.reshape(-1).clone())
    return torch.cat(parts) if parts else None


def _cosine(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    return torch.dot(v1, v2).item() / (v1.norm().item() * v2.norm().item() + 1e-10)


# ================================================================
#  Held-out elasticity evaluation (§11.4)
# ================================================================

@torch.no_grad()
def evaluate_heldout_elasticity(
    features_array, teacher_cache, codec, tail,
    G, alpha, norm_mode, batch_size, device,
    max_images=None,
    probe_group_chunk=4,
):
    """Compute full-group ε_g, q_g, κ_g on a held-out set (§13.2).

    Statistics are accumulated per-image and aggregated at dataset level
    with image-count weighting, so results do not depend on batch size,
    last-batch size, or sample grouping order.
    """
    codec.eval()
    N = features_array.shape[0]
    if max_images is not None:
        N = min(N, max_images)

    probe_group_chunk = max(1, int(probe_group_chunk))
    D0_per_img = []
    Dp_per_img = {g: [] for g in range(G)}
    Dm_per_img = {g: [] for g in range(G)}
    qg_norm_per_img = {g: [] for g in range(G)}
    qg_orig_per_img = {g: [] for g in range(G)}
    interaction_pairs = ([(0, 1), (G // 4, G // 2), (G // 2, G - 1)]
                         if G >= 2 else [])
    interaction_per_img = {pair: [] for pair in interaction_pairs}
    D = features_array.shape[2]

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        X = torch.from_numpy(features_array[start:end]).float().to(device)
        Yt = torch.from_numpy(teacher_cache[start:end]).float().to(device)
        B = X.shape[0]

        Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
        Y_hat, info = codec(
            Y, return_details=True, detail_level='train')
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)

        X_hat_out = tail.forward_nograd(X_hat)
        per_img_d0 = ((Yt - X_hat_out) ** 2).reshape(B, -1).sum(dim=1)
        D0_batch = per_img_d0.sum() / B
        D0_per_img.extend(per_img_d0.cpu().tolist())

        T_tok = X.shape[1]
        r_g = info['r_g']
        if (codec.transform is None
                or not hasattr(codec.transform, 'get_rotation')):
            raise RuntimeError(
                "Compact held-out evaluation currently requires an "
                "OrthogonalTransform")
        R_t = codec.transform.get_rotation().detach().t()
        d = codec.pq.d
        Dplus_img_batch = {}
        pair_groups = {g for pair in interaction_pairs for g in pair}
        pair_e_g = {}

        for g_start in range(0, G, probe_group_chunk):
            g_end = min(g_start + probe_group_chunk, G)
            chunk_groups = list(range(g_start, g_end))
            e_g_dict = {
                g: r_g[g] @ R_t[g*d:(g+1)*d, :]
                for g in chunk_groups
            }

            _, _, _, per_img = compute_elasticity(
                X_hat, e_g_dict, Std, tail, Yt, D0_batch, alpha,
                probe_group_chunk=len(chunk_groups),
                return_per_image=True)

            for g in chunk_groups:
                dp_img = per_img['Dplus_per_image'][g]
                dm_img = per_img['Dminus_per_image'][g]
                Dplus_img_batch[g] = dp_img
                Dp_per_img[g].extend(dp_img.cpu().tolist())
                Dm_per_img[g].extend(dm_img.cpu().tolist())
                if g in pair_groups:
                    pair_e_g[g] = e_g_dict[g]

        qg_result = compute_qg_per_image_from_rg(
            r_g, B=B, T=T_tok, Std=Std)
        for g in range(G):
            qg_norm_per_img[g].extend(
                qg_result['q_g_normalized'][g].cpu().tolist())
            if 'q_g_original' in qg_result:
                qg_orig_per_img[g].extend(
                    qg_result['q_g_original'][g].cpu().tolist())

        if interaction_pairs:
            _, _, ixn_img = compute_interaction(
                X_hat, pair_e_g, Std, tail, Yt,
                per_img_d0, Dplus_img_batch, alpha, interaction_pairs,
                return_per_image=True)
            for pair, values in ixn_img.items():
                interaction_per_img[pair].extend(values.cpu().tolist())

        del X, Yt, Y, Mu, Std, Y_hat, X_hat, X_hat_out, info

    D0_arr = np.array(D0_per_img)
    n_total = len(D0_arr)

    def _dataset_eps_g(g):
        dp = np.array(Dp_per_img[g])
        dm = np.array(Dm_per_img[g])
        d0 = D0_arr
        mean_d0 = d0.mean()
        eps_per_img = (dp - dm) / (2.0 * alpha * (mean_d0 + 1e-8))
        std = float(eps_per_img.std())
        mean = float(eps_per_img.mean())
        half = 1.96 * std / max(np.sqrt(n_total), 1.0)
        return {
            'mean': mean,
            'std': std,
            'cv': float(std / (abs(mean) + 1e-10)),
            'min': float(eps_per_img.min()),
            'max': float(eps_per_img.max()),
            'span': float(eps_per_img.max() - eps_per_img.min()),
            'ci95': [mean - half, mean + half],
            'n': n_total,
        }

    def _dataset_kappa_g(g):
        dp = np.array(Dp_per_img[g])
        dm = np.array(Dm_per_img[g])
        d0 = D0_arr
        mean_d0 = d0.mean()
        kappa_per_img = (dp + dm - 2 * d0) / (alpha ** 2 * (mean_d0 + 1e-8))
        mean = float(kappa_per_img.mean())
        std = float(kappa_per_img.std())
        half = 1.96 * std / max(np.sqrt(n_total), 1.0)
        return {
            'mean': mean,
            'std': std,
            'min': float(kappa_per_img.min()),
            'max': float(kappa_per_img.max()),
            'span': float(kappa_per_img.max() - kappa_per_img.min()),
            'ci95': [mean - half, mean + half],
            'n': n_total,
        }

    eps_result = {}
    kappa_result = {}
    for g in range(G):
        eps_result[g] = _dataset_eps_g(g)
        kappa_result[g] = _dataset_kappa_g(g)

    eps_means = np.array([eps_result[g]['mean'] for g in range(G)])
    kappa_means = np.array([kappa_result[g]['mean'] for g in range(G)])

    def _build_summary(per_group, means):
        return {
            'per_group': per_group,
            'global_mean': float(means.mean()),
            'global_std': float(means.std()),
            'global_cv': float(means.std() / (abs(means.mean()) + 1e-10)),
            'global_span': float(means.max() - means.min()),
            'n_images': n_total,
        }

    result = {
        'eps_g': _build_summary(eps_result, eps_means),
        'kappa_g': _build_summary(kappa_result, kappa_means),
    }

    def _moment_row(values):
        arr = np.asarray(values, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std())
        half = 1.96 * std / max(np.sqrt(arr.size), 1.0)
        return {
            'mean': mean,
            'std': std,
            'ci95': [mean - half, mean + half],
            'n': int(arr.size),
        }

    qg_norm_means = np.array([np.mean(qg_norm_per_img[g])
                               for g in range(G)])
    result['q_g_normalized'] = {
        'per_group': {g: _moment_row(qg_norm_per_img[g])
                      for g in range(G)},
        'global_mean': float(qg_norm_means.mean()),
        'global_std': float(qg_norm_means.std()),
        'n_images': n_total,
    }
    if qg_orig_per_img[0]:
        qg_orig_means = np.array([np.mean(qg_orig_per_img[g])
                                   for g in range(G)])
        result['q_g_original'] = {
            'per_group': {g: _moment_row(qg_orig_per_img[g])
                          for g in range(G)},
            'global_mean': float(qg_orig_means.mean()),
            'global_std': float(qg_orig_means.std()),
            'n_images': n_total,
        }

    if interaction_pairs:
        per_pair = {}
        all_norm = []
        mean_d0 = D0_arr.mean()
        for pair in interaction_pairs:
            arr = np.asarray(interaction_per_img[pair])
            norm = arr / (mean_d0 + 1e-8)
            all_norm.extend(norm.tolist())
            per_pair[f'{pair[0]}-{pair[1]}'] = {
                'mean': float(norm.mean()),
                'std': float(norm.std()),
                'max_abs': float(np.abs(norm).max()),
                'n': int(norm.size),
            }
        ixn_arr = np.asarray(all_norm)
        result['interaction'] = {
            'mean': float(ixn_arr.mean()),
            'std': float(ixn_arr.std()),
            'max_abs': float(np.abs(ixn_arr).max()),
            'n_samples': len(ixn_arr),
            'per_pair': per_pair,
        }
    batches = int(math.ceil(N / batch_size))
    result['performance_contract'] = {
        'n_images': n_total,
        'n_batches': batches,
        'probe_group_chunk': probe_group_chunk,
        'elasticity_tail_calls_per_batch': int(math.ceil(
            G / probe_group_chunk)),
        'interaction_tail_calls_per_batch': 1 if interaction_pairs else 0,
        'total_tail_calls_per_batch': (
            1 + int(math.ceil(G / probe_group_chunk))
            + (1 if interaction_pairs else 0)),
        'group_by_image_tail_loop': False,
    }
    return result


# ================================================================
#  V1.1 comprehensive held-out evaluation (§15.3 / §15.7)
# ================================================================

@torch.no_grad()
def evaluate_heldout_v1_1(
    features_array, teacher_cache, codec, tail,
    G, norm_mode, batch_size, device,
    alphas=None,
    max_images=None,
    probe_group_chunk=4,
):
    """Compute full-group S_g, M_g, N_g, q_g, κ_g on a held-out set.

    Unlike ``evaluate_heldout_elasticity``, this computes:
      - Fixed-energy response S_g^(α) for all groups and alphas
      - Operational repair response M_g^(α)
      - Scale nonlinearity N_g
      - Per-image q_g (normalised and original)
      - Legacy κ_g (from bilateral ±α probe with α=min(alphas))

    All statistics are accumulated per-image with image-count weighting.
    """
    if alphas is None:
        alphas = [0.1, 0.5, 1.0]
    alpha_ref = min(alphas)

    codec.eval()
    N = features_array.shape[0]
    if max_images is not None:
        N = min(N, max_images)

    D_dim = features_array.shape[2]
    T_tok = features_array.shape[1]

    D0_per_img = []
    # Per-group, per-alpha accumulators
    Dfe_per_img = {(a, g): [] for a in alphas for g in range(G)}
    Dop_per_img = {(a, g): [] for a in alphas for g in range(G)}
    qg_norm_per_img = {g: [] for g in range(G)}
    qg_orig_per_img = {g: [] for g in range(G)}
    Dp_per_img = {g: [] for g in range(G)}
    Dm_per_img = {g: [] for g in range(G)}
    interaction_pairs = (
        [(0, 1), (G // 4, G // 2), (G // 2, G - 1)]
        if G >= 2 else [])
    interaction_per_img = {pair: [] for pair in interaction_pairs}
    em_audits = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        X = torch.from_numpy(features_array[start:end]).float().to(device)
        Yt = torch.from_numpy(teacher_cache[start:end]).float().to(device)
        B = X.shape[0]

        Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
        Y_hat, info = codec(
            Y, return_details=True, detail_level='train')
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)

        X_hat_out = tail.forward_nograd(X_hat)
        per_img_d0 = ((Yt - X_hat_out) ** 2).reshape(B, -1).sum(dim=1)
        D0_batch = per_img_d0.mean()
        D0_per_img.extend(per_img_d0.cpu().tolist())

        r_g = info['r_g']
        if not (codec.transform is not None
                and hasattr(codec.transform, 'get_rotation')):
            raise RuntimeError(
                "V1.1 held-out evaluation requires OrthogonalTransform")
        R_t = codec.transform.get_rotation().detach().t()
        d = codec.pq.d

        # The fixed-energy target is the per-image mean over all G groups.
        # Compute it once so it cannot depend on probe chunking.
        qg_result = compute_qg_per_image_from_rg(
            r_g, B=B, T=T_tok, Std=Std)
        q_bar_all = qg_result['q_g_original'].mean(dim=0).detach()
        for g in range(G):
            qg_norm_per_img[g].extend(
                qg_result['q_g_normalized'][g].cpu().tolist())
            qg_orig_per_img[g].extend(
                qg_result['q_g_original'][g].cpu().tolist())

        for g_start in range(0, G, probe_group_chunk):
            g_end = min(g_start + probe_group_chunk, G)
            chunk_groups = list(range(g_start, g_end))

            e_g_dict = {
                g: r_g[g] @ R_t[g*d:(g+1)*d, :]
                for g in chunk_groups
            }

            # Energy matching for fixed-energy probe
            e_g_matched, q_g_img, q_bar, em_audit = \
                energy_match_residuals(
                    e_g_dict, B, T_tok,
                    q_bar_per_image=q_bar_all, Std=Std)
            em_audits.append(em_audit)

            # Fixed-energy single-sided probe
            D_fe, D_fe_img = compute_single_sided_probe(
                X_hat, e_g_matched, Std, tail, Yt,
                alphas, probe_chunk=len(chunk_groups) * len(alphas))

            # Operational single-sided probe
            D_op, D_op_img = compute_single_sided_probe(
                X_hat, e_g_dict, Std, tail, Yt,
                alphas, probe_chunk=len(chunk_groups) * len(alphas))

            for g in chunk_groups:
                for a in alphas:
                    Dfe_per_img[(a, g)].extend(
                        D_fe_img[(a, g)].cpu().tolist())
                    Dop_per_img[(a, g)].extend(
                        D_op_img[(a, g)].cpu().tolist())

            # Legacy bilateral probe for κ_g (using alpha_ref)
            _, _, _, legacy_per_img = compute_elasticity(
                X_hat, e_g_dict, Std, tail, Yt, D0_batch,
                alpha_ref,
                probe_group_chunk=len(chunk_groups),
                return_per_image=True)
            for g in chunk_groups:
                Dp_per_img[g].extend(
                    legacy_per_img['Dplus_per_image'][g].cpu().tolist())
                Dm_per_img[g].extend(
                    legacy_per_img['Dminus_per_image'][g].cpu().tolist())

        # Interaction
        if interaction_pairs:
            pair_groups = {g for p in interaction_pairs for g in p}
            pair_e_g = {
                g: r_g[g] @ R_t[g*d:(g+1)*d, :]
                for g in pair_groups
            }
            Dplus_ixn = {}
            for g in pair_groups:
                eg_o = pair_e_g[g].reshape(B, T_tok, D_dim) * Std
                x_p = X_hat + alpha_ref * eg_o
                y_p = tail.forward_nograd(x_p)
                dp_img = ((Yt - y_p) ** 2).reshape(B, -1).sum(dim=1)
                Dplus_ixn[g] = dp_img
            _, _, ixn_img = compute_interaction(
                X_hat, pair_e_g, Std, tail, Yt,
                per_img_d0, Dplus_ixn, alpha_ref, interaction_pairs,
                return_per_image=True)
            for pair, values in ixn_img.items():
                interaction_per_img[pair].extend(values.cpu().tolist())

        del X, Yt, Y, Mu, Std, Y_hat, X_hat, X_hat_out, info

    D0_arr = np.array(D0_per_img)
    n_total = len(D0_arr)
    mean_d0 = D0_arr.mean()

    # Convert raw per-image probe distortions after the dataset-wide mean D0
    # is known.  Per-batch normalisation would make the result depend on batch
    # size and sample ordering.
    S_per_img = {}
    M_per_img = {}
    for a in alphas:
        if a <= 0:
            raise ValueError("Response statistics require alpha > 0")
        for g in range(G):
            dfe = np.asarray(Dfe_per_img[(a, g)], dtype=np.float64)
            dop = np.asarray(Dop_per_img[(a, g)], dtype=np.float64)
            S_per_img[(a, g)] = (
                (D0_arr - dfe) / (a * (mean_d0 + 1e-8)))
            M_per_img[(a, g)] = (
                (D0_arr - dop) / (mean_d0 + 1e-8))

    def _moment_row(values):
        arr = np.asarray(values, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std())
        half = 1.96 * std / max(np.sqrt(arr.size), 1.0)
        return {
            'mean': mean, 'std': std,
            'ci95': [mean - half, mean + half],
            'n': int(arr.size),
        }

    # S_g and M_g per alpha
    result = {'n_images': n_total, 'alphas': alphas}
    for prefix, store in [('S_g', S_per_img), ('M_g', M_per_img)]:
        for a in alphas:
            per_group = {}
            means = []
            for g in range(G):
                row = _moment_row(store[(a, g)])
                per_group[g] = row
                means.append(row['mean'])
            means_arr = np.array(means)
            pairwise = float(np.var(means_arr) * 2 * G / (G - 1)) \
                if G > 1 else 0.0
            result[f'{prefix}_alpha{a}'] = {
                'per_group': per_group,
                'global_mean': float(means_arr.mean()),
                'global_std': float(means_arr.std()),
                'global_span': float(means_arr.max() - means_arr.min()),
                'global_mad': float(np.median(np.abs(
                    means_arr - np.median(means_arr)))),
                'min_group': int(np.argmin(means_arr)),
                'max_group': int(np.argmax(means_arr)),
                'pairwise_dispersion': pairwise,
                'n_images': n_total,
            }

    def _nonlinearity_rows(store, already_divided_by_alpha):
        ref_alpha = min(alphas)
        rows = {}
        for g in range(G):
            ref = np.asarray(store[(ref_alpha, g)], dtype=np.float64)
            if not already_divided_by_alpha:
                ref = ref / ref_alpha
            deviations = []
            for a in alphas:
                values = np.asarray(store[(a, g)], dtype=np.float64)
                if not already_divided_by_alpha:
                    values = values / a
                deviations.append(np.abs(values - ref))
            rows[g] = _moment_row(np.max(np.stack(deviations), axis=0))
        means = np.asarray([rows[g]['mean'] for g in range(G)])
        return rows, {
            'mean': float(means.mean()),
            'std': float(means.std()),
            'max': float(means.max()),
            'max_group': int(np.argmax(means)),
        }

    # Operational M does not divide by alpha; fixed-energy S already does.
    N_rows, N_summary = _nonlinearity_rows(
        M_per_img, already_divided_by_alpha=False)
    Ns_rows, Ns_summary = _nonlinearity_rows(
        S_per_img, already_divided_by_alpha=True)
    result['N_g'] = {g: float(N_rows[g]['mean']) for g in range(G)}
    result['N_g_stats'] = {'per_group': N_rows, **N_summary}
    result['N_g_summary'] = N_summary
    result['N_fixed_g'] = {
        g: float(Ns_rows[g]['mean']) for g in range(G)}
    result['N_fixed_g_stats'] = {'per_group': Ns_rows, **Ns_summary}

    # κ_g from bilateral probe
    eps_result = {}
    kappa_result = {}
    for g in range(G):
        dp = np.array(Dp_per_img[g])
        dm = np.array(Dm_per_img[g])
        eps_per_img = (dp - dm) / (2.0 * alpha_ref * (mean_d0 + 1e-8))
        kappa_per_img = (dp + dm - 2 * D0_arr) / (
            alpha_ref ** 2 * (mean_d0 + 1e-8))
        eps_result[g] = _moment_row(eps_per_img)
        kappa_result[g] = _moment_row(kappa_per_img)

    eps_means = np.array([eps_result[g]['mean'] for g in range(G)])
    kappa_means = np.array([kappa_result[g]['mean'] for g in range(G)])
    result['legacy_eps_g'] = {
        'per_group': eps_result,
        'global_mean': float(eps_means.mean()),
        'global_std': float(eps_means.std()),
        'global_span': float(eps_means.max() - eps_means.min()),
        'alpha': alpha_ref,
    }
    result['kappa_g'] = {
        'per_group': kappa_result,
        'global_mean': float(kappa_means.mean()),
        'global_std': float(kappa_means.std()),
        'global_span': float(kappa_means.max() - kappa_means.min()),
        'alpha': alpha_ref,
    }

    # q_g
    qg_norm_means = np.array([np.mean(qg_norm_per_img[g])
                               for g in range(G)])
    result['q_g_normalized'] = {
        'per_group': {g: _moment_row(qg_norm_per_img[g])
                      for g in range(G)},
        'global_mean': float(qg_norm_means.mean()),
        'global_std': float(qg_norm_means.std()),
        'n_images': n_total,
    }
    if qg_orig_per_img[0]:
        qg_orig_means = np.array([np.mean(qg_orig_per_img[g])
                                   for g in range(G)])
        result['q_g_original'] = {
            'per_group': {g: _moment_row(qg_orig_per_img[g])
                          for g in range(G)},
            'global_mean': float(qg_orig_means.mean()),
            'global_std': float(qg_orig_means.std()),
            'n_images': n_total,
        }

    # Interaction
    if interaction_pairs:
        per_pair = {}
        all_norm = []
        for pair in interaction_pairs:
            arr = np.asarray(interaction_per_img[pair])
            norm = arr / (mean_d0 + 1e-8)
            all_norm.extend(norm.tolist())
            per_pair[f'{pair[0]}-{pair[1]}'] = {
                'mean': float(norm.mean()),
                'std': float(norm.std()),
                'max_abs': float(np.abs(norm).max()),
                'ci95': _moment_row(norm)['ci95'],
                'n': int(norm.size),
            }
        ixn_arr = np.asarray(all_norm)
        result['interaction'] = {
            'mean': float(ixn_arr.mean()),
            'std': float(ixn_arr.std()),
            'max_abs': float(np.abs(ixn_arr).max()),
            'ci95': _moment_row(ixn_arr)['ci95'],
            'n_samples': len(ixn_arr),
            'per_pair': per_pair,
        }

    # Energy match audit
    result['energy_match_audit_max'] = float(max(em_audits)) \
        if em_audits else 0.0

    # D0 summary
    result['D0_raw_mean'] = float(mean_d0)
    result['D0_per_element'] = float(mean_d0 / (T_tok * D_dim))

    return result


# ================================================================
#  Main training function
# ================================================================

def train_v1(
    features_array,     # [N, T, D] numpy (already stacked!)
    T_tokens,           # int — number of tokens per image
    tail: FrozenTail,
    G, K, d,
    norm_mode='per_image',
    epochs=100,
    lr=1e-3,
    batch_size=4,
    device='cuda',
    seeds=None,
    val_array=None,     # [N_val, T, D] numpy or None
    verbose=True,
    # Initialisation
    transform=None,
    R_init=None,
    codebooks_init=None,
    kmeans_max_samples=2_000_000,
    # V1 elastic loss
    beta=0.0,
    alpha=0.1,
    elastic_tau=1.0,
    n_groups_per_batch=4,
    # Diagnostics
    compute_diagnostics=True,
    n_interaction_pairs=4,
    # Temperature
    tau_start=0.5,
    tau_end=0.005,
    tau_schedule='exponential',
    # Freezing
    freeze_transform=False,
    freeze_codebooks=False,
    # Rate (disabled in first round)
    lmbda=0.0,
    prior_floor=0.0,
    # Misc
    grad_clip=1.0,
    # Phase C: alternating training (§12)
    step_mode='joint',       # 'joint' | 'alternating'
    alt_u_steps=1,
    alt_c_steps=1,
    # Held-out evaluation interval (§11.4)
    val_elasticity_interval=20,
    # Batched tail probe (§13.3b)
    probe_group_chunk=4,
    # Held-out can use a larger no-grad chunk than the training probe.
    heldout_probe_group_chunk=4,
    # Pre-computed caches (avoid redundant computation across processes)
    teacher_cache_precomputed=None,
    val_teacher_cache_precomputed=None,
    # V1.1: response objective (§15)
    response_objective='legacy',    # 'legacy' | 'fixed_energy' | 'operational'
    alpha_list=None,                # multi-alpha list, e.g. [0.1, 0.5, 1.0]
    alpha_weights=None,             # per-alpha weights for response loss
    d0_normalize='raw',             # 'raw' (v1) | 'per_element' (v1.1 §15.2)
    beta_target_ratio=0.0,          # calibrate beta on first minibatch
):
    """Train FeatureCodecV1 with elastic loss and diagnostics.

    FIX vs original soft_pq.train_soft_pq:
      - features_array is pre-stacked numpy; T_tokens passed explicitly
        (avoids the del-then-access bug).
      - Returns comprehensive per-epoch diagnostics.

    Parameters (cache)
    ------------------
    teacher_cache_precomputed : numpy array or None
        If provided, skip internal teacher cache computation (shared artifact).
    val_teacher_cache_precomputed : numpy array or None
        If provided, skip internal val teacher cache computation.

    Returns
    -------
    codec   : FeatureCodecV1
    history : list[dict]
    """
    device = torch.device(device)
    seed_init = seeds['init'] if seeds else 42
    seed_mb = seeds['minibatch'] if seeds else 42
    seed_gs = seeds['group_sampling'] if seeds else 42

    torch.manual_seed(seed_init)
    np.random.seed(seed_init)

    if isinstance(device, str):
        device = torch.device(device)

    # V1.1 defaults
    if alpha_list is None:
        alpha_list = [alpha] if response_objective == 'legacy' else [0.1, 0.5, 1.0]
    if alpha_weights is None:
        alpha_weights = [1.0 / len(alpha_list)] * len(alpha_list)
    _is_v1_1 = response_objective != 'legacy'

    N_img = features_array.shape[0]
    D = features_array.shape[2]

    # Build codec
    pq = SoftPQ(G, K, d, lmbda=lmbda, prior_floor=prior_floor).to(device)
    if transform is not None:
        transform = transform.to(device)
    codec = FeatureCodecV1(pq, transform).to(device)

    _use_soft = (tau_start > 0)
    _use_elastic = (beta > 0 or beta_target_ratio > 0)
    effective_beta = float(beta)
    beta_calibration = None

    # Initialisation
    if R_init is not None and codebooks_init is not None and transform is not None:
        if verbose:
            print(f"  Warm-start from OPQ (transform + codebooks)")
        transform.init_from_opq(R_init)
        pq.init_codebooks(codebooks_init)
    elif codebooks_init is not None:
        if verbose:
            print(f"  Warm-start codebooks only")
        pq.init_codebooks(codebooks_init)
    else:
        if verbose:
            print(f"  K-means init ({N_img} images)...")
        all_Z = []
        for start in range(0, N_img, 200):
            end = min(start + 200, N_img)
            X = torch.from_numpy(features_array[start:end]).float().to(device)
            with torch.no_grad():
                Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
                flat = Y.reshape(-1, D)
                Z = transform.encode(flat) if transform else flat
            all_Z.append(Z.cpu())
            del X, Y, flat, Z
        Z_flat = torch.cat(all_Z, dim=0)
        max_km = kmeans_max_samples
        if Z_flat.shape[0] > max_km:
            idx = np.random.choice(Z_flat.shape[0], max_km, replace=False)
            Z_flat = Z_flat[idx]
        pq.init_from_kmeans(Z_flat, device=device)
        del all_Z, Z_flat
        torch.cuda.empty_cache()

    # Freeze parameters
    if freeze_transform and transform is not None:
        for p in transform.parameters():
            p.requires_grad_(False)
        if verbose:
            n_p = sum(p.numel() for p in transform.parameters())
            print(f"  Frozen: transform ({n_p:,} params)")
    if freeze_codebooks:
        pq.codebooks.requires_grad_(False)
        if pq.use_rate:
            pq.log_prior.requires_grad_(False)
        if verbose:
            print(f"  Frozen: codebooks ({pq.codebooks.numel():,} params)")

    trainable = [p for p in codec.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01)

    # Separate param groups for gradient decomposition (§11.6)
    transform_params = [p for p in (transform.parameters() if transform else [])
                        if p.requires_grad]
    codebook_params = [p for p in [pq.codebooks] if p.requires_grad]

    if verbose:
        n_trainable = sum(p.numel() for p in trainable)
        print(f"  Trainable params: {n_trainable:,}")
        if transform and hasattr(transform, 'orth_error'):
            print(f"  ||R'R-I||_F = {transform.orth_error():.2e}")
        if _use_soft:
            print(f"  Soft PQ: τ {tau_start:.2f} → {tau_end:.4f} ({tau_schedule})")
        if _use_elastic:
            if _is_v1_1:
                print(f"  Response: objective={response_objective}, "
                      f"β={beta}, alphas={alpha_list}, "
                      f"weights={alpha_weights}, "
                      f"groups/batch={n_groups_per_batch}")
                if beta_target_ratio > 0:
                    print(f"  β calibration target gradient ratio: "
                          f"{beta_target_ratio}")
                print(f"  D0 normalisation: {d0_normalize}")
            else:
                print(f"  Elastic: β={beta}, α={alpha}, τ_el={elastic_tau}, "
                      f"groups/batch={n_groups_per_batch}")
        else:
            print(f"  β=0 (D₀ only, no elastic loss)")
        print(f"  Step mode: {step_mode}")

    # Teacher cache (use pre-computed if available)
    if teacher_cache_precomputed is not None:
        teacher_cache = teacher_cache_precomputed
        if verbose:
            print(f"  Teacher cache: pre-computed "
                  f"({teacher_cache.nbytes / 1e9:.1f} GB)")
    else:
        if verbose:
            print(f"  Pre-computing teacher outputs ({N_img} images)...")
        t_pre = time.time()
        teacher_cache = np.empty_like(features_array)
        with torch.no_grad():
            for start in range(0, N_img, batch_size):
                end = min(start + batch_size, N_img)
                X_chunk = torch.from_numpy(
                    features_array[start:end]).float().to(device)
                teacher_cache[start:end] = tail.forward_nograd(
                    X_chunk).cpu().numpy()
                del X_chunk
        torch.cuda.empty_cache()
        if verbose:
            print(f"  Teacher cache: {teacher_cache.nbytes / 1e9:.1f} GB "
                  f"({time.time() - t_pre:.1f}s)")

    # Validation teacher cache (use pre-computed if available)
    if val_teacher_cache_precomputed is not None:
        val_teacher_cache = val_teacher_cache_precomputed
    elif val_array is not None:
        val_teacher_cache = None
        n_val = val_array.shape[0]
        val_teacher_cache = np.empty_like(val_array)
        with torch.no_grad():
            for vs in range(0, n_val, batch_size):
                ve = min(vs + batch_size, n_val)
                X_v = torch.from_numpy(val_array[vs:ve]).float().to(device)
                val_teacher_cache[vs:ve] = tail.forward_nograd(
                    X_v).cpu().numpy()
                del X_v
        torch.cuda.empty_cache()
    else:
        val_teacher_cache = None

    # DataLoader
    g_mb = torch.Generator()
    g_mb.manual_seed(seed_mb)
    train_dataset = FeatureDataset(features_array, teacher_cache=teacher_cache)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True,
        generator=g_mb,
    )

    group_sampler = GroupSampler(G, n_groups_per_batch, seed=seed_gs)

    history = []

    for epoch in range(epochs):
        t_epoch = time.time()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        # Temperature annealing
        if _use_soft and epochs > 1:
            progress = epoch / (epochs - 1)
            if tau_schedule == 'linear':
                tau = tau_start + (tau_end - tau_start) * progress
            else:
                tau = tau_start * (tau_end / tau_start) ** progress
            pq.temperature = tau
        elif _use_soft:
            pq.temperature = tau_start
        else:
            pq.temperature = 0.0

        total_D0_raw = 0.0
        total_D0 = 0.0
        total_response = 0.0
        total_loss = 0.0
        total_tokens = 0
        usage_acc = torch.zeros(G, K, device=device)

        eps_g_acc = GroupStatsAccumulator(G)
        kappa_g_acc = GroupStatsAccumulator(G)
        qg_norm_acc = GroupStatsAccumulator(G)
        interaction_values = []
        # V1.1 response accumulators (per-alpha per-group)
        response_acc = {a: GroupStatsAccumulator(G) for a in alpha_list} if _is_v1_1 else {}
        em_audit_max = 0.0

        # Gradient diagnostics (§11.6 / §15.2)
        grad_d0_norms = []
        grad_response_norms = []
        grad_response_raw_norms = []
        grad_cosines = []
        grad_ratios = []
        clip_counts = 0
        clip_total = 0
        clip_pre_norms = []
        clip_post_norms = []

        group_sampler.reset_epoch()

        codec.train()

        # Phase C: alternating step counter
        _alt_counter = 0
        _batch_idx = 0

        for batch in train_loader:
            X, Y_teacher = batch
            X = X.to(device, non_blocking=True)
            Y_teacher = Y_teacher.to(device, non_blocking=True)
            B = X.shape[0]

            # Phase C alternating: toggle what is trainable
            if step_mode == 'alternating':
                total_alt = alt_u_steps + alt_c_steps
                phase_pos = _alt_counter % total_alt
                if phase_pos < alt_u_steps:
                    # U step: train transform, freeze codebooks
                    for p in transform_params:
                        p.requires_grad_(True)
                    for p in codebook_params:
                        p.requires_grad_(False)
                else:
                    # C step: freeze transform, train codebooks
                    for p in transform_params:
                        p.requires_grad_(False)
                    for p in codebook_params:
                        p.requires_grad_(True)
                _alt_counter += 1

            with torch.no_grad():
                Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)

            # Determine which groups need differentiable residuals
            group_subset = group_sampler.sample() if (_use_elastic or compute_diagnostics) else []
            diff_groups = group_subset if _use_elastic else None

            need_details = _use_elastic or compute_diagnostics
            dl = 'train' if need_details else 'diagnostic'
            Y_hat, info = codec(Y, return_details=need_details,
                                differentiable_groups=diff_groups,
                                detail_level=dl)

            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
            X_hat_out = tail(X_hat)
            D0_raw = ((Y_teacher - X_hat_out) ** 2).sum() / B
            D0_per_elem = D0_raw / (T_tokens * D)
            D0 = D0_per_elem if d0_normalize == 'per_element' else D0_raw

            L_response = torch.tensor(0.0, device=device)
            if _use_elastic and need_details:
                e_g_diff = info['e_g_diff']  # {g: [N, D]} with grad

                if response_objective == 'legacy':
                    eps_g_dict, Dplus, Dminus = compute_elasticity(
                        X_hat, e_g_diff, Std, tail, Y_teacher,
                        D0_raw, alpha,
                        probe_group_chunk=probe_group_chunk)
                    L_el = elastic_loss(eps_g_dict, elastic_tau)
                    L_response = L_el
                    if d0_normalize == 'per_element':
                        loss = D0 + beta * L_response
                    else:
                        loss = compute_loss_v1(D0_raw, L_el, beta)
                    eps_g_acc.add(
                        {g: v.detach() for g, v in eps_g_dict.items()})

                    if compute_diagnostics:
                        kappa = compute_curvature(
                            Dplus, Dminus, D0_raw, alpha)
                        kappa_g_acc.add(kappa)

                elif response_objective == 'fixed_energy':
                    with torch.no_grad():
                        qg_all = compute_qg_per_image_from_rg(
                            info['r_g'], B=B, T=T_tokens, Std=Std)
                        q_bar_all = qg_all['q_g_original'].mean(
                            dim=0).detach()
                    e_g_matched, q_g_img, q_bar, em_audit = \
                        energy_match_residuals(
                            e_g_diff, B, T_tokens,
                            q_bar_per_image=q_bar_all, Std=Std)
                    em_audit_max = max(em_audit_max, em_audit)
                    _probe_chunk = max(
                        1, probe_group_chunk * len(alpha_list))
                    D_probe, _ = compute_single_sided_probe(
                        X_hat, e_g_matched, Std, tail, Y_teacher,
                        alpha_list, probe_chunk=_probe_chunk)
                    S = fixed_energy_response(
                        D0_raw, D_probe, alpha_list, group_subset)
                    L_response, per_alpha_loss = compute_response_loss(
                        S, alpha_list, group_subset,
                        alpha_weights=alpha_weights)
                    loss = D0 + beta * L_response

                    for a in alpha_list:
                        response_acc[a].add(
                            {g: S[(a, g)].detach()
                             for g in group_subset})

                elif response_objective == 'operational':
                    _probe_chunk = max(
                        1, probe_group_chunk * len(alpha_list))
                    D_probe, _ = compute_single_sided_probe(
                        X_hat, e_g_diff, Std, tail, Y_teacher,
                        alpha_list, probe_chunk=_probe_chunk)
                    M = operational_response(
                        D0_raw, D_probe, alpha_list, group_subset)
                    L_response, per_alpha_loss = compute_response_loss(
                        M, alpha_list, group_subset,
                        alpha_weights=alpha_weights)
                    loss = D0 + beta * L_response

                    for a in alpha_list:
                        response_acc[a].add(
                            {g: M[(a, g)].detach()
                             for g in group_subset})

                if beta_target_ratio > 0 and beta_calibration is None:
                    if d0_normalize != 'per_element':
                        raise ValueError(
                            "beta_target_ratio requires per_element D0")
                    optimizer.zero_grad()
                    D0.backward(retain_graph=True)
                    g0_cal = _collect_grad_vec(trainable)
                    gn0_cal = (
                        g0_cal.norm().item() if g0_cal is not None else 0.0)
                    optimizer.zero_grad()
                    L_response.backward(retain_graph=True)
                    gr_cal = _collect_grad_vec(trainable)
                    gnr_cal = (
                        gr_cal.norm().item() if gr_cal is not None else 0.0)
                    if gn0_cal <= 0 or gnr_cal <= 0:
                        raise RuntimeError(
                            "Cannot calibrate beta from zero gradient norm: "
                            f"g_D0={gn0_cal}, g_response={gnr_cal}")
                    effective_beta = (
                        beta_target_ratio * gn0_cal / gnr_cal)
                    beta_calibration = {
                        'target_ratio': float(beta_target_ratio),
                        'grad_D0_norm': float(gn0_cal),
                        'grad_response_raw_norm': float(gnr_cal),
                        'effective_beta': float(effective_beta),
                        'epoch': int(epoch),
                        'batch': int(_batch_idx),
                    }
                    optimizer.zero_grad()
                    if verbose:
                        print(
                            f"  Calibrated β={effective_beta:.8g} "
                            f"for target ratio={beta_target_ratio:.3f}")

                if beta_target_ratio > 0:
                    loss = D0 + effective_beta * L_response

                # Interaction: only on calibration batches (§13.3)
                if (compute_diagnostics
                        and n_interaction_pairs > 0
                        and len(group_subset) >= 2
                        and _batch_idx < 3):
                    if response_objective == 'legacy':
                        _Dplus_ixn = Dplus
                    else:
                        with torch.no_grad():
                            _Dplus_ixn = {}
                            for g in group_subset:
                                eg_o = e_g_diff[g].detach().reshape(
                                    B, T_tokens, D) * Std
                                x_p = X_hat.detach() + alpha * eg_o
                                y_p = tail.forward_nograd(x_p)
                                dp_img = ((Y_teacher - y_p) ** 2
                                          ).reshape(B, -1).sum(dim=1)
                                _Dplus_ixn[g] = dp_img.mean()
                    pairs = []
                    gs = list(group_subset)
                    for i_p in range(
                            min(n_interaction_pairs, len(gs) - 1)):
                        pairs.append(
                            (gs[i_p], gs[(i_p + 1) % len(gs)]))
                    with torch.no_grad():
                        e_g_ixn = {
                            g: info['e_g_diff'][g].detach()
                            for g in set(gs)
                        }
                        ixn, ixn_norm = compute_interaction(
                            X_hat.detach(), e_g_ixn, Std,
                            tail, Y_teacher, D0_raw, _Dplus_ixn,
                            alpha, pairs)
                    interaction_values.extend(ixn_norm.values())
            else:
                loss = D0

            # Quantisation difficulty (§13.3: r_g-based for normalised space)
            if need_details and compute_diagnostics:
                with torch.no_grad():
                    qg_norm = compute_qg_from_rg(info['r_g'])
                    qg_norm_acc.add({g: qg_norm[g] for g in range(G)})

            # Gradient diagnostics (§11.6 / §15.2): 1-3 batches per
            # diagnostic epoch (every 10 epochs)
            do_grad_diag = (_use_elastic
                            and (epoch % 10 == 0)
                            and _batch_idx < 3)
            if do_grad_diag:
                optimizer.zero_grad()
                D0.backward(retain_graph=True)
                g0_vec = _collect_grad_vec(trainable)
                gn_d0 = g0_vec.norm().item() if g0_vec is not None else 0.0

                optimizer.zero_grad()
                (effective_beta * L_response).backward(retain_graph=True)
                gr_vec = _collect_grad_vec(trainable)
                gn_r = gr_vec.norm().item() if gr_vec is not None else 0.0

                optimizer.zero_grad()
                L_response.backward(retain_graph=True)
                gr_raw_vec = _collect_grad_vec(trainable)
                gn_r_raw = gr_raw_vec.norm().item() if gr_raw_vec is not None else 0.0

                cos_angle = _cosine(g0_vec, gr_vec)
                ratio = gn_r / (gn_d0 + 1e-10)

                grad_d0_norms.append(gn_d0)
                grad_response_norms.append(gn_r)
                grad_response_raw_norms.append(gn_r_raw)
                grad_cosines.append(cos_angle)
                grad_ratios.append(ratio)

                optimizer.zero_grad()
                loss.backward()
            else:
                optimizer.zero_grad()
                loss.backward()

            # Gradient clipping with stats (§11.6 / §15.2)
            clip_total += 1
            if grad_clip > 0:
                pre_clip_norm = _grad_norm(trainable)
                torch.nn.utils.clip_grad_norm_(codec.parameters(), grad_clip)
                post_clip_norm = _grad_norm(trainable)
                clip_pre_norms.append(pre_clip_norm)
                clip_post_norms.append(post_clip_norm)
                if pre_clip_norm > grad_clip:
                    clip_counts += 1

            optimizer.step()

            total_D0_raw += D0_raw.item() * B
            total_D0 += D0.item() * B
            total_response += L_response.item() * B if _use_elastic else 0.0
            total_loss += loss.item() * B
            total_tokens += B * T_tokens
            if need_details:
                usage_acc += info['usage'].detach()

            _batch_idx += 1
            del X, Y, Mu, Std, Y_hat, X_hat, X_hat_out, Y_teacher
            del info, loss, D0, D0_raw, D0_per_elem

        # Restore all params' requires_grad after alternating epoch
        if step_mode == 'alternating':
            for p in transform_params:
                p.requires_grad_(not freeze_transform)
            for p in codebook_params:
                p.requires_grad_(not freeze_codebooks)

        scheduler.step()

        avg_D0_raw = total_D0_raw / N_img
        avg_D0 = total_D0 / N_img
        avg_response = total_response / N_img if _use_elastic else 0.0
        avg_loss = total_loss / N_img
        perplexity = compute_perplexity(usage_acc)
        dead_entries = int((usage_acc == 0).sum().item())

        # Validation D0
        val_D0_raw = None
        val_D0 = None
        if val_array is not None:
            val_sum = 0.0
            n_val = val_array.shape[0]
            codec.eval()
            with torch.no_grad():
                for vs in range(0, n_val, batch_size):
                    ve = min(vs + batch_size, n_val)
                    X_v = torch.from_numpy(val_array[vs:ve]).float().to(device)
                    Bv = X_v.shape[0]
                    Y_v, Mu_v, Std_v = batch_normalize_gpu(X_v, mode=norm_mode)
                    Yh_v, _ = codec(Y_v)
                    Xh_v = batch_inv_normalize_gpu(Yh_v, Mu_v, Std_v)

                    if val_teacher_cache is not None:
                        Yt_v = torch.from_numpy(
                            val_teacher_cache[vs:ve]).float().to(device)
                    else:
                        Yt_v = tail.forward_nograd(X_v)
                    Xo_v = tail.forward_nograd(Xh_v)
                    val_sum += ((Yt_v - Xo_v) ** 2).sum().item() / Bv * Bv
                    del X_v, Y_v, Mu_v, Std_v, Yh_v, Xh_v, Yt_v, Xo_v
            val_D0_raw = val_sum / n_val
            val_D0 = val_D0_raw / (T_tokens * D) if d0_normalize == 'per_element' else val_D0_raw

        epoch_time = time.time() - t_epoch
        epoch_info = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'D0_raw': avg_D0_raw,
            'D0_per_token': avg_D0_raw / T_tokens,
            'D0_per_element': avg_D0_raw / (T_tokens * D),
            'D0': avg_D0,
            'response_objective': response_objective,
            'effective_beta': float(effective_beta),
            'beta_calibration': beta_calibration,
            'd0_normalize': d0_normalize,
            'sampled_response_loss': avg_response,
            'total_loss': avg_loss,
            'perplexity': perplexity,
            'dead_entries': dead_entries,
            'temperature': pq.temperature,
            'val_D0_raw': val_D0_raw,
            'val_D0': val_D0,
            'time': epoch_time,
            'images_per_sec': N_img / max(epoch_time, 1e-6),
            'tokens_per_sec': total_tokens / max(epoch_time, 1e-6),
        }
        if device.type == 'cuda':
            epoch_info['peak_memory_mb'] = (
                torch.cuda.max_memory_allocated(device) / 1e6)

        if step_mode == 'alternating':
            total_steps = _alt_counter
            u_steps_epoch = sum(1 for i in range(total_steps)
                                if i % (alt_u_steps + alt_c_steps) < alt_u_steps)
            c_steps_epoch = total_steps - u_steps_epoch
            epoch_info['u_steps'] = u_steps_epoch
            epoch_info['c_steps'] = c_steps_epoch
            epoch_info['total_optimizer_steps'] = total_steps

        if transform and hasattr(transform, 'orth_error'):
            epoch_info['orth_error'] = transform.orth_error()

        if _use_elastic:
            if response_objective == 'legacy':
                epoch_info['sampled_eps_g_stats'] = eps_g_acc.summarise()
            epoch_info['group_coverage'] = group_sampler.coverage_stats
            if _is_v1_1:
                for a in alpha_list:
                    key = f'sampled_response_alpha{a}'
                    epoch_info[key] = response_acc[a].summarise()
                epoch_info['energy_match_audit_max'] = em_audit_max

        if compute_diagnostics:
            epoch_info['q_g_normalized'] = qg_norm_acc.summarise()
            if _use_elastic:
                if response_objective == 'legacy':
                    epoch_info['kappa_g_stats'] = kappa_g_acc.summarise()
                if interaction_values:
                    ixn_arr = np.array(interaction_values)
                    epoch_info['interaction_stats'] = {
                        'mean': float(ixn_arr.mean()),
                        'std': float(ixn_arr.std()),
                        'max_abs': float(np.abs(ixn_arr).max()),
                        'n_samples': len(ixn_arr),
                    }

        # Gradient diagnostics (§11.6 / §15.2)
        if grad_d0_norms:
            epoch_info['grad_D0_norm'] = float(np.mean(grad_d0_norms))
            epoch_info['grad_response_norm'] = float(
                np.mean(grad_response_norms))
            epoch_info['grad_response_raw_norm'] = float(
                np.mean(grad_response_raw_norms))
            epoch_info['grad_cos_d0_response'] = float(
                np.mean(grad_cosines))
            epoch_info['grad_ratio_response_d0'] = float(
                np.mean(grad_ratios))
        epoch_info['grad_clip_ratio'] = (clip_counts / clip_total
                                         if clip_total > 0 else 0.0)
        if clip_pre_norms:
            epoch_info['grad_clip_pre_norm_mean'] = float(
                np.mean(clip_pre_norms))
            epoch_info['grad_clip_post_norm_mean'] = float(
                np.mean(clip_post_norms))

        # Periodic held-out evaluation (§11.4 / §15.7)
        if (val_array is not None and val_teacher_cache is not None
                and _use_elastic
                and epoch % val_elasticity_interval == 0
                and epoch != epochs - 1):
            _max_img = 200
            if _is_v1_1:
                if verbose:
                    print(f"  [ep {epoch}] Computing validation v1.1 "
                          f"responses...")
                val_el = evaluate_heldout_v1_1(
                    val_array, val_teacher_cache, codec, tail,
                    G, norm_mode, batch_size, device,
                    alphas=alpha_list,
                    max_images=_max_img,
                    probe_group_chunk=heldout_probe_group_chunk)
            else:
                if verbose:
                    print(f"  [ep {epoch}] Computing validation "
                          f"elasticity...")
                val_el = evaluate_heldout_elasticity(
                    val_array, val_teacher_cache, codec, tail,
                    G, alpha, norm_mode, batch_size, device,
                    max_images=_max_img,
                    probe_group_chunk=heldout_probe_group_chunk)
            epoch_info['val_elasticity'] = val_el

        history.append(epoch_info)

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            val_str = (f"  val={val_D0_raw:.1f}"
                       if val_D0_raw is not None else "")
            tau_str = f"  τ={pq.temperature:.4f}" if _use_soft else ""
            resp_str = ""
            if _use_elastic:
                if response_objective == 'legacy':
                    el_summary = eps_g_acc.summarise()
                    resp_str = (
                        f"  sL_el={avg_response:.2f}"
                        f"  sε_cv={el_summary['global_cv']:.3f}"
                        f"  sε_span={el_summary['global_span']:.4f}")
                else:
                    a_main = alpha_list[-1]
                    r_sum = response_acc[a_main].summarise()
                    resp_str = (
                        f"  L_resp={avg_response:.4f}"
                        f"  {response_objective}_cv="
                        f"{r_sum['global_cv']:.3f}"
                        f"  span={r_sum['global_span']:.4f}")
            orth_str = ""
            if transform and hasattr(transform, 'orth_error'):
                orth_str = f"  ||RR'-I||={transform.orth_error():.1e}"
            grad_str = ""
            if grad_d0_norms:
                grad_str = (
                    f"  |g0|={np.mean(grad_d0_norms):.2e}"
                    f" |gr|={np.mean(grad_response_norms):.2e}"
                    f" cos={np.mean(grad_cosines):.3f}"
                    f" ratio={np.mean(grad_ratios):.3f}"
                    f" clip={epoch_info['grad_clip_ratio']:.0%}")
            print(f"  ep {epoch:3d}/{epochs}  D0={avg_D0_raw:.1f}"
                  f"  D0/elem={avg_D0_raw/(T_tokens*D):.4f}"
                  f"  ppl={perplexity:.1f}{resp_str}{tau_str}"
                  f"{val_str}{orth_str}{grad_str}"
                  f"  ({time.time() - t_epoch:.1f}s)")

    # Final full-group response on training calibration subset
    if _use_elastic and verbose and history:
        if _is_v1_1:
            print(f"  Computing full-group v1.1 response on "
                  f"calibration subset...")
            cal_el = evaluate_heldout_v1_1(
                features_array, teacher_cache, codec, tail,
                G, norm_mode, batch_size, device,
                alphas=alpha_list,
                max_images=200,
                probe_group_chunk=heldout_probe_group_chunk)
            history[-1]['full_v1_1_stats'] = cal_el
            a_main = alpha_list[-1]
            key = f'M_g_alpha{a_main}'
            if key in cal_el:
                m_sum = cal_el[key]
                print(f"  Full M_g(α={a_main}): "
                      f"mean={m_sum['global_mean']:.4f} "
                      f"std={m_sum['global_std']:.4f} "
                      f"span={m_sum['global_span']:.4f} "
                      f"pairwise={m_sum['pairwise_dispersion']:.6f}")
        else:
            print(f"  Computing full-group elasticity on "
                  f"calibration subset...")
            cal_el = evaluate_heldout_elasticity(
                features_array, teacher_cache, codec, tail,
                G, alpha, norm_mode, batch_size, device,
                max_images=200,
                probe_group_chunk=heldout_probe_group_chunk)
            history[-1]['full_eps_g_stats'] = cal_el.get('eps_g')
            full_summary = cal_el['eps_g']
            print(f"  Full ε_g: "
                  f"mean={full_summary['global_mean']:.4f} "
                  f"CV={full_summary['global_cv']:.4f} "
                  f"span={full_summary['global_span']:.4f}")

    return codec, history
