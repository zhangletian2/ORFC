"""Content-adaptive pairwise token merge + position residual decoder.

No channel rotation R, no PQ, no entropy.  Patch tokens only (CLS identity).

Encoder (hard, no grad):
    greedy disjoint matching of ``r`` pairs.  Pair score is either cosine
    similarity or an ``L_ref`` proxy (Hutchinson estimate of
    ``||J δ||²`` for the mean-merge perturbation).  Each pair ``(i, j)``
    is replaced by the mean ``y_m = (x_i + x_j) / 2``.  Unmatched patch
    tokens pass through as identity.  Coded sequence is
    ``CLS + unmatched + y_m`` (length ``T_full - r``).
    With ``r = T_patch / 2`` this is a perfect matching (no unmatched).

Decoder (learned):
    ``x̂_i = y_m + r_ψ(y_m, Δp_i)``,  ``x̂_j = y_m + r_ψ(y_m, Δp_j)``
    where ``Δp`` is the 2-D grid offset from the pair midpoint.
    ``r_ψ`` is a shared MLP, last layer zero-init so training starts at
    ``x̂ = y_m`` (copy-mean reconstruction).

``forward(Y, pairs=None) -> (Y_hat, None)``.  ``pairs`` is required when
matching is the ``L_ref`` proxy (or pass ``X`` + bound tail).
"""

import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ORFC_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'orfc'))
if _ORFC_DIR not in sys.path:
    sys.path.insert(0, _ORFC_DIR)

from opq import (  # noqa: E402
    batch_normalize_gpu, batch_inv_normalize_gpu, learn_opq_rotation,
)
from soft_pq import SoftPQ, OrthogonalTransform  # noqa: E402


# Must be below any finite affinity (L_ref costs can be >> 1e4).
_NEG = float('-inf')


def infer_patch_hw(n_patch, grid=None):
    """Return ``(H, W)`` for a rasterised patch token sequence."""
    if grid is not None:
        if isinstance(grid, (int, float)):
            h = w = int(grid)
        else:
            h, w = int(grid[0]), int(grid[1])
        if h * w != n_patch:
            raise ValueError(f"grid {h}x{w} != n_patch={n_patch}")
        return h, w
    h = int(round(math.sqrt(n_patch)))
    if h * h != n_patch:
        raise ValueError(
            f"n_patch={n_patch} is not a square grid; pass grid=(H,W)")
    return h, h


def greedy_match_from_affinity(affinity, r):
    """Greedy disjoint matching.  ``affinity [B, T, T]``, higher = merge first."""
    B, T, _ = affinity.shape
    if r > T // 2:
        raise ValueError(f"r={r} > T/2={T // 2}")
    triu = torch.triu(torch.ones(T, T, device=affinity.device, dtype=torch.bool),
                      diagonal=1)
    work = affinity.masked_fill(~triu, _NEG)
    used = torch.zeros(B, T, dtype=torch.bool, device=affinity.device)
    left = torch.empty(B, r, dtype=torch.long, device=affinity.device)
    right = torch.empty(B, r, dtype=torch.long, device=affinity.device)
    arange = torch.arange(B, device=affinity.device)
    for _k in range(r):
        work.masked_fill_(used.unsqueeze(2), _NEG)
        work.masked_fill_(used.unsqueeze(1), _NEG)
        arg = work.reshape(B, -1).argmax(dim=1)
        i = torch.div(arg, T, rounding_mode='floor')
        j = arg - i * T
        left[:, _k] = i
        right[:, _k] = j
        used[arange, i] = True
        used[arange, j] = True
    return left, right


@torch.no_grad()
def greedy_match_pairs(tokens, r, affinity=None):
    """Global greedy matching.  Default affinity = cosine similarity."""
    if affinity is None:
        x = F.normalize(tokens, dim=-1)
        affinity = x @ x.transpose(-1, -2)
    return greedy_match_from_affinity(affinity, r)


def lref_pair_cost_from_grad(X_patch, G_patch):
    """Pair cost ``((g_i - g_j) · (x_i - x_j))²``.  ``X,G [B, T, D]`` -> ``[B, T, T]``.

    For a mean-merge perturbation ``δ_i = (x_j-x_i)/2``, ``δ_j = (x_i-x_j)/2``,
    one Hutchinson probe ``u`` gives ``u·ΔF ≈ ((g_j-g_i)·Δx)/2`` with
    ``g = ∂(u·F)/∂X``.  Squared cost ranks pairs by estimated ``||J δ||²``.
    """
    S = torch.bmm(G_patch, X_patch.transpose(1, 2))
    a = (G_patch * X_patch).sum(-1)
    diff = a.unsqueeze(2) + a.unsqueeze(1) - S - S.transpose(1, 2)
    return diff.square()


def lref_proxy_match_pairs(X, tail, r, n_prefix=1, n_probe=1, generator=None,
                           probe='hutchinson'):
    """Match pairs by ``L_ref`` Jacobian proxy.  ``X [B, T_full, D]`` unnormalised.

    ``probe='hutchinson'``: all probes are Gaussian (unbiased estimate of
    ``||J δ||²``).  ``probe='energy'``: probe 0 uses ``u = tail(X)``, extra
    probes are Gaussian.
    """
    p = int(n_prefix)
    n_probe = max(int(n_probe), 1)
    device = X.device
    Xp = X[:, p:, :].detach()
    cost = None
    with torch.enable_grad():
        for k in range(n_probe):
            Xg = X.detach().requires_grad_(True)
            F = tail(Xg)
            if probe == 'energy' and k == 0:
                u = F.detach()
            else:
                u = torch.randn(F.shape, device=device, generator=generator,
                                dtype=F.dtype)
                flat = u.reshape(u.shape[0], -1)
                u = u / flat.norm(dim=1).clamp_min(1e-8).view(-1, 1, 1)
            (F * u).sum().backward()
            G = Xg.grad.detach()[:, p:, :]
            term = lref_pair_cost_from_grad(Xp, G)
            cost = term if cost is None else cost + term
            del Xg, F, u, G, term
    affinity = -cost
    del cost
    return greedy_match_from_affinity(affinity, r)


def spatial_undirected_edges(H, W, device):
    """4-connected undirected edges on an ``H×W`` raster grid."""
    ii, jj = [], []
    for row in range(H):
        for col in range(W):
            t = row * W + col
            if col + 1 < W:
                ii.append(t)
                jj.append(t + 1)
            if row + 1 < H:
                ii.append(t)
                jj.append(t + W)
    return (torch.tensor(ii, device=device, dtype=torch.long),
            torch.tensor(jj, device=device, dtype=torch.long))


def propose_lref_candidates(patch, used, n_cand, H, W):
    """Propose unused pairs: half top-cosine, half spatial/random.

    Cosine only *proposes*; the caller scores with real ΔL_ref.
    Spatial/random pairs are included so L_ref can pick pairs cosine would reject.
    """
    B, T, D = patch.shape
    device = patch.device
    n_cand = int(n_cand)
    n_cos = max(n_cand // 2, 1)
    n_oth = n_cand - n_cos

    x = F.normalize(patch, dim=-1)
    sim = x @ x.transpose(-1, -2)
    triu = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), 1)
    blocked = used.unsqueeze(1) | used.unsqueeze(2) | ~triu
    sim = sim.masked_fill(blocked, _NEG)
    cos_vals, cos_arg = sim.reshape(B, -1).topk(n_cos, dim=1)
    cos_i = torch.div(cos_arg, T, rounding_mode='floor')
    cos_j = cos_arg - cos_i * T
    valid_cos = torch.isfinite(cos_vals) & (cos_vals > -1.0e3)

    si, sj = spatial_undirected_edges(H, W, device)
    spa_ok = (~used[:, si]) & (~used[:, sj])
    spa_prob = spa_ok.float()
    spa_sum = spa_prob.sum(dim=1, keepdim=True)
    uniform = torch.full_like(spa_prob, 1.0 / max(spa_prob.shape[1], 1))
    spa_prob = torch.where(spa_sum > 0, spa_prob / spa_sum.clamp_min(1e-8), uniform)
    n_spa = min(n_oth, max(int(si.numel()), 0))
    if n_spa > 0:
        spa_idx = torch.multinomial(spa_prob, n_spa, replacement=True)
        spa_i = si[spa_idx]
        spa_j = sj[spa_idx]
        valid_spa = spa_ok.gather(1, spa_idx) & (spa_i != spa_j)
    else:
        spa_i = torch.zeros(B, 0, dtype=torch.long, device=device)
        spa_j = spa_i
        valid_spa = torch.zeros(B, 0, dtype=torch.bool, device=device)

    n_rand = n_oth - n_spa
    parts_i = [cos_i, spa_i]
    parts_j = [cos_j, spa_j]
    parts_v = [valid_cos, valid_spa]
    if n_rand > 0:
        avail = (~used).float()
        avail_sum = avail.sum(dim=1, keepdim=True)
        uni_t = torch.full_like(avail, 1.0 / T)
        avail = torch.where(avail_sum > 0, avail / avail_sum.clamp_min(1e-8), uni_t)
        ri = torch.multinomial(avail, n_rand, replacement=True)
        rj = torch.multinomial(avail, n_rand, replacement=True)
        bi = torch.arange(B, device=device)[:, None]
        valid_r = (ri != rj) & (~used[bi, ri]) & (~used[bi, rj])
        parts_i.append(ri)
        parts_j.append(rj)
        parts_v.append(valid_r)

    ci = torch.cat(parts_i, dim=1)
    cj = torch.cat(parts_j, dim=1)
    valid = torch.cat(parts_v, dim=1)
    return ci, cj, valid


def _delta_l_merge_candidates(current, teacher, tail, left, right, n_prefix,
                              valid):
    """Exact ΔL_ref of copy-mean merging each candidate pair on ``current``."""
    B, Tfull, D = current.shape
    C = left.shape[1]
    Xb = current.repeat_interleave(C, dim=0)
    L = left.reshape(-1)
    Rgt = right.reshape(-1)
    v = valid.reshape(-1)
    bb = torch.arange(B * C, device=current.device)
    p = int(n_prefix)
    mean = 0.5 * (Xb[bb, L + p] + Xb[bb, Rgt + p])
    Xb = Xb.clone()
    if v.any():
        idx = v.nonzero(as_tuple=False).squeeze(1)
        Xb[bb[idx], L[idx] + p] = mean[idx]
        Xb[bb[idx], Rgt[idx] + p] = mean[idx]
    Fhat = tail.forward_nograd(Xb)
    Tch = teacher.repeat_interleave(C, dim=0)
    dl = ((Tch - Fhat) ** 2).flatten(1).sum(dim=1).view(B, C)
    return dl.masked_fill(~valid, 1.0e30)


def _commit_disjoint(dl, left, right, used, commit_k):
    """Greedy take up to ``commit_k`` disjoint min-ΔL_ref pairs per image."""
    B, C = dl.shape
    device = dl.device
    work = dl.clone()
    pick_i = torch.full((B, commit_k), -1, dtype=torch.long, device=device)
    pick_j = torch.full((B, commit_k), -1, dtype=torch.long, device=device)
    n_ok = torch.zeros(B, dtype=torch.long, device=device)
    local_used = used.clone()
    arange = torch.arange(B, device=device)
    for k in range(commit_k):
        blocked = local_used.gather(1, left.clamp(min=0)) | local_used.gather(
            1, right.clamp(min=0))
        work = work.masked_fill(blocked, 1.0e30)
        best, arg = work.min(dim=1)
        ok = best < 1.0e29
        if not ok.any():
            break
        ii = left[arange, arg]
        jj = right[arange, arg]
        pick_i[ok, k] = ii[ok]
        pick_j[ok, k] = jj[ok]
        n_ok[ok] += 1
        local_used[arange[ok], ii[ok]] = True
        local_used[arange[ok], jj[ok]] = True
        work[arange, arg] = 1.0e30
    return pick_i, pick_j, n_ok, local_used


@torch.no_grad()
def exact_lref_greedy_match(X, tail, r, n_prefix=1, n_cand=8, commit_k=1,
                            grid=None):
    """Sequential greedy matching scored by real ΔL_ref (copy-mean + tail).

    At each round, ``n_cand`` unused pairs are proposed, each is scored by
    ``||tail(H)-tail(Ĥ)||²`` of merging that pair on the current reconstruction,
    then up to ``commit_k`` disjoint min-cost pairs are applied.
    """
    B, Tfull, D = X.shape
    p = int(n_prefix)
    T = Tfull - p
    H, W = infer_patch_hw(T, grid)
    teacher = tail.forward_nograd(X)
    current = X.clone()
    used = torch.zeros(B, T, dtype=torch.bool, device=X.device)
    left = torch.empty(B, r, dtype=torch.long, device=X.device)
    right = torch.empty(B, r, dtype=torch.long, device=X.device)
    filled = torch.zeros(B, dtype=torch.long, device=X.device)
    commit_k = max(int(commit_k), 1)
    n_cand = max(int(n_cand), 1)
    max_rounds = r + 4
    for _rnd in range(max_rounds):
        if (filled >= r).all():
            break
        ci, cj, valid = propose_lref_candidates(
            current[:, p:, :], used, n_cand, H, W)
        still = filled < r
        valid = valid & still.unsqueeze(1)
        dl = _delta_l_merge_candidates(
            current, teacher, tail, ci, cj, p, valid)
        k = min(commit_k, int((r - filled.min()).item()))
        pi, pj, n_ok, _ = _commit_disjoint(dl, ci, cj, used, k)
        for b in range(B):
            n_take = int(min(int(n_ok[b]), int(r - filled[b])))
            for t in range(n_take):
                i = int(pi[b, t])
                j = int(pj[b, t])
                slot = int(filled[b])
                left[b, slot] = i
                right[b, slot] = j
                ii, jj = i + p, j + p
                mean = 0.5 * (current[b, ii] + current[b, jj])
                current[b, ii] = mean
                current[b, jj] = mean
                used[b, i] = True
                used[b, j] = True
                filled[b] += 1
        if int(n_ok.max()) == 0:
            for b in range(B):
                unused = (~used[b]).nonzero(as_tuple=False).squeeze(1)
                k = 0
                while int(filled[b]) < r and k + 1 < int(unused.numel()):
                    i = int(unused[k])
                    j = int(unused[k + 1])
                    slot = int(filled[b])
                    left[b, slot] = i
                    right[b, slot] = j
                    mean = 0.5 * (current[b, i + p] + current[b, j + p])
                    current[b, i + p] = mean
                    current[b, j + p] = mean
                    used[b, i] = True
                    used[b, j] = True
                    filled[b] += 1
                    k += 2
            break
    if (filled < r).any():
        raise RuntimeError(
            f"exact L_ref matching failed to fill r={r} pairs "
            f"(filled min={int(filled.min())})")
    return left, right


def precompute_exact_lref_pairs(features, tail, r, n_prefix, device,
                                batch_size=4, n_cand=8, commit_k=1,
                                grid=None, verbose=True):
    """Precompute exact-ΔL_ref pairs.  Independent of ``r_ψ``."""
    if isinstance(features, np.ndarray) and features.ndim == 3:
        arr = features
    else:
        arr = np.stack(features)
    N = arr.shape[0]
    left_all = np.empty((N, r), dtype=np.int64)
    right_all = np.empty((N, r), dtype=np.int64)
    t0 = time.time()
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        X = torch.from_numpy(arr[s:e]).float().to(device)
        left, right = exact_lref_greedy_match(
            X, tail, r, n_prefix=n_prefix, n_cand=n_cand,
            commit_k=commit_k, grid=grid)
        left_all[s:e] = left.cpu().numpy()
        right_all[s:e] = right.cpu().numpy()
        del X, left, right
        if verbose and ((s // batch_size) % 5 == 0 or e == N):
            print(f"    exact L_ref pairs {e}/{N}  ({time.time() - t0:.1f}s)",
                  flush=True)
    return left_all, right_all


def precompute_lref_pairs(features, tail, r, n_prefix, device, batch_size=16,
                          n_probe=1, seed=42, verbose=True, probe='hutchinson'):
    """Precompute L_ref-proxy pairs for a feature list/array.  Matching is
    independent of ``r_ψ``, so this is done once before training/eval.
    """
    if isinstance(features, np.ndarray) and features.ndim == 3:
        arr = features
    else:
        arr = np.stack(features)
    N = arr.shape[0]
    left_all = np.empty((N, r), dtype=np.int64)
    right_all = np.empty((N, r), dtype=np.int64)
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    t0 = time.time()
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        X = torch.from_numpy(arr[s:e]).float().to(device)
        left, right = lref_proxy_match_pairs(
            X, tail, r, n_prefix=n_prefix, n_probe=n_probe, generator=g,
            probe=probe)
        left_all[s:e] = left.cpu().numpy()
        right_all[s:e] = right.cpu().numpy()
        del X, left, right
        if verbose and ((s // batch_size) % 20 == 0 or e == N):
            print(f"    lref pairs {e}/{N}  ({time.time() - t0:.1f}s)",
                  flush=True)
    return left_all, right_all


def matching_jaccard(left_a, right_a, left_b, right_b):
    """Mean Jaccard overlap of two disjoint pairings.  Arrays ``[N, r]``."""
    n = left_a.shape[0]
    acc = 0.0
    for i in range(n):
        a = set(tuple(sorted((int(u), int(v))))
                for u, v in zip(left_a[i], right_a[i]))
        b = set(tuple(sorted((int(u), int(v))))
                for u, v in zip(left_b[i], right_b[i]))
        acc += len(a & b) / max(len(a | b), 1)
    return float(acc / max(n, 1))


def cosine_match_pairs_numpy(features, r, n_prefix, device, batch_size=64,
                             norm_mode='per_image'):
    """Cosine greedy pairs as numpy ``[N, r]`` (for overlap diagnostics)."""
    if isinstance(features, np.ndarray) and features.ndim == 3:
        arr = features
    else:
        arr = np.stack(features)
    N = arr.shape[0]
    left_all = np.empty((N, r), dtype=np.int64)
    right_all = np.empty((N, r), dtype=np.int64)
    p = int(n_prefix)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        X = torch.from_numpy(arr[s:e]).float().to(device)
        Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
        left, right = greedy_match_pairs(Y[:, p:, :], r)
        left_all[s:e] = left.cpu().numpy()
        right_all[s:e] = right.cpu().numpy()
        del X, Y, left, right
    return left_all, right_all


def unmatched_indices(used, n_patch):
    """``used [B, T]`` bool → sorted unmatched indices ``[B, T - n_used]``."""
    B = used.shape[0]
    n_unm = n_patch - int(used[0].sum().item())
    if n_unm <= 0:
        return used.new_zeros(B, 0, dtype=torch.long)
    idx = torch.arange(n_patch, device=used.device).expand(B, -1)
    keys = torch.where(used, torch.full_like(idx, n_patch), idx)
    return keys.argsort(dim=1)[:, :n_unm]


def pair_stats(left, right, H, W):
    """Mean cosine is filled by the caller; here: mean grid L2 distance."""
    t = torch.arange(H * W, device=left.device)
    rr, cc = t // W, t % W
    di = (rr[left] - rr[right]).float()
    dj = (cc[left] - cc[right]).float()
    dist = (di ** 2 + dj ** 2).sqrt()
    return dist.mean().item()


class ResidualDecoder(nn.Module):
    """``r_ψ(y_m, Δp) -> R^D``.  Last layer zeros => identity-at-init."""

    def __init__(self, D, hidden=256, pos_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D + pos_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, D),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, y_m, dp):
        return self.net(torch.cat([y_m, dp], dim=-1))


class SimilarityMergeCodec(nn.Module):
    """Merge ``r`` pairs; decode with ``y_m + r_ψ(y_m, Δp)``.

    ``match_mode``:
      ``cosine``     — greedy on cosine similarity of normalised patch tokens
      ``tome``       — ToMe bipartite even/odd matching (size-weighted dest)
      ``lref``       — sequential greedy scored by real ΔL_ref (tail forward)
      ``lref_proxy`` — Hutchinson Jacobian proxy (inaccurate; kept for ablation)

    Set ``use_decoder=False`` to reconstruct both slots as ``y_m``.
    """

    def __init__(self, D, n_prefix=1, r=128, grid=None, hidden=256,
                 use_decoder=True, match_mode='cosine', n_probe=1,
                 probe='hutchinson', n_cand=8, commit_k=1):
        super().__init__()
        self.D = int(D)
        self.n_prefix = int(n_prefix)
        self.r = int(r)
        self.grid = grid
        self.hidden = int(hidden)
        self.use_decoder = bool(use_decoder)
        self.match_mode = str(match_mode)
        self.n_probe = int(n_probe)
        self.probe = str(probe)
        self.n_cand = int(n_cand)
        self.commit_k = int(commit_k)
        self.tail = None
        self.decoder = ResidualDecoder(D, hidden=hidden)

    def bind_tail(self, tail):
        self.tail = tail
        return self

    def _hw(self, n_patch):
        return infer_patch_hw(n_patch, self.grid)

    def coded_tokens(self, T_full=None):
        """``CLS + unmatched + y_m`` has length ``T_full - r``."""
        if T_full is None:
            return None
        return int(T_full) - int(self.r)

    def _delta_p(self, idx, H, W):
        """Grid coords of token indices ``idx [B, r]`` -> ``[B, r, 2]``."""
        rows = torch.div(idx, W, rounding_mode='floor').float()
        cols = (idx - rows.long() * W).float()
        return torch.stack([rows, cols], dim=-1)

    def _resolve_pairs(self, Y, X=None, pairs=None):
        if pairs is not None:
            return pairs
        p = self.n_prefix
        if self.match_mode == 'cosine':
            return greedy_match_pairs(Y[:, p:, :], self.r)
        if self.match_mode == 'lref':
            if X is None or self.tail is None:
                raise RuntimeError(
                    "lref matching needs unnormalised X and bind_tail(), "
                    "or precomputed pairs=")
            return exact_lref_greedy_match(
                X, self.tail, self.r, n_prefix=p, n_cand=self.n_cand,
                commit_k=self.commit_k, grid=self.grid)
        if self.match_mode == 'lref_proxy':
            if X is None or self.tail is None:
                raise RuntimeError(
                    "lref_proxy matching needs unnormalised X and bind_tail(), "
                    "or precomputed pairs=")
            return lref_proxy_match_pairs(
                X, self.tail, self.r, n_prefix=p, n_probe=self.n_probe,
                probe=self.probe)
        raise ValueError(f"unknown match_mode={self.match_mode!r}")

    def _encode_tome(self, Y):
        """CLS + unmerged A + size-weighted B.  Length ``T_full - r``."""
        B, T, D = Y.shape
        p = self.n_prefix
        patch = Y[:, p:, :]
        n_patch = patch.shape[1]
        src_idx, dst_idx, unm_idx, n_a, n_b = tome_bipartite_match(
            patch, self.r)
        src, dst = patch[:, ::2, :], patch[:, 1::2, :]
        src_m = src.gather(1, src_idx.unsqueeze(-1).expand(-1, -1, D))
        dst_acc = dst.clone()
        cnt = torch.ones(B, n_b, 1, device=Y.device, dtype=dst.dtype)
        dst_acc.scatter_add_(
            1, dst_idx.unsqueeze(-1).expand(-1, -1, D), src_m)
        cnt.scatter_add_(
            1, dst_idx.unsqueeze(-1),
            torch.ones(B, self.r, 1, device=Y.device, dtype=dst.dtype))
        dst_m = dst_acc / cnt
        n_unm = unm_idx.shape[1]
        parts = []
        if p:
            parts.append(Y[:, :p, :])
        if n_unm:
            xm = src.gather(1, unm_idx.unsqueeze(-1).expand(-1, -1, D))
            parts.append(xm)
        parts.append(dst_m)
        aux = {
            "kind": "tome",
            "src_idx": src_idx, "dst_idx": dst_idx, "unm_idx": unm_idx,
            "n_patch": n_patch, "n_a": n_a, "n_b": n_b,
        }
        return torch.cat(parts, dim=1), aux

    def _decode_tome(self, seq, aux):
        """Unmerged A + all B; merged A copies its dest token."""
        B, Tm, D = seq.shape
        p = self.n_prefix
        n_unm = aux["unm_idx"].shape[1]
        n_a, n_b = aux["n_a"], aux["n_b"]
        xm = seq[:, p:p + n_unm, :]
        dst_hat = seq[:, p + n_unm:, :]
        if dst_hat.shape[1] != n_b:
            raise RuntimeError(
                f"ToMe decode expected n_b={n_b} dest tokens, "
                f"got {dst_hat.shape[1]}")
        out_a = dst_hat.new_zeros(B, n_a, D)
        if n_unm:
            out_a.scatter_(
                1, aux["unm_idx"].unsqueeze(-1).expand(-1, -1, D), xm)
        src_hat = dst_hat.gather(
            1, aux["dst_idx"].unsqueeze(-1).expand(-1, -1, D))
        out_a.scatter_(
            1, aux["src_idx"].unsqueeze(-1).expand(-1, -1, D), src_hat)
        out = dst_hat.new_zeros(B, aux["n_patch"], D)
        out[:, ::2, :] = out_a
        out[:, 1::2, :] = dst_hat
        if p == 0:
            return out
        return torch.cat([seq[:, :p, :], out], dim=1)

    def encode_pairs(self, Y, X=None, pairs=None):
        """Match + mean.  Returns ``seq [B, T_full-r, D]`` = CLS + unmatched + y_m."""
        if self.match_mode == "tome":
            return self._encode_tome(Y)
        B, T, D = Y.shape
        p = self.n_prefix
        patch = Y[:, p:, :]
        n_patch = patch.shape[1]
        if 2 * self.r > n_patch:
            raise ValueError(f"r={self.r} needs 2r <= T_patch={n_patch}")
        H, W = self._hw(n_patch)
        left, right = self._resolve_pairs(Y, X=X, pairs=pairs)
        used = torch.zeros(B, n_patch, dtype=torch.bool, device=Y.device)
        used.scatter_(1, left, True)
        used.scatter_(1, right, True)
        unmatched = unmatched_indices(used, n_patch)
        xi = torch.gather(patch, 1, left.unsqueeze(-1).expand(-1, -1, D))
        xj = torch.gather(patch, 1, right.unsqueeze(-1).expand(-1, -1, D))
        ym = 0.5 * (xi + xj)
        xm = torch.gather(patch, 1, unmatched.unsqueeze(-1).expand(-1, -1, D))
        pi = self._delta_p(left, H, W)
        pj = self._delta_p(right, H, W)
        mid = 0.5 * (pi + pj)
        aux = {
            'left': left, 'right': right, 'unmatched': unmatched,
            'dpi': (pi - mid) / max(H, W),
            'dpj': (pj - mid) / max(H, W),
            'n_patch': n_patch,
        }
        parts = []
        if p:
            parts.append(Y[:, :p, :])
        parts.append(xm)
        parts.append(ym)
        return torch.cat(parts, dim=1), aux

    def decode_pairs(self, seq, aux):
        """Expand ``seq [B, T_full-r, D]`` back to ``[B, T_full, D]``."""
        if aux.get("kind") == "tome":
            return self._decode_tome(seq, aux)
        B, Tm, D = seq.shape
        p = self.n_prefix
        n_unm = aux['unmatched'].shape[1]
        xm = seq[:, p:p + n_unm, :]
        ym = seq[:, p + n_unm:, :]
        if ym.shape[1] != self.r:
            raise RuntimeError(
                f"decode expected r={self.r} merged tokens, got {ym.shape[1]}")
        if self.use_decoder:
            flat_y = ym.reshape(B * self.r, D)
            res_i = self.decoder(flat_y, aux['dpi'].reshape(B * self.r, 2))
            res_j = self.decoder(flat_y, aux['dpj'].reshape(B * self.r, 2))
            hat_i = ym + res_i.view(B, self.r, D)
            hat_j = ym + res_j.view(B, self.r, D)
        else:
            hat_i = ym
            hat_j = ym
        out = ym.new_zeros(B, aux['n_patch'], D)
        if n_unm:
            out.scatter_(1, aux['unmatched'].unsqueeze(-1).expand(-1, -1, D), xm)
        out.scatter_(1, aux['left'].unsqueeze(-1).expand(-1, -1, D), hat_i)
        out.scatter_(1, aux['right'].unsqueeze(-1).expand(-1, -1, D), hat_j)
        if p == 0:
            return out
        return torch.cat([seq[:, :p, :], out], dim=1)

    def forward(self, Y, X=None, pairs=None):
        seq, aux = self.encode_pairs(Y, X=X, pairs=pairs)
        return self.decode_pairs(seq, aux), None

    @torch.no_grad()
    def match_diagnostics(self, Y, X=None, pairs=None):
        """Return mean matched cosine and mean grid L2 distance of pairs."""
        p = self.n_prefix
        patch = Y[:, p:, :]
        n_patch = patch.shape[1]
        H, W = self._hw(n_patch)
        left, right = self._resolve_pairs(Y, X=X, pairs=pairs)
        x = F.normalize(patch, dim=-1)
        B, T, D = patch.shape
        bi = torch.arange(B, device=Y.device)[:, None]
        cos = (x[bi, left] * x[bi, right]).sum(-1).mean().item()
        dist = pair_stats(left, right, H, W)
        return {'mean_matched_cos': float(cos), 'mean_grid_l2': float(dist)}


def train_similarity_merge_codec(
    features_train,
    tail,
    D,
    n_prefix=1,
    r=128,
    grid=None,
    hidden=256,
    norm_mode='per_image',
    epochs=100,
    lr=3e-4,
    batch_size=32,
    device='cuda',
    seed=42,
    val_features=None,
    verbose=True,
    grad_clip=1.0,
    match_mode='cosine',
    n_probe=1,
    probe='hutchinson',
    n_cand=8,
    commit_k=1,
    pair_left=None,
    pair_right=None,
    val_pair_left=None,
    val_pair_right=None,
):
    """Train ``r_ψ`` with ΔL_ref.  Matching is hard and not trained.

    For ``match_mode`` in ``{lref, lref_proxy}`` pass precomputed pairs
    (or they are computed once here).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if isinstance(features_train, np.ndarray) and features_train.ndim == 3:
        features_array = features_train
    else:
        features_array = np.stack(features_train)
    N_img = features_array.shape[0]
    T_full = features_array.shape[1]

    codec = SimilarityMergeCodec(
        D, n_prefix=n_prefix, r=r, grid=grid, hidden=hidden,
        use_decoder=True, match_mode=match_mode, n_probe=n_probe,
        probe=probe, n_cand=n_cand, commit_k=commit_k).to(device)
    if match_mode in ('lref', 'lref_proxy'):
        codec.bind_tail(tail)
        if pair_left is None:
            if verbose:
                print(f"  precomputing {match_mode} pairs on train...")
            if match_mode == 'lref':
                pair_left, pair_right = precompute_exact_lref_pairs(
                    features_array, tail, r, n_prefix, device,
                    batch_size=min(batch_size, 4), n_cand=n_cand,
                    commit_k=commit_k, grid=grid, verbose=verbose)
            else:
                pair_left, pair_right = precompute_lref_pairs(
                    features_array, tail, r, n_prefix, device,
                    batch_size=min(batch_size, 16), n_probe=n_probe, seed=seed,
                    verbose=verbose, probe=probe)
        if val_features is not None and val_pair_left is None:
            if verbose:
                print(f"  precomputing {match_mode} pairs on val...")
            if match_mode == 'lref':
                val_pair_left, val_pair_right = precompute_exact_lref_pairs(
                    val_features, tail, r, n_prefix, device,
                    batch_size=min(batch_size, 4), n_cand=n_cand,
                    commit_k=commit_k, grid=grid, verbose=verbose)
            else:
                val_pair_left, val_pair_right = precompute_lref_pairs(
                    val_features, tail, r, n_prefix, device,
                    batch_size=min(batch_size, 16), n_probe=n_probe,
                    seed=seed + 7, verbose=verbose, probe=probe)

    trainable = [p for p in codec.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=lr * 0.01)
    if verbose:
        n_tr = sum(p.numel() for p in trainable)
        print(f"  Trainable params: {n_tr:,}  (objective: ΔL_ref)")
        print(f"  match={match_mode}  probe={probe}  n_probe={n_probe}  r={r}  "
              f"coded tokens/img={codec.coded_tokens(T_full)}  "
              f"(T_full={T_full})")

    def _pairs_for(idx, left_np, right_np):
        if left_np is None:
            return None
        return (
            torch.from_numpy(left_np[idx]).long().to(device),
            torch.from_numpy(right_np[idx]).long().to(device),
        )

    teacher_cache = np.empty_like(features_array)
    with torch.no_grad():
        for s in range(0, N_img, batch_size):
            e = min(s + batch_size, N_img)
            Xc = torch.from_numpy(features_array[s:e]).float().to(device)
            teacher_cache[s:e] = tail.forward_nograd(Xc).cpu().numpy()
            del Xc
    torch.cuda.empty_cache()

    val_array = np.stack(val_features) if val_features else None
    history = []
    for epoch in range(epochs):
        t_ep = time.time()
        perm = np.random.permutation(N_img)
        total_d = 0.0
        codec.train()
        for s in range(0, N_img, batch_size):
            idx = perm[s:s + batch_size]
            X = torch.from_numpy(features_array[idx]).float().to(device)
            B = X.shape[0]
            with torch.no_grad():
                Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode,
                                                 n_prefix=n_prefix)
            Y_hat, _ = codec(Y, pairs=_pairs_for(idx, pair_left, pair_right))
            Y_teacher = torch.from_numpy(teacher_cache[idx]).float().to(device)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
            distortion = ((Y_teacher - tail(X_hat)) ** 2).sum() / B
            optimizer.zero_grad()
            distortion.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
            total_d += distortion.item() * B
            del X, Y, Mu, Std, Y_hat, Y_teacher, X_hat, distortion
        scheduler.step()

        avg_d = total_d / N_img
        val_loss = None
        if val_array is not None:
            codec.eval()
            vs_sum = 0.0
            n_val = val_array.shape[0]
            with torch.no_grad():
                for vs in range(0, n_val, batch_size):
                    ve = min(vs + batch_size, n_val)
                    vidx = np.arange(vs, ve)
                    Xv = torch.from_numpy(val_array[vs:ve]).float().to(device)
                    Yv, Muv, Stdv = batch_normalize_gpu(
                        Xv, mode=norm_mode, n_prefix=n_prefix)
                    Yhv, _ = codec(
                        Yv, pairs=_pairs_for(vidx, val_pair_left, val_pair_right))
                    Ytv = tail.forward_nograd(Xv)
                    Xhv = batch_inv_normalize_gpu(Yhv, Muv, Stdv)
                    vs_sum += ((Ytv - tail.forward_nograd(Xhv)) ** 2).sum().item()
                    del Xv, Yv, Muv, Stdv, Yhv, Ytv, Xhv
            val_loss = vs_sum / n_val

        info = {
            'epoch': epoch,
            'loss_distortion': avg_d,
            'val_loss': val_loss,
            'lr': scheduler.get_last_lr()[0],
            'time': time.time() - t_ep,
        }
        history.append(info)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            vstr = f"  val={val_loss:.1f}" if val_loss is not None else ""
            print(f"  ep {epoch:3d}/{epochs}  D={avg_d:.1f}{vstr}  "
                  f"({info['time']:.1f}s)")
    codec._pair_left = pair_left
    codec._pair_right = pair_right
    return codec, history


def save_merge_codec(codec, path, meta_extra=None):
    meta = {
        'format': 'similarity_merge_v1',
        'D': codec.D,
        'n_prefix': codec.n_prefix,
        'r': codec.r,
        'grid': codec.grid,
        'hidden': codec.hidden,
        'match_mode': getattr(codec, 'match_mode', 'cosine'),
        'n_probe': getattr(codec, 'n_probe', 1),
        'probe': getattr(codec, 'probe', 'hutchinson'),
        'n_cand': getattr(codec, 'n_cand', 8),
        'commit_k': getattr(codec, 'commit_k', 1),
        'state_dict': codec.state_dict(),
    }
    if meta_extra:
        meta.update(meta_extra)
    torch.save(meta, path)


def load_merge_codec(path, device='cuda'):
    meta = torch.load(path, map_location='cpu')
    sd = meta['state_dict']
    hidden = int(meta.get('hidden') or sd['decoder.net.0.weight'].shape[0])
    codec = SimilarityMergeCodec(
        D=meta['D'], n_prefix=meta['n_prefix'], r=meta['r'],
        grid=meta.get('grid'), hidden=hidden, use_decoder=True,
        match_mode=meta.get('match_mode', 'cosine'),
        n_probe=int(meta.get('n_probe', 1)),
        probe=meta.get('probe', 'hutchinson'),
        n_cand=int(meta.get('n_cand', 8)),
        commit_k=int(meta.get('commit_k', 1)))
    codec.load_state_dict(sd)
    return codec.to(device).eval(), meta


def matching_side_bits(r, T_patch):
    """Bits to send a disjoint matching: 2r token indices."""
    return float(2 * r * np.log2(max(T_patch, 2)))


@torch.no_grad()
def tome_bipartite_match(tokens, r):
    """ToMe bipartite soft matching on patch tokens ``[B, T, D]``.

    A = even indices, B = odd.  Each A token links to its most similar B;
    keep the top-``r`` edges (dest may repeat).  Returns indices into A/B.
    """
    B, T, _D = tokens.shape
    n_a = (T + 1) // 2
    n_b = T // 2
    if r < 1 or r > n_a:
        raise ValueError(f"ToMe r={r} not in [1, n_a={n_a}] (T={T})")
    if n_b < 1:
        raise ValueError(f"ToMe needs >=1 odd token, T={T}")
    k = F.normalize(tokens, dim=-1)
    a, b = k[:, ::2, :], k[:, 1::2, :]
    scores = a @ b.transpose(-1, -2)
    node_max, node_idx = scores.max(dim=-1)
    edge_idx = node_max.argsort(dim=-1, descending=True)
    src_idx = edge_idx[:, :r]
    unm_idx = edge_idx[:, r:].sort(dim=-1).values
    dst_idx = node_idx.gather(1, src_idx)
    return src_idx, dst_idx, unm_idx, n_a, n_b


def tome_side_bits(r, T_patch):
    """Naive ToMe side-info: r src indices in A plus r dest indices in B."""
    n_a = (int(T_patch) + 1) // 2
    n_b = int(T_patch) // 2
    return float(r * np.log2(max(n_a, 2)) + r * np.log2(max(n_b, 2)))


def merge_side_bits(r, T_patch, match_mode="cosine"):
    if match_mode == "tome":
        return tome_side_bits(r, T_patch)
    return matching_side_bits(r, T_patch)


# ================================================================
#            frozen merge + channel R + PQ
# ================================================================

class SimilarityMergePQCodec(nn.Module):
    """Quantise ``CLS + unmatched + y_m`` (T-r tokens), frozen ``r_ψ`` expands to T.

    matching + r_ψ are frozen.  R and SoftPQ are trained from scratch on
    the short sequence.
    """

    def __init__(self, merge, pq, transform):
        super().__init__()
        self.merge = merge
        self.pq = pq
        self.transform = transform
        self.quantize = True
        for p in self.merge.parameters():
            p.requires_grad_(False)

    def forward(self, Y_norm, damage_group=None, X=None, pairs=None):
        B, T, D = Y_norm.shape
        seq, aux = self.merge.encode_pairs(Y_norm, X=X, pairs=pairs)
        Bm, Tm, _ = seq.shape
        R = self.transform.get_rotation()
        Z = (seq.reshape(Bm * Tm, D) @ R)
        if self.quantize:
            Z_hat, usage = self.pq._quantise(Z)
            if damage_group is not None:
                g = int(damage_group)
                d = self.pq.d
                mean_cw = self.pq.codebooks[g].mean(0)
                Z_hat[:, g * d:(g + 1) * d] = mean_cw.unsqueeze(0)
        else:
            Z_hat = Z
            usage = torch.zeros(self.pq.G, self.pq.K, device=Y_norm.device)
        seq_hat = (Z_hat @ R.t()).reshape(Bm, Tm, D)
        Y_hat = self.merge.decode_pairs(seq_hat, aux)
        return Y_hat, usage

    @property
    def use_rate(self):
        return self.pq.use_rate

    @property
    def _last_rate(self):
        return self.pq._last_rate

    @property
    def lmbda(self):
        return self.pq.lmbda

    @torch.no_grad()
    def get_prior_pmf(self):
        return self.pq.get_prior_pmf()


def _compute_perplexity(usage):
    p = usage / usage.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    entropy = -(p * (p + 1e-10).log()).sum(dim=-1)
    return entropy.exp().mean().item()


@torch.no_grad()
def collect_ym_latents(features, merge, norm_mode, device, n_prefix,
                       batch_size=64, max_rows=200_000, seed=42):
    """Gather unrotated ``CLS + unmatched + y_m`` rows ``[N, D]`` for OPQ init."""
    rows = []
    for s in range(0, len(features), batch_size):
        e = min(s + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[s:e])).float().to(device)
        Y, _, _ = batch_normalize_gpu(X, mode=norm_mode, n_prefix=n_prefix)
        seq, _ = merge.encode_pairs(Y)
        rows.append(seq.reshape(-1, seq.shape[-1]).cpu().numpy())
        del X, Y, seq
    Z = np.concatenate(rows, axis=0)
    if Z.shape[0] > max_rows:
        rng = np.random.RandomState(seed)
        Z = Z[rng.choice(Z.shape[0], max_rows, replace=False)]
    return Z


def train_merge_pq_codec(
    features_train,
    tail,
    merge,
    G, K, d,
    n_prefix=1,
    norm_mode='per_image',
    epochs=100,
    lr=3e-4,
    batch_size=32,
    device='cuda',
    seed=42,
    val_features=None,
    verbose=True,
    lmbda=0.5,
    grad_clip=1.0,
    tau_start=0.5,
    tau_end=0.005,
    tau_schedule='exponential',
    opq_init=True,
):
    """Freeze merge+r_ψ; OPQ-init then train R + PQ on y_m with ΔL_ref."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if isinstance(features_train, np.ndarray) and features_train.ndim == 3:
        features_array = features_train
    else:
        features_array = np.stack(features_train)
    N_img = features_array.shape[0]
    T_full = features_array.shape[1]
    D = features_array.shape[2]
    Tm = T_full - merge.r

    merge = merge.to(device)
    for p in merge.parameters():
        p.requires_grad_(False)
    merge.eval()

    pq = SoftPQ(G, K, d, lmbda=lmbda).to(device)
    transform = OrthogonalTransform(D).to(device)
    codec = SimilarityMergePQCodec(merge, pq, transform).to(device)

    if opq_init:
        if verbose:
            print("  OPQ init of R + codebooks on y_m latents...")
        Z0 = collect_ym_latents(
            list(features_array), merge, norm_mode, device, n_prefix,
            batch_size=max(batch_size, 64), seed=seed)
        R_np, cbs, _ = learn_opq_rotation(
            Z0, G, d, K, max_iter_opq=20, max_iter_kmeans=50,
            device=device, verbose=verbose)
        transform.init_from_opq(R_np)
        pq.init_codebooks(cbs)
        del Z0
        torch.cuda.empty_cache()
    elif verbose:
        print("  R = I, codebooks random; no OPQ")

    use_rate = bool(pq.use_rate)
    trainable = [p for p in codec.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=lr * 0.01)
    _use_soft = (tau_start > 0)
    if verbose:
        n_tr = sum(p.numel() for p in trainable)
        n_fr = sum(p.numel() for p in merge.parameters())
        print(f"  Trainable params: {n_tr:,}  (R+PQ)  frozen merge: {n_fr:,}")
        print(f"  objective: {'R + %g·D' % lmbda if use_rate else 'D'}  "
              f"K={K} Tm={Tm}/{T_full}  ||R'R-I||={transform.orth_error():.2e}")

    teacher_cache = np.empty_like(features_array)
    with torch.no_grad():
        for s in range(0, N_img, batch_size):
            e = min(s + batch_size, N_img)
            Xc = torch.from_numpy(features_array[s:e]).float().to(device)
            teacher_cache[s:e] = tail.forward_nograd(Xc).cpu().numpy()
            del Xc
    torch.cuda.empty_cache()

    val_array = np.stack(val_features) if val_features else None
    history = []
    for epoch in range(epochs):
        t_ep = time.time()
        if _use_soft and epochs > 1:
            prog = epoch / (epochs - 1)
            if tau_schedule == 'linear':
                tau = tau_start + (tau_end - tau_start) * prog
            else:
                tau = tau_start * (tau_end / tau_start) ** prog
            pq.temperature = tau
        elif _use_soft:
            pq.temperature = tau_start
        else:
            pq.temperature = 0.0

        perm = np.random.permutation(N_img)
        total_d = 0.0
        total_r = 0.0
        usage_acc = torch.zeros(G, K, device=device)
        codec.train()
        merge.eval()
        for s in range(0, N_img, batch_size):
            idx = perm[s:s + batch_size]
            X = torch.from_numpy(features_array[idx]).float().to(device)
            B = X.shape[0]
            with torch.no_grad():
                Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode,
                                                 n_prefix=n_prefix)
            Y_hat, usage = codec(Y)
            Y_teacher = torch.from_numpy(teacher_cache[idx]).float().to(device)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
            distortion = ((Y_teacher - tail(X_hat)) ** 2).sum() / B
            if use_rate:
                rate_bits = codec._last_rate * Tm
                loss = rate_bits + lmbda * distortion
                total_r += codec._last_rate.item() * B
            else:
                loss = distortion
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
            total_d += distortion.item() * B
            usage_acc += usage.detach()
            del X, Y, Mu, Std, Y_hat, Y_teacher, X_hat, distortion, loss
        scheduler.step()

        avg_d = total_d / N_img
        avg_r = total_r / N_img if use_rate else 0.0
        val_loss = None
        if val_array is not None:
            codec.eval()
            vs_sum = 0.0
            n_val = val_array.shape[0]
            with torch.no_grad():
                for vs in range(0, n_val, batch_size):
                    ve = min(vs + batch_size, n_val)
                    Xv = torch.from_numpy(val_array[vs:ve]).float().to(device)
                    Yv, Muv, Stdv = batch_normalize_gpu(
                        Xv, mode=norm_mode, n_prefix=n_prefix)
                    Yhv, _ = codec(Yv)
                    Ytv = tail.forward_nograd(Xv)
                    Xhv = batch_inv_normalize_gpu(Yhv, Muv, Stdv)
                    vs_sum += ((Ytv - tail.forward_nograd(Xhv)) ** 2).sum().item()
                    del Xv, Yv, Muv, Stdv, Yhv, Ytv, Xhv
            val_loss = vs_sum / n_val

        info = {
            'epoch': epoch,
            'loss_distortion': avg_d,
            'rate_bits': avg_r,
            'perplexity': _compute_perplexity(usage_acc),
            'val_loss': val_loss,
            'temperature': pq.temperature,
            'time': time.time() - t_ep,
        }
        history.append(info)
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            vstr = f"  val={val_loss:.1f}" if val_loss is not None else ""
            rstr = f"  R={avg_r:.2f}b/t" if use_rate else ""
            print(f"  ep {epoch:3d}/{epochs}  D={avg_d:.1f}{rstr}  "
                  f"ppl={info['perplexity']:.1f}  "
                  f"τ={pq.temperature:.4f}{vstr}  ({info['time']:.1f}s)")
    return codec, history


def save_merge_pq_codec(codec, path, meta_extra=None):
    pq = codec.pq
    m = codec.merge
    meta = {
        'format': 'similarity_merge_pq_v1',
        'G': pq.G, 'K': pq.K, 'd': pq.d, 'D': pq.D,
        'lmbda': pq.lmbda, 'prior_floor': pq.prior_floor,
        'n_prefix': m.n_prefix, 'r': m.r, 'hidden': m.hidden,
        'grid': m.grid,
        'match_mode': getattr(m, 'match_mode', 'cosine'),
        'use_decoder': bool(m.use_decoder),
        'state_dict': codec.state_dict(),
    }
    if meta_extra:
        meta.update(meta_extra)
    torch.save(meta, path)


def load_merge_pq_codec(path, device='cuda'):
    """Rebuild ``SimilarityMergePQCodec`` saved by ``save_merge_pq_codec``."""
    meta = torch.load(path, map_location='cpu')
    fmt = meta.get('format')
    if fmt not in (None, 'similarity_merge_pq_v1'):
        raise ValueError(f"unexpected merge-PQ format {fmt!r} in {path}")
    G, K, d, D = int(meta['G']), int(meta['K']), int(meta['d']), int(meta['D'])
    match_mode = meta.get('match_mode', 'cosine')
    use_decoder = bool(meta.get('use_decoder', match_mode != 'tome'))
    merge = SimilarityMergeCodec(
        D=D, n_prefix=int(meta['n_prefix']), r=int(meta['r']),
        grid=meta.get('grid'), hidden=int(meta.get('hidden', 256)),
        use_decoder=use_decoder, match_mode=match_mode)
    pq = SoftPQ(G, K, d,
                lmbda=meta.get('lmbda', 0.0),
                prior_floor=meta.get('prior_floor', 0.0))
    transform = OrthogonalTransform(D)
    codec = SimilarityMergePQCodec(merge, pq, transform)
    codec.load_state_dict(meta['state_dict'])
    return codec.to(device).eval(), meta
