"""
ORFC-v1 finite-perturbation elasticity probe, elastic loss, and
mechanism diagnostics (curvature κ_g, cross-group interaction I_gh).

All functions operate on a single minibatch and are called from the
training loop in ``train_v1.py``.
"""

import math
import torch
import torch.nn.functional as F_fn
import numpy as np


# ================================================================
#  Smooth extreme operators
# ================================================================

def smooth_max(x, tau):
    """τ · log Σ exp(x_g / τ)  —  smooth approximation to max."""
    return tau * torch.logsumexp(x / tau, dim=0)


def smooth_min(x, tau):
    """-τ · log Σ exp(-x_g / τ)  —  smooth approximation to min."""
    return -tau * torch.logsumexp(-x / tau, dim=0)


# ================================================================
#  Elasticity probe (Section 4 of TODO)
# ================================================================

def compute_elasticity(
    X_hat,            # [B, T, D]  actual hard-reconstructed features (orig space)
    e_g_sampled,      # dict[int, Tensor[B*T, D]]  per-group normed-space residuals
    Std,              # [B, 1, D] or broadcastable  —  normalisation std
    tail,             # FrozenTail
    Y_teacher,        # [B, T, D_out]  teacher output (precomputed)
    D0,               # scalar tensor  —  current workpoint distortion
    alpha,            # perturbation scale
    eps=1e-8,
    probe_group_chunk=0,
    return_per_image=False,
):
    """Compute finite-difference elasticity ε_g for selected groups (§13.3b).

    Uses the actual hard reconstruction X_hat as the perturbation centre
    (§11.2), so  D_g± = ||tail(X_hat ± α·e_g·Std) − Y₀||².

    When ``probe_group_chunk > 0``, stacks ±α·e_g for multiple groups
    along the batch dimension and passes them through tail in one call,
    reducing total tail invocations.

    Returns
    -------
    eps_g   : dict[int, Tensor]  —  ε_g scalar for each probed group
    Dplus   : dict[int, Tensor]  —  D_g⁺
    Dminus  : dict[int, Tensor]  —  D_g⁻
    """
    B, T, D = X_hat.shape
    D0_sg = D0.detach()

    eps_g = {}
    Dplus = {}
    Dminus = {}
    Dplus_per_image = {}
    Dminus_per_image = {}

    group_list = list(e_g_sampled.keys())
    chunk = probe_group_chunk if probe_group_chunk > 0 else len(group_list)

    for ci in range(0, len(group_list), chunk):
        chunk_groups = group_list[ci:ci + chunk]
        n_ch = len(chunk_groups)

        eg_origs = []
        for g in chunk_groups:
            eg_origs.append(e_g_sampled[g].reshape(B, T, D) * Std)

        # Stack: [n_ch * B, T, D] for plus; [n_ch * B, T, D] for minus
        X_hat_rep = X_hat.unsqueeze(0).expand(n_ch, -1, -1, -1).reshape(
            n_ch * B, T, D)
        eg_stack = torch.cat(eg_origs, dim=0)  # [n_ch * B, T, D]

        X_plus = X_hat_rep + alpha * eg_stack
        X_minus = X_hat_rep - alpha * eg_stack

        X_both = torch.cat([X_plus, X_minus], dim=0)  # [2*n_ch*B, T, D]
        Y_both = tail(X_both)

        Y_plus_all = Y_both[:n_ch * B]
        Y_minus_all = Y_both[n_ch * B:]

        Y_teacher_rep = Y_teacher.unsqueeze(0).expand(
            n_ch, -1, -1, -1).reshape(n_ch * B, *Y_teacher.shape[1:])

        err_plus_img = ((Y_teacher_rep - Y_plus_all) ** 2).reshape(
            n_ch, B, -1).sum(dim=-1)
        err_minus_img = ((Y_teacher_rep - Y_minus_all) ** 2).reshape(
            n_ch, B, -1).sum(dim=-1)
        err_plus = err_plus_img.mean(dim=-1)
        err_minus = err_minus_img.mean(dim=-1)

        for i, g in enumerate(chunk_groups):
            d_plus = err_plus[i]
            d_minus = err_minus[i]
            eps_val = (d_plus - d_minus) / (2.0 * alpha * (D0_sg + eps))
            eps_g[g] = eps_val
            Dplus[g] = d_plus.detach()
            Dminus[g] = d_minus.detach()
            if return_per_image:
                Dplus_per_image[g] = err_plus_img[i].detach()
                Dminus_per_image[g] = err_minus_img[i].detach()

        del X_both, Y_both, X_plus, X_minus, eg_stack, X_hat_rep

    if return_per_image:
        return eps_g, Dplus, Dminus, {
            'Dplus_per_image': Dplus_per_image,
            'Dminus_per_image': Dminus_per_image,
        }
    return eps_g, Dplus, Dminus


# ================================================================
#  Elastic loss (Section 5 of TODO)
# ================================================================

def elastic_loss(eps_g_values, tau):
    """Zero-baselined elastic loss (§11.9):

        L_elastic⁰ = smax_τ(ε) − smin_τ(ε) − 2τ log|S|

    When all ε_g are equal the raw smooth range equals 2τlog|S|; subtracting
    this constant makes the loss zero at perfect balance and removes the
    dependence on |S| from the loss magnitude.  The gradient is unaffected
    by the constant.

    Args
    ----
    eps_g_values : list[Tensor] or dict or Tensor  —  ε_g for sampled groups
    tau          : float  —  smooth-extreme temperature

    Returns
    -------
    loss : scalar Tensor
    """
    if isinstance(eps_g_values, dict):
        eps_g_values = list(eps_g_values.values())
    if isinstance(eps_g_values, list):
        stacked = torch.stack(eps_g_values)  # [|S|]
    else:
        stacked = eps_g_values

    S = stacked.shape[0]
    baseline = 2.0 * tau * math.log(S)
    return smooth_max(stacked, tau) - smooth_min(stacked, tau) - baseline


def compute_loss_v1(D0, L_elastic, beta):
    """L_v1 = D₀ + β · sg(D₀) · L_elastic

    Uses stopgrad(D0) for dimensional rescaling only.
    """
    return D0 + beta * D0.detach() * L_elastic


# ================================================================
#  Group subset sampling (Section 4)
# ================================================================

class GroupSampler:
    """Round-robin group subset sampler ensuring uniform coverage.

    Each epoch, every group is sampled approximately the same number
    of times.  Within each batch a random subset of size ``n_per_batch``
    is drawn from the current round-robin cursor.
    """

    def __init__(self, G, n_per_batch, seed=0):
        self.G = G
        self.n_per_batch = min(n_per_batch, G)
        self.rng = np.random.RandomState(seed)
        self._deck = []
        self._counts = np.zeros(G, dtype=int)

    def sample(self):
        """Return a list of ``n_per_batch`` group indices."""
        while len(self._deck) < self.n_per_batch:
            new_perm = self.rng.permutation(self.G).tolist()
            self._deck.extend(new_perm)
        chosen = self._deck[:self.n_per_batch]
        self._deck = self._deck[self.n_per_batch:]
        for g in chosen:
            self._counts[g] += 1
        return chosen

    def reset_epoch(self):
        """Optional: reset the deck at epoch boundary for exact balance."""
        self._deck = []

    @property
    def coverage_stats(self):
        return {
            'min': int(self._counts.min()),
            'max': int(self._counts.max()),
            'mean': float(self._counts.mean()),
        }


# ================================================================
#  Mechanism diagnostics (Section 7)
# ================================================================

@torch.no_grad()
def compute_quantisation_difficulty(e_g_all, Std=None, T=None):
    """Compute per-group quantisation difficulty (§11.7).

    Returns dict with:
        q_g_normalized : [G]  —  E||e_g^norm||₂²  (normalised-space)
        q_g_original   : [G]  —  E||e_g^norm ⊙ Std||₂²  (original feature space)
                                 Only present when Std is provided.

    Args:
        e_g_all : [G, N, D]  per-group normalised-space residuals (N = B*T)
        Std     : [B, 1, D]  normalisation std (per-image, shared across tokens)
        T       : int, tokens per image (needed to expand Std to [B*T, D])
    """
    q_norm = (e_g_all ** 2).sum(dim=-1).mean(dim=-1)  # [G]
    result = {'q_g_normalized': q_norm}
    if Std is not None:
        D = e_g_all.shape[-1]
        N = e_g_all.shape[1]
        if Std.dim() == 3 and Std.shape[1] == 1:
            B = Std.shape[0]
            T_infer = N // B if T is None else T
            std_expanded = Std.expand(B, T_infer, D).reshape(1, N, D)
        elif Std.dim() == 2:
            std_expanded = Std.unsqueeze(0)
        else:
            std_expanded = Std.reshape(1, -1, D)
        scaled = e_g_all * std_expanded
        q_orig = (scaled ** 2).sum(dim=-1).mean(dim=-1)
        result['q_g_original'] = q_orig
    return result


@torch.no_grad()
def compute_qg_from_rg(r_g):
    """Compute q_g_normalized directly from rotated-space r_g (§13.3).

    Orthogonal transform preserves norms, so ||e_g^norm||₂ = ||r_g||₂.
    This avoids materializing the full [G, N, D] e_g tensor.

    Args:
        r_g : [G, N, d]  per-group rotated residuals
    Returns:
        q_g_normalized : [G]  —  E||r_g||₂²
    """
    return (r_g ** 2).sum(dim=-1).mean(dim=-1)


@torch.no_grad()
def compute_qg_per_image_from_rg(r_g, B, T, Std=None):
    """Return per-image quantisation difficulty without lifting to D dims.

    Parameters
    ----------
    r_g : Tensor [G, B*T, d]
        Rotated-space group residuals.
    B, T : int
        Batch size and token count.
    Std : Tensor or None
        Normalisation scale. Supported project modes use a scalar scale per
        image/token (`Std.shape[-1] == 1`), so orthogonality preserves the
        residual norm exactly.

    Returns
    -------
    dict
        q_g_normalized: [G, B]
        q_g_original:   [G, B], when Std is provided
    """
    G, N, _ = r_g.shape
    if N != B * T:
        raise ValueError(f"r_g N={N} does not equal B*T={B*T}")
    token_energy = r_g.reshape(G, B, T, -1).square().sum(dim=-1)
    result = {'q_g_normalized': token_energy.mean(dim=-1)}
    if Std is not None:
        if Std.shape[-1] != 1:
            raise ValueError(
                "Compact original-space q_g requires scalar Std per token")
        if Std.shape[1] == 1:
            scale_sq = Std[:, 0, 0].square().reshape(1, B, 1)
        elif Std.shape[1] == T:
            scale_sq = Std[:, :, 0].square().reshape(1, B, T)
        else:
            raise ValueError(
                f"Unsupported Std shape {tuple(Std.shape)} for T={T}")
        result['q_g_original'] = (token_energy * scale_sq).mean(dim=-1)
    return result


@torch.no_grad()
def compute_curvature(Dplus, Dminus, D0, alpha, eps=1e-8):
    """κ_g = (D_g⁺ + D_g⁻ − 2D₀) / (α² · [sg(D₀) + ε])

    Args:
        Dplus, Dminus : dict[int, float or Tensor]
        D0            : float or Tensor
        alpha         : float

    Returns:
        kappa_g : dict[int, float]
    """
    D0_val = D0.item() if isinstance(D0, torch.Tensor) else D0
    kappa = {}
    for g in Dplus:
        dp = Dplus[g].item() if isinstance(Dplus[g], torch.Tensor) else Dplus[g]
        dm = Dminus[g].item() if isinstance(Dminus[g], torch.Tensor) else Dminus[g]
        kappa[g] = (dp + dm - 2 * D0_val) / (alpha ** 2 * (D0_val + eps))
    return kappa


@torch.no_grad()
def compute_interaction(
    X_hat,        # [B, T, D]  actual hard reconstruction (orig space)
    e_g_all,      # [G, B*T, D]  normed-space residuals
    Std,          # normalisation Std
    tail,
    Y_teacher,    # [B, T, D_out]
    D0,           # scalar
    Dplus,        # dict[int, Tensor]  from elasticity probe
    alpha,        # float
    group_pairs,  # list of (g, h) tuples to probe
    eps=1e-8,
    return_per_image=False,
):
    """Cross-group interaction I_gh = D_gh^{++} − D_g⁺ − D_h⁺ + D₀.

    Uses actual hard reconstruction X_hat as centre (§11.2).

    Returns:
        interactions : dict[(g,h), float]  — raw I_gh
        interactions_norm : dict[(g,h), float]  — Ĩ_gh = I_gh / (D₀ + ε)
    """
    B, T, D = X_hat.shape
    if not group_pairs:
        if return_per_image:
            return {}, {}, {}
        return {}, {}

    if isinstance(D0, torch.Tensor):
        D0_tensor = D0.detach()
        D0_val = D0_tensor.mean().item()
    else:
        D0_tensor = torch.tensor(float(D0), device=X_hat.device)
        D0_val = float(D0)

    interactions = {}
    interactions_norm = {}
    interactions_per_image = {}

    def _eg(g):
        return e_g_all[g].reshape(B, T, D) * Std

    X_pp = torch.cat(
        [X_hat + alpha * _eg(g) + alpha * _eg(h)
         for g, h in group_pairs],
        dim=0)
    Y_pp = tail.forward_nograd(X_pp)
    Yt_rep = Y_teacher.unsqueeze(0).expand(
        len(group_pairs), -1, -1, -1).reshape(
            len(group_pairs) * B, *Y_teacher.shape[1:])
    Dpp_img = ((Yt_rep - Y_pp) ** 2).reshape(
        len(group_pairs), B, -1).sum(dim=-1)

    def _as_per_image(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().reshape(-1)
            if value.numel() == B:
                return value
            if value.numel() == 1:
                return value.expand(B)
        return torch.full(
            (B,), float(value), device=X_hat.device, dtype=X_hat.dtype)

    d0_img = _as_per_image(D0_tensor)
    for i, (g, h) in enumerate(group_pairs):
        dp_g = _as_per_image(Dplus[g])
        dp_h = _as_per_image(Dplus[h])
        I_img = Dpp_img[i] - dp_g - dp_h + d0_img
        I_gh = I_img.mean().item()
        interactions[(g, h)] = I_gh
        interactions_norm[(g, h)] = I_gh / (D0_val + eps)
        if return_per_image:
            interactions_per_image[(g, h)] = I_img.detach()

    if return_per_image:
        return interactions, interactions_norm, interactions_per_image
    return interactions, interactions_norm


# ================================================================
#  Per-group statistics aggregator
# ================================================================

class GroupStatsAccumulator:
    """Accumulates per-group scalars across batches and computes
    mean / std / CV / min / max / span at epoch end."""

    def __init__(self, G):
        self.G = G
        self._data = {g: [] for g in range(G)}

    def add(self, values_dict):
        """values_dict: {group_idx: scalar}"""
        for g, v in values_dict.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            self._data[g].append(val)

    def summarise(self):
        per_group = {}
        for g in range(self.G):
            vals = np.array(self._data[g]) if self._data[g] else np.array([0.0])
            abs_mean = float(np.abs(vals).mean())
            mad = float(np.median(np.abs(vals - np.median(vals))))
            per_group[g] = {
                'mean': float(vals.mean()),
                'abs_mean': abs_mean,
                'std': float(vals.std()),
                'cv': float(vals.std() / (abs(vals.mean()) + 1e-10)),
                'mad': mad,
                'min': float(vals.min()),
                'max': float(vals.max()),
                'span': float(vals.max() - vals.min()),
                'n': len(vals),
            }
        means = np.array([per_group[g]['mean'] for g in range(self.G)])
        abs_means = np.array([per_group[g]['abs_mean'] for g in range(self.G)])
        summary = {
            'per_group': per_group,
            'global_mean': float(means.mean()),
            'global_abs_mean': float(abs_means.mean()),
            'global_std': float(means.std()),
            'global_cv': float(means.std() / (abs(means.mean()) + 1e-10)),
            'global_span': float(means.max() - means.min()),
        }
        return summary

    def reset(self):
        self._data = {g: [] for g in range(self.G)}


# ================================================================
#  V1.1: Pairwise dispersion loss (§15.4)
# ================================================================

def pairwise_dispersion_loss(z):
    r"""Sampled pairwise dispersion loss.

    .. math::
        \mathcal L_{\rm pair}(z)
        = \frac{2}{|S|(|S|-1)}
          \sum_{g<h \in S}(z_g - z_h)^2

    Uses the identity  Σ_{g<h}(z_g-z_h)² = S·Σz² − (Σz)²  to avoid
    materialising the full pairwise matrix.

    Contract (§15.4):
      - All components equal → exactly zero
      - Permutation invariant
      - Unbiased estimator of full-group pairwise dispersion
      - Invariant to chunk/batch partitioning
    """
    S = z.shape[0]
    if S < 2:
        return z.new_tensor(0.0)
    sum_sq = (z * z).sum()
    sq_sum = z.sum() ** 2
    pair_sum = S * sum_sq - sq_sum
    return pair_sum * (2.0 / (S * (S - 1)))


# ================================================================
#  V1.1: Per-image energy matching (§15.3.1)
# ================================================================

def energy_match_residuals(
    e_g_dict, B, T, eps=1e-12, q_bar_per_image=None, Std=None,
):
    r"""Per-image energy matching with stop-gradient scaling.

    .. math::
        \widetilde e_{g,i}
        = e_{g,i}\,\sqrt{\frac{\operatorname{sg}(\bar q_i)}
                               {\operatorname{sg}(q_{g,i})+\varepsilon}}

    All groups end up with per-image probe energy equal to q̄_i.
    q_g and q̄ in the denominator are detached; gradient flows only
    through e_{g,i} itself (§15.3.1).

    Parameters
    ----------
    e_g_dict : dict[int, Tensor[B*T, D]]
        Per-group residuals in normalised space (may carry gradient).
    B, T : int
        Batch size and tokens per image.
    q_bar_per_image : Tensor[B] or None
        Detached target energy computed across *all* G groups.  Passing this
        argument is required when ``e_g_dict`` contains only sampled/chunked
        groups.  When omitted, the mean over ``e_g_dict`` is used, which is
        only valid when the dictionary contains the complete group set.
    Std : Tensor or None
        Inverse-normalisation scale.  When provided, q_g and the energy audit
        are measured in original feature space.

    Returns
    -------
    e_g_matched : dict[int, Tensor[B*T, D]]
    q_g_per_image : dict[int, Tensor[B]]
    q_bar_per_image : Tensor[B]
    audit_max_rel_error : float
        max_{i,g} |‖ẽ_{g,i}‖²_F/T − q̄_i| / (q̄_i + ε).
    """
    groups = sorted(e_g_dict.keys())
    D = e_g_dict[groups[0]].shape[-1]

    q_g_per_image = {}
    for g in groups:
        e_g_3d = e_g_dict[g].detach().reshape(B, T, D)
        if Std is not None:
            e_g_3d = e_g_3d * Std
        q_g_per_image[g] = (e_g_3d ** 2).sum(dim=(1, 2)) / T  # [B]

    if q_bar_per_image is None:
        q_bar = torch.stack(
            [q_g_per_image[g] for g in groups]).mean(dim=0)  # [B]
    else:
        q_bar = q_bar_per_image.detach().reshape(B).to(
            device=e_g_dict[groups[0]].device,
            dtype=e_g_dict[groups[0]].dtype)

    e_g_matched = {}
    audit_max_rel = 0.0
    for g in groups:
        q_g = q_g_per_image[g]                              # [B]
        scale = torch.sqrt(q_bar / (q_g + eps))             # [B]
        scale_flat = scale.unsqueeze(1).expand(B, T).reshape(B * T, 1)
        e_g_matched[g] = e_g_dict[g] * scale_flat

        with torch.no_grad():
            matched_3d = e_g_matched[g].detach().reshape(B, T, D)
            if Std is not None:
                matched_3d = matched_3d * Std
            matched_energy = (matched_3d ** 2).sum(dim=(1, 2)) / T
            rel = ((matched_energy - q_bar).abs() / (q_bar + eps)).max().item()
            audit_max_rel = max(audit_max_rel, rel)

    return e_g_matched, q_g_per_image, q_bar, audit_max_rel


# ================================================================
#  V1.1: Batched single-sided probe (§15.3)
# ================================================================

def compute_single_sided_probe(
    X_hat,          # [B, T, D]
    e_g_dict,       # dict[int, Tensor[B*T, D]]
    Std,            # [B, 1, D]
    tail,           # FrozenTail
    Y_teacher,      # [B, T, D_out]
    alphas,         # list[float]
    eps=1e-8,
    probe_chunk=0,
):
    r"""Single-sided probe: D_tail(X̂ − α·e_g·Std) for all (α, g).

    Batches along (alpha × group × B) to minimise tail invocations.
    Does **not** detach e_g — gradient flows back to U through e_g_dict.

    Returns
    -------
    D_probe : dict[(float, int), Tensor]
        Mean D_tail value for each (α, g).
    D_probe_per_image : dict[(float, int), Tensor[B]]
        Per-image D_tail values (always detached).
    """
    B, T, D = X_hat.shape
    groups = sorted(e_g_dict.keys())

    probe_pairs = [(a, g) for a in alphas for g in groups]
    n_probes = len(probe_pairs)
    chunk = probe_chunk if probe_chunk > 0 else n_probes

    D_probe = {}
    D_probe_per_image = {}

    for ci in range(0, n_probes, chunk):
        chunk_pairs = probe_pairs[ci:ci + chunk]
        n_ch = len(chunk_pairs)

        X_rep = X_hat.unsqueeze(0).expand(
            n_ch, -1, -1, -1).reshape(n_ch * B, T, D)

        perts = []
        for a_val, g in chunk_pairs:
            eg_orig = e_g_dict[g].reshape(B, T, D) * Std
            perts.append(a_val * eg_orig)
        pert_stack = torch.cat(perts, dim=0)        # [n_ch*B, T, D]

        X_perturbed = X_rep - pert_stack
        Y_perturbed = tail(X_perturbed)

        Yt_rep = Y_teacher.unsqueeze(0).expand(
            n_ch, -1, -1, -1).reshape(n_ch * B, *Y_teacher.shape[1:])

        err_img = ((Yt_rep - Y_perturbed) ** 2).reshape(
            n_ch, B, -1).sum(dim=-1)               # [n_ch, B]
        err_mean = err_img.mean(dim=-1)             # [n_ch]

        for i, (a_val, g) in enumerate(chunk_pairs):
            D_probe[(a_val, g)] = err_mean[i]
            D_probe_per_image[(a_val, g)] = err_img[i].detach()

        del X_rep, pert_stack, X_perturbed, Y_perturbed, Yt_rep

    return D_probe, D_probe_per_image


# ================================================================
#  V1.1: Response computation (§15.3.1 / §15.3.2)
# ================================================================

def fixed_energy_response(D0, D_probe, alphas, groups, eps=1e-8):
    r"""Fixed-energy response (§15.3.1).

    .. math::
        S_g^{(\alpha)}
        = \frac{D_{\rm tail}(\hat F) - D_{\rm tail}(\hat F - \alpha\tilde e_g)}
               {\alpha\,[D_{\rm tail}(\hat F)+\varepsilon]}

    D0 in the denominator is detached.
    """
    D0_sg = D0.detach()
    S = {}
    for a in alphas:
        for g in groups:
            S[(a, g)] = (D0 - D_probe[(a, g)]) / (a * (D0_sg + eps))
    return S


def operational_response(D0, D_probe, alphas, groups, eps=1e-8):
    r"""Operational repair response (§15.3.2).

    .. math::
        M_g^{(\alpha)}
        = \frac{D_{\rm tail}(\hat F) - D_{\rm tail}(\hat F - \alpha e_g)}
               {D_{\rm tail}(\hat F)+\varepsilon}

    D0 in the denominator is detached.
    """
    D0_sg = D0.detach()
    M = {}
    for a in alphas:
        for g in groups:
            M[(a, g)] = (D0 - D_probe[(a, g)]) / (D0_sg + eps)
    return M


@torch.no_grad()
def scale_nonlinearity(response_dict, alphas, groups):
    r"""Scale nonlinearity indicator (§15.3.2).

    .. math::
        N_g = \max_{\alpha\in\mathcal A}
              \left|\frac{M_g^{(\alpha)}}{\alpha}
                    - \frac{M_g^{(\alpha_{\rm ref})}}{\alpha_{\rm ref}}\right|

    where α_ref = min(alphas).
    """
    ref_alpha = min(alphas)
    N = {}
    for g in groups:
        m_ref = response_dict[(ref_alpha, g)]
        if isinstance(m_ref, torch.Tensor):
            m_ref = m_ref.item()
        rate_ref = m_ref / ref_alpha
        max_dev = 0.0
        for a in alphas:
            m_a = response_dict[(a, g)]
            if isinstance(m_a, torch.Tensor):
                m_a = m_a.item()
            max_dev = max(max_dev, abs(m_a / a - rate_ref))
        N[g] = max_dev
    return N


# ================================================================
#  V1.1: Response loss aggregation (§15.4)
# ================================================================

def compute_response_loss(response_dict, alphas, groups,
                          alpha_weights=None):
    r"""Aggregate pairwise dispersion across alphas.

    .. math::
        \sum_{\alpha} w_\alpha\,\mathcal L_{\rm pair}(z^{(\alpha)})

    Returns (total_loss, per_alpha_loss_dict).
    """
    if alpha_weights is None:
        alpha_weights = [1.0 / len(alphas)] * len(alphas)

    total_loss = None
    per_alpha_loss = {}
    for i, a in enumerate(alphas):
        z = torch.stack([response_dict[(a, g)] for g in groups])
        loss_a = pairwise_dispersion_loss(z)
        per_alpha_loss[a] = loss_a.detach().item()

        weighted = alpha_weights[i] * loss_a
        total_loss = weighted if total_loss is None else total_loss + weighted

    if total_loss is None:
        total_loss = torch.tensor(0.0)

    return total_loss, per_alpha_loss


def compute_loss_v1_1(D0_per_element, L_response, beta):
    r"""V1.1 combined loss (§15.4).

    .. math::
        \mathcal L = \bar D_0 + \beta\,\mathcal L_{\rm response}

    No sg(D0) scaling; β is calibrated via gradient-ratio measurements.
    """
    return D0_per_element + beta * L_response
