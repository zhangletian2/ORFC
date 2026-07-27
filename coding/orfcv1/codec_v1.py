"""
ORFC-v1 extended codec: per-group residual decomposition and
reconstruction audit for the complete ViT tail distortion pipeline.

Reuses OrthogonalTransform and SoftPQ from ``orfc/soft_pq.py``; this
module only adds the diagnostic forward path and audit check.
"""

import math
import numpy as np
import torch
import torch.nn as nn

import sys, os
_ORFC_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'orfc'))
if _ORFC_DIR not in sys.path:
    sys.path.insert(0, _ORFC_DIR)

from opq import batch_normalize_gpu, batch_inv_normalize_gpu
from soft_pq import (
    SoftPQ, OrthogonalTransform, FeatureTransform, FeatureCodec,
    FrozenTail, save_codec as _save_codec_base, load_codec as _load_codec_base,
)


class FeatureCodecV1(nn.Module):
    """Extended FeatureCodec with per-group residual decomposition.

    Forward returns
    ---------------
    Y_hat : [B, T, D]  —  reconstructed normalised features
    info  : dict        —  optional diagnostics (only when ``return_details=True``)
        Z        : [B*T, D']    clean rotated features
        Z_hat    : [B*T, D']    hard-reconstructed rotated features
        r_g      : [G, B*T, d]  per-group rotated residual
        e_g      : [G, B*T, D]  per-group original-space residual
        labels   : [G, B*T]     hard assignment labels
        usage    : [G, K]       usage counts
    """

    def __init__(self, pq: SoftPQ, transform=None):
        super().__init__()
        self.pq = pq
        self.transform = transform

    # ------------------------------------------------------------------
    #  Core forward (drop-in compatible with FeatureCodec)
    # ------------------------------------------------------------------
    def forward(self, Y_norm, return_details=False, differentiable_groups=None,
                detail_level='diagnostic'):
        """Forward pass with optional per-group residual decomposition.

        Parameters
        ----------
        Y_norm : [B, T, D]
        return_details : bool
            If True, return per-group info dict.
        differentiable_groups : list[int] or None
            Group indices whose original-space residuals ``e_g`` should
            retain the computation graph (for elastic loss backprop to U).
            Groups not listed are returned detached.
            ``None`` means all detached (diagnostic mode).
        detail_level : str
            'train'      — only sampled-group e_g_diff + r_g for q_g;
                           skips full e_g, Z, Z_hat, labels to save memory.
            'diagnostic' — full output including e_g [G,N,D] for held-out.
        """
        B, T, D = Y_norm.shape
        flat = Y_norm.reshape(B * T, D)

        is_orth = (self.transform is not None
                   and hasattr(self.transform, 'get_rotation'))

        if is_orth:
            R = self.transform.get_rotation()
            Z = flat @ R
        else:
            Z = self.transform.encode(flat) if self.transform else flat

        Z_hat, usage = self.pq._quantise(Z)

        if is_orth:
            Y_hat = Z_hat @ R.t()
        else:
            Y_hat = (self.transform.decode(Z_hat)
                     if self.transform else Z_hat)

        Y_hat = Y_hat.reshape(B, T, D)

        if not return_details:
            return Y_hat, usage

        N = B * T
        G, d = self.pq.G, self.pq.d
        r_full = Z_hat - Z                               # [N, D']
        r_g = r_full.reshape(N, G, d).permute(1, 0, 2)   # [G, N, d]

        diff_set = set(differentiable_groups) if differentiable_groups else set()

        if is_orth:
            R_t = R.t()                                  # [D', D]

            # Differentiable e_g only for sampled groups (dict)
            e_g_diff = {}
            for g in diff_set:
                e_g_diff[g] = r_g[g] @ R_t[g*d:(g+1)*d, :]

            if detail_level == 'diagnostic':
                with torch.no_grad():
                    e_g_detached = torch.stack(
                        [r_g[g] @ R_t[g*d:(g+1)*d, :] for g in range(G)],
                        dim=0)
            else:
                e_g_detached = None
        else:
            e_g_diff = {}
            e_g_detached = r_g.detach() if detail_level == 'diagnostic' else None

        info = {
            'r_g': r_g.detach(),          # [G, N, d] always available for q_g
            'e_g_diff': e_g_diff,         # {g: [N, D]} differentiable, for loss
            'usage': usage.detach(),
        }

        if detail_level == 'diagnostic':
            info['Z'] = Z.detach()
            info['Z_hat'] = Z_hat.detach()
            info['e_g'] = e_g_detached    # [G, N, D] all detached
            info['labels'] = self.pq._last_labels.detach()

        return Y_hat, info

    # ------------------------------------------------------------------
    #  Convenience properties (compatible with FeatureCodec)
    # ------------------------------------------------------------------
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


# ================================================================
#  Reconstruction audit
# ================================================================

@torch.no_grad()
def reconstruction_audit(
    features,
    codec: FeatureCodecV1,
    norm_mode: str,
    device,
    batch_size: int = 8,
    tol: float = 1e-5,
    return_details: bool = False,
):
    """Audit quantisation decomposition separately from U round-trip error.

    In finite precision, Cayley ``R @ R.T`` is close to but not exactly I.
    The exact decomposition being tested is

        X_hat - X = (X_roundtrip - X) + sum_g e_g_original.

    The PQ identity is ``X_hat - X_roundtrip == sum_g e_g_original``.

    Returns
    -------
    max_relative_error : float
        max over batch of  ||F̂ − F − Σe_g||₂ / (||F̂ − F||₂ + ε)
    passed : bool
        True if max_relative_error < tol
    """
    codec.eval()
    max_quant_rel = 0.0
    max_combined_rel = 0.0
    max_roundtrip_rel = 0.0
    max_orth_error = 0.0
    eps = 1e-8

    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        X = torch.from_numpy(
            np.stack(features[start:end])
        ).float().to(device)
        B, T, D = X.shape

        Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
        Y_hat, info = codec(
            Y, return_details=True, detail_level='diagnostic')
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)

        if (codec.transform is not None
                and hasattr(codec.transform, 'get_rotation')):
            R = codec.transform.get_rotation()
            flat = Y.reshape(B * T, D)
            Y_roundtrip = (flat @ R @ R.t()).reshape(B, T, D)
            orth_err = (
                R.t() @ R - torch.eye(D, device=R.device, dtype=R.dtype)
            ).norm().item()
        elif codec.transform is None:
            Y_roundtrip = Y
            orth_err = 0.0
        else:
            raise RuntimeError(
                "Reconstruction audit currently requires an orthogonal "
                "transform or no transform")
        X_roundtrip = batch_inv_normalize_gpu(
            Y_roundtrip, Mu, Std)

        total_err = X_hat - X                                    # [B, T, D]
        roundtrip_err = X_roundtrip - X
        quant_err = X_hat - X_roundtrip
        e_g = info['e_g']                                        # [G, B*T, D]
        sum_eg = e_g.sum(dim=0).reshape(B, T, D)                 # [B, T, D]
        # inv-normalise contribution: e_g lives in normalised space,
        # so sum_eg * Std gives original-space contribution
        sum_eg_orig = sum_eg * Std

        quant_diff = quant_err - sum_eg_orig
        combined_diff = total_err - roundtrip_err - sum_eg_orig
        quant_rel = (
            quant_diff.reshape(B, -1).norm(dim=1)
            / (quant_err.reshape(B, -1).norm(dim=1) + eps)
        ).max().item()
        combined_rel = (
            combined_diff.reshape(B, -1).norm(dim=1)
            / (total_err.reshape(B, -1).norm(dim=1) + eps)
        ).max().item()
        roundtrip_rel = (
            roundtrip_err.reshape(B, -1).norm(dim=1)
            / (total_err.reshape(B, -1).norm(dim=1) + eps)
        ).max().item()
        max_quant_rel = max(max_quant_rel, quant_rel)
        max_combined_rel = max(max_combined_rel, combined_rel)
        max_roundtrip_rel = max(max_roundtrip_rel, roundtrip_rel)
        max_orth_error = max(max_orth_error, orth_err)

        del X, Y, Mu, Std, Y_hat, X_hat, X_roundtrip, info

    passed = max_quant_rel < tol and max_combined_rel < tol
    details = {
        'quantisation_relative_error': max_quant_rel,
        'combined_relative_error': max_combined_rel,
        'roundtrip_relative_magnitude': max_roundtrip_rel,
        'orthogonality_frobenius': max_orth_error,
        'tolerance': tol,
        'passed': passed,
    }
    if return_details:
        return details
    return max_quant_rel, passed


# ================================================================
#  Save / load  (wraps base implementation, adds V1 tag)
# ================================================================

def save_codec_v1(codec, path):
    """Save V1 codec with architecture metadata.

    Uses atomic write (temp file + rename) to avoid half-written files
    from concurrent or interrupted saves (§13.1).
    """
    import tempfile
    pq = codec.pq
    meta = {
        'version': 'v1',
        'G': pq.G, 'K': pq.K, 'd': pq.d,
        'lmbda': pq.lmbda, 'prior_floor': pq.prior_floor,
        'has_transform': codec.transform is not None,
        'transform_type': (type(codec.transform).__name__
                           if codec.transform else None),
    }
    if codec.transform is not None and hasattr(codec.transform, 'D'):
        meta['D'] = codec.transform.D
    elif codec.transform is not None and hasattr(codec.transform, 'D_in'):
        meta['D_in'] = codec.transform.D_in
        meta['D_out'] = codec.transform.D_out
    meta['state_dict'] = codec.state_dict()
    dir_name = os.path.dirname(path) or '.'
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.pt.tmp')
    os.close(fd)
    try:
        torch.save(meta, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_codec_v1(path, device='cuda'):
    """Load V1 codec."""
    meta = torch.load(path, map_location='cpu')
    G, K, d = meta['G'], meta['K'], meta['d']
    pq = SoftPQ(G, K, d,
                lmbda=meta.get('lmbda', 0.0),
                prior_floor=meta.get('prior_floor', 0.0))
    transform = None
    if meta.get('has_transform'):
        ttype = meta.get('transform_type')
        if ttype == 'OrthogonalTransform':
            transform = OrthogonalTransform(meta['D'])
        elif ttype == 'FeatureTransform':
            transform = FeatureTransform(meta['D_in'], meta['D_out'])
    codec = FeatureCodecV1(pq, transform)
    codec.load_state_dict(meta['state_dict'])
    return codec.to(device).eval()
