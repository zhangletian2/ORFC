"""GPU k-means, k-means++ seeding and the OPQ alternation.

Lifted unchanged from the v8 ``phase1/init.py`` -- the numerics were audited
there and v9 re-uses them so that the only thing that differs between the two
rounds is the menu, not the estimator.

Two deliberate deviations from what the repo's own ``batched_kmeans`` did, both
inherited from v8 and both still wanted here:

* centroids are seeded by k-means++ rather than by random data points, because
  random seeding produces near-duplicate centroids far more often;
* training vectors come from train-core only.  The repo's older preparation
  path sampled from the whole 4,500-row cache, which contains the 500 cal rows,
  and cal is the set that selects the candidate.
"""

import numpy as np
import torch

from opq import batch_normalize_gpu

from . import config as C
from . import splits

CHUNK_BYTES = 1 << 30


def _chunk_size(groups, width):
    return max(1, min(1 << 15, CHUNK_BYTES // (groups * width * 4)))


def load_training_vectors(name="train_core", device="cuda", batch=64,
                          max_images=None):
    """Normalised train-core features, flattened to [N, D] on the GPU."""
    splits.assert_training_split(name)
    feat, _, rows, names = splits.load_split(name)
    if max_images is not None:
        rows, names = rows[:max_images], names[:max_images]
    array = np.load(feat, mmap_mode="r")
    out = []
    for start in range(0, len(rows), batch):
        block = np.asarray(array[rows[start:start + batch]])
        x = torch.from_numpy(block).float().to(device)
        y, _, _ = batch_normalize_gpu(x, mode=C.NORM_MODE)
        out.append(y.reshape(-1, y.shape[-1]))
        del x
    return torch.cat(out), names, feat


def _grouped(y, rotation, first, last, groups, dim):
    """One chunk of rotated features, viewed as [G, chunk, d]."""
    z = y[first:last] if rotation is None else y[first:last] @ rotation
    return z.reshape(-1, groups, dim).permute(1, 0, 2).contiguous()


@torch.no_grad()
def kmeans_plusplus(y, rotation, groups, dim, k, generator, device):
    """D^2 seeding, run in parallel across all G groups."""
    total = y.shape[0]
    chunk = _chunk_size(groups, max(k, dim))
    centroids = torch.empty(groups, k, dim, device=device)
    closest = torch.full((groups, total), float("inf"), device=device)

    pick = torch.randint(total, (1,), generator=generator, device=device)
    for index in range(k):
        if index == 0:
            chosen = pick.expand(groups).clone()
        else:
            weights = closest.clamp_min(0)
            flat = weights.sum(1, keepdim=True)
            # a group whose points are all already covered falls back to uniform
            degenerate = (flat.squeeze(1) <= 0)
            probs = torch.where(
                degenerate.unsqueeze(1),
                torch.full_like(weights, 1.0 / total),
                weights / flat.clamp_min(torch.finfo(weights.dtype).tiny))
            chosen = torch.multinomial(probs, 1, generator=generator).squeeze(1)

        for g in range(groups):
            first = int(chosen[g])
            block = _grouped(y, rotation, first, first + 1, groups, dim)
            centroids[g, index] = block[g, 0]

        for first in range(0, total, chunk):
            last = min(first + chunk, total)
            block = _grouped(y, rotation, first, last, groups, dim)
            distance = (block - centroids[:, index:index + 1]).square().sum(-1)
            torch.minimum(closest[:, first:last], distance,
                          out=closest[:, first:last])
    return centroids


@torch.no_grad()
def lloyd(y, rotation, centroids, iterations, groups, dim):
    """Chunked Lloyd iterations; empty clusters keep their previous centroid."""
    device = centroids.device
    total, k = y.shape[0], centroids.shape[1]
    chunk = _chunk_size(groups, max(k, dim))
    offset = (torch.arange(groups, device=device) * k).unsqueeze(1)
    for _ in range(iterations):
        sums = torch.zeros(groups * k, dim, device=device)
        counts = torch.zeros(groups * k, device=device)
        for first in range(0, total, chunk):
            last = min(first + chunk, total)
            block = _grouped(y, rotation, first, last, groups, dim)
            labels = torch.cdist(block, centroids).argmin(-1)
            flat = (labels + offset).reshape(-1)
            sums.scatter_add_(
                0, flat.unsqueeze(1).expand(-1, dim), block.reshape(-1, dim))
            counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=sums.dtype))
        alive = counts > 0
        updated = torch.where(
            alive.unsqueeze(1), sums / counts.clamp_min(1).unsqueeze(1),
            centroids.reshape(groups * k, dim))
        centroids = updated.reshape(groups, k, dim)
    return centroids


@torch.no_grad()
def quantisation_mse(y, rotation, centroids, groups, dim):
    total = y.shape[0]
    chunk = _chunk_size(groups, max(centroids.shape[1], dim))
    error = 0.0
    for first in range(0, total, chunk):
        last = min(first + chunk, total)
        block = _grouped(y, rotation, first, last, groups, dim)
        error += float(torch.cdist(block, centroids).square().amin(-1).sum())
    return error / total


@torch.no_grad()
def dead_fraction(y, rotation, centroids, groups, dim):
    """Fraction of codewords no train-core vector maps to.  Recorded only."""
    total, k = y.shape[0], centroids.shape[1]
    chunk = _chunk_size(groups, max(k, dim))
    counts = torch.zeros(groups, k, device=centroids.device)
    for first in range(0, total, chunk):
        last = min(first + chunk, total)
        block = _grouped(y, rotation, first, last, groups, dim)
        labels = torch.cdist(block, centroids).argmin(-1)
        counts.scatter_add_(1, labels, torch.ones_like(labels, dtype=counts.dtype))
    return (counts == 0).float().mean(-1).cpu().numpy()


@torch.no_grad()
def opq_warmup(y, groups, dim, k, alternations, lloyd_iters, generator,
               device, log=print):
    """Alternate PQ codebook refit and a Procrustes update of the rotation."""
    width = groups * dim
    rotation = torch.eye(width, device=device)
    centroids = kmeans_plusplus(y, rotation, groups, dim, k, generator, device)
    history = []
    chunk = _chunk_size(groups, max(k, dim))
    for step in range(alternations):
        centroids = lloyd(y, rotation, centroids, lloyd_iters, groups, dim)

        # M = Y^T Zhat accumulated in chunks, so Zhat is never materialised
        moment = torch.zeros(width, width, device=device, dtype=torch.float64)
        for first in range(0, y.shape[0], chunk):
            last = min(first + chunk, y.shape[0])
            block = _grouped(y, rotation, first, last, groups, dim)
            labels = torch.cdist(block, centroids).argmin(-1)
            hat = torch.gather(
                centroids, 1, labels.unsqueeze(-1).expand(-1, -1, dim))
            hat = hat.permute(1, 0, 2).reshape(last - first, width)
            moment += (y[first:last].t() @ hat).double()
        # float64 SVD: the fp32 one is only orthogonal to ~1e-2 at D=1024,
        # three orders of magnitude worse than the gate, and that is a real
        # defect in U rather than a measurement artifact.
        left, _, right = torch.linalg.svd(moment)
        rotation = (left @ right).float()             # argmax tr(U^T M)

        mse = quantisation_mse(y, rotation, centroids, groups, dim)
        history.append(mse)
        log(f"  opq {step + 1:2d}/{alternations}  mse {mse:.6f}")
    return rotation, history
