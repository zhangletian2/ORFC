"""Fused hard-PQ forwards for exact-budget allocation samples."""

import numpy as np
import torch

from opq import batch_inv_normalize_gpu

from . import config as C
from ..v11.qhard import revive_dead_codewords


def quantise(codec, y, allocations):
    """Quantise several allocations with one rotation and distance bank."""
    allocations = torch.as_tensor(
        allocations, dtype=torch.long, device=y.device)
    if allocations.ndim == 1:
        allocations = allocations[None]
    pq = codec.pq
    rotation = codec.transform.get_rotation()
    b, tokens, dim = y.shape
    z = y.reshape(b * tokens, dim) @ rotation
    sub = z.reshape(-1, pq.G, pq.d).permute(1, 0, 2)
    banks, labels = [], []
    for quantizer in pq.quantizers:
        book = quantizer.codebooks
        with torch.no_grad():
            index = torch.cdist(
                sub.detach(), book.detach()).square().argmin(dim=-1)
        banks.append(torch.gather(
            book, 1, index.unsqueeze(-1).expand(-1, -1, pq.d)))
        labels.append(index)
    bank, labels = torch.stack(banks), torch.stack(labels)
    groups = torch.arange(pq.G, device=y.device)
    chosen = bank[allocations, groups[None]]
    z_hat = chosen.permute(0, 2, 1, 3).reshape(
        allocations.shape[0], b * tokens, dim)
    z_hat = z_hat + (z[None] - z.detach()[None])
    decoded = z_hat @ rotation.t()
    selected_labels = labels[allocations, groups[None]]
    return decoded.reshape(-1, tokens, dim), selected_labels


def distortions(codec, tail, y, mu, std, teacher, allocations):
    """Hard tail distortion ``[allocations, images]`` in one tail call."""
    allocations = torch.as_tensor(
        allocations, dtype=torch.long, device=y.device)
    if allocations.ndim == 1:
        allocations = allocations[None]
    count = int(allocations.shape[0])
    decoded, labels = quantise(codec, y, allocations)
    b = y.shape[0]
    expanded_mu = mu[None].expand(count, *mu.shape).reshape(
        count * b, *mu.shape[1:])
    expanded_std = std[None].expand(count, *std.shape).reshape(
        count * b, *std.shape[1:])
    output = tail(batch_inv_normalize_gpu(decoded, expanded_mu, expanded_std))
    target = teacher[None].expand(
        count, *teacher.shape).reshape(count * b, *teacher.shape[1:])
    return ((output - target).square().reshape(count, b, -1).sum(-1),
            labels)


@torch.no_grad()
def selfcheck(codec, tail, resident, modes, image_batch=C.EVAL_IMAGE_BATCH):
    """Compare the fused forward with the frozen hard evaluator."""
    from ..engine import evaluate_allocations

    allocations = torch.as_tensor(modes, device=resident.device)
    if allocations.ndim == 1:
        allocations = allocations[None]
    reference = evaluate_allocations(
        codec, tail, resident, allocations.cpu().numpy(),
        image_batch=image_batch)
    chunks = []
    for start in range(0, resident.count, image_batch):
        value, _ = distortions(
            codec, tail, *resident.slice(start, start + image_batch),
            allocations)
        chunks.append(value.cpu().numpy())
    mine = np.concatenate(chunks, axis=1)
    scale = float(np.abs(reference).mean())
    gap = float(np.abs(reference - mine).max())
    return {"max_abs_gap": gap, "rel_gap": gap / scale,
            "mean_reference": scale, "tolerance": C.REPLAY_REL_TOL}
