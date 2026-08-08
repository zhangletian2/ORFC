"""Tail distortion evaluators for V33 (sparse + multi-allocation)."""

from __future__ import annotations

import torch

from opq import batch_inv_normalize_gpu

from . import quantise as Q


def distortion_sparse(codec, tail, y, mu, std, teacher, allocation,
                      temperature=0.0):
    """One allocation → per-image tail MSE ``[B]`` (training stream path)."""
    decoded, labels = Q.quantise_sparse(
        codec, y, allocation, temperature=temperature)
    output = tail(batch_inv_normalize_gpu(decoded, mu, std))
    value = (output - teacher).square().reshape(y.shape[0], -1).sum(-1)
    return value, labels


def distortions(codec, tail, y, mu, std, teacher, allocations,
                temperature=0.0):
    """Score many allocations; returns ``[S, B]`` like ``v12.qhard.distortions``.

    Implementation streams :func:`distortion_sparse` so every candidate shares
    the same kernel path (plan requirement for search / fair slate).
    """
    allocations = torch.as_tensor(
        allocations, dtype=torch.long, device=y.device)
    if allocations.ndim == 1:
        allocations = allocations[None]
    rows, label_rows = [], []
    for allocation in allocations:
        value, labels = distortion_sparse(
            codec, tail, y, mu, std, teacher, allocation,
            temperature=temperature)
        rows.append(value)
        label_rows.append(labels)
    return torch.stack(rows), torch.stack(label_rows)


def distortion(codec, tail, y, mu, std, teacher, allocation,
               temperature=0.0):
    """Alias of :func:`distortion_sparse` matching ``v30.nested.distortion``."""
    return distortion_sparse(
        codec, tail, y, mu, std, teacher, allocation,
        temperature=temperature)


@torch.no_grad()
def evaluate(codec, tail, resident, allocations, image_batch=16,
             temperature=0.0):
    """Offline multi-allocation evaluation on a resident GPU cache."""
    from .quantise import frozen_rotations

    allocations = torch.as_tensor(
        allocations, dtype=torch.long, device=resident.device)
    if allocations.ndim == 1:
        allocations = allocations[None]
    rows = []
    with frozen_rotations(codec):
        for allocation in allocations:
            pieces = []
            for first in range(0, resident.count, int(image_batch)):
                value, _ = distortion_sparse(
                    codec, tail, *resident.slice(first, first + image_batch),
                    allocation, temperature=temperature)
                pieces.append(value)
            rows.append(torch.cat(pieces))
    return torch.stack(rows).cpu().numpy()
