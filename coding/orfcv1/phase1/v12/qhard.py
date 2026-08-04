"""Fused hard-PQ forwards for exact-budget allocation samples."""

import math
import numpy as np
import torch
import torch.nn.functional as F

from opq import batch_inv_normalize_gpu

from . import config as C
from ..v11.qhard import revive_dead_codewords


def _rate_cost(quantizer, groups=None):
    """Per-codeword ``-log2(p)`` using ORFC's prior-floor convention."""
    if not getattr(quantizer, "use_rate", False):
        return None
    logits = quantizer.log_prior if groups is None else \
        quantizer.log_prior.index_select(0, groups)
    log_p = F.log_softmax(logits, dim=-1)
    if quantizer.prior_floor > 0:
        p = ((1 - quantizer.prior_floor) * log_p.exp()
             + quantizer.prior_floor / quantizer.K)
        return -p.clamp_min(1e-30).log() / math.log(2)
    return -log_p / math.log(2)


def quantise(codec, y, allocations, marginals=None, codeword_temperature=0.0,
             return_rate=False):
    """Hard PQ forward with optional DP/codeword surrogate backward."""
    allocations = torch.as_tensor(
        allocations, dtype=torch.long, device=y.device)
    if allocations.ndim == 1:
        allocations = allocations[None]
    pq = codec.pq
    rotation = codec.transform.get_rotation()
    b, tokens, dim = y.shape
    z = y.reshape(b * tokens, dim) @ rotation
    sub = z.reshape(-1, pq.G, pq.d).permute(1, 0, 2)
    banks, labels, rates = [], [], []
    soft_codewords = float(codeword_temperature) > 0
    for quantizer in pq.quantizers:
        book = quantizer.codebooks
        distances = torch.cdist(sub, book).square()
        rate = _rate_cost(quantizer)
        cost = (distances if rate is None else
                distances + rate.unsqueeze(1) / quantizer.lmbda)
        index = cost.detach().argmin(dim=-1)
        if soft_codewords:
            soft = torch.softmax(-cost / float(codeword_temperature), -1)
            hard = F.one_hot(index, book.shape[1]).to(soft.dtype)
            weights = hard.detach() + soft - soft.detach()
            banks.append(torch.einsum("gnk,gkd->gnd", weights, book))
        else:
            banks.append(torch.gather(
                book, 1, index.unsqueeze(-1).expand(-1, -1, pq.d)))
        labels.append(index)
        rates.append(torch.zeros_like(index, dtype=distances.dtype) if rate is None
                     else torch.gather(rate, 1, index))
    bank, labels, rates = map(torch.stack, (banks, labels, rates))
    groups = torch.arange(pq.G, device=y.device)
    if marginals is None:
        chosen = bank[allocations, groups[None]]
        chosen_rates = rates[allocations, groups[None]]
    else:
        marginals = torch.as_tensor(marginals, device=y.device, dtype=bank.dtype)
        if marginals.shape != (pq.G, len(pq.quantizers)):
            raise ValueError("marginals must have shape [groups, modes]")
        hard = F.one_hot(allocations, len(pq.quantizers)).to(bank.dtype)
        hard_chosen = torch.einsum("sgm,mgnc->sgnc", hard, bank)
        soft_chosen = torch.einsum("gm,mgnc->gnc", marginals, bank)
        chosen = hard_chosen.detach() + soft_chosen[None] - soft_chosen.detach()[None]
        hard_rates = rates[allocations, groups[None]]
        soft_rates = torch.einsum("gm,mgn->gn", marginals, rates)
        chosen_rates = (hard_rates.detach() + soft_rates[None]
                        - soft_rates.detach()[None])
    z_hat = chosen.permute(0, 2, 1, 3).reshape(
        allocations.shape[0], b * tokens, dim)
    if not soft_codewords:
        z_hat = z_hat + (z[None] - z.detach()[None])
    decoded = z_hat @ rotation.t()
    selected_labels = labels[allocations, groups[None]]
    rate_bpt = chosen_rates.reshape(
        allocations.shape[0], pq.G, b, tokens).sum(1).mean(-1)
    result = (decoded.reshape(-1, tokens, dim), selected_labels)
    return (*result, rate_bpt) if return_rate else result


def distortions(codec, tail, y, mu, std, teacher, allocations,
                marginals=None, codeword_temperature=0.0, return_rate=False):
    """Hard tail distortion ``[allocations, images]`` in one tail call."""
    allocations = torch.as_tensor(
        allocations, dtype=torch.long, device=y.device)
    if allocations.ndim == 1:
        allocations = allocations[None]
    count = int(allocations.shape[0])
    result = quantise(
        codec, y, allocations, marginals=marginals,
        codeword_temperature=codeword_temperature, return_rate=return_rate)
    decoded, labels = result[:2]
    b = y.shape[0]
    expanded_mu = mu[None].expand(count, *mu.shape).reshape(
        count * b, *mu.shape[1:])
    expanded_std = std[None].expand(count, *std.shape).reshape(
        count * b, *std.shape[1:])
    output = tail(batch_inv_normalize_gpu(decoded, expanded_mu, expanded_std))
    target = teacher[None].expand(
        count, *teacher.shape).reshape(count * b, *teacher.shape[1:])
    distortion = (output - target).square().reshape(count, b, -1).sum(-1)
    return (distortion, labels, result[2]) if return_rate else (distortion, labels)


def distortion_sparse(codec, tail, y, mu, std, teacher, allocation,
                      codeword_temperature=0.0, rotation=None,
                      return_rate=False):
    """One hard allocation without materialising unused group-mode banks."""
    allocation = torch.as_tensor(
        allocation, dtype=torch.long, device=y.device).reshape(-1)
    pq = codec.pq
    if allocation.numel() != pq.G:
        raise ValueError("allocation must contain one mode per group")
    rotation = (codec.transform.get_rotation() if rotation is None else rotation)
    b, tokens, dim = y.shape
    z = y.reshape(b * tokens, dim) @ rotation
    sub = z.reshape(-1, pq.G, pq.d).permute(1, 0, 2)
    chosen = torch.empty_like(sub)
    labels = torch.empty(
        pq.G, b * tokens, dtype=torch.long, device=y.device)
    soft_codewords = float(codeword_temperature) > 0
    rate_per_token = torch.zeros(b * tokens, device=y.device)
    for mode, quantizer in enumerate(pq.quantizers):
        groups = torch.where(allocation == mode)[0]
        if groups.numel() == 0:
            continue
        vectors = sub.index_select(0, groups)
        book = quantizer.codebooks.index_select(0, groups)
        distances = torch.cdist(vectors, book).square()
        rate = _rate_cost(quantizer, groups)
        cost = (distances if rate is None else
                distances + rate.unsqueeze(1) / quantizer.lmbda)
        index = cost.detach().argmin(dim=-1)
        if soft_codewords:
            soft = torch.softmax(-cost / float(codeword_temperature), -1)
            hard = F.one_hot(index, book.shape[1]).to(soft.dtype)
            weights = hard.detach() + soft - soft.detach()
            reconstruction = torch.einsum("gnk,gkd->gnd", weights, book)
        else:
            reconstruction = torch.gather(
                book, 1, index.unsqueeze(-1).expand(-1, -1, pq.d))
            reconstruction = reconstruction + vectors - vectors.detach()
        chosen.index_copy_(0, groups, reconstruction)
        labels.index_copy_(0, groups, index)
        if rate is not None:
            rate_per_token = rate_per_token + torch.gather(rate, 1, index).sum(0)
    z_hat = chosen.permute(1, 0, 2).reshape(b * tokens, dim)
    decoded = z_hat @ rotation.t()
    output = tail(batch_inv_normalize_gpu(
        decoded.reshape(b, tokens, dim), mu, std))
    distortion = (output - teacher).square().reshape(b, -1).sum(-1)
    rate_bpt = rate_per_token.reshape(b, tokens).mean(-1)
    return (distortion, labels, rate_bpt) if return_rate else (distortion, labels)


@torch.no_grad()
def rates(codec, y, allocations):
    """Hard ECVQ cross-entropy rate ``[allocations, images]`` in bits/token."""
    allocations = torch.as_tensor(
        allocations, dtype=torch.long, device=y.device)
    if allocations.ndim == 1:
        allocations = allocations[None]
    pq = codec.pq
    b, tokens, dim = y.shape
    z = y.reshape(b * tokens, dim) @ codec.transform.get_rotation()
    sub = z.reshape(-1, pq.G, pq.d).permute(1, 0, 2)
    tables = []
    for quantizer in pq.quantizers:
        distance = torch.cdist(sub, quantizer.codebooks).square()
        rate = _rate_cost(quantizer)
        if rate is None:
            tables.append(torch.zeros_like(distance[..., 0]))
            continue
        labels = (distance + rate.unsqueeze(1) / quantizer.lmbda).argmin(-1)
        tables.append(torch.gather(rate, 1, labels))
    tables = torch.stack(tables)
    groups = torch.arange(pq.G, device=y.device)
    selected = tables[allocations, groups[None]]
    return selected.reshape(
        allocations.shape[0], pq.G, b, tokens).sum(1).mean(-1)


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
