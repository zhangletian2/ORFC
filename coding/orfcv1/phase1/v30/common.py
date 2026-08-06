"""Shared training and candidate utilities for V30."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch

from ..v12.allocation_policy import FixedBudgetAllocationPolicy
from ..v22.allocation_dp import topk_allocations
from . import nested


def mode_bits(codec):
    return tuple(codec.pq.mode_bits)


def nominal_rate(allocation, bits):
    return int(sum(bits[int(mode)] for mode in allocation))


def local_probes(base, modes):
    probes, keys = [np.asarray(base).copy()], [(None, None)]
    for group in range(len(base)):
        for mode in range(int(modes)):
            if mode == base[group]:
                continue
            row = np.asarray(base).copy(); row[group] = mode
            probes.append(row); keys.append((group, mode))
    return np.stack(probes), keys


def candidate_pool(codec, tail, resident, base, budget, topk, random_count,
                   image_batch, seed):
    bits = mode_bits(codec)
    probes, keys = local_probes(base, len(bits))
    values = nested.evaluate(codec, tail, resident, probes, image_batch).mean(1)
    costs = np.zeros((codec.pq.G, len(bits)), dtype=np.float64)
    for index, (group, mode) in enumerate(keys[1:], 1):
        costs[group, mode] = values[index] - values[0]
    dp = [np.asarray(row, dtype=np.int64) for _, row in topk_allocations(
        costs, bits, budget, topk=topk)]
    policy = FixedBudgetAllocationPolicy(codec.pq.G, bits, budget).to(
        resident.device)
    generator = torch.Generator(device=resident.device).manual_seed(int(seed))
    random_rows = policy.build(1.0).sample(
        max(1, int(random_count)), generator=generator).cpu().numpy()
    pool, seen = [], set()
    for row in [np.asarray(base)] + dp + list(random_rows):
        key = tuple(map(int, row))
        if key not in seen:
            pool.append(np.asarray(row, dtype=np.int64)); seen.add(key)
    predicted = np.asarray([
        values[0] + sum(costs[g, int(mode)] for g, mode in enumerate(row))
        for row in pool])
    return np.stack(pool), predicted, costs


def train_fixed(codec, tail, resident, allocation, steps, batch, lr,
                temperature, seed, train_u=True):
    parameters = list(codec.pq.parameters())
    for parameter in codec.transform.parameters():
        parameter.requires_grad_(bool(train_u))
    if train_u:
        parameters += list(codec.transform.parameters())
    optimizer = torch.optim.Adam(parameters, lr=float(lr))
    generator = torch.Generator(device=resident.device).manual_seed(int(seed))
    losses = []
    for _ in range(int(steps)):
        index = torch.randint(
            resident.count, (int(batch),), generator=generator,
            device=resident.device)
        value, _ = nested.distortion(
            codec, tail,
            resident.y.index_select(0, index),
            resident.mu.index_select(0, index),
            resident.std.index_select(0, index),
            resident.teacher.index_select(0, index),
            allocation, temperature=temperature)
        loss = value.mean()
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0); optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else None


def adapted_scores(codec, tail, train, report, allocations, steps, batch, lr,
                   temperature, image_batch, seed, train_u=True):
    scores = []
    for index, allocation in enumerate(allocations):
        branch = copy.deepcopy(codec)
        train_fixed(branch, tail, train, allocation, steps, batch, lr,
                    temperature, seed, train_u=train_u)
        score = float(nested.evaluate(
            branch, tail, report, allocation, image_batch).mean())
        scores.append(score)
        del branch
    return np.asarray(scores)


def ranks(values):
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result


def spearman(left, right):
    if len(left) < 2:
        return 1.0
    return float(np.corrcoef(ranks(left), ranks(right))[0, 1])
