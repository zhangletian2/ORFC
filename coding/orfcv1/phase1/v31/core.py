"""Shared deterministic state machinery for the frozen V31 protocol."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch

from ..v12.allocation_policy import FixedBudgetAllocationPolicy
from ..v12.strict_fair import strict_fair_slate
from ..v30 import nested


CODEC_LR = 3e-4
POLICY_LR = 1e-2
POLICY_TEMPERATURE = 1.0
CODEWORD_TEMPERATURE = 0.05
SCHEDULE_MAX = 5580
WARMUP_STEPS = 300
AUTHORIZATION_STEPS = 500
SHORT_ADAPT_STEPS = 8
SHORT_ADAPT_BATCH = 16


class ResidentView:
    """Zero-copy contiguous view of a resident GPU data set."""

    def __init__(self, resident, first, count):
        last = int(first) + int(count)
        if first < 0 or last > resident.count:
            raise ValueError("resident view is out of range")
        self.y = resident.y[int(first):last]
        self.mu = resident.mu[int(first):last]
        self.std = resident.std[int(first):last]
        self.teacher = resident.teacher[int(first):last]
        self.count, self.device = int(count), resident.device

    def slice(self, first, last):
        last = min(int(last), self.count)
        return (self.y[int(first):last], self.mu[int(first):last],
                self.std[int(first):last], self.teacher[int(first):last])


class CommittedCosine:
    """Cosine schedule indexed only by updates committed to shared state."""

    def __init__(self, optimizer, base_lr=CODEC_LR,
                 total=SCHEDULE_MAX, count=0):
        self.optimizer = optimizer
        self.base_lr = float(base_lr); self.total = int(total)
        self.count = int(count); self._apply()

    def _apply(self):
        progress = min(self.count, self.total) / self.total
        lr = self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        if self.count >= self.total:
            lr = 0.0
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def step(self):
        self.count += 1; self._apply()

    def state_dict(self):
        return {"base_lr": self.base_lr, "total": self.total,
                "count": self.count}

    def load_state_dict(self, state):
        if int(state["total"]) != self.total or not math.isclose(
                float(state["base_lr"]), self.base_lr,
                rel_tol=0.0, abs_tol=0.0):
            raise ValueError("scheduler contract differs")
        self.count = int(state["count"]); self._apply()

    @property
    def lr(self):
        return float(self.optimizer.param_groups[0]["lr"])


def build_policy(groups, bits, rate, uniform_mode, device):
    logits = torch.zeros(groups, len(bits))
    logits[torch.arange(groups), int(uniform_mode)] = 2.0
    policy = FixedBudgetAllocationPolicy(
        groups, bits, rate, init_logits=logits).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=POLICY_LR)
    return policy, optimizer


def build_codec_optimizer(codec, optimizer_state=None, scheduler_state=None):
    optimizer = torch.optim.Adam(codec.parameters(), lr=CODEC_LR)
    if optimizer_state is not None:
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    scheduler = CommittedCosine(optimizer)
    if scheduler_state is not None:
        scheduler.load_state_dict(copy.deepcopy(scheduler_state))
    return optimizer, scheduler


def clone_codec_branch(codec, optimizer, scheduler):
    branch = copy.deepcopy(codec)
    branch_optimizer, branch_scheduler = build_codec_optimizer(
        branch, optimizer.state_dict(), scheduler.state_dict())
    return branch, branch_optimizer, branch_scheduler


def codec_step(codec, optimizer, scheduler, tail, resident, allocation,
               index, temperature=CODEWORD_TEMPERATURE):
    value, _ = nested.distortion(
        codec, tail,
        resident.y.index_select(0, index),
        resident.mu.index_select(0, index),
        resident.std.index_select(0, index),
        resident.teacher.index_select(0, index), allocation,
        temperature=temperature)
    loss = value.mean(); optimizer.zero_grad(set_to_none=True)
    loss.backward(); torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
    optimizer.step(); scheduler.step()
    return float(loss.detach())


def generators(device, seed):
    batch = torch.Generator(device=device).manual_seed(int(seed) + 1)
    allocation = torch.Generator(device=device).manual_seed(int(seed) + 2)
    return batch, allocation


def inner_allocation(step, policy, fair, groups, bits, rate, device,
                     allocation_generator):
    if step <= WARMUP_STEPS:
        if fair is None or (step - 1) % len(bits) == 0:
            fair = strict_fair_slate(
                groups, bits, rate, device, allocation_generator)
        allocation = fair[(step - 1) % len(bits)]
    else:
        allocation = policy.build(POLICY_TEMPERATURE).sample(
            1, generator=allocation_generator)[0]
    return allocation, fair


def run_authorization_prefix(codec, policy, codec_optimizer, scheduler,
                             tail, train, groups, bits, rate, seed,
                             steps=AUTHORIZATION_STEPS, batch=32):
    batch_generator, allocation_generator = generators(train.device, seed)
    fair = None; allocations, batches, losses = [], [], []
    exposure = torch.zeros(groups, len(bits), dtype=torch.long,
                           device=train.device)
    for step in range(1, int(steps) + 1):
        allocation, fair = inner_allocation(
            step, policy, fair, groups, bits, rate, train.device,
            allocation_generator)
        index = torch.randint(
            train.count, (int(batch),), generator=batch_generator,
            device=train.device)
        exposure[torch.arange(groups, device=train.device), allocation] += 1
        losses.append(codec_step(
            codec, codec_optimizer, scheduler, tail, train, allocation, index))
        allocations.append(allocation.detach().cpu())
        batches.append(index.detach().cpu())
    return {
        "allocations": torch.stack(allocations),
        "batches": torch.stack(batches),
        "exposure": exposure.cpu(), "losses": losses,
        "batch_generator_state": batch_generator.get_state().cpu(),
        "allocation_generator_state": allocation_generator.get_state().cpu(),
        "batch_generator": batch_generator,
        "allocation_generator": allocation_generator, "fair": fair}


def fixed_adaptation_batches(resident, batch=SHORT_ADAPT_BATCH):
    if resident.count != SHORT_ADAPT_STEPS * int(batch):
        raise ValueError("adaptation set must be exactly 8x16 images")
    return [torch.arange(first, first + int(batch), device=resident.device)
            for first in range(0, resident.count, int(batch))]


def adapt_branch(codec, optimizer, scheduler, tail, resident, allocation,
                 batches):
    branch, branch_optimizer, branch_scheduler = clone_codec_branch(
        codec, optimizer, scheduler)
    losses = [codec_step(
        branch, branch_optimizer, branch_scheduler, tail, resident,
        allocation, index) for index in batches]
    return branch, branch_optimizer, branch_scheduler, losses


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    first = 0
    while first < len(values):
        last = first + 1
        while last < len(values) and values[order[last]] == values[order[first]]:
            last += 1
        ranks[order[first:last]] = 0.5 * (first + last - 1)
        first = last
    return ranks


def spearman(left, right):
    left, right = average_ranks(left), average_ranks(right)
    if len(left) < 2:
        return 1.0
    if float(left.std()) == 0 or float(right.std()) == 0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


def relative_tensor_error(left, right):
    left = left.detach().double().cpu(); right = right.detach().double().cpu()
    return float((left - right).norm() / max(float(right.norm()), 1e-12))


def max_state_error(left, right):
    """Recursively compare tensor state; non-floating objects must be exact."""
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError("state dictionary keys differ")
        return max((max_state_error(left[k], right[k]) for k in left),
                   default=0.0)
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise ValueError("state sequence lengths differ")
        return max((max_state_error(a, b) for a, b in zip(left, right)),
                   default=0.0)
    if torch.is_tensor(left):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError("state tensors differ structurally")
        if left.is_floating_point():
            return relative_tensor_error(left, right)
        if not torch.equal(left.cpu(), right.cpu()):
            raise ValueError("integer tensor state differs")
        return 0.0
    if isinstance(left, float):
        return abs(left - right) / max(abs(right), 1e-12)
    if left != right:
        raise ValueError("discrete state differs")
    return 0.0
