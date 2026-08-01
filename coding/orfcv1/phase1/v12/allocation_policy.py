"""Exact-budget structured distribution over per-group PQ modes.

The trainable object is ``logits[g, mode]``.  For integer per-mode costs
``bits[mode]`` and a fixed total budget ``R``, it defines

    p(m | R) propto 1[sum_g bits[m_g] == R]
                     exp(sum_g logits[g, m_g] / temperature).

All samples and the MAP allocation are therefore valid hard allocations.  The
dynamic program never evaluates the ViT tail and costs O(G * M * B), where B is
the shifted rate budget.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn


def _validate_costs(groups, bit_costs, total_bits):
    def integer(value, name):
        try:
            converted = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be an integer") from error
        if isinstance(value, bool) or float(value) != converted:
            raise ValueError(f"{name} must be an integer")
        return converted

    groups = integer(groups, "groups")
    if groups < 1:
        raise ValueError("groups must be positive")
    costs = tuple(integer(value, "bit cost") for value in bit_costs)
    if not costs or any(value < 0 for value in costs):
        raise ValueError("bit_costs must be a non-empty sequence of nonnegative integers")
    if tuple(sorted(costs)) != costs or len(set(costs)) != len(costs):
        raise ValueError("bit_costs must be strictly increasing")
    minimum = groups * costs[0]
    maximum = groups * costs[-1]
    total = integer(total_bits, "total_bits")
    if total < minimum or total > maximum:
        raise ValueError(
            f"total_bits={total} lies outside the reachable interval "
            f"[{minimum}, {maximum}]")
    shifted = tuple(value - costs[0] for value in costs)
    budget = total - minimum
    reachable = {0}
    for _ in range(groups):
        reachable = {left + right for left in reachable for right in shifted
                     if left + right <= budget}
    if budget not in reachable:
        raise ValueError(
            f"total_bits={total} is not reachable with {groups} groups and "
            f"mode costs {costs}")
    return costs, shifted, budget


def _constant(value, reference):
    return reference.new_tensor(float(value))


def _log_dp_rows(scores, costs, budget, reverse=False):
    """Return log-DP rows and Python reachability masks.

    Only reachable states enter ``logsumexp``.  This avoids the undefined
    gradient of ``logsumexp([-inf, ... , -inf])`` at unreachable states.
    """
    groups, _ = scores.shape
    sequence = range(groups - 1, -1, -1) if reverse else range(groups)
    initial = [_constant(float("-inf"), scores) for _ in range(budget + 1)]
    initial[0] = _constant(0.0, scores)
    rows = [torch.stack(initial)]
    masks = [tuple(index == 0 for index in range(budget + 1))]
    previous, previous_mask = rows[0], masks[0]
    for group in sequence:
        candidates = []
        for mode, cost in enumerate(costs):
            shifted = previous.new_full((budget + 1,), float("-inf"))
            shifted[cost:] = previous[:budget + 1 - cost] + scores[group, mode]
            candidates.append(shifted)
        candidates = torch.stack(candidates)
        previous_mask = tuple(
            any(cost <= rate and masks[-1][rate - cost] for cost in costs)
            for rate in range(budget + 1))
        reachable = torch.tensor(previous_mask, device=scores.device)
        current = previous.new_full((budget + 1,), float("-inf"))
        current[reachable] = torch.logsumexp(candidates[:, reachable], dim=0)
        previous = current
        rows.append(previous)
        masks.append(previous_mask)
    if reverse:
        rows = list(reversed(rows))
        masks = list(reversed(masks))
    return rows, masks


@dataclass
class FixedBudgetDistribution:
    """One-use differentiable distribution built from current policy logits."""

    scores: torch.Tensor
    shifted_costs: tuple
    shifted_budget: int
    forward: list
    forward_mask: list
    backward: list
    backward_mask: list

    @property
    def groups(self):
        return int(self.scores.shape[0])

    @property
    def modes(self):
        return int(self.scores.shape[1])

    def log_partition(self):
        return self.forward[-1][self.shifted_budget]

    def marginals(self):
        log_z = self.log_partition()
        output = []
        for group in range(self.groups):
            row = []
            for mode, cost in enumerate(self.shifted_costs):
                prefixes = [
                    prefix for prefix in range(self.shifted_budget - cost + 1)
                    if (self.forward_mask[group][prefix]
                        and self.backward_mask[group + 1][
                            self.shifted_budget - prefix - cost])]
                if not prefixes:
                    row.append(_constant(0.0, self.scores))
                else:
                    prefix = torch.tensor(prefixes, device=self.scores.device)
                    suffix = self.shifted_budget - prefix - cost
                    terms = (self.forward[group][prefix]
                             + self.scores[group, mode]
                             + self.backward[group + 1][suffix] - log_z)
                    row.append(torch.exp(torch.logsumexp(terms, 0)))
            output.append(torch.stack(row))
        return torch.stack(output)

    def entropy(self):
        probabilities = self.marginals()
        return self.log_partition() - (probabilities * self.scores).sum()

    def log_prob(self, allocations, validate=True):
        allocations = torch.as_tensor(
            allocations, dtype=torch.long, device=self.scores.device)
        squeeze = allocations.ndim == 1
        if squeeze:
            allocations = allocations.unsqueeze(0)
        if allocations.ndim != 2 or allocations.shape[1] != self.groups:
            raise ValueError(
                f"allocations must have shape [N, {self.groups}] or [{self.groups}]")
        if validate and bool(
                ((allocations < 0) | (allocations >= self.modes)).any()):
            raise ValueError("allocation contains an invalid mode index")
        cost_tensor = torch.as_tensor(
            self.shifted_costs, device=allocations.device, dtype=torch.long)
        totals = cost_tensor[allocations].sum(dim=1)
        if validate and bool((totals != self.shifted_budget).any()):
            bad = totals[totals != self.shifted_budget][0].item()
            raise ValueError(
                f"allocation has shifted cost {bad}, expected {self.shifted_budget}")
        group = torch.arange(self.groups, device=allocations.device)
        selected = self.scores[group.unsqueeze(0), allocations].sum(dim=1)
        result = selected - self.log_partition()
        return result[0] if squeeze else result

    @torch.no_grad()
    def sample(self, count=1, generator: Optional[torch.Generator] = None):
        count = int(count)
        if count < 1:
            raise ValueError("count must be positive")
        samples = torch.empty(
            count, self.groups, dtype=torch.long, device=self.scores.device)
        costs = torch.tensor(
            self.shifted_costs, device=self.scores.device, dtype=torch.long)
        masks = torch.tensor(
            self.backward_mask, device=self.scores.device, dtype=torch.bool)
        remaining = torch.full(
            (count,), self.shifted_budget, device=self.scores.device,
            dtype=torch.long)
        for group in range(self.groups):
            suffix = remaining[:, None] - costs[None, :]
            clamped = suffix.clamp(min=0, max=self.shifted_budget)
            valid = (suffix >= 0) & masks[group + 1][clamped]
            values = (self.scores[group][None, :]
                      + self.backward[group + 1][clamped])
            probabilities = torch.softmax(
                values.masked_fill(~valid, float("-inf")), dim=1)
            chosen = torch.multinomial(
                probabilities, 1, generator=generator).squeeze(1)
            samples[:, group] = chosen
            remaining.sub_(costs[chosen])
        return samples

    @torch.no_grad()
    def map_allocation(self):
        """Deterministic max-sum DP; lowest mode index wins exact ties."""
        negative = float("-inf")
        previous = [negative] * (self.shifted_budget + 1)
        previous[0] = 0.0
        backpointers = []
        for group in range(self.groups):
            current = [negative] * (self.shifted_budget + 1)
            choices = [-1] * (self.shifted_budget + 1)
            for rate in range(self.shifted_budget + 1):
                for mode, cost in enumerate(self.shifted_costs):
                    if cost > rate or previous[rate - cost] == negative:
                        continue
                    value = previous[rate - cost] + float(self.scores[group, mode])
                    if value > current[rate]:
                        current[rate] = value
                        choices[rate] = mode
            previous = current
            backpointers.append(choices)
        if previous[self.shifted_budget] == negative:
            raise RuntimeError("MAP budget became unreachable")
        allocation = [-1] * self.groups
        remaining = self.shifted_budget
        for group in range(self.groups - 1, -1, -1):
            mode = backpointers[group][remaining]
            if mode < 0:
                raise RuntimeError("MAP backtracking reached an impossible state")
            allocation[group] = mode
            remaining -= self.shifted_costs[mode]
        return torch.tensor(allocation, dtype=torch.long, device=self.scores.device)


class FixedBudgetAllocationPolicy(nn.Module):
    """Trainable logits for a fixed-budget hard allocation distribution."""

    def __init__(self, groups, bit_costs, total_bits, init_logits=None):
        super().__init__()
        costs, shifted, budget = _validate_costs(groups, bit_costs, total_bits)
        self.groups = int(groups)
        self.bit_costs = costs
        self.shifted_costs = shifted
        self.total_bits = int(total_bits)
        self.shifted_budget = int(budget)
        if init_logits is None:
            initial = torch.zeros(self.groups, len(costs))
        else:
            initial = torch.as_tensor(init_logits, dtype=torch.float32)
            if initial.shape != (self.groups, len(costs)):
                raise ValueError(
                    f"init_logits must have shape {(self.groups, len(costs))}")
        self.logits = nn.Parameter(initial.clone())

    def build(self, temperature=1.0, validate=True):
        temperature = float(temperature)
        if not math.isfinite(temperature) or not temperature > 0:
            raise ValueError("temperature must be finite and positive")
        if validate and not bool(torch.isfinite(self.logits).all()):
            raise ValueError("policy logits must be finite")
        # Training centers each row after an optimizer step.  Keep the raw
        # logits here so log_partition retains its literal mathematical value
        # for arbitrary user-supplied logits and remains testable by enumeration.
        scores = self.logits / temperature
        forward, forward_mask = _log_dp_rows(
            scores, self.shifted_costs, self.shifted_budget, reverse=False)
        backward, backward_mask = _log_dp_rows(
            scores, self.shifted_costs, self.shifted_budget, reverse=True)
        if not forward_mask[-1][self.shifted_budget]:
            raise RuntimeError("fixed budget unexpectedly became unreachable")
        return FixedBudgetDistribution(
            scores=scores,
            shifted_costs=self.shifted_costs,
            shifted_budget=self.shifted_budget,
            forward=forward,
            forward_mask=forward_mask,
            backward=backward,
            backward_mask=backward_mask,
        )

    def actual_rate(self, allocations):
        allocations = torch.as_tensor(
            allocations, dtype=torch.long, device=self.logits.device)
        costs = torch.as_tensor(
            self.bit_costs, dtype=torch.long, device=self.logits.device)
        return costs[allocations].sum(dim=-1)
