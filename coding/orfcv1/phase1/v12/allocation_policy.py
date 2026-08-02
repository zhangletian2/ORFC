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


def _reachability_mask(costs, budget, groups, device):
    """Reachable shifted rates after ``k`` group transitions, ``k = 0..groups``."""
    mask = torch.zeros(groups + 1, budget + 1, dtype=torch.bool, device=device)
    mask[0, 0] = True
    for step in range(groups):
        for cost in costs:
            if cost <= budget:
                mask[step + 1, cost:] |= mask[step, :budget + 1 - cost]
    return mask


def _dp_step(previous, scores_g, costs, budget, reachable):
    """One group transition; only reachable rates enter ``logsumexp``."""
    modes = scores_g.shape[0]
    candidates = previous.new_full((modes, budget + 1), float("-inf"))
    for mode, cost in enumerate(costs):
        candidates[mode, cost:] = previous[:budget + 1 - cost] + scores_g[mode]
    current = previous.new_full((budget + 1,), float("-inf"))
    current = current.clone()
    current[reachable] = torch.logsumexp(candidates[:, reachable], dim=0)
    return current


def _compute_dp_tables(scores, costs, budget, reach_forward, reach_backward):
    """Static ``[G+1, B+1]`` forward/backward log-DP tables."""
    groups, _ = scores.shape
    forward_rows = [scores.new_full((budget + 1,), float("-inf"))]
    forward_rows[0] = forward_rows[0].clone()
    forward_rows[0][0] = 0.0
    for group in range(groups):
        forward_rows.append(_dp_step(
            forward_rows[-1], scores[group], costs, budget,
            reach_forward[group + 1]))
    forward = torch.stack(forward_rows)

    backward_rows = [scores.new_full((budget + 1,), float("-inf"))]
    backward_rows[0] = backward_rows[0].clone()
    backward_rows[0][0] = 0.0
    # Walk groups from the end; ``reach_backward[g] == reach_forward[G - g]``.
    for group in range(groups - 1, -1, -1):
        backward_rows.append(_dp_step(
            backward_rows[-1], scores[group], costs, budget,
            reach_backward[group]))
    backward = torch.stack(list(reversed(backward_rows)))
    return forward, backward


def _compute_marginals(scores, forward, backward, forward_mask, backward_mask,
                       costs, budget):
    """Vectorized per-group mode marginals under the exact-budget DP."""
    log_z = forward[-1, budget]
    groups, modes = scores.shape
    prefixes = torch.arange(budget + 1, device=scores.device)
    suffix = budget - prefixes[None, None, :] - costs[None, :, None]
    suffix_clamped = suffix.clamp(min=0, max=budget)
    forward_ok = forward_mask[:-1, None, :]
    backward_at = torch.gather(
        backward_mask[1:, None, :].expand(groups, modes, budget + 1),
        2,
        suffix_clamped.expand(groups, modes, budget + 1))
    valid = (suffix >= 0) & forward_ok & backward_at
    backward_term = torch.gather(
        backward[1:, None, :].expand(groups, modes, budget + 1),
        2,
        suffix_clamped.expand(groups, modes, budget + 1))
    terms = (forward[:-1, None, :]
             + scores[:, :, None]
             + backward_term
             - log_z)
    terms = terms.masked_fill(~valid, float("-inf"))
    return torch.exp(torch.logsumexp(terms, dim=-1))


_COMPUTE_DP_TABLES = _compute_dp_tables
_COMPUTE_MARGINALS = _compute_marginals


def _enable_compiled_marginals():
    """Best-effort compile for the dense marginal gather (optional).

    Full DP tables stay eager: reachable-state ``index`` has dynamic shape and
    breaks ``torch.compile`` on this stack while remaining required for safe
    ``logsumexp`` gradients.
    """
    global _COMPUTE_MARGINALS
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return
    try:
        _COMPUTE_MARGINALS = compile_fn(
            _compute_marginals, fullgraph=False, dynamic=False)
    except Exception:
        _COMPUTE_MARGINALS = _compute_marginals


_enable_compiled_marginals()


@dataclass
class FixedBudgetDistribution:
    """One-use differentiable distribution built from current policy logits."""

    scores: torch.Tensor
    shifted_costs: tuple
    shifted_budget: int
    forward: torch.Tensor
    forward_mask: torch.Tensor
    backward: torch.Tensor
    backward_mask: torch.Tensor
    _cost_tensor: torch.Tensor

    @property
    def groups(self):
        return int(self.scores.shape[0])

    @property
    def modes(self):
        return int(self.scores.shape[1])

    def log_partition(self):
        return self.forward[-1, self.shifted_budget]

    def marginals(self):
        return _COMPUTE_MARGINALS(
            self.scores, self.forward, self.backward,
            self.forward_mask, self.backward_mask,
            self._cost_tensor, self.shifted_budget)

    def entropy(self, probabilities=None):
        if probabilities is None:
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
        totals = self._cost_tensor[allocations].sum(dim=1)
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
        costs = self._cost_tensor
        masks = self.backward_mask
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
    def sample_conditioned(self, group, mode, count=1,
                           generator: Optional[torch.Generator] = None):
        """Sample exact-budget allocations with one group-mode pair forced."""
        group, mode = int(group), int(mode)
        if not 0 <= group < self.groups or not 0 <= mode < self.modes:
            raise ValueError("conditioned group or mode is out of range")
        scores = self.scores.detach().clone()
        scores[group].fill_(float("-inf"))
        scores[group, mode] = 0.0
        forward, backward = _COMPUTE_DP_TABLES(
            scores, self.shifted_costs, self.shifted_budget,
            self.forward_mask, self.backward_mask)
        if not bool(torch.isfinite(forward[-1, self.shifted_budget])):
            raise ValueError("forced group-mode pair makes budget unreachable")
        conditioned = FixedBudgetDistribution(
            scores=scores, shifted_costs=self.shifted_costs,
            shifted_budget=self.shifted_budget, forward=forward,
            forward_mask=self.forward_mask, backward=backward,
            backward_mask=self.backward_mask,
            _cost_tensor=self._cost_tensor)
        samples = conditioned.sample(count, generator=generator)
        if not bool((samples[:, group] == mode).all()):
            raise RuntimeError("conditioned sampler violated its forced mode")
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
        # Reachability depends only on (groups, costs, budget); cache once.
        reach = _reachability_mask(
            shifted, budget, self.groups, torch.device("cpu"))
        self.register_buffer("_reach_forward", reach, persistent=False)
        self.register_buffer(
            "_reach_backward", torch.flip(reach, dims=(0,)), persistent=False)
        self.register_buffer(
            "_shifted_cost_tensor",
            torch.as_tensor(shifted, dtype=torch.long),
            persistent=False)

    def build(self, temperature=1.0, validate=True):
        temperature = float(temperature)
        if not math.isfinite(temperature) or not temperature > 0:
            raise ValueError("temperature must be finite and positive")
        if validate and not bool(torch.isfinite(self.logits).all()):
            raise ValueError("policy logits must be finite")
        # Constructor already proved the budget is reachable; skip the CUDA
        # syncing ``bool(mask)`` check on the training hot path.
        if validate and not bool(
                self._reach_forward[-1, self.shifted_budget]):
            raise RuntimeError("fixed budget unexpectedly became unreachable")
        # Training centers each row after an optimizer step.  Keep the raw
        # logits here so log_partition retains its literal mathematical value
        # for arbitrary user-supplied logits and remains testable by enumeration.
        scores = self.logits / temperature
        reach_forward = self._reach_forward
        reach_backward = self._reach_backward
        if reach_forward.device != scores.device:
            reach_forward = reach_forward.to(device=scores.device, non_blocking=True)
            reach_backward = reach_backward.to(
                device=scores.device, non_blocking=True)
        forward, backward = _COMPUTE_DP_TABLES(
            scores, self.shifted_costs, self.shifted_budget,
            reach_forward, reach_backward)
        return FixedBudgetDistribution(
            scores=scores,
            shifted_costs=self.shifted_costs,
            shifted_budget=self.shifted_budget,
            forward=forward,
            forward_mask=reach_forward,
            backward=backward,
            backward_mask=reach_backward,
            _cost_tensor=self._shifted_cost_tensor.to(
                device=scores.device, non_blocking=True),
        )

    def actual_rate(self, allocations):
        allocations = torch.as_tensor(
            allocations, dtype=torch.long, device=self.logits.device)
        costs = torch.as_tensor(
            self.bit_costs, dtype=torch.long, device=self.logits.device)
        return costs[allocations].sum(dim=-1)
