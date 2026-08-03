"""Strictly fair exact-budget slates for shared-codec training."""

from functools import lru_cache
from itertools import permutations

import torch


@lru_cache(maxsize=None)
def _base_slate(groups, bits, rate):
    """Return M allocations: every group uses every mode once, each at rate R."""
    groups, bits, rate = int(groups), tuple(map(int, bits)), int(rate)
    modes = len(bits)
    if modes < 2 or modes > 3:
        raise ValueError("strict fairness currently supports two or three modes")
    if groups * sum(bits) != modes * rate:
        raise ValueError("full mode fairness is incompatible with this rate")
    options = tuple(permutations(range(modes)))
    target = (rate,) * (modes - 1)
    states = {(0,) * (modes - 1): None}
    parents = []
    for group in range(groups):
        remaining = groups - group - 1
        current = {}
        for state in states:
            for option in options:
                value = tuple(state[row] + bits[option[row]]
                              for row in range(modes - 1))
                if any(x > rate or x + remaining * min(bits) > rate
                       or x + remaining * max(bits) < rate for x in value):
                    continue
                current.setdefault(value, (state, option))
        if not current:
            raise ValueError("no strictly fair exact-budget slate exists")
        parents.append(current)
        states = current
    if target not in states:
        raise ValueError("no strictly fair exact-budget slate reaches the rate")
    chosen, state = [], target
    for table in reversed(parents):
        previous, option = table[state]
        chosen.append(option)
        state = previous
    slate = torch.tensor(list(reversed(chosen)), dtype=torch.long).t()
    costs = torch.tensor(bits, dtype=torch.long)
    assert bool((costs[slate].sum(1) == rate).all())
    assert bool(torch.equal(
        torch.sort(slate, dim=0).values,
        torch.arange(modes)[:, None].expand_as(slate)))
    return tuple(tuple(map(int, row)) for row in slate.tolist())


@torch.no_grad()
def strict_fair_slate(groups, bits, rate, device, generator=None):
    """Randomly relabel a deterministic fair slate without breaking fairness."""
    base = torch.tensor(_base_slate(groups, tuple(bits), rate), device=device)
    group_order = torch.randperm(groups, device=device, generator=generator)
    row_order = torch.randperm(len(bits), device=device, generator=generator)
    return base.index_select(1, group_order).index_select(0, row_order)
