"""Top-k multiple-choice knapsack for group-wise bit allocations."""

from __future__ import annotations


def topk_allocations(costs, bit_costs, budget, topk=8):
    """Return the lowest-cost exact-budget allocations.

    ``costs[g][m]`` is the local cost of assigning mode ``m`` to group ``g``.
    Ties are deterministic: allocation tuples are ordered lexicographically.
    """
    costs = tuple(tuple(map(float, row)) for row in costs)
    bits = tuple(map(int, bit_costs))
    budget, topk = int(budget), int(topk)
    if not costs or topk < 1 or len(bits) != len(costs[0]):
        raise ValueError("invalid costs, bit costs, or top-k")
    if any(len(row) != len(bits) for row in costs):
        raise ValueError("all groups must have the same mode count")

    states = {0: [(0.0, ())]}
    for row in costs:
        updated = {}
        for used, entries in states.items():
            for value, prefix in entries:
                for mode, (bit, cost) in enumerate(zip(bits, row)):
                    target = used + bit
                    if target <= budget:
                        updated.setdefault(target, []).append(
                            (value + cost, prefix + (mode,)))
        states = {}
        for used, entries in updated.items():
            entries.sort(key=lambda item: (item[0], item[1]))
            states[used] = entries[:topk]
    if budget not in states:
        raise ValueError(f"budget {budget} is unreachable")
    return states[budget]
