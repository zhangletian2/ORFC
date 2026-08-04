"""CPU-only exact-budget and geometry contracts for V20."""

import torch

from multimode_pq import MultiModeSoftPQ
from ..v12.allocation_policy import FixedBudgetAllocationPolicy
from ..v12.strict_fair import strict_fair_slate
from .config import SPECS


def test_profiles():
    for groups, dim, anchor in SPECS.values():
        slate = strict_fair_slate(
            groups, anchor.mode_bits, anchor.rate, torch.device("cpu"))
        policy = FixedBudgetAllocationPolicy(
            groups, anchor.mode_bits, anchor.rate)
        assert slate.shape == (len(anchor.mode_bits), groups)
        assert bool((policy.actual_rate(slate) == anchor.rate).all())
        assert MultiModeSoftPQ(groups, anchor.mode_sizes, dim).G == groups
        assert groups * dim == 1024


if __name__ == "__main__":
    test_profiles()
    print("V20_CONTRACTS_PASS")
