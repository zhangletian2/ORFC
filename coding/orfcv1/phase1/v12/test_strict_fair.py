"""Small exhaustive contracts for strict fair exact-budget slates."""

import torch

from .strict_fair import strict_fair_slate
from .train import fair_policy_weights


class _Distribution:
    def log_prob(self, allocations, validate=False):
        del validate
        return torch.tensor([-4.0, -2.0, 0.0], device=allocations.device)


def main():
    for bits, rate in (((1, 2, 3), 64), ((2, 3, 4), 96)):
        slate = strict_fair_slate(32, bits, rate, torch.device("cpu"))
        costs = torch.tensor(bits)
        assert slate.shape == (3, 32)
        assert bool((costs[slate].sum(1) == rate).all())
        assert bool(torch.equal(
            torch.sort(slate, dim=0).values,
            torch.arange(3)[:, None].expand_as(slate)))
        weights = fair_policy_weights(_Distribution(), slate, 1e-4)
        assert torch.isclose(weights.sum(), torch.tensor(1.0))
        assert bool((weights >= 1e-4).all())
    print("strict fair slate contracts: PASS")


if __name__ == "__main__":
    main()
