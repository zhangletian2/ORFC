"""Contracts for hard-forward DP/codeword soft-backward quantisation."""

import unittest

import torch
import torch.nn as nn

from .allocation_policy import FixedBudgetAllocationPolicy
from .qhard import quantise


class _Transform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rotation = nn.Parameter(torch.eye(dim))

    def get_rotation(self):
        return self.rotation


class _Quantizer(nn.Module):
    def __init__(self, groups, size, dim):
        super().__init__()
        self.codebooks = nn.Parameter(torch.randn(groups, size, dim))


class _PQ(nn.Module):
    def __init__(self, groups, sizes, dim):
        super().__init__()
        self.G, self.d = groups, dim
        self.quantizers = nn.ModuleList([
            _Quantizer(groups, size, dim) for size in sizes])


class _Codec(nn.Module):
    def __init__(self, groups, sizes, dim):
        super().__init__()
        self.transform = _Transform(groups * dim)
        self.pq = _PQ(groups, sizes, dim)


class DPSTContracts(unittest.TestCase):
    def test_forward_is_hard_and_backward_reaches_every_mode(self):
        torch.manual_seed(7)
        groups, dim, bits = 4, 2, (1, 2, 3)
        codec = _Codec(groups, tuple(2 ** bit for bit in bits), dim)
        policy = FixedBudgetAllocationPolicy(groups, bits, 8)
        distribution = policy.build(1.0)
        allocations = distribution.sample(
            3, generator=torch.Generator().manual_seed(11))
        y = torch.randn(2, 3, groups * dim)
        hard, _ = quantise(codec, y, allocations)
        relaxed, _ = quantise(
            codec, y, allocations, marginals=distribution.marginals(),
            codeword_temperature=0.5)
        torch.testing.assert_close(relaxed, hard)
        relaxed.square().mean().backward()
        self.assertGreater(float(policy.logits.grad.norm()), 0)
        self.assertGreater(float(codec.transform.rotation.grad.norm()), 0)
        for quantizer in codec.pq.quantizers:
            per_group = quantizer.codebooks.grad.square().sum((1, 2))
            self.assertTrue(bool((per_group > 0).all()))


if __name__ == "__main__":
    unittest.main()
