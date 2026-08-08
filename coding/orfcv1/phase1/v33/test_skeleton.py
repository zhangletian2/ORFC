"""CPU/GPU contracts for the V33 U0+L+nested skeleton."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from phase1.v30 import nested

from .codec import V33Codec
from .quantise import linear_decode, linear_encode, quantise_sparse, reconstruct
from .transform import ConditionalBlockDiagL, skew_param_count


class TinyIdentityTail(nn.Module):
    def forward(self, x):
        return x


class V33SkeletonContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(33)
        self.G, self.d, self.T = 4, 8, 3
        self.bits = (1, 2, 3)
        self.device = torch.device("cpu")

    def _codec(self, groups=None, dim=None, seed_books=True):
        groups = self.G if groups is None else int(groups)
        dim = self.d if dim is None else int(dim)
        codec = V33Codec.build(
            groups, self.bits, dim, parameterization="direct",
            device=self.device)
        if seed_books:
            for stage in codec.pq.stages:
                nn.init.normal_(stage.codebooks, std=0.05)
        return codec

    def test_skew_count_and_bank_shape(self):
        self.assertEqual(skew_param_count(32), 496)
        bank = ConditionalBlockDiagL(32, 3, 32)
        self.assertEqual(tuple(bank.triu_params.shape), (32, 3, 496))
        rotations = bank.get_rotations()
        self.assertEqual(tuple(rotations.shape), (32, 3, 32, 32))

    def test_identity_and_random_orthogonality(self):
        bank = ConditionalBlockDiagL(32, 3, 32)
        self.assertLess(bank.orth_error(), 1e-5)
        nn.init.normal_(bank.triu_params, std=0.02)
        self.assertLess(bank.orth_error(), 1e-4)

    def test_linear_roundtrip(self):
        codec = self._codec(seed_books=False)
        nn.init.normal_(codec.L.triu_params, std=0.03)
        with torch.no_grad():
            u0 = codec.transform.get_rotation()
            q, _ = torch.linalg.qr(torch.randn(u0.shape))
            codec.transform.rotation.copy_(q)
        y = torch.randn(2, 5, self.G * self.d)
        allocation = torch.tensor([0, 2, 1, 0])
        sub_L, _ = linear_encode(codec, y, allocation)
        recovered = linear_decode(
            codec, sub_L, allocation, y.shape[0], y.shape[1])
        self.assertTrue(torch.allclose(recovered, y, atol=1e-5, rtol=1e-5))

    def test_allocation_conditions_L(self):
        bank = ConditionalBlockDiagL(self.G, self.T, self.d)
        bank.reset_identity()
        # Only group 1 differs across modes; other groups stay identity.
        bank.triu_params.data[1, 2].fill_(0.04)
        a0 = torch.zeros(self.G, dtype=torch.long)
        a2 = torch.full((self.G,), 2, dtype=torch.long)
        gap = (bank.select(a0)[1] - bank.select(a2)[1]).norm().detach()
        self.assertGreater(float(gap), 1e-3)
        sub = torch.randn(self.G, 6, self.d)
        rotated0 = bank.rotate_groups(sub, a0)
        rotated2 = bank.rotate_groups(sub, a2)
        self.assertGreater(
            float((rotated0[1] - rotated2[1]).norm().detach()), 1e-3)
        self.assertTrue(torch.allclose(rotated0[0], rotated2[0], atol=1e-6))
        self.assertTrue(torch.allclose(rotated0[0], sub[0], atol=1e-6))

    def test_identity_L_matches_nested_reconstruct(self):
        codec = self._codec()
        codec.L.reset_identity()
        nested_codec = nested.NestedCodec(
            codec.transform, codec.pq)
        y = torch.randn(3, 4, self.G * self.d)
        allocation = torch.tensor([0, 1, 2, 1])
        got, got_labels = reconstruct(codec, y, allocation)
        ref, ref_labels = nested.reconstruct(
            nested_codec, y, allocation)
        self.assertTrue(torch.equal(got_labels, ref_labels))
        self.assertTrue(torch.allclose(got, ref, atol=1e-6, rtol=1e-6))

    def test_freeze_u0_stops_u0_grads(self):
        codec = self._codec()
        nn.init.normal_(codec.L.triu_params, std=0.02)
        y = torch.randn(2, 3, self.G * self.d, requires_grad=False)
        allocation = torch.tensor([1, 0, 2, 1])
        codec.freeze_u0()
        self.assertTrue(codec.u0_frozen)
        decoded, _ = quantise_sparse(codec, y, allocation)
        decoded.square().sum().backward()
        self.assertIsNone(codec.transform.rotation.grad)
        self.assertIsNotNone(codec.L.triu_params.grad)
        self.assertTrue(
            any(stage.codebooks.grad is not None for stage in codec.pq.stages))

    def test_full_geometry_bank_smoke(self):
        """R64 geometry: G=32, d=32, T=3 → 96 SO(32) cells."""
        codec = V33Codec.build(32, (1, 2, 3), 32)
        self.assertEqual(tuple(codec.L.triu_params.shape), (32, 3, 496))
        nn.init.normal_(codec.L.triu_params, std=0.01)
        self.assertLess(codec.L.orth_error(), 1e-4)
        y = torch.randn(1, 2, 1024)
        allocation = torch.zeros(32, dtype=torch.long)
        allocation[::3] = 1
        allocation[1::3] = 2
        decoded, labels = quantise_sparse(codec, y, allocation)
        self.assertEqual(tuple(decoded.shape), (1, 2, 1024))
        self.assertEqual(tuple(labels.shape), (32, 2))

    def test_parameter_groups_for_phase1(self):
        codec = self._codec()
        groups = {g["name"]: g for g in codec.parameter_groups(l_decay=1e-4)}
        self.assertIn("u0", groups)
        self.assertEqual(groups["L"]["weight_decay"], 1e-4)
        codec.freeze_u0()
        names = {g["name"] for g in codec.parameter_groups(l_decay=1e-4)}
        self.assertNotIn("u0", names)
        self.assertEqual(names, {"L", "pq"})


if __name__ == "__main__":
    unittest.main()
