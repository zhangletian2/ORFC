"""Contracts for the ``use_L=False`` ablation and the exact ``L`` fold."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from . import checkpoint as ckpt
from . import quantise as Q
from .codec import V33Codec


def _random_codec(groups=6, bits=(1, 2, 3), dim=8, seed=0, use_L=True):
    torch.manual_seed(seed)
    codec = V33Codec.build(
        groups, bits, dim, parameterization="orfc_cayley", use_L=use_L)
    for stage in codec.pq.stages:
        nn.init.normal_(stage.codebooks, std=0.5)
    if use_L:
        nn.init.normal_(codec.L.triu_params, std=0.05)
    nn.init.normal_(codec.transform.triu_params, std=0.02)
    return codec.eval()


class NoLContracts(unittest.TestCase):

    def setUp(self):
        self.groups, self.bits, self.dim = 6, (1, 2, 3), 8
        torch.manual_seed(7)
        self.y = torch.randn(2, 5, self.groups * self.dim)
        self.allocation = torch.tensor([0, 2, 1, 2, 0, 1])

    def test_fold_is_exact(self):
        codec = _random_codec(self.groups, self.bits, self.dim)
        with torch.no_grad():
            before, labels = Q.quantise_sparse(codec, self.y, self.allocation)
        codec.fold_L_into_codebooks(self.allocation)
        self.assertFalse(codec.uses_L)
        with torch.no_grad():
            after, folded_labels = Q.quantise_sparse(
                codec, self.y, self.allocation)
        torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-5)
        self.assertTrue(torch.equal(labels, folded_labels))

    def test_no_L_codec_is_pure_u0(self):
        codec = _random_codec(use_L=False)
        self.assertIsNone(codec.L)
        self.assertFalse(codec.uses_L)
        self.assertEqual(codec.l_orth_error(), 0.0)
        names = [g["name"] for g in codec.parameter_groups()]
        self.assertEqual(names, ["u0", "pq"])
        with torch.no_grad():
            decoded, _ = Q.quantise_sparse(codec, self.y, self.allocation)
        self.assertEqual(decoded.shape, self.y.shape)

    def test_frozen_rotations_without_bank(self):
        codec = _random_codec(use_L=False)
        with torch.no_grad():
            plain, _ = Q.quantise_sparse(codec, self.y, self.allocation)
            with Q.frozen_rotations(codec):
                cached, _ = Q.quantise_sparse(codec, self.y, self.allocation)
        torch.testing.assert_close(plain, cached, rtol=1e-6, atol=1e-6)

    def test_checkpoint_roundtrip_both_ways(self):
        for use_L in (True, False):
            with self.subTest(use_L=use_L), tempfile.TemporaryDirectory() as tmp:
                codec = _random_codec(use_L=use_L)
                path = Path(tmp) / "checkpoint.pt"
                ckpt.save_checkpoint(codec, path, meta={"step": 1})
                restored, payload = ckpt.load_checkpoint(path)
                self.assertIs(payload["geometry"]["use_L"], use_L)
                self.assertEqual(restored.uses_L, use_L)
                with torch.no_grad():
                    a, _ = Q.quantise_sparse(codec, self.y, self.allocation)
                    b, _ = Q.quantise_sparse(restored, self.y, self.allocation)
                torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)

    def test_legacy_payload_without_use_L_key(self):
        codec = _random_codec(use_L=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            ckpt.save_checkpoint(codec, path, meta={"step": 1})
            payload = torch.load(path, weights_only=False)
            payload["geometry"].pop("use_L")
            torch.save(payload, path)
            restored, _ = ckpt.load_checkpoint(path)
        self.assertTrue(restored.uses_L)


if __name__ == "__main__":
    unittest.main()
