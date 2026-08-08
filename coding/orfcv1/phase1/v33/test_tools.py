"""Unit / smoke contracts for V33 noise-floor, switch-probe, and search tools."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from . import checkpoint as ckpt
from . import noise_floor
from . import ranking
from . import search
from . import switch_probe
from . import verify
from .codec import V33Codec


class RankingContracts(unittest.TestCase):
    def test_sparse_kendall_perfect_and_ties(self):
        left = np.array([1.0, 2.0, 3.0, 4.0])
        right = np.array([0.5, 1.5, 2.5, 3.5])
        tau, c, d, decisive, dropped = ranking.sparse_kendall_tau(
            left, right, tie_threshold=0.0)
        self.assertEqual(tau, 1.0)
        self.assertEqual(decisive, 6)
        self.assertEqual(dropped, 0)

        # Large threshold drops everything.
        tau2, _, _, decisive2, dropped2 = ranking.sparse_kendall_tau(
            left, right, tie_threshold=10.0)
        self.assertIsNone(tau2)
        self.assertEqual(decisive2, 0)
        self.assertEqual(dropped2, 6)


class NoiseFloorContracts(unittest.TestCase):
    def test_sigma_from_known_noise(self):
        rng = np.random.default_rng(0)
        true = rng.normal(size=12)
        sigma = 0.07
        matrix = true[None, :] + rng.normal(scale=sigma, size=(8, 12))
        stats = noise_floor.estimate_sigma_noise(matrix)
        # RMS pair-std should be near sqrt(2)*sigma for independent noise.
        expected = np.sqrt(2) * sigma
        self.assertLess(abs(stats["sigma_noise"] - expected) / expected, 0.35)

    def test_calibrate_sets_threshold(self):
        rng = np.random.default_rng(1)
        matrix = rng.normal(size=(5, 10))
        report = noise_floor.calibrate(matrix, tie_mult=2.0)
        self.assertEqual(report["tie_threshold"], 2.0 * report["sigma_noise"])
        self.assertIn("subset0_vs_subset1_sparse_tau", report)


class CheckpointContracts(unittest.TestCase):
    def test_roundtrip(self):
        codec = V33Codec.build(4, (1, 2, 3), 8, parameterization="direct")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "codec.pt"
            ckpt.save_checkpoint(codec, path, meta={"step": 3, "phase": 1})
            loaded, payload = ckpt.load_checkpoint(path, device="cpu")
            self.assertEqual(payload["format"], ckpt.FORMAT)
            self.assertEqual(payload["meta"]["step"], 3)
            U, geometry, meta = ckpt.load_u0_rotation(path, device="cpu")
            self.assertEqual(tuple(U.shape), (32, 32))
            self.assertEqual(geometry["groups"], 4)
            self.assertTrue(torch.allclose(
                loaded.transform.get_rotation(), U, atol=1e-6))


class SearchContracts(unittest.TestCase):
    def test_two_group_preserves_rate(self):
        bits = (1, 2, 3)
        base = search.uniform_allocation(8, bits, 16)
        neighbors, moves = search.two_group_transfers(base, bits)
        self.assertGreater(len(neighbors), 0)
        for row in neighbors:
            self.assertEqual(search.nominal_rate(row, bits), 16)
        self.assertEqual(len(neighbors), len(moves))

    def test_smoke_multi_start_and_certificate(self):
        groups, bits, rate = 8, (1, 2, 3), 16
        score, table = search.synthetic_score_fn(bits, rate, seed=33)(groups)
        result = search.run_search(
            score, score, groups, bits, rate,
            eval_budget=600, propose_top_k=20, max_steps=16,
            n_random_starts=2, topk=3, seed=33, simplified=False,
            certify=True)
        self.assertIsNotNone(result["winner"])
        self.assertIsNotNone(result["certificate"])
        self.assertLessEqual(result["body"]["evals_used"], 600)
        # Certificate should be boolean.
        self.assertIn(result["certificate"]["is_1opt"], (True, False))

    def test_cost_table_proposal_is_prefix(self):
        groups, bits, rate = 8, (1, 2, 3), 16
        score, _ = search.synthetic_score_fn(bits, rate, seed=1)(groups)
        base = search.uniform_allocation(groups, bits, rate)
        costs, _ = search.build_cost_table(score, base, len(bits))
        neighbors, moves = search.two_group_transfers(base, bits)
        top, top_moves, deltas = search.propose_top(
            neighbors, moves, costs, base, top=10)
        self.assertEqual(len(top), 10)
        self.assertTrue(np.all(np.diff(deltas) >= -1e-12))


class SwitchProbeContracts(unittest.TestCase):
    def test_overlap_identity_is_one(self):
        groups, dim = 4, 8
        U = torch.eye(groups * dim)
        o = switch_probe.group_subspace_overlap(U, U, groups, dim)
        self.assertTrue(np.allclose(o, 1.0, atol=1e-5))

    def test_suggest_requires_past_peak(self):
        # Rising then flat — peak at end → must refuse.
        curve = [
            {"drift": {"drift_max": 0.0}, "search_hamming": 0,
             "kendall": {"sparse_kendall_tau": 0.5 + 0.1 * i}}
            for i in range(5)
        ]
        decision = switch_probe.suggest_switch(curve, window=2)
        self.assertFalse(decision["suggest"])
        self.assertEqual(decision["reason"], "have_not_passed_peak")

    def test_suggest_both_stable_past_peak(self):
        curve = []
        taus = [0.5, 0.7, 0.95, 0.9, 0.88, 0.87]
        drifts = [0.05, 0.02, 0.01, 5e-4, 2e-4, 1e-4]
        moves = [5, 3, 1, 0, 0, 0]
        for tau, drift, move in zip(taus, drifts, moves):
            curve.append({
                "drift": {"drift_max": drift},
                "search_hamming": move,
                "kendall": {"sparse_kendall_tau": tau},
            })
        decision = switch_probe.suggest_switch(
            curve, drift_stable_max=1e-3, move_stable_max=0, window=3)
        self.assertTrue(decision["suggest"])
        self.assertEqual(decision["peak_index"], 2)

    def test_smoke_cli_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "probe.json"
            switch_probe.main([
                "--smoke", "--out", str(out),
                "--n-candidates", "6", "--window", "2",
                "--drift-stable-max", "1.0", "--move-stable-max", "100",
            ])
            payload = json.loads(out.read_text())
            self.assertTrue(payload["smoke"])
            self.assertGreaterEqual(len(payload["curve"]), 2)
            self.assertIn("note", payload["curve"][0])


class CliSmoke(unittest.TestCase):
    def test_noise_floor_smoke_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "noise.json"
            noise_floor.main([
                "--smoke", "--out", str(out),
                "--n-parts", "4", "--n-alloc", "8",
            ])
            payload = json.loads(out.read_text())
            self.assertTrue(payload["smoke"])
            self.assertGreater(payload["sigma_noise"], 0.0)

    def test_search_smoke_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "search.json"
            search.main([
                "--smoke", "--out", str(out),
                "--eval-budget", "500", "--propose-top", "15",
                "--max-steps", "10", "--random-starts", "2", "--topk", "2",
            ])
            payload = json.loads(out.read_text())
            self.assertTrue(payload["smoke"])
            self.assertIsNotNone(payload["winner"])


class VerifyContracts(unittest.TestCase):
    def test_nominal_rate_exact(self):
        bits = (1, 2, 3)
        allocation = search.uniform_allocation(8, bits, 16)
        report = verify.check_nominal_rate(allocation, bits, 16)
        self.assertTrue(report["passed"])
        self.assertEqual(report["nominal_rate"], 16)

    def test_topk_ranking_consistent_and_discordant(self):
        ok = verify.check_topk_ranking_consistency(
            [3.0, 1.0, 2.0], [2.9, 1.1, 2.05], ids=["a", "b", "c"])
        self.assertTrue(ok["argsort_identical"])
        self.assertTrue(ok["top1_identical"])
        self.assertEqual(ok["sparse_kendall_tau"], 1.0)

        bad = verify.check_topk_ranking_consistency(
            [1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        self.assertFalse(bad["argsort_identical"])
        self.assertFalse(bad["top1_identical"])
        self.assertEqual(bad["sparse_kendall_tau"], -1.0)

    def test_one_opt_on_synthetic(self):
        groups, bits, rate = 8, (1, 2, 3), 16
        score, _ = search.synthetic_score_fn(bits, rate, seed=7)(groups)
        allocation = search.uniform_allocation(groups, bits, rate)
        cert = verify.check_one_opt(score, allocation, bits)
        self.assertIn("is_1opt", cert)
        self.assertEqual(cert["neighborhood_size"],
                         len(search.two_group_transfers(allocation, bits)[0]))

    def test_smoke_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "verify.json"
            code = verify.main(["--smoke", "--out", str(out)])
            self.assertEqual(code, 0)
            payload = json.loads(out.read_text())
            self.assertTrue(payload["smoke"])
            self.assertTrue(payload["passed"])
            self.assertTrue(payload["nominal_rate"]["passed"])
            self.assertIn("is_1opt", payload["one_opt"])
            self.assertTrue(payload["vs_orfc"]["skipped"])
            self.assertEqual(payload["rans_downstream"]["status"], "TODO")

    def test_print_pipeline(self):
        code = verify.main(["--print-pipeline"])
        self.assertEqual(code, 0)

    def test_default_orfc_ref_exists(self):
        path = verify.default_orfc_ref("R64")
        self.assertTrue(path.exists(), msg=str(path))
        stem, pt, npz = verify.resolve_orfc_paths(None, anchor_name="R64")
        self.assertEqual(Path(pt), path)
        self.assertTrue(Path(npz).exists())


if __name__ == "__main__":
    unittest.main()
