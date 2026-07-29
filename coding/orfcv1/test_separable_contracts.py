#!/usr/bin/env python3
"""CPU contract tests for the separable projection redefinition."""

import unittest

import numpy as np

from separable_projection import (
    attainable_rank, design_matrix, dp_summary, dp_topk, fit_separable,
    gauge_null_space, leave_one_out_residual, range_gradient_weights,
    residual_operator, separable_values, strip_gauge,
)


def enumerate_fixed(groups, bits, budget):
    rows, modes = [], len(bits)

    def walk(group, used, path):
        if group == groups:
            if used == budget:
                rows.append(tuple(path))
            return
        for mode in range(modes):
            if used + bits[mode] <= budget:
                walk(group + 1, used + bits[mode], path + [mode])

    walk(0, 0, [])
    return np.asarray(rows, dtype=np.int64)


class SeparableContracts(unittest.TestCase):
    def setUp(self):
        self.groups, self.bits, self.budget = 5, (1, 2, 3), 10
        self.modes = len(self.bits)
        self.allocations = enumerate_fixed(self.groups, self.bits, self.budget)
        rng = np.random.default_rng(0)
        self.table = rng.normal(size=(self.groups, self.modes))

    def test_design_rank_is_capped_by_the_gauge(self):
        design = design_matrix(self.allocations, self.modes)
        values = np.linalg.svd(design, compute_uv=False)
        rank = int((values > values.max() * 1e-10).sum())
        self.assertEqual(rank, attainable_rank(self.groups, self.modes))

    def test_gauge_directions_leave_feasible_values_unchanged(self):
        null = gauge_null_space(
            self.groups, self.modes, self.bits, self.budget)
        self.assertEqual(null.shape[0], self.groups)
        base = separable_values(
            self.table.reshape(-1), self.allocations, self.modes)
        shifted = self.table.reshape(-1) + 3.7 * null[0] - 1.2 * null[-1]
        moved = separable_values(shifted, self.allocations, self.modes)
        np.testing.assert_allclose(base, moved, atol=1e-9)

    def test_gauge_directions_leave_the_dp_solution_unchanged(self):
        null = gauge_null_space(
            self.groups, self.modes, self.bits, self.budget)
        first = dp_summary(self.table, self.bits, self.budget)
        shifted = (self.table.reshape(-1) + 2.5 * null[1]).reshape(
            self.groups, self.modes)
        second = dp_summary(shifted, self.bits, self.budget)
        self.assertAlmostEqual(first["span"], second["span"], places=9)
        self.assertAlmostEqual(first["top2_gap"], second["top2_gap"], places=9)
        np.testing.assert_array_equal(first["argmin"], second["argmin"])

    def test_strip_gauge_removes_only_the_unidentified_part(self):
        null = gauge_null_space(
            self.groups, self.modes, self.bits, self.budget)
        delta = np.random.default_rng(1).normal(size=self.groups * self.modes)
        stripped = strip_gauge(delta, null)
        np.testing.assert_allclose(null @ stripped, 0.0, atol=1e-9)
        base = separable_values(delta, self.allocations, self.modes)
        kept = separable_values(stripped, self.allocations, self.modes)
        np.testing.assert_allclose(base - base.mean(), kept - kept.mean(),
                                   atol=1e-9)

    def test_separable_measurements_are_fitted_exactly(self):
        values = separable_values(
            self.table.reshape(-1), self.allocations, self.modes)
        fitted = fit_separable(values, self.allocations, self.modes)
        self.assertLess(np.abs(fitted["residual"]).max(), 1e-9)

    def test_projector_is_idempotent_and_orthogonal_to_the_design(self):
        design = design_matrix(self.allocations, self.modes)
        operator, rank = residual_operator(design)
        self.assertEqual(rank, attainable_rank(self.groups, self.modes))
        np.testing.assert_allclose(operator @ operator, operator, atol=1e-9)
        np.testing.assert_allclose(operator @ design, 0.0, atol=1e-9)

    def test_dp_matches_brute_force_on_the_full_feasible_set(self):
        values = separable_values(
            self.table.reshape(-1), self.allocations, self.modes)
        report = dp_summary(self.table, self.bits, self.budget)
        order = np.sort(values)
        self.assertAlmostEqual(report["minimum"], order[0], places=9)
        self.assertAlmostEqual(report["maximum"], order[-1], places=9)
        self.assertAlmostEqual(report["second"], order[1], places=9)
        top = dp_topk(self.table, self.bits, self.budget, count=4)
        np.testing.assert_allclose(
            [value for value, _ in top], order[:4], atol=1e-9)

    def test_range_weights_reproduce_the_exact_gradient(self):
        rng = np.random.default_rng(2)
        measured = separable_values(
            self.table.reshape(-1), self.allocations, self.modes)
        measured = measured + rng.normal(scale=0.3, size=len(measured))
        design = design_matrix(self.allocations, self.modes)
        operator, _ = residual_operator(design)
        residual = operator @ measured
        weights, high, low = range_gradient_weights(residual, operator)
        self.assertAlmostEqual(
            float(weights @ measured), float(residual[high] - residual[low]),
            places=9)
        step = 1e-6
        for index in rng.permutation(len(measured))[:5]:
            bumped = measured.copy()
            bumped[index] += step
            moved = operator @ bumped
            numeric = ((moved[high] - moved[low]) -
                       (residual[high] - residual[low])) / step
            self.assertAlmostEqual(numeric, float(weights[index]), places=5)

    def test_leave_one_out_masks_unresolvable_rows(self):
        values = np.random.default_rng(3).normal(size=len(self.allocations))
        fitted = fit_separable(values, self.allocations, self.modes)
        design = design_matrix(self.allocations, self.modes)
        loo, keep = leave_one_out_residual(fitted["residual"], design)
        self.assertTrue(keep.any())
        self.assertTrue(np.isfinite(loo[keep]).all())
        self.assertTrue(np.isnan(loo[~keep]).all())


if __name__ == "__main__":
    unittest.main()
