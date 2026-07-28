#!/usr/bin/env python3
"""Small CPU contracts for the current fixed-rate allocation pipeline."""

from argparse import Namespace

import numpy as np
import torch

from allocation_train import (
    _allocation_source, _curve_report, _design, _dynamic_source,
    _objective_terms, _positive_fit, _protect_primary, _select_state,
)
from fixed_rate_remainder import (
    decompose_output_vectors, validate_fixed_total_rate,
)


def test_positive_projection_and_candidate_binding():
    rates = np.asarray([[1, 3], [2, 2], [3, 1]], dtype=np.float64)
    A = _design(rates, 1)
    intercept, expected = 7.0, np.asarray([2.0, 8.0])
    distortion = intercept + A @ expected
    fitted_intercept, fitted = _positive_fit(
        A, distortion, expected, ridge=1e-8)
    assert np.allclose(fitted_intercept, intercept, atol=1e-5)
    assert np.allclose(fitted, expected, atol=1e-4)

    source = {
        "rates": rates,
        "allocations": np.asarray([[0, 2], [1, 1], [2, 0]]),
    }
    calibration = {
        "rate_dimension": np.asarray(1),
        "c_g": expected,
        "mode_bits": np.asarray([1, 2, 3]),
    }
    args = Namespace(
        ridge=1e-8, allocations=3, seed=42, reference_bit=2,
        tie_atol=1e-8, tie_rtol=1e-8)
    state = _select_state(source, distortion, calibration, args)
    assert state["target"] == int(distortion.argmin())
    assert state["allocations"][state["minimizer_local"][0]].tolist() == (
        source["allocations"][state["minimizers"][0]].tolist())
    assert state["reference"] == 1

    tied = _select_state(
        source, np.ones(3), {
            "rate_dimension": np.asarray(1),
            "c_g": np.zeros(2),
            "mode_bits": np.asarray([1, 2, 3]),
        }, args)
    assert len(tied["minimizers"]) == 3
    assert np.isinf(tied["gap"])


def test_fixed_rate_and_curve_contracts():
    allocations = np.asarray([[0, 2], [1, 1], [2, 0]])
    costs = np.broadcast_to(np.asarray([1.0, 2.0, 3.0]), (2, 3))
    _, totals, target = validate_fixed_total_rate(allocations, costs)
    assert target == 4.0 and np.all(totals == target)
    passed = _curve_report([3, 4, 5], [9.0, 6.0, 4.0], 0.0)
    failed = _curve_report([3, 4, 5], [9.0, 10.0, 4.0], 0.0)
    assert passed["monotonic"] and not failed["monotonic"]


def test_dynamic_pool_and_selection_objective():
    costs = np.broadcast_to(np.arange(1, 4, dtype=float), (2, 3))
    calibration = {
        "cost_table": costs, "mode_bits": np.arange(1, 4),
        "rate_dimension": np.asarray(1), "c_g": np.ones(2)}
    audit = _allocation_source(np.asarray([[0, 2], [1, 1], [2, 0]]),
                               calibration)
    args = Namespace(
        dynamic_allocations=True, dynamic_single=2, dynamic_random=2,
        seed=42)
    pool = _dynamic_source(audit, calibration, np.asarray([2.0, 1.0]),
                           args, 0)
    assert len(pool["allocations"]) >= len(audit["allocations"])
    validate_fixed_total_rate(pool["allocations"], costs)

    select_args = Namespace(
        ridge=1e-8, allocations=3, seed=42, reference_bit=2,
        tie_atol=1e-8, tie_rtol=1e-8, lse_temperature=0.1,
        recovery_fraction=1.0)
    distortion = np.asarray([5.0, 3.0, 4.0])
    state = _select_state(audit, distortion, calibration, select_args)
    terms = _objective_terms(
        distortion[state["selected"]], state, select_args, 0.5)
    assert terms["score"] >= terms["base"] and terms["omega"] >= 0


def test_primary_gradient_protection():
    primary = torch.tensor([1.0, 0.0])
    auxiliary = torch.tensor([-2.0, 1.0])
    protected, cosine, projected = _protect_primary(auxiliary, primary)
    assert projected and cosine < 0
    assert torch.dot(primary, protected).abs() < 1e-7


def test_remainder_decomposition():
    response = torch.tensor([[[[1.0, 0.0]], [[0.0, 2.0]]]])
    delta = response.sum(1)
    parts = decompose_output_vectors(3.0, response, delta)
    total = (
        parts["phi"] + parts["menu"] + parts["cross"]
        + parts["nonlinear"])
    assert torch.allclose(parts["distortion"], total)
    assert float(parts["cross"]) == 0.0
    assert float(parts["nonlinear"]) == 0.0


if __name__ == "__main__":
    test_positive_projection_and_candidate_binding()
    test_fixed_rate_and_curve_contracts()
    test_dynamic_pool_and_selection_objective()
    test_primary_gradient_protection()
    test_remainder_decomposition()
    print("PASS: allocation contracts")
