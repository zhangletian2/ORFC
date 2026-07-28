#!/usr/bin/env python3
"""Small CPU contracts for the current fixed-rate allocation pipeline."""

from argparse import Namespace

import numpy as np
import torch

from allocation_train import (
    _curve_report, _design, _positive_fit, _seed_from_anchor, _select_state,
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


def test_nested_anchor_initialisation():
    class Quantizer:
        def __init__(self, size):
            self.codebooks = torch.zeros(1, size, 1)

    class PQ:
        def __init__(self):
            self.quantizers = [Quantizer(size) for size in (2, 4, 8)]

    class Codec:
        def __init__(self):
            self.pq = PQ()

    codec = Codec()
    anchor = torch.arange(4.0).reshape(1, 4, 1)
    codec.pq.quantizers[1].codebooks.copy_(anchor)
    _seed_from_anchor(codec, 1)
    low = set(codec.pq.quantizers[0].codebooks.flatten().tolist())
    middle = set(anchor.flatten().tolist())
    assert low < middle
    assert torch.equal(codec.pq.quantizers[1].codebooks, anchor)
    assert torch.equal(codec.pq.quantizers[2].codebooks[:, :4], anchor)


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
    test_nested_anchor_initialisation()
    test_remainder_decomposition()
    print("PASS: allocation contracts")
