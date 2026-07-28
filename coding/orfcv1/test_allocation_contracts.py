#!/usr/bin/env python3
"""Small CPU contracts for the current fixed-rate allocation pipeline."""

from argparse import Namespace
from itertools import product

import numpy as np
import torch

from cayley import CayleySGD, DirectOrthogonalTransform
from allocation_train import (
    _allocation_source, _backward, _curve_report, _dynamic_source,
    _objective_terms, _protect_primary, _select_state,
    _tangent_gradient, _validate_training_slices, _with_ideal_set,
)
from fixed_rate_remainder import (
    decompose_output_vectors, validate_fixed_total_rate,
)
from ideal_set_statistics import _solve_batch, _suffix_counts
from p1_fixed_rate import (
    make_allocations, top2_allocate, top2_cost_allocate, topk_cost_allocate,
)


def test_fixed_ideal_candidate_binding():
    rates = np.asarray([[1, 3], [2, 2], [3, 1]], dtype=np.float64)
    expected = np.asarray([1.0, 16.0])
    distortion = np.asarray([10.0, 1.0, 0.0])
    source = {
        "rates": rates,
        "allocations": np.asarray([[0, 2], [1, 1], [2, 0]]),
    }
    calibration = {
        "rate_dimension": np.asarray(1),
        "c_g": expected,
        "mode_bits": np.asarray([1, 2, 3]),
        "ideal_bits": np.asarray([1, 3]),
        "ideal_gap": np.asarray(0.5625),
    }
    args = Namespace(
        allocations=3, seed=42, reference_bit=2,
        tie_atol=1e-8, tie_rtol=1e-8)
    state = _select_state(source, distortion, calibration, args)
    assert state["target"] == 0
    assert state["full_phi"][0] < state["full_phi"][2]
    assert state["allocations"][state["minimizer_local"][0]].tolist() == (
        source["allocations"][state["minimizers"][0]].tolist())
    assert state["reference"] == 1

    tied = _select_state(
        source, np.ones(3), {
            "rate_dimension": np.asarray(1),
            "c_g": np.zeros(2),
            "mode_bits": np.asarray([1, 2, 3]),
            "ideal_bits": np.asarray([1, 3]),
            "ideal_gap": np.asarray(0.0),
        }, args)
    assert len(tied["minimizers"]) == 3
    assert tied["gap"] == 0


def test_top2_matches_brute_force():
    c, bits, budget = np.asarray([1.0, 2.0, 3.0]), (1, 2, 3), 6
    exact = top2_allocate(c, bits, budget, 2)
    rows = sorted(
        (sum(c[g] * 2 ** (-choice[g]) for g in range(3)), choice)
        for choice in product(bits, repeat=3) if sum(choice) == budget)
    assert tuple(exact["ideal_bits"]) == rows[0][1]
    assert tuple(exact["second_bits"]) == rows[1][1]
    assert np.isclose(exact["ideal_gap"], rows[1][0] - rows[0][0])
    table = np.asarray([[3, 2, 1], [1, 2, 4]], dtype=float)
    discrete = top2_cost_allocate(table, bits, 4)
    brute = sorted(
        (sum(table[g, bits.index(choice[g])] for g in range(2)), choice)
        for choice in product(bits, repeat=2) if sum(choice) == 4)
    assert tuple(discrete["ideal_bits"]) == brute[0][1]
    assert tuple(discrete["second_bits"]) == brute[1][1]
    top = topk_cost_allocate(table, bits, 4, 3)
    assert [tuple(row) for row in top["bits"]] == [
        row[1] for row in brute[:3]]
    allocations, _, _ = make_allocations(
        c, bits, budget, 0, 0, 42, 2)
    realised = {tuple(np.asarray(bits)[row]) for row in allocations}
    assert rows[0][1] in realised and rows[1][1] in realised


def test_fixed_rate_and_curve_contracts():
    allocations = np.asarray([[0, 2], [1, 1], [2, 0]])
    costs = np.broadcast_to(np.asarray([1.0, 2.0, 3.0]), (2, 3))
    _, totals, target = validate_fixed_total_rate(allocations, costs)
    assert target == 4.0 and np.all(totals == target)
    passed = _curve_report([3, 4, 5], [9.0, 6.0, 4.0], 0.0)
    failed = _curve_report([3, 4, 5], [9.0, 10.0, 4.0], 0.0)
    assert passed["monotonic"] and not failed["monotonic"]


def test_statistical_allocation_helpers():
    bits = np.asarray([1, 2, 3])
    table = np.asarray([[[3., 2., 1.], [1., 2., 4.]]])
    modes = _solve_batch(table, bits, 4)
    assert modes.tolist() == [[2, 0]]
    counts = _suffix_counts(2, tuple(bits), 4)
    assert counts[0][4] == 3
    slices = Namespace(
        outer_calibration_offset=0, outer_calibration_images=10,
        outer_mining_offset=10, outer_mining_images=5,
        train_image_offset=15, train_images=20)
    _validate_training_slices(slices, 35)
    slices.train_image_offset = 14
    try:
        _validate_training_slices(slices, 35)
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping training slices were accepted")


def test_dynamic_pool_and_selection_objective():
    costs = np.broadcast_to(np.arange(1, 4, dtype=float), (2, 3))
    calibration = {
        "cost_table": costs, "mode_bits": np.arange(1, 4),
        "rate_dimension": np.asarray(1), "c_g": np.ones(2),
        "ideal_bits": np.asarray([2, 2]),
        "ideal_gap": np.asarray(0.140625)}
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
        allocations=3, seed=42, reference_bit=2,
        tie_atol=1e-8, tie_rtol=1e-8, lse_temperature=0.1,
        recovery_margin=0.0, candidate_mean_weight=0.0)
    distortion = np.asarray([5.0, 3.0, 4.0])
    state = _select_state(audit, distortion, calibration, select_args)
    terms = _objective_terms(
        distortion[state["selected"]], state, select_args, 0.5)
    assert terms["score"] >= terms["base"] and terms["omega"] >= 0
    assert terms["empirical_margin"] == 1.0

    discrete = dict(calibration)
    discrete["ideal_cost_table"] = np.asarray([[3, 2, 1], [1, 2, 4]])
    discrete["ideal_bits"] = np.asarray([3, 1])
    state = _select_state(audit, distortion, discrete, select_args)
    assert state["target"] == 2 and state["full_phi"].tolist() == [7, 4, 2]
    set_args = Namespace(ideal_set_size=2)
    discrete = _with_ideal_set(discrete, set_args)
    select_args.ideal_batch_size = 1
    state = _select_state(audit, distortion, discrete, select_args)
    assert state["target_set"].tolist() == [2, 1]
    assert state["active_target_set"].tolist() == [1]
    assert state["selected"][state["competitor_local"]].tolist() == [0]
    assert state["gap"] == 5 and state["empirical_margin"] == 2


def test_primary_gradient_protection():
    primary = torch.tensor([1.0, 0.0])
    auxiliary = torch.tensor([-2.0, 1.0])
    protected, cosine, projected = _protect_primary(auxiliary, primary)
    assert projected and cosine < 0
    assert torch.dot(primary, protected).abs() < 1e-7


def test_joint_recovery_gradients():
    rotation = torch.nn.Parameter(torch.eye(2))
    codebook = torch.nn.Parameter(torch.ones(2))
    base = ((rotation - torch.tensor([[1., 1.], [0., 1.]])) ** 2).sum()
    base = base + codebook.square().sum()
    recovery = -rotation[0, 1] - codebook.sum()
    _backward(
        {"base": base, "recovery": recovery},
        [rotation, codebook], rotation, 0.5)
    assert rotation.grad is not None and codebook.grad is not None
    assert torch.allclose(
        rotation.t() @ rotation.grad + rotation.grad.t() @ rotation,
        torch.zeros(2), atol=1e-6)


def test_direct_cayley_descent_and_orthogonality():
    torch.manual_seed(42)
    transform = DirectOrthogonalTransform(8).double()
    target, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
    optimizer = CayleySGD(
        [transform.rotation], lr=0.05, reorthogonalize_every=10)
    initial = (transform.rotation - target).square().sum().item()
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = (transform.rotation - target).square().sum()
        loss.backward()
        optimizer.step()
    assert loss.item() < initial
    assert transform.orth_error() < 1e-10


def test_rotation_gradient_is_tangent():
    rotation, _ = torch.linalg.qr(torch.randn(8, 8))
    tangent = _tangent_gradient(rotation, torch.randn(8, 8))
    assert torch.allclose(
        rotation.t() @ tangent + tangent.t() @ rotation,
        torch.zeros(8, 8), atol=1e-5)


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
    test_fixed_ideal_candidate_binding()
    test_top2_matches_brute_force()
    test_fixed_rate_and_curve_contracts()
    test_statistical_allocation_helpers()
    test_dynamic_pool_and_selection_objective()
    test_primary_gradient_protection()
    test_joint_recovery_gradients()
    test_direct_cayley_descent_and_orthogonality()
    test_rotation_gradient_is_tangent()
    test_remainder_decomposition()
    print("PASS: allocation contracts")
