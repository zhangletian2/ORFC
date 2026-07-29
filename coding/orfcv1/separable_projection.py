#!/usr/bin/env python3
"""Separable projection contract for fixed-total-rate rate allocation.

The ideal term is redefined as the orthogonal projection of the measured
distortion onto the separable family ``Phi(r)=sum_g phi[g, r_g]`` instead of a
prescribed ``c_g 2^{-2r_g/d}`` law or a JVP cost table.  Two structural facts
drive every routine here.

First, under a single fixed total rate the coefficients are identifiable only
modulo a per-group affine gauge ``phi[g,k] -> phi[g,k] + alpha_g + beta*b_k``,
so the attainable design rank is ``G*M-G`` rather than ``G*M``.  Fitted values,
residuals, ranges, argmins and the dynamic-programming bounds over the feasible
set are all invariant to that gauge; individual coefficients are not, and are
only ever compared after :func:`strip_gauge`.

Second, the projector depends on the allocation set alone and not on the
measured distortion.  The separable part therefore needs no alternating
estimation block, and the range of the non-separable residual is an exact
linear functional of the measured distortion with weights given by
:func:`range_gradient_weights`.
"""

import numpy as np


def design_matrix(allocations, modes):
    """Return the ``[A, G*M]`` one-hot separable design."""
    allocations = np.asarray(allocations, dtype=np.int64)
    if allocations.ndim != 2 or allocations.shape[0] == 0:
        raise ValueError("allocations must be a non-empty rank-two array")
    if allocations.min() < 0 or allocations.max() >= modes:
        raise ValueError("allocations contain an invalid mode index")
    rows, groups = allocations.shape
    design = np.zeros((rows, groups * modes), dtype=np.float64)
    columns = np.arange(groups, dtype=np.int64)[None, :] * modes + allocations
    np.put_along_axis(design, columns, 1.0, axis=1)
    return design


def attainable_rank(groups, modes):
    """Maximum design rank reachable at one fixed total rate."""
    return groups * modes - groups


def constant_directions(groups, modes, mode_bits, budget):
    """Coefficient directions whose separable value is constant on the budget."""
    bits = np.asarray(mode_bits, dtype=np.float64)
    if bits.shape != (modes,):
        raise ValueError("mode_bits must contain one entry per mode")
    directions = np.zeros((groups + 1, groups * modes), dtype=np.float64)
    for group in range(groups):
        directions[group, group * modes:(group + 1) * modes] = 1.0
    directions[groups] = np.tile(bits, groups)
    responses = np.concatenate([np.ones(groups), [float(budget)]])
    return directions, responses


def gauge_null_space(groups, modes, mode_bits, budget, rcond=1e-10):
    """Orthonormal rows spanning the unidentified coefficient directions."""
    directions, responses = constant_directions(
        groups, modes, mode_bits, budget)
    normal = responses / np.linalg.norm(responses)
    basis = (np.eye(len(responses)) - np.outer(normal, normal)) @ directions
    left, values, _ = np.linalg.svd(basis.T, full_matrices=False)
    keep = values > values.max() * rcond
    return left[:, keep].T


def strip_gauge(delta, null_space):
    """Project out the unidentified component of a coefficient difference."""
    basis = np.asarray(null_space, dtype=np.float64)
    flat = np.asarray(delta, dtype=np.float64).reshape(-1)
    if flat.shape[0] != basis.shape[1]:
        raise ValueError("delta and null_space disagree on the coefficient size")
    return flat - basis.T @ (basis @ flat)


def residual_operator(design, rcond=1e-10):
    """Return ``I-P`` for the separable design and its numerical rank."""
    left, values, _ = np.linalg.svd(np.asarray(design), full_matrices=False)
    keep = values > values.max() * rcond
    basis = left[:, keep]
    return np.eye(design.shape[0]) - basis @ basis.T, int(keep.sum())


def fit_separable(values, allocations, modes, rcond=1e-10):
    """Least-squares separable fit; ``values`` may be ``[A]`` or ``[A,N]``."""
    design = design_matrix(allocations, modes)
    values = np.asarray(values, dtype=np.float64)
    coefficients, _, rank, _ = np.linalg.lstsq(design, values, rcond=rcond)
    fitted = design @ coefficients
    groups = np.asarray(allocations).shape[1]
    table = coefficients.reshape(groups, modes, *coefficients.shape[1:])
    return {
        "coefficients": coefficients,
        "table": table,
        "fitted": fitted,
        "residual": values - fitted,
        "rank": int(rank),
        "attainable_rank": attainable_rank(groups, modes),
    }


def separable_values(coefficients, allocations, modes):
    """Evaluate a fitted separable model on any allocation set."""
    return design_matrix(allocations, modes) @ np.asarray(coefficients)


def leverage(design, rcond=1e-10):
    """Diagonal of the hat matrix, used to gate leave-one-out residuals."""
    operator, _ = residual_operator(design, rcond=rcond)
    return 1.0 - np.diag(operator)


def leave_one_out_residual(residual, design, threshold=1.0 - 1e-6,
                           rcond=1e-10):
    """Leave-one-out residuals with unresolvable high-leverage rows masked."""
    hat = leverage(design, rcond=rcond)
    keep = hat < threshold
    out = np.full(np.asarray(residual).shape, np.nan, dtype=np.float64)
    scale = 1.0 - hat[keep]
    out[keep] = np.asarray(residual)[keep] / (
        scale if np.ndim(residual) == 1 else scale[:, None])
    return out, keep


def range_gradient_weights(residual, operator):
    """Weights ``w`` with ``range(residual) = w @ distortion`` at the active pair.

    ``operator`` is ``I-P``.  Because ``P`` does not depend on the measured
    distortion, the range of the non-separable residual is exactly linear in
    the measured distortion once the extremal pair is fixed, so a single
    weighted backward pass gives the exact gradient of the population range.
    """
    residual = np.asarray(residual, dtype=np.float64)
    high, low = int(np.argmax(residual)), int(np.argmin(residual))
    selector = np.zeros(residual.shape[0], dtype=np.float64)
    selector[high] += 1.0
    selector[low] -= 1.0
    return np.asarray(operator) @ selector, high, low


def dp_topk(table, mode_bits, budget, count=1, maximise=False):
    """Exact top-``count`` separable allocations over the whole feasible set."""
    table = np.asarray(table, dtype=np.float64)
    if table.ndim != 2:
        raise ValueError("table must be [G, M]")
    groups, modes = table.shape
    bits = np.asarray(mode_bits, dtype=np.int64)
    if bits.shape != (modes,):
        raise ValueError("mode_bits must contain one entry per mode")
    if count < 1:
        raise ValueError("count must be positive")
    sign = -1.0 if maximise else 1.0
    low, high = int(bits.min()), int(bits.max())
    frontier = {0: [(0.0, ())]}
    for group in range(groups):
        rest = groups - group - 1
        nxt = {}
        for used, entries in frontier.items():
            for mode in range(modes):
                total = used + int(bits[mode])
                left = budget - total
                if left < low * rest or left > high * rest:
                    continue
                bucket = nxt.setdefault(total, [])
                cost = sign * table[group, mode]
                bucket.extend(
                    (value + cost, path + (mode,)) for value, path in entries)
        frontier = {
            key: sorted(value, key=lambda item: item[0])[:count]
            for key, value in nxt.items()
        }
    best = sorted(frontier.get(budget, []), key=lambda item: item[0])[:count]
    return [
        (sign * value, np.asarray(path, dtype=np.int64)) for value, path in best
    ]


def dp_summary(table, mode_bits, budget):
    """Gauge-invariant global minimum, maximum, span and top-two gap."""
    best = dp_topk(table, mode_bits, budget, count=2)
    worst = dp_topk(table, mode_bits, budget, count=1, maximise=True)
    return {
        "minimum": float(best[0][0]),
        "argmin": best[0][1],
        "second": float(best[1][0]) if len(best) > 1 else None,
        "argsecond": best[1][1] if len(best) > 1 else None,
        "top2_gap": (
            float(best[1][0] - best[0][0]) if len(best) > 1 else None),
        "maximum": float(worst[0][0]),
        "argmax": worst[0][1],
        "span": float(worst[0][0] - best[0][0]),
    }


def cell_coverage(allocations, modes):
    """Count how often each ``(group, mode)`` cell appears in the design."""
    allocations = np.asarray(allocations, dtype=np.int64)
    groups = allocations.shape[1]
    counts = np.zeros((groups, modes), dtype=np.int64)
    for group in range(groups):
        counts[group] = np.bincount(allocations[:, group], minlength=modes)
    return counts
