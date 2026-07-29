"""Fixed-total-rate allocation and remainder measurements.

This module reuses FeatureCodecV1, the frozen-tail teacher cache and the
existing normalisation path.  It contains no training or experiment launcher.
"""

import itertools

import numpy as np
import torch

from opq import batch_inv_normalize_gpu, batch_normalize_gpu


def nominal_cost_table(pq):
    """Return ``[G,M]`` fixed-length mode costs in bits/token/group."""
    rates = np.log2(np.asarray(pq.mode_sizes, dtype=np.float64))
    return np.broadcast_to(rates[None, :], (pq.G, len(rates))).copy()


def allocation_rates(allocations, cost_table):
    """Map mode-index allocations ``[A,G]`` to per-group rates ``[A,G]``."""
    allocations = np.asarray(allocations, dtype=np.int64)
    costs = np.asarray(cost_table, dtype=np.float64)
    if allocations.ndim != 2 or costs.ndim != 2:
        raise ValueError("allocations and cost_table must both be rank two")
    if allocations.shape[0] == 0:
        raise ValueError("allocations must be non-empty")
    if allocations.shape[1] != costs.shape[0]:
        raise ValueError("allocation group count does not match cost_table")
    if (allocations < 0).any() or (allocations >= costs.shape[1]).any():
        raise ValueError("allocation contains an invalid mode index")
    groups = np.arange(costs.shape[0])[None, :]
    return costs[groups, allocations]


def validate_fixed_total_rate(allocations, cost_table, tolerance=1e-8):
    """Validate one common total-rate budget and return rate metadata."""
    rates = allocation_rates(allocations, cost_table)
    totals = rates.sum(axis=1)
    target = float(totals[0])
    error = np.abs(totals - target)
    if float(error.max(initial=0.0)) > tolerance:
        raise ValueError(
            f"allocations violate fixed total rate: "
            f"max error={error.max():.6g}, tolerance={tolerance:.6g}")
    return rates, totals, target


def adjacent_exchange_allocations(base, cost_table, tolerance=1e-8):
    """Generate exact-budget one-step donor/receiver exchanges."""
    base = np.asarray(base, dtype=np.int64)
    costs = np.asarray(cost_table, dtype=np.float64)
    if base.shape != (costs.shape[0],):
        raise ValueError("base must contain one mode index per group")
    target = allocation_rates(base[None, :], costs).sum()
    result = {tuple(base.tolist())}
    for donor in range(len(base)):
        if base[donor] == 0:
            continue
        for receiver in range(len(base)):
            if donor == receiver or base[receiver] + 1 >= costs.shape[1]:
                continue
            candidate = base.copy()
            candidate[donor] -= 1
            candidate[receiver] += 1
            total = allocation_rates(candidate[None, :], costs).sum()
            if abs(float(total - target)) <= tolerance:
                result.add(tuple(candidate.tolist()))
    return np.asarray(sorted(result), dtype=np.int64)


def random_exchange_walk(
    base, cost_table, n_samples, max_steps=8, tolerance=1e-8, seed=42,
):
    """Sample multi-group allocations by composing valid adjacent exchanges."""
    base = np.asarray(base, dtype=np.int64)
    rng = np.random.default_rng(seed)
    seen = {tuple(base.tolist())}
    current = base.copy()
    attempts = 0
    max_attempts = max(100, n_samples * max_steps * 20)
    while len(seen) < n_samples and attempts < max_attempts:
        neighbours = adjacent_exchange_allocations(
            current, cost_table, tolerance=tolerance)
        neighbours = [
            row for row in neighbours
            if tuple(row.tolist()) != tuple(current.tolist())
        ]
        if not neighbours:
            current = base.copy()
            attempts += 1
            continue
        for _ in range(int(rng.integers(1, max_steps + 1))):
            current = neighbours[int(rng.integers(len(neighbours)))].copy()
            seen.add(tuple(current.tolist()))
            neighbours = adjacent_exchange_allocations(
                current, cost_table, tolerance=tolerance)
            neighbours = [
                row for row in neighbours
                if tuple(row.tolist()) != tuple(current.tolist())
            ]
            if not neighbours or len(seen) >= n_samples:
                break
        attempts += 1
    return np.asarray(sorted(seen), dtype=np.int64)


def enumerate_fixed_rate(
    groups, num_modes, cost_table, target_rate, tolerance=1e-8,
    max_states=1_000_000,
):
    """Exhaust a small allocation menu; refuse unexpectedly large products."""
    n_states = int(num_modes) ** int(groups)
    if n_states > max_states:
        raise ValueError(
            f"requested {n_states:,} states exceeds max_states={max_states:,}")
    selected = []
    for state in itertools.product(range(num_modes), repeat=groups):
        total = allocation_rates(
            np.asarray(state)[None, :], cost_table).sum()
        if abs(float(total - target_rate)) <= tolerance:
            selected.append(state)
    if not selected:
        raise ValueError("no allocation satisfies the requested rate")
    return np.asarray(selected, dtype=np.int64)


def ideal_phi(c_g, rates, rate_dimension=1):
    """Evaluate ``Phi(r)=sum_g c_g 2^(-2 r_g / d)``."""
    c_g = np.asarray(c_g, dtype=np.float64)
    rates = np.asarray(rates, dtype=np.float64)
    if rates.ndim != 2 or rates.shape[1] != c_g.size:
        raise ValueError("rates must have shape [A,G] matching c_g")
    return (
        c_g[None, :] * np.exp2(-2.0 * rates / float(rate_dimension))
    ).sum(axis=1)


def allocation_phi(
    allocations, cost_table, c_g, rate_dimension, ideal_cost_table=None,
):
    """Evaluate the one canonical separable ideal model for allocations."""
    rates = allocation_rates(allocations, cost_table)
    if ideal_cost_table is None:
        return ideal_phi(c_g, rates, rate_dimension=rate_dimension)
    table = np.asarray(ideal_cost_table, dtype=np.float64)
    if table.shape != np.asarray(cost_table).shape:
        raise ValueError("ideal_cost_table must match cost_table")
    groups = np.arange(table.shape[0])[None, :]
    return table[groups, np.asarray(allocations, dtype=np.int64)].sum(axis=1)


def bootstrap_remainder_range(
    distortion_per_image, phi, comparison=None, comparison_phi=None,
    bootstraps=1000, batch_size=32, seed=42,
):
    """Bootstrap allocation extrema, reselecting max/min on every resample."""
    distortion = np.asarray(distortion_per_image, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    if distortion.ndim != 2 or phi.shape != (distortion.shape[0],):
        raise ValueError("distortion must be [A,N] and phi must be [A]")
    other = None if comparison is None else np.asarray(
        comparison, dtype=np.float64)
    if other is not None and other.shape != distortion.shape:
        raise ValueError("comparison must match distortion")
    other_phi = (
        phi if comparison_phi is None else
        np.asarray(comparison_phi, dtype=np.float64))
    if other_phi.shape != phi.shape:
        raise ValueError("comparison_phi must match phi")

    def omega(values, reference):
        return float(np.ptp(values.mean(1) - reference))

    point = omega(distortion, phi)
    if bootstraps < 1 or distortion.shape[1] < 2:
        samples = np.asarray([point])
        differences = (
            np.asarray([point - omega(other, other_phi)])
            if other is not None else None)
    else:
        rng = np.random.default_rng(seed)
        rows, deltas = [], []
        for start in range(0, bootstraps, batch_size):
            count = min(batch_size, bootstraps - start)
            ids = rng.integers(
                0, distortion.shape[1], size=(count, distortion.shape[1]))
            means = distortion[:, ids].mean(2)
            current = np.ptp(means - phi[:, None], axis=0)
            rows.append(current)
            if other is not None:
                baseline = np.ptp(
                    other[:, ids].mean(2) - other_phi[:, None], axis=0)
                deltas.append(current - baseline)
        samples = np.concatenate(rows)
        differences = np.concatenate(deltas) if deltas else None
    result = {
        "omega_point": point,
        "omega_bootstrap_ci95": np.quantile(
            samples, [0.025, 0.975]).tolist(),
        "omega_bootstrap_count": int(bootstraps),
        "omega_extrema_reselected": True,
    }
    if differences is not None:
        result.update({
            "paired_omega_change": float(
                point - omega(other, other_phi)),
            "paired_omega_change_ci95": np.quantile(
                differences, [0.025, 0.975]).tolist(),
        })
    return result


def paired_contract_statistics(
    distortion, quad_phi, analytic_phi, bootstraps=1000,
    batch_size=32, seed=42,
):
    """Audit the unified per-image contract with paired resampling.

    Arrays use shape ``[allocations, images]``.  Every bootstrap resamples the
    same images for ``D``, ``Phi_quad`` and ``Phi_ana`` and reselects both the
    remainder extrema and the sampled analytic optimum.
    """
    D, Q, A = (
        np.asarray(value, dtype=np.float64)
        for value in (distortion, quad_phi, analytic_phi)
    )
    if D.ndim != 2 or Q.shape != D.shape or A.shape != D.shape:
        raise ValueError("D, Phi_quad and Phi_ana must share shape [A,N]")
    structural, rate = D - Q, Q - A
    total = D - A

    def metrics(indices=None):
        arrays = (structural, rate, total, A)
        means = [
            value.mean(1) if indices is None else value[:, indices].mean(2)
            for value in arrays
        ]
        if indices is None:
            s, m, e, phi = means
            order = np.argsort(phi)
            gap = float(phi[order[1]] - phi[order[0]]) if len(phi) > 1 else np.inf
            omega = float(np.ptp(e))
            return np.asarray([np.ptp(s), np.ptp(m), omega, gap, gap - omega])
        s, m, e, phi = means
        order = np.argsort(phi, axis=0)
        columns = np.arange(phi.shape[1])
        gap = (
            phi[order[1], columns] - phi[order[0], columns]
            if len(phi) > 1 else np.full(phi.shape[1], np.inf)
        )
        omega = np.ptp(e, axis=0)
        return np.stack([
            np.ptp(s, axis=0), np.ptp(m, axis=0), omega, gap, gap - omega])

    names = (
        "structural_range", "rate_model_mismatch_range",
        "analytic_remainder_range", "sampled_analytic_gap",
        "sampled_recovery_margin",
    )
    point = metrics()
    samples = []
    if bootstraps > 0 and D.shape[1] > 1:
        rng = np.random.default_rng(seed)
        for start in range(0, bootstraps, batch_size):
            count = min(batch_size, bootstraps - start)
            indices = rng.integers(
                0, D.shape[1], size=(count, D.shape[1]))
            samples.append(metrics(indices))
    draws = np.concatenate(samples, axis=1) if samples else point[:, None]
    result = {
        f"{name}_point": float(point[index])
        for index, name in enumerate(names)
    }
    result.update({
        f"{name}_ci95": np.quantile(
            draws[index], [0.025, 0.975]).tolist()
        for index, name in enumerate(names)
    })
    result.update({
        "contract_bootstrap_count": int(bootstraps),
        "contract_extrema_reselected": True,
        "contract_analytic_optimum_reselected": True,
        "sampled_recovery_condition_point": bool(point[-1] > 0),
        "sampled_recovery_condition_confident": bool(
            result["analytic_remainder_range_ci95"][1]
            < result["sampled_analytic_gap_ci95"][0]),
    })
    return result


def allocation_correlation(first, second):
    """Return Pearson and rank correlation across allocation-level values."""
    first, second = (
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in (first, second)
    )
    if first.shape != second.shape:
        raise ValueError("allocation correlation inputs must match")

    def correlation(x, y):
        x, y = x - x.mean(), y - y.mean()
        scale = np.linalg.norm(x) * np.linalg.norm(y)
        return float(x @ y / scale) if scale > 0 else None

    def rank(value):
        order = np.argsort(value, kind="mergesort")
        result = np.empty(len(value), dtype=np.float64)
        result[order] = np.arange(len(value), dtype=np.float64)
        return result

    return {
        "pearson": correlation(first, second),
        "spearman": correlation(rank(first), rank(second)),
    }


def split_half_stability(per_image, repeats=200, seed=42):
    """Measure whether allocation ordering replicates across disjoint halves."""
    values = np.asarray(per_image, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("split-half stability requires shape [A,N]")
    if values.shape[1] < 4 or repeats < 1:
        return {"available": False, "repeats": 0}
    rng, correlations, range_ratios = np.random.default_rng(seed), [], []
    half = values.shape[1] // 2
    for _ in range(repeats):
        order = rng.permutation(values.shape[1])
        first = values[:, order[:half]].mean(1)
        second = values[:, order[half:2 * half]].mean(1)
        correlation = allocation_correlation(first, second)["spearman"]
        if correlation is not None:
            correlations.append(correlation)
        ranges = float(np.ptp(first)), float(np.ptp(second))
        if max(ranges) > 0:
            range_ratios.append(min(ranges) / max(ranges))

    def report(rows):
        rows = np.asarray(rows, dtype=np.float64)
        return {
            "median": float(np.median(rows)) if rows.size else None,
            "ci95": (
                np.quantile(rows, [0.025, 0.975]).tolist()
                if rows.size else None),
        }

    return {
        "available": True,
        "repeats": int(repeats),
        "split_half_spearman": report(correlations),
        "split_half_range_ratio": report(range_ratios),
    }


def decompose_output_vectors(phi, group_response, output_delta):
    """Split one output distortion into analytic, menu, cross and nonlinear terms.

    ``group_response`` has shape ``[B,G,...]`` and contains the local linear
    response to each realised group error.  ``output_delta`` is the exact
    frozen-tail output change with shape ``[B,...]``.
    """
    response = group_response.flatten(2)
    delta = output_delta.flatten(1)
    response_sum = response.sum(1)
    self_quad = response.square().sum((1, 2)).double()
    paired_quad = response_sum.square().sum(1).double()
    distortion = delta.square().sum(1).double()
    phi = torch.as_tensor(
        phi, dtype=distortion.dtype, device=distortion.device
    ).expand_as(distortion)
    rho = delta - response_sum
    menu = self_quad - phi
    cross = paired_quad - self_quad
    nonlinear = distortion - paired_quad
    cross_bound = (
        response.norm(dim=2).sum(1).square() - self_quad
    ).clamp_min(0)
    rho_norm = rho.norm(dim=1)
    nonlinear_bound = 2 * response_sum.norm(dim=1) * rho_norm + rho_norm.square()
    return {
        "distortion": distortion, "phi": phi, "self_quad": self_quad,
        "paired_quad": paired_quad, "menu": menu, "cross": cross,
        "nonlinear": nonlinear, "cross_bound": cross_bound,
        "nonlinear_bound": nonlinear_bound,
    }


def _swap_edge_range(allocations, remainder):
    index = {
        tuple(row.tolist()): i for i, row in enumerate(allocations)
    }
    maximum = 0.0
    count = 0
    for i, row in enumerate(allocations):
        for donor in range(row.size):
            if row[donor] == 0:
                continue
            for receiver in range(row.size):
                if donor == receiver:
                    continue
                neighbour = row.copy()
                neighbour[donor] -= 1
                neighbour[receiver] += 1
                j = index.get(tuple(neighbour.tolist()))
                if j is not None and i < j:
                    maximum = max(
                        maximum, abs(float(remainder[i] - remainder[j])))
                    count += 1
    return maximum, count


@torch.no_grad()
def evaluate_fixed_rate_remainder(
    features_array,
    teacher_cache,
    codec,
    tail,
    allocations,
    cost_table,
    c_g,
    norm_mode,
    device,
    batch_size=8,
    allocation_chunk=4,
    rate_tolerance=1e-8,
    ideal_cost_table=None,
    bootstrap_count=1000,
    bootstrap_batch=32,
    bootstrap_seed=42,
):
    """Measure ``D``, ``Phi`` and ``E=D-Phi`` on fixed-rate allocations.

    The expensive frozen-tail call is batched over allocation chunks.  Returned
    per-image arrays preserve paired comparisons and can be saved separately
    from the compact summary.
    """
    allocations = np.asarray(allocations, dtype=np.int64)
    rates, totals, target = validate_fixed_total_rate(
        allocations, cost_table, tolerance=rate_tolerance)
    phi = allocation_phi(
        allocations, cost_table, c_g, codec.pq.d, ideal_cost_table)
    n_alloc, n_images = allocations.shape[0], len(features_array)
    distortion = np.empty((n_alloc, n_images), dtype=np.float64)

    codec.eval()
    for start in range(0, n_images, batch_size):
        end = min(start + batch_size, n_images)
        X = torch.from_numpy(
            np.array(features_array[start:end], copy=True)).float().to(device)
        teacher = torch.from_numpy(
            np.array(teacher_cache[start:end], copy=True)).float().to(device)
        Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
        B = X.shape[0]

        if not hasattr(codec.pq, "quantizers"):
            raise TypeError("fixed-rate evaluation requires MultiModeSoftPQ")
        R = codec.transform.get_rotation()
        Z = Y.reshape(B * Y.shape[1], -1) @ R
        bank = torch.stack([
            quantizer._quantise(Z)[0].reshape(
                B, Y.shape[1], codec.pq.G, codec.pq.d)
            for quantizer in codec.pq.quantizers
        ])
        bank_g = bank.permute(3, 0, 1, 2, 4)
        groups = torch.arange(codec.pq.G, device=device)[None, :]

        for a_start in range(0, n_alloc, allocation_chunk):
            a_end = min(a_start + allocation_chunk, n_alloc)
            count = a_end - a_start
            modes = torch.as_tensor(
                allocations[a_start:a_end],
                device=device, dtype=torch.long)
            selected = bank_g[groups, modes].permute(
                0, 2, 3, 1, 4).reshape(
                    count * B, Y.shape[1], -1)
            Y_hat = selected @ R.t()
            mu = Mu.unsqueeze(0).expand(
                count, *Mu.shape).reshape(count * B, *Mu.shape[1:])
            std = Std.unsqueeze(0).expand(
                count, *Std.shape).reshape(count * B, *Std.shape[1:])
            X_hat = batch_inv_normalize_gpu(Y_hat, mu, std)
            output = tail.forward_nograd(X_hat)
            teacher_rep = teacher.unsqueeze(0).expand(
                a_end - a_start, -1, *teacher.shape[1:]).reshape(
                    (a_end - a_start) * B, *teacher.shape[1:])
            values = ((output - teacher_rep) ** 2).reshape(
                a_end - a_start, B, -1).sum(dim=-1)
            distortion[a_start:a_end, start:end] = (
                values.cpu().numpy())

    D = distortion.mean(axis=1)
    E = D - phi
    e_min, e_max = int(E.argmin()), int(E.argmax())
    omega = float(E[e_max] - E[e_min])
    phi_best = int(phi.argmin())
    gaps = phi - phi[phi_best]
    tied_best = int(np.count_nonzero(np.abs(gaps) <= 1e-12))
    positive_gaps = gaps[gaps > 1e-12]
    if tied_best != 1:
        phi_gap = 0.0
    else:
        phi_gap = (
            float(positive_gaps.min()) if positive_gaps.size
            else float("inf"))
    candidates = np.flatnonzero(gaps <= omega + 1e-12)
    edge_max, edge_count = _swap_edge_range(allocations, E)

    summary = {
        "n_allocations": int(n_alloc),
        "n_images": int(n_images),
        "target_rate": target,
        "max_rate_error": float(np.abs(totals - target).max()),
        "omega_sampled": omega,
        "phi_best_index": phi_best,
        "phi_gap": phi_gap,
        "chi_sampled": (
            float(omega / phi_gap)
            if np.isfinite(phi_gap) and phi_gap > 0 else None),
        "candidate_indices": candidates.tolist(),
        "candidate_count": int(candidates.size),
        "remainder_min_index": e_min,
        "remainder_max_index": e_max,
        "swap_edge_max": float(edge_max),
        "swap_edge_count": int(edge_count),
    }
    summary.update(bootstrap_remainder_range(
        distortion, phi, bootstraps=bootstrap_count,
        batch_size=bootstrap_batch, seed=bootstrap_seed))
    arrays = {
        "allocations": allocations,
        "rates": rates,
        "total_rates": totals,
        "distortion_per_image": distortion,
        "distortion_mean": D,
        "phi": phi,
        "remainder": E,
    }
    return summary, arrays


@torch.no_grad()
def evaluate_fixed_rate_decomposition(
    features_array,
    teacher_cache,
    codec,
    tail,
    allocations,
    cost_table,
    c_g,
    norm_mode,
    device,
    allocation_chunk=4,
    jvp_eps=0.01,
    jvp_chunk=8,
    rate_tolerance=1e-8,
    ideal_cost_table=None,
    mode_bits=None,
    reference_bit=None,
    bootstrap_count=1000,
    bootstrap_batch=32,
    bootstrap_seed=42,
    stability_repeats=200,
):
    """Measure the unified ``D``, quadratic and analytic ideal contracts.

    The primary exact split is ``D=Phi_quad+E_struct``.  With a reference mode,
    the common exponential model adds
    ``D=Phi_ana+E_rate+E_struct``.  Central finite differences estimate every
    per-image, per-group, per-mode local response under the current codec.
    """
    if jvp_eps <= 0 or jvp_chunk < 1:
        raise ValueError("jvp_eps and jvp_chunk must be positive")
    if not hasattr(codec.pq, "quantizers"):
        raise TypeError("fixed-rate decomposition requires MultiModeSoftPQ")
    allocations = np.asarray(allocations, dtype=np.int64)
    rates, totals, target = validate_fixed_total_rate(
        allocations, cost_table, tolerance=rate_tolerance)
    calibrated_phi = allocation_phi(
        allocations, cost_table, c_g, codec.pq.d, ideal_cost_table)
    keys = (
        "distortion", "self_quad", "paired_quad", "menu", "cross",
        "nonlinear", "cross_bound", "nonlinear_bound",
    )
    values = {
        key: np.empty((len(allocations), len(features_array)), np.float64)
        for key in keys
    }
    codec.eval()
    groups = codec.pq.G
    modes_count = len(codec.pq.quantizers)
    mode_bits = (
        np.log2(codec.pq.mode_sizes)
        if mode_bits is None else np.asarray(mode_bits, dtype=np.float64))
    if mode_bits.shape != (modes_count,):
        raise ValueError("mode_bits must match the multi-mode PQ menu")
    if reference_bit is None:
        reference_bit = float(mode_bits[len(mode_bits) // 2])
    matches = np.flatnonzero(np.isclose(mode_bits, reference_bit))
    if len(matches) != 1:
        raise ValueError("reference_bit must identify exactly one mode")
    reference_mode = int(matches[0])
    quad_cost_per_image = np.empty(
        (len(features_array), groups, modes_count), np.float64)

    for image_index in range(len(features_array)):
        x = torch.from_numpy(np.array(
            features_array[image_index:image_index + 1], copy=True
        )).float().to(device)
        teacher = torch.from_numpy(np.array(
            teacher_cache[image_index:image_index + 1], copy=True
        )).float().to(device)
        y, mu, std = batch_normalize_gpu(x, mode=norm_mode)
        rotation = codec.transform.get_rotation()
        z = y.reshape(-1, y.shape[-1]) @ rotation
        bank = torch.stack([
            quantizer._quantise(z)[0].reshape(
                1, y.shape[1], groups, codec.pq.d)
            for quantizer in codec.pq.quantizers
        ])
        residual = bank[:, 0] - z.reshape(y.shape[1], groups, codec.pq.d)
        errors = torch.stack([
            residual[:, :, g] @ rotation.t()[
                g * codec.pq.d:(g + 1) * codec.pq.d]
            for g in range(groups)
        ], dim=1) * std[0]
        magnitude = errors.flatten(2).norm(dim=2)
        directions = errors / magnitude.clamp_min(1e-12)[..., None, None]
        flat_directions = directions.reshape(
            modes_count * groups, y.shape[1], y.shape[-1])
        response_rows = []
        for start in range(0, len(flat_directions), jvp_chunk):
            direction = flat_directions[start:start + jvp_chunk]
            base = x.expand(len(direction), -1, -1)
            plus = tail.forward_nograd(base + jvp_eps * direction)
            minus = tail.forward_nograd(base - jvp_eps * direction)
            response_rows.append((plus - minus) / (2 * jvp_eps))
        response_bank = torch.cat(response_rows).reshape(
            modes_count, groups, *teacher.shape[1:])
        response_bank *= magnitude.reshape(
            modes_count, groups, *([1] * (teacher.ndim - 1)))
        quad_cost_per_image[image_index] = (
            response_bank.flatten(2).square().sum(2).t().cpu().numpy())

        bank_g = bank.permute(3, 0, 1, 2, 4)
        group_index = torch.arange(groups, device=device)[None, :]
        for start in range(0, len(allocations), allocation_chunk):
            stop = min(start + allocation_chunk, len(allocations))
            selected_modes = torch.as_tensor(
                allocations[start:stop], dtype=torch.long, device=device)
            selected = bank_g[group_index, selected_modes].permute(
                0, 2, 3, 1, 4).reshape(stop - start, y.shape[1], -1)
            reconstructed = batch_inv_normalize_gpu(
                selected @ rotation.t(),
                mu.expand(stop - start, *mu.shape[1:]),
                std.expand(stop - start, *std.shape[1:]))
            delta = tail.forward_nograd(reconstructed) - teacher
            for offset, mode_row in enumerate(selected_modes):
                response = response_bank[
                    mode_row, torch.arange(groups, device=device)]
                parts = decompose_output_vectors(
                    calibrated_phi[start + offset], response.unsqueeze(0),
                    delta[offset:offset + 1])
                for key in keys:
                    values[key][start + offset, image_index] = (
                        parts[key].item())

    quad_phi = np.stack([
        quad_cost_per_image[:, np.arange(groups), modes].sum(1)
        for modes in allocations])
    scales = np.exp2(-2.0 * mode_bits / codec.pq.d)
    c_per_image = quad_cost_per_image[
        :, :, reference_mode] / scales[reference_mode]
    analytic_phi = np.stack([
        (c_per_image * scales[modes][None]).sum(1)
        for modes in allocations])
    structural = values["distortion"] - quad_phi
    rate_mismatch = quad_phi - analytic_phi
    analytic_remainder = values["distortion"] - analytic_phi
    calibration_drift = quad_phi - calibrated_phi[:, None]
    calibrated_remainder = values["distortion"] - calibrated_phi[:, None]
    quad_lookup_error = quad_phi - values["self_quad"]
    structural_component_error = structural - (
        values["cross"] + values["nonlinear"])
    reconstruction_error = values["distortion"] - (
        analytic_phi + rate_mismatch + structural)
    relative_error = np.abs(reconstruction_error) / np.maximum(
        np.abs(values["distortion"]), 1.0)
    quad_lookup_relative = np.abs(quad_lookup_error) / np.maximum(
        np.abs(quad_phi), 1.0)
    structural_component_relative = np.abs(
        structural_component_error) / np.maximum(
            np.abs(values["distortion"]), 1.0)
    contract_components = {
        "rate_model_mismatch": rate_mismatch,
        "cross": values["cross"],
        "nonlinear": values["nonlinear"],
    }
    component_ranges = {
        key: float(np.ptp(value.mean(1)))
        for key, value in contract_components.items()
    }
    summary = {
        "measurement_contract": "paired_unified_v1",
        "n_allocations": int(len(allocations)),
        "n_images": int(len(features_array)),
        "target_rate": target,
        "calibrated_ideal_model": (
            "discrete_table"
            if ideal_cost_table is not None else "common_exponential"),
        "analytic_model": "reference_mode_common_exponential",
        "reference_bit": float(reference_bit),
        "max_rate_error": float(np.abs(totals - target).max()),
        "jvp_eps": float(jvp_eps),
        "component_mean": {
            key: float(value.mean())
            for key, value in contract_components.items()
        },
        "component_range": component_ranges,
        "structural_remainder_range": float(np.ptp(structural.mean(1))),
        "rate_model_mismatch_range": float(np.ptp(rate_mismatch.mean(1))),
        "analytic_remainder_range": float(
            np.ptp(analytic_remainder.mean(1))),
        "calibration_drift_range": float(
            np.ptp(calibration_drift.mean(1))),
        "max_abs_quad_lookup_error": float(
            np.abs(quad_lookup_error).max()),
        "max_rel_quad_lookup_error": float(quad_lookup_relative.max()),
        "max_abs_structural_component_error": float(
            np.abs(structural_component_error).max()),
        "max_rel_structural_component_error": float(
            structural_component_relative.max()),
        "dominant_component_by_range": max(
            component_ranges, key=component_ranges.get),
        "max_abs_decomposition_error": float(
            np.abs(reconstruction_error).max()),
        "max_rel_decomposition_error": float(relative_error.max()),
        "cross_bound_violations": int(np.count_nonzero(
            np.abs(values["cross"]) > values["cross_bound"] + 1e-5)),
        "nonlinear_bound_violations": int(np.count_nonzero(
            np.abs(values["nonlinear"]) >
            values["nonlinear_bound"] + 1e-5)),
    }
    summary.update(paired_contract_statistics(
        values["distortion"], quad_phi, analytic_phi,
        bootstraps=bootstrap_count, batch_size=bootstrap_batch,
        seed=bootstrap_seed))
    summary["split_half_stability"] = {
        name: split_half_stability(
            value, repeats=stability_repeats, seed=bootstrap_seed + 1)
        for name, value in (
            ("structural_remainder", structural),
            ("rate_model_mismatch", rate_mismatch),
            ("analytic_remainder", analytic_remainder),
        )
    }
    summary["structural_rate_allocation_correlation"] = (
        allocation_correlation(
            structural.mean(1), rate_mismatch.mean(1)))
    arrays = {
        "allocations": allocations, "rates": rates, "total_rates": totals,
        "calibrated_phi": calibrated_phi,
        "quad_cost_per_image": quad_cost_per_image,
        "quad_phi": quad_phi, "analytic_phi": analytic_phi,
        "structural_remainder": structural,
        "rate_model_mismatch": rate_mismatch,
        "analytic_remainder": analytic_remainder,
        "calibration_drift": calibration_drift,
        "calibrated_remainder": calibrated_remainder,
        "quad_lookup_error": quad_lookup_error,
        "structural_component_error": structural_component_error,
        "decomposition_error": reconstruction_error,
        "current_c_per_image": c_per_image, **values,
    }
    return summary, arrays
