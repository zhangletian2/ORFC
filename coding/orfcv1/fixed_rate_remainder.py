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
    distortion_per_image, phi, comparison=None, bootstraps=1000,
    batch_size=32, seed=42,
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

    def omega(values):
        return float(np.ptp(values.mean(1) - phi))

    point = omega(distortion)
    if bootstraps < 1 or distortion.shape[1] < 2:
        samples = np.asarray([point])
        differences = (
            np.asarray([point - omega(other)]) if other is not None else None)
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
                    other[:, ids].mean(2) - phi[:, None], axis=0)
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
            "paired_omega_change": float(point - omega(other)),
            "paired_omega_change_ci95": np.quantile(
                differences, [0.025, 0.975]).tolist(),
        })
    return result


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
):
    """Measure ``D=Phi+M+C+N`` for realised multi-mode PQ errors.

    ``M`` is the realised self-quadratic/menu mismatch, ``C`` is quadratic
    cross-group coupling and ``N`` is finite-amplitude nonlinear propagation.
    Central finite differences estimate the local group responses.
    """
    if jvp_eps <= 0 or jvp_chunk < 1:
        raise ValueError("jvp_eps and jvp_chunk must be positive")
    if not hasattr(codec.pq, "quantizers"):
        raise TypeError("fixed-rate decomposition requires MultiModeSoftPQ")
    allocations = np.asarray(allocations, dtype=np.int64)
    rates, totals, target = validate_fixed_total_rate(
        allocations, cost_table, tolerance=rate_tolerance)
    phi = allocation_phi(
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
                    phi[start + offset], response.unsqueeze(0),
                    delta[offset:offset + 1])
                for key in keys:
                    values[key][start + offset, image_index] = (
                        parts[key].item())

    means = {key: array.mean(1) for key, array in values.items()}
    remainder = (
        values["menu"] + values["cross"] + values["nonlinear"])
    reconstruction_error = values["distortion"] - (
        phi[:, None] + remainder)
    relative_error = np.abs(reconstruction_error) / np.maximum(
        np.abs(values["distortion"]), 1.0)
    component_ranges = {
        key: float(np.ptp(means[key]))
        for key in ("menu", "cross", "nonlinear")
    }
    summary = {
        "n_allocations": int(len(allocations)),
        "n_images": int(len(features_array)),
        "target_rate": target,
        "ideal_model": (
            "discrete_table"
            if ideal_cost_table is not None else "common_exponential"),
        "max_rate_error": float(np.abs(totals - target).max()),
        "jvp_eps": float(jvp_eps),
        "component_mean": {
            key: float(means[key].mean())
            for key in ("menu", "cross", "nonlinear")
        },
        "component_range": component_ranges,
        "remainder_range": float(np.ptp(remainder.mean(1))),
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
    arrays = {
        "allocations": allocations, "rates": rates, "total_rates": totals,
        "phi": phi, "remainder": remainder,
        "decomposition_error": reconstruction_error, **values,
    }
    return summary, arrays
