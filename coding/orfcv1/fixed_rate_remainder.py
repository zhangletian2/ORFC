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
):
    """Measure ``D``, ``Phi`` and ``E=D-Phi`` on fixed-rate allocations.

    The expensive frozen-tail call is batched over allocation chunks.  Returned
    per-image arrays preserve paired comparisons and can be saved separately
    from the compact summary.
    """
    allocations = np.asarray(allocations, dtype=np.int64)
    rates, totals, target = validate_fixed_total_rate(
        allocations, cost_table, tolerance=rate_tolerance)
    phi = ideal_phi(c_g, rates, rate_dimension=codec.pq.d)
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

    paired = (
        distortion[e_max] - distortion[e_min]
        - (phi[e_max] - phi[e_min]))
    paired_std = float(paired.std(ddof=1)) if n_images > 1 else 0.0
    half = 1.96 * paired_std / max(np.sqrt(n_images), 1.0)
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
        "paired_omega_ci95": [
            float(paired.mean() - half),
            float(paired.mean() + half),
        ],
    }
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
