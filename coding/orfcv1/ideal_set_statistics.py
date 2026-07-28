#!/usr/bin/env python3
"""Statistical diagnostics for fixed-budget ideal allocation sets."""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from p1_fixed_rate import topk_cost_allocate


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _solve_batch(tables, bits, budget):
    """Exact best path for each separable table in a batch."""
    count, groups, _ = tables.shape
    dp = np.full((count, budget + 1), np.inf)
    dp[:, 0] = 0
    parents = np.full((count, groups, budget + 1), -1, np.int8)
    for group in range(groups):
        new = np.full_like(dp, np.inf)
        choice = np.full((count, budget + 1), -1, np.int8)
        for mode, bit in enumerate(bits):
            candidate = dp[:, :budget - bit + 1] + tables[
                :, group, mode, None]
            target = new[:, bit:]
            better = candidate < target
            target[better] = candidate[better]
            choice[:, bit:][better] = mode
        dp, parents[:, group] = new, choice
    rows = np.arange(count)
    remaining = np.full(count, budget, dtype=int)
    modes = np.empty((count, groups), dtype=np.int8)
    for group in range(groups - 1, -1, -1):
        mode = parents[rows, group, remaining]
        if np.any(mode < 0):
            raise RuntimeError("fixed budget is infeasible")
        modes[:, group] = mode
        remaining -= bits[mode]
    return modes


def _bootstrap_optima(q, bits, budget, count, batch, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for start in range(0, count, batch):
        size = min(batch, count - start)
        ids = rng.integers(0, q.shape[1], size=(size, q.shape[1]))
        tables = q[:, ids, :].mean(2).transpose(1, 2, 0)
        rows.append(_solve_batch(tables, bits, budget))
    return np.concatenate(rows)


def command_bootstrap(args):
    with np.load(args.calibration, allow_pickle=False) as saved:
        q = np.asarray(saved["q_per_image_by_bit"], dtype=np.float64)
        bits = np.asarray(saved["mode_bits"], dtype=int)
        budget = int(np.asarray(saved["ideal_bits"]).sum())
    modes = _bootstrap_optima(
        q, bits, budget, args.bootstraps, args.bootstrap_batch, args.seed)
    unique, counts = np.unique(modes, axis=0, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    unique, counts = unique[order], counts[order]
    cumulative = np.cumsum(counts) / args.bootstraps
    confidence_size = int(np.searchsorted(
        cumulative, args.confidence) + 1)
    mean_table = q.mean(1).T
    nominal_value = mean_table[
        np.arange(mean_table.shape[0])[None], modes].sum(1)
    excess = nominal_value - nominal_value.min()
    top = topk_cost_allocate(
        mean_table, tuple(map(int, bits)), budget, args.max_topk)
    rank = {tuple(row): index + 1 for index, row in enumerate(
        np.searchsorted(bits, top["bits"]))}
    bootstrap_rank = np.asarray([
        rank.get(tuple(row), args.max_topk + 1) for row in modes])
    coverages = {
        str(k): float(np.mean(bootstrap_rank <= k))
        for k in args.report_topk if k <= args.max_topk
    }
    output = Path(args.output)
    np.savez_compressed(
        output.with_suffix(".npz"), bootstrap_optimal_modes=modes,
        unique_modes=unique, frequencies=counts / args.bootstraps,
        nominal_top_modes=np.searchsorted(bits, top["bits"]),
        nominal_top_values=top["values"])
    _write(output, {
        "calibration": args.calibration,
        "images": int(q.shape[1]),
        "bootstraps": args.bootstraps,
        "confidence": args.confidence,
        "unique_optima": int(len(unique)),
        "maximum_optimum_frequency": float(counts[0] / args.bootstraps),
        "frequency_ranked_confidence_set_size": confidence_size,
        "frequency_ranked_confidence_achieved": float(
            cumulative[confidence_size - 1]),
        "nominal_excess_at_confidence": float(np.quantile(
            excess, args.confidence, method="higher")),
        "nominal_topk_bootstrap_coverage": coverages,
        "scope": (
            "nonparametric bootstrap stability diagnostic; "
            "not a finite-sample or global recovery certificate"),
        "arrays": str(output.with_suffix(".npz")),
    })
    print(output)


def _suffix_counts(groups, bits, budget):
    counts = [[0] * (budget + 1) for _ in range(groups + 1)]
    counts[groups][0] = 1
    for group in range(groups - 1, -1, -1):
        for remaining in range(budget + 1):
            counts[group][remaining] = sum(
                counts[group + 1][remaining - bit]
                for bit in bits if bit <= remaining)
    return counts


def command_uniform(args):
    with np.load(args.calibration, allow_pickle=False) as saved:
        bits = tuple(map(int, saved["mode_bits"]))
        groups = int(saved["cost_table"].shape[0])
        budget = int(np.asarray(saved["ideal_bits"]).sum())
    counts = _suffix_counts(groups, bits, budget)
    rng, rows = random.Random(args.seed), set()
    while len(rows) < args.samples:
        remaining, modes = budget, []
        for group in range(groups):
            weights = [
                counts[group + 1][remaining - bit]
                if bit <= remaining else 0 for bit in bits]
            draw, total = rng.randrange(sum(weights)), 0
            for mode, weight in enumerate(weights):
                total += weight
                if draw < total:
                    modes.append(mode)
                    remaining -= bits[mode]
                    break
        rows.add(tuple(modes))
    allocations = np.asarray(sorted(rows), dtype=np.int64)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, allocations=allocations, mode_bits=bits, budget=budget,
        feasible_allocation_count=str(counts[0][budget]), seed=args.seed)
    _write(output.with_suffix(".json"), {
        "samples": int(len(allocations)),
        "groups": groups,
        "budget": budget,
        "feasible_allocation_count": str(counts[0][budget]),
        "sampling": "exact uniform without replacement",
        "scope": (
            "supports distributional audits only; it does not bound the "
            "worst allocation"),
        "arrays": str(output),
    })
    print(output)


def parser():
    main = argparse.ArgumentParser()
    sub = main.add_subparsers(dest="command", required=True)
    p = sub.add_parser("bootstrap")
    p.add_argument("--calibration", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bootstraps", type=int, default=5000)
    p.add_argument("--bootstrap-batch", type=int, default=256)
    p.add_argument("--confidence", type=float, default=.95)
    p.add_argument("--max-topk", type=int, default=256)
    p.add_argument(
        "--report-topk", type=int, nargs="+",
        default=(1, 4, 8, 16, 32, 64, 128, 256))
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("uniform")
    p.add_argument("--calibration", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return main


if __name__ == "__main__":
    args = parser().parse_args()
    globals()[f"command_{args.command}"](args)
