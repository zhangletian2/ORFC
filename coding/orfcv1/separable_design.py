#!/usr/bin/env python3
"""Build a fixed-total-rate allocation design that identifies the separable model.

The existing candidate pools were mined for large remainder range and span only
rank 73 of the attainable 160, leaving 83 of the 192 ``(group, mode)`` cells
unobserved.  This builder produces three explicitly labelled blocks:

``fit``      compensated exchanges around the uniform base, constructed so that
             every cell is observed and the design reaches full attainable
             rank, plus an independent uniform sample so the model is never
             asked to extrapolate outside the fitted hull;
``holdout``  an exact uniform sample of the whole feasible set, never used for
             fitting, which is the only honest test of extrapolation away from
             the base;
``anchor``   allocations carried over from earlier pools so the decision-level
             comparison stays on a common footing.

Blocks are disjoint and the whole design is validated against one common total
rate.  The output is a drop-in ``--calibration`` file for
``p1_fixed_rate.py decompose``.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

from ideal_set_statistics import _suffix_counts
from separable_projection import (
    attainable_rank, cell_coverage, design_matrix,
)

BLOCK_NAMES = ("fit", "holdout", "anchor")


def _rank(allocations, modes, rcond=1e-10):
    values = np.linalg.svd(design_matrix(allocations, modes), compute_uv=False)
    return int((values > values.max() * rcond).sum())


def _uniform_base(groups, modes, bits, budget):
    for mode in range(modes):
        if groups * int(bits[mode]) == budget:
            return np.full(groups, mode, dtype=np.int64)
    raise ValueError(
        "no uniform allocation meets the budget; pass --base explicitly")


def _shifted(base, offsets, modes):
    """Compensated exchange rounds covering every cell of every group."""
    groups = len(base)
    middle = int(base[0])
    rows = []
    for offset in offsets:
        if offset % groups == 0 or (2 * offset) % groups == 0:
            raise ValueError(f"offset {offset} degenerates for {groups} groups")
        for group in range(groups):
            for down in range(middle):
                debt = middle - down
                row = base.copy()
                row[group] = down
                partner = (group + offset) % groups
                if middle + debt < modes:
                    row[partner] = middle + debt
                else:
                    row[partner] = modes - 1
                    spill = debt - (modes - 1 - middle)
                    second = (group + 2 * offset) % groups
                    if middle + spill >= modes or second in (group, partner):
                        continue
                    row[second] = middle + spill
                rows.append(row)
    return rows


def _topup(rows, base, modes, bits, budget, target, seed, attempts):
    """Add random compensated exchanges until the design reaches full rank."""
    rng = np.random.default_rng(seed)
    groups = len(base)
    seen = {tuple(map(int, row)) for row in rows}
    current = np.asarray(rows, dtype=np.int64)
    for _ in range(attempts):
        if _rank(current, modes) >= target:
            break
        row = base.copy()
        moved = rng.permutation(groups)[:rng.integers(2, 6)]
        row[moved] = rng.integers(0, modes, size=len(moved))
        deficit = budget - int(bits[row].sum())
        order = rng.permutation(groups)
        for group in order:
            if deficit == 0:
                break
            for mode in range(modes):
                if int(bits[mode]) - int(bits[row[group]]) == deficit:
                    row[group] = mode
                    deficit = 0
                    break
        if deficit != 0:
            continue
        key = tuple(map(int, row))
        if key in seen:
            continue
        seen.add(key)
        current = np.vstack([current, row[None, :]])
    return current


def _uniform_sample(groups, bits, budget, samples, seed):
    import random

    counts = _suffix_counts(groups, tuple(map(int, bits)), budget)
    rng, rows = random.Random(seed), set()
    guard = 0
    while len(rows) < samples and guard < samples * 100:
        guard += 1
        remaining, modes = budget, []
        for group in range(groups):
            weights = [
                counts[group + 1][remaining - bit] if bit <= remaining else 0
                for bit in map(int, bits)
            ]
            draw, total = rng.randrange(sum(weights)), 0
            for mode, weight in enumerate(weights):
                total += weight
                if draw < total:
                    modes.append(mode)
                    remaining -= int(bits[mode])
                    break
        rows.add(tuple(modes))
    return np.asarray(sorted(rows), dtype=np.int64), str(counts[0][budget])


def command_build(args):
    source = np.load(args.calibration, allow_pickle=False)
    fields = {name: source[name] for name in source.files}

    def field(name):
        key = name if name in fields else f"calibration__{name}"
        if key not in fields:
            raise KeyError(f"{name} is absent from {args.calibration}")
        return fields[key]

    cost_table = field("cost_table")
    groups, modes = int(cost_table.shape[0]), int(cost_table.shape[1])
    bits = np.asarray(field("mode_bits"), dtype=np.int64)
    budget = (
        args.budget if args.budget else int(np.asarray(field("ideal_bits")).sum()))
    base = (
        np.asarray([int(v) for v in args.base.split(",")], dtype=np.int64)
        if args.base else _uniform_base(groups, modes, bits, budget))
    if base.shape != (groups,) or int(bits[base].sum()) != budget:
        raise ValueError("base allocation does not meet the fixed total rate")

    offsets = [int(v) for v in args.offsets.split(",")]
    rows, seen = [], set()
    for row in [base.copy()] + _shifted(base, offsets, modes):
        key = tuple(map(int, row))
        if key not in seen and int(bits[row].sum()) == budget:
            seen.add(key)
            rows.append(row)
    rows = np.asarray(rows, dtype=np.int64)
    target = min(attainable_rank(groups, modes), args.max_rank or 10 ** 9)
    near = _topup(
        rows, base, modes, bits, budget, target, args.seed, args.topup_attempts)
    far, feasible = _uniform_sample(
        groups, bits, budget, args.fit_uniform, args.seed + 7) \
        if args.fit_uniform else (np.zeros((0, groups), np.int64), None)
    near_keys = {tuple(map(int, row)) for row in near}
    far = np.asarray(
        [row for row in far if tuple(map(int, row)) not in near_keys],
        dtype=np.int64).reshape(-1, groups)
    fit_rows = np.vstack([near, far])
    fit_keys = {tuple(map(int, row)) for row in fit_rows}

    holdout, feasible = _uniform_sample(
        groups, bits, budget, args.holdout_samples, args.seed + 1)
    holdout = np.asarray(
        [row for row in holdout if tuple(map(int, row)) not in fit_keys],
        dtype=np.int64).reshape(-1, groups)
    holdout_keys = fit_keys | {tuple(map(int, row)) for row in holdout}

    anchor = np.zeros((0, groups), dtype=np.int64)
    if args.anchor:
        loaded = np.load(args.anchor, allow_pickle=False)
        key = args.anchor_key if args.anchor_key in loaded.files else "allocations"
        candidate = np.asarray(loaded[key], dtype=np.int64)
        if args.anchor_limit:
            candidate = candidate[:args.anchor_limit]
        anchor = np.asarray(
            [row for row in candidate
             if tuple(map(int, row)) not in holdout_keys],
            dtype=np.int64).reshape(-1, groups)

    allocations = np.vstack([fit_rows, holdout, anchor])
    block = np.concatenate([
        np.zeros(len(fit_rows), np.int64),
        np.ones(len(holdout), np.int64),
        np.full(len(anchor), 2, np.int64),
    ])
    totals = bits[allocations].sum(1)
    if not np.all(totals == budget):
        raise ValueError("design violates the fixed total rate")

    coverage = cell_coverage(allocations, modes)
    fit_rank = _rank(fit_rows, modes)
    full_rank = _rank(allocations, modes)
    summary = {
        "groups": groups,
        "modes": modes,
        "budget": budget,
        "base": base.tolist(),
        "offsets": offsets,
        "n_fit": int(len(fit_rows)),
        "n_fit_near_base": int(len(near)),
        "n_fit_uniform": int(len(far)),
        "n_holdout": int(len(holdout)),
        "n_anchor": int(len(anchor)),
        "n_total": int(len(allocations)),
        "fit_block_rank": fit_rank,
        "full_design_rank": full_rank,
        "attainable_rank": attainable_rank(groups, modes),
        "rank_saturated": bool(fit_rank >= target),
        "cells_never_used": int((coverage == 0).sum()),
        "min_cell_count": int(coverage.min()),
        "median_cell_count": float(np.median(coverage)),
        "feasible_allocation_count": feasible,
        "blocks": {name: index for index, name in enumerate(BLOCK_NAMES)},
        "gauge": (
            "coefficients identified modulo a per-group affine gauge; "
            "fitted values, residuals, ranges, argmins and DP bounds are "
            "gauge invariant"),
        "holdout_scope": (
            "exact uniform sample of the full feasible set, disjoint from the "
            "fit block, never used for fitting"),
    }
    if not summary["rank_saturated"]:
        raise ValueError(
            f"fit block rank {fit_rank} below target {target}; "
            "increase --offsets or --topup-attempts")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields["allocations"] = allocations
    fields["design_block"] = block
    fields["design_base"] = base
    fields["design_rank"] = np.asarray(full_rank)
    fields["design_attainable_rank"] = np.asarray(
        attainable_rank(groups, modes))
    np.savez_compressed(output, **fields)
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parser():
    main = argparse.ArgumentParser()
    sub = main.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--calibration", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--base")
    p.add_argument("--budget", type=int)
    p.add_argument("--offsets", default="1,7")
    p.add_argument("--holdout-samples", type=int, default=96)
    p.add_argument("--fit-uniform", type=int, default=64)
    p.add_argument("--anchor")
    p.add_argument("--anchor-key", default="allocations")
    p.add_argument("--anchor-limit", type=int, default=48)
    p.add_argument("--max-rank", type=int)
    p.add_argument("--topup-attempts", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    return main


if __name__ == "__main__":
    parsed = parser().parse_args()
    globals()[f"command_{parsed.command}"](parsed)
