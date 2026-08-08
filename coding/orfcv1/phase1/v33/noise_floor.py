"""Evaluation noise floor → sparse Kendall-τ same-rank threshold.

Protocol
--------
1. Fix the v12 ``train_val`` list (``N_VAL=500``).
2. Partition it into ``n_parts`` *disjoint* subsets.
3. On one frozen checkpoint, score the same candidate allocations on every
   subset (one allocation at a time).
4. For each allocation pair, the across-subset score-difference series has an
   empirical std; aggregate those into ``sigma_noise``.
5. Same-rank threshold for :func:`sparse_kendall_tau` is
   ``tie_threshold = tie_mult * sigma_noise``.

This is a *measurement* of evaluation noise, not a correctness certificate for
any ranking.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .. import engine, tail as tail_mod
from ..v12 import config as C
from ..v21.config import SPECS, activate
from . import checkpoint as ckpt
from . import distortion as D
from . import ranking
from . import valset


def _score_matrix(score_fn, allocations, subset_row_lists, build_subset_scorer):
    """Return ``[n_parts, n_alloc]`` means.

    ``build_subset_scorer(rows) -> callable(allocation) -> float`` lets the
    caller inject either a real resident evaluator or a synthetic stub.
    """
    allocations = np.asarray(allocations, dtype=np.int64)
    if allocations.ndim == 1:
        allocations = allocations[None]
    matrix = np.zeros((len(subset_row_lists), len(allocations)), dtype=np.float64)
    for r, rows in enumerate(subset_row_lists):
        scorer = build_subset_scorer(rows)
        for a, allocation in enumerate(allocations):
            matrix[r, a] = float(scorer(allocation))
    return matrix


def estimate_sigma_noise(score_matrix):
    """Aggregate paired-difference std across disjoint subset replicates.

    ``score_matrix[r, a]`` = mean distortion of allocation ``a`` on subset ``r``.
    For every allocation pair ``(i, j)``, take ``std_r (score[r,i] - score[r,j])``
    and return the RMS of those pair stds as ``sigma_noise``.
    """
    matrix = np.asarray(score_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError("need >=2 subsets and >=2 allocations")
    n_parts, n_alloc = matrix.shape
    pair_stds = []
    for i in range(n_alloc):
        for j in range(i + 1, n_alloc):
            diffs = matrix[:, i] - matrix[:, j]
            pair_stds.append(float(diffs.std(ddof=1)))
    pair_stds = np.asarray(pair_stds, dtype=np.float64)
    sigma = float(np.sqrt(np.mean(pair_stds ** 2)))
    return {
        "sigma_noise": sigma,
        "pair_std_median": float(np.median(pair_stds)),
        "pair_std_mean": float(np.mean(pair_stds)),
        "pair_std_max": float(np.max(pair_stds)),
        "n_pairs": int(len(pair_stds)),
        "n_parts": int(n_parts),
        "n_allocations": int(n_alloc),
        "pair_stds": pair_stds,
    }


def tie_threshold_from_noise(sigma_noise, tie_mult=1.0):
    """Same-rank threshold for sparse Kendall-τ."""
    return float(tie_mult) * float(sigma_noise)


def calibrate(score_matrix, tie_mult=1.0):
    """Full calibration report from a ``[R, A]`` score matrix."""
    stats = estimate_sigma_noise(score_matrix)
    threshold = tie_threshold_from_noise(stats["sigma_noise"], tie_mult)
    # Self-consistency: sparse τ between subset 0 and subset 1 at this threshold.
    tau_01, c, d, decisive, dropped = ranking.sparse_kendall_tau(
        score_matrix[0], score_matrix[1], tie_threshold=threshold)
    stats = {k: v for k, v in stats.items() if k != "pair_stds"}
    stats.update({
        "tie_mult": float(tie_mult),
        "tie_threshold": threshold,
        "subset0_vs_subset1_sparse_tau": tau_01,
        "subset0_vs_subset1_decisive": int(decisive),
        "subset0_vs_subset1_dropped": int(dropped),
        "subset0_vs_subset1_concordant": int(c),
        "subset0_vs_subset1_discordant": int(d),
        "score_matrix": np.asarray(score_matrix, dtype=np.float64).tolist(),
    })
    return stats


def make_resident_scorer(codec, tail, resident, image_batch=16):
    """One-allocation-at-a-time mean Tail MSE over a resident set."""

    @torch.no_grad()
    def score(allocation):
        values = D.evaluate(
            codec, tail, resident, allocation, image_batch=image_batch)
        return float(np.asarray(values).reshape(-1).mean())

    return score


def make_subset_builder(codec, tail, device, image_batch=16):
    def build(rows):
        resident = valset.resident_from_rows(device, rows)
        return make_resident_scorer(codec, tail, resident, image_batch)

    return build


def random_exact_budget_allocations(groups, bits, rate, count, seed=0):
    """Unique exact-budget random allocations (CPU)."""
    from ..v12.allocation_policy import FixedBudgetAllocationPolicy

    policy = FixedBudgetAllocationPolicy(groups, bits, rate)
    generator = torch.Generator().manual_seed(int(seed))
    draws = policy.build(1.0).sample(int(count) * 4, generator=generator)
    unique, seen = [], set()
    for row in draws.cpu().numpy():
        key = tuple(map(int, row))
        if key not in seen:
            seen.add(key)
            unique.append(np.asarray(row, dtype=np.int64))
        if len(unique) >= int(count):
            break
    if len(unique) < int(count):
        raise RuntimeError("could not draw enough unique exact-budget allocations")
    return np.stack(unique)


def run_calibration(codec, tail, device, allocations, n_parts=5, n_val=valset.N_VAL,
                    seed=0, image_batch=16, tie_mult=1.0):
    parts = valset.disjoint_partitions(n_val, n_parts, seed=seed)
    # Map partition indices → absolute val-list rows (identity for fixed list).
    rows = valset.fixed_val_rows(n_val)
    subset_rows = [rows[part] for part in parts]
    builder = make_subset_builder(codec, tail, device, image_batch=image_batch)
    matrix = _score_matrix(
        None, allocations, subset_rows, builder)
    report = calibrate(matrix, tie_mult=tie_mult)
    report["subset_sizes"] = [int(len(r)) for r in subset_rows]
    report["n_val"] = int(n_val)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", default="blk20", choices=tuple(SPECS))
    parser.add_argument("--anchor", default="R64")
    parser.add_argument("--checkpoint",
                        help=f"{ckpt.FORMAT} path (required unless --smoke)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-parts", type=int, default=5)
    parser.add_argument("--n-val", type=int, default=valset.N_VAL)
    parser.add_argument("--n-alloc", type=int, default=16)
    parser.add_argument("--image-batch", type=int, default=16)
    parser.add_argument("--tie-mult", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny synthetic path (no real features/tail)")
    args = parser.parse_args(argv)

    started = time.time()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        report = _smoke_report(args)
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required unless --smoke")
        activate(args.block)
        engine.configure_precision(C.ALLOW_TF32)
        device = torch.device(args.device)
        codec, payload = ckpt.load_checkpoint(args.checkpoint, device=device)
        anchor = C.ANCHOR_BY_NAME[args.anchor]
        allocations = random_exact_budget_allocations(
            codec.pq.G, tuple(codec.pq.mode_bits), anchor.rate,
            args.n_alloc, seed=args.seed)
        tail = tail_mod.build_tail(C.LAYER, device)
        report = run_calibration(
            codec, tail, device, allocations,
            n_parts=args.n_parts, n_val=args.n_val, seed=args.seed,
            image_batch=args.image_batch, tie_mult=args.tie_mult)
        report["checkpoint"] = str(Path(args.checkpoint).resolve())
        report["meta"] = payload.get("meta", {})
        report["anchor"] = args.anchor
        report["allocations"] = allocations.tolist()

    report["plan"] = "v33_noise_floor"
    report["seconds"] = time.time() - started
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k != "score_matrix"}, indent=2))


def _smoke_report(args):
    """Deterministic synthetic calibration (no GPU features required)."""
    rng = np.random.default_rng(args.seed)
    n_alloc = max(4, int(args.n_alloc))
    n_parts = max(2, int(args.n_parts))
    # True scores + iid subset noise.
    true = rng.normal(size=n_alloc)
    sigma = 0.05
    matrix = true[None, :] + rng.normal(scale=sigma, size=(n_parts, n_alloc))
    report = calibrate(matrix, tie_mult=args.tie_mult)
    report["smoke"] = True
    report["injected_sigma"] = sigma
    report["subset_sizes"] = [args.n_val // n_parts] * n_parts
    report["n_val"] = int(args.n_val)
    return report


if __name__ == "__main__":
    main()
