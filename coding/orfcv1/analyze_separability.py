#!/usr/bin/env python3
"""Pre-registered separability report for one fixed-total-rate decomposition.

Consumes a ``p1_fixed_rate.py decompose`` output measured on a design built by
``separable_design.py`` and answers three questions that the current remainder
definition cannot answer, because ``E:=D-Phi`` makes ``D-E=Phi`` an identity:

H1  how much of the measured remainder range survives once the ideal term is
    the best separable model rather than a prescribed one, judged against the
    image-sampling noise floor and on an allocation block that was never fitted;
H2  whether the separable model's global argmin, obtained by exact dynamic
    programming over the whole feasible set, is decision-equivalent to the
    measured distortion, reported as regret rather than exact recovery;
H3  how far the JVP and analytic coefficient tables sit from the fitted ones,
    after the unidentified gauge component is projected out.

Every threshold is recorded in the output so the verdict cannot be restated
after the fact.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from separable_projection import (
    attainable_rank, cell_coverage, design_matrix, dp_summary, fit_separable,
    gauge_null_space, leave_one_out_residual, leverage, residual_operator,
    separable_values, strip_gauge,
)


def _ptp(values):
    return float(np.ptp(np.asarray(values, dtype=np.float64)))


def _noise_floor(per_image, repeats, seed):
    """Range induced by image sampling alone, from random half splits."""
    rng = np.random.default_rng(seed)
    images = per_image.shape[1]
    half = images // 2
    draws = []
    for _ in range(repeats):
        order = rng.permutation(images)
        left = per_image[:, order[:half]].mean(1)
        right = per_image[:, order[half:2 * half]].mean(1)
        draws.append(_ptp(left - right))
    return {
        "median": float(np.median(draws)),
        "p95": float(np.quantile(draws, 0.95)),
        "repeats": int(repeats),
        "images_per_half": int(half),
    }


def _excess(observed, floor):
    """Component of a range not explained by the sampling floor, in quadrature."""
    return float(np.sqrt(max(0.0, observed ** 2 - floor ** 2)))


def _shuffle_control(values, allocations, modes, repeats, seed):
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        permuted = np.asarray(allocations)[rng.permutation(len(allocations))]
        draws.append(_ptp(fit_separable(values, permuted, modes)["residual"]))
    return float(np.median(draws))


def _rank_of(value, pool):
    return int((np.asarray(pool) < value).sum())


def _separable_null_band(distortion, allocations, modes, fit_idx, hold_idx,
                         repeats, seed):
    """Transfer-residual distribution when the mean truly is separable.

    The null world keeps the observed per-image fluctuations but replaces the
    mean by its separable projection, so the band measures exactly how much
    transfer residual the image sampling and the design leverage manufacture on
    their own.  Comparing the observed transfer against this band tests
    separability without an arbitrary tolerance.
    """
    values = np.asarray(distortion, dtype=np.float64)
    truth = fit_separable(values.mean(1), allocations, modes)["fitted"]
    fluctuation = values - values.mean(1, keepdims=True)
    rng = np.random.default_rng(seed)
    images = values.shape[1]
    draws = []
    for _ in range(repeats):
        picks = rng.integers(0, images, size=images)
        sample = truth + fluctuation[:, picks].mean(1)
        fitted = fit_separable(sample[fit_idx], allocations[fit_idx], modes)
        transferred = separable_values(
            fitted["coefficients"], allocations[hold_idx], modes)
        draws.append(_ptp(sample[hold_idx] - transferred))
    return {
        "median": float(np.median(draws)),
        "p95": float(np.quantile(draws, 0.95)),
        "repeats": int(repeats),
    }


def command_report(args):
    data = np.load(args.decomposition, allow_pickle=False)
    design = np.load(args.design, allow_pickle=False)
    allocations = np.asarray(data["allocations"], dtype=np.int64)
    if not np.array_equal(allocations, np.asarray(design["allocations"])):
        raise ValueError(
            "decomposition and design disagree on the allocation order")
    block = np.asarray(design["design_block"], dtype=np.int64)
    bits = np.asarray(
        design["mode_bits"] if "mode_bits" in design.files
        else design["calibration__mode_bits"], dtype=np.int64)
    groups, modes = allocations.shape[1], int(len(bits))
    budget = int(bits[allocations[0]].sum())

    distortion = np.asarray(data["distortion"], dtype=np.float64)
    mean_d = distortion.mean(1)
    quad_phi = np.asarray(data["quad_phi"], dtype=np.float64).mean(1)
    analytic_phi = np.asarray(data["analytic_phi"], dtype=np.float64).mean(1)

    fit_idx = np.flatnonzero(block == 0)
    hold_idx = np.flatnonzero(block == 1)
    if len(fit_idx) < attainable_rank(groups, modes):
        raise ValueError("fit block is smaller than the attainable rank")
    if len(hold_idx) == 0:
        raise ValueError("design carries no holdout block")

    fitted = fit_separable(mean_d[fit_idx], allocations[fit_idx], modes)
    table = fitted["table"]
    design_fit = design_matrix(allocations[fit_idx], modes)
    operator, rank = residual_operator(design_fit)
    hold_pred = separable_values(
        fitted["coefficients"], allocations[hold_idx], modes)
    hold_resid = mean_d[hold_idx] - hold_pred
    loo, loo_keep = leave_one_out_residual(fitted["residual"], design_fit)

    floor_fit = _noise_floor(distortion[fit_idx], args.half_repeats, args.seed)
    floor_hold = _noise_floor(
        distortion[hold_idx], args.half_repeats, args.seed + 1)

    ranges = {
        "fit_block": {
            "analytic": _ptp(mean_d[fit_idx] - analytic_phi[fit_idx]),
            "jvp": _ptp(mean_d[fit_idx] - quad_phi[fit_idx]),
            "separable_in_sample": _ptp(fitted["residual"]),
            "separable_loo": _ptp(loo[loo_keep]),
            "shuffled_control": _shuffle_control(
                mean_d[fit_idx], allocations[fit_idx], modes,
                args.shuffle_repeats, args.seed + 2),
            "noise_floor": floor_fit,
        },
        "holdout_block": {
            "analytic": _ptp(mean_d[hold_idx] - analytic_phi[hold_idx]),
            "jvp": _ptp(mean_d[hold_idx] - quad_phi[hold_idx]),
            "separable_transferred": _ptp(hold_resid),
            "noise_floor": floor_hold,
        },
    }

    null_band = _separable_null_band(
        distortion, allocations, modes, fit_idx, hold_idx,
        args.null_repeats, args.seed + 3)
    ranges["holdout_block"]["separable_null_band"] = null_band

    observed_transfer = ranges["holdout_block"]["separable_transferred"]
    h1_reference = ranges["holdout_block"]["analytic"]
    h1_excess = _excess(observed_transfer, floor_hold["median"])
    h1_pass = bool(observed_transfer <= null_band["p95"])
    h1_shrink = bool(h1_excess < args.h1_fraction * h1_reference)

    global_fit = dp_summary(table, bits, budget)
    global_jvp = dp_summary(
        np.asarray(data["quad_cost_per_image"], dtype=np.float64).mean(0),
        bits, budget)
    c_g = np.asarray(data["current_c_per_image"], dtype=np.float64).mean(0)
    analytic_table = c_g[:, None] * np.exp2(-2.0 * bits[None, :] / args.dimension)
    global_ana = dp_summary(analytic_table, bits, budget)

    pool_pred = separable_values(fitted["coefficients"], allocations, modes)
    pool_best = int(np.argmin(pool_pred))
    measured_best = int(np.argmin(mean_d))
    decision = {
        "measured_argmin_index": measured_best,
        "measured_argmin_allocation": allocations[measured_best].tolist(),
        "separable_pool_argmin_index": pool_best,
        "separable_pool_argmin_rank_under_d": _rank_of(
            mean_d[pool_best], mean_d),
        "jvp_pool_argmin_rank_under_d": _rank_of(
            mean_d[int(np.argmin(quad_phi))], mean_d),
        "analytic_pool_argmin_rank_under_d": _rank_of(
            mean_d[int(np.argmin(analytic_phi))], mean_d),
        "pool_regret_separable": float(mean_d[pool_best] - mean_d.min()),
        "pool_size": int(len(allocations)),
        "global_separable": {
            "top2_gap": global_fit["top2_gap"],
            "span": global_fit["span"],
            "argmin": global_fit["argmin"].tolist(),
            "argmin_in_pool": bool(
                any(np.array_equal(global_fit["argmin"], row)
                    for row in allocations)),
        },
        "global_jvp_span": global_jvp["span"],
        "global_analytic_span": global_ana["span"],
    }
    residual_bound = ranges["holdout_block"]["separable_transferred"]
    decision["regret_bound_two_omega"] = float(2.0 * residual_bound)
    h2_pass = bool(
        decision["pool_regret_separable"] <= decision["regret_bound_two_omega"])

    null = gauge_null_space(groups, modes, bits, budget)
    jvp_table = np.asarray(
        data["quad_cost_per_image"], dtype=np.float64).mean(0)
    reference = np.linalg.norm(strip_gauge(jvp_table, null))
    coefficients = {
        "jvp_gap_relative": float(
            np.linalg.norm(strip_gauge(jvp_table - table, null)) /
            max(reference, 1e-12)),
        "analytic_gap_relative": float(
            np.linalg.norm(strip_gauge(analytic_table - table, null)) /
            max(reference, 1e-12)),
        "gauge_dimension": int(null.shape[0]),
    }

    coverage = cell_coverage(allocations, modes)
    hat = leverage(design_fit)
    summary = {
        "decomposition": str(args.decomposition),
        "design": str(args.design),
        "groups": groups,
        "modes": modes,
        "budget": budget,
        "images": int(distortion.shape[1]),
        "design_rank": int(rank),
        "attainable_rank": attainable_rank(groups, modes),
        "cells_never_used": int((coverage == 0).sum()),
        "high_leverage_rows": int((hat >= 1.0 - 1e-6).sum()),
        "ranges": ranges,
        "decision": decision,
        "coefficients": coefficients,
        "prereg": {
            "h1_statement": (
                "the holdout-block separable residual range does not exceed "
                "the 95th percentile of the range produced by image sampling "
                "and design leverage alone under an exactly separable mean"),
            "h1_observed_transfer": observed_transfer,
            "h1_null_p95": null_band["p95"],
            "h1_pass": h1_pass,
            "h1_shrink_statement": (
                "the same residual range, after removing the image-sampling "
                "floor in quadrature, is below "
                f"{args.h1_fraction} of the analytic remainder range on the "
                "same block"),
            "h1_excess_over_floor": h1_excess,
            "h1_shrink_threshold": float(args.h1_fraction * h1_reference),
            "h1_shrink_pass": h1_shrink,
            "h2_statement": (
                "the pool regret of the separable global argmin is within "
                "twice the holdout residual range"),
            "h2_pass": h2_pass,
            "scope": (
                "exact recovery of a unique argmin is not tested; the top-two "
                "gap over the full feasible set is reported but is not a "
                "target"),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.followup:
        np.savez_compressed(
            args.followup,
            allocations=np.vstack([
                global_fit["argmin"][None, :],
                global_fit["argmax"][None, :],
                global_jvp["argmin"][None, :],
                global_ana["argmin"][None, :],
            ]),
            labels=np.asarray(
                ["separable_min", "separable_max", "jvp_min", "analytic_min"]))
    print(json.dumps(summary["prereg"], indent=2, sort_keys=True))
    print(json.dumps(summary["ranges"], indent=2, sort_keys=True))


def parser():
    main = argparse.ArgumentParser()
    sub = main.add_subparsers(dest="command", required=True)
    p = sub.add_parser("report")
    p.add_argument("--decomposition", required=True)
    p.add_argument("--design", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--followup")
    p.add_argument("--dimension", type=int, default=32)
    p.add_argument("--half-repeats", type=int, default=200)
    p.add_argument("--shuffle-repeats", type=int, default=20)
    p.add_argument("--null-repeats", type=int, default=200)
    p.add_argument("--h1-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return main


if __name__ == "__main__":
    parsed = parser().parse_args()
    globals()[f"command_{parsed.command}"](parsed)
