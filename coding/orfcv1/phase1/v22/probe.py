"""Validate a task-aware local cost table and exact-budget DP allocation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from codec_v1 import load_codec_v1
from .. import engine, tail as tail_mod
from ..v12 import qhard
from ..v12.allocation_policy import FixedBudgetAllocationPolicy
from ..v21.config import SPECS, activate
from .allocation_dp import topk_allocations


def split_resident(config, rows, first, count, device):
    paths = config.load_split("train_val")
    selected = np.asarray(rows[first:first + count], dtype=np.int64)
    if len(selected) != count:
        raise ValueError("validation split is too small")
    return engine.ResidentSet(paths[0], paths[1], selected, device)


@torch.no_grad()
def evaluate(codec, tail, resident, allocations, image_batch, rate_lambda):
    allocations = np.asarray(allocations, dtype=np.int64)
    distortion = engine.evaluate_allocations(
        codec, tail, resident, allocations, image_batch=image_batch,
        pair_budget=image_batch, per_image=True)
    if rate_lambda <= 0:
        rates = np.zeros_like(distortion)
        objective = distortion
    else:
        chunks = []
        for first in range(0, resident.count, image_batch):
            chunks.append(qhard.rates(
                codec, resident.y[first:first + image_batch], allocations
            ).cpu().numpy())
        rates = np.concatenate(chunks, axis=1)
        objective = rates * resident.tokens + float(rate_lambda) * distortion
    return distortion, rates, objective


def local_table(base, modes):
    allocations, keys = [base.copy()], [(None, None)]
    for group in range(len(base)):
        for mode in range(modes):
            if mode == base[group]:
                continue
            candidate = base.copy()
            candidate[group] = mode
            allocations.append(candidate)
            keys.append((group, mode))
    return np.stack(allocations), keys


def random_allocations(groups, bits, rate, count, device, seed):
    policy = FixedBudgetAllocationPolicy(groups, bits, rate).to(device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    draws = policy.build(1.0).sample(int(count), generator=generator)
    return np.unique(draws.cpu().numpy(), axis=0)


def order_agreement(predicted, measured):
    agree = total = 0
    for left in range(len(predicted)):
        for right in range(left + 1, len(predicted)):
            a = np.sign(predicted[left] - predicted[right])
            b = np.sign(measured[left] - measured[right])
            if a and b:
                total += 1
                agree += int(a == b)
    return float(agree / total) if total else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", default="blk20", choices=tuple(SPECS))
    parser.add_argument("--anchor", default="R64")
    parser.add_argument("--codec", required=True)
    parser.add_argument("--allocation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cal-images", type=int, default=64)
    parser.add_argument("--select-images", type=int, default=64)
    parser.add_argument("--report-images", type=int, default=128)
    parser.add_argument("--image-batch", type=int, default=16)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--random", type=int, default=16)
    parser.add_argument("--rate-lambda", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    started = time.time()
    config = activate(args.block)
    anchor = config.ANCHOR_BY_NAME[args.anchor]
    device = torch.device(args.device)
    codec = load_codec_v1(Path(args.codec), device=device).eval()
    bits = tuple(int(round(np.log2(q.codebooks.shape[1])))
                 for q in codec.pq.quantizers)
    if bits != tuple(anchor.mode_bits):
        raise SystemExit(f"menu {bits} does not match {anchor.mode_bits}")
    base = (np.load(args.allocation).astype(np.int64)
            if args.allocation else
            engine.uniform_allocation(anchor, config.GROUPS))
    if engine.nominal_rate(base, anchor) != anchor.rate:
        raise SystemExit("base allocation violates the nominal budget")

    all_rows = config.load_split("train_val")[2]
    total_images = args.cal_images + args.select_images + args.report_images
    if total_images > len(all_rows):
        raise ValueError("requested disjoint splits exceed train_val")
    for count in (args.cal_images, args.select_images, args.report_images):
        if count % args.image_batch:
            raise ValueError("every split size must be divisible by image-batch")
    cal = split_resident(config, all_rows, 0, args.cal_images, device)
    select = split_resident(
        config, all_rows, args.cal_images, args.select_images, device)
    report = split_resident(
        config, all_rows, args.cal_images + args.select_images,
        args.report_images, device)
    tail = tail_mod.build_tail(config.LAYER, device)

    probes, keys = local_table(base, len(bits))
    d_cal, r_cal, j_cal = evaluate(
        codec, tail, cal, probes, args.image_batch, args.rate_lambda)
    base_j = float(j_cal[0].mean())
    costs = np.zeros((config.GROUPS, len(bits)), dtype=np.float64)
    for index, (group, mode) in enumerate(keys[1:], start=1):
        costs[group, mode] = float(j_cal[index].mean() - base_j)
    winners = topk_allocations(costs, bits, anchor.rate, args.topk)
    dp_allocations = np.asarray([entry[1] for entry in winners], dtype=np.int64)

    random_rows = random_allocations(
        config.GROUPS, bits, anchor.rate, args.random, device, args.seed)
    d_random, r_random, j_random = evaluate(
        codec, tail, cal, random_rows, args.image_batch, args.rate_lambda)
    predicted_random = np.asarray([
        base_j + sum(costs[g, mode] for g, mode in enumerate(row))
        for row in random_rows])
    measured_random = j_random.mean(1)
    correlation = (float(np.corrcoef(predicted_random, measured_random)[0, 1])
                   if len(random_rows) > 1 else None)

    candidates = np.concatenate((base[None], dp_allocations), axis=0)
    d_select, r_select, j_select = evaluate(
        codec, tail, select, candidates, args.image_batch, args.rate_lambda)
    selection_means = j_select.mean(1)
    selected_index = int(np.argmin(selection_means))
    selected = candidates[selected_index]
    d_report, r_report, j_report = evaluate(
        codec, tail, report, np.stack((base, selected)),
        args.image_batch, args.rate_lambda)

    result = {
        "plan": "v22_task_cost_dp_probe", "block": args.block,
        "anchor": anchor.name, "nominal_rate": anchor.rate,
        "codec": str(Path(args.codec).resolve()),
        "allocation_source": args.allocation,
        "splits": {"cal": args.cal_images, "select": args.select_images,
                   "report": args.report_images, "disjoint": True},
        "rate_lambda": args.rate_lambda, "mode_bits": list(bits),
        "base_allocation": base.tolist(),
        "dp_topk": [{"rank": rank + 1, "predicted_objective":
                     float(base_j + value), "allocation": list(allocation)}
                    for rank, (value, allocation) in enumerate(winners)],
        "surrogate_random_audit": {
            "allocations": int(len(random_rows)), "pearson": correlation,
            "pairwise_order_agreement": order_agreement(
                predicted_random, measured_random),
            "mae": float(np.abs(predicted_random - measured_random).mean()),
            "max_abs_error": float(
                np.abs(predicted_random - measured_random).max())},
        "selection": {
            "winner_index_in_base_plus_topk": selected_index,
            "allocation": selected.tolist(),
            "base_objective": float(selection_means[0]),
            "winner_objective": float(selection_means[selected_index]),
            "gain": float(selection_means[0] - selection_means[selected_index])},
        "report": {
            "base_distortion": float(d_report[0].mean()),
            "selected_distortion": float(d_report[1].mean()),
            "distortion_gain": float(d_report[0].mean() - d_report[1].mean()),
            "base_rate_bpt": float(r_report[0].mean()),
            "selected_rate_bpt": float(r_report[1].mean()),
            "base_objective": float(j_report[0].mean()),
            "selected_objective": float(j_report[1].mean()),
            "objective_gain": float(j_report[0].mean() - j_report[1].mean())},
        "seconds": time.time() - started}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
