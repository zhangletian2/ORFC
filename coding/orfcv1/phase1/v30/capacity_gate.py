"""Gate 1: fixed-U/allocation capacity of independent versus nested PQ."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from codec_v1 import load_codec_v1
from opq import batch_inv_normalize_gpu
from .. import engine, tail as tail_mod
from ..v12 import qhard
from ..v12 import train as joint
from ..v21.config import SPECS, activate
from . import common, nested


def mixed_exact_allocation(groups, bits, budget, uniform_mode):
    """Exact-budget row containing every mode with minimum nonuniform cells."""
    states = {(0, 0): (0, ())}
    full = (1 << len(bits)) - 1
    for _ in range(int(groups)):
        updated = {}
        for (used, mask), (penalty, prefix) in states.items():
            for mode, bit in enumerate(bits):
                target = used + bit
                if target > budget:
                    continue
                key = (target, mask | (1 << mode))
                value = (penalty + int(mode != uniform_mode), prefix + (mode,))
                if key not in updated or value < updated[key]:
                    updated[key] = value
        states = updated
    if (budget, full) not in states:
        raise ValueError("no exact-budget allocation contains every mode")
    return np.asarray(states[(budget, full)][1], dtype=np.int64)


@torch.no_grad()
def evaluate_pair(independent, nested_codec, tail, resident, allocation,
                  image_batch):
    independent_rows, nested_rows = [], []
    for first in range(0, resident.count, int(image_batch)):
        y, mu, std, teacher = resident.slice(first, first + image_batch)
        decoded_i = qhard.quantise(independent, y, allocation)[0]
        decoded_n = nested.reconstruct(nested_codec, y, allocation)[0]
        output = tail(batch_inv_normalize_gpu(
            torch.cat((decoded_i, decoded_n)),
            torch.cat((mu, mu)), torch.cat((std, std))))
        target = torch.cat((teacher, teacher))
        value = (output - target).square().reshape(2, y.shape[0], -1).sum(-1)
        independent_rows.append(value[0]); nested_rows.append(value[1])
    return (torch.cat(independent_rows).cpu().numpy(),
            torch.cat(nested_rows).cpu().numpy())


def convergence(values, window, tolerance):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < int(window):
        return {"converged": False, "relative_span": None}
    tail = values[-int(window):]
    span = float((tail.max() - tail.min()) / max(abs(tail.mean()), 1e-12))
    return {"converged": bool(span <= tolerance), "relative_span": span}


def paired_interval(independent, nested_values, samples, seed):
    independent = np.asarray(independent, dtype=np.float64)
    nested_values = np.asarray(nested_values, dtype=np.float64)
    rng, ratios = np.random.default_rng(seed), []
    remaining = int(samples)
    while remaining:
        count = min(1000, remaining)
        index = rng.integers(0, len(independent), (count, len(independent)))
        ratios.append(nested_values[index].mean(1) /
                      independent[index].mean(1) - 1.0)
        remaining -= count
    ratios = np.concatenate(ratios)
    return {
        "mean_relative_loss": float(
            nested_values.mean() / independent.mean() - 1.0),
        "ci95": [float(np.quantile(ratios, 0.025)),
                 float(np.quantile(ratios, 0.975))],
        "one_sided_ucb95": float(np.quantile(ratios, 0.95)),
        "bootstrap_samples": int(samples)}


@torch.no_grad()
def absolute_centroids(codec, tree=False):
    if tree:
        return [codec.pq.composed_codebook(mode).detach().clone()
                for mode in range(codec.pq.num_modes)]
    return [quantizer.codebooks.detach().clone()
            for quantizer in codec.pq.quantizers]


@torch.no_grad()
def centroid_update_norms(codec, initial, tree=False):
    current = absolute_centroids(codec, tree=tree)
    return [float((now - start).norm())
            for now, start in zip(current, initial)]


def paired_train(independent, nested_codec, tail, train, val, allocation,
                 epochs, batch, lr, tau_start, tau_end, check_every,
                 image_batch, seed):
    for codec in (independent, nested_codec):
        for parameter in codec.transform.parameters():
            parameter.requires_grad_(False)
    parameters_i = [q.codebooks for q in independent.pq.quantizers]
    parameters_n = list(nested_codec.pq.parameters())
    optimizer_i = torch.optim.Adam(parameters_i, lr=float(lr))
    optimizer_n = torch.optim.Adam(parameters_n, lr=float(lr))
    scheduler_i = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_i, T_max=int(epochs), eta_min=0.0)
    scheduler_n = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_n, T_max=int(epochs), eta_min=0.0)
    generator = torch.Generator(device=train.device).manual_seed(int(seed))
    centroids_i0 = absolute_centroids(independent)
    centroids_n0 = absolute_centroids(nested_codec, tree=True)
    curve = []
    initial_i, initial_n = evaluate_pair(
        independent, nested_codec, tail, val, allocation, image_batch)
    curve.append({"epoch": 0, "independent": float(initial_i.mean()),
                  "nested": float(initial_n.mean()), "lr": float(lr),
                  "absolute_centroid_update_frobenius": {
                      "independent": centroid_update_norms(
                          independent, centroids_i0),
                      "tree": centroid_update_norms(
                          nested_codec, centroids_n0, tree=True)}})
    for epoch in range(int(epochs)):
        permutation = torch.randperm(
            train.count, generator=generator, device=train.device)
        tau = joint.codeword_tau(
            epoch + 1, int(epochs), tau_start, tau_end)
        losses_i, losses_n = [], []
        for first in range(0, train.count, int(batch)):
            index = permutation[first:first + int(batch)]
            y = train.y.index_select(0, index)
            mu = train.mu.index_select(0, index)
            std = train.std.index_select(0, index)
            teacher = train.teacher.index_select(0, index)
            decoded_i = qhard.quantise(
                independent, y, allocation, codeword_temperature=tau)[0]
            decoded_n = nested.reconstruct(
                nested_codec, y, allocation, temperature=tau)[0]
            output = tail(batch_inv_normalize_gpu(
                torch.cat((decoded_i, decoded_n)),
                torch.cat((mu, mu)), torch.cat((std, std))))
            target = torch.cat((teacher, teacher))
            value = (output - target).square().reshape(
                2, y.shape[0], -1).sum(-1).mean(1)
            optimizer_i.zero_grad(set_to_none=True)
            optimizer_n.zero_grad(set_to_none=True)
            value.sum().backward()
            torch.nn.utils.clip_grad_norm_(parameters_i, 1.0)
            torch.nn.utils.clip_grad_norm_(parameters_n, 1.0)
            optimizer_i.step(); optimizer_n.step()
            losses_i.append(float(value[0].detach()))
            losses_n.append(float(value[1].detach()))
        scheduler_i.step(); scheduler_n.step()
        if ((epoch + 1) % int(check_every) == 0 or epoch + 1 == epochs):
            hard_i, hard_n = evaluate_pair(
                independent, nested_codec, tail, val, allocation, image_batch)
            curve.append({
                "epoch": epoch + 1,
                "independent": float(hard_i.mean()),
                "nested": float(hard_n.mean()),
                "train_independent": float(np.mean(losses_i)),
                "train_nested": float(np.mean(losses_n)),
                "lr": float(optimizer_i.param_groups[0]["lr"]),
                "temperature": float(tau),
                "absolute_centroid_update_frobenius": {
                    "independent": centroid_update_norms(
                        independent, centroids_i0),
                    "tree": centroid_update_norms(
                        nested_codec, centroids_n0, tree=True)}})
    final_i, final_n = evaluate_pair(
        independent, nested_codec, tail, val, allocation, image_batch)
    return curve, final_i, final_n


def train_coverage(codec, tail, train, allocations, epochs, batch, lr,
                   tau_start, tau_end, seed):
    for parameter in codec.transform.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(codec.pq.parameters(), lr=float(lr))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(epochs), eta_min=0.0)
    generator = torch.Generator(device=train.device).manual_seed(int(seed))
    step = 0
    for epoch in range(int(epochs)):
        permutation = torch.randperm(
            train.count, generator=generator, device=train.device)
        tau = joint.codeword_tau(epoch + 1, int(epochs), tau_start, tau_end)
        for first in range(0, train.count, int(batch)):
            index = permutation[first:first + int(batch)]
            allocation = allocations[step % len(allocations)]; step += 1
            value, _ = nested.distortion(
                codec, tail,
                train.y.index_select(0, index),
                train.mu.index_select(0, index),
                train.std.index_select(0, index),
                train.teacher.index_select(0, index), allocation,
                temperature=tau)
            loss = value.mean(); optimizer.zero_grad(set_to_none=True)
            loss.backward(); torch.nn.utils.clip_grad_norm_(
                codec.pq.parameters(), 1.0); optimizer.step()
        scheduler.step()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", default="blk20", choices=tuple(SPECS))
    parser.add_argument("--anchor", default="R64")
    parser.add_argument("--source-codec", required=True)
    parser.add_argument("--allocation")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tau-start", type=float, default=0.5)
    parser.add_argument("--tau-end", type=float, default=0.005)
    parser.add_argument("--train-images", type=int, default=5000)
    parser.add_argument("--val-images", type=int, default=500)
    parser.add_argument("--image-batch", type=int, default=32)
    parser.add_argument("--check-every", type=int, default=5)
    parser.add_argument("--convergence-window", type=int, default=4)
    parser.add_argument("--convergence-tolerance", type=float, default=0.002)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    started = time.time()
    config = activate(args.block); anchor = config.ANCHOR_BY_NAME[args.anchor]
    device, out = torch.device(args.device), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = load_codec_v1(Path(args.source_codec), device=device)
    if any(getattr(q, "use_rate", False) for q in source.pq.quantizers):
        raise SystemExit("capacity gate requires a Tail-MSE codec without ECVQ")
    bits = tuple(int(round(np.log2(q.codebooks.shape[1])))
                 for q in source.pq.quantizers)
    if bits != tuple(anchor.mode_bits):
        raise SystemExit(f"source menu {bits} does not match {anchor.mode_bits}")
    base = (np.load(args.allocation).astype(np.int64)
            if args.allocation else
            engine.uniform_allocation(anchor, config.GROUPS))
    if common.nominal_rate(base, bits) != anchor.rate:
        raise SystemExit("allocation violates the exact nominal budget")
    mixed = mixed_exact_allocation(
        config.GROUPS, bits, anchor.rate, anchor.uniform_mode)
    shifts = (0, max(1, config.GROUPS // 3), max(1, 2 * config.GROUPS // 3))
    allocations = np.unique(np.stack(
        [base] + [np.roll(mixed, shift) for shift in shifts]), axis=0)
    if any(common.nominal_rate(row, bits) != anchor.rate for row in allocations):
        raise SystemExit("capacity allocation violates the exact budget")
    train_n = min(args.train_images, 16) if args.smoke else args.train_images
    val_n = min(args.val_images, 16) if args.smoke else args.val_images
    epochs = min(args.epochs, 2) if args.smoke else args.epochs
    check_every = 1 if args.smoke else args.check_every
    train_paths = config.load_split("train_fit")
    val_paths = config.load_split("train_val")
    train = engine.ResidentSet(*train_paths[:2], train_paths[2][:train_n], device)
    val = engine.ResidentSet(*val_paths[:2], val_paths[2][:val_n], device)
    tail = tail_mod.build_tail(config.LAYER, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    nested_initial = nested.from_independent(source)
    records = []
    for arm, allocation in enumerate(allocations):
        independent = copy.deepcopy(source)
        nested_codec = copy.deepcopy(nested_initial)
        for parameter in nested_codec.transform.parameters():
            parameter.requires_grad_(False)
        curve, final_i, final_n = paired_train(
            independent, nested_codec, tail, train, val, allocation,
            epochs, args.batch, args.lr, args.tau_start, args.tau_end,
            check_every, args.image_batch, args.seed)
        conv_i = convergence(
            [item["independent"] for item in curve],
            args.convergence_window, args.convergence_tolerance)
        conv_n = convergence(
            [item["nested"] for item in curve],
            args.convergence_window, args.convergence_tolerance)
        interval = paired_interval(
            final_i, final_n,
            min(args.bootstrap_samples, 100) if args.smoke
            else args.bootstrap_samples, args.seed + arm)
        records.append({
            "arm": arm, "allocation": allocation.tolist(),
            "modes_present": sorted(set(map(int, allocation))),
            "curve": curve, "convergence": {
                "independent": conv_i, "nested": conv_n},
            "final": {"independent": float(final_i.mean()),
                      "nested": float(final_n.mean())},
            "paired_capacity_loss": interval})
        nested.save_checkpoint(
            nested_codec, args.source_codec,
            out / f"nested_arm{arm}.pt", extra={"arm": arm})
    all_converged = all(
        record["convergence"][side]["converged"]
        for record in records for side in ("independent", "nested"))
    worst_ucb = max(record["paired_capacity_loss"]["one_sided_ucb95"]
                    for record in records)
    verdict = ("INCONCLUSIVE" if not all_converged else
               "PASS" if worst_ucb <= args.threshold else "FAIL")
    result = {
        "plan": "v30_hierarchical_tree_capacity_gate", "block": args.block,
        "anchor": args.anchor, "allocations": allocations.tolist(),
        "source_codec": str(Path(args.source_codec).resolve()),
        "mode_bits": list(bits),
        "parameterization": "parent_conditioned_residual_tree",
        "stage_sizes": list(nested_initial.pq.stage_sizes),
        "branch_sizes": list(nested_initial.pq.branch_sizes), "fixed_u": True,
        "epochs": epochs, "batch": args.batch, "lr": args.lr,
        "lr_schedule": "cosine_to_zero", "tau_schedule": [
            args.tau_start, args.tau_end], "same_batch_order": True,
        "same_updates": True, "train_images": train.count,
        "centroid_update_contract": {
            "object": "absolute composed centroid",
            "reference": "arm initialization",
            "metric": "per-mode Frobenius norm"},
        "updates_per_arm": int(epochs * int(np.ceil(train.count / args.batch))),
        "validation_select": val.count, "check_every": check_every,
        "convergence_contract": {
            "window": args.convergence_window,
            "relative_span_tolerance": args.convergence_tolerance},
        "arms": records, "all_sides_converged": all_converged,
        "worst_one_sided_ucb95": worst_ucb, "threshold": args.threshold,
        "verdict": verdict,
        "peak_memory_bytes": (int(torch.cuda.max_memory_allocated(device))
                              if device.type == "cuda" else 0),
        "seconds_before_optional_export": time.time() - started}
    (out / "capacity_gate.json").write_text(json.dumps(result, indent=2))
    if verdict == "PASS":
        nested_codec = copy.deepcopy(nested_initial)
        train_coverage(
            nested_codec, tail, train, allocations, epochs, args.batch,
            args.lr, args.tau_start, args.tau_end, args.seed)
        nested.save_checkpoint(
            nested_codec, args.source_codec, out / "nested_codec.pt",
            extra={"capacity_gate": result})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
