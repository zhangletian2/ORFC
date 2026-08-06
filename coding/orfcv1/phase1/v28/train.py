"""Exact-budget branch-coordinate training on train-5k and fixed val-500."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from codec_v1 import load_codec_v1, save_codec_v1
from .. import engine, frozen, tail as tail_mod
from ..v12 import qhard
from ..v12 import train as joint
from ..v21.config import SPECS, activate
from ..v22.allocation_dp import topk_allocations
from ..v22.probe import local_table


def hard_distortion(codec, tail, resident, allocations, image_batch, alloc_chunk):
    return engine.evaluate_allocations(
        codec, tail, resident, np.asarray(allocations, dtype=np.int64),
        image_batch=image_batch, pair_budget=image_batch * alloc_chunk,
        per_image=True)


@torch.no_grad()
def candidates(codec, tail, train, base, bits, budget, topk, image_batch,
               alloc_chunk, first_cycle):
    probes, keys = local_table(base, len(bits))
    values = hard_distortion(
        codec, tail, train, probes, image_batch, alloc_chunk).mean(1)
    costs = np.zeros((len(base), len(bits)), dtype=np.float64)
    for index, (group, mode) in enumerate(keys[1:], 1):
        costs[group, mode] = float(values[index] - values[0])
    ranked = topk_allocations(costs, bits, budget, max(topk * 2, topk + 2))
    rows, seen = [base.copy()], {tuple(base.tolist())}
    uniform = np.full(len(base), bits.index(budget // len(base)), dtype=np.int64)
    if first_cycle and tuple(uniform.tolist()) not in seen:
        rows.append(uniform); seen.add(tuple(uniform.tolist()))
    for _, row in ranked:
        key = tuple(row)
        if key not in seen:
            rows.append(np.asarray(row, dtype=np.int64)); seen.add(key)
        if len(rows) == topk + 1:
            break
    return np.stack(rows), {
        "base_train_mse": float(values[0]),
        "probe_count": len(probes),
        "cost_min": float(costs.min()), "cost_max": float(costs.max()),
        "dp_predicted": [float(item[0]) for item in ranked[:topk]]}


def adapt_branch(parent, optimizer_state, tail, train, allocation, permutations,
                 batch, lr, lrs, taus, train_rate_lambda):
    branch = copy.deepcopy(parent)
    joint.make_trainable(branch)
    optimizer = torch.optim.Adam(branch.parameters(), lr=lr)
    if optimizer_state is not None:
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    books_before = [q.codebooks.detach().clone() for q in branch.pq.quantizers]
    groups = torch.arange(len(allocation), device=train.device)
    selected = torch.as_tensor(allocation, device=train.device)
    losses = []
    for permutation, epoch_lr, tau in zip(permutations, lrs, taus):
        for group in optimizer.param_groups:
            group["lr"] = float(epoch_lr)
        for first in range(0, len(permutation), batch):
            index = permutation[first:first + batch]
            if len(index) != batch:
                continue
            y = train.y.index_select(0, index)
            mu = train.mu.index_select(0, index)
            std = train.std.index_select(0, index)
            teacher = train.teacher.index_select(0, index)
            distortion, _, rate = qhard.distortion_sparse(
                branch, tail, y, mu, std, teacher, allocation,
                codeword_temperature=tau, return_rate=True)
            loss = (distortion + train_rate_lambda * rate * train.tokens).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(branch.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    mask = torch.zeros(len(allocation), len(branch.pq.quantizers),
                       dtype=torch.bool, device=train.device)
    mask[groups, selected] = True
    drift = torch.stack([
        (q.codebooks.detach() - old).flatten(1).norm(dim=1)
        for q, old in zip(branch.pq.quantizers, books_before)], dim=1)
    return (branch, optimizer.state_dict(), float(np.mean(losses)), mask,
            float(drift[mask].min()), float(drift[~mask].max()))


@torch.no_grad()
def val_record(codec, tail, val, allocation, image_batch):
    values = hard_distortion(codec, tail, val, allocation[None], image_batch, 1)[0]
    return {"hard_tail_mse": float(values.mean()),
            "per_image_std": float(values.std()), "images": int(len(values))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", default="blk20", choices=tuple(SPECS))
    parser.add_argument("--anchor", default="R64")
    parser.add_argument("--source-codec", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--branch-epochs", type=int, default=1)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"),
                        default="constant")
    parser.add_argument("--tau-start", type=float, default=0.5)
    parser.add_argument("--tau-end", type=float, default=0.005)
    parser.add_argument("--tau-floor", type=float, default=0.0)
    parser.add_argument("--train-rate-lambda", type=float, default=0.0)
    parser.add_argument("--image-batch", type=int, default=20)
    parser.add_argument("--alloc-chunk", type=int, default=2)
    parser.add_argument("--train-images", type=int, default=5000)
    parser.add_argument("--val-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    started = time.time()
    config = activate(args.block)
    anchor = config.ANCHOR_BY_NAME[args.anchor]
    device, out = torch.device(args.device), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is non-empty")
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    val_rows = np.asarray(manifest["validation_select"]["indices"][:args.val_images])
    if len(val_rows) != args.val_images or len(np.unique(val_rows)) != len(val_rows):
        raise SystemExit("invalid validation manifest")
    engine.configure_precision(config.ALLOW_TF32)
    codec = load_codec_v1(Path(args.source_codec), device=device)
    joint.make_trainable(codec)
    if any(getattr(q, "use_rate", False) for q in codec.pq.quantizers) \
            != bool(args.train_rate_lambda):
        raise SystemExit("codec prior and training objective disagree")
    bits = tuple(int(round(math.log2(q.codebooks.shape[1])))
                 for q in codec.pq.quantizers)
    allocation = engine.uniform_allocation(anchor, config.GROUPS)
    train_paths = config.load_split("train_fit")
    train = engine.ResidentSet(
        train_paths[0], train_paths[1], train_paths[2][:args.train_images], device)
    val_paths = config.load_split("train_val")
    val = engine.ResidentSet(val_paths[0], val_paths[1], val_rows, device)
    tail = tail_mod.build_tail(config.LAYER, device)
    initial_orth = frozen.orthogonality_error(codec)
    initial = val_record(codec, tail, val, allocation, args.image_batch)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    optimizer_state, events, best = None, [], None
    best_payload = None
    cycles = 1 if args.smoke else args.cycles
    total_epochs = cycles * args.branch_epochs
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    for cycle in range(cycles):
        rows, table = candidates(
            codec, tail, train, allocation, bits, anchor.rate, args.topk,
            args.image_batch, args.alloc_chunk, cycle == 0)
        permutations = [torch.randperm(
            train.count, generator=generator, device=device)
            for _ in range(args.branch_epochs)]
        branches = []
        taus = [joint.codeword_tau(
            cycle * args.branch_epochs + epoch + 1, total_epochs,
            args.tau_start, args.tau_end) for epoch in range(args.branch_epochs)]
        taus = [max(value, args.tau_floor) for value in taus]
        lrs = [(config.cosine_lr(
            cycle * args.branch_epochs + epoch, total_epochs, args.lr)
            if args.lr_schedule == "cosine" else args.lr)
            for epoch in range(args.branch_epochs)]
        for index, row in enumerate(rows):
            branch, state, train_loss, selected, active_drift, inactive_drift = adapt_branch(
                codec, optimizer_state, tail, train, row, permutations, args.batch,
                args.lr, lrs, taus, args.train_rate_lambda)
            record = val_record(branch, tail, val, row, args.image_batch)
            record.update(index=index, allocation=row.tolist(), train_loss=train_loss,
                          selected_book_drift_min=active_drift,
                          unselected_book_drift_max=inactive_drift)
            branches.append((record, branch, state, selected))
        winner = min(range(len(branches)),
                     key=lambda i: branches[i][0]["hard_tail_mse"])
        event = {"cycle": cycle + 1, "table": table, "winner": winner,
                 "branches": [item[0] for item in branches]}
        events.append(event)
        codec, optimizer_state = branches[winner][1], branches[winner][2]
        allocation = rows[winner].copy()
        score = branches[winner][0]["hard_tail_mse"]
        if best is None or score < best:
            best = score
            save_codec_v1(codec, out / "codec.pt")
            np.save(out / "allocation.npy", allocation)
            torch.save(optimizer_state, out / "optimizer.pt")
            best_payload = {"cycle": cycle + 1, **branches[winner][0]}
        del branches
        print(f"cycle={cycle + 1}/{cycles} candidates={len(rows)} "
              f"winner={winner} val_D={score:.3f} alloc={allocation.tolist()}",
              flush=True)
    codec = load_codec_v1(out / "codec.pt", device=device).eval()
    allocation = np.load(out / "allocation.npy")
    final = val_record(codec, tail, val, allocation, args.image_batch)
    parity = qhard.selfcheck(codec, tail, val, allocation, args.image_batch)
    final_orth = frozen.orthogonality_error(codec)
    if engine.nominal_rate(allocation, anchor) != anchor.rate:
        raise SystemExit("INVALID_EXPERIMENT: nominal rate")
    if parity["rel_gap"] > parity["tolerance"]:
        raise SystemExit("INVALID_EXPERIMENT: hard parity")
    if final_orth - initial_orth > config.ORTH_TOL:
        raise SystemExit("INVALID_EXPERIMENT: orthogonality")
    payload = {
        "plan": "v28_train5k_fixed_val500_branch_coordinate",
        "block": args.block, "anchor": args.anchor, "nominal_rate": anchor.rate,
        "manifest": str(Path(args.manifest).resolve()),
        "data": {"train": train.count, "validation_select": val.count,
                 "validation_rest_used": False},
        "cycles": cycles, "branch_epochs": args.branch_epochs,
        "total_path_epochs": total_epochs, "topk": args.topk, "batch": args.batch,
        "lr": args.lr, "tau_start": args.tau_start, "tau_end": args.tau_end,
        "lr_schedule": args.lr_schedule, "tau_floor": args.tau_floor,
        "train_rate_lambda": args.train_rate_lambda,
        "image_batch": args.image_batch, "alloc_chunk": args.alloc_chunk,
        "initial": initial, "best": best_payload, "final": final,
        "allocation": allocation.tolist(), "events": events,
        "hard_parity": parity, "orthogonality_initial": initial_orth,
        "orthogonality_final": final_orth,
        "peak_memory_gb": (torch.cuda.max_memory_allocated(device) / 2**30
                           if device.type == "cuda" else 0.0),
        "seconds": time.time() - started}
    (out / "train.json").write_text(json.dumps(payload, indent=2))
    (out / "training_complete.json").write_text(json.dumps({
        "best": best_payload, "allocation": allocation.tolist(),
        "hard_parity": parity}, indent=2))
    print(json.dumps({k: payload[k] for k in (
        "initial", "best", "final", "allocation", "peak_memory_gb", "seconds")},
        indent=2))


if __name__ == "__main__":
    main()
