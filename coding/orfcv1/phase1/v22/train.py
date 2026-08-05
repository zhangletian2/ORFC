"""Deterministic allocation DP alternating with ORFC codec updates."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from codec_v1 import load_codec_v1, save_codec_v1
from opq import batch_normalize_gpu
from .. import engine, frozen, tail as tail_mod
from ..v12 import qhard
from ..v12 import train as joint
from ..v15.audit_final import centroid_usage
from ..v21.config import SPECS, activate
from .allocation_dp import topk_allocations
from .probe import evaluate, local_table, split_resident


@torch.no_grad()
def propose(codec, tail, cal, select, base, bits, rate, image_batch,
            topk, rate_lambda):
    probes, keys = local_table(base, len(bits))
    _, _, cal_objective = evaluate(
        codec, tail, cal, probes, image_batch, rate_lambda)
    base_cal = float(cal_objective[0].mean())
    costs = np.zeros((len(base), len(bits)), dtype=np.float64)
    for index, (group, mode) in enumerate(keys[1:], start=1):
        costs[group, mode] = float(cal_objective[index].mean() - base_cal)
    winners = topk_allocations(costs, bits, rate, topk)
    rows, seen = [base.copy()], {tuple(base.tolist())}
    for _, row in winners:
        key = tuple(row.tolist())
        if key not in seen:
            rows.append(row.copy())
            seen.add(key)
    candidates = np.asarray(rows, dtype=np.int64)
    distortion, rates, objective = evaluate(
        codec, tail, select, candidates, image_batch, rate_lambda)
    values = objective.mean(1)
    winner = int(np.argmin(values))
    accepted = bool(winner != 0 and values[winner] < values[0])
    chosen = candidates[winner].copy() if accepted else base.copy()
    return chosen, {
        "base": base.tolist(), "proposed": candidates[winner].tolist(),
        "accepted": accepted, "topk": int(topk),
        "candidate_count": int(len(candidates)),
        "predicted": [float(base_cal + item[0]) for item in winners],
        "select_base_distortion": float(distortion[0].mean()),
        "select_winner_distortion": float(distortion[winner].mean()),
        "select_base_rate_bpt": float(rates[0].mean()),
        "select_winner_rate_bpt": float(rates[winner].mean()),
        "select_base_objective": float(values[0]),
        "select_winner_objective": float(values[winner]),
        "select_objective_gain": float(values[0] - values[winner])}


@torch.no_grad()
def validate(codec, tail, resident, allocation, image_batch, rate_lambda):
    distortion, rates, objective = evaluate(
        codec, tail, resident, allocation[None], image_batch, rate_lambda)
    return {"distortion": float(distortion.mean()),
            "rate_bpt": float(rates.mean()),
            "objective": float(objective.mean())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", default="blk20", choices=tuple(SPECS))
    parser.add_argument("--anchor", default="R64")
    parser.add_argument("--source-codec", required=True)
    parser.add_argument("--source-allocation")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tau-start", type=float, default=0.5)
    parser.add_argument("--tau-end", type=float, default=0.05)
    parser.add_argument("--rate-lambda", type=float, default=0.5)
    parser.add_argument("--outer-every", type=int, default=250)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--cal-images", type=int, default=64)
    parser.add_argument("--select-images", type=int, default=64)
    parser.add_argument("--report-images", type=int, default=128)
    parser.add_argument("--image-batch", type=int, default=16)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--skip-centroid", action="store_true")
    args = parser.parse_args(argv)
    if args.rate_lambda <= 0:
        raise SystemExit("V22 currently requires the V21 ECVQ objective")

    started = time.time()
    config = activate(args.block)
    np.random.seed(config.TRAIN_SEED)
    torch.manual_seed(config.TRAIN_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.TRAIN_SEED)
    anchor = config.ANCHOR_BY_NAME[args.anchor]
    device = torch.device(args.device)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is non-empty; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    engine.configure_precision(config.ALLOW_TF32)
    codec = load_codec_v1(Path(args.source_codec), device=device)
    joint.make_trainable(codec)
    bits = joint.actual_mode_bits(codec)
    if bits != tuple(anchor.mode_bits):
        raise SystemExit(f"menu {bits} does not match {anchor.mode_bits}")
    allocation = (np.load(args.source_allocation).astype(np.int64)
                  if args.source_allocation else
                  engine.uniform_allocation(anchor, config.GROUPS))
    if engine.nominal_rate(allocation, anchor) != anchor.rate:
        raise SystemExit("source allocation violates the nominal budget")
    if bool(getattr(codec.pq, "use_rate", False)) != (args.rate_lambda > 0):
        raise SystemExit("codec prior and rate objective disagree")

    optimizer = torch.optim.Adam(codec.parameters(), lr=float(args.lr))
    tail = tail_mod.build_tail(config.LAYER, device)
    rows = config.load_split("train_val")[2]
    cal = split_resident(config, rows, 0, args.cal_images, device)
    select = split_resident(
        config, rows, args.cal_images, args.select_images, device)
    report = split_resident(
        config, rows, args.cal_images + args.select_images,
        args.report_images, device)

    train_paths = config.load_split("train_fit")
    dataset = joint.CachedFeatureDataset(train_paths[0], train_paths[2])
    teacher_host = torch.from_numpy(np.load(train_paths[1])).float()
    if device.type == "cuda":
        teacher_host = teacher_host.pin_memory()
    staging = torch.empty(
        (args.batch, *teacher_host.shape[1:]), dtype=teacher_host.dtype,
        pin_memory=device.type == "cuda")
    generator = torch.Generator().manual_seed(config.TRAIN_SEED)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        generator=generator)
    per_epoch = len(loader)
    total = int(args.steps) if args.steps else int(args.epochs) * per_epoch
    schedule_epochs = max(1, math.ceil(total / per_epoch))
    iterator = iter(loader)
    initial_rotation = codec.transform.get_rotation().detach().clone()
    initial_books = [q.codebooks.detach().clone() for q in codec.pq.quantizers]
    initial_orth = frozen.orthogonality_error(codec)
    initial_report = validate(
        codec, tail, report, allocation, args.image_batch, args.rate_lambda)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    outer, trace, validation = [], [], []
    gradient_coverage = torch.zeros(
        config.GROUPS, len(bits), dtype=torch.long, device=device)

    allocation, event = propose(
        codec, tail, cal, select, allocation, bits, anchor.rate,
        args.image_batch, args.topk, args.rate_lambda)
    event["step"] = 0
    outer.append(event)
    validation.append({"step": 0, **validate(
        codec, tail, report, allocation, args.image_batch, args.rate_lambda),
                       "allocation": allocation.tolist()})
    train_started = time.time()

    for step in range(1, total + 1):
        try:
            batch_rows, x = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch_rows, x = next(iterator)
        x = x.float().to(device, non_blocking=True)
        n = int(batch_rows.shape[0])
        torch.index_select(teacher_host, 0, batch_rows, out=staging[:n])
        teacher = staging[:n].to(device, non_blocking=True)
        with torch.no_grad():
            y, mu, std = batch_normalize_gpu(x, mode=config.NORM_MODE)
        epoch = min((step - 1) // per_epoch + 1, schedule_epochs)
        joint.set_lr(optimizer, epoch - 1, schedule_epochs, args.lr)
        tau = joint.codeword_tau(
            epoch, schedule_epochs, args.tau_start, args.tau_end)
        optimizer.zero_grad(set_to_none=True)
        distortion, _, rate = qhard.distortion_sparse(
            codec, tail, y, mu, std, teacher, allocation,
            codeword_temperature=tau, return_rate=args.rate_lambda > 0)
        objective = (rate * y.shape[1] + args.rate_lambda * distortion
                     if args.rate_lambda > 0 else distortion)
        loss = objective.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
        with torch.no_grad():
            for mode, quantizer in enumerate(codec.pq.quantizers):
                if quantizer.codebooks.grad is not None:
                    active = quantizer.codebooks.grad.square().sum((1, 2)).gt(0)
                    gradient_coverage[:, mode].add_(active)
        optimizer.step()
        trace.append([step, float(distortion.mean().detach()),
                      float(rate.mean().detach()), float(loss.detach()), tau])

        if args.outer_every > 0 and step < total and step % args.outer_every == 0:
            allocation, event = propose(
                codec, tail, cal, select, allocation, bits, anchor.rate,
                args.image_batch, args.topk, args.rate_lambda)
            event["step"] = step
            outer.append(event)
        if step % args.val_every == 0 or step == total:
            record = validate(
                codec, tail, report, allocation,
                args.image_batch, args.rate_lambda)
            record.update(step=step, allocation=allocation.tolist())
            validation.append(record)
            print(f"[{anchor.name}] step={step}/{total} "
                  f"D={record['distortion']:.1f} J={record['objective']:.1f} "
                  f"R={record['rate_bpt']:.3f} tau={tau:.4f}", flush=True)

    final_report = validate(
        codec, tail, report, allocation, args.image_batch, args.rate_lambda)
    hard_parity = qhard.selfcheck(
        codec, tail, report, allocation, image_batch=args.image_batch)
    if hard_parity["rel_gap"] > hard_parity["tolerance"]:
        raise SystemExit("INVALID_EXPERIMENT: hard parity")
    final_orth = frozen.orthogonality_error(codec)
    if final_orth - initial_orth > config.ORTH_TOL:
        raise SystemExit("INVALID_EXPERIMENT: orthogonality drift")
    save_codec_v1(codec, out / "codec.pt")
    np.save(out / "allocation.npy", allocation)
    (out / "training_complete.json").write_text(json.dumps({
        "allocation": allocation.tolist(), "final_report": final_report,
        "hard_parity": hard_parity}, indent=2))
    selected = torch.zeros(config.GROUPS, len(bits), dtype=torch.bool)
    selected[torch.arange(config.GROUPS), torch.from_numpy(allocation)] = True
    drift = torch.stack([
        (q.codebooks.detach().cpu() - old.cpu()).flatten(1).norm(dim=1)
        for q, old in zip(codec.pq.quantizers, initial_books)], dim=1)
    usage = ([] if args.skip_centroid else centroid_usage(
        codec, config.TRAIN_FEATURES, config.N_TRAIN, device,
        config.NORM_MODE, 16))
    payload = {
        "plan": "v22_deterministic_dp_block_coordinate",
        "block": args.block, "anchor": anchor.name, "rate": anchor.rate,
        "source_codec": args.source_codec,
        "source_allocation": args.source_allocation,
        "steps": total, "epochs_equivalent": total / per_epoch,
        "batch": args.batch, "lr": args.lr,
        "rate_lambda": args.rate_lambda,
        "tau_start": args.tau_start, "tau_end": args.tau_end,
        "outer_every": args.outer_every, "topk": args.topk,
        "allocation": allocation.tolist(), "outer_events": outer,
        "initial_report": initial_report, "final_report": final_report,
        "hard_parity": hard_parity,
        "validation": validation, "trace": trace,
        "orthogonality_initial": initial_orth,
        "orthogonality_final": final_orth,
        "rotation_relative_drift": float(
            (codec.transform.get_rotation().detach() - initial_rotation).norm()
            / initial_rotation.norm()),
        "active_gradient_coverage_min": int(
            gradient_coverage.cpu()[selected].min()),
        "inactive_gradient_coverage_max": int(
            gradient_coverage.cpu()[~selected].max()),
        "active_codebook_drift_min": float(drift[selected].min()),
        "centroid_usage_train5k": usage,
        "training_seconds": time.time() - train_started,
        "training_images_per_second": total * args.batch /
        max(time.time() - train_started, 1e-9),
        "peak_memory_gb": (torch.cuda.max_memory_allocated(device) / 2**30
                           if device.type == "cuda" else 0.0),
        "seconds": time.time() - started}
    (out / "train.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in (
        "plan", "allocation", "initial_report", "final_report",
        "active_gradient_coverage_min", "active_codebook_drift_min",
        "training_seconds")}, indent=2))


if __name__ == "__main__":
    main()
