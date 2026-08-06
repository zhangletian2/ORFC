"""Deterministic allocation DP alternating with ORFC codec updates."""

from __future__ import annotations

import argparse
import copy
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
            topk, rate_lambda, active_slate_size=0):
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
        row = np.asarray(row, dtype=np.int64)
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
    count = min(max(1, int(active_slate_size)), len(candidates))
    order = np.argsort(values, kind="stable")[:count]
    slate = candidates[order].copy() if active_slate_size else chosen[None]
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
        "select_objective_gain": float(values[0] - values[winner]),
        "active_slate": slate.tolist(),
        "active_slate_objectives": [float(values[index]) for index in order]}, slate


@torch.no_grad()
def validate(codec, tail, resident, allocation, image_batch, rate_lambda):
    distortion, rates, objective = evaluate(
        codec, tail, resident, allocation[None], image_batch, rate_lambda)
    return {"distortion": float(distortion.mean()),
            "rate_bpt": float(rates.mean()),
            "objective": float(objective.mean())}


@torch.no_grad()
def audit_event(codec, tail, resident, event, image_batch, rate_lambda):
    modes = np.asarray([event["base"], event["proposed"]], dtype=np.int64)
    distortion, rates, objective = evaluate(
        codec, tail, resident, modes, image_batch, rate_lambda)
    event.update(
        report_base_distortion=float(distortion[0].mean()),
        report_proposed_distortion=float(distortion[1].mean()),
        report_base_rate_bpt=float(rates[0].mean()),
        report_proposed_rate_bpt=float(rates[1].mean()),
        report_base_objective=float(objective[0].mean()),
        report_proposed_objective=float(objective[1].mean()),
        report_objective_gain=float(
            objective[0].mean() - objective[1].mean()))


def lookahead(codec, tail, cal, select, report, base, bits, rate, image_batch,
              topk, size, steps, inner_batch, lr, tau, rate_lambda,
              freeze_u=False, fit=None, u_optimizer_state=None):
    """Compare equally adapted exact-budget branches and keep the winner."""
    _, proposal, slate = propose(
        codec, tail, cal, select, base, bits, rate, image_batch, topk,
        rate_lambda, size)
    rows = [base.copy()]
    rows.extend(row.copy() for row in slate
                if not np.array_equal(row, base))
    rows = np.asarray(rows[:size], dtype=np.int64)
    records, winner, winner_codec, winner_u_state = [], None, None, None
    for index, row in enumerate(rows):
        branch = copy.deepcopy(codec)
        for parameter in branch.transform.parameters():
            parameter.requires_grad_(not freeze_u)
        for quantizer in branch.pq.quantizers:
            for parameter in quantizer.parameters():
                parameter.requires_grad_(True)
        q_parameters = [parameter for quantizer in branch.pq.quantizers
                        for parameter in quantizer.parameters()]
        if freeze_u:
            optimizers = [torch.optim.Adam(q_parameters, lr=float(lr))]
            branch_u_optimizer = None
        elif u_optimizer_state is not None:
            branch_u_optimizer = torch.optim.Adam(
                branch.transform.parameters(), lr=float(lr))
            branch_u_optimizer.load_state_dict(
                copy.deepcopy(u_optimizer_state))
            optimizers = [branch_u_optimizer,
                          torch.optim.Adam(q_parameters, lr=float(lr))]
        else:
            branch_u_optimizer = None
            optimizers = [torch.optim.Adam(
                branch.parameters(), lr=float(lr))]
        rotation_before = branch.transform.get_rotation().detach().clone()
        books_before = [q.codebooks.detach().clone()
                        for q in branch.pq.quantizers]
        adapt = cal if fit is None else fit
        for inner in range(steps):
            first = ((inner * inner_batch)
                     % max(1, adapt.count - inner_batch + 1))
            batch = adapt.slice(
                first, min(first + inner_batch, adapt.count))
            distortion, _, rates = qhard.distortion_sparse(
                branch, tail, *batch, row, codeword_temperature=tau,
                return_rate=True)
            objective = rates * adapt.tokens + rate_lambda * distortion
            for branch_optimizer in optimizers:
                branch_optimizer.zero_grad(set_to_none=True)
            objective.mean().backward()
            torch.nn.utils.clip_grad_norm_(branch.parameters(), 1.0)
            for branch_optimizer in optimizers:
                branch_optimizer.step()
        selected = validate(
            branch, tail, select, row, image_batch, rate_lambda)
        reported = validate(
            branch, tail, report, row, image_batch, rate_lambda)
        branch_device = next(branch.parameters()).device
        selected_mask = torch.zeros(
            len(base), len(bits), dtype=torch.bool, device=branch_device)
        selected_mask[
            torch.arange(len(base), device=branch_device),
            torch.as_tensor(row, device=branch_device)] = True
        drift = torch.stack([
            (q.codebooks.detach() - old).flatten(1).norm(dim=1)
            for q, old in zip(branch.pq.quantizers, books_before)], dim=1)
        records.append({"allocation": row.tolist(), "select": selected,
                        "report": reported,
                        "u_drift": float(
                            (branch.transform.get_rotation().detach()
                             - rotation_before).abs().max()),
                        "selected_book_drift_min": float(
                            drift[selected_mask].min()),
                        "unselected_book_drift_max": float(
                            drift[~selected_mask].max())})
        if winner is None or selected["objective"] < records[winner]["select"]["objective"]:
            del winner_codec
            winner, winner_codec = index, branch
            winner_u_state = (copy.deepcopy(
                branch_u_optimizer.state_dict())
                if branch_u_optimizer is not None else None)
        else:
            del branch
    del optimizers
    codec.load_state_dict(winner_codec.state_dict())
    del winner_codec
    chosen = rows[winner].copy()
    base_select, chosen_select = records[0]["select"], records[winner]["select"]
    base_report, chosen_report = records[0]["report"], records[winner]["report"]
    return chosen, {
        "base": base.tolist(), "proposed": chosen.tolist(),
        "accepted": bool(winner != 0), "topk": int(topk),
        "candidate_count": int(len(rows)), "lookahead_steps": int(steps),
        "lookahead_freeze_u": bool(freeze_u),
        "branches": records,
        "select_base_distortion": base_select["distortion"],
        "select_winner_distortion": chosen_select["distortion"],
        "select_base_rate_bpt": base_select["rate_bpt"],
        "select_winner_rate_bpt": chosen_select["rate_bpt"],
        "select_base_objective": base_select["objective"],
        "select_winner_objective": chosen_select["objective"],
        "select_objective_gain": (base_select["objective"]
                                  - chosen_select["objective"]),
        "report_base_distortion": base_report["distortion"],
        "report_proposed_distortion": chosen_report["distortion"],
        "report_base_rate_bpt": base_report["rate_bpt"],
        "report_proposed_rate_bpt": chosen_report["rate_bpt"],
        "report_base_objective": base_report["objective"],
        "report_proposed_objective": chosen_report["objective"],
        "report_objective_gain": (base_report["objective"]
                                  - chosen_report["objective"]),
        "u_optimizer_state_reused": bool(u_optimizer_state is not None)}, \
        winner_u_state


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
    parser.add_argument("--candidate-adapt", action="store_true")
    parser.add_argument("--active-slate-size", type=int, default=0)
    parser.add_argument("--active-slate-temperature", choices=("sem", "range"),
                        default="sem")
    parser.add_argument("--lookahead-steps", type=int, default=0)
    parser.add_argument("--lookahead-size", type=int, default=3)
    parser.add_argument("--lookahead-batch", type=int, default=16)
    parser.add_argument("--lookahead-fit-images", type=int, default=0)
    parser.add_argument("--lookahead-freeze-u", action="store_true")
    parser.add_argument("--main-u-only", action="store_true")
    parser.add_argument("--defer-initial-outer", action="store_true")
    parser.add_argument("--skip-centroid", action="store_true")
    args = parser.parse_args(argv)
    if args.rate_lambda <= 0:
        raise SystemExit("V22 currently requires the V21 ECVQ objective")
    if sum(bool(value) for value in (
            args.candidate_adapt, args.active_slate_size,
            args.lookahead_steps)) > 1:
        raise SystemExit("candidate adaptation, active slate, and lookahead are exclusive")

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
    if args.main_u_only:
        for quantizer in codec.pq.quantizers:
            for parameter in quantizer.parameters():
                parameter.requires_grad_(False)
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

    optimizer = torch.optim.Adam(
        codec.transform.parameters() if args.main_u_only else codec.parameters(),
        lr=float(args.lr))
    tail = tail_mod.build_tail(config.LAYER, device)
    rows = config.load_split("train_val")[2]
    cal = split_resident(config, rows, 0, args.cal_images, device)
    select = split_resident(
        config, rows, args.cal_images, args.select_images, device)
    report = split_resident(
        config, rows, args.cal_images + args.select_images,
        args.report_images, device)

    train_paths = config.load_split("train_fit")
    lookahead_fit = (engine.ResidentSet(
        train_paths[0], train_paths[1],
        np.asarray(train_paths[2][:args.lookahead_fit_images]), device)
        if args.lookahead_fit_images else None)
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
    candidate_coverage = torch.zeros_like(gradient_coverage)
    candidate_trace, candidate_u_grad_delta = [], 0.0
    slate_exposure = torch.zeros_like(gradient_coverage)
    slate_trace = []

    event = None
    if args.lookahead_steps and not args.defer_initial_outer:
        allocation, event, winner_u_state = lookahead(
            codec, tail, cal, select, report, allocation, bits, anchor.rate,
            args.image_batch, args.topk, args.lookahead_size,
            args.lookahead_steps, args.lookahead_batch, args.lr,
            args.tau_start, args.rate_lambda, args.lookahead_freeze_u,
            lookahead_fit, optimizer.state_dict()
            if args.main_u_only and not args.lookahead_freeze_u else None)
        active_slate = allocation[None]
        if winner_u_state is not None:
            optimizer = torch.optim.Adam(
                codec.transform.parameters(), lr=float(args.lr))
            optimizer.load_state_dict(winner_u_state)
        elif not (args.lookahead_freeze_u and args.main_u_only):
            optimizer = torch.optim.Adam(
                codec.transform.parameters() if args.main_u_only
                else codec.parameters(), lr=float(args.lr))
    elif not args.lookahead_steps and not args.defer_initial_outer:
        allocation, event, active_slate = propose(
            codec, tail, cal, select, allocation, bits, anchor.rate,
            args.image_batch, args.topk, args.rate_lambda,
            args.active_slate_size)
    else:
        active_slate = allocation[None]
    if event is not None:
        event["step"] = 0
        if not args.lookahead_steps:
            audit_event(
                codec, tail, report, event,
                args.image_batch, args.rate_lambda)
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
        if args.active_slate_size:
            distortions, _, rates = qhard.distortions(
                codec, tail, y, mu, std, teacher, active_slate,
                codeword_temperature=tau, return_rate=True)
            objectives = rates * y.shape[1] + args.rate_lambda * distortions
            means = objectives.mean(1)
            best = int(means.detach().argmin())
            differences = objectives - objectives[best:best + 1]
            sem = differences.std(1) / math.sqrt(y.shape[0])
            mask = torch.arange(len(means), device=device) != best
            if args.active_slate_temperature == "range" and len(means) > 1:
                softmin_tau = ((means.max() - means.min()).detach()
                               / math.log(len(means))).clamp_min(1.0)
            else:
                softmin_tau = (sem[mask].median().detach().clamp_min(1.0)
                               if mask.any() else means.new_tensor(1.0))
            loss = -softmin_tau * torch.logsumexp(
                -means / softmin_tau, dim=0)
            weights = torch.softmax(-means.detach() / softmin_tau, dim=0)
            distortion = (weights[:, None] * distortions).sum(0)
            rate = (weights[:, None] * rates).sum(0)
            objective = (weights[:, None] * objectives).sum(0)
            with torch.no_grad():
                for row in active_slate:
                    groups = torch.arange(config.GROUPS, device=device)
                    modes = torch.as_tensor(row, device=device)
                    slate_exposure.index_put_(
                        (groups, modes), torch.ones_like(groups), accumulate=True)
            slate_trace.append([step, softmin_tau.item(), best,
                                weights.cpu().tolist(), means.detach().cpu().tolist()])
        else:
            distortion, _, rate = qhard.distortion_sparse(
                codec, tail, y, mu, std, teacher, allocation,
                codeword_temperature=tau, return_rate=True)
            objective = rate * y.shape[1] + args.rate_lambda * distortion
            loss = objective.mean()
        loss.backward()
        if args.candidate_adapt:
            pairs = [(group, mode) for group in range(config.GROUPS)
                     for mode in range(len(bits))
                     if mode != int(allocation[group])]
            group, mode = pairs[(step - 1) % len(pairs)]
            candidate = allocation.copy()
            candidate[group] = mode
            before = ([parameter.grad.detach().clone()
                       for parameter in codec.transform.parameters()
                       if parameter.grad is not None] if step == 1 else [])
            candidate_distortion, _, candidate_rate = qhard.distortion_sparse(
                codec, tail, y, mu, std, teacher, candidate,
                codeword_temperature=tau,
                rotation=codec.transform.get_rotation().detach(),
                return_rate=True)
            candidate_objective = (
                candidate_rate * y.shape[1]
                + args.rate_lambda * candidate_distortion).mean()
            quantizer = codec.pq.quantizers[mode]
            parameters = [quantizer.codebooks, quantizer.log_prior]
            gradients = torch.autograd.grad(
                candidate_objective, parameters, allow_unused=True)
            for parameter, gradient in zip(parameters, gradients):
                if gradient is None:
                    continue
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                parameter.grad[group].add_(gradient[group])
            if gradients[0] is not None and gradients[0][group].square().sum() > 0:
                candidate_coverage[group, mode].add_(1)
            if step == 1:
                after = [parameter.grad for parameter in codec.transform.parameters()
                         if parameter.grad is not None]
                candidate_u_grad_delta = max(
                    (float((new - old).abs().max())
                     for old, new in zip(before, after)), default=0.0)
                if candidate_u_grad_delta != 0:
                    raise SystemExit("INVALID_EXPERIMENT: candidate gradient entered U")
            candidate_trace.append([
                step, group, mode,
                float(candidate_distortion.mean().detach()),
                float(candidate_rate.mean().detach()),
                float(candidate_objective.detach())])
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
            if args.lookahead_steps:
                allocation, event, winner_u_state = lookahead(
                    codec, tail, cal, select, report, allocation, bits,
                    anchor.rate, args.image_batch, args.topk,
                    args.lookahead_size, args.lookahead_steps,
                    args.lookahead_batch, args.lr, tau, args.rate_lambda,
                    args.lookahead_freeze_u, lookahead_fit,
                    optimizer.state_dict()
                    if args.main_u_only and not args.lookahead_freeze_u
                    else None)
                active_slate = allocation[None]
                if winner_u_state is not None:
                    optimizer = torch.optim.Adam(
                        codec.transform.parameters(), lr=float(args.lr))
                    optimizer.load_state_dict(winner_u_state)
                elif not (args.lookahead_freeze_u and args.main_u_only):
                    optimizer = torch.optim.Adam(
                        codec.transform.parameters() if args.main_u_only
                        else codec.parameters(), lr=float(args.lr))
            else:
                allocation, event, active_slate = propose(
                    codec, tail, cal, select, allocation, bits, anchor.rate,
                    args.image_batch, args.topk, args.rate_lambda,
                    args.active_slate_size)
            event["step"] = step
            if not args.lookahead_steps:
                audit_event(
                    codec, tail, report, event,
                    args.image_batch, args.rate_lambda)
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
        "plan": ("v26_allocation_conditioned_lookahead" if args.lookahead_steps
                 else "v24_full_information_active_slate" if args.active_slate_size
                 else "v23_candidate_adapted_dp" if args.candidate_adapt else
                 "v22_deterministic_dp_block_coordinate"),
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
        "candidate_adapt": bool(args.candidate_adapt),
        "candidate_gradient_coverage_min": int(
            candidate_coverage.cpu()[~selected].min()
            if args.candidate_adapt else 0),
        "candidate_u_grad_delta": candidate_u_grad_delta,
        "candidate_trace": candidate_trace,
        "active_slate_size": int(args.active_slate_size),
        "active_slate_temperature": args.active_slate_temperature,
        "lookahead_steps": int(args.lookahead_steps),
        "lookahead_size": int(args.lookahead_size),
        "lookahead_batch": int(args.lookahead_batch),
        "lookahead_fit_images": int(args.lookahead_fit_images),
        "lookahead_freeze_u": bool(args.lookahead_freeze_u),
        "main_u_only": bool(args.main_u_only),
        "defer_initial_outer": bool(args.defer_initial_outer),
        "active_slate_final": active_slate.tolist(),
        "active_slate_trace": slate_trace,
        "active_slate_exposure_min": int(
            slate_exposure[slate_exposure > 0].min()
            if args.active_slate_size else 0),
        "active_slate_gradient_min": int(
            gradient_coverage[slate_exposure > 0].min()
            if args.active_slate_size else 0),
        "active_codebook_drift_min": float(drift[selected].min()),
        "inactive_codebook_drift_min": float(drift[~selected].min()),
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
