#!/usr/bin/env python3
"""Repair an operational RD menu and test candidate-bound remainder training."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import lsq_linear

from codec_v1 import load_codec_v1, save_codec_v1
from fixed_rate_remainder import allocation_rates
from opq import batch_inv_normalize_gpu, batch_normalize_gpu
from p1_fixed_rate import build_tail, make_allocations


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(path)


def _design(rates, dimension):
    return np.exp2(-2.0 * np.asarray(rates) / float(dimension))


def _positive_fit(A, y, c0, ridge=1e-3):
    """Fit ``intercept + A @ c`` with non-negative ``c``."""
    n, groups = A.shape
    X = np.c_[np.ones(n), A] / np.sqrt(n)
    target = np.asarray(y, dtype=np.float64) / np.sqrt(n)
    if ridge > 0:
        reg = np.zeros((groups, groups + 1))
        reg[:, 1:] = np.sqrt(ridge) * np.eye(groups)
        X = np.r_[X, reg]
        target = np.r_[target, np.sqrt(ridge) * c0]
    fit = lsq_linear(
        X, target, bounds=(np.r_[-np.inf, np.zeros(groups)], np.inf))
    if not fit.success:
        raise RuntimeError(f"coefficient fit failed: {fit.message}")
    return float(fit.x[0]), fit.x[1:]


def _sample(features, teachers, count, rng, device, norm_mode, ids=None):
    if ids is None:
        ids = rng.choice(
            len(features), min(count, len(features)), replace=False)
    x = torch.from_numpy(np.array(features[ids], copy=True)).float().to(device)
    teacher = torch.from_numpy(
        np.array(teachers[ids], copy=True)).float().to(device)
    y, mu, std = batch_normalize_gpu(x, mode=norm_mode)
    return y, mu, std, teacher


def _distortions(
    codec, tail, batch, allocations, chunk, temperature, ste,
    reduce_images=True,
):
    """Build one differentiable mode bank and evaluate many allocations."""
    y, mu, std, teacher = batch
    b, tokens, _ = y.shape
    codec.pq.temperature = temperature
    rotation = codec.transform.get_rotation()
    z = y.reshape(b * tokens, -1) @ rotation
    sub = z.reshape(-1, codec.pq.G, codec.pq.d).permute(1, 0, 2)
    banks = []
    for quantizer in codec.pq.quantizers:
        cost = torch.cdist(sub, quantizer.codebooks).square()
        weights = torch.softmax(-cost / temperature, -1)
        if ste:
            labels = cost.argmin(-1, keepdim=True)
            hard = torch.zeros_like(weights).scatter_(-1, labels, 1.0)
            weights = hard - weights.detach() + weights
        banks.append(torch.einsum(
            "gnk,gkd->gnd", weights, quantizer.codebooks
        ).permute(1, 0, 2).reshape(
            b, tokens, codec.pq.G, codec.pq.d))
    bank = torch.stack(banks).permute(3, 0, 1, 2, 4)
    groups = torch.arange(codec.pq.G, device=y.device)[None]
    values = []
    for start in range(0, len(allocations), chunk):
        modes = torch.as_tensor(
            allocations[start:start + chunk], device=y.device)
        count = len(modes)
        selected = bank[groups, modes].permute(
            0, 2, 3, 1, 4).reshape(count * b, tokens, -1)
        xhat = batch_inv_normalize_gpu(
            selected @ rotation.t(),
            mu.unsqueeze(0).expand(count, *mu.shape).reshape(
                count * b, *mu.shape[1:]),
            std.unsqueeze(0).expand(count, *std.shape).reshape(
                count * b, *std.shape[1:]))
        output = tail(xhat)
        target = teacher.unsqueeze(0).expand(
            count, -1, *teacher.shape[1:]).reshape(
                count * b, *teacher.shape[1:])
        value = (output - target).square().reshape(count, b, -1).sum(-1)
        values.append(value.mean(1) if reduce_images else value)
    return torch.cat(values)


@torch.no_grad()
def _hard_eval(
    codec, tail, features, teachers, allocations, args,
    return_per_image=False,
):
    rows = []
    first = args.hard_image_offset
    count = min(args.hard_images, len(features) - first)
    if count < 1:
        raise ValueError("hard evaluation image slice is empty")
    for start in range(first, first + count, args.hard_batch_size):
        stop = min(start + args.hard_batch_size, first + count)
        x = torch.from_numpy(
            np.array(features[start:stop], copy=True)).float().to(args.device)
        target = torch.from_numpy(
            np.array(teachers[start:stop], copy=True)).float().to(args.device)
        y, mu, std = batch_normalize_gpu(x, mode=args.norm_mode)
        value = _distortions(
            codec, tail, (y, mu, std, target), allocations,
            args.allocation_chunk, args.pq_temperature, True,
            reduce_images=False)
        rows.append(value.cpu().numpy())
    matrix = np.concatenate(rows, axis=1)
    return matrix if return_per_image else matrix.mean(1)


def _uniform_allocations(codec):
    return np.repeat(
        np.arange(codec.pq.num_modes)[:, None], codec.pq.G, axis=1)


def _curve_report(bits, distortion, tolerance):
    distortion = np.asarray(distortion)
    relative = np.diff(distortion) / np.maximum(distortion[:-1], 1e-12)
    return {
        "bits": list(map(int, bits)),
        "distortion": distortion.tolist(),
        "adjacent_relative_change": relative.tolist(),
        "violations": int(np.count_nonzero(relative > tolerance)),
        "max_relative_violation": float(max(relative.max(initial=0), 0)),
        "monotonic": bool(np.all(relative <= tolerance)),
    }


@torch.no_grad()
def _seed_from_anchor(codec, anchor, split_scale=0.001):
    """Build a nested menu around the anchor using data-driven mode centres."""
    base = codec.pq.quantizers[anchor].codebooks
    groups, size, dimension = base.shape
    order = torch.empty(groups, size, dtype=torch.long, device=base.device)
    for group in range(groups):
        points = base[group]
        first = (points - points.mean(0)).square().sum(1).argmin()
        chosen = torch.zeros(size, dtype=torch.bool, device=base.device)
        distance = (points - points[first]).square().sum(1)
        for index in range(size):
            current = first if index == 0 else distance.argmax()
            order[group, index] = current
            chosen[current] = True
            distance = torch.minimum(
                distance, (points - points[current]).square().sum(1))
            distance[chosen] = -1
    for mode, quantizer in enumerate(codec.pq.quantizers[:anchor]):
        target = quantizer.codebooks
        indices = order[:, :target.shape[1], None].expand(
            -1, -1, dimension)
        target.copy_(torch.gather(base, 1, indices))
    previous = base.detach().clone()
    for quantizer in codec.pq.quantizers[anchor + 1:]:
        target, candidates = quantizer.codebooks, quantizer.codebooks.clone()
        for group in range(groups):
            centres, pool = previous[group], candidates[group]
            used = torch.zeros(
                len(pool), dtype=torch.bool, device=pool.device)
            additions = []
            distance = torch.cdist(pool, centres).square().amin(1)
            for _ in range(target.shape[1] - centres.shape[0]):
                index = distance.argmax()
                candidate = pool[index]
                nearest = torch.cdist(
                    candidate[None], centres).argmin()
                point = centres[nearest] + split_scale * (
                    candidate - centres[nearest])
                additions.append(point)
                used[index] = True
                distance = torch.minimum(
                    distance, (pool - point).square().sum(1))
                distance[used] = -1
            target[group].copy_(torch.cat([centres, torch.stack(additions)]))
        previous = target.detach().clone()


def _base(args):
    device = torch.device(args.device)
    codec = load_codec_v1(args.codec, device).train()
    tail = build_tail(args.layer, device)
    train_x = np.load(args.features, mmap_mode="r")
    train_y = np.load(args.teachers, mmap_mode="r")
    val_x = np.load(args.hard_features, mmap_mode="r")
    val_y = np.load(args.hard_teachers, mmap_mode="r")
    return device, codec, tail, train_x, train_y, val_x, val_y


def command_repair(args):
    device, codec, tail, train_x, train_y, val_x, val_y = _base(args)
    bits = np.log2(codec.pq.mode_sizes).astype(int)
    if args.anchor_bit not in bits:
        raise ValueError("anchor bit is absent from the mode menu")
    anchor = int(np.flatnonzero(bits == args.anchor_bit)[0])
    allocations = _uniform_allocations(codec)
    original = _hard_eval(
        codec, tail, val_x, val_y, allocations, args)
    anchor_codebook = codec.pq.quantizers[anchor].codebooks.detach().clone()
    _seed_from_anchor(codec, anchor, args.split_scale)
    seeded = _hard_eval(
        codec, tail, val_x, val_y, allocations, args)
    scale = max(float(seeded.mean()), 1.0)
    codec.requires_grad_(False)
    params = []
    for mode, quantizer in enumerate(codec.pq.quantizers):
        if mode != anchor:
            quantizer.codebooks.requires_grad_(True)
            params.append(quantizer.codebooks)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    rng = np.random.default_rng(args.seed)
    seeded_report = _curve_report(
        bits, seeded, args.monotonic_tolerance)
    best_key = (
        seeded_report["violations"],
        int((seeded[anchor] - seeded[-1]) / max(
            seeded[anchor], 1e-12) < args.minimum_high_rate_gain),
        float(np.mean(np.delete(seeded, anchor))),
        seeded_report["max_relative_violation"])
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in codec.state_dict().items()}
    history = []
    for step in range(args.steps):
        batch = _sample(
            train_x, train_y, args.images, rng, device, args.norm_mode)
        optimizer.zero_grad(set_to_none=True)
        D = _distortions(
            codec, tail, batch, allocations, args.allocation_chunk,
            args.pq_temperature, True)
        violation = torch.relu(
            D[1:] - (1.0 - args.margin) * D[:-1]).mean() / scale
        mean = D.mean() / scale
        loss = mean + args.monotonic_weight * violation
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        row = {
            "step": step + 1, "loss": float(loss),
            "mean": float(mean), "monotonic": float(violation),
            "grad_norm": float(grad),
        }
        if (step + 1) % args.eval_steps == 0 or step + 1 == args.steps:
            curve = _hard_eval(
                codec, tail, val_x, val_y, allocations, args)
            report = _curve_report(bits, curve, args.monotonic_tolerance)
            key = (
                report["violations"],
                int((curve[anchor] - curve[-1]) / max(
                    curve[anchor], 1e-12) < args.minimum_high_rate_gain),
                float(np.mean(np.delete(curve, anchor))),
                report["max_relative_violation"])
            row["validation"] = report
            if best_key is None or key < best_key:
                best_key = key
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in codec.state_dict().items()}
        history.append(row)
    codec.load_state_dict(best_state)
    after = _hard_eval(codec, tail, val_x, val_y, allocations, args)
    anchor_exact = torch.equal(
        codec.pq.quantizers[anchor].codebooks, anchor_codebook)
    if not anchor_exact:
        raise RuntimeError("frozen anchor codebook changed")
    save_codec_v1(codec, args.checkpoint)
    report = {
        "before": _curve_report(bits, original, args.monotonic_tolerance),
        "anchor_seeded": seeded_report,
        "after": _curve_report(bits, after, args.monotonic_tolerance),
        "anchor_bit": args.anchor_bit, "anchor_codebook_exact": anchor_exact,
        "steps": args.steps, "checkpoint": args.checkpoint,
        "history": history,
    }
    high_gain = (
        after[anchor] - after[-1]) / max(after[anchor], 1e-12)
    report["high_rate_gain"] = float(high_gain)
    _write(args.output, report)
    if (
        not report["after"]["monotonic"]
        or high_gain < args.minimum_high_rate_gain
    ):
        raise RuntimeError("repaired menu did not pass the monotonic RD gate")


def _select_state(source, distortion, calibration, args):
    A = _design(source["rates"], int(calibration["rate_dimension"]))
    intercept, c = _positive_fit(
        A, distortion, calibration["c_g"], args.ridge)
    phi = intercept + A @ c
    remainder = distortion - phi
    tolerance = max(
        args.tie_atol, args.tie_rtol * max(abs(float(phi.min())), 1.0))
    minimizers = np.flatnonzero(phi <= phi.min() + tolerance)
    outside = np.setdiff1d(
        np.arange(len(phi)), minimizers, assume_unique=True)
    gap = (
        float(phi[outside].min() - phi.min()) if len(outside)
        else float("inf"))
    target = int(minimizers[np.argmin(distortion[minimizers])])
    order = np.argsort(remainder)
    edge = max(2, args.allocations // 4)
    selected = (
        set(order[:edge]) | set(order[-edge:])
        | set(map(int, minimizers)) | {target})
    distortion_order = outside[np.argsort(distortion[outside])]
    selected.update(map(int, distortion_order[:edge]))
    selected.update(map(int, distortion_order[-edge:]))
    reference = None
    bits = calibration["mode_bits"][source["allocations"]]
    hits = np.flatnonzero(np.all(bits == args.reference_bit, axis=1))
    if len(hits):
        reference = int(hits[0])
        selected.add(reference)
    rng = np.random.default_rng(args.seed)
    for index in rng.permutation(len(phi)):
        selected.add(int(index))
        if len(selected) >= min(args.allocations, len(phi)):
            break
    selected = np.asarray(sorted(selected), dtype=int)
    local = {index: position for position, index in enumerate(selected)}
    return {
        "selected": selected, "allocations": source["allocations"][selected],
        "A": A[selected], "full_A": A, "intercept": intercept, "c": c,
        "target": target, "minimizers": minimizers,
        "minimizer_local": np.asarray(
            [local[int(index)] for index in minimizers]),
        "gap": gap,
        "reference": reference,
        "reference_local": local.get(reference),
        "omega_scale": max(float(np.ptp(remainder)), 1.0),
        "distortion_scale": max(float(distortion[target]), 1.0),
    }


def _allocation_source(allocations, calibration):
    allocations = np.asarray(allocations, dtype=np.int64)
    return {
        "allocations": allocations,
        "rates": allocation_rates(allocations, calibration["cost_table"]),
    }


def _subset_distortion(source, distortion, subset):
    lookup = {
        tuple(row.tolist()): value
        for row, value in zip(source["allocations"], distortion)}
    return np.asarray([
        lookup[tuple(row.tolist())] for row in subset["allocations"]])


def _dynamic_source(audit, calibration, c, args, refresh):
    if not args.dynamic_allocations:
        return audit
    budget = int(round(float(audit["rates"][0].sum())))
    generated, _, _ = make_allocations(
        c, tuple(map(int, calibration["mode_bits"])), budget,
        args.dynamic_single, args.dynamic_random, args.seed + refresh,
        int(calibration["rate_dimension"]))
    pool = np.unique(
        np.concatenate([audit["allocations"], generated]), axis=0)
    return _allocation_source(pool, calibration)


def _set_scales(state, scales):
    state["omega_scale"], state["distortion_scale"] = scales
    return state


def _objective_terms(distortion, state, args, remainder_weight):
    D = np.asarray(distortion, dtype=np.float64)
    complete = len(D) == len(state["full_A"])
    A = state["full_A"] if complete else state["A"]
    minimizers = state["minimizers"] if complete else state["minimizer_local"]
    reference_index = (
        state["reference"] if complete else state["reference_local"])
    e = D - state["intercept"] - A @ state["c"]
    scale = args.lse_temperature * state["omega_scale"]
    high = scale * np.log(np.exp((e - e.max()) / scale).sum()) + e.max()
    low_e = -e
    low = (
        scale * np.log(np.exp((low_e - low_e.max()) / scale).sum())
        + low_e.max())
    omega_raw = high + low - 2 * scale * math.log(len(e))
    recovery = max(
        omega_raw - args.recovery_fraction * state["gap"], 0.0
    ) / state["omega_scale"]
    candidate = D[minimizers].mean() / state["distortion_scale"]
    topk = np.sort(D)[-min(args.topk, len(D)):].mean()
    topk /= state["distortion_scale"]
    reference = (
        D[reference_index] / state["distortion_scale"]
        if reference_index is not None else 0.0)
    base = (
        args.candidate_weight * candidate + args.topk_weight * topk
        + args.reference_weight * reference)
    return {
        "score": float(base + remainder_weight * recovery),
        "base": float(base), "recovery": float(recovery),
        "omega": float(np.ptp(e)), "gap": float(state["gap"]),
        "recovery_ratio": float(np.ptp(e) / max(state["gap"], 1e-12)),
        "candidate": float(candidate), "topk": float(topk),
        "reference": float(reference),
    }


def _loss(codec, tail, batch, state, args, remainder_weight):
    D = _distortions(
        codec, tail, batch, state["allocations"], args.allocation_chunk,
        args.pq_temperature, True)
    A = torch.as_tensor(state["A"], device=D.device)
    c = torch.as_tensor(state["c"], device=D.device)
    e = D.double() - state["intercept"] - A.double() @ c
    tau = args.lse_temperature
    scaled_tau = tau * state["omega_scale"]
    omega_raw = (
        scaled_tau * torch.logsumexp(e / scaled_tau, 0)
        + scaled_tau * torch.logsumexp(-e / scaled_tau, 0)
        - 2 * scaled_tau * math.log(len(e)))
    omega = omega_raw / state["omega_scale"]
    recovery = torch.relu(
        omega_raw - args.recovery_fraction * state["gap"]
    ) / state["omega_scale"]
    candidate = D[state["minimizer_local"]].mean() / state["distortion_scale"]
    topk = torch.topk(
        D, min(args.topk, len(D))).values.mean() / state["distortion_scale"]
    reference = (
        D[state["reference_local"]] / state["distortion_scale"]
        if state["reference_local"] is not None else D.new_zeros(()))
    base = (
        args.candidate_weight * candidate + args.topk_weight * topk
        + args.reference_weight * reference)
    return base + remainder_weight * recovery, {
        "base": base, "omega": omega, "recovery": recovery,
        "candidate": candidate,
        "topk": topk, "reference": reference,
    }


def _joint_parameters(codec):
    codec.requires_grad_(False)
    codec.transform.triu_params.requires_grad_(True)
    params = [codec.transform.triu_params]
    for quantizer in codec.pq.quantizers:
        quantizer.codebooks.requires_grad_(True)
        params.append(quantizer.codebooks)
    return params


def _calibrate_remainder(codec, tail, batch, state, args, params):
    if args.remainder_grad_ratio < 0:
        return args.remainder_weight
    _, terms = _loss(codec, tail, batch, state, args, 0.0)
    targets = params if args.remainder_parameters == "joint" else params[:1]
    base = torch.autograd.grad(
        terms["base"], targets, retain_graph=True, allow_unused=True)
    remainder = torch.autograd.grad(
        terms["recovery"], targets, allow_unused=True)
    norm = lambda values: torch.sqrt(sum(
        value.square().sum() for value in values if value is not None))
    base_norm, remainder_norm = norm(base), norm(remainder)
    codec.zero_grad(set_to_none=True)
    return (
        args.remainder_grad_ratio * float(base_norm)
        / max(float(remainder_norm), 1e-12)
        if float(remainder_norm) > 1e-12 else 0.0)


def _backward(loss, terms, params, args, remainder_weight):
    if remainder_weight <= 0 or args.remainder_parameters == "joint":
        loss.backward()
        return
    terms["base"].backward(retain_graph=True)
    gradient = torch.autograd.grad(
        terms["recovery"], params[0], allow_unused=True)[0]
    if gradient is not None:
        if params[0].grad is None:
            params[0].grad = remainder_weight * gradient
        else:
            params[0].grad.add_(gradient, alpha=remainder_weight)


def _report(source, distortion, state, dimension):
    phi = state["intercept"] + _design(source["rates"], dimension) @ state["c"]
    remainder = distortion - phi
    minimizers = np.asarray(state["minimizers"])
    gap = state["gap"]
    return {
        "mean_distortion": float(distortion.mean()),
        "target_index": int(state["target"]),
        "target_bits": source["rates"][state["target"]].tolist(),
        "target_distortion": float(distortion[state["target"]]),
        "minimizer_indices": minimizers.tolist(),
        "minimizer_count": int(len(minimizers)),
        "best_minimizer_index": int(
            minimizers[np.argmin(distortion[minimizers])]),
        "best_minimizer_distortion": float(distortion[minimizers].min()),
        "numeric_best_index": int(distortion.argmin()),
        "numeric_best_distortion": float(distortion.min()),
        "reference_distortion": (
            float(distortion[state["reference"]])
            if state["reference"] is not None else None),
        "omega": float(np.ptp(remainder)),
        "set_external_gap": float(gap),
        "recovery_ratio": float(np.ptp(remainder) / max(gap, 1e-12)),
        "zero_coefficients": int(np.count_nonzero(state["c"] < 1e-8)),
        "candidate_count": int(np.count_nonzero(
            phi - phi.min() <= np.ptp(remainder) + 1e-12)),
    }


def _training_batches(args, count, rng):
    if args.epochs <= 0:
        for step in range(args.steps):
            yield step, None, None
        return
    count = min(args.train_images, count)
    step = 0
    for epoch in range(args.epochs):
        order = rng.permutation(count)
        for start in range(0, count, args.images):
            yield step, epoch, order[start:start + args.images]
            step += 1


def _state_report(state):
    coverage = sum(
        len(np.unique(state["allocations"][:, group]))
        for group in range(state["allocations"].shape[1]))
    return {
        "target": int(state["target"]),
        "minimizers": state["minimizers"].tolist(),
        "gap": float(state["gap"]),
        "selected": state["selected"].tolist(),
        "selected_allocations": state["allocations"].tolist(),
        "selected_count": int(len(state["selected"])),
        "group_mode_coverage": int(coverage),
        "c": state["c"].tolist(),
    }


def command_short(args):
    device, codec, tail, train_x, train_y, val_x, val_y = _base(args)
    saved = np.load(args.allocation_file, allow_pickle=False)
    calibration = np.load(args.calibration, allow_pickle=False)
    audit = {
        "allocations": np.asarray(saved["allocations"]),
        "rates": np.asarray(saved["rates"]),
    }
    bits = np.log2(codec.pq.mode_sizes).astype(int)
    params = _joint_parameters(codec)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    rng = np.random.default_rng(args.seed)
    refresh_x = train_x[:args.refresh_images]
    refresh_y = train_y[:args.refresh_images]
    refresh_args = argparse.Namespace(**vars(args))
    refresh_args.hard_images = args.refresh_images
    refresh_args.hard_image_offset = 0
    training_source = _dynamic_source(
        audit, calibration, calibration["c_g"], args, 0)
    initial_D = _hard_eval(
        codec, tail, refresh_x, refresh_y,
        training_source["allocations"], refresh_args)
    state = _select_state(training_source, initial_D, calibration, args)
    training_scales = (state["omega_scale"], state["distortion_scale"])
    audit_D = _subset_distortion(training_source, initial_D, audit)
    audit_state = _select_state(audit, audit_D, calibration, args)
    audit_scales = (
        audit_state["omega_scale"], audit_state["distortion_scale"])
    state = _set_scales(state, training_scales)
    audit_state = _set_scales(audit_state, audit_scales)
    before_matrix = _hard_eval(
        codec, tail, val_x, val_y, audit["allocations"], args,
        return_per_image=True)
    before_val_D = before_matrix.mean(1)
    before = _report(
        audit, before_val_D, audit_state,
        int(calibration["rate_dimension"]))
    total_steps = (
        args.epochs * math.ceil(
            min(args.train_images, len(train_x)) / args.images)
        if args.epochs > 0 else args.steps)
    steps_per_epoch = (
        math.ceil(min(args.train_images, len(train_x)) / args.images)
        if args.epochs > 0 else 0)
    schedule_length = (
        args.epochs
        if args.schedule_unit == "epoch" and args.epochs > 0
        else total_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(schedule_length, 1), eta_min=args.lr * 0.01)
    args.pq_temperature = args.tau_start
    calibration_batch = _sample(
        train_x, train_y, args.images, np.random.default_rng(args.seed + 1),
        device, args.norm_mode, np.arange(min(args.images, len(train_x))))
    remainder_weight = _calibrate_remainder(
        codec, tail, calibration_batch, state, args, params)
    history, active, refresh = [], [], 0
    initial_terms = _objective_terms(
        before_val_D, audit_state, args, remainder_weight)
    initial_base = initial_terms["base"]
    selection = [{"step": 0, **initial_terms, "eligible": True}]
    best_score, best_step = initial_terms["score"], 0
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in codec.state_dict().items()}
    for step, epoch, ids in _training_batches(args, len(train_x), rng):
        progress = (
            epoch / max(args.epochs - 1, 1)
            if args.schedule_unit == "epoch" and epoch is not None
            else step / max(total_steps - 1, 1))
        temperature = args.tau_start * (
            args.tau_end / args.tau_start) ** progress
        args.pq_temperature = temperature
        batch = _sample(
            train_x, train_y, args.images, rng, device, args.norm_mode, ids)
        optimizer.zero_grad(set_to_none=True)
        loss, terms = _loss(
            codec, tail, batch, state, args, remainder_weight)
        _backward(loss, terms, params, args, remainder_weight)
        grad = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        if (
            args.schedule_unit == "step" or not steps_per_epoch
            or (step + 1) % steps_per_epoch == 0
        ):
            scheduler.step()
        if step == 0 or (step + 1) % args.log_steps == 0:
            history.append({
                "step": step + 1, "loss": float(loss),
                "base": float(terms["base"]), "omega": float(terms["omega"]),
                "recovery": float(terms["recovery"]),
                "candidate": float(terms["candidate"]),
                "topk": float(terms["topk"]),
                "reference": float(terms["reference"]),
                "grad_norm": float(grad),
                "remainder_weight": remainder_weight,
                "training_pool_size": int(
                    len(training_source["allocations"])),
                "target_index": int(state["target"]),
                "minimizer_count": int(len(state["minimizers"])),
                "set_external_gap": float(state["gap"]),
                "epoch": epoch, "temperature": temperature,
            })
        if (
            args.refresh_steps > 0 and (step + 1) % args.refresh_steps == 0
        ):
            refresh += 1
            training_source = _dynamic_source(
                audit, calibration, state["c"], args, refresh)
            current_D = _hard_eval(
                codec, tail, refresh_x, refresh_y,
                training_source["allocations"], refresh_args)
            state = _select_state(
                training_source, current_D, calibration, args)
            state = _set_scales(state, training_scales)
            audit_D = _subset_distortion(
                training_source, current_D, audit)
            audit_state = _select_state(
                audit, audit_D, calibration, args)
            audit_state = _set_scales(audit_state, audit_scales)
            report = _state_report(state)
            report["pool_size"] = int(len(training_source["allocations"]))
            active.append(report)
        if args.select_steps > 0 and (step + 1) % args.select_steps == 0:
            validation_D = _hard_eval(
                codec, tail, val_x, val_y, audit["allocations"], args)
            terms_val = _objective_terms(
                validation_D, audit_state, args, remainder_weight)
            eligible = (
                terms_val["base"]
                <= initial_base * (1 + args.selection_base_tolerance))
            selection.append({
                "step": step + 1, **terms_val, "eligible": eligible})
            print(
                f"step={step + 1}/{total_steps} epoch={epoch} "
                f"score={terms_val['score']:.6g} "
                f"base={terms_val['base']:.6g} "
                f"omega={terms_val['omega']:.6g} "
                f"pool={len(training_source['allocations'])} "
                f"eligible={eligible}", flush=True)
            if eligible and terms_val["score"] < best_score:
                best_score, best_step = terms_val["score"], step + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in codec.state_dict().items()}
    if args.select_steps > 0:
        codec.load_state_dict(best_state)
    final_D = _hard_eval(
        codec, tail, refresh_x, refresh_y,
        audit["allocations"], refresh_args)
    state = _select_state(audit, final_D, calibration, args)
    after_matrix = _hard_eval(
        codec, tail, val_x, val_y, audit["allocations"], args,
        return_per_image=True)
    after_val_D = after_matrix.mean(1)
    after = _report(
        audit, after_val_D, state, int(calibration["rate_dimension"]))
    curve = _hard_eval(
        codec, tail, val_x, val_y, _uniform_allocations(codec), args)
    curve_report = _curve_report(
        bits, curve, args.monotonic_tolerance)
    save_codec_v1(codec, args.checkpoint)
    target = int(state["target"])
    paired = after_matrix[target] - before_matrix[target]
    half = 1.96 * paired.std(ddof=1) / np.sqrt(len(paired))
    array_path = Path(args.output).with_suffix(".npz")
    np.savez_compressed(
        array_path, allocations=audit["allocations"],
        before_per_image=before_matrix, after_per_image=after_matrix,
        final_target_index=target,
        final_minimizer_indices=state["minimizers"])
    _write(args.output, {
        "before": before, "after": after, "menu": curve_report,
        "history": history, "active_states": active,
        "validation_selection": selection,
        "initial_validation_score": initial_terms["score"],
        "selected_validation_score": (
            best_score if args.select_steps > 0 else None),
        "selected_validation_step": (
            best_step if args.select_steps > 0 else None),
        "final_state": _state_report(state),
        "joint_codebooks": True,
        "remainder_parameters": args.remainder_parameters,
        "dynamic_allocations": args.dynamic_allocations,
        "audit_allocation_count": int(len(audit["allocations"])),
        "epochs": args.epochs, "train_images": args.train_images,
        "batch_size": args.images, "learning_rate": args.lr,
        "tau": [args.tau_start, args.tau_end],
        "schedule_unit": args.schedule_unit,
        "total_steps": total_steps,
        "remainder_weight": remainder_weight,
        "remainder_grad_ratio": args.remainder_grad_ratio,
        "final_target_paired_delta": float(paired.mean()),
        "final_target_paired_delta_ci95": [
            float(paired.mean() - half), float(paired.mean() + half)],
        "checkpoint": args.checkpoint, "arrays": str(array_path),
    })
    if not curve_report["monotonic"]:
        raise RuntimeError("short training broke the monotonic RD gate")


def _common(parser):
    parser.add_argument("--codec", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--teachers", required=True)
    parser.add_argument("--hard-features", required=True)
    parser.add_argument("--hard-teachers", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--norm-mode", default="per_image")
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument("--hard-images", type=int, default=32)
    parser.add_argument("--hard-image-offset", type=int, default=0)
    parser.add_argument("--hard-batch-size", type=int, default=4)
    parser.add_argument("--allocation-chunk", type=int, default=4)
    parser.add_argument("--pq-temperature", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--anchor-bit", type=int, default=6)
    parser.add_argument("--monotonic-tolerance", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)


def parser():
    main = argparse.ArgumentParser()
    sub = main.add_subparsers(dest="command", required=True)
    repair = sub.add_parser("repair")
    _common(repair)
    repair.add_argument("--monotonic-weight", type=float, default=10.0)
    repair.add_argument("--margin", type=float, default=0.0)
    repair.add_argument("--eval-steps", type=int, default=5)
    repair.add_argument("--minimum-high-rate-gain", type=float, default=0.0)
    repair.add_argument("--split-scale", type=float, default=0.001)
    short = sub.add_parser("short")
    _common(short)
    short.add_argument("--allocation-file", required=True)
    short.add_argument("--calibration", required=True)
    short.add_argument("--refresh-images", type=int, default=8)
    short.add_argument("--refresh-steps", type=int, default=5)
    short.add_argument("--allocations", type=int, default=64)
    short.add_argument("--epochs", type=int, default=0)
    short.add_argument("--train-images", type=int, default=4500)
    short.add_argument("--log-steps", type=int, default=1)
    short.add_argument("--select-steps", type=int, default=0)
    short.add_argument("--tau-start", type=float, default=0.01)
    short.add_argument("--tau-end", type=float, default=0.01)
    short.add_argument(
        "--schedule-unit", choices=("step", "epoch"), default="step")
    short.add_argument("--ridge", type=float, default=1e-3)
    short.add_argument("--tie-atol", type=float, default=1e-8)
    short.add_argument("--tie-rtol", type=float, default=1e-8)
    short.add_argument("--lse-temperature", type=float, default=0.1)
    short.add_argument("--candidate-weight", type=float, default=1.0)
    short.add_argument("--topk-weight", type=float, default=0.25)
    short.add_argument("--topk", type=int, default=8)
    short.add_argument("--reference-bit", type=int, default=6)
    short.add_argument("--reference-weight", type=float, default=1.0)
    short.add_argument("--remainder-weight", type=float, default=0.0)
    short.add_argument("--remainder-grad-ratio", type=float, default=-1.0)
    short.add_argument(
        "--remainder-parameters", choices=("u", "joint"), default="u")
    short.add_argument("--recovery-fraction", type=float, default=1.0)
    short.add_argument("--dynamic-allocations", action="store_true")
    short.add_argument("--dynamic-single", type=int, default=32)
    short.add_argument("--dynamic-random", type=int, default=32)
    short.add_argument("--selection-base-tolerance", type=float, default=0.0)
    return main


if __name__ == "__main__":
    args = parser().parse_args()
    globals()[f"command_{args.command}"](args)
