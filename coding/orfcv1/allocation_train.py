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
from opq import batch_inv_normalize_gpu, batch_normalize_gpu
from p1_fixed_rate import build_tail


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
    selected.update(map(int, outside[np.argsort(distortion[outside])[:edge]]))
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
        "A": A[selected], "intercept": intercept, "c": c,
        "target": target, "minimizers": minimizers,
        "minimizer_local": np.asarray(
            [local[int(index)] for index in minimizers]),
        "gap": gap,
        "reference": reference,
        "reference_local": local.get(reference),
        "omega_scale": max(float(np.ptp(remainder[selected])), 1.0),
        "distortion_scale": max(float(distortion[target]), 1.0),
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
    return {
        "target": int(state["target"]),
        "minimizers": state["minimizers"].tolist(),
        "gap": float(state["gap"]),
        "selected": state["selected"].tolist(),
        "c": state["c"].tolist(),
    }


def command_short(args):
    device, codec, tail, train_x, train_y, val_x, val_y = _base(args)
    source = np.load(args.allocation_file, allow_pickle=False)
    calibration = np.load(args.calibration, allow_pickle=False)
    bits = np.log2(codec.pq.mode_sizes).astype(int)
    params = _joint_parameters(codec)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    rng = np.random.default_rng(args.seed)
    refresh_x = train_x[:args.refresh_images]
    refresh_y = train_y[:args.refresh_images]
    refresh_args = argparse.Namespace(**vars(args))
    refresh_args.hard_images = args.refresh_images
    refresh_args.hard_image_offset = 0
    all_allocations = source["allocations"]
    initial_D = _hard_eval(
        codec, tail, refresh_x, refresh_y, all_allocations, refresh_args)
    state = _select_state(source, initial_D, calibration, args)
    before_matrix = _hard_eval(
        codec, tail, val_x, val_y, all_allocations, args,
        return_per_image=True)
    before_val_D = before_matrix.mean(1)
    before = _report(
        source, before_val_D, state, int(calibration["rate_dimension"]))
    total_steps = (
        args.epochs * math.ceil(
            min(args.train_images, len(train_x)) / args.images)
        if args.epochs > 0 else args.steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=args.lr * 0.01)
    remainder_weight, history, active = args.remainder_weight, [], []
    selection, best_score, best_state = [], float("inf"), None
    recalibrate = True
    for step, epoch, ids in _training_batches(args, len(train_x), rng):
        progress = step / max(total_steps - 1, 1)
        temperature = args.tau_start * (
            args.tau_end / args.tau_start) ** progress
        args.pq_temperature = temperature
        batch = _sample(
            train_x, train_y, args.images, rng, device, args.norm_mode, ids)
        optimizer.zero_grad(set_to_none=True)
        _, terms = _loss(codec, tail, batch, state, args, 0.0)
        if recalibrate and args.remainder_grad_ratio >= 0:
            base_grad = torch.autograd.grad(
                terms["base"], params, retain_graph=True, allow_unused=True)
            omega_grad = torch.autograd.grad(
                terms["recovery"], params, retain_graph=True, allow_unused=True)
            base_norm = torch.sqrt(sum(
                value.square().sum() for value in base_grad
                if value is not None))
            omega_norm = torch.sqrt(sum(
                value.square().sum() for value in omega_grad
                if value is not None))
            remainder_weight = (
                args.remainder_grad_ratio * float(base_norm)
                / max(float(omega_norm), 1e-12)
                if float(omega_norm) > 1e-12 else 0.0)
            recalibrate = False
        loss, terms = _loss(
            codec, tail, batch, state, args, remainder_weight)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % args.log_steps == 0:
            history.append({
                "step": step + 1, "loss": float(loss),
                "base": float(terms["base"]), "omega": float(terms["omega"]),
                "recovery": float(terms["recovery"]),
                "candidate": float(terms["candidate"]),
                "reference": float(terms["reference"]),
                "grad_norm": float(grad),
                "remainder_weight": remainder_weight,
                "target_index": int(state["target"]),
                "minimizer_count": int(len(state["minimizers"])),
                "set_external_gap": float(state["gap"]),
                "epoch": epoch, "temperature": temperature,
            })
        if (
            args.refresh_steps > 0 and (step + 1) % args.refresh_steps == 0
        ):
            current_D = _hard_eval(
                codec, tail, refresh_x, refresh_y,
                all_allocations, refresh_args)
            state = _select_state(source, current_D, calibration, args)
            active.append(_state_report(state))
            recalibrate = True
        if args.select_steps > 0 and (step + 1) % args.select_steps == 0:
            validation_D = _hard_eval(
                codec, tail, val_x, val_y, state["allocations"], args)
            score = float(validation_D[state["minimizer_local"]].mean())
            if state["reference_local"] is not None:
                score += args.reference_weight * float(
                    validation_D[state["reference_local"]])
            selection.append({"step": step + 1, "score": score})
            if score < best_score:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in codec.state_dict().items()}
    if best_state is not None:
        codec.load_state_dict(best_state)
    final_D = _hard_eval(
        codec, tail, refresh_x, refresh_y, all_allocations, refresh_args)
    state = _select_state(source, final_D, calibration, args)
    after_matrix = _hard_eval(
        codec, tail, val_x, val_y, all_allocations, args,
        return_per_image=True)
    after_val_D = after_matrix.mean(1)
    after = _report(
        source, after_val_D, state, int(calibration["rate_dimension"]))
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
        array_path, allocations=all_allocations,
        before_per_image=before_matrix, after_per_image=after_matrix,
        final_target_index=target,
        final_minimizer_indices=state["minimizers"])
    _write(args.output, {
        "before": before, "after": after, "menu": curve_report,
        "history": history, "active_states": active,
        "validation_selection": selection,
        "selected_validation_score": (
            best_score if best_state is not None else None),
        "final_state": _state_report(state),
        "joint_codebooks": True,
        "epochs": args.epochs, "train_images": args.train_images,
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
    short.add_argument("--recovery-fraction", type=float, default=1.0)
    return main


if __name__ == "__main__":
    args = parser().parse_args()
    globals()[f"command_{args.command}"](args)
