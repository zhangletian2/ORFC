#!/usr/bin/env python3
"""Warm up a multi-rate PQ menu and train fixed-rate recovery."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from cayley import CayleySGD, DirectOrthogonalTransform
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


def _base(args):
    device = torch.device(args.device)
    codec = load_codec_v1(args.codec, device).train()
    tail = build_tail(args.layer, device)
    train_x = np.load(args.features, mmap_mode="r")
    train_y = np.load(args.teachers, mmap_mode="r")
    val_x = np.load(args.hard_features, mmap_mode="r")
    val_y = np.load(args.hard_teachers, mmap_mode="r")
    return device, codec, tail, train_x, train_y, val_x, val_y


def command_warmup(args):
    device, codec, tail, train_x, train_y, val_x, val_y = _base(args)
    bits = np.log2(codec.pq.mode_sizes).astype(int)
    allocations = _uniform_allocations(codec)
    before = _hard_eval(
        codec, tail, val_x, val_y, allocations, args)
    rotation = codec.transform.get_rotation().detach().clone()
    scale = max(float(before.mean()), 1.0)
    codec.requires_grad_(False)
    params = [quantizer.codebooks for quantizer in codec.pq.quantizers]
    for parameter in params:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    rng = np.random.default_rng(args.seed)
    history = []
    for step in range(args.steps):
        batch = _sample(
            train_x, train_y, args.images, rng, device, args.norm_mode)
        optimizer.zero_grad(set_to_none=True)
        D = _distortions(
            codec, tail, batch, allocations, args.allocation_chunk,
            args.pq_temperature, True)
        loss = D.mean() / scale
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        if step == 0 or (step + 1) % args.log_steps == 0:
            history.append({
                "step": step + 1, "loss": float(loss),
                "grad_norm": float(grad)})
    after = _hard_eval(codec, tail, val_x, val_y, allocations, args)
    rotation_exact = torch.equal(codec.transform.get_rotation(), rotation)
    if not rotation_exact:
        raise RuntimeError("warmup changed the frozen rotation")
    save_codec_v1(codec, args.checkpoint)
    _write(args.output, {
        "before": _curve_report(bits, before, args.monotonic_tolerance),
        "after": _curve_report(bits, after, args.monotonic_tolerance),
        "rotation_frozen": rotation_exact,
        "steps": args.steps, "checkpoint": args.checkpoint,
        "history": history,
    })


def _select_state(source, distortion, calibration, args):
    A = _design(source["rates"], int(calibration["rate_dimension"]))
    intercept, c = 0.0, np.asarray(calibration["c_g"], dtype=np.float64)
    phi = A @ c
    remainder = distortion - phi
    tolerance = max(
        args.tie_atol, args.tie_rtol * max(abs(float(phi.min())), 1.0))
    minimizers = np.flatnonzero(phi <= phi.min() + tolerance)
    outside = np.setdiff1d(
        np.arange(len(phi)), minimizers, assume_unique=True)
    gap = float(calibration["ideal_gap"])
    lookup = {
        int(bit): index
        for index, bit in enumerate(calibration["mode_bits"])}
    target_modes = np.asarray(
        [lookup[int(bit)] for bit in calibration["ideal_bits"]])
    hits = np.flatnonzero(np.all(
        source["allocations"] == target_modes, axis=1))
    if len(hits) != 1:
        raise ValueError("allocation pool must contain the fixed ideal solution")
    target = int(hits[0])
    if target not in minimizers:
        raise RuntimeError("fixed ideal solution does not minimize fixed Phi")
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
    base = candidate
    return {
        "score": float(base + remainder_weight * recovery),
        "base": float(base), "recovery": float(recovery),
        "omega": float(np.ptp(e)), "gap": float(state["gap"]),
        "recovery_ratio": float(np.ptp(e) / max(state["gap"], 1e-12)),
        "candidate": float(candidate),
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
    base = candidate
    return base + remainder_weight * recovery, {
        "base": base, "omega": omega, "recovery": recovery,
        "candidate": candidate,
    }


def _joint_parameters(codec):
    codec.requires_grad_(False)
    if not isinstance(codec.transform, DirectOrthogonalTransform):
        raise TypeError(
            "allocation training requires DirectOrthogonalTransform")
    rotation = codec.transform.rotation
    rotation.requires_grad_(True)
    codebooks = []
    for quantizer in codec.pq.quantizers:
        quantizer.codebooks.requires_grad_(True)
        codebooks.append(quantizer.codebooks)
    return rotation, codebooks


def _tangent_gradient(rotation, gradient):
    if gradient is None:
        return None
    value = rotation.detach()
    product = value.t() @ gradient
    return gradient - value @ (0.5 * (product + product.t()))


def _calibrate_remainder(codec, tail, batch, state, args, rotation):
    if args.remainder_grad_ratio < 0:
        return args.remainder_weight
    _, terms = _loss(codec, tail, batch, state, args, 0.0)
    base = torch.autograd.grad(
        terms["base"], rotation, retain_graph=True, allow_unused=True)[0]
    remainder = torch.autograd.grad(
        terms["recovery"], rotation, allow_unused=True)[0]
    base = _tangent_gradient(rotation, base)
    remainder = _tangent_gradient(rotation, remainder)
    norm = lambda value: (
        value.norm() if value is not None
        else torch.zeros((), device=rotation.device))
    base_norm, remainder_norm = norm(base), norm(remainder)
    codec.zero_grad(set_to_none=True)
    return (
        args.remainder_grad_ratio * float(base_norm)
        / max(float(remainder_norm), 1e-12)
        if float(remainder_norm) > 1e-12 else 0.0)


def _protect_primary(auxiliary, primary):
    if auxiliary is None or primary is None:
        return auxiliary, 0.0, False
    dot = torch.dot(auxiliary.flatten(), primary.flatten())
    denom = primary.square().sum().clamp_min(1e-24)
    cosine = dot / torch.sqrt(
        denom * auxiliary.square().sum().clamp_min(1e-24))
    if dot < 0:
        auxiliary = auxiliary - dot / denom * primary
    return auxiliary, float(cosine), bool(dot < 0)


def _backward(terms, rotation, remainder_weight):
    terms["base"].backward(retain_graph=True)
    primary = _tangent_gradient(rotation, rotation.grad)
    rotation.grad = primary
    if remainder_weight <= 0:
        return 0.0, False
    gradient = torch.autograd.grad(
        terms["recovery"], rotation, allow_unused=True)[0]
    gradient = _tangent_gradient(rotation, gradient)
    gradient, cosine, projected = _protect_primary(
        gradient, primary)
    if gradient is not None and primary is None:
        rotation.grad = remainder_weight * gradient
    elif gradient is not None:
        rotation.grad.add_(gradient, alpha=remainder_weight)
    return cosine, projected


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
    calibration = np.load(args.calibration, allow_pickle=False)
    audit = _allocation_source(calibration["allocations"], calibration)
    bits = np.log2(codec.pq.mode_sizes).astype(int)
    rotation, codebooks = _joint_parameters(codec)
    params = [rotation, *codebooks]
    rotation_lr = args.rotation_lr or args.lr
    rotation_optimizer = CayleySGD(
        [rotation], lr=rotation_lr,
        fixed_point_iterations=args.cayley_iterations,
        reorthogonalize_every=args.reorthogonalize_every)
    codebook_optimizer = torch.optim.Adam(codebooks, lr=args.lr)
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
    rotation_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        rotation_optimizer, T_max=max(schedule_length, 1),
        eta_min=rotation_lr * 0.01)
    codebook_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        codebook_optimizer, T_max=max(schedule_length, 1),
        eta_min=args.lr * 0.01)
    args.pq_temperature = args.tau_start
    calibration_batch = _sample(
        train_x, train_y, args.images, np.random.default_rng(args.seed + 1),
        device, args.norm_mode, np.arange(min(args.images, len(train_x))))
    remainder_weight = _calibrate_remainder(
        codec, tail, calibration_batch, state, args, rotation)
    history, active, refresh = [], [], 0
    initial_terms = _objective_terms(
        before_val_D, audit_state, args, remainder_weight)
    validation_trace = [{"step": 0, **initial_terms}]
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
        rotation_optimizer.zero_grad(set_to_none=True)
        codebook_optimizer.zero_grad(set_to_none=True)
        loss, terms = _loss(
            codec, tail, batch, state, args, remainder_weight)
        gradient_cosine, gradient_projected = _backward(
            terms, rotation, remainder_weight)
        grad = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        codebook_optimizer.step()
        rotation_optimizer.step()
        if (
            args.schedule_unit == "step" or not steps_per_epoch
            or (step + 1) % steps_per_epoch == 0
        ):
            rotation_scheduler.step()
            codebook_scheduler.step()
        if step == 0 or (step + 1) % args.log_steps == 0:
            history.append({
                "step": step + 1, "loss": float(loss),
                "base": float(terms["base"]), "omega": float(terms["omega"]),
                "recovery": float(terms["recovery"]),
                "candidate": float(terms["candidate"]),
                "grad_norm": float(grad),
                "gradient_cosine": gradient_cosine,
                "gradient_projected": gradient_projected,
                "rotation_step_size": rotation_optimizer.state[
                    rotation].get("last_step_size"),
                "orthogonality_frobenius": codec.transform.orth_error(),
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
                audit, calibration, calibration["c_g"], args, refresh)
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
            validation_trace.append({"step": step + 1, **terms_val})
            print(
                f"step={step + 1}/{total_steps} epoch={epoch} "
                f"score={terms_val['score']:.6g} "
                f"base={terms_val['base']:.6g} "
                f"omega={terms_val['omega']:.6g} "
                f"pool={len(training_source['allocations'])}", flush=True)
    final_D = _hard_eval(
        codec, tail, refresh_x, refresh_y,
        audit["allocations"], refresh_args)
    state = _select_state(audit, final_D, calibration, args)
    state = _set_scales(state, audit_scales)
    after_matrix = _hard_eval(
        codec, tail, val_x, val_y, audit["allocations"], args,
        return_per_image=True)
    after_val_D = after_matrix.mean(1)
    final_terms = _objective_terms(
        after_val_D, state, args, remainder_weight)
    if validation_trace[-1]["step"] != total_steps:
        validation_trace.append({"step": total_steps, **final_terms})
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
        "validation_trace": validation_trace,
        "initial_validation_score": initial_terms["score"],
        "final_validation_score": final_terms["score"],
        "final_state": _state_report(state),
        "joint_codebooks": True,
        "fixed_ideal_model": True,
        "coefficient_source": "independent_jvp",
        "remainder_parameters": "u",
        "dynamic_allocations": args.dynamic_allocations,
        "audit_allocation_count": int(len(audit["allocations"])),
        "epochs": args.epochs, "train_images": args.train_images,
        "batch_size": args.images, "learning_rate": args.lr,
        "rotation_learning_rate": rotation_lr,
        "rotation_optimizer": "CayleySGD",
        "cayley_iterations": args.cayley_iterations,
        "reorthogonalize_every": args.reorthogonalize_every,
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
    parser.add_argument("--monotonic-tolerance", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)


def parser():
    main = argparse.ArgumentParser()
    sub = main.add_subparsers(dest="command", required=True)
    warmup = sub.add_parser("warmup")
    _common(warmup)
    warmup.add_argument("--log-steps", type=int, default=10)
    short = sub.add_parser("short")
    _common(short)
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
    short.add_argument("--tie-atol", type=float, default=1e-8)
    short.add_argument("--tie-rtol", type=float, default=1e-8)
    short.add_argument("--lse-temperature", type=float, default=0.1)
    short.add_argument("--reference-bit", type=int, default=6)
    short.add_argument("--remainder-weight", type=float, default=0.0)
    short.add_argument("--remainder-grad-ratio", type=float, default=-1.0)
    short.add_argument("--rotation-lr", type=float)
    short.add_argument("--cayley-iterations", type=int, default=5)
    short.add_argument("--reorthogonalize-every", type=int, default=100)
    short.add_argument("--recovery-fraction", type=float, default=1.0)
    short.add_argument("--dynamic-allocations", action="store_true")
    short.add_argument("--dynamic-single", type=int, default=32)
    short.add_argument("--dynamic-random", type=int, default=32)
    return main


if __name__ == "__main__":
    args = parser().parse_args()
    globals()[f"command_{args.command}"](args)
