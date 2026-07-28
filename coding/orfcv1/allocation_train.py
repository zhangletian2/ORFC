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
from p1_fixed_rate import (
    build_tail, estimate_c, make_allocations, topk_cost_allocate,
)


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(path)


def _design(rates, dimension):
    return np.exp2(-2.0 * np.asarray(rates) / float(dimension))


def _calibration_dict(source):
    return {
        key: np.asarray(source[key]).copy()
        for key in source.files
    }


def _ideal_phi(source, calibration):
    allocations = source["allocations"]
    if "ideal_cost_table" in calibration:
        table = np.asarray(calibration["ideal_cost_table"], dtype=np.float64)
        groups = np.arange(table.shape[0])[None]
        return table[groups, allocations].sum(1)
    A = _design(source["rates"], int(calibration["rate_dimension"]))
    return A @ np.asarray(calibration["c_g"], dtype=np.float64)


def _ideal_table(calibration):
    if "ideal_cost_table" in calibration:
        return np.asarray(calibration["ideal_cost_table"], dtype=np.float64)
    bits = np.asarray(calibration["mode_bits"], dtype=np.float64)
    c = np.asarray(calibration["c_g"], dtype=np.float64)
    dimension = int(calibration["rate_dimension"])
    return c[:, None] * np.exp2(-2.0 * bits[None] / dimension)


def _with_ideal_set(calibration, args):
    """Attach an exact Top-K ideal set and its separation from the complement."""
    if args.ideal_set_size < 1:
        raise ValueError("ideal-set-size must be positive")
    updated = dict(calibration)
    bits = tuple(map(int, calibration["mode_bits"]))
    budget = int(np.asarray(calibration["ideal_bits"]).sum())
    top = topk_cost_allocate(
        _ideal_table(calibration), bits, budget, args.ideal_set_size + 1)
    count = min(args.ideal_set_size, len(top["bits"]))
    members = top["bits"][:count]
    outside = top["bits"][count] if len(top["bits"]) > count else np.empty(0)
    outside_value = (
        float(top["values"][count])
        if len(top["values"]) > count else float("inf"))
    second = top["bits"][1] if len(top["bits"]) > 1 else np.empty(0)
    second_value = (
        float(top["values"][1]) if len(top["values"]) > 1 else float("inf"))
    updated.update({
        "ideal_bits": members[0],
        "ideal_value": np.asarray(top["values"][0]),
        "second_bits": second,
        "second_value": np.asarray(second_value),
        "ideal_gap": np.asarray(second_value - top["values"][0]),
        "ideal_set_bits": members,
        "ideal_set_values": top["values"][:count],
        "ideal_set_outside_bits": outside,
        "ideal_set_gap": np.asarray(outside_value - top["values"][0]),
    })
    return updated


def _ideal_anchor_modes(calibration):
    lookup = {
        int(bit): index
        for index, bit in enumerate(calibration["mode_bits"])}
    rows = np.atleast_2d(calibration.get(
        "ideal_set_bits", calibration["ideal_bits"]))
    outside = np.asarray(
        calibration.get("ideal_set_outside_bits", np.empty(0)))
    if outside.size:
        rows = np.concatenate([rows, outside[None]], axis=0)
    return np.asarray([
        [lookup[int(bit)] for bit in row] for row in rows
    ], dtype=np.int64)


def _feasible_count(groups, mode_bits, budget):
    counts = {0: 1}
    for _ in range(groups):
        next_counts = {}
        for used, count in counts.items():
            for bit in mode_bits:
                total = used + int(bit)
                if total <= budget:
                    next_counts[total] = next_counts.get(total, 0) + count
        counts = next_counts
    return counts.get(budget, 0)


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


def _validate_training_slices(args, count):
    ranges = {
        "calibration": (
            args.outer_calibration_offset,
            args.outer_calibration_offset + args.outer_calibration_images),
        "mining": (
            args.outer_mining_offset,
            args.outer_mining_offset + args.outer_mining_images),
        "inner": (
            args.train_image_offset,
            args.train_image_offset + args.train_images),
    }
    for name, (first, last) in ranges.items():
        if first < 0 or last > count or first >= last:
            raise ValueError(f"{name} image slice is invalid")
    names = list(ranges)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = ranges[left], ranges[right]
            if max(a[0], b[0]) < min(a[1], b[1]):
                raise ValueError(f"{left} and {right} image slices overlap")


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
    phi = _ideal_phi(source, calibration)
    remainder = distortion - phi
    tolerance = max(
        args.tie_atol, args.tie_rtol * max(abs(float(phi.min())), 1.0))
    minimizers = np.flatnonzero(phi <= phi.min() + tolerance)
    lookup = {
        int(bit): index
        for index, bit in enumerate(calibration["mode_bits"])}
    target_bits = np.atleast_2d(calibration.get(
        "ideal_set_bits", calibration["ideal_bits"]))
    target_set = []
    for row in target_bits:
        modes = np.asarray([lookup[int(bit)] for bit in row])
        hits = np.flatnonzero(np.all(source["allocations"] == modes, axis=1))
        if len(hits) != 1:
            raise ValueError(
                "allocation pool must contain every ideal-set member")
        target_set.append(int(hits[0]))
    target_set = np.asarray(list(dict.fromkeys(target_set)), dtype=int)
    if not np.intersect1d(target_set, minimizers).size:
        raise RuntimeError("ideal set does not contain the ideal minimizer")
    outside = np.setdiff1d(np.arange(len(phi)), target_set)
    target = int(target_set[np.argmin(distortion[target_set])])
    active_count = min(
        getattr(args, "ideal_batch_size", len(target_set)),
        len(target_set))
    active_targets = [target]
    if active_count > 1 and int(target_set[0]) != target:
        active_targets.append(int(target_set[0]))
    for index in target_set[np.argsort(distortion[target_set])]:
        if int(index) not in active_targets:
            active_targets.append(int(index))
        if len(active_targets) >= active_count:
            break
    active_targets = np.asarray(active_targets, dtype=int)
    gap = float(calibration.get(
        "ideal_set_gap", calibration["ideal_gap"]))
    order = np.argsort(remainder)
    edge = max(2, args.allocations // 4)
    selected = (
        set(order[:edge]) | set(order[-edge:])
        | set(map(int, minimizers)) | set(map(int, active_targets)))
    outside_bits = np.asarray(
        calibration.get("ideal_set_outside_bits", np.empty(0)))
    if outside_bits.size:
        modes = np.asarray([lookup[int(bit)] for bit in outside_bits])
        hits = np.flatnonzero(np.all(source["allocations"] == modes, axis=1))
        if len(hits) == 1:
            selected.add(int(hits[0]))
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
    target_local = local[target]
    target_set_local = np.asarray([
        local[int(index)] for index in active_targets], dtype=int)
    full_target_set = set(map(int, target_set))
    competitor_local = np.asarray([
        position for position, index in enumerate(selected)
        if int(index) not in full_target_set], dtype=int)
    empirical_margin = (
        float(distortion[outside].min() - distortion[target_set].min())
        if len(outside) else float("inf"))
    return {
        "selected": selected, "allocations": source["allocations"][selected],
        "phi": phi[selected], "full_phi": phi,
        "target": target, "target_set": target_set,
        "active_target_set": active_targets,
        "minimizers": minimizers,
        "target_local": target_local,
        "target_set_local": target_set_local,
        "competitor_local": competitor_local,
        "minimizer_local": np.asarray(
            [local[int(index)] for index in minimizers]),
        "gap": gap,
        "empirical_margin": empirical_margin,
        "reference": reference,
        "reference_local": local.get(reference),
        "omega_scale": max(float(np.ptp(remainder)), 1.0),
        "distortion_scale": max(float(distortion[target_set].min()), 1.0),
    }


def _allocation_source(allocations, calibration):
    allocations = np.asarray(allocations, dtype=np.int64)
    return {
        "allocations": allocations,
        "rates": allocation_rates(allocations, calibration["cost_table"]),
    }


def _dynamic_source(audit, calibration, c, args, refresh):
    if not args.dynamic_allocations:
        return audit
    budget = int(round(float(audit["rates"][0].sum())))
    generated, _, _ = make_allocations(
        c, tuple(map(int, calibration["mode_bits"])), budget,
        args.dynamic_single, args.dynamic_random, args.seed + refresh,
        int(calibration["rate_dimension"]),
        calibration.get("ideal_cost_table"))
    pool = np.unique(
        np.concatenate([audit["allocations"], generated]), axis=0)
    return _allocation_source(pool, calibration)


def _set_scales(state, scales):
    state["omega_scale"], state["distortion_scale"] = scales
    return state


def _objective_terms(distortion, state, args, remainder_weight):
    D = np.asarray(distortion, dtype=np.float64)
    complete = len(D) == len(state["full_phi"])
    phi = state["full_phi"] if complete else state["phi"]
    targets = (
        state["target_set"] if complete else state["target_set_local"])
    competitors = np.setdiff1d(np.arange(len(D)), targets)
    e = D - phi
    scale = args.lse_temperature * state["omega_scale"]
    high = scale * np.log(np.exp((e - e.max()) / scale).sum()) + e.max()
    low_e = -e
    low = (
        scale * np.log(np.exp((low_e - low_e.max()) / scale).sum())
        + low_e.max())
    omega_raw = high + low - 2 * scale * math.log(len(e))
    target_value = float(D[targets].min())
    margin = (
        float(D[competitors].min() - target_value)
        if len(competitors) else float("inf"))
    recovery = max(args.recovery_margin - margin, 0.0) / (
        state["distortion_scale"])
    candidate = target_value / state["distortion_scale"]
    base = candidate + (
        args.candidate_mean_weight * D.mean() / state["distortion_scale"])
    return {
        "score": float(base + remainder_weight * recovery),
        "base": float(base), "recovery": float(recovery),
        "omega": float(np.ptp(e)), "gap": float(state["gap"]),
        "sampled_omega_to_set_gap": float(
            np.ptp(e) / max(state["gap"], 1e-12)),
        "candidate": float(candidate), "empirical_margin": margin,
    }


def _loss(codec, tail, batch, state, args, remainder_weight):
    D = _distortions(
        codec, tail, batch, state["allocations"], args.allocation_chunk,
        args.pq_temperature, True)
    phi = torch.as_tensor(state["phi"], device=D.device)
    e = D.double() - phi
    tau = args.lse_temperature
    scaled_tau = tau * state["omega_scale"]
    omega_raw = (
        scaled_tau * torch.logsumexp(e / scaled_tau, 0)
        + scaled_tau * torch.logsumexp(-e / scaled_tau, 0)
        - 2 * scaled_tau * math.log(len(e)))
    omega = omega_raw / state["omega_scale"]
    targets = torch.as_tensor(
        state["target_set_local"], device=D.device)
    competitors = torch.as_tensor(
        state["competitor_local"], device=D.device)
    normalized = D / state["distortion_scale"]
    candidate = normalized[targets].min()
    if len(competitors):
        outside = normalized[competitors].min()
        recovery_constraint = (
            candidate - outside
            + args.recovery_margin / state["distortion_scale"])
        recovery = torch.relu(recovery_constraint)
        empirical_margin = D[competitors].min() - D[targets].min()
    else:
        recovery = candidate.new_zeros(())
        recovery_constraint = candidate.new_zeros(())
        empirical_margin = candidate.new_tensor(float("inf"))
    base = candidate + (
        args.candidate_mean_weight * D.mean() / state["distortion_scale"])
    return base + remainder_weight * recovery, {
        "base": base, "omega": omega, "recovery": recovery,
        "recovery_constraint": recovery_constraint,
        "candidate": candidate,
        "empirical_margin": empirical_margin,
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


def _calibrate_remainder(codec, tail, batch, state, args, parameters, rotation):
    if args.remainder_grad_ratio < 0:
        return args.remainder_weight
    _, terms = _loss(codec, tail, batch, state, args, 0.0)
    base = list(torch.autograd.grad(
        terms["base"], parameters, retain_graph=True, allow_unused=True))
    # Calibrate with the unhinged constraint.  An already satisfied first
    # minibatch must not silently disable the auxiliary for the whole run.
    recovery = list(torch.autograd.grad(
        terms["recovery_constraint"], parameters, allow_unused=True))
    base[0] = _tangent_gradient(rotation, base[0])
    recovery[0] = _tangent_gradient(rotation, recovery[0])
    def norm(values):
        terms = [
            value.square().sum() for value in values if value is not None]
        return (
            torch.sqrt(torch.stack(terms).sum()) if terms
            else torch.zeros((), device=rotation.device))
    base_norm, remainder_norm = norm(base), norm(recovery)
    codec.zero_grad(set_to_none=True)
    if args.remainder_grad_ratio > 0 and float(remainder_norm) <= 1e-12:
        raise RuntimeError(
            "recovery constraint has zero calibration gradient")
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


def _backward(terms, parameters, rotation, remainder_weight):
    primary = list(torch.autograd.grad(
        terms["base"], parameters, retain_graph=remainder_weight > 0,
        allow_unused=True))
    primary[0] = _tangent_gradient(rotation, primary[0])
    if remainder_weight <= 0:
        for parameter, gradient in zip(parameters, primary):
            parameter.grad = gradient
        return 0.0, False
    auxiliary = list(torch.autograd.grad(
        terms["recovery"], parameters, allow_unused=True))
    auxiliary[0] = _tangent_gradient(rotation, auxiliary[0])
    cosines, projected = [], False
    for parameter, base, extra in zip(parameters, primary, auxiliary):
        extra, cosine, changed = _protect_primary(extra, base)
        projected = projected or changed
        if base is None:
            parameter.grad = (
                remainder_weight * extra if extra is not None else None)
        else:
            parameter.grad = base
            if extra is not None:
                parameter.grad.add_(extra, alpha=remainder_weight)
        if extra is not None and base is not None:
            cosines.append(cosine)
    return float(np.mean(cosines)) if cosines else 0.0, projected


def _report(source, distortion, state, dimension):
    phi = state["full_phi"]
    remainder = distortion - phi
    minimizers = np.asarray(state["minimizers"])
    targets = np.asarray(state["target_set"])
    outside = np.setdiff1d(np.arange(len(distortion)), targets)
    target = int(targets[np.argmin(distortion[targets])])
    gap = state["gap"]
    return {
        "mean_distortion": float(distortion.mean()),
        "target_index": target,
        "target_bits": source["rates"][target].tolist(),
        "target_distortion": float(distortion[target]),
        "ideal_set_indices": targets.tolist(),
        "ideal_set_count": int(len(targets)),
        "ideal_set_best_distortion": float(distortion[targets].min()),
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
        "empirical_margin": (
            float(distortion[outside].min() - distortion[targets].min())
            if len(outside) else float("inf")),
        "set_external_gap": float(gap),
        "omega_scope": "sampled_pool",
        "sampled_omega_to_set_gap": float(
            np.ptp(remainder) / max(gap, 1e-12)),
        "sampled_near_ideal_count": int(np.count_nonzero(
            phi - phi.min() <= np.ptp(remainder) + 1e-12)),
    }


def _training_batches(args, count, rng):
    first = args.train_image_offset
    count = min(args.train_images, count - first)
    if count < 1:
        raise ValueError("training image slice is empty")
    if args.epochs <= 0:
        for step in range(args.steps):
            ids = first + rng.choice(
                count, min(args.images, count), replace=False)
            yield step, None, ids
        return
    step = 0
    for epoch in range(args.epochs):
        order = first + rng.permutation(count)
        for start in range(0, count, args.images):
            yield step, epoch, order[start:start + args.images]
            step += 1


def _state_report(state):
    coverage = sum(
        len(np.unique(state["allocations"][:, group]))
        for group in range(state["allocations"].shape[1]))
    return {
        "target": int(state["target"]),
        "target_allocation": state["allocations"][
            state["target_local"]].tolist(),
        "target_phi": float(state["phi"][state["target_local"]]),
        "ideal_set_count": int(len(state["target_set"])),
        "ideal_set_indices": state["target_set"].tolist(),
        "active_ideal_set_count": int(len(state["active_target_set"])),
        "active_ideal_set_indices": state["active_target_set"].tolist(),
        "active_ideal_set_allocations": state["allocations"][
            state["target_set_local"]].tolist(),
        "minimizers": state["minimizers"].tolist(),
        "gap": float(state["gap"]),
        "selected": state["selected"].tolist(),
        "selected_allocations": state["allocations"].tolist(),
        "selected_count": int(len(state["selected"])),
        "group_mode_coverage": int(coverage),
        "empirical_margin": float(state["empirical_margin"]),
    }


@torch.no_grad()
def _refresh_ideal(codec, tail, features, calibration, args):
    bits = tuple(map(int, calibration["mode_bits"]))
    first = args.outer_calibration_offset
    count = min(args.outer_calibration_images, len(features) - first)
    if count < 1:
        raise ValueError("outer calibration image slice is empty")
    probe = features[first:first + count]
    estimate_args = argparse.Namespace(
        images=count, batch_size=args.outer_batch_size,
        norm_mode=args.norm_mode, group_chunk=args.outer_group_chunk,
        eps=args.outer_eps)
    estimates = {
        bit: estimate_c(
            codec, tail, probe, mode, bit, estimate_args, args.device)
        for mode, bit in enumerate(bits)
    }
    table = np.stack([
        estimates[bit][1].mean(0) for bit in bits], axis=1)
    updated = dict(calibration)
    updated.update({
        "c_g": estimates[args.reference_bit][0],
        "ideal_cost_table": table,
        "ideal_model": np.asarray("discrete_jvp"),
    })
    return _with_ideal_set(updated, args)


def _refresh_training_state(
    codec, tail, train_x, train_y, audit, previous, calibration, args,
    refresh,
):
    calibration = (
        _refresh_ideal(codec, tail, train_x, calibration, args)
        if args.outer_refresh else calibration)
    anchors = audit["allocations"]
    if previous is not None:
        anchors = np.unique(np.concatenate([
            anchors, previous["allocations"]]), axis=0)
    anchors = np.unique(np.concatenate([
        anchors, _ideal_anchor_modes(calibration)]), axis=0)
    source = _dynamic_source(
        _allocation_source(anchors, calibration), calibration,
        calibration["c_g"], args, refresh)
    mine_args = argparse.Namespace(**vars(args))
    mine_args.hard_images = args.outer_mining_images
    mine_args.hard_image_offset = args.outer_mining_offset
    distortion = _hard_eval(
        codec, tail, train_x, train_y, source["allocations"], mine_args)
    return calibration, source, _select_state(
        source, distortion, calibration, args)


def command_short(args):
    device, codec, tail, train_x, train_y, val_x, val_y = _base(args)
    with np.load(args.calibration, allow_pickle=False) as saved:
        calibration = _calibration_dict(saved)
    if args.outer_refresh and not args.dynamic_allocations:
        raise ValueError("outer refresh requires dynamic allocations")
    if args.outer_refresh:
        _validate_training_slices(args, len(train_x))
    saved_images = (
        calibration["q_per_image_by_bit"].shape[1]
        if "q_per_image_by_bit" in calibration else 0)
    if saved_images < args.minimum_saved_calibration_images:
        raise ValueError(
            f"saved calibration has {saved_images} images; "
            f"{args.minimum_saved_calibration_images} required")
    calibration = _with_ideal_set(calibration, args)
    audit_calibration = dict(calibration)
    audit_allocations = np.unique(np.concatenate([
        calibration["allocations"], _ideal_anchor_modes(calibration)]), axis=0)
    audit = _allocation_source(audit_allocations, calibration)
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
    refresh_args = argparse.Namespace(**vars(args))
    refresh_args.hard_images = args.outer_mining_images
    refresh_args.hard_image_offset = args.outer_mining_offset
    calibration, training_source, state = _refresh_training_state(
        codec, tail, train_x, train_y, audit, None, calibration, args, 0)
    training_scales = (state["omega_scale"], state["distortion_scale"])
    audit_D = _hard_eval(
        codec, tail, train_x, train_y, audit["allocations"], refresh_args)
    audit_state = _select_state(
        audit, audit_D, audit_calibration, args)
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
        int(audit_calibration["rate_dimension"]))
    initial_dynamic_D = _hard_eval(
        codec, tail, val_x, val_y, training_source["allocations"], args)
    initial_dynamic_state = _select_state(
        training_source, initial_dynamic_D, calibration, args)
    initial_dynamic_state = _set_scales(
        initial_dynamic_state, training_scales)
    initial_dynamic_report = _report(
        training_source, initial_dynamic_D, initial_dynamic_state,
        int(calibration["rate_dimension"]))
    total_steps = (
        args.epochs * math.ceil(
            min(args.train_images, len(train_x) - args.train_image_offset)
            / args.images)
        if args.epochs > 0 else args.steps)
    steps_per_epoch = (
        math.ceil(
            min(args.train_images, len(train_x) - args.train_image_offset)
            / args.images)
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
        device, args.norm_mode,
        args.train_image_offset + np.arange(min(
            args.images, len(train_x) - args.train_image_offset)))
    remainder_weight = _calibrate_remainder(
        codec, tail, calibration_batch, state, args, params, rotation)
    history, active, refresh, last_refresh_step = [], [], 0, 0
    initial_terms = _objective_terms(
        initial_dynamic_D, initial_dynamic_state, args, remainder_weight)
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
            terms, params, rotation, remainder_weight)
        parameter_grad_norms = [
            float(parameter.grad.norm()) if parameter.grad is not None else 0.0
            for parameter in params]
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
                "empirical_margin": float(terms["empirical_margin"]),
                "grad_norm": float(grad),
                "rotation_grad_norm": parameter_grad_norms[0],
                "codebook_grad_norms": parameter_grad_norms[1:],
                "gradient_cosine": gradient_cosine,
                "gradient_projected": gradient_projected,
                "rotation_step_size": rotation_optimizer.state[
                    rotation].get("last_step_size"),
                "orthogonality_frobenius": codec.transform.orth_error(),
                "remainder_weight": remainder_weight,
                "training_pool_size": int(
                    len(training_source["allocations"])),
                "target_index": int(state["target"]),
                "ideal_set_count": int(len(state["target_set"])),
                "active_ideal_set_count": int(
                    len(state["active_target_set"])),
                "minimizer_count": int(len(state["minimizers"])),
                "set_external_gap": float(state["gap"]),
                "epoch": epoch, "temperature": temperature,
            })
        if (
            args.refresh_steps > 0 and (step + 1) % args.refresh_steps == 0
        ):
            refresh += 1
            last_refresh_step = step + 1
            calibration, training_source, state = _refresh_training_state(
                codec, tail, train_x, train_y, audit, state,
                calibration, args, refresh)
            state = _set_scales(state, training_scales)
            report = _state_report(state)
            report["pool_size"] = int(len(training_source["allocations"]))
            active.append(report)
        if args.select_steps > 0 and (step + 1) % args.select_steps == 0:
            validation_D = _hard_eval(
                codec, tail, val_x, val_y,
                training_source["allocations"], args)
            validation_state = _select_state(
                training_source, validation_D, calibration, args)
            validation_state = _set_scales(
                validation_state, training_scales)
            terms_val = _objective_terms(
                validation_D, validation_state, args, remainder_weight)
            validation_trace.append({"step": step + 1, **terms_val})
            print(
                f"step={step + 1}/{total_steps} epoch={epoch} "
                f"score={terms_val['score']:.6g} "
                f"base={terms_val['base']:.6g} "
                f"margin={terms_val['empirical_margin']:.6g} "
                f"omega={terms_val['omega']:.6g} "
                f"pool={len(training_source['allocations'])}", flush=True)
    if args.outer_refresh and last_refresh_step != total_steps:
        refresh += 1
        calibration, training_source, state = _refresh_training_state(
            codec, tail, train_x, train_y, audit, state,
            calibration, args, refresh)
        state = _set_scales(state, training_scales)
        active.append(_state_report(state))
    final_dynamic_matrix = _hard_eval(
        codec, tail, val_x, val_y, training_source["allocations"], args,
        return_per_image=True)
    final_dynamic_D = final_dynamic_matrix.mean(1)
    final_dynamic_state = _select_state(
        training_source, final_dynamic_D, calibration, args)
    final_dynamic_state = _set_scales(
        final_dynamic_state, training_scales)
    final_terms = _objective_terms(
        final_dynamic_D, final_dynamic_state, args, remainder_weight)
    after_matrix = _hard_eval(
        codec, tail, val_x, val_y, audit["allocations"], args,
        return_per_image=True)
    after_val_D = after_matrix.mean(1)
    if validation_trace[-1]["step"] != total_steps:
        validation_trace.append({"step": total_steps, **final_terms})
    final_audit_state = _select_state(
        audit, after_val_D, audit_calibration, args)
    final_audit_state = _set_scales(final_audit_state, audit_scales)
    after = _report(
        audit, after_val_D, final_audit_state,
        int(audit_calibration["rate_dimension"]))
    final_dynamic_report = _report(
        training_source, final_dynamic_D, final_dynamic_state,
        int(calibration["rate_dimension"]))
    curve = _hard_eval(
        codec, tail, val_x, val_y, _uniform_allocations(codec), args)
    curve_report = _curve_report(
        bits, curve, args.monotonic_tolerance)
    save_codec_v1(codec, args.checkpoint)
    target = int(audit_state["target"])
    paired = after_matrix[target] - before_matrix[target]
    half = 1.96 * paired.std(ddof=1) / np.sqrt(len(paired))
    array_path = Path(args.output).with_suffix(".npz")
    np.savez_compressed(
        array_path, allocations=audit["allocations"],
        before_per_image=before_matrix, after_per_image=after_matrix,
        final_dynamic_allocations=training_source["allocations"],
        final_dynamic_per_image=final_dynamic_matrix,
        final_ideal_cost_table=calibration.get(
            "ideal_cost_table", np.empty((0, 0))),
        final_ideal_bits=calibration["ideal_bits"],
        final_ideal_gap=calibration["ideal_gap"],
        final_ideal_set_bits=calibration["ideal_set_bits"],
        final_ideal_set_values=calibration["ideal_set_values"],
        final_ideal_set_gap=calibration["ideal_set_gap"],
        fixed_audit_target_index=target,
        fixed_audit_target_indices=audit_state["target_set"],
        final_dynamic_target_index=final_dynamic_state["target"],
        final_dynamic_target_indices=final_dynamic_state["target_set"])
    _write(args.output, {
        "before": before, "after": after, "menu": curve_report,
        "initial_dynamic": initial_dynamic_report,
        "final_dynamic": final_dynamic_report,
        "history": history, "active_states": active,
        "validation_trace": validation_trace,
        "initial_validation_score": initial_terms["score"],
        "final_validation_score": final_terms["score"],
        "final_state": _state_report(final_dynamic_state),
        "joint_codebooks": True,
        "fixed_ideal_model": not args.outer_refresh,
        "coefficient_source": "outer_discrete_jvp",
        "remainder_parameters": "u+codebooks",
        "dynamic_allocations": args.dynamic_allocations,
        "audit_allocation_count": int(len(audit["allocations"])),
        "audit_scope": "fixed_empirical_pool",
        "feasible_allocation_count": str(_feasible_count(
            codec.pq.G, calibration["mode_bits"],
            int(np.asarray(calibration["ideal_bits"]).sum()))),
        "sampled_remainder_is_global_lower_bound": True,
        "global_remainder_upper_bound": None,
        "strict_recovery_certified": False,
        "ideal_set_size": int(len(calibration["ideal_set_bits"])),
        "ideal_batch_size": args.ideal_batch_size,
        "ideal_set_definition": "exact_top_k_phi",
        "recovery_objective": "empirical_best_outside_minus_best_inside",
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
        "outer_refresh": args.outer_refresh,
        "outer_calibration_images": args.outer_calibration_images,
        "saved_calibration_images": saved_images,
        "outer_mining_images": args.outer_mining_images,
        "candidate_mean_weight": args.candidate_mean_weight,
        "recovery_margin": args.recovery_margin,
        "fixed_member_paired_delta": float(paired.mean()),
        "fixed_member_paired_delta_ci95": [
            float(paired.mean() - half), float(paired.mean() + half)],
        "paired_member_selection": (
            "lowest training-audit distortion in initial ideal set"),
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
    short.add_argument("--train-image-offset", type=int, default=0)
    short.add_argument("--log-steps", type=int, default=1)
    short.add_argument("--select-steps", type=int, default=0)
    short.add_argument("--tau-start", type=float, default=0.01)
    short.add_argument("--tau-end", type=float, default=0.01)
    short.add_argument(
        "--schedule-unit", choices=("step", "epoch"), default="step")
    short.add_argument("--tie-atol", type=float, default=1e-8)
    short.add_argument("--tie-rtol", type=float, default=1e-8)
    short.add_argument("--lse-temperature", type=float, default=0.1)
    short.add_argument("--recovery-margin", type=float, default=0.0)
    short.add_argument("--candidate-mean-weight", type=float, default=0.0)
    short.add_argument("--ideal-set-size", type=int, default=16)
    short.add_argument("--ideal-batch-size", type=int, default=16)
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
    short.add_argument("--outer-refresh", action="store_true")
    short.add_argument("--outer-calibration-images", type=int, default=64)
    short.add_argument("--outer-calibration-offset", type=int, default=0)
    short.add_argument("--outer-mining-images", type=int, default=64)
    short.add_argument("--outer-mining-offset", type=int, default=64)
    short.add_argument("--outer-batch-size", type=int, default=4)
    short.add_argument("--outer-group-chunk", type=int, default=8)
    short.add_argument("--outer-eps", type=float, default=0.01)
    short.add_argument(
        "--minimum-saved-calibration-images", type=int, default=0)
    return main


if __name__ == "__main__":
    args = parser().parse_args()
    globals()[f"command_{args.command}"](args)
