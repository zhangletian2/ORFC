#!/usr/bin/env python3
"""Three pre-training gates for the differentiable fixed-rate remainder loss."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr

from codec_v1 import load_codec_v1, save_codec_v1
from fixed_rate_remainder import (
    evaluate_fixed_rate_remainder,
    random_exchange_walk,
    validate_fixed_total_rate,
)
from opq import batch_inv_normalize_gpu, batch_normalize_gpu
from p1_fixed_rate import build_tail, dp_allocate


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(path)


def _design(rates, dimension):
    return np.exp2(-2.0 * np.asarray(rates) / float(dimension))


def _positive_fit(A, y, c0, ridge):
    """Non-negative ridge fit with an unpenalised intercept."""
    n, g = A.shape
    X = np.c_[np.ones(n), A] / np.sqrt(n)
    target = np.asarray(y, dtype=np.float64) / np.sqrt(n)
    if ridge > 0:
        reg = np.zeros((g, g + 1))
        reg[:, 1:] = np.sqrt(ridge) * np.eye(g)
        X = np.r_[X, reg]
        target = np.r_[target, np.sqrt(ridge) * c0]
    lower = np.r_[-np.inf, np.zeros(g)]
    fit = lsq_linear(X, target, bounds=(lower, np.inf), lsmr_tol="auto")
    if not fit.success:
        raise RuntimeError(f"coefficient fit failed: {fit.message}")
    return fit.x[0], fit.x[1:]


def _projection_report(measured, c0, dimension, seed, alloc_fraction,
                       image_fraction, ridge):
    A, D = _design(measured["rates"], dimension), measured["distortion_per_image"]
    if len(A) < 2 or D.shape[1] < 2:
        raise ValueError("projection report needs at least two allocations/images")
    rng = np.random.default_rng(seed)
    alloc = rng.permutation(len(A))
    images = rng.permutation(D.shape[1])
    na = min(max(33, int(len(A) * alloc_fraction)), len(A) - 1)
    ni = min(max(1, int(D.shape[1] * image_fraction)), D.shape[1] - 1)
    fit_a, test_a = alloc[:na], alloc[na:]
    fit_i, test_i = images[:ni], images[ni:]
    y_fit = D[np.ix_(fit_a, fit_i)].mean(1)
    intercept, c = _positive_fit(A[fit_a], y_fit, c0, ridge)
    _, c_repeat = _positive_fit(
        A[fit_a], D[np.ix_(fit_a, test_i)].mean(1), c0, ridge)
    y_test = D[:, test_i].mean(1)
    e_fit, e_jvp = y_test - A @ c, y_test - A @ c0
    phi = A @ c
    ordered = np.sort(phi)
    return {
        "n_fit_allocations": int(len(fit_a)),
        "n_test_allocations": int(len(test_a)),
        "n_fit_images": int(len(fit_i)),
        "n_test_images": int(len(test_i)),
        "omega_projected_heldout": float(np.ptp(e_fit[test_a])),
        "omega_jvp_heldout": float(np.ptp(e_jvp[test_a])),
        "omega_projected_all": float(np.ptp(e_fit)),
        "mean_distortion_heldout": float(y_test[test_a].mean()),
        "coefficient_min": float(c.min()),
        "coefficient_max": float(c.max()),
        "coefficient_image_split_cosine": float(
            np.dot(c, c_repeat) / (np.linalg.norm(c) * np.linalg.norm(c_repeat))),
        "coefficient_image_split_relative_l2": float(
            np.linalg.norm(c - c_repeat) / np.linalg.norm(c)),
        "heldout_phi_distortion_spearman": float(
            spearmanr(phi[test_a], y_test[test_a]).statistic),
        "phi_gap": float(ordered[1] - ordered[0]),
        "intercept": float(intercept),
        "coefficients": c.tolist(),
    }


def command_fit(args):
    measured = np.load(args.measurement, allow_pickle=False)
    calibration = np.load(args.calibration, allow_pickle=False)
    report = _projection_report(
        measured, calibration["c_g"], int(calibration["rate_dimension"]),
        args.seed, args.allocation_train, args.image_train, args.ridge)
    c = np.asarray(report["coefficients"])
    bits = np.asarray(calibration["mode_bits"], dtype=int)
    dimension = int(calibration["rate_dimension"])
    target = int(round(float(measured["rates"][0].sum())))
    table = {
        g: {int(b): float(c[g] * np.exp2(-2.0 * b / dimension))
            for b in bits} for g in range(len(c))
    }
    ideal_bits, ideal_value = dp_allocate(table, target, menu=bits.tolist())
    index = {int(bit): i for i, bit in enumerate(bits)}
    artifact = Path(args.output).with_suffix(".npz")
    np.savez_compressed(
        artifact, c_g=c, intercept=report["intercept"],
        coefficient_kind="projected", cost_table=calibration["cost_table"],
        mode_bits=bits, rate_dimension=dimension,
        ideal_bits=ideal_bits,
        ideal_modes=[index[bit] for bit in ideal_bits],
        target_rate=target)
    report.update({
        "measurement": args.measurement, "calibration": args.calibration,
        "coefficient_kind": "projected",
        "coefficient_artifact": str(artifact),
        "ideal_bits": list(ideal_bits), "ideal_value": ideal_value,
    })
    _write(args.output, report)


def _select_allocations(remainder, count, seed, phi=None):
    n = len(remainder)
    count = min(max(4, count), n)
    order = np.argsort(remainder)
    edge = max(2, count // 4)
    chosen = set(order[:edge].tolist() + order[-edge:].tolist())
    if phi is not None and len(chosen) < count:
        chosen.add(int(np.argmin(phi)))
    rng = np.random.default_rng(seed)
    for index in rng.permutation(n):
        chosen.add(int(index))
        if len(chosen) >= count:
            break
    return np.asarray(sorted(chosen), dtype=np.int64)


def _soft_distortions(
    codec, tail, y, mu, std, teacher, allocations, chunk,
    straight_through=False,
):
    """Reuse one differentiable mode bank for all sampled allocations."""
    b, tokens, _ = y.shape
    rotation = codec.transform.get_rotation()
    z = y.reshape(b * tokens, -1) @ rotation
    sub = z.reshape(-1, codec.pq.G, codec.pq.d).permute(1, 0, 2)
    modes = []
    for quantizer in codec.pq.quantizers:
        cost = torch.cdist(sub, quantizer.codebooks).square()
        if quantizer.use_rate:
            log2_p = -torch.log_softmax(
                quantizer.log_prior, -1) / math.log(2.0)
            cost = cost + log2_p[:, None] / quantizer.lmbda
        weights = torch.softmax(-cost / codec.pq.temperature, -1)
        if straight_through:
            labels = cost.argmin(-1, keepdim=True)
            hard = torch.zeros_like(weights).scatter_(-1, labels, 1.0)
            weights = hard - weights.detach() + weights
        modes.append(torch.einsum(
            "gnk,gkd->gnd", weights, quantizer.codebooks
        ).permute(1, 0, 2).reshape(b, tokens, codec.pq.G, codec.pq.d))
    bank = torch.stack(modes).permute(3, 0, 1, 2, 4)
    groups = torch.arange(codec.pq.G, device=y.device)[None]
    values = []
    for start in range(0, len(allocations), chunk):
        modes = torch.as_tensor(
            allocations[start:start + chunk], device=y.device)
        count = len(modes)
        selected = bank[groups, modes].permute(
            0, 2, 3, 1, 4).reshape(count * b, tokens, -1)
        y_hat = selected @ rotation.t()
        x_hat = batch_inv_normalize_gpu(
            y_hat,
            mu.unsqueeze(0).expand(count, *mu.shape).reshape(
                count * b, *mu.shape[1:]),
            std.unsqueeze(0).expand(count, *std.shape).reshape(
                count * b, *std.shape[1:]))
        output = tail(x_hat)
        target = teacher.unsqueeze(0).expand(
            count, -1, *teacher.shape[1:]).reshape(
                count * b, *teacher.shape[1:])
        values.append((output - target).square().reshape(
            count, b, -1).sum(-1).mean(1))
    return torch.cat(values)


def _objective(
    codec, tail, batch, allocations, A, intercept, c, args,
    scale, anchor_scale,
):
    codec.train()
    codec.pq.temperature = args.pq_temperature
    y, mu, std, teacher = batch
    D = _soft_distortions(
        codec, tail, y, mu, std, teacher, allocations, args.allocation_chunk,
        straight_through=args.assignment == "ste")
    e = (D.double() - intercept - A.double() @ c) / scale
    tau = args.lse_temperature
    remainder = (
        tau * torch.logsumexp(e / tau, 0)
        + tau * torch.logsumexp(-e / tau, 0)
        - 2.0 * tau * math.log(len(e)))
    anchor = D.mean() / anchor_scale
    loss = args.remainder_weight * remainder + args.anchor_weight * anchor
    return loss, {
        "remainder": remainder.detach(), "anchor": anchor.detach(),
        "_remainder": remainder, "_anchor": anchor,
        "c_min": c.min().detach(), "c_max": c.max().detach(),
    }


def _term_grad_stats(info, params):
    grads = [
        torch.autograd.grad(
            info[key], params, retain_graph=True, allow_unused=True)
        for key in ("_remainder", "_anchor")]
    norms = [
        torch.sqrt(sum(g.square().sum() for g in row if g is not None))
        for row in grads]
    dot = sum(
        (a * b).sum() for a, b in zip(*grads)
        if a is not None and b is not None)
    return {
        "remainder_grad_norm": float(norms[0]),
        "anchor_grad_norm": float(norms[1]),
        "grad_cosine": float(dot / (norms[0] * norms[1]).clamp_min(1e-12)),
    }


@torch.no_grad()
def _rotation_distance(codec, reference):
    current = codec.transform.get_rotation()
    return float((current - reference).norm() / math.sqrt(current.shape[0]))


def _fit_state(measured, distortion, calibration, args, device, seed):
    A_all = _design(
        measured["rates"], int(calibration["rate_dimension"]))
    intercept, c = _positive_fit(
        A_all, distortion, calibration["c_g"], args.ridge)
    phi = A_all @ c
    remainder = np.asarray(distortion) - intercept - phi
    selected = _select_allocations(
        remainder, args.allocations, seed, phi=phi)
    return (
        selected, measured["allocations"][selected],
        torch.as_tensor(A_all[selected], device=device),
        torch.as_tensor(intercept, device=device),
        torch.as_tensor(c, device=device),
        max(float(np.ptp(remainder[selected])), 1e-12),
        max(float(np.asarray(distortion)[selected].mean()), 1e-12),
    )


@torch.no_grad()
def _hard_distortions(codec, tail, batch, allocations, args):
    codec.train()
    codec.pq.temperature = args.pq_temperature
    return _soft_distortions(
        codec, tail, *batch, allocations, args.allocation_chunk,
        straight_through=True).cpu().numpy()


def _load_problem(args):
    device = torch.device(args.device)
    codec = load_codec_v1(args.codec, device).train()
    tail = build_tail(args.layer, device)
    measured = np.load(args.measurement, allow_pickle=False)
    calibration = np.load(args.calibration, allow_pickle=False)
    features = np.load(args.features, mmap_mode="r")
    teachers = np.load(args.teachers, mmap_mode="r")
    sl = slice(args.image_offset, args.image_offset + args.images)
    x = torch.from_numpy(np.array(features[sl], copy=True)).float().to(device)
    teacher = torch.from_numpy(np.array(teachers[sl], copy=True)).float().to(device)
    y, mu, std = batch_normalize_gpu(x, mode=args.norm_mode)
    batch = (y, mu, std, teacher)
    state = _fit_state(
        measured, _hard_distortions(
            codec, tail, batch, measured["allocations"], args),
        calibration, args, device, args.seed)
    return (codec, tail, batch, measured, calibration, *state)


def _parameters(codec, kind):
    codec.requires_grad_(False)
    params = []
    if kind in ("u", "joint"):
        codec.transform.triu_params.requires_grad_(True)
        params.append(codec.transform.triu_params)
    if kind in ("codebook", "joint"):
        for quantizer in codec.pq.quantizers:
            quantizer.codebooks.requires_grad_(True)
            params.append(quantizer.codebooks)
    return params


def command_gradcheck(args):
    if args.assignment != "soft":
        raise ValueError("finite-difference gradcheck requires soft assignment")
    (codec, tail, batch, _, _, selected, allocations, A, intercept, c, scale,
     anchor_scale) = _load_problem(args)
    params = _parameters(codec, args.parameters)

    def evaluate():
        return _objective(
            codec, tail, batch, allocations, A, intercept, c, args,
            scale, anchor_scale)

    codec.zero_grad(set_to_none=True)
    loss, info = evaluate()
    loss.backward()
    grads = [p.grad.detach().clone() for p in params]
    norm = torch.sqrt(sum(g.square().sum() for g in grads))
    if not torch.isfinite(norm) or norm == 0:
        raise RuntimeError("non-finite or zero gradient")
    originals = [p.detach().clone() for p in params]
    gradient = [g / norm for g in grads]
    coordinate = [torch.zeros_like(g) for g in grads]
    target = max(range(len(grads)), key=lambda i: float(grads[i].abs().max()))
    coordinate[target].view(-1)[grads[target].abs().argmax()] = 1.0
    rows = {}
    try:
        for name, direction in (("coordinate", coordinate), ("gradient", gradient)):
            ad = float(sum((g * d).sum() for g, d in zip(grads, direction)))
            scans = []
            for eps in map(float, args.eps.split(",")):
                with torch.no_grad():
                    for p, base, d in zip(params, originals, direction):
                        p.copy_(base + eps * d)
                    plus = float(evaluate()[0])
                    for p, base, d in zip(params, originals, direction):
                        p.copy_(base - eps * d)
                    minus = float(evaluate()[0])
                fd = (plus - minus) / (2 * eps)
                scans.append({
                    "eps": eps, "autograd": ad, "finite_difference": fd,
                    "relative_error": abs(fd - ad) / max(abs(fd), abs(ad), 1e-12),
                    "sign_match": bool(np.sign(fd) == np.sign(ad)),
                })
            rows[name] = scans
    finally:
        with torch.no_grad():
            for p, base in zip(params, originals):
                p.copy_(base)
    best = min(rows["coordinate"], key=lambda row: row["relative_error"])
    _write(args.output, {
        "parameters": args.parameters, "selected_indices": selected.tolist(),
        "loss": float(loss), "gradient_norm": float(norm),
        "coefficient_min": float(info["c_min"]),
        "pass": bool(best["sign_match"] and best["relative_error"] < 0.05),
        "directions": rows,
    })


def _hard_report(codec, tail, args, measured, calibration):
    features = np.load(args.hard_features, mmap_mode="r")[:args.hard_images]
    teachers = np.load(args.hard_teachers, mmap_mode="r")[:args.hard_images]
    _, arrays = evaluate_fixed_rate_remainder(
        features, teachers, codec, tail, measured["allocations"],
        calibration["cost_table"], calibration["c_g"], args.norm_mode,
        torch.device(args.device), batch_size=args.hard_batch_size,
        allocation_chunk=args.hard_allocation_chunk)
    return _projection_report(
        arrays, calibration["c_g"], int(calibration["rate_dimension"]),
        args.seed, args.allocation_train, args.image_train, args.ridge)


def command_step(args):
    if args.parameters in ("codebook", "joint") and args.assignment != "ste":
        raise ValueError("codebook updates require --assignment ste")
    (codec, tail, batch, measured, calibration, selected, allocations, A,
     intercept, c, scale, anchor_scale) = _load_problem(args)
    params = _parameters(codec, args.parameters)
    before = _hard_report(codec, tail, args, measured, calibration)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    rotation0 = codec.transform.get_rotation().detach().clone()
    history, stages, active_sets = [], [], [selected.tolist()]
    stage_start = None
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss, info = _objective(
            codec, tail, batch, allocations, A, intercept, c, args,
            scale, anchor_scale)
        collect_stats = (
            step == 0 and (
                args.grad_stats_every >= 0
                or args.target_remainder_grad_ratio >= 0)
            or args.grad_stats_every > 0
            and (step + 1) % args.grad_stats_every == 0)
        stats = _term_grad_stats(info, params) if collect_stats else {}
        if step == 0 and args.target_remainder_grad_ratio >= 0:
            if (args.anchor_weight <= 0
                    or not all(math.isfinite(value) and value > 0 for value in (
                        stats["anchor_grad_norm"],
                        stats["remainder_grad_norm"]))):
                raise RuntimeError("cannot calibrate non-positive gradient norms")
            args.remainder_weight = (
                args.target_remainder_grad_ratio * args.anchor_weight
                * stats["anchor_grad_norm"]
                / max(stats["remainder_grad_norm"], 1e-12))
            loss = (
                args.remainder_weight * info["_remainder"]
                + args.anchor_weight * info["_anchor"])
        if stats:
            stats["weighted_grad_ratio"] = (
                args.remainder_weight * stats["remainder_grad_norm"]
                / max(args.anchor_weight * stats["anchor_grad_norm"], 1e-12))
        if stage_start is None:
            stage_start = float(loss.detach())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        row = {
            "step": step + 1, "loss": float(loss.detach()),
            "remainder": float(info["remainder"]),
            "anchor": float(info["anchor"]), "grad_norm": float(grad_norm),
            "coefficient_min": float(info["c_min"]),
            "rotation_distance": _rotation_distance(codec, rotation0),
            "remainder_weight": args.remainder_weight,
        }
        row.update(stats)
        if (args.checkpoint and args.checkpoint_every > 0
                and (step + 1) % args.checkpoint_every == 0):
            path = Path(args.checkpoint)
            snapshot = path.with_name(
                f"{path.stem}.step{step + 1:03d}{path.suffix}")
            save_codec_v1(codec, snapshot)
            row["checkpoint"] = str(snapshot)
        history.append(row)
        refresh = (
            args.refresh_steps > 0 and (step + 1) % args.refresh_steps == 0
            and step + 1 < args.steps)
        if refresh:
            with torch.no_grad():
                stage_end = float(_objective(
                    codec, tail, batch, allocations, A, intercept, c, args,
                    scale, anchor_scale)[0])
                all_D = _hard_distortions(
                    codec, tail, batch, measured["allocations"], args)
            stages.append({"end_step": step + 1, "before": stage_start,
                           "after": stage_end})
            (selected, allocations, A, intercept, c, scale,
             anchor_scale) = _fit_state(
                measured, all_D, calibration, args, A.device,
                args.seed)
            active_sets.append(selected.tolist())
            stage_start = None
    with torch.no_grad():
        final_loss, final_info = _objective(
            codec, tail, batch, allocations, A, intercept, c, args,
            scale, anchor_scale)
    stages.append({"end_step": args.steps, "before": stage_start,
                   "after": float(final_loss)})
    after = _hard_report(codec, tail, args, measured, calibration)
    if args.checkpoint:
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        save_codec_v1(codec, args.checkpoint)
    _write(args.output, {
        "parameters": args.parameters, "steps": args.steps, "lr": args.lr,
        "remainder_weight": args.remainder_weight,
        "assignment": args.assignment, "active_sets": active_sets,
        "protocol": {
            key: getattr(args, key) for key in (
                "seed", "images", "image_offset", "allocations",
                "allocation_chunk", "hard_images", "hard_batch_size",
                "hard_allocation_chunk", "allocation_train", "image_train",
                "steps", "refresh_steps", "lr", "ridge", "pq_temperature",
                "lse_temperature", "remainder_weight", "anchor_weight")
        },
        "before": before, "after": after,
        "soft_before": stages[0]["before"], "soft_after": float(final_loss),
        "soft_after_remainder": float(final_info["remainder"]),
        "soft_decreased": bool(all(s["after"] < s["before"] for s in stages)),
        "hard_projected_decreased": bool(
            after["omega_projected_heldout"]
            < before["omega_projected_heldout"]),
        "stages": stages, "history": history, "checkpoint": args.checkpoint,
    })


def command_compare(args):
    if len(args.inputs) != len(args.names):
        raise ValueError("--inputs and --names must have equal length")
    rows, errors, reference = [], [], None
    for name, path in zip(args.names, args.inputs):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        before, after = data["before"], data["after"]
        initial = data["active_sets"][0]
        signature = (data.get("protocol"), initial)
        if reference is None:
            reference = signature
        elif signature != reference:
            errors.append(f"protocol or initial active set mismatch: {name}")
        omega0, omega1 = (
            before["omega_projected_heldout"],
            after["omega_projected_heldout"])
        d0, d1 = before["mean_distortion_heldout"], after["mean_distortion_heldout"]
        rows.append({
            "name": name, "omega_before": omega0, "omega_after": omega1,
            "omega_change_pct": 100.0 * (omega1 / omega0 - 1.0),
            "distortion_change_pct": 100.0 * (d1 / d0 - 1.0),
            "all_stages_decreased": data["soft_decreased"],
            "active_generations": len(data["active_sets"]),
        })
    baseline = [row["omega_before"] for row in rows]
    spread = np.ptp(baseline) / max(np.mean(baseline), 1e-12)
    if spread > args.baseline_tolerance:
        errors.append(f"baseline omega relative spread {spread:.3g}")
    rows.sort(key=lambda row: row["omega_change_pct"])
    _write(args.output, {
        "passed": not errors, "errors": errors,
        "baseline_relative_spread": float(spread), "ranking": rows,
    })
    if errors:
        raise RuntimeError("; ".join(errors))


def command_manifest(args):
    source = np.load(args.source, allow_pickle=False)
    calibration = np.load(args.calibration, allow_pickle=False)
    costs = calibration["cost_table"]
    _, _, target = validate_fixed_total_rate(source["allocations"], costs)
    excluded = set()
    for path in args.exclude:
        excluded.update(map(tuple, np.load(
            path, allow_pickle=False)["allocations"]))
    uniform = [
        row for row in source["allocations"] if np.all(row == row[0])]
    if uniform:
        base = uniform[0]
    else:
        lookup = {int(bit): i for i, bit in enumerate(
            calibration["mode_bits"])}
        base = np.asarray([lookup[int(bit)]
                           for bit in calibration["ideal_bits"]])
    need = args.samples + len(excluded) + 32
    pool = random_exchange_walk(
        base, costs, need, max_steps=args.max_steps, seed=args.seed)
    fresh = np.asarray(
        [row for row in pool if tuple(row) not in excluded], dtype=np.int64)
    if len(fresh) < args.samples:
        raise RuntimeError("could not generate enough disjoint allocations")
    fresh = fresh[np.random.default_rng(args.seed).permutation(
        len(fresh))[:args.samples]]
    rates, totals, checked = validate_fixed_total_rate(fresh, costs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, allocations=fresh, rates=rates, total_rates=totals)
    _write(output.with_suffix(".json"), {
        "source": args.source, "calibration": args.calibration,
        "exclude": args.exclude, "seed": args.seed,
        "n_allocations": len(fresh), "excluded_count": len(excluded),
        "target_rate": checked, "max_rate_error": float(
            np.abs(totals - target).max()), "overlap": 0,
    })


def command_arrays(args):
    device = torch.device(args.device)
    codec = load_codec_v1(args.codec, device)
    tail = build_tail(args.layer, device)
    measured = np.load(args.measurement, allow_pickle=False)
    calibration = np.load(args.calibration, allow_pickle=False)
    coefficient_path = args.coefficients or args.calibration
    coefficients = np.load(coefficient_path, allow_pickle=False)
    if "cost_table" in coefficients and not np.allclose(
            coefficients["cost_table"], calibration["cost_table"]):
        raise ValueError("coefficient and calibration cost tables differ")
    kind = (
        str(coefficients["coefficient_kind"].item())
        if "coefficient_kind" in coefficients else "legacy")
    sl = slice(args.image_offset, args.image_offset + args.images)
    features = np.load(args.features, mmap_mode="r")[sl]
    teachers = np.load(args.teachers, mmap_mode="r")[sl]
    if len(features) != args.images or len(teachers) != args.images:
        raise ValueError("evaluation image slice is incomplete")
    summary, arrays = evaluate_fixed_rate_remainder(
        features, teachers, codec, tail, measured["allocations"],
        calibration["cost_table"], coefficients["c_g"], args.norm_mode,
        device, batch_size=args.batch_size,
        allocation_chunk=args.allocation_chunk)
    arrays["c_g"] = np.asarray(coefficients["c_g"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    summary["mean_distortion"] = float(arrays["distortion_mean"].mean())
    if args.reference_codec:
        reference = load_codec_v1(args.reference_codec, device)
        summary["rotation_distance"] = _rotation_distance(
            codec, reference.transform.get_rotation())
    summary.update({
        "codec": args.codec, "measurement": args.measurement,
        "calibration": args.calibration, "features": args.features,
        "coefficients": coefficient_path, "coefficient_kind": kind,
        "teachers": args.teachers, "image_offset": args.image_offset,
        "reference_codec": args.reference_codec,
    })
    _write(output.with_suffix(".json"), summary)


def command_match(args):
    def read(paths):
        rows = []
        for path in paths:
            row = json.loads(Path(path).read_text(encoding="utf-8"))
            rows.append({
                "summary": path, "codec": row["codec"],
                "mean_distortion": row["mean_distortion"],
                "omega": row["omega_sampled"],
                "rotation_distance": row["rotation_distance"],
            })
        return rows

    pairs = []
    for mean in read(args.mean_inputs):
        for joint in read(args.joint_inputs):
            d_scale = max(
                (mean["mean_distortion"] + joint["mean_distortion"]) / 2, 1e-12)
            u_scale = max(
                (mean["rotation_distance"] + joint["rotation_distance"]) / 2,
                1e-12)
            pairs.append({
                "mean": mean, "joint": joint,
                "mean_relative_gap": abs(
                    mean["mean_distortion"] - joint["mean_distortion"]) / d_scale,
                "rotation_relative_gap": abs(
                    mean["rotation_distance"] - joint["rotation_distance"])
                / u_scale,
                "omega_delta_joint_minus_mean":
                    joint["omega"] - mean["omega"],
            })
    mean = sorted(pairs, key=lambda row: row["mean_relative_gap"])[:args.top]
    rotation = sorted(
        pairs, key=lambda row: row["rotation_relative_gap"])[:args.top]
    _write(args.output, {
        "mean_match_pass": mean[0]["mean_relative_gap"] <= args.mean_tolerance,
        "rotation_match_pass":
            rotation[0]["rotation_relative_gap"] <= args.rotation_tolerance,
        "matched_mean_distortion": mean,
        "matched_rotation_distance": rotation,
    })


def command_paired(args):
    if len(args.inputs) != len(args.names) or len(args.inputs) < 2:
        raise ValueError("paired CI needs matching baseline and variant inputs")
    arrays = [np.load(path, allow_pickle=False) for path in args.inputs]
    reference = arrays[0]
    for data in arrays[1:]:
        for key in ("allocations", "rates"):
            if not np.array_equal(data[key], reference[key]):
                raise ValueError(f"paired input mismatch: {key}")
        if data["distortion_per_image"].shape != reference[
                "distortion_per_image"].shape:
            raise ValueError("paired input distortion shape mismatch")
    if args.refit_c and not args.calibration:
        raise ValueError("--calibration is required with --refit-c")
    calibration = (
        np.load(args.calibration, allow_pickle=False)
        if args.refit_c else None)
    A = (
        _design(reference["rates"], int(calibration["rate_dimension"]))
        if args.refit_c else None)
    D = [data["distortion_per_image"] for data in arrays]
    rng = np.random.default_rng(args.seed)
    images = np.arange(D[0].shape[1])
    if args.refit_c:
        alloc = rng.permutation(len(A))
        images = rng.permutation(len(images))
        na = min(max(33, int(len(A) * args.allocation_train)), len(A) - 1)
        ni = min(max(1, int(len(images) * args.image_train)), len(images) - 1)
        fit_a, test_a = alloc[:na], alloc[na:]
        fit_i, test_i = images[:ni], images[ni:]
    else:
        fit_a = fit_i = np.asarray([], dtype=int)
        test_a, test_i = np.arange(len(reference["allocations"])), images

    def statistic(index, train_images, test_images):
        values = D[index]
        heldout = values[np.ix_(test_a, test_images)].mean(1)
        if args.refit_c:
            y = values[np.ix_(fit_a, train_images)].mean(1)
            intercept, c = _positive_fit(
                A[fit_a], y, calibration["c_g"], args.ridge)
            remainder = heldout - intercept - A[test_a] @ c
        else:
            remainder = heldout - arrays[index]["phi"][test_a]
        return np.ptp(remainder), heldout.mean()

    point = np.asarray([
        statistic(index, fit_i, test_i) for index in range(len(D))])
    samples = np.empty((args.repeats, len(D), 2))
    for repeat in range(args.repeats):
        train = (
            rng.choice(fit_i, len(fit_i), replace=True)
            if args.refit_c else fit_i)
        test = rng.choice(test_i, len(test_i), replace=True)
        samples[repeat] = [
            statistic(index, train, test) for index in range(len(D))]
    rows = []
    for index in range(1, len(D)):
        delta = samples[:, index] - samples[:, 0]
        rows.append({
            "name": args.names[index],
            "omega": float(point[index, 0]),
            "omega_delta": float(point[index, 0] - point[0, 0]),
            "omega_delta_ci95": np.percentile(
                delta[:, 0], [2.5, 97.5]).tolist(),
            "distortion_delta": float(point[index, 1] - point[0, 1]),
            "distortion_delta_ci95": np.percentile(
                delta[:, 1], [2.5, 97.5]).tolist(),
            "bootstrap_probability_omega_lower": float(
                np.mean(delta[:, 0] < 0)),
        })
    _write(args.output, {
        "baseline": args.names[0], "baseline_omega": float(point[0, 0]),
        "coefficient_policy": "refit" if args.refit_c else "frozen",
        "repeats": args.repeats, "seed": args.seed,
        "fit_allocations": len(fit_a), "test_allocations": len(test_a),
        "fit_images": len(fit_i), "test_images": len(test_i),
        "comparisons": rows,
    })


def _shared(parser):
    parser.add_argument("--codec", required=True)
    parser.add_argument("--measurement", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--teachers", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parameters", choices=("u", "codebook", "joint"),
                        default="u")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--norm-mode", default="per_image")
    parser.add_argument("--image-offset", type=int, default=0)
    parser.add_argument("--images", type=int, default=1)
    parser.add_argument("--allocations", type=int, default=40)
    parser.add_argument("--allocation-chunk", type=int, default=4)
    parser.add_argument("--pq-temperature", type=float, default=0.01)
    parser.add_argument("--assignment", choices=("soft", "ste"), default="soft")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--lse-temperature", type=float, default=0.1)
    parser.add_argument("--remainder-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)


def parser():
    main = argparse.ArgumentParser()
    sub = main.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--measurement", required=True)
    fit.add_argument("--calibration", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--allocation-train", type=float, default=0.7)
    fit.add_argument("--image-train", type=float, default=0.5)
    fit.add_argument("--ridge", type=float, default=1e-3)
    fit.add_argument("--seed", type=int, default=42)
    grad = sub.add_parser("gradcheck")
    _shared(grad)
    grad.add_argument("--eps", default="0.00001,0.00003,0.0001,0.0003")
    step = sub.add_parser("step")
    _shared(step)
    step.add_argument("--hard-features", required=True)
    step.add_argument("--hard-teachers", required=True)
    step.add_argument("--hard-images", type=int, default=32)
    step.add_argument("--hard-batch-size", type=int, default=4)
    step.add_argument("--hard-allocation-chunk", type=int, default=8)
    step.add_argument("--allocation-train", type=float, default=0.7)
    step.add_argument("--image-train", type=float, default=0.5)
    step.add_argument("--steps", type=int, default=10)
    step.add_argument("--refresh-steps", type=int, default=5)
    step.add_argument("--lr", type=float, default=1e-5)
    step.add_argument("--grad-clip", type=float, default=1.0)
    step.add_argument("--checkpoint", default="")
    step.add_argument("--checkpoint-every", type=int, default=0)
    step.add_argument("--grad-stats-every", type=int, default=-1)
    step.add_argument(
        "--target-remainder-grad-ratio", type=float, default=-1.0)
    compare = sub.add_parser("compare")
    compare.add_argument("--inputs", nargs="+", required=True)
    compare.add_argument("--names", nargs="+", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--baseline-tolerance", type=float, default=1e-5)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--source", required=True)
    manifest.add_argument("--calibration", required=True)
    manifest.add_argument("--exclude", nargs="+", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--samples", type=int, default=261)
    manifest.add_argument("--max-steps", type=int, default=32)
    manifest.add_argument("--seed", type=int, default=1042)
    arrays = sub.add_parser("arrays")
    for name in ("codec", "measurement", "calibration", "features",
                 "teachers", "output"):
        arrays.add_argument(f"--{name}", required=True)
    arrays.add_argument("--coefficients", default="")
    arrays.add_argument("--device", default="cuda")
    arrays.add_argument("--layer", type=int, default=20)
    arrays.add_argument("--norm-mode", default="per_image")
    arrays.add_argument("--image-offset", type=int, default=0)
    arrays.add_argument("--images", type=int, default=128)
    arrays.add_argument("--batch-size", type=int, default=4)
    arrays.add_argument("--allocation-chunk", type=int, default=8)
    arrays.add_argument("--reference-codec", default="")
    match = sub.add_parser("match")
    match.add_argument("--mean-inputs", nargs="+", required=True)
    match.add_argument("--joint-inputs", nargs="+", required=True)
    match.add_argument("--output", required=True)
    match.add_argument("--top", type=int, default=5)
    match.add_argument("--mean-tolerance", type=float, default=1e-3)
    match.add_argument("--rotation-tolerance", type=float, default=2e-2)
    paired = sub.add_parser("paired")
    paired.add_argument("--inputs", nargs="+", required=True)
    paired.add_argument("--names", nargs="+", required=True)
    paired.add_argument("--calibration", default="")
    paired.add_argument("--refit-c", action="store_true")
    paired.add_argument("--output", required=True)
    paired.add_argument("--repeats", type=int, default=1000)
    paired.add_argument("--allocation-train", type=float, default=0.7)
    paired.add_argument("--image-train", type=float, default=0.5)
    paired.add_argument("--ridge", type=float, default=1e-3)
    paired.add_argument("--seed", type=int, default=42)
    return main


if __name__ == "__main__":
    args = parser().parse_args()
    globals()[f"command_{args.command}"](args)
