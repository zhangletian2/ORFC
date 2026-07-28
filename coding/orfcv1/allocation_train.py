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


def _sample(features, teachers, count, rng, device, norm_mode):
    ids = rng.choice(len(features), min(count, len(features)), replace=False)
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
def _seed_from_anchor(codec, anchor):
    """Make lower modes nested subsets and higher modes exact refinements."""
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
    for mode, quantizer in enumerate(codec.pq.quantizers):
        if mode == anchor:
            continue
        target = quantizer.codebooks
        if target.shape[1] < size:
            indices = order[:, :target.shape[1], None].expand(
                -1, -1, dimension)
            target.copy_(torch.gather(base, 1, indices))
        else:
            repeat = math.ceil(target.shape[1] / size)
            target.copy_(base.repeat(1, repeat, 1)[:, :target.shape[1]])


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
    _seed_from_anchor(codec, anchor)
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
        seeded_report["max_relative_violation"],
        float(np.mean(np.delete(seeded, anchor))))
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
                report["violations"], report["max_relative_violation"],
                float(np.mean(np.delete(curve, anchor))))
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
    ideal = int(phi.argmin())
    order = np.argsort(remainder)
    edge = max(2, args.allocations // 4)
    selected = set(order[:edge]) | set(order[-edge:]) | {ideal}
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
        "ideal": ideal, "ideal_local": local[ideal],
        "reference": reference,
        "reference_local": local.get(reference),
        "omega_scale": max(float(np.ptp(remainder[selected])), 1.0),
        "distortion_scale": max(float(distortion[ideal]), 1.0),
    }


def _loss(codec, tail, batch, state, args, remainder_weight):
    D = _distortions(
        codec, tail, batch, state["allocations"], args.allocation_chunk,
        args.pq_temperature, True)
    A = torch.as_tensor(state["A"], device=D.device)
    c = torch.as_tensor(state["c"], device=D.device)
    e = (D.double() - state["intercept"] - A.double() @ c)
    e = e / state["omega_scale"]
    tau = args.lse_temperature
    omega = (
        tau * torch.logsumexp(e / tau, 0)
        + tau * torch.logsumexp(-e / tau, 0)
        - 2 * tau * math.log(len(e)))
    candidate = D[state["ideal_local"]] / state["distortion_scale"]
    topk = torch.topk(
        D, min(args.topk, len(D))).values.mean() / state["distortion_scale"]
    reference = (
        D[state["reference_local"]] / state["distortion_scale"]
        if state["reference_local"] is not None else D.new_zeros(()))
    base = (
        args.candidate_weight * candidate + args.topk_weight * topk
        + args.reference_weight * reference)
    return base + remainder_weight * omega, {
        "base": base, "omega": omega, "candidate": candidate,
        "topk": topk, "reference": reference,
    }


def _joint_parameters(codec, anchor):
    codec.requires_grad_(False)
    codec.transform.triu_params.requires_grad_(True)
    params = [codec.transform.triu_params]
    for mode, quantizer in enumerate(codec.pq.quantizers):
        if mode != anchor:
            quantizer.codebooks.requires_grad_(True)
            params.append(quantizer.codebooks)
    return params


def _report(source, distortion, state, dimension):
    phi = state["intercept"] + _design(source["rates"], dimension) @ state["c"]
    remainder = distortion - phi
    gap = np.sort(phi)[1] - phi.min()
    return {
        "mean_distortion": float(distortion.mean()),
        "ideal_index": int(state["ideal"]),
        "ideal_bits": source["rates"][state["ideal"]].tolist(),
        "ideal_distortion": float(distortion[state["ideal"]]),
        "numeric_best_index": int(distortion.argmin()),
        "numeric_best_distortion": float(distortion.min()),
        "reference_distortion": (
            float(distortion[state["reference"]])
            if state["reference"] is not None else None),
        "omega": float(np.ptp(remainder)), "phi_gap": float(gap),
        "candidate_count": int(np.count_nonzero(
            phi - phi.min() <= np.ptp(remainder) + 1e-12)),
    }


def command_short(args):
    device, codec, tail, train_x, train_y, val_x, val_y = _base(args)
    source = np.load(args.allocation_file, allow_pickle=False)
    calibration = np.load(args.calibration, allow_pickle=False)
    bits = np.log2(codec.pq.mode_sizes).astype(int)
    anchor = int(np.flatnonzero(bits == args.anchor_bit)[0])
    params = _joint_parameters(codec, anchor)
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
    remainder_weight, history, active = args.remainder_weight, [], []
    for step in range(args.steps):
        batch = _sample(
            train_x, train_y, args.images, rng, device, args.norm_mode)
        optimizer.zero_grad(set_to_none=True)
        _, terms = _loss(codec, tail, batch, state, args, 0.0)
        if step == 0 and args.remainder_grad_ratio >= 0:
            base_grad = torch.autograd.grad(
                terms["base"], params, retain_graph=True, allow_unused=True)
            omega_grad = torch.autograd.grad(
                terms["omega"], params, retain_graph=True, allow_unused=True)
            base_norm = torch.sqrt(sum(
                value.square().sum() for value in base_grad
                if value is not None))
            omega_norm = torch.sqrt(sum(
                value.square().sum() for value in omega_grad
                if value is not None))
            remainder_weight = (
                args.remainder_grad_ratio * float(base_norm)
                / max(float(omega_norm), 1e-12))
        loss, terms = _loss(
            codec, tail, batch, state, args, remainder_weight)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        optimizer.step()
        history.append({
            "step": step + 1, "loss": float(loss),
            "base": float(terms["base"]), "omega": float(terms["omega"]),
            "candidate": float(terms["candidate"]),
            "reference": float(terms["reference"]),
            "grad_norm": float(grad),
            "remainder_weight": remainder_weight,
            "ideal_index": int(state["ideal"]),
        })
        if (
            args.refresh_steps > 0 and (step + 1) % args.refresh_steps == 0
            and step + 1 < args.steps
        ):
            current_D = _hard_eval(
                codec, tail, refresh_x, refresh_y,
                all_allocations, refresh_args)
            state = _select_state(source, current_D, calibration, args)
            active.append(int(state["ideal"]))
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
    target = int(state["ideal"])
    paired = after_matrix[target] - before_matrix[target]
    half = 1.96 * paired.std(ddof=1) / np.sqrt(len(paired))
    array_path = Path(args.output).with_suffix(".npz")
    np.savez_compressed(
        array_path, allocations=all_allocations,
        before_per_image=before_matrix, after_per_image=after_matrix,
        final_ideal_index=target)
    _write(args.output, {
        "before": before, "after": after, "menu": curve_report,
        "history": history, "active_ideal_indices": active,
        "remainder_weight": remainder_weight,
        "remainder_grad_ratio": args.remainder_grad_ratio,
        "final_candidate_paired_delta": float(paired.mean()),
        "final_candidate_paired_delta_ci95": [
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
    short = sub.add_parser("short")
    _common(short)
    short.add_argument("--allocation-file", required=True)
    short.add_argument("--calibration", required=True)
    short.add_argument("--refresh-images", type=int, default=8)
    short.add_argument("--refresh-steps", type=int, default=5)
    short.add_argument("--allocations", type=int, default=64)
    short.add_argument("--ridge", type=float, default=1e-3)
    short.add_argument("--lse-temperature", type=float, default=0.1)
    short.add_argument("--candidate-weight", type=float, default=1.0)
    short.add_argument("--topk-weight", type=float, default=0.25)
    short.add_argument("--topk", type=int, default=8)
    short.add_argument("--reference-bit", type=int, default=6)
    short.add_argument("--reference-weight", type=float, default=1.0)
    short.add_argument("--remainder-weight", type=float, default=0.0)
    short.add_argument("--remainder-grad-ratio", type=float, default=-1.0)
    return main


if __name__ == "__main__":
    args = parser().parse_args()
    globals()[f"command_{args.command}"](args)
