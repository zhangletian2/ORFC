#!/usr/bin/env python3
"""Fixed-total-rate P1 experiment; one process handles one representation."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ORFC = HERE.parent / "orfc"
THEORY = HERE.parents[2] / "theory_verify"
sys.path[:0] = [str(HERE), str(ORFC), str(THEORY)]

from codec_v1 import (
    DirectOrthogonalTransform, FeatureCodecV1, load_codec_v1, save_codec_v1,
)
from fixed_rate_remainder import (
    adjacent_exchange_allocations,
    evaluate_fixed_rate_decomposition,
    evaluate_fixed_rate_remainder,
    nominal_cost_table,
    random_exchange_walk,
    validate_fixed_total_rate,
)
from multimode_pq import MultiModeSoftPQ
from opq import batch_normalize_gpu, learn_opq_rotation
def csv_ints(value):
    return tuple(int(x) for x in value.split(","))


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def build_tail(layer, device):
    from backbone.wrapper import Dinov2Wrapper
    from soft_pq import FrozenTail

    wrapper = Dinov2Wrapper(
        head_layers=1, model_name="dinov2_vitl14", device=device)
    blocks = list(wrapper.backbone.blocks)
    for block in blocks[:layer + 1]:
        block.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()
    return FrozenTail(
        blocks[layer + 1:], wrapper.backbone.norm, device=device)


@torch.no_grad()
def command_prepare(args):
    output = Path(args.output)
    if output.exists() and not args.force:
        print(f"reuse {output}")
        return
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    features = np.load(args.features, mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    ids = rng.choice(
        len(features), min(args.images, len(features)), replace=False)
    vectors = []
    for start in range(0, len(ids), args.batch_size):
        x = torch.from_numpy(
            np.asarray(features[ids[start:start + args.batch_size]])).float().to(device)
        y, _, _ = batch_normalize_gpu(x, mode=args.norm_mode)
        vectors.append(y.reshape(-1, y.shape[-1]).cpu())
    y = torch.cat(vectors)
    if len(y) > args.max_vectors:
        keep = torch.randperm(len(y), generator=torch.Generator().manual_seed(
            args.seed))[:args.max_vectors]
        y = y[keep]
    sizes = [2**bits for bits in csv_ints(args.mode_bits)]
    transform = DirectOrthogonalTransform(features.shape[-1]).to(device)
    history = []
    if args.source_kind == "opq":
        rotation, _, history = learn_opq_rotation(
            y, args.groups, features.shape[-1] // args.groups,
            2**args.opq_bits, max_iter_opq=args.opq_iters,
            max_iter_kmeans=args.kmeans_iters, device=device, verbose=True)
        transform.init_from_opq(rotation)
    transform = transform.eval()
    transform.requires_grad_(False)
    rotation = transform.get_rotation().detach()
    z = y.to(device) @ rotation
    pq = MultiModeSoftPQ(args.groups, sizes, features.shape[-1] // args.groups)
    pq.init_from_kmeans(
        z, device=device, max_iter=args.kmeans_iters, seed=args.seed)
    probe = z[:min(len(z), 512)].reshape(-1, pq.G, pq.d).permute(1, 0, 2)
    aligned_mse = [
        float(torch.cdist(probe, q.codebooks.to(device)).square().amin(2).mean())
        for q in pq.quantizers]
    codec = FeatureCodecV1(pq.to(device), transform)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_codec_v1(codec, output)
    dump_json(output.with_suffix(".json"), {
        "arm": args.arm, "mode_bits": list(csv_ints(args.mode_bits)),
        "training_images": int(len(ids)), "vectors": int(len(z)),
        "kmeans_iters": args.kmeans_iters, "seed": args.seed,
        "source_kind": args.source_kind, "opq_bits": args.opq_bits,
        "opq_iters": args.opq_iters, "opq_history": [
            [mse, delta if np.isfinite(delta) else None]
            for mse, delta in history],
        "rotation_orth_error": transform.orth_error(),
        "codebook_coordinates": "normalised_features@effective_rotation",
        "initial_rotated_mse": aligned_mse,
    })
    print(f"saved {output}")


@torch.no_grad()
def estimate_c(codec, tail, features, mode, bit, args, device):
    q_rows, n = [], min(args.images, len(features))
    rotation_t = codec.transform.get_rotation().detach().t()
    groups, d = codec.pq.G, codec.pq.d
    modes = torch.full((groups,), mode, dtype=torch.long, device=device)
    for start in range(0, n, args.batch_size):
        x = torch.from_numpy(
            np.array(features[start:min(start + args.batch_size, n)], copy=True)
        ).float().to(device)
        y, _, std = batch_normalize_gpu(x, mode=args.norm_mode)
        _, info = codec(
            y, return_details=True, detail_level="train", modes=modes)
        batch, tokens = x.shape[:2]
        residual = info["r_g"].reshape(
            groups, batch, tokens, d)
        q_batch = torch.empty(batch, groups, device=device)
        for first in range(0, groups, args.group_chunk):
            last = min(first + args.group_chunk, groups)
            blocks = torch.stack([
                rotation_t[g * d:(g + 1) * d] for g in range(first, last)])
            error = torch.einsum(
                "cbtd,cde->cbte", residual[first:last], blocks)
            error = error * std.unsqueeze(0)
            magnitude = error.flatten(2).norm(dim=2).clamp_min(1e-12)
            direction = error / magnitude[:, :, None, None]
            base = x.unsqueeze(0).expand(last - first, *x.shape)
            plus_flat = tail.forward_nograd(
                (base + args.eps * direction).reshape(
                    -1, tokens, x.shape[-1]))
            plus = plus_flat.reshape(
                last - first, batch, *plus_flat.shape[1:])
            minus_flat = tail.forward_nograd(
                (base - args.eps * direction).reshape(
                    -1, tokens, x.shape[-1]))
            minus = minus_flat.reshape(plus.shape)
            delta = (plus - minus) / (2.0 * args.eps)
            q_batch[:, first:last] = (
                delta.flatten(2).square().sum(2)
                * magnitude.square()).t()
        q_rows.append(q_batch.cpu().numpy())
    q = np.concatenate(q_rows)
    return np.exp2(2.0 * bit / codec.pq.d) * q.mean(0), q


def topk_cost_allocate(ideal_cost_table, mode_bits, budget, k):
    """Exact top-k allocations for any separable fixed-budget mode table."""
    ideal_cost_table = np.asarray(ideal_cost_table, dtype=np.float64)
    if ideal_cost_table.ndim != 2 or ideal_cost_table.shape[1] != len(mode_bits):
        raise ValueError("ideal cost table must have shape [groups, modes]")
    if k < 1:
        raise ValueError("k must be positive")
    states = {0: [(0.0, ())]}
    for costs in ideal_cost_table:
        next_states = {}
        for used, rows in states.items():
            for value, path in rows:
                for mode, bit in enumerate(mode_bits):
                    total = used + bit
                    if total <= budget:
                        next_states.setdefault(total, []).append((
                            value + costs[mode], path + (bit,)))
        states = {}
        for used, rows in next_states.items():
            unique = []
            for row in sorted(rows, key=lambda item: (item[0], item[1])):
                if all(row[1] != kept[1] for kept in unique):
                    unique.append(row)
                if len(unique) == k:
                    break
            states[used] = unique
    rows = states.get(budget, [])
    if not rows:
        raise ValueError(f"budget {budget} is infeasible")
    return {
        "values": np.asarray([row[0] for row in rows], dtype=np.float64),
        "bits": np.asarray([row[1] for row in rows], dtype=np.int64),
    }


def top2_cost_allocate(ideal_cost_table, mode_bits, budget):
    """Exact best and runner-up allocations for any separable mode table."""
    top = topk_cost_allocate(ideal_cost_table, mode_bits, budget, 2)
    rows = list(zip(top["values"], top["bits"]))
    second = rows[1] if len(rows) > 1 else (float("inf"), ())
    return {
        "ideal_value": float(rows[0][0]),
        "ideal_bits": np.asarray(rows[0][1], dtype=np.int64),
        "second_value": float(second[0]),
        "second_bits": np.asarray(second[1], dtype=np.int64),
        "ideal_gap": float(second[0] - rows[0][0]),
    }


def top2_allocate(c_g, mode_bits, budget, dimension):
    table = np.stack([
        np.asarray(c_g) * np.exp2(-2.0 * bit / dimension)
        for bit in mode_bits], axis=1)
    return top2_cost_allocate(table, mode_bits, budget)


def make_allocations(
    c_g, mode_bits, budget, n_single, n_random, seed, dimension,
    ideal_cost_table=None,
):
    optimum = (
        top2_allocate(c_g, mode_bits, budget, dimension)
        if ideal_cost_table is None else
        top2_cost_allocate(ideal_cost_table, mode_bits, budget))
    ideal_bits = optimum["ideal_bits"]
    index = {bits: i for i, bits in enumerate(mode_bits)}
    base = np.asarray([index[b] for b in ideal_bits], dtype=np.int64)
    second = np.asarray(
        [index[b] for b in optimum["second_bits"]], dtype=np.int64)
    costs = np.broadcast_to(
        np.asarray(mode_bits, dtype=np.float64)[None],
        (len(c_g), len(mode_bits))).copy()
    neighbours = adjacent_exchange_allocations(base, costs)
    rng = np.random.default_rng(seed)
    others = neighbours[(neighbours != base).any(axis=1)]
    if len(others) > n_single:
        others = others[rng.choice(len(others), n_single, replace=False)]
    randoms = random_exchange_walk(
        base, costs, n_random + 1, seed=seed)
    rows = {tuple(base)}
    if second.size == base.size:
        rows.add(tuple(second))
    rows.update(map(tuple, others))
    rows.update(map(tuple, randoms))
    if budget % len(c_g) == 0 and budget // len(c_g) in index:
        rows.add(tuple([index[budget // len(c_g)]] * len(c_g)))
    allocations = np.asarray(sorted(rows), dtype=np.int64)
    validate_fixed_total_rate(allocations, costs)
    return allocations, np.asarray(ideal_bits), costs


@torch.no_grad()
def command_calibrate(args):
    device = torch.device("cuda")
    codec = load_codec_v1(args.codec, device=device)
    tail = build_tail(args.layer, device)
    source = np.load(args.features, mmap_mode="r")
    end = min(args.image_offset + args.images, len(source))
    features = source[args.image_offset:end]
    if len(features) != args.images:
        raise ValueError("calibration image slice is incomplete")
    mode_bits = csv_ints(args.mode_bits)
    budgets, refs = csv_ints(args.budgets), csv_ints(args.reference_bits)
    if len(budgets) != len(refs):
        raise ValueError("budgets and reference-bits must have equal length")
    requested = csv_ints(args.coefficient_bits) if args.coefficient_bits else ()
    probe_bits = tuple(sorted(set(refs) | set(requested)))
    if any(bit not in mode_bits for bit in (*refs, *probe_bits)):
        raise ValueError("reference/coefficient bits must belong to mode-bits")
    estimates = {
        bit: estimate_c(
            codec, tail, features, mode_bits.index(bit), bit, args, device)
        for bit in probe_bits
    }
    c_by_bit = np.stack([estimates[bit][0] for bit in probe_bits])
    q_by_bit = np.stack([estimates[bit][1] for bit in probe_bits])
    mean = c_by_bit.mean(0).clip(1e-12)
    rate_cv = c_by_bit.std(0) / mean
    rate_span = c_by_bit.max(0) / c_by_bit.min(0).clip(1e-12)
    stability = {
        f"{name}_{stat}": float(function(values))
        for name, values in (("rate_cv", rate_cv), ("rate_span", rate_span))
        for stat, function in (
            ("median", np.median), ("p95", lambda x: np.quantile(x, .95)),
            ("max", np.max))
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for budget, ref in zip(budgets, refs):
        c_g, q = estimates[ref]
        if args.ideal_model == "discrete_jvp":
            if set(probe_bits) != set(mode_bits):
                raise ValueError(
                    "discrete_jvp requires every mode in coefficient-bits")
            ideal_cost_table = np.stack([
                estimates[bit][1].mean(0) for bit in mode_bits], axis=1)
        else:
            ideal_cost_table = np.stack([
                c_g * np.exp2(-2.0 * bit / codec.pq.d)
                for bit in mode_bits], axis=1)
        allocations, ideal_bits, costs = make_allocations(
            c_g, mode_bits, budget, args.single, args.random, args.seed,
            codec.pq.d, ideal_cost_table)
        optimum = top2_cost_allocate(
            ideal_cost_table, mode_bits, budget)
        if not np.array_equal(ideal_bits, optimum["ideal_bits"]):
            raise RuntimeError("allocation generator and exact DP disagree")
        np.savez_compressed(
            out / f"calibration_R{budget}.npz", c_g=c_g, q_per_image=q,
            allocations=allocations, ideal_bits=ideal_bits,
            ideal_value=optimum["ideal_value"],
            second_bits=optimum["second_bits"],
            second_value=optimum["second_value"],
            ideal_gap=optimum["ideal_gap"],
            ideal_cost_table=ideal_cost_table,
            ideal_model=args.ideal_model,
            coefficient_bits=probe_bits, c_by_bit=c_by_bit,
            q_per_image_by_bit=q_by_bit,
            cost_table=costs, mode_bits=mode_bits,
            rate_dimension=codec.pq.d,
            reference_bit=ref,
            coefficient_kind="central_jvp_actual_residual",
            image_offset=args.image_offset)
        dump_json(out / f"calibration_R{budget}.json", {
            "arm": args.arm, "budget": budget, "reference_bit": ref,
            "n_images": int(len(q)), "n_allocations": int(len(allocations)),
            "c_min": float(c_g.min()), "c_max": float(c_g.max()),
            "ideal_bits": ideal_bits.tolist(),
            "ideal_value": optimum["ideal_value"],
            "second_bits": optimum["second_bits"].tolist(),
            "second_value": optimum["second_value"],
            "ideal_gap": optimum["ideal_gap"],
            "coefficient_bits": list(probe_bits), **stability,
            "ideal_model": args.ideal_model,
            "rate_dimension": codec.pq.d,
            "reference_bit": ref,
            "coefficient_kind": "central_jvp_actual_residual",
            "image_offset": args.image_offset, "codec": args.codec,
        })
        print(f"R={budget}: {len(allocations)} allocations")


def command_reallocate(args):
    """Reuse saved JVP rows and build one common allocation set."""
    source, output = Path(args.source_run), Path(args.output_run)
    arms, mode_bits = args.arms.split(","), csv_ints(args.mode_bits)
    budgets, refs = csv_ints(args.budgets), csv_ints(args.reference_bits)
    if len(budgets) != len(refs):
        raise ValueError("budgets and reference-bits must have equal length")
    for budget, ref in zip(budgets, refs):
        records, union = {}, set()
        for index, arm in enumerate(arms):
            old_path = source / arm / f"calibration_R{budget}.npz"
            old = np.load(old_path, allow_pickle=False)
            if tuple(old["mode_bits"]) != mode_bits:
                raise ValueError(f"mode menu mismatch in {old_path}")
            q, c_g = np.asarray(old["q_per_image"]), np.asarray(old["c_g"])
            own, ideal, costs = make_allocations(
                c_g, mode_bits, budget, args.single, args.random,
                args.seed + index, args.dimension)
            optimum = top2_allocate(c_g, mode_bits, budget, args.dimension)
            records[arm] = (
                c_g, q, own, ideal, costs, old_path, optimum)
            union.update(map(tuple, own))
        allocations = np.asarray(sorted(union), dtype=np.int64)
        validate_fixed_total_rate(allocations, records[arms[0]][4])
        for arm, record in records.items():
            c_g, q, own, ideal, costs, old_path, optimum = record
            directory = output / arm
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                directory / f"calibration_R{budget}.npz",
                c_g=c_g, q_per_image=q, allocations=allocations,
                own_allocations=own, ideal_bits=ideal, cost_table=costs,
                ideal_value=optimum["ideal_value"],
                second_bits=optimum["second_bits"],
                second_value=optimum["second_value"],
                ideal_gap=optimum["ideal_gap"],
                mode_bits=mode_bits, rate_dimension=args.dimension,
                coefficient_kind="independent_saved_jvp")
            dump_json(directory / f"calibration_R{budget}.json", {
                "arm": arm, "budget": budget, "reference_bit": ref,
                "rate_dimension": args.dimension,
                "n_images": int(len(q)), "n_own": int(len(own)),
                "n_union": int(len(allocations)),
                "ideal_gap": optimum["ideal_gap"],
                "source_calibration": str(old_path),
            })
        print(f"R={budget}: common union has {len(allocations)} allocations")


@torch.no_grad()
def command_measure(args):
    device = torch.device("cuda")
    codec = load_codec_v1(args.codec, device=device)
    tail = build_tail(args.layer, device)
    features = np.load(args.features, mmap_mode="r")[:args.images]
    teachers = np.load(args.teachers, mmap_mode="r")[:args.images]
    out = Path(args.output_dir)
    for budget in csv_ints(args.budgets):
        calibration = np.load(
            out / f"calibration_R{budget}.npz", allow_pickle=False)
        rate_dimension = int(calibration["rate_dimension"])
        if rate_dimension != codec.pq.d:
            raise ValueError(
                f"rate dimension {rate_dimension} != PQ dimension {codec.pq.d}")
        summary, arrays = evaluate_fixed_rate_remainder(
            features, teachers, codec, tail, calibration["allocations"],
            calibration["cost_table"], calibration["c_g"], args.norm_mode,
            device, batch_size=args.batch_size,
            allocation_chunk=args.allocation_chunk,
            ideal_cost_table=(
                calibration["ideal_cost_table"]
                if "ideal_cost_table" in calibration.files else None))
        sampled_gap = summary["phi_gap"]
        exact_gap = float(calibration["ideal_gap"])
        summary.update({
            "phi_gap_sampled": sampled_gap,
            "phi_gap": exact_gap,
            "phi_gap_exact": exact_gap,
            "chi_sampled": (
                float(summary["omega_sampled"] / exact_gap)
                if np.isfinite(exact_gap) and exact_gap > 0 else None),
            "omega_scope": "sampled_allocation_lower_bound",
            "ideal_model_fixed_from_independent_calibration": True,
            "ideal_model": (
                str(calibration["ideal_model"])
                if "ideal_model" in calibration.files
                else "common_exponential"),
        })
        summary.update({
            "arm": args.arm, "budget": budget,
            "rate_dimension": rate_dimension,
        })
        np.savez_compressed(out / f"measurement_R{budget}.npz", **arrays)
        dump_json(out / f"measurement_R{budget}.json", summary)
        print(f"R={budget}: omega={summary['omega_sampled']:.6g}")


@torch.no_grad()
def command_decompose(args):
    device = torch.device("cuda")
    codec = load_codec_v1(args.codec, device=device)
    tail = build_tail(args.layer, device)
    features = np.load(args.features, mmap_mode="r")[:args.images]
    teachers = np.load(args.teachers, mmap_mode="r")[:args.images]
    out = Path(args.output_dir)
    for budget in csv_ints(args.budgets):
        source_path = (
            Path(args.calibration) if args.calibration else
            out / f"calibration_R{budget}.npz")
        calibration = np.load(source_path, allow_pickle=False)

        def field(name):
            key = (
                name if name in calibration.files else
                f"calibration__{name}")
            if key not in calibration.files:
                raise KeyError(f"{name} is absent from {source_path}")
            return calibration[key]

        allocations = (
            calibration[args.allocation_key]
            if args.allocation_key in calibration.files else
            field("allocations"))
        if (
            "reference_bit" in calibration.files
            or "calibration__reference_bit" in calibration.files
        ):
            reference_bit = int(field("reference_bit"))
        else:
            distance = np.square(
                field("c_by_bit") - field("c_g")[None]).mean(1)
            reference_bit = int(
                field("coefficient_bits")[distance.argmin()])
        summary, arrays = evaluate_fixed_rate_decomposition(
            features, teachers, codec, tail, allocations,
            field("cost_table"), field("c_g"), args.norm_mode,
            device, allocation_chunk=args.allocation_chunk,
            jvp_eps=args.jvp_eps, jvp_chunk=args.jvp_chunk,
            mode_bits=field("mode_bits"),
            reference_bit=reference_bit,
            bootstrap_count=args.bootstrap_count,
            bootstrap_batch=args.bootstrap_batch,
            bootstrap_seed=args.seed + 2000,
            stability_repeats=args.stability_repeats,
            ideal_cost_table=(
                field("ideal_cost_table")
                if (
                    "ideal_cost_table" in calibration.files
                    or "calibration__ideal_cost_table" in calibration.files)
                else None))
        summary.update({
            "arm": args.arm, "budget": budget,
            "calibration_source": str(source_path),
            "allocation_key": args.allocation_key,
            "allocation_scope": "sampled_lower_bound",
            "ideal_terms_recomputed_for_current_codec": True,
            "complete_and_ideal_terms_paired_by_image": True,
        })
        np.savez_compressed(out / f"decomposition_R{budget}.npz", **arrays)
        dump_json(out / f"decomposition_R{budget}.json", summary)
        print(
            f"R={budget}: structural="
            f"{summary['structural_range_point']:.6g}, analytic="
            f"{summary['analytic_remainder_range_point']:.6g}, "
            f"recovery={summary['sampled_recovery_condition_confident']}")


def command_audit(args):
    root, rows, errors, common = Path(args.run_dir), [], [], {}
    for arm in args.arms.split(","):
        for budget in csv_ints(args.budgets):
            path = root / arm / f"measurement_R{budget}.json"
            if not path.exists():
                errors.append(f"missing {path}")
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["max_rate_error"] > 1e-8:
                errors.append(f"rate mismatch: {arm}/R{budget}")
            if row["n_allocations"] < 2 or row["n_images"] < 1:
                errors.append(f"insufficient data: {arm}/R{budget}")
            measured = np.load(path.with_suffix(".npz"))["allocations"]
            calibrated = np.load(
                root / arm / f"calibration_R{budget}.npz")
            bits = calibrated["mode_bits"][measured]
            if not np.array_equal(
                    bits[row["phi_best_index"]], calibrated["ideal_bits"]):
                errors.append(f"ideal allocation mismatch: {arm}/R{budget}")
            if budget in common and not np.array_equal(common[budget], measured):
                errors.append(f"allocation union mismatch: {arm}/R{budget}")
            common.setdefault(budget, measured)
            rows.append(row)
    report = {"passed": not errors, "errors": errors, "results": rows}
    dump_json(root / "audit.json", report)
    if errors:
        raise RuntimeError("; ".join(errors))
    print(f"PASS: {len(rows)} arm-budget results")


def parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--arm", required=True)
    common.add_argument("--codec", required=True)
    common.add_argument("--features", required=True)
    common.add_argument("--norm-mode", default="per_image")
    common.add_argument("--layer", type=int, default=20)
    common.add_argument("--batch-size", type=int, default=4)
    common.add_argument("--seed", type=int, default=42)
    main = argparse.ArgumentParser()
    sub = main.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--arm", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source-kind", choices=("scratch", "opq"),
                   required=True)
    p.add_argument("--mode-bits", required=True)
    p.add_argument("--groups", type=int, default=32)
    p.add_argument("--images", type=int, default=2000)
    p.add_argument("--max-vectors", type=int, default=200000)
    p.add_argument("--kmeans-iters", type=int, default=50)
    p.add_argument("--opq-bits", type=int, default=6)
    p.add_argument("--opq-iters", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--norm-mode", default="per_image")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("calibrate", parents=[common])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mode-bits", required=True)
    p.add_argument("--budgets", required=True)
    p.add_argument("--reference-bits", required=True)
    p.add_argument("--coefficient-bits")
    p.add_argument(
        "--ideal-model", choices=("common_exponential", "discrete_jvp"),
        default="common_exponential")
    p.add_argument("--images", type=int, default=64)
    p.add_argument("--image-offset", type=int, default=0)
    p.add_argument("--group-chunk", type=int, default=8)
    p.add_argument("--eps", type=float, default=0.01)
    p.add_argument("--single", type=int, default=32)
    p.add_argument("--random", type=int, default=32)
    p = sub.add_parser("reallocate")
    p.add_argument("--source-run", required=True)
    p.add_argument("--output-run", required=True)
    p.add_argument("--arms", default="identity,opq,orfc,response")
    p.add_argument("--mode-bits", default="3,4,5,6,7,8")
    p.add_argument("--budgets", default="128,192")
    p.add_argument("--reference-bits", default="4,6")
    p.add_argument("--dimension", type=int, default=32)
    p.add_argument("--single", type=int, default=32)
    p.add_argument("--random", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("measure", parents=[common])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--teachers", required=True)
    p.add_argument("--budgets", required=True)
    p.add_argument("--images", type=int, default=300)
    p.add_argument("--allocation-chunk", type=int, default=8)
    p = sub.add_parser("decompose", parents=[common])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--calibration")
    p.add_argument("--allocation-key", default="allocations")
    p.add_argument("--teachers", required=True)
    p.add_argument("--budgets", required=True)
    p.add_argument("--images", type=int, default=32)
    p.add_argument("--allocation-chunk", type=int, default=4)
    p.add_argument("--jvp-eps", type=float, default=0.01)
    p.add_argument("--jvp-chunk", type=int, default=8)
    p.add_argument("--bootstrap-count", type=int, default=1000)
    p.add_argument("--bootstrap-batch", type=int, default=32)
    p.add_argument("--stability-repeats", type=int, default=200)
    p = sub.add_parser("audit")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--arms", default="identity,opq,orfc,response")
    p.add_argument("--budgets", required=True)
    return main


if __name__ == "__main__":
    args = parser().parse_args()
    globals()[f"command_{args.command}"](args)
