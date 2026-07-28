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
from allocate import dp_allocate


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
        base = tail.forward_nograd(x)
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
            perturbed = (
                x.unsqueeze(0) + args.eps * direction).reshape(
                    -1, tokens, x.shape[-1])
            response = tail.forward_nograd(perturbed).reshape(
                last - first, batch, *base.shape[1:])
            delta = response - base.unsqueeze(0)
            q_batch[:, first:last] = (
                delta.flatten(2).square().sum(2)
                * (magnitude / args.eps).square()).t()
        q_rows.append(q_batch.cpu().numpy())
    q = np.concatenate(q_rows)
    return np.exp2(2.0 * bit / codec.pq.d) * q.mean(0), q


def make_allocations(
    c_g, mode_bits, budget, n_single, n_random, seed, dimension,
):
    phi = {
        g: {bits: float(c_g[g] * np.exp2(-2.0 * bits / dimension))
            for bits in mode_bits}
        for g in range(len(c_g))
    }
    ideal_bits, _ = dp_allocate(phi, budget, menu=mode_bits)
    index = {bits: i for i, bits in enumerate(mode_bits)}
    base = np.asarray([index[b] for b in ideal_bits], dtype=np.int64)
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
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for budget, ref in zip(budgets, refs):
        c_g, q = estimate_c(
            codec, tail, features, mode_bits.index(ref), ref, args, device)
        allocations, ideal_bits, costs = make_allocations(
            c_g, mode_bits, budget, args.single, args.random, args.seed,
            codec.pq.d)
        np.savez_compressed(
            out / f"calibration_R{budget}.npz", c_g=c_g, q_per_image=q,
            allocations=allocations, ideal_bits=ideal_bits,
            cost_table=costs, mode_bits=mode_bits,
            rate_dimension=codec.pq.d, coefficient_kind="jvp",
            image_offset=args.image_offset)
        dump_json(out / f"calibration_R{budget}.json", {
            "arm": args.arm, "budget": budget, "reference_bit": ref,
            "n_images": int(len(q)), "n_allocations": int(len(allocations)),
            "c_min": float(c_g.min()), "c_max": float(c_g.max()),
            "rate_dimension": codec.pq.d, "coefficient_kind": "jvp",
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
            q = np.asarray(old["q_per_image"])
            c_g = np.exp2(2.0 * ref / args.dimension) * q.mean(0)
            own, ideal, costs = make_allocations(
                c_g, mode_bits, budget, args.single, args.random,
                args.seed + index, args.dimension)
            records[arm] = (c_g, q, own, ideal, costs, old_path)
            union.update(map(tuple, own))
        allocations = np.asarray(sorted(union), dtype=np.int64)
        validate_fixed_total_rate(allocations, records[arms[0]][4])
        for arm, (c_g, q, own, ideal, costs, old_path) in records.items():
            directory = output / arm
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                directory / f"calibration_R{budget}.npz",
                c_g=c_g, q_per_image=q, allocations=allocations,
                own_allocations=own, ideal_bits=ideal, cost_table=costs,
                mode_bits=mode_bits, rate_dimension=args.dimension)
            dump_json(directory / f"calibration_R{budget}.json", {
                "arm": arm, "budget": budget, "reference_bit": ref,
                "rate_dimension": args.dimension,
                "n_images": int(len(q)), "n_own": int(len(own)),
                "n_union": int(len(allocations)),
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
            allocation_chunk=args.allocation_chunk)
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
        calibration = np.load(
            out / f"calibration_R{budget}.npz", allow_pickle=False)
        summary, arrays = evaluate_fixed_rate_decomposition(
            features, teachers, codec, tail, calibration["allocations"],
            calibration["cost_table"], calibration["c_g"], args.norm_mode,
            device, allocation_chunk=args.allocation_chunk,
            jvp_eps=args.jvp_eps, jvp_chunk=args.jvp_chunk)
        summary.update({"arm": args.arm, "budget": budget})
        np.savez_compressed(out / f"decomposition_R{budget}.npz", **arrays)
        dump_json(out / f"decomposition_R{budget}.json", summary)
        print(
            f"R={budget}: remainder={summary['remainder_range']:.6g}, "
            f"dominant={summary['dominant_component_by_range']}")


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
    p.add_argument("--teachers", required=True)
    p.add_argument("--budgets", required=True)
    p.add_argument("--images", type=int, default=32)
    p.add_argument("--allocation-chunk", type=int, default=4)
    p.add_argument("--jvp-eps", type=float, default=0.01)
    p.add_argument("--jvp-chunk", type=int, default=8)
    p = sub.add_parser("audit")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--arms", default="identity,opq,orfc,response")
    p.add_argument("--budgets", required=True)
    return main


if __name__ == "__main__":
    args = parser().parse_args()
    globals()[f"command_{args.command}"](args)
