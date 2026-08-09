"""V34: measure whether token/channel rotations improve Tail-MSE separability."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from codec_v1 import load_codec_v1
from opq import batch_inv_normalize_gpu, batch_normalize_gpu

from .. import engine, kmeans
from .. import tail as tail_mod
from ..v21 import config as v21_config
from ..v33.noise_floor import random_exact_budget_allocations
from .model import TwoSidedPQ


ARMS = ("identity", "opq", "task_u", "task_v", "task_vu")
BITS = (0, 1, 2)
GROUPS, GROUP_DIM = 32, 32


def configure(block):
    cfg = v21_config.activate(block)
    if cfg.GROUPS != GROUPS or cfg.DIM != GROUP_DIM:
        raise ValueError("V34 is registered for 32 x 32 channel groups")
    return cfg


def batches(path, rows, batch, device):
    source = np.load(path, mmap_mode="r")
    rows = np.asarray(rows, dtype=np.int64)
    for first in range(0, len(rows), int(batch)):
        block = np.asarray(source[rows[first:first + int(batch)]])
        yield torch.from_numpy(block).float().to(device)


def canonical_eigenbasis(matrix):
    matrix = 0.5 * (matrix + matrix.t())
    _, vectors = torch.linalg.eigh(matrix.double())
    vectors = vectors.flip(1)
    # Deterministic signs: the largest-magnitude entry in each column is +.
    index = vectors.abs().argmax(0)
    signs = vectors[index, torch.arange(vectors.shape[1])].sign()
    signs[signs == 0] = 1
    return (vectors * signs).float()


def effective_rank(matrix):
    values = torch.linalg.eigvalsh(0.5 * (matrix + matrix.t())).clamp_min(0)
    probabilities = values / values.sum().clamp_min(torch.finfo(values.dtype).tiny)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return float(entropy.exp())


def estimate_task_marginals(cfg, device, images, batch, probes, seed, log):
    """Hutchinson contractions of J^T J over token and channel axes."""
    tail = tail_mod.build_tail(cfg.LAYER, device)
    source = np.load(cfg.TRAIN_FEATURES, mmap_mode="r")
    count = min(int(images), cfg.N_TRAIN)
    token = torch.zeros(257, 257, dtype=torch.float64, device=device)
    channel = torch.zeros(1024, 1024, dtype=torch.float64, device=device)
    token_half = [torch.zeros_like(token), torch.zeros_like(token)]
    channel_half = [torch.zeros_like(channel), torch.zeros_like(channel)]
    generator = torch.Generator(device=device).manual_seed(int(seed))
    seen = [0, 0]
    for first in range(0, count, int(batch)):
        last = min(first + int(batch), count)
        x = torch.from_numpy(np.array(source[first:last], copy=True)).float().to(device)
        with torch.no_grad():
            y0, mu, std = batch_normalize_gpu(x, mode=cfg.NORM_MODE)
        y = y0.detach().requires_grad_(True)
        output = tail(batch_inv_normalize_gpu(y, mu, std))
        scale = math.sqrt(output[0].numel())
        half = int(first >= count // 2)
        for probe in range(int(probes)):
            signs = torch.randint(
                0, 2, output.shape, generator=generator, device=device,
                dtype=torch.int8).float().mul_(2).sub_(1)
            scalar = (output * signs).sum() / scale
            gradient, = torch.autograd.grad(
                scalar, y, retain_graph=(probe + 1 < int(probes)))
            kt = torch.einsum("btd,bsd->ts", gradient, gradient).double()
            kc = torch.einsum("btd,bte->de", gradient, gradient).double()
            token.add_(kt); channel.add_(kc)
            token_half[half].add_(kt); channel_half[half].add_(kc)
            seen[half] += int(y.shape[0])
        del output, y, y0, x
        if last == count or last % max(1, 8 * int(batch)) == 0:
            log(f"probe {last}/{count}")
    token /= max(sum(seen), 1)
    channel /= max(sum(seen), 1)
    for half in range(2):
        token_half[half] /= max(seen[half], 1)
        channel_half[half] /= max(seen[half], 1)
    def stability(a, b):
        return float(torch.nn.functional.cosine_similarity(
            a.reshape(1, -1), b.reshape(1, -1)).item())
    result = {
        "token": token.cpu(), "channel": channel.cpu(),
        "V": canonical_eigenbasis(token).cpu(),
        "U": canonical_eigenbasis(channel).cpu(),
        "diagnostics": {
            "images": count, "probes": int(probes),
            "token_effective_rank": effective_rank(token),
            "channel_effective_rank": effective_rank(channel),
            "token_half_cosine": stability(token_half[0], token_half[1]),
            "channel_half_cosine": stability(channel_half[0], channel_half[1]),
        },
    }
    del tail
    torch.cuda.empty_cache()
    return result


def probe(args):
    cfg = configure(args.block)
    device = torch.device(args.device)
    engine.configure_precision(False)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = estimate_task_marginals(
        cfg, device, args.probe_images, args.probe_batch, args.probes,
        args.seed, print)
    np.savez(
        out, V=result["V"].numpy(), U=result["U"].numpy(),
        token=result["token"].numpy(), channel=result["channel"].numpy())
    meta = dict(result["diagnostics"], block=args.block,
                seconds=time.time() - started, output=str(out))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def arm_rotations(cfg, arm, probe_path, device):
    probe_data = np.load(probe_path)
    eye_t = torch.eye(257, device=device)
    eye_c = torch.eye(1024, device=device)
    task_v = torch.from_numpy(probe_data["V"]).float().to(device)
    task_u = torch.from_numpy(probe_data["U"]).float().to(device)
    if arm == "identity":
        return eye_t, eye_c
    if arm == "task_u":
        return eye_t, task_u
    if arm == "task_v":
        return task_v, eye_c
    if arm == "task_vu":
        return task_v, task_u
    if arm == "opq":
        root = cfg.V12 / "init_orfc_adam" / "R32" / "codec.pt"
        source = load_codec_v1(root, device=device)
        return eye_t, source.transform.get_rotation().detach()
    raise ValueError(arm)


@torch.no_grad()
def load_transformed_training(cfg, V, device, images, batch):
    rows = np.arange(min(int(images), cfg.N_TRAIN), dtype=np.int64)
    vectors = []
    for x in batches(cfg.TRAIN_FEATURES, rows, batch, device):
        y, _, _ = batch_normalize_gpu(x, mode=cfg.NORM_MODE)
        token = torch.einsum("ts,btd->bsd", V, y)
        vectors.append(token.reshape(-1, token.shape[-1]))
    return torch.cat(vectors)


@torch.no_grad()
def fit_codec(cfg, V, U, device, images, iterations, seed, log):
    y = load_transformed_training(cfg, V, device, images, 32)
    books, fit = [], []
    for mode, bits in enumerate(BITS):
        generator = torch.Generator(device=device).manual_seed(int(seed) + mode)
        book = kmeans.kmeans_plusplus(
            y, U, GROUPS, GROUP_DIM, 2 ** bits, generator, device)
        book = kmeans.lloyd(
            y, U, book, int(iterations), GROUPS, GROUP_DIM)
        books.append(book)
        fit.append({
            "bits": bits,
            "mse": kmeans.quantisation_mse(
                y, U, book, GROUPS, GROUP_DIM),
            "dead_fraction_max": float(kmeans.dead_fraction(
                y, U, book, GROUPS, GROUP_DIM).max()),
        })
        log(f"fit {bits}b K={2 ** bits} mse={fit[-1]['mse']:.6f}")
    codec = TwoSidedPQ(V, U, books, BITS, GROUPS, GROUP_DIM).to(device).eval()
    del y
    torch.cuda.empty_cache()
    return codec, fit


def expand_stats(values, denominator=None):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    sem = float(values.std(ddof=1) / math.sqrt(len(values)))
    result = {"mean": mean, "ci95": [mean - 1.96 * sem, mean + 1.96 * sem],
              "mean_abs": float(np.abs(values).mean())}
    if denominator is not None:
        result["mean_abs_over_full"] = float(
            np.abs(values).mean() / max(abs(float(np.mean(denominator))), 1e-30))
    return result


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    first = 0
    while first < len(values):
        last = first + 1
        while last < len(values) and values[order[last]] == values[order[first]]:
            last += 1
        ranks[order[first:last]] = 0.5 * (first + last - 1)
        first = last
    return ranks


def spearman(left, right):
    a, b = rankdata(left), rankdata(right)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def decoded_distortion(tail, codec, y, mu, std, teacher, coefficient_errors):
    """Vectorised candidates [C,B,T,D] -> per-image Tail MSE [C,B]."""
    count, batch = coefficient_errors.shape[:2]
    decoded = y.unsqueeze(0) + codec.decode_error(coefficient_errors)
    flat = decoded.reshape(count * batch, *decoded.shape[2:])
    mu2 = mu.unsqueeze(0).expand(count, *mu.shape).reshape(
        count * batch, *mu.shape[1:])
    std2 = std.unsqueeze(0).expand(count, *std.shape).reshape(
        count * batch, *std.shape[1:])
    output = tail(batch_inv_normalize_gpu(flat, mu2, std2))
    target = teacher.unsqueeze(0).expand(
        count, *teacher.shape).reshape(count * batch, *teacher.shape[1:])
    return (output - target).square().reshape(count, batch, -1).sum(-1)


@torch.no_grad()
def evaluate_self_costs(cfg, codec, tail, feature_path, teacher_path, rows,
                        device, image_batch, candidate_chunk):
    """Per-group/mode self distortions, averaged over the selected rows."""
    teacher_source = np.load(teacher_path, mmap_mode="r")
    total = torch.zeros(GROUPS, len(BITS), dtype=torch.float64)
    count = 0
    for first in range(0, len(rows), int(image_batch)):
        selected = rows[first:first + int(image_batch)]
        x = next(batches(feature_path, selected, len(selected), device))
        teacher = torch.from_numpy(np.asarray(teacher_source[selected])).float().to(device)
        y, mu, std = batch_normalize_gpu(x, mode=cfg.NORM_MODE)
        bank = codec.mode_error_bank(y).reshape(
            len(BITS), len(selected), 257, GROUPS, GROUP_DIM)
        candidates, indices = [], []
        for mode in range(len(BITS)):
            for group in range(GROUPS):
                error = torch.zeros_like(bank[mode, :, :, :, :])
                error[:, :, group] = bank[mode, :, :, group]
                candidates.append(error.reshape(len(selected), 257, 1024))
                indices.append((group, mode))
                if len(candidates) == int(candidate_chunk):
                    values = decoded_distortion(
                        tail, codec, y, mu, std, teacher, torch.stack(candidates))
                    for row, (g, m) in zip(values, indices):
                        total[g, m] += row.double().sum().cpu()
                    candidates, indices = [], []
        if candidates:
            values = decoded_distortion(
                tail, codec, y, mu, std, teacher, torch.stack(candidates))
            for row, (g, m) in zip(values, indices):
                total[g, m] += row.double().sum().cpu()
        count += len(selected)
    return (total / count).numpy()


@torch.no_grad()
def evaluate_coupling_and_allocations(
        cfg, codec, tail, rows, allocations, device, image_batch,
        candidate_chunk):
    feature_source = np.load(cfg.VAL_FEATURES, mmap_mode="r")
    teacher_source = np.load(cfg.VAL_TEACHERS, mmap_mode="r")
    full_rows, token_residual, channel_residual = [], [], []
    allocation_values = [[] for _ in range(len(allocations))]
    uniform = np.ones(GROUPS, dtype=np.int64)
    for first in range(0, len(rows), int(image_batch)):
        selected = rows[first:first + int(image_batch)]
        x = torch.from_numpy(np.asarray(feature_source[selected])).float().to(device)
        teacher = torch.from_numpy(np.asarray(teacher_source[selected])).float().to(device)
        y, mu, std = batch_normalize_gpu(x, mode=cfg.NORM_MODE)
        bank = codec.mode_error_bank(y)
        full_error = codec.error_for_allocation(bank, uniform)
        full = decoded_distortion(
            tail, codec, y, mu, std, teacher, full_error.unsqueeze(0))[0]
        full_rows.append(full.cpu())

        token_sum = torch.zeros_like(full)
        for start in range(0, 257, int(candidate_chunk)):
            stop = min(start + int(candidate_chunk), 257)
            errors = torch.zeros(
                stop - start, *full_error.shape, device=device)
            for local, token in enumerate(range(start, stop)):
                errors[local, :, token] = full_error[:, token]
            token_sum += decoded_distortion(
                tail, codec, y, mu, std, teacher, errors).sum(0)
        token_residual.append((full - token_sum).cpu())

        grouped = full_error.reshape(len(selected), 257, GROUPS, GROUP_DIM)
        channel_sum = torch.zeros_like(full)
        for start in range(0, GROUPS, int(candidate_chunk)):
            stop = min(start + int(candidate_chunk), GROUPS)
            errors = torch.zeros(
                stop - start, len(selected), 257, GROUPS, GROUP_DIM,
                device=device)
            for local, group in enumerate(range(start, stop)):
                errors[local, :, :, group] = grouped[:, :, group]
            channel_sum += decoded_distortion(
                tail, codec, y, mu, std, teacher,
                errors.reshape(stop - start, len(selected), 257, 1024)).sum(0)
        channel_residual.append((full - channel_sum).cpu())

        for start in range(0, len(allocations), int(candidate_chunk)):
            part = allocations[start:start + int(candidate_chunk)]
            errors = torch.stack([
                codec.error_for_allocation(bank, allocation) for allocation in part])
            values = decoded_distortion(tail, codec, y, mu, std, teacher, errors)
            for offset, row in enumerate(values):
                allocation_values[start + offset].append(row.cpu())
    full = torch.cat(full_rows).numpy()
    return {
        "full": full,
        "token_residual": torch.cat(token_residual).numpy(),
        "channel_residual": torch.cat(channel_residual).numpy(),
        "allocation_values": np.stack([
            torch.cat(parts).numpy() for parts in allocation_values]),
    }


def run_arm(args):
    cfg = configure(args.block)
    device = torch.device(args.device)
    engine.configure_precision(False)
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is non-empty")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    V, U = arm_rotations(cfg, args.arm, args.probe, device)
    codec, fit = fit_codec(
        cfg, V, U, device, args.fit_images, args.kmeans_iters,
        args.seed, print)
    torch.save({
        "format": "v34_two_sided_v1", "block": args.block, "arm": args.arm,
        "bits": BITS, "V": codec.V.cpu(), "U": codec.U.cpu(),
        "codebooks": [book.detach().cpu() for book in codec.books]},
        out / "codec.pt")
    tail = tail_mod.build_tail(cfg.LAYER, device)
    train_rows = np.arange(min(args.cost_images, cfg.N_TRAIN), dtype=np.int64)
    cost = evaluate_self_costs(
        cfg, codec, tail, cfg.TRAIN_FEATURES, cfg.TRAIN_TEACHERS,
        train_rows, device, args.image_batch, args.candidate_chunk)
    allocations = random_exact_budget_allocations(
        GROUPS, BITS, 32, args.allocations - 1, seed=args.seed)
    uniform = np.ones((1, GROUPS), dtype=np.int64)
    allocations = allocations[(allocations != uniform[0]).any(1)]
    if len(allocations) != args.allocations - 1:
        raise SystemExit("random allocation draw unexpectedly contained uniform")
    allocations = np.concatenate([uniform, allocations], 0)
    val_rows = np.arange(min(args.val_images, 500), dtype=np.int64)
    measured = evaluate_coupling_and_allocations(
        cfg, codec, tail, val_rows, allocations, device,
        args.image_batch, args.candidate_chunk)
    true_scores = measured["allocation_values"].mean(1)
    predicted = np.asarray([
        sum(cost[g, int(mode)] for g, mode in enumerate(row))
        for row in allocations], dtype=np.float64)
    nrmse = float(np.sqrt(np.mean((predicted - true_scores) ** 2)) /
                  max(true_scores.std(), 1e-30))
    predicted_delta = predicted - predicted[0]
    true_delta = true_scores - true_scores[0]
    delta_nrmse = float(np.sqrt(np.mean(
        (predicted_delta - true_delta) ** 2)) /
        max(true_delta.std(), 1e-30))
    best = int(predicted.argmin())
    result = {
        "plan": "v34_two_sided_separability", "block": args.block,
        "arm": args.arm, "rate": 32, "bits": list(BITS),
        "fit": fit, "orthogonality": codec.orthogonality(),
        "data": {"fit_images": args.fit_images, "cost_images": args.cost_images,
                 "val_images": len(val_rows), "allocations": len(allocations)},
        "coupling": {
            "full_mse": expand_stats(measured["full"]),
            "token_residual": expand_stats(
                measured["token_residual"], measured["full"]),
            "channel_residual": expand_stats(
                measured["channel_residual"], measured["full"]),
        },
        "separable_allocation": {
            "spearman": spearman(predicted, true_scores), "nrmse": nrmse,
            "delta_nrmse": delta_nrmse,
            "predicted_best_index": best,
            "predicted_best_true_mse": float(true_scores[best]),
            "uniform_true_mse": float(true_scores[0]),
            "registered_true_best_mse": float(true_scores.min()),
            "registered_true_best_index": int(true_scores.argmin()),
            "predicted_scores": predicted.tolist(),
            "true_scores": true_scores.tolist(),
            "allocations": allocations.tolist(),
            "self_cost_table": cost.tolist(),
        },
        "seconds": time.time() - started,
        "peak_memory_gib": (float(torch.cuda.max_memory_allocated(device)) /
                            (1024 ** 3) if device.type == "cuda" else 0.0),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "block": args.block, "arm": args.arm,
        "token_residual": result["coupling"]["token_residual"],
        "channel_residual": result["coupling"]["channel_residual"],
        "spearman": result["separable_allocation"]["spearman"],
        "nrmse": nrmse, "delta_nrmse": delta_nrmse,
        "seconds": result["seconds"]}, indent=2))


def smoke(args):
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(34)
    q_t, _ = torch.linalg.qr(torch.randn(257, 257, generator=generator, device=device))
    q_c, _ = torch.linalg.qr(torch.randn(1024, 1024, generator=generator, device=device))
    books = [torch.randn(GROUPS, 2 ** b, GROUP_DIM, generator=generator,
                         device=device) for b in BITS]
    codec = TwoSidedPQ(q_t, q_c, books, BITS).to(device)
    y = torch.randn(2, 257, 1024, generator=generator, device=device)
    roundtrip = codec.synthesis(codec.analysis(y))
    relative = float((roundtrip - y).norm() / y.norm())
    allocation = torch.ones(GROUPS, dtype=torch.long, device=device)
    reconstructed = codec.reconstruct(y, allocation)
    loss = reconstructed.square().mean()
    if not torch.isfinite(loss) or relative > 2e-5:
        raise SystemExit(f"smoke failed: relative={relative}")
    print(json.dumps({"roundtrip_relative": relative,
                      "orthogonality": codec.orthogonality(),
                      "reconstruction_shape": list(reconstructed.shape)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--block", choices=("blk05", "blk20"), required=True)
    common.add_argument("--device", default="cuda")
    common.add_argument("--seed", type=int, default=20260809)
    p = sub.add_parser("probe", parents=[common])
    p.add_argument("--output", required=True)
    p.add_argument("--probe-images", type=int, default=512)
    p.add_argument("--probe-batch", type=int, default=4)
    p.add_argument("--probes", type=int, default=2)
    a = sub.add_parser("arm", parents=[common])
    a.add_argument("--arm", choices=ARMS, required=True)
    a.add_argument("--probe", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--fit-images", type=int, default=5000)
    a.add_argument("--cost-images", type=int, default=500)
    a.add_argument("--val-images", type=int, default=500)
    a.add_argument("--allocations", type=int, default=256)
    a.add_argument("--kmeans-iters", type=int, default=100)
    a.add_argument("--image-batch", type=int, default=4)
    a.add_argument("--candidate-chunk", type=int, default=4)
    s = sub.add_parser("smoke")
    s.add_argument("--device", default="cuda")
    args = parser.parse_args()
    {"probe": probe, "arm": run_arm, "smoke": smoke}[args.command](args)


if __name__ == "__main__":
    main()
