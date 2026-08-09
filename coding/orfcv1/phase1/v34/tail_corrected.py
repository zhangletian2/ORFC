"""Tail-MSE-trained two-sided rotations for the corrected V34 experiment."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cayley import AnchoredCayleyTransform
from opq import batch_inv_normalize_gpu, batch_normalize_gpu

from .. import engine
from .. import tail as tail_mod
from ..v12.train import CachedFeatureDataset
from ..v21 import config as v21_config
from ..v33.noise_floor import random_exact_budget_allocations
from .model import TwoSidedPQ
from .run import (BITS, GROUPS, GROUP_DIM, batches, decoded_distortion,
                  evaluate_coupling_and_allocations, evaluate_self_costs,
                  expand_stats, spearman)


ARMS = ("identity_fixed", "opq_fixed", "tail_u", "tail_v", "tail_vu")


def configure(block):
    cfg = v21_config.activate(block)
    if cfg.GROUPS != GROUPS or cfg.DIM != GROUP_DIM:
        raise ValueError("corrected V34 requires 32 x 32 channel groups")
    return cfg


class TrainableTwoSidedPQ(nn.Module):
    """Hard PQ forward, soft codeword backward, Cayley V/U."""

    def __init__(self, V, U, books, train_v, train_u):
        super().__init__()
        self.token = AnchoredCayleyTransform(257)
        self.channel = AnchoredCayleyTransform(1024)
        self.token.init_from_opq(V)
        self.channel.init_from_opq(U)
        self.books = nn.ParameterList([
            nn.Parameter(torch.as_tensor(book).float().clone()) for book in books])
        self.token.triu_params.requires_grad_(bool(train_v))
        self.channel.triu_params.requires_grad_(bool(train_u))

    def rotations(self):
        return self.token.get_rotation(), self.channel.get_rotation()

    def reconstruct_mode(self, y, mode, tau):
        V, U = self.rotations()
        z = torch.einsum("ts,btd->bsd", V, y) @ U
        b, t, d = z.shape
        sub = z.reshape(-1, GROUPS, GROUP_DIM).permute(1, 0, 2)
        book = self.books[int(mode)]
        distance = torch.cdist(sub, book).square()
        index = distance.detach().argmin(-1)
        soft = torch.softmax(-distance / float(tau), -1)
        hard = F.one_hot(index, book.shape[1]).to(soft.dtype)
        weight = hard.detach() + soft - soft.detach()
        chosen = torch.einsum("gnk,gkd->gnd", weight, book)
        zhat = chosen.permute(1, 0, 2).reshape(b, t, d)
        decoded = (zhat @ U.t())
        return torch.einsum("ts,bsd->btd", V, decoded)

    @torch.no_grad()
    def fixed(self):
        V, U = self.rotations()
        return TwoSidedPQ(V.detach(), U.detach(),
                          [book.detach() for book in self.books], BITS)


def load_init(path, arm, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    books = payload["codebooks"]
    if len(books) != len(BITS):
        raise SystemExit("INVALID_EXPERIMENT: initialization mode count")
    if arm == "identity_fixed":
        V, U = torch.eye(257), torch.eye(1024)
    else:
        V, U = torch.eye(257), payload["U"]
    train_v = arm in ("tail_v", "tail_vu")
    train_u = arm in ("tail_u", "tail_vu")
    return TrainableTwoSidedPQ(V, U, books, train_v, train_u).to(device)


def hard_mode_mse(model, tail, cfg, rows, device, batch=16):
    teacher_source = np.load(cfg.VAL_TEACHERS, mmap_mode="r")
    total = torch.zeros(len(BITS), dtype=torch.float64)
    count = 0
    fixed = model.fixed().to(device).eval()
    for first in range(0, len(rows), int(batch)):
        selected = rows[first:first + int(batch)]
        x = next(batches(cfg.VAL_FEATURES, selected, len(selected), device))
        teacher = torch.from_numpy(np.asarray(teacher_source[selected])).float().to(device)
        y, mu, std = batch_normalize_gpu(x, mode=cfg.NORM_MODE)
        bank = fixed.mode_error_bank(y)
        values = []
        for mode in range(len(BITS)):
            allocation = torch.full((GROUPS,), mode, device=device)
            error = fixed.error_for_allocation(bank, allocation)
            values.append(decoded_distortion(
                tail, fixed, y, mu, std, teacher, error.unsqueeze(0))[0])
        total += torch.stack(values).double().sum(1).cpu()
        count += len(selected)
    return (total / count).tolist()


def cosine_lr(epoch, epochs, base):
    return float(base) * 0.5 * (1 + math.cos(math.pi * epoch / int(epochs)))


def tau_at(epoch, epochs, start, end):
    if int(epochs) <= 1:
        return float(end)
    progress = epoch / (int(epochs) - 1)
    return float(start * (end / start) ** progress)


def train(args):
    cfg = configure(args.block)
    device = torch.device(args.device)
    engine.configure_precision(False)
    torch.manual_seed(args.seed)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is non-empty")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{args.block}/{args.arm}] load initialization", flush=True)
    model = load_init(args.init_codec, args.arm, device).train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    tail = tail_mod.build_tail(cfg.LAYER, device)
    print(f"[{args.block}/{args.arm}] tail ready", flush=True)
    train_set = CachedFeatureDataset(
        cfg.TRAIN_FEATURES, np.arange(cfg.N_TRAIN, dtype=np.int64))
    loader = DataLoader(
        train_set, batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
        generator=torch.Generator().manual_seed(args.seed))
    teacher_host = torch.from_numpy(np.load(cfg.TRAIN_TEACHERS)).float().pin_memory()
    val_rows = np.arange(min(args.val_images, 500), dtype=np.int64)
    initial = hard_mode_mse(model, tail, cfg, val_rows, device, args.val_batch)
    print(f"[{args.block}/{args.arm}] initial hard={initial}", flush=True)
    validation = [{"epoch": 0, "hard_mode_mse": initial}]
    trace, started, steps = [], time.time(), 0
    initial_V, initial_U = [x.detach().clone() for x in model.rotations()]
    initial_books = [book.detach().clone() for book in model.books]
    stop = False
    for epoch in range(args.epochs):
        lr, tau = cosine_lr(epoch, args.epochs, args.lr), tau_at(
            epoch, args.epochs, args.tau_start, args.tau_end)
        for group in optimizer.param_groups:
            group["lr"] = lr
        for rows, x in loader:
            x = x.float().to(device, non_blocking=True)
            teacher = teacher_host.index_select(0, rows).to(device, non_blocking=True)
            with torch.no_grad():
                y, mu, std = batch_normalize_gpu(x, mode=cfg.NORM_MODE)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            # Three uniform-rate objectives give every registered mode one
            # equally frequent full Tail-MSE update without allocation search.
            for mode in range(len(BITS)):
                decoded = model.reconstruct_mode(y, mode, tau)
                output = tail(batch_inv_normalize_gpu(decoded, mu, std))
                loss = (output - teacher).square().reshape(len(x), -1).sum(-1).mean()
                (loss / len(BITS)).backward()
                losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            steps += 1
            trace.append([steps, *losses, tau, lr])
            if args.max_steps and steps >= args.max_steps:
                stop = True
                break
        if ((epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs):
            hard = hard_mode_mse(model, tail, cfg, val_rows, device, args.val_batch)
            validation.append({"epoch": epoch + 1, "hard_mode_mse": hard,
                               "temperature": tau, "lr": lr})
            print(f"[{args.block}/{args.arm}] epoch={epoch+1}/{args.epochs} "
                  f"hard={hard} tau={tau:.5f}", flush=True)
        if stop:
            break
    final = hard_mode_mse(model, tail, cfg, val_rows, device, args.val_batch)
    fixed = model.fixed().cpu()
    V, U = fixed.V, fixed.U
    payload = {
        "format": "v34_tail_corrected_v1", "block": args.block,
        "arm": args.arm, "bits": list(BITS), "V": V,
        "U": U, "codebooks": [book.detach().cpu() for book in fixed.books]}
    torch.save(payload, out / "codec.pt")
    book_drift = [float((book.detach() - old).norm())
                  for book, old in zip(model.books, initial_books)]
    result = {
        "plan": "v34_tail_mse_corrected", "stage": "train",
        "block": args.block, "arm": args.arm, "objective":
        "mean full Tail MSE of uniform K1/K2/K4 allocations",
        "images": cfg.N_TRAIN, "val_images": len(val_rows),
        "epochs": args.epochs, "steps": steps, "batch": args.batch,
        "lr": args.lr, "tau": [args.tau_start, args.tau_end],
        "initial_hard_mode_mse": initial, "final_hard_mode_mse": final,
        "V_relative_drift": float((V - initial_V.cpu()).norm() / initial_V.cpu().norm()),
        "U_relative_drift": float((U - initial_U.cpu()).norm() / initial_U.cpu().norm()),
        "book_drift": book_drift, "orthogonality": fixed.orthogonality(),
        "validation": validation, "trace": trace,
        "seconds": time.time() - started}
    (out / "train.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in (
        "block", "arm", "final_hard_mode_mse", "V_relative_drift",
        "U_relative_drift", "orthogonality", "seconds")}, indent=2))


def load_fixed(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return TwoSidedPQ(payload["V"], payload["U"], payload["codebooks"],
                      payload["bits"]).to(device).eval()


def image_stats(values):
    values = np.asarray(values, dtype=np.float64)
    sem = values.std(ddof=1) / math.sqrt(len(values))
    return {"mean": float(values.mean()),
            "ci95": [float(values.mean() - 1.96 * sem),
                     float(values.mean() + 1.96 * sem)]}


@torch.no_grad()
def mixed_differences(cfg, codec, tail, rows, device, image_batch,
                      chunk, token_pair_count, seed):
    rng = np.random.default_rng(seed)
    all_token = np.asarray([(i, j) for i in range(257)
                            for j in range(i + 1, 257)], dtype=np.int64)
    token_pairs = all_token[rng.choice(
        len(all_token), size=min(token_pair_count, len(all_token)), replace=False)]
    channel_pairs = np.asarray([(i, j) for i in range(GROUPS)
                                for j in range(i + 1, GROUPS)], dtype=np.int64)
    teacher_source = np.load(cfg.VAL_TEACHERS, mmap_mode="r")
    token_rows, channel_rows, full_rows = [], [], []
    for first in range(0, len(rows), image_batch):
        selected = rows[first:first + image_batch]
        x = next(batches(cfg.VAL_FEATURES, selected, len(selected), device))
        teacher = torch.from_numpy(np.asarray(teacher_source[selected])).float().to(device)
        y, mu, std = batch_normalize_gpu(x, mode=cfg.NORM_MODE)
        bank = codec.mode_error_bank(y)
        allocation = torch.ones(GROUPS, dtype=torch.long, device=device)
        error = codec.error_for_allocation(bank, allocation)
        zero = torch.zeros_like(error)
        d0 = decoded_distortion(tail, codec, y, mu, std, teacher, zero[None])[0]
        full = decoded_distortion(tail, codec, y, mu, std, teacher, error[None])[0]
        full_rows.append(full.cpu())

        token_single = torch.empty(257, len(selected), device=device)
        for start in range(0, 257, chunk):
            ids = range(start, min(start + chunk, 257))
            errors = torch.zeros(len(ids), *error.shape, device=device)
            for local, index in enumerate(ids):
                errors[local, :, index] = error[:, index]
            token_single[start:start + len(ids)] = decoded_distortion(
                tail, codec, y, mu, std, teacher, errors)
        mixed = []
        for start in range(0, len(token_pairs), chunk):
            pairs = token_pairs[start:start + chunk]
            errors = torch.zeros(len(pairs), *error.shape, device=device)
            for local, (left, right) in enumerate(pairs):
                errors[local, :, left] = error[:, left]
                errors[local, :, right] = error[:, right]
            pair_d = decoded_distortion(tail, codec, y, mu, std, teacher, errors)
            left = torch.as_tensor(pairs[:, 0], device=device)
            right = torch.as_tensor(pairs[:, 1], device=device)
            mixed.append(pair_d - token_single[left] - token_single[right] + d0)
        token_rows.append(torch.cat(mixed).abs().mean(0).cpu())

        grouped = error.reshape(len(selected), 257, GROUPS, GROUP_DIM)
        channel_single = torch.empty(GROUPS, len(selected), device=device)
        for start in range(0, GROUPS, chunk):
            ids = range(start, min(start + chunk, GROUPS))
            errors = torch.zeros(len(ids), len(selected), 257, GROUPS,
                                 GROUP_DIM, device=device)
            for local, index in enumerate(ids):
                errors[local, :, :, index] = grouped[:, :, index]
            channel_single[start:start + len(ids)] = decoded_distortion(
                tail, codec, y, mu, std, teacher,
                errors.reshape(len(ids), len(selected), 257, 1024))
        mixed = []
        for start in range(0, len(channel_pairs), chunk):
            pairs = channel_pairs[start:start + chunk]
            errors = torch.zeros(len(pairs), len(selected), 257, GROUPS,
                                 GROUP_DIM, device=device)
            for local, (left, right) in enumerate(pairs):
                errors[local, :, :, left] = grouped[:, :, left]
                errors[local, :, :, right] = grouped[:, :, right]
            pair_d = decoded_distortion(
                tail, codec, y, mu, std, teacher,
                errors.reshape(len(pairs), len(selected), 257, 1024))
            left = torch.as_tensor(pairs[:, 0], device=device)
            right = torch.as_tensor(pairs[:, 1], device=device)
            mixed.append(pair_d - channel_single[left] - channel_single[right] + d0)
        channel_rows.append(torch.cat(mixed).abs().mean(0).cpu())
    full = torch.cat(full_rows).numpy()
    token = torch.cat(token_rows).numpy()
    channel = torch.cat(channel_rows).numpy()
    return {"full": full, "token": token, "channel": channel,
            "token_pairs": token_pairs, "channel_pairs": channel_pairs}


def evaluate(args):
    cfg = configure(args.block)
    device = torch.device(args.device)
    engine.configure_precision(False)
    codec = load_fixed(args.codec, device)
    tail = tail_mod.build_tail(cfg.LAYER, device)
    rows = np.arange(min(args.val_images, 500), dtype=np.int64)
    cost_rows = np.arange(min(args.cost_images, cfg.N_TRAIN), dtype=np.int64)
    cost = evaluate_self_costs(
        cfg, codec, tail, cfg.TRAIN_FEATURES, cfg.TRAIN_TEACHERS,
        cost_rows, device, args.image_batch, args.chunk)
    allocations = random_exact_budget_allocations(
        GROUPS, BITS, 32, args.allocations - 1, seed=args.seed)
    uniform = np.ones((1, GROUPS), dtype=np.int64)
    allocations = allocations[(allocations != uniform[0]).any(1)]
    if len(allocations) != args.allocations - 1:
        raise SystemExit("INVALID_EXPERIMENT: random pool contained uniform")
    allocations = np.concatenate([uniform, allocations], 0)
    measured = evaluate_coupling_and_allocations(
        cfg, codec, tail, rows, allocations, device,
        args.image_batch, args.chunk)
    true_scores = measured["allocation_values"].mean(1)
    predicted = np.asarray([sum(cost[g, int(m)] for g, m in enumerate(row))
                            for row in allocations])
    pd, td = predicted - predicted[0], true_scores - true_scores[0]
    mixed = mixed_differences(
        cfg, codec, tail, rows, device, args.image_batch,
        args.chunk, args.token_pairs, args.seed)
    denominator = max(float(mixed["full"].mean()), 1e-30)
    best = int(predicted.argmin())
    result = {
        "plan": "v34_tail_mse_corrected", "stage": "heldout_evaluation",
        "block": args.block, "arm": args.arm, "rate": 32,
        "data": {"cost_images": len(cost_rows), "val_images": len(rows),
                 "allocations": len(allocations),
                 "token_pairs_sampled": len(mixed["token_pairs"]),
                 "channel_pairs_complete": len(mixed["channel_pairs"])},
        "orthogonality": codec.orthogonality(),
        "interaction": {
            "token_pair_mean_abs_per_image": image_stats(mixed["token"]),
            "channel_pair_mean_abs_per_image": image_stats(mixed["channel"]),
            "token_normalized": float(mixed["token"].mean() / denominator),
            "channel_normalized": float(mixed["channel"].mean() / denominator)},
        "aggregate_residual": {
            "token": expand_stats(measured["token_residual"], measured["full"]),
            "channel": expand_stats(measured["channel_residual"], measured["full"])},
        "separable_allocation": {
            "spearman": spearman(predicted, true_scores),
            "delta_nrmse": float(np.sqrt(np.mean((pd - td) ** 2)) /
                                 max(td.std(), 1e-30)),
            "predicted_best_index": best,
            "predicted_best_true_mse": float(true_scores[best]),
            "uniform_true_mse": float(true_scores[0]),
            "registered_true_best_mse": float(true_scores.min()),
            "registered_true_best_index": int(true_scores.argmin()),
            "predicted_scores": predicted.tolist(),
            "true_scores": true_scores.tolist(),
            "allocations": allocations.tolist(),
            "self_cost_table": cost.tolist()}}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "mixed_differences.npz", **mixed)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"block": args.block, "arm": args.arm,
                      "interaction": result["interaction"],
                      "separable_allocation": {k: result["separable_allocation"][k]
                      for k in ("spearman", "delta_nrmse",
                                "predicted_best_true_mse", "uniform_true_mse",
                                "registered_true_best_mse")}}, indent=2))


def smoke(args):
    device = torch.device(args.device)
    books = [torch.randn(GROUPS, 2 ** bit, GROUP_DIM, device=device)
             for bit in BITS]
    model = TrainableTwoSidedPQ(torch.eye(257), torch.eye(1024), books,
                                train_v=True, train_u=True).to(device)
    y = torch.randn(2, 257, 1024, device=device)
    loss = sum(model.reconstruct_mode(y, mode, 0.5).square().mean()
               for mode in range(len(BITS)))
    loss.backward()
    values = {"loss": float(loss),
              "V_grad": float(model.token.triu_params.grad.norm()),
              "U_grad": float(model.channel.triu_params.grad.norm()),
              "book_grad_min": min(float(x.grad.norm()) for x in model.books)}
    if not all(math.isfinite(x) and x > 0 for x in values.values()):
        raise SystemExit(f"smoke failed: {values}")
    print(json.dumps(values, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--block", choices=("blk05", "blk20"), required=True)
    common.add_argument("--device", default="cuda")
    common.add_argument("--seed", type=int, default=20260809)
    train_p = sub.add_parser("train", parents=[common])
    train_p.add_argument("--arm", choices=ARMS, required=True)
    train_p.add_argument("--init-codec", required=True)
    train_p.add_argument("--output", required=True)
    train_p.add_argument("--epochs", type=int, default=100)
    train_p.add_argument("--batch", type=int, default=32)
    train_p.add_argument("--workers", type=int, default=4)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--tau-start", type=float, default=0.5)
    train_p.add_argument("--tau-end", type=float, default=0.005)
    train_p.add_argument("--val-images", type=int, default=500)
    train_p.add_argument("--val-batch", type=int, default=16)
    train_p.add_argument("--val-every", type=int, default=10)
    train_p.add_argument("--max-steps", type=int, default=0,
                         help="smoke/debug limit; zero runs the full contract")
    eval_p = sub.add_parser("evaluate", parents=[common])
    eval_p.add_argument("--arm", choices=ARMS, required=True)
    eval_p.add_argument("--codec", required=True)
    eval_p.add_argument("--output", required=True)
    eval_p.add_argument("--cost-images", type=int, default=500)
    eval_p.add_argument("--val-images", type=int, default=500)
    eval_p.add_argument("--allocations", type=int, default=256)
    eval_p.add_argument("--token-pairs", type=int, default=512)
    eval_p.add_argument("--image-batch", type=int, default=8)
    eval_p.add_argument("--chunk", type=int, default=4)
    smoke_p = sub.add_parser("smoke")
    smoke_p.add_argument("--device", default="cuda")
    args = parser.parse_args()
    {"train": train, "evaluate": evaluate, "smoke": smoke}[args.command](args)


if __name__ == "__main__":
    main()
