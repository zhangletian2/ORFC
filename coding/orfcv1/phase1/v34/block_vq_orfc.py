"""Matched-rate patch-only ORFC and token-channel block quantizers."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ORFC = Path("/data4/workspace/zlt/featcodec/ORFC/coding/orfc")
ORFCV1 = Path("/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1")
sys.path[:0] = [str(ORFC), str(ORFCV1)]

from opq import (batch_inv_normalize_gpu, batch_normalize_gpu, batched_assign,
                 batched_kmeans, learn_opq_rotation)  # noqa: E402
from soft_pq import FeatureDataset, OrthogonalTransform, SoftPQ, compute_perplexity  # noqa: E402
from phase1 import engine, tail as tail_mod  # noqa: E402
from phase1.v21 import config as v21_config  # noqa: E402
from phase1.v34.block_quantizer import (  # noqa: E402
    DIM, GROUPS, GROUP_DIM, PATCHES, SIDE, BlockVQ, PatchOnlyCodec,
    ProductBlockVQ, TokenPairVQ, blockify,
)
from phase1.v34.token_transform_orfc import tau_at  # noqa: E402


def normalized_patches(features, cfg, device, images):
    rows = []
    for start in range(0, min(images, len(features)), 64):
        x = torch.from_numpy(np.array(features[start:start + 64], copy=True)).float().to(device)
        with torch.no_grad():
            y, _, _ = batch_normalize_gpu(x, mode=cfg.NORM_MODE)
        rows.append(y[:, 1:].cpu())
    return torch.cat(rows).numpy()


def prepare_initialisation(features, cfg, args, device, path):
    patch_images = normalized_patches(features, cfg, device, args.init_images)
    patch = patch_images.reshape(-1, DIM)
    if len(patch) > args.opq_samples:
        patch = patch[np.random.RandomState(args.seed).choice(
            len(patch), args.opq_samples, replace=False)]
    rotation, token_books, _ = learn_opq_rotation(
        patch, GROUPS, GROUP_DIM, 4, args.opq_iters, args.kmeans_iters,
        device=device, verbose=False)
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
        token_books[-1][:, -1] *= -1
    np.savez(path, rotation=rotation, token_books=np.stack(token_books))


@torch.no_grad()
def prepare_block_initialisation(features, cfg, args, device, path):
    patches = normalized_patches(features, cfg, device, args.init_images)
    if args.shared_init:
        rotation = np.asarray(np.load(args.shared_init)["rotation"], dtype=np.float32)
        warm_mse, warmstart = [], "e32_k4"
    else:
        tokens = patches.reshape(-1, DIM)
        if len(tokens) > args.opq_samples:
            ids = np.random.RandomState(args.seed).choice(
                len(tokens), args.opq_samples, replace=False)
            tokens = tokens[ids]
        rotation, _, warm = learn_opq_rotation(
            tokens, 64, 16, 2, args.opq_iters, args.kmeans_iters,
            device=device, verbose=False)
        warm_mse = [row[0] for row in warm]
        warmstart = "e16_k2"
    blocks = blockify(torch.from_numpy(patches).to(device).reshape(
        -1, SIDE, SIDE, GROUPS, GROUP_DIM)).reshape(-1, GROUPS, 4, GROUP_DIM)
    count = min(len(blocks), args.opq_samples // 4)
    blocks = blocks.index_select(0, torch.randperm(len(blocks), device=device)[:count])
    x = blocks.permute(0, 2, 1, 3).reshape(count, 4, DIM)
    r = torch.from_numpy(rotation).to(device)
    pq = ProductBlockVQ().to(device)
    best_mse, best_r, best_books, history = float("inf"), None, None, []
    books = None
    for _ in range(args.opq_iters):
        z = x @ r
        z_blocks = z.reshape(count, 4, GROUPS, GROUP_DIM).permute(
            0, 2, 1, 3).reshape(1, count, GROUPS, 4 * GROUP_DIM)
        sub = pq.split(z_blocks).squeeze(0).permute(1, 0, 2).contiguous()
        books = batched_kmeans(sub, pq.K, args.kmeans_iters, device=device,
                               initial_centroids=books)
        recon, _ = batched_assign(sub, books, device=device)
        merged = pq.merge(recon.permute(1, 0, 2).unsqueeze(0)).squeeze(0)
        zhat = merged.reshape(count, GROUPS, 4, GROUP_DIM).permute(
            0, 2, 1, 3).reshape(count, 4, DIM)
        mse = float((z - zhat).square().mean())
        history.append(mse)
        if mse < best_mse:
            best_mse, best_r, best_books = mse, r.clone(), books.clone()
        cross = x.reshape(-1, DIM).t() @ zhat.reshape(-1, DIM)
        left, _, right = torch.linalg.svd(cross)
        if torch.det(left @ right) < 0:
            left[:, -1] *= -1
        r = left @ right
    np.savez(path, rotation=best_r.cpu().numpy(),
             block_books=best_books.cpu().numpy(),
             block_opq_mse=np.asarray(history), block_count=count,
             warm_opq_mse=np.asarray(warm_mse), warmstart=warmstart)


def initialise(features, cfg, args, device):
    if not args.shared_init:
        raise SystemExit("--shared-init is required for both training arms")
    init = np.load(args.shared_init)
    rotation = np.asarray(init["rotation"], dtype=np.float32)
    token_books = (np.asarray(init["token_books"], dtype=np.float32)
                   if "token_books" in init else None)
    patch_images = normalized_patches(features, cfg, device, args.init_images)
    transform = OrthogonalTransform(DIM).to(device)
    transform.init_from_opq(rotation)
    if args.arm == "orfc":
        pq = SoftPQ(GROUPS, 4, GROUP_DIM).to(device)
        pq.init_codebooks(token_books)
    else:
        quantizer = {"block": BlockVQ, "block_pq": ProductBlockVQ,
                     "pair": TokenPairVQ}[args.arm]
        pq = quantizer().to(device)
        if args.arm == "block_pq" and "block_books" in init:
            books = torch.from_numpy(np.asarray(init["block_books"], dtype=np.float32)).to(device)
        else:
            r = torch.from_numpy(rotation).float().to(device)
            rotated = torch.from_numpy(patch_images).to(device) @ r
            blocks = blockify(rotated.reshape(-1, SIDE, SIDE, GROUPS, GROUP_DIM),
                              block_shape=pq.block_shape)
            sub = pq.split(blocks).permute(2, 0, 1, 3).reshape(pq.G, -1, pq.d)
            if sub.shape[1] > args.kmeans_samples:
                ids = torch.randperm(sub.shape[1], device=device)[:args.kmeans_samples]
                sub = sub.index_select(1, ids)
            books = batched_kmeans(sub, pq.K, args.kmeans_iters, device=device)
        pq.init_codebooks(books)
    return PatchOnlyCodec(transform, pq, args.arm).to(device)


@torch.no_grad()
def evaluate(codec, features, teachers, cfg, tail, device, batch):
    codec.eval()
    total, cls_error = 0.0, 0.0
    for start in range(0, len(features), batch):
        x = torch.from_numpy(np.array(features[start:start + batch], copy=True)).float().to(device)
        teacher = torch.from_numpy(np.array(teachers[start:start + batch], copy=True)).float().to(device)
        y, mu, std = batch_normalize_gpu(x, mode=cfg.NORM_MODE)
        yhat, _ = codec(y)
        cls_error = max(cls_error, float((yhat[:, 0] - y[:, 0]).abs().max()))
        output = tail.forward_nograd(batch_inv_normalize_gpu(yhat, mu, std))
        total += (output - teacher).square().reshape(len(x), -1).sum(-1).sum().item()
    return total / len(features), cls_error


def train(args):
    cfg, device = v21_config.activate(args.block), torch.device(args.device)
    tail_mod.check_layer(cfg.LAYER, cfg.TRAIN_FEATURES, cfg.TRAIN_TEACHERS, cfg.VAL_FEATURES, cfg.VAL_TEACHERS)
    engine.configure_precision(False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is non-empty")
    out.mkdir(parents=True, exist_ok=True)
    features = np.load(cfg.TRAIN_FEATURES, mmap_mode="r")[:args.train_images]
    if args.arm in ("prepare", "prepare_block_pq"):
        prepare = (prepare_initialisation if args.arm == "prepare"
                   else prepare_block_initialisation)
        prepare(features, cfg, args, device, out / "shared_init.npz")
        print(out / "shared_init.npz")
        return
    teachers = np.load(cfg.TRAIN_TEACHERS, mmap_mode="r")[:args.train_images]
    val_x = np.load(cfg.VAL_FEATURES, mmap_mode="r")[:args.val_images]
    val_y = np.load(cfg.VAL_TEACHERS, mmap_mode="r")[:args.val_images]
    codec = initialise(features, cfg, args, device)
    tail = tail_mod.build_tail(cfg.LAYER, device)
    initial, cls_initial = evaluate(codec, val_x, val_y, cfg, tail, device, args.batch)
    loader = DataLoader(FeatureDataset(np.asarray(features), teacher_cache=np.asarray(teachers)),
                        batch_size=args.batch, shuffle=True, num_workers=2,
                        pin_memory=True, persistent_workers=True)
    opt = torch.optim.Adam(codec.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr * .01)
    history, steps, started = [], 0, time.time()
    for epoch in range(args.epochs):
        codec.train()
        codec.pq.temperature = tau_at(epoch, args.epochs, args.tau_start,
                                      args.tau_end, "exponential")
        total, seen = 0.0, 0
        usage = torch.zeros(codec.pq.G, codec.pq.K, device=device)
        for x, teacher in loader:
            x, teacher = x.to(device, non_blocking=True), teacher.to(device, non_blocking=True)
            y, mu, std = batch_normalize_gpu(x, mode=cfg.NORM_MODE)
            yhat, used = codec(y)
            output = tail(batch_inv_normalize_gpu(yhat, mu, std))
            loss = (output - teacher).square().sum() / len(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), args.grad_clip)
            opt.step()
            total += loss.item() * len(x)
            seen += len(x)
            usage += used.detach()
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        sched.step()
        val = None
        if epoch % args.val_every == 0 or epoch + 1 == args.epochs:
            val, _ = evaluate(codec, val_x, val_y, cfg, tail, device, args.batch)
        history.append({"epoch": epoch, "train_tail_mse": total / seen,
                        "val_tail_mse": val, "temperature": codec.pq.temperature,
                        "perplexity": compute_perplexity(usage),
                        "dead": int((usage == 0).sum()), "lr": opt.param_groups[0]["lr"]})
        if args.max_steps and steps >= args.max_steps:
            break
    final, cls_final = evaluate(codec, val_x, val_y, cfg, tail, device, args.batch)
    nominal = PATCHES * GROUPS * 2
    if args.arm != "orfc":
        positions = PATCHES // codec.pq.block_tokens
        nominal = positions * codec.pq.G * int(math.log2(codec.pq.K))
    if nominal != 16384 or max(cls_initial, cls_final) != 0:
        raise SystemExit("INVALID_EXPERIMENT: rate mismatch or CLS was modified")
    variant = {"orfc": "orfc_k4", "block": "block_k256",
               "block_pq": "block_pq_2xk16",
               "pair": "token_pair_k16"}[args.arm]
    torch.save({"format": "v34_patch_block_vq", "arm": args.arm,
                "variant": variant,
                "state_dict": codec.state_dict()}, out / "codec.pt")
    result = {"block": args.block, "arm": args.arm, "train_images": len(features),
              "val_images": len(val_x), "steps": steps, "epochs": args.epochs,
              "batch": args.batch, "nominal_patch_bits_per_image": nominal,
              "quantizer_variant": variant,
              "codebook_parameters": int(codec.pq.codebooks.numel()),
              "shared_init": str(args.shared_init),
              "cls_bypass_max_error": cls_final, "initial_val_tail_mse": initial,
              "final_val_tail_mse": final, "history": history,
              "seconds": time.time() - started}
    (out / "train.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("arm", "nominal_patch_bits_per_image",
                                             "cls_bypass_max_error", "initial_val_tail_mse",
                                             "final_val_tail_mse")}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True,
                   choices=("prepare", "prepare_block_pq", "orfc", "block",
                            "block_pq", "pair"))
    p.add_argument("--block", default="blk20", choices=("blk05", "blk10", "blk15", "blk20"))
    p.add_argument("--train-images", type=int, default=5000)
    p.add_argument("--val-images", type=int, default=500)
    p.add_argument("--init-images", type=int, default=512)
    p.add_argument("--opq-samples", type=int, default=100000)
    p.add_argument("--kmeans-samples", type=int, default=20000)
    p.add_argument("--opq-iters", type=int, default=20)
    p.add_argument("--kmeans-iters", type=int, default=50)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--tau-start", type=float, default=.5)
    p.add_argument("--tau-end", type=float, default=.005)
    p.add_argument("--grad-clip", type=float, default=1.)
    p.add_argument("--val-every", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shared-init")
    p.add_argument("--output", required=True)
    train(p.parse_args())


if __name__ == "__main__":
    main()
