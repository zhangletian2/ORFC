"""Fixed-allocation continuation used to isolate codeword ST temperature."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from codec_v1 import load_codec_v1, save_codec_v1
from opq import batch_normalize_gpu

from . import config as C
from . import qhard
from . import train as joint
from .. import engine, frozen
from .. import tail as tail_mod


def temperature_at(step, total, start, end):
    if total <= 1 or start == end:
        return float(end)
    progress = (step - 1) / (total - 1)
    return float(start * (end / start) ** progress)


def run(anchor, source, run_id, device, epochs=20, steps=None,
        batch=C.DEFAULT_BATCH, lr_u=C.DEFAULT_LR_U,
        lr_theta=C.DEFAULT_LR_THETA, tau_start=0.5, tau_end=0.005,
        num_workers=C.DATALOADER_WORKERS, log=print):
    source = Path(source)
    out = C.output_dir(anchor, run_id)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is non-empty; refusing to overwrite")
    for name in ("codec.pt", "allocation.npy", "train.json"):
        if not (source / name).exists():
            raise FileNotFoundError(source / name)
    out.mkdir(parents=True, exist_ok=True)
    engine.configure_precision(C.ALLOW_TF32)

    codec = load_codec_v1(source / "codec.pt", device=device)
    allocation = torch.from_numpy(np.load(source / "allocation.npy")).long().to(device)
    bits = joint.actual_mode_bits(codec)
    if allocation.numel() != C.GROUPS:
        raise SystemExit("INVALID_EXPERIMENT: allocation group count")
    rate = sum(bits[int(mode)] for mode in allocation)
    if rate != anchor.rate:
        raise SystemExit(f"INVALID_EXPERIMENT: rate {rate} != {anchor.rate}")
    joint.make_trainable(codec)
    initial_rotation = codec.transform.rotation.detach().clone()
    initial_books = [q.codebooks.detach().clone() for q in codec.pq.quantizers]
    rotation_opt, book_opt = joint.build_optimizers(
        codec, lr_u, lr_theta, book_optimizer="adam")

    tail = tail_mod.build_tail(C.LAYER, device)
    train_paths = C.load_split("train_fit")
    train_set = joint.CachedFeatureDataset(train_paths[0], train_paths[2])
    teacher_host = torch.from_numpy(np.load(train_paths[1])).float()
    if device.type == "cuda":
        teacher_host = teacher_host.pin_memory()
    loader = DataLoader(
        train_set, batch_size=int(batch), shuffle=True, drop_last=True,
        num_workers=int(num_workers), pin_memory=device.type == "cuda",
        persistent_workers=int(num_workers) > 0,
        prefetch_factor=4 if int(num_workers) > 0 else None,
        generator=torch.Generator().manual_seed(C.TRAIN_SEED))
    per_epoch = len(loader)
    total = int(steps) if steps is not None else int(epochs) * per_epoch
    if total < 1:
        raise ValueError("training requires at least one step")
    val_paths = C.load_split("train_val")
    val = engine.ResidentSet(*val_paths[:3], device)
    initial_parity = qhard.selfcheck(codec, tail, val, allocation)
    initial_mse = joint.validate(codec, tail, val, allocation)
    initial_orth = frozen.orthogonality_error(codec)
    scale = initial_mse
    validation = [{"step": 0, "hard_mse": initial_mse}]
    trace = []
    iterator = iter(loader)
    started = time.time()

    for step in range(1, total + 1):
        try:
            rows, x = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            rows, x = next(iterator)
        x = x.float().to(device, non_blocking=True)
        teacher = teacher_host.index_select(0, rows).to(device, non_blocking=True)
        with torch.no_grad():
            y, mu, std = batch_normalize_gpu(x, mode=C.NORM_MODE)
        tau = temperature_at(step, total, tau_start, tau_end)
        for optimizer, base in ((rotation_opt, lr_u), (book_opt, lr_theta)):
            lr = C.cosine_lr(step - 1, total, base)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
        per_image, labels = qhard.distortions(
            codec, tail, y, mu, std, teacher, allocation[None],
            codeword_temperature=tau)
        loss = per_image.mean() / scale
        loss.backward()
        torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
        rotation_opt.step()
        book_opt.step()
        if step % C.REVIVE_EVERY == 0:
            qhard.revive_dead_codewords(codec, y, allocation, labels[0])
        trace.append([step, float(per_image.mean().detach()), tau])
        if step % C.VAL_EVERY == 0 or step == total:
            hard_mse = joint.validate(codec, tail, val, allocation)
            validation.append({"step": step, "hard_mse": hard_mse,
                               "temperature": tau})
            log(f"[{anchor.name}] {step}/{total} train={trace[-1][1]:.1f} "
                f"hard={hard_mse:.1f} tau={tau:.5f}")

    final_mse = joint.validate(codec, tail, val, allocation)
    final_parity = qhard.selfcheck(codec, tail, val, allocation)
    final_orth = frozen.orthogonality_error(codec)
    if final_orth - initial_orth > C.ORTH_TOL:
        raise SystemExit("INVALID_EXPERIMENT: orthogonality drift")
    selected = torch.zeros(C.GROUPS, len(bits), dtype=torch.bool)
    selected[torch.arange(C.GROUPS), allocation.cpu()] = True
    book_drift = torch.stack([
        (q.codebooks.detach() - old).flatten(1).norm(dim=1)
        for q, old in zip(codec.pq.quantizers, initial_books)], dim=1).cpu()
    payload = {
        "plan": "fixed-allocation temperature isolation",
        "anchor": anchor.name, "rate": anchor.rate, "source": str(source),
        "allocation": allocation.cpu().tolist(), "epochs": float(epochs),
        "steps": total, "batch": int(batch), "lr_U": float(lr_u),
        "lr_theta": float(lr_theta), "tau_start": float(tau_start),
        "tau_end": float(tau_end), "initial_hard_mse": initial_mse,
        "final_hard_mse": final_mse,
        "relative_change": final_mse / initial_mse - 1.0,
        "hard_parity_initial": initial_parity, "hard_parity_final": final_parity,
        "orthogonality_initial": initial_orth, "orthogonality_final": final_orth,
        "rotation_relative_drift": float(
            (codec.transform.rotation.detach() - initial_rotation).norm()
            / initial_rotation.norm()),
        "selected_book_drift_min": float(book_drift[selected].min()),
        "selected_book_drift_max": float(book_drift[selected].max()),
        "validation": validation, "trace": trace,
        "training_seconds": time.time() - started}
    save_codec_v1(codec, out / "codec.pt")
    np.save(out / "allocation.npy", allocation.cpu().numpy())
    (out / "train.json").write_text(json.dumps(payload, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", required=True, choices=list(C.ANCHOR_BY_NAME))
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=float, default=20)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch", type=int, default=C.DEFAULT_BATCH)
    parser.add_argument("--lr-u", type=float, default=C.DEFAULT_LR_U)
    parser.add_argument("--lr-theta", type=float, default=C.DEFAULT_LR_THETA)
    parser.add_argument("--tau-start", type=float, default=0.5)
    parser.add_argument("--tau-end", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=C.DATALOADER_WORKERS)
    args = parser.parse_args()
    payload = run(C.ANCHOR_BY_NAME[args.anchor], args.source,
                  args.run_id, torch.device(args.device), epochs=args.epochs,
                  steps=args.steps, batch=args.batch, lr_u=args.lr_u,
                  lr_theta=args.lr_theta, tau_start=args.tau_start,
                  tau_end=args.tau_end, num_workers=args.num_workers)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
