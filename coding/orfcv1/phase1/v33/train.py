"""V33 two-phase training: strict-fair supernet, then frozen-U0 specialisation.

Phase 1 streams the exact-budget strict-fair slate (no policy / REINFORCE),
updates full ``U0``, every ``L[g,t]`` cell (with small weight decay), and the
nested tree.  Phase 2 loads a searched allocation, freezes ``U0``, and trains
only ``L`` + nested PQ on that fixed assignment.

Checkpoint I/O for the search agent lives in :mod:`phase1.v33.checkpoint`.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from opq import batch_normalize_gpu

from .. import engine, frozen
from .. import tail as tail_mod
from ..v12 import init as v12_init
from ..v12.strict_fair import strict_fair_slate
from ..v21.config import SPECS
from ..v30.nested import NestedMultiModePQ
from . import checkpoint as ckpt
from . import config as C
from .codec import V33Codec
from .distortion import distortion_sparse, evaluate as evaluate_allocations


class CachedFeatureDataset(Dataset):
    def __init__(self, features, rows, max_images=None):
        self.features = np.load(features, mmap_mode="r")
        self.rows = np.asarray(rows, dtype=np.int64)
        if max_images is not None:
            self.rows = self.rows[:max_images]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        feature = torch.from_numpy(np.array(self.features[row], copy=True))
        return int(row), feature


def codeword_tau(step, total, start, end):
    end = start if end is None else end
    if start == end:
        return float(start)
    if start <= 0 or end <= 0:
        raise ValueError("annealed codeword temperatures must be positive")
    progress = (step - 1) / max(total - 1, 1)
    return float(start * (end / start) ** progress)


def set_lr(optimizer, step, total, base):
    value = C.cosine_lr(step, total, base)
    for group in optimizer.param_groups:
        group["lr"] = value


def build_optimizer(codec, lr, l_decay):
    groups = codec.parameter_groups(l_decay=float(l_decay))
    return torch.optim.Adam(groups, lr=float(lr))


def codec_from_opq(anchor, device, parameterization="orfc_cayley",
                   require_full=True, use_L=True):
    """OPQ init → nested tree + identity ``L`` (v12 init path)."""
    source, meta = v12_init.load_checked(
        anchor, device, parameterization=parameterization,
        require_full=require_full)
    bits = tuple(anchor.mode_bits)
    pq = NestedMultiModePQ(source.pq.G, bits, source.pq.d).to(device)
    pq.init_from_independent(source.pq)
    codec = V33Codec(
        copy.deepcopy(source.transform), pq, use_L=bool(use_L)).to(device)
    for parameter in codec.parameters():
        parameter.requires_grad_(True)
    return codec, meta


def make_loader(train_set, batch, device, num_workers, seed):
    kwargs = dict(
        batch_size=int(batch), shuffle=True, drop_last=True,
        num_workers=max(0, int(num_workers)),
        pin_memory=(device.type == "cuda"),
        generator=torch.Generator().manual_seed(int(seed)))
    if kwargs["num_workers"] > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(train_set, **kwargs)


@torch.no_grad()
def validate(codec, tail, resident, allocation, image_batch=None):
    image_batch = C.EVAL_IMAGE_BATCH if image_batch is None else int(image_batch)
    matrix = evaluate_allocations(
        codec, tail, resident, allocation, image_batch=image_batch)
    return float(matrix.mean())


def stream_slate_loss(codec, tail, y, mu, std, teacher, allocations,
                      temperature, image_microbatch=0):
    """Equal-weight slate stream; accumulates grads; returns mean distortion."""
    n = int(y.shape[0])
    micro = n if int(image_microbatch) <= 0 else int(image_microbatch)
    count = int(allocations.shape[0])
    weight = 1.0 / count
    values = []
    for allocation in allocations:
        pieces = []
        for first in range(0, n, micro):
            last = min(first + micro, n)
            value, _ = distortion_sparse(
                codec, tail, y[first:last], mu[first:last],
                std[first:last], teacher[first:last], allocation,
                temperature=temperature)
            (weight * value.mean() * ((last - first) / n)).backward()
            pieces.append(value.detach())
        values.append(torch.cat(pieces).mean())
    return torch.stack(values)


def stream_fixed_loss(codec, tail, y, mu, std, teacher, allocation,
                      temperature, image_microbatch=0):
    n = int(y.shape[0])
    micro = n if int(image_microbatch) <= 0 else int(image_microbatch)
    pieces = []
    for first in range(0, n, micro):
        last = min(first + micro, n)
        value, _ = distortion_sparse(
            codec, tail, y[first:last], mu[first:last],
            std[first:last], teacher[first:last], allocation,
            temperature=temperature)
        (value.mean() * ((last - first) / n)).backward()
        pieces.append(value.detach())
    return torch.cat(pieces).mean()


def _prepare_io(anchor, run_id, device, images, batch, num_workers):
    out = C.output_dir(anchor, C.ensure_run_id(run_id))
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is non-empty; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    engine.configure_precision(C.ALLOW_TF32)
    train_paths = C.load_split("train_fit")
    tail_mod.check_layer(C.LAYER, train_paths[0], train_paths[1])
    train_set = CachedFeatureDataset(
        train_paths[0], train_paths[2], max_images=images)
    teacher_host = torch.from_numpy(np.load(train_paths[1])).float()
    if device.type == "cuda":
        teacher_host = teacher_host.pin_memory()
    val_paths = C.load_split("train_val")
    val_n = None if images is None else min(int(images), C.N_VAL)
    val = engine.ResidentSet(
        *val_paths[:3], device, max_images=val_n)
    loader = make_loader(
        train_set, batch, device, num_workers, C.TRAIN_SEED)
    if len(loader) == 0:
        raise ValueError(f"batch {batch} exceeds {len(train_set)} images")
    teacher_staging = torch.empty(
        (int(batch), *teacher_host.shape[1:]),
        dtype=teacher_host.dtype,
        pin_memory=(device.type == "cuda"))
    tail = tail_mod.build_tail(C.LAYER, device)
    return out, train_set, teacher_host, teacher_staging, loader, val, tail


def _next_batch(iterator, loader, teacher_host, teacher_staging, device):
    try:
        rows, x = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        rows, x = next(iterator)
    x = x.float().to(device, non_blocking=True)
    rows_cpu = rows if rows.device.type == "cpu" else rows.cpu()
    n = int(rows_cpu.shape[0])
    torch.index_select(teacher_host, 0, rows_cpu, out=teacher_staging[:n])
    teacher = teacher_staging[:n].to(device, non_blocking=True)
    with torch.no_grad():
        y, mu, std = batch_normalize_gpu(x, mode=C.NORM_MODE)
    return iterator, y, mu, std, teacher


def run_phase1(anchor, run_id, device, *, epochs=C.DEFAULT_EPOCHS, steps=None,
               batch=C.DEFAULT_BATCH, lr=C.DEFAULT_LR, l_decay=C.DEFAULT_L_DECAY,
               tau_start=C.DEFAULT_TAU_START, tau_end=C.DEFAULT_TAU_END,
               images=None, num_workers=C.DATALOADER_WORKERS,
               image_microbatch=0, parameterization="orfc_cayley",
               grad_clip=C.GRAD_CLIP, ckpt_every=C.CKPT_EVERY,
               resume=None, use_L=True, log=print):
    """Strict-fair supernet: slate stream, no policy."""
    started = time.time()
    if parameterization != "orfc_cayley":
        raise ValueError("phase-1 Adam path expects orfc_cayley U0 (triu_params)")
    out, train_set, teacher_host, teacher_staging, loader, val, tail = \
        _prepare_io(anchor, run_id, device, images, batch, num_workers)
    per_epoch = len(loader)
    schedule_total = int(epochs) * per_epoch
    total = int(steps) if steps is not None else schedule_total
    if total < 1:
        raise ValueError("training requires at least one step")

    start_step = 1
    if resume is not None:
        codec, payload = ckpt.load_checkpoint(
            resume, device=device, train=True)
        if tuple(codec.pq.mode_bits) != tuple(anchor.mode_bits):
            raise SystemExit("INVALID_EXPERIMENT: resume menu mismatch")
        meta_payload = payload.get("meta") or {}
        start_step = int(meta_payload.get("step", 0)) + 1
        meta = meta_payload.get("opq_init", {"resumed": True})
        optimizer = build_optimizer(codec, lr, l_decay)
        if payload.get("optimizer_state") is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
        log(f"[{anchor.name}] resumed phase1 from step {meta_payload.get('step')}")
    else:
        codec, meta = codec_from_opq(
            anchor, device, parameterization=parameterization,
            require_full=images is None, use_L=use_L)
        optimizer = build_optimizer(codec, lr, l_decay)

    bits = tuple(codec.pq.mode_bits)
    if bits != tuple(anchor.mode_bits):
        raise SystemExit(f"INVALID_EXPERIMENT: menu {bits} != {anchor.mode_bits}")
    if not hasattr(codec.transform, "triu_params"):
        raise ValueError("orfc_cayley transform required for phase-1 Adam")

    uniform = torch.as_tensor(
        engine.uniform_allocation(anchor, C.GROUPS), device=device)
    initial_u0_orth = frozen.orthogonality_error(codec)
    initial_L_orth = codec.l_orth_error()
    initial_mse = validate(codec, tail, val, uniform)
    generator = torch.Generator(device=device).manual_seed(C.SLATE_SEED)
    slate0 = strict_fair_slate(
        C.GROUPS, bits, anchor.rate, device, generator)
    coverage = torch.zeros(C.GROUPS, len(bits), dtype=torch.long, device=device)
    validation = [{"step": start_step - 1, "uniform_mse": initial_mse,
                   "slate_mean_mse": None}]
    trace = []
    iterator = iter(loader)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    training_started = time.time()

    for step in range(start_step, total + 1):
        iterator, y, mu, std, teacher = _next_batch(
            iterator, loader, teacher_host, teacher_staging, device)
        epoch_index = (step - 1) // per_epoch
        set_lr(optimizer, epoch_index, int(epochs), lr)
        tau = codeword_tau(
            epoch_index + 1, int(epochs), tau_start, tau_end)
        allocations = strict_fair_slate(
            C.GROUPS, bits, anchor.rate, device, generator)
        for allocation in allocations:
            coverage[torch.arange(C.GROUPS, device=device), allocation] += 1
        optimizer.zero_grad(set_to_none=True)
        per_alloc = stream_slate_loss(
            codec, tail, y, mu, std, teacher, allocations, tau,
            image_microbatch=image_microbatch)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in codec.parameters() if p.requires_grad],
                float(grad_clip))
        optimizer.step()
        train_mean = float(per_alloc.mean().detach())
        trace.append({
            "step": step, "train_slate_mean": train_mean,
            "train_slate": [float(v) for v in per_alloc.detach()],
            "tau": float(tau),
            "lr": float(optimizer.param_groups[0]["lr"]),
        })
        if step == 1 or step % C.LOG_EVERY == 0 or step == total:
            log(f"[{anchor.name}/p1] {step}/{total} "
                f"Dslate={train_mean:.1f} tau={tau:.5f} "
                f"cov_min={int(coverage.min())}")
        if step % C.VAL_EVERY == 0 or step == total:
            uniform_mse = validate(codec, tail, val, uniform)
            slate_vals = [
                validate(codec, tail, val, row) for row in slate0]
            record = {
                "step": step, "uniform_mse": uniform_mse,
                "slate_mean_mse": float(np.mean(slate_vals)),
                "slate_mse": slate_vals, "tau": float(tau),
                "L_orth": codec.l_orth_error(),
            }
            validation.append(record)
            log(f"[{anchor.name}/p1] val step={step} "
                f"uniform={uniform_mse:.1f} "
                f"slate_mean={record['slate_mean_mse']:.1f}")
        if ckpt_every > 0 and (step % int(ckpt_every) == 0 or step == total):
            ckpt.save_checkpoint(
                codec, out / f"checkpoint_step{step:06d}.pt",
                meta={
                    "step": step, "phase": 1, "anchor": anchor.name,
                    "run_id": str(run_id), "l_decay": float(l_decay),
                    "allocation": None, "u0_frozen": False,
                    "opq_init": meta if isinstance(meta, dict) else {},
                },
                optimizer=optimizer)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    final_mse = validate(codec, tail, val, uniform)
    final_L_orth = codec.l_orth_error()
    ckpt.save_checkpoint(
        codec, out / "checkpoint.pt",
        meta={
            "step": total, "phase": 1, "anchor": anchor.name,
            "run_id": str(run_id), "l_decay": float(l_decay),
            "allocation": None, "u0_frozen": False,
            "opq_init": meta if isinstance(meta, dict) else {},
        },
        optimizer=optimizer)
    # Convenience alias for the search agent.
    (out / "latest.pt").write_bytes((out / "checkpoint.pt").read_bytes())
    payload = {
        "plan": "v33_phase1_strict_fair",
        "phase": 1,
        "anchor": anchor.name,
        "rate": anchor.rate,
        "run_id": str(run_id),
        "parameterization": parameterization,
        "images": len(train_set),
        "val_images": val.count,
        "batch": int(batch),
        "epochs": float(total / per_epoch),
        "steps": total,
        "steps_per_epoch": per_epoch,
        "schedule_epochs": int(epochs),
        "lr": float(lr),
        "l_decay": float(l_decay),
        "tau_start": float(tau_start),
        "tau_end": float(tau_end),
        "image_microbatch": int(image_microbatch),
        "coverage": coverage.cpu().tolist(),
        "coverage_min": int(coverage.min()),
        "initial_uniform_mse": initial_mse,
        "final_uniform_mse": final_mse,
        "initial_u0_orth": initial_u0_orth,
        "initial_L_orth": initial_L_orth,
        "final_u0_orth": frozen.orthogonality_error(codec),
        "final_L_orth": final_L_orth,
        "slate0": slate0.detach().cpu().tolist(),
        "validation": validation,
        "trace": trace,
        "training_seconds": time.time() - training_started,
        "seconds": time.time() - started,
        "peak_memory_gb": (
            torch.cuda.max_memory_allocated(device) / 2 ** 30
            if device.type == "cuda" else 0.0),
        "checkpoint": str(out / "checkpoint.pt"),
    }
    (out / "train.json").write_text(json.dumps(payload, indent=2))
    log(f"[{anchor.name}/p1] complete in {payload['seconds']:.1f}s "
        f"→ {out / 'checkpoint.pt'}")
    return payload


def run_phase2(anchor, run_id, device, *, source, allocation,
               epochs=C.DEFAULT_EPOCHS, steps=None, batch=C.DEFAULT_BATCH,
               lr=C.DEFAULT_LR, l_decay=0.0,
               tau_start=C.DEFAULT_TAU_START, tau_end=C.DEFAULT_TAU_END,
               images=None, num_workers=C.DATALOADER_WORKERS,
               image_microbatch=0, parameterization="orfc_cayley",
               grad_clip=C.GRAD_CLIP, ckpt_every=C.CKPT_EVERY,
               resume=None, freeze_u0=True, drop_L=False, log=print):
    """Specialise on a fixed allocation (U0 frozen by default).

    Pass ``freeze_u0=False`` for the thaw ablation that jointly updates U0,
    L, and the nested tree.  Pass ``drop_L=True`` to fold the used ``L`` blocks
    into the tree and continue without the bank; the fold is exact, so the
    reported ``initial_hard_mse`` must match the un-dropped run.
    """
    started = time.time()
    out, train_set, teacher_host, teacher_staging, loader, val, tail = \
        _prepare_io(anchor, run_id, device, images, batch, num_workers)
    per_epoch = len(loader)
    schedule_total = int(epochs) * per_epoch
    total = int(steps) if steps is not None else schedule_total
    if total < 1:
        raise ValueError("training requires at least one step")

    start_step = 1
    payload = {}
    if resume is not None:
        codec, payload = ckpt.load_checkpoint(
            resume, device=device, train=True)
        meta_payload = payload.get("meta") or {}
        start_step = int(meta_payload.get("step", 0)) + 1
        if allocation is None:
            allocation = ckpt.meta_allocation(payload)
        parameterization = payload["geometry"].get(
            "parameterization", parameterization)
        log(f"[{anchor.name}] resumed phase2 from step {meta_payload.get('step')}")
    else:
        source_path = Path(source)
        ckpt_path = (source_path if source_path.is_file()
                     else source_path / "checkpoint.pt")
        if not ckpt_path.exists():
            latest = source_path / "latest.pt"
            ckpt_path = latest if latest.exists() else ckpt_path
        codec, payload = ckpt.load_checkpoint(
            ckpt_path, device=device, train=True)
        parameterization = payload["geometry"].get(
            "parameterization", parameterization)
        if allocation is None:
            allocation = ckpt.meta_allocation(payload)

    if allocation is None:
        raise SystemExit(
            "phase 2 requires --allocation (search result) or a checkpoint "
            "that already stores one")
    allocation = ckpt.load_allocation(
        allocation, groups=C.GROUPS, device=device)
    rate = ckpt.allocation_rate(allocation, codec.pq.mode_bits)
    if rate != anchor.rate:
        raise SystemExit(
            f"INVALID_EXPERIMENT: allocation rate {rate} != {anchor.rate}")
    if tuple(codec.pq.mode_bits) != tuple(anchor.mode_bits):
        raise SystemExit("INVALID_EXPERIMENT: menu mismatch")

    if drop_L:
        codec.fold_L_into_codebooks(allocation)
    if freeze_u0:
        codec.freeze_u0()
    else:
        codec.unfreeze_u0()
    # Unused L cells receive no grads from a fixed allocation; keep l_decay at
    # 0 by default so weight decay does not pull them off the switch-point.
    optimizer = build_optimizer(codec, lr, l_decay)
    if resume is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])

    initial_mse = validate(codec, tail, val, allocation)
    initial_L_orth = codec.l_orth_error()
    validation = [{"step": start_step - 1, "hard_mse": initial_mse}]
    trace = []
    iterator = iter(loader)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    training_started = time.time()
    u0_frozen_flag = bool(freeze_u0)

    for step in range(start_step, total + 1):
        iterator, y, mu, std, teacher = _next_batch(
            iterator, loader, teacher_host, teacher_staging, device)
        epoch_index = (step - 1) // per_epoch
        set_lr(optimizer, epoch_index, int(epochs), lr)
        tau = codeword_tau(
            epoch_index + 1, int(epochs), tau_start, tau_end)
        optimizer.zero_grad(set_to_none=True)
        train_value = stream_fixed_loss(
            codec, tail, y, mu, std, teacher, allocation, tau,
            image_microbatch=image_microbatch)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in codec.parameters() if p.requires_grad],
                float(grad_clip))
        # Belt-and-suspenders: clear grads on frozen U0 if any leaked.
        if freeze_u0:
            for parameter in codec.transform.parameters():
                parameter.grad = None
        optimizer.step()
        trace.append({
            "step": step, "train": float(train_value.detach()),
            "tau": float(tau),
            "lr": float(optimizer.param_groups[0]["lr"]),
        })
        if step == 1 or step % C.LOG_EVERY == 0 or step == total:
            log(f"[{anchor.name}/p2] {step}/{total} "
                f"D={float(train_value):.1f} tau={tau:.5f}")
        if step % C.VAL_EVERY == 0 or step == total:
            hard = validate(codec, tail, val, allocation)
            validation.append({
                "step": step, "hard_mse": hard, "tau": float(tau),
                "L_orth": codec.l_orth_error(),
            })
            log(f"[{anchor.name}/p2] val step={step} hard={hard:.1f}")
        if ckpt_every > 0 and (step % int(ckpt_every) == 0 or step == total):
            ckpt.save_checkpoint(
                codec, out / f"checkpoint_step{step:06d}.pt",
                meta={
                    "step": step, "phase": 2, "anchor": anchor.name,
                    "run_id": str(run_id), "l_decay": float(l_decay),
                    "u0_frozen": u0_frozen_flag,
                    **({} if freeze_u0 else {"ablation": "u0_unfrozen"}),
                },
                allocation=allocation, optimizer=optimizer)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    final_mse = validate(codec, tail, val, allocation)
    if freeze_u0 and not codec.u0_frozen:
        raise SystemExit("INVALID_EXPERIMENT: U0 thawed during phase 2")
    if (not freeze_u0) and codec.u0_frozen:
        raise SystemExit("INVALID_EXPERIMENT: U0 frozen during thaw ablation")
    ckpt.save_checkpoint(
        codec, out / "checkpoint.pt",
        meta={
            "step": total, "phase": 2, "anchor": anchor.name,
            "run_id": str(run_id), "l_decay": float(l_decay),
            "u0_frozen": u0_frozen_flag,
            **({} if freeze_u0 else {"ablation": "u0_unfrozen"}),
        },
        allocation=allocation, optimizer=optimizer)
    (out / "latest.pt").write_bytes((out / "checkpoint.pt").read_bytes())
    np.save(out / "allocation.npy", allocation.detach().cpu().numpy())
    payload = {
        "plan": "v33_phase2_specialise",
        "phase": 2,
        "anchor": anchor.name,
        "rate": anchor.rate,
        "run_id": str(run_id),
        "source": str(source),
        "allocation": allocation.detach().cpu().tolist(),
        "parameterization": parameterization,
        "images": len(train_set),
        "val_images": val.count,
        "batch": int(batch),
        "epochs": float(total / per_epoch),
        "steps": total,
        "steps_per_epoch": per_epoch,
        "schedule_epochs": int(epochs),
        "lr": float(lr),
        "l_decay": float(l_decay),
        "tau_start": float(tau_start),
        "tau_end": float(tau_end),
        "u0_frozen": u0_frozen_flag,
        "ablation": None if freeze_u0 else "u0_unfrozen",
        "initial_hard_mse": initial_mse,
        "final_hard_mse": final_mse,
        "initial_L_orth": initial_L_orth,
        "final_L_orth": codec.l_orth_error(),
        "validation": validation,
        "trace": trace,
        "training_seconds": time.time() - training_started,
        "seconds": time.time() - started,
        "peak_memory_gb": (
            torch.cuda.max_memory_allocated(device) / 2 ** 30
            if device.type == "cuda" else 0.0),
        "checkpoint": str(out / "checkpoint.pt"),
    }
    (out / "train.json").write_text(json.dumps(payload, indent=2))
    log(f"[{anchor.name}/p2] complete in {payload['seconds']:.1f}s "
        f"→ {out / 'checkpoint.pt'}")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, choices=(1, 2), required=True)
    parser.add_argument("--block", default=None, choices=tuple(SPECS),
                        help="omit for the legacy blk20 wiring (v12 OPQ init); "
                             "set blk05/blk10/... to retarget caches, tail "
                             "layer, and the v21 OPQ init root")
    parser.add_argument("--anchor", default="R64",
                        choices=list(C.ANCHOR_BY_NAME))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=C.DEFAULT_EPOCHS)
    parser.add_argument("--steps", type=int, default=None,
                        help="override total steps (smoke / probes)")
    parser.add_argument("--batch", type=int, default=C.DEFAULT_BATCH)
    parser.add_argument("--num-workers", type=int, default=C.DATALOADER_WORKERS)
    parser.add_argument("--lr", type=float, default=C.DEFAULT_LR)
    parser.add_argument("--l-decay", type=float, default=None,
                        help="L weight decay (phase1 default 1e-4, phase2 0)")
    parser.add_argument("--tau-start", type=float, default=C.DEFAULT_TAU_START)
    parser.add_argument("--tau-end", type=float, default=C.DEFAULT_TAU_END)
    parser.add_argument("--image-microbatch", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=C.GRAD_CLIP)
    parser.add_argument("--ckpt-every", type=int, default=C.CKPT_EVERY)
    parser.add_argument("--images", type=int, default=None,
                        help="cap train/val images (smoke only)")
    parser.add_argument("--resume", default=None,
                        help="resume from a V33 checkpoint.pt")
    parser.add_argument("--source", default=None,
                        help="phase-2: phase-1 run dir or checkpoint.pt")
    parser.add_argument("--allocation", default=None,
                        help="phase-2: searched allocation (.npy / .json)")
    parser.add_argument("--no-freeze-u0", action="store_true",
                        help="phase-2 ablation: keep U0 trainable")
    parser.add_argument("--no-L", dest="no_L", action="store_true",
                        help="drop the block-diagonal L bank; phase 2 folds "
                             "the used blocks into the tree first (exact)")
    args = parser.parse_args(argv)
    if args.block is not None:
        C.activate(args.block)
    anchor = C.ANCHOR_BY_NAME[args.anchor]
    device = torch.device(args.device)
    if args.phase == 1:
        l_decay = (C.DEFAULT_L_DECAY if args.l_decay is None
                   else float(args.l_decay))
        result = run_phase1(
            anchor, args.run_id, device, epochs=args.epochs, steps=args.steps,
            batch=args.batch, lr=args.lr, l_decay=l_decay,
            tau_start=args.tau_start, tau_end=args.tau_end,
            images=args.images, num_workers=args.num_workers,
            image_microbatch=args.image_microbatch, grad_clip=args.grad_clip,
            ckpt_every=args.ckpt_every, resume=args.resume,
            use_L=not args.no_L)
    else:
        if args.source is None and args.resume is None:
            raise SystemExit("phase 2 requires --source or --resume")
        l_decay = 0.0 if args.l_decay is None else float(args.l_decay)
        result = run_phase2(
            anchor, args.run_id, device, source=args.source,
            allocation=args.allocation, epochs=args.epochs, steps=args.steps,
            batch=args.batch, lr=args.lr, l_decay=l_decay,
            tau_start=args.tau_start, tau_end=args.tau_end,
            images=args.images, num_workers=args.num_workers,
            image_microbatch=args.image_microbatch, grad_clip=args.grad_clip,
            ckpt_every=args.ckpt_every, resume=args.resume,
            freeze_u0=not args.no_freeze_u0, drop_L=args.no_L)
    print(json.dumps({
        "phase": result["phase"], "anchor": result["anchor"],
        "seconds": result["seconds"], "checkpoint": result["checkpoint"],
        "validation": result["validation"][-3:],
    }, indent=2))


if __name__ == "__main__":
    main()
