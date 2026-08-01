"""Freeze U/codebooks and diagnose the exact-budget allocation policy."""

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from opq import batch_normalize_gpu

from . import config as C
from . import init as init_mod
from .allocation_policy import FixedBudgetAllocationPolicy
from .train import CachedFeatureDataset, actual_mode_bits, joint_validation
from .. import engine
from .. import tail as tail_mod
from . import qhard


def run(anchor, run_id, arm, device, steps=8000, schedule_steps=15600,
        batch=32, samples=12, policy_lr=1e-2, entropy_weight=0.0,
        val_every=500, val_samples=32, images=None, log=print):
    root = C.V12 / "policy_only" / C.ensure_run_id(run_id) / anchor.name / arm
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"{root} is non-empty; refusing to overwrite")
    root.mkdir(parents=True, exist_ok=True)
    engine.configure_precision(C.ALLOW_TF32)

    codec, metadata = init_mod.load_checked(anchor, device)
    codec.eval()
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    bits = actual_mode_bits(codec)
    policy = FixedBudgetAllocationPolicy(C.GROUPS, bits, anchor.rate).to(device)
    optimizer = torch.optim.Adam([policy.logits], lr=float(policy_lr))
    tail = tail_mod.build_tail(C.LAYER, device)

    train_paths = C.load_split("train_fit")
    train_set = CachedFeatureDataset(train_paths[0], train_paths[2], images)
    teacher_host = torch.from_numpy(np.load(train_paths[1])).float()
    if device.type == "cuda":
        teacher_host = teacher_host.pin_memory()
    batch = int(batch)
    teacher_staging = torch.empty(
        (batch, *teacher_host.shape[1:]),
        dtype=teacher_host.dtype,
        pin_memory=(device.type == "cuda"))
    generator = torch.Generator().manual_seed(C.TRAIN_SEED)
    loader = DataLoader(train_set, batch_size=batch, shuffle=True,
                        drop_last=True, num_workers=C.DATALOADER_WORKERS,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=True, prefetch_factor=4,
                        generator=generator)
    iterator = iter(loader)
    val_paths = C.load_split("train_val")
    val = engine.ResidentSet(*val_paths[:3], device,
                             max_images=None if images is None else min(images, C.N_VAL))
    sample_generator = torch.Generator(device=device).manual_seed(C.POLICY_SEED)
    initial = policy.build(C.DEFAULT_TEMPERATURE)
    validation = [joint_validation(
        codec, tail, val, initial, 0, int(val_samples))]
    violations = 0
    trace, started = [], time.time()

    for step in range(1, int(steps) + 1):
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
        distribution = policy.build(C.DEFAULT_TEMPERATURE, validate=False)
        allocations = distribution.sample(int(samples), generator=sample_generator)
        violations += int((policy.actual_rate(allocations) != anchor.rate).sum())
        with torch.no_grad():
            per_image, _ = qhard.distortions(
                codec, tail, y, mu, std, teacher, allocations)
            distortion = per_image.mean(1)
        scale = validation[0]["sample_mean"]
        detached = distortion.detach()
        baseline = (detached.sum() - detached) / (len(detached) - 1)
        advantage = (detached - baseline) / scale
        log_probability = distribution.log_prob(allocations, validate=False)
        entropy = distribution.entropy()
        coefficient = C.cosine_lr(
            step, schedule_steps, float(entropy_weight), floor_ratio=0.0)
        loss = (advantage * log_probability).mean()
        loss = loss - coefficient * entropy / C.GROUPS
        lr = C.cosine_lr(step - 1, schedule_steps, float(policy_lr))
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            policy.logits.sub_(policy.logits.mean(1, keepdim=True))
        trace.append({"step": step, "batch_mean": float(distortion.mean()),
                      "entropy": float(entropy), "entropy_coefficient": coefficient,
                      "policy_lr": lr})
        if step % int(val_every) == 0 or step == int(steps):
            record = joint_validation(codec, tail, val,
                                      policy.build(C.DEFAULT_TEMPERATURE),
                                      step, int(val_samples))
            validation.append(record)
            log(f"[{anchor.name}/{arm}] step={step} "
                f"MAP={record['map_distortion']:.1f} "
                f"E[D]={record['sample_mean']:.1f} H={record['entropy']:.3f} "
                f"pmax={record['marginal_max_mean']:.3f}")

    final = joint_validation(codec, tail, val,
                             policy.build(C.DEFAULT_TEMPERATURE),
                             int(steps), max(128, int(val_samples)))
    validation[-1] = final
    payload = {"plan": "policy_only_diagnostic_v1", "anchor": anchor.name,
               "run_id": run_id, "arm": arm, "init": metadata,
               "codec_frozen": True, "steps": int(steps),
               "schedule_steps": int(schedule_steps), "images": len(train_set),
               "batch": int(batch), "policy_samples": int(samples),
               "policy_lr": float(policy_lr),
               "entropy_weight": float(entropy_weight),
               "rate_violations": int(violations), "validation": validation,
               "trace": trace, "seconds": time.time() - started}
    torch.save({"policy_state": policy.state_dict(), "groups": policy.groups,
                "bit_costs": bits, "total_bits": anchor.rate,
                "temperature": C.DEFAULT_TEMPERATURE}, root / "policy.pt")
    (root / "train.json").write_text(json.dumps(payload, indent=2))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", default="R96", choices=list(C.ANCHOR_BY_NAME))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--schedule-steps", type=int, default=15600)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--policy-samples", type=int, default=12)
    parser.add_argument("--policy-lr", type=float, default=1e-2)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-samples", type=int, default=32)
    parser.add_argument("--images", type=int, default=None)
    args = parser.parse_args(argv)
    result = run(C.ANCHOR_BY_NAME[args.anchor], args.run_id, args.arm,
                 torch.device(args.device), steps=args.steps,
                 schedule_steps=args.schedule_steps, batch=args.batch,
                 samples=args.policy_samples, policy_lr=args.policy_lr,
                 entropy_weight=args.entropy_weight,
                 val_every=args.val_every, val_samples=args.val_samples,
                 images=args.images)
    print(json.dumps({"arm": result["arm"], "seconds": result["seconds"],
                      "final": result["validation"][-1]}, indent=2))


if __name__ == "__main__":
    main()
