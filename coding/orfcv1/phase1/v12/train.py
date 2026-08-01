"""Joint hard-tail training of U, all PQ modes, and an exact-budget policy."""

import argparse
import json
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cayley import CayleySGD
from codec_v1 import save_codec_v1
from opq import batch_normalize_gpu

from . import config as C
from . import init as init_mod
from .allocation_policy import FixedBudgetAllocationPolicy
from .. import engine, frozen
from .. import tail as tail_mod
from ..v11 import qhard
from ..v11.train import MenuStream


class CachedFeatureDataset(Dataset):
    """Memory-mapped blk features; teachers stay in a separate CPU cache."""

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


def actual_mode_bits(codec):
    bits = tuple(int(round(math.log2(q.codebooks.shape[1])))
                 for q in codec.pq.quantizers)
    if any(2 ** bit != q.codebooks.shape[1]
           for bit, q in zip(bits, codec.pq.quantizers)):
        raise SystemExit("INVALID_EXPERIMENT: non-power-of-two codebook")
    return bits


def make_trainable(codec):
    codec.transform.rotation.requires_grad_(True)
    for quantizer in codec.pq.quantizers:
        quantizer.codebooks.requires_grad_(True)


def build_optimizers(codec, lr_u, lr_theta, book_optimizer="sgd"):
    rotation = CayleySGD(
        [codec.transform.rotation], lr=float(lr_u),
        fixed_point_iterations=C.CAYLEY_FIXED_POINT_ITERATIONS,
        reorthogonalize_every=C.CAYLEY_REORTH_EVERY)
    parameters = [q.codebooks for q in codec.pq.quantizers]
    if book_optimizer == "adam":
        books = torch.optim.Adam(parameters, lr=float(lr_theta))
    else:
        books = torch.optim.SGD(parameters, lr=float(lr_theta),
                                momentum=C.CODEBOOK_MOMENTUM)
    return rotation, books


def set_lrs(rotation, books, policy, step, total, lr_u, lr_theta, lr_policy):
    for optimizer, base in ((rotation, lr_u), (books, lr_theta),
                            (policy, lr_policy)):
        value = C.cosine_lr(step, total, base)
        for group in optimizer.param_groups:
            group["lr"] = value


def add_coverage(table, allocations):
    rows = torch.as_tensor(allocations, device=table.device, dtype=torch.long)
    if rows.ndim == 1:
        rows = rows[None]
    index = rows.t().contiguous()
    table.scatter_add_(1, index, torch.ones_like(index, dtype=table.dtype))


def hard_distortion(codec, tail, y, mu, std, teacher, modes):
    value, labels = qhard.distortion(codec, tail, y, mu, std, teacher, modes)
    return value.mean(), labels


@torch.no_grad()
def validate_many(codec, tail, resident, allocations):
    allocations = torch.as_tensor(allocations)
    if allocations.ndim == 1:
        allocations = allocations[None]
    matrix = engine.evaluate_allocations(
        codec, tail, resident, allocations.detach().cpu().numpy(),
        image_batch=C.EVAL_IMAGE_BATCH, pair_budget=C.EVAL_PAIR_BUDGET,
        per_image=True)
    return matrix.mean(axis=1)


def validate(codec, tail, resident, allocation):
    return float(validate_many(codec, tail, resident, allocation)[0])


def stage1_validation(codec, tail, resident, distribution, step, samples):
    allocation = distribution.map_allocation()
    generator = torch.Generator(device=allocation.device).manual_seed(
        C.STAGE1_VAL_SEED + int(step))
    draws = distribution.sample(int(samples), generator=generator)
    values = validate_many(codec, tail, resident,
                           torch.cat((allocation[None], draws), dim=0))
    marginals = distribution.marginals().detach()
    return {"step": int(step), "map_distortion": float(values[0]),
            "sample_mean": float(values[1:].mean()),
            "sample_min": float(values[1:].min()),
            "sample_max": float(values[1:].max()),
            "map_vs_sample_mean": float(values[0] - values[1:].mean()),
            "map_allocation": allocation.cpu().tolist(),
            "entropy": float(distribution.entropy().detach()),
            "marginal_min": float(marginals.min()),
            "marginal_max_mean": float(marginals.max(1).values.mean()),
            "logit_rms": float(
                distribution.scores.square().mean().sqrt().detach())}


def snapshot(policy, temperature, policy_coverage, support_coverage):
    distribution = policy.build(temperature)
    marginals = distribution.marginals().detach()
    allocation = distribution.map_allocation()
    bits = torch.tensor(policy.bit_costs, device=marginals.device,
                        dtype=marginals.dtype)
    return {"map_allocation": allocation.cpu().tolist(),
            "map_rate": int(policy.actual_rate(allocation)),
            "entropy": float(distribution.entropy().detach()),
            "marginals": marginals.cpu().tolist(),
            "marginal_min": float(marginals.min()),
            "expected_rate": float((marginals * bits).sum()),
            "policy_coverage": policy_coverage.cpu().tolist(),
            "policy_coverage_min": int(policy_coverage.min()),
            "support_coverage": support_coverage.cpu().tolist(),
            "support_coverage_min": int(support_coverage.min())}


def run(anchor, run_id, device, epochs=C.DEFAULT_EPOCHS, steps=None,
        batch=C.DEFAULT_BATCH, lr_u=C.DEFAULT_LR_U,
        lr_theta=C.DEFAULT_LR_THETA, policy_lr=C.DEFAULT_POLICY_LR,
        temperature=C.DEFAULT_TEMPERATURE,
        entropy_weight=C.DEFAULT_ENTROPY_WEIGHT, images=None, log=print,
        book_optimizer="adam", grad_clip=1.0,
        coverage_weight=C.COVERAGE_WEIGHT):
    started = time.time()
    run_id = C.ensure_run_id(run_id)
    out = C.output_dir(anchor, run_id)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is non-empty; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    engine.configure_precision(C.ALLOW_TF32)

    codec, metadata = init_mod.load_checked(anchor, device)
    make_trainable(codec)
    initial_rotation = codec.transform.rotation.detach().clone()
    initial_books = [q.codebooks.detach().clone() for q in codec.pq.quantizers]
    book_optimizer_name = book_optimizer
    bits = actual_mode_bits(codec)
    if bits != tuple(anchor.mode_bits):
        raise SystemExit(f"INVALID_EXPERIMENT: menu {bits} != {anchor.mode_bits}")
    policy = FixedBudgetAllocationPolicy(C.GROUPS, bits, anchor.rate).to(device)
    policy_optimizer = torch.optim.Adam([policy.logits], lr=float(policy_lr))
    rotation_optimizer, book_optimizer = build_optimizers(
        codec, lr_u, lr_theta, book_optimizer_name)

    tail = tail_mod.build_tail(C.LAYER, device)
    train_paths = C.load_split("train_fit")
    tail_mod.check_layer(C.LAYER, train_paths[0], train_paths[1])
    train_set = CachedFeatureDataset(
        train_paths[0], train_paths[2], max_images=images)
    teacher_cache = np.load(train_paths[1], mmap_mode="r")
    val_paths = C.load_split("train_val")
    val = engine.ResidentSet(*val_paths[:3], device,
                             max_images=None if images is None else min(images, C.N_VAL))
    batch = int(batch)
    loader_generator = torch.Generator().manual_seed(C.TRAIN_SEED)
    train_loader = DataLoader(
        train_set, batch_size=batch, shuffle=True, drop_last=True,
        num_workers=2, pin_memory=True, persistent_workers=True,
        prefetch_factor=4, generator=loader_generator)
    per_epoch = len(train_loader)
    if per_epoch == 0:
        raise ValueError(f"batch {batch} exceeds {len(train_set)} training images")
    schedule_total = int(epochs) * per_epoch
    total = int(steps) if steps is not None else schedule_total
    if total < 1:
        raise ValueError("training requires at least one step")
    train_iterator = iter(train_loader)
    generator = torch.Generator(device=device).manual_seed(C.POLICY_SEED)
    support_stream = MenuStream(anchor, seed=C.COVERAGE_SEED)
    policy_coverage = torch.zeros(
        C.GROUPS, len(bits), dtype=torch.long, device=device)
    support_coverage = torch.zeros_like(policy_coverage)
    gradient_coverage = torch.zeros_like(policy_coverage)
    budget_violations = torch.zeros((), dtype=torch.long, device=device)

    uniform = torch.as_tensor(engine.uniform_allocation(anchor), device=device)
    initial_parity = qhard.selfcheck(codec, tail, val, uniform)
    initial_distortion = validate(codec, tail, val, uniform)
    initial_entropy = float(policy.build(temperature).entropy().detach())
    scale = initial_distortion
    initial_orth = frozen.orthogonality_error(codec)
    trace, validation = [], [{"step": 0, "uniform": initial_distortion}]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    training_started = time.time()

    for step in range(1, total + 1):
        try:
            rows, x = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            rows, x = next(train_iterator)
        teacher = torch.from_numpy(
            np.asarray(teacher_cache[rows.numpy()], dtype=np.float32))
        x = x.float().to(device, non_blocking=True)
        teacher = teacher.float().to(device, non_blocking=True)
        with torch.no_grad():
            y, mu, std = batch_normalize_gpu(x, mode=C.NORM_MODE)
        distribution = policy.build(temperature, validate=False)
        allocations = distribution.sample(C.POLICY_SAMPLES, generator=generator)
        budget_violations.add_(
            (policy.actual_rate(allocations) != anchor.rate).sum())
        _, support_np = support_stream.allocation(step)
        support = torch.as_tensor(support_np, dtype=torch.long, device=device)
        if sum(bits[int(mode)] for mode in support_np) != anchor.rate:
            raise SystemExit("INVALID_EXPERIMENT: support allocation violated budget")
        add_coverage(policy_coverage, allocations)
        add_coverage(support_coverage, support)

        set_lrs(rotation_optimizer, book_optimizer, policy_optimizer,
                step - 1, schedule_total, lr_u, lr_theta, policy_lr)
        rotation_optimizer.zero_grad(set_to_none=True)
        book_optimizer.zero_grad(set_to_none=True)
        policy_optimizer.zero_grad(set_to_none=True)

        means, labels = [], []
        for modes in allocations:
            mean, selected = hard_distortion(codec, tail, y, mu, std, teacher, modes)
            (0.5 * mean / scale).backward()
            means.append(mean.detach())
            labels.append(selected)

        # A legal support allocation is used on every step.  It updates only
        # codebooks, so all modes track the changing rotated feature space
        # without steering U or the allocation policy.
        codec.transform.rotation.requires_grad_(False)
        support_mean, support_labels = hard_distortion(
            codec, tail, y, mu, std, teacher, support)
        (coverage_weight * support_mean / scale).backward()
        codec.transform.rotation.requires_grad_(True)

        per_allocation = torch.stack(means)
        log_probability = distribution.log_prob(allocations, validate=False)
        advantage = (per_allocation[0] - per_allocation[1]) / scale
        policy_loss = 0.5 * advantage * (log_probability[0] - log_probability[1])
        entropy = distribution.entropy()
        entropy_coefficient = C.cosine_lr(
            step, schedule_total, float(entropy_weight))
        policy_loss = policy_loss - entropy_coefficient * entropy / C.GROUPS
        policy_loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(codec.parameters(), float(grad_clip))
        with torch.no_grad():
            for mode, quantizer in enumerate(codec.pq.quantizers):
                gradient = quantizer.codebooks.grad
                if gradient is not None:
                    gradient_coverage[:, mode].add_(
                        gradient.square().sum(dim=(1, 2)).gt(0))

        rotation_optimizer.step()
        book_optimizer.step()
        policy_optimizer.step()
        with torch.no_grad():
            policy.logits.sub_(policy.logits.mean(dim=1, keepdim=True))
        if step % C.REVIVE_EVERY == 0:
            for modes, selected in zip(allocations, labels):
                qhard.revive_dead_codewords(codec, y, modes, selected)
            qhard.revive_dead_codewords(codec, y, support, support_labels)

        trace.append((step, per_allocation.mean().detach(), support_mean.detach(),
                      policy_loss.detach(), entropy.detach()))
        if step % C.VAL_EVERY == 0 or step == total:
            current = policy.build(temperature)
            record = stage1_validation(
                codec, tail, val, current, step, C.STAGE1_VAL_SAMPLES)
            validation.append(record)
            log(f"[{anchor.name}] validation step={step} "
                f"MAP={record['map_distortion']:.1f} H={record['entropy']:.3f}")
        if step == 1 or step % C.LOG_EVERY == 0:
            log(f"[{anchor.name}] {step}/{total} D={float(per_allocation.mean()):.1f} "
                f"support={float(support_mean):.1f} H={float(entropy):.3f} "
                f"coverage={int(support_coverage.min())}.."
                f"{int(support_coverage.max())}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.time() - training_started
    peak_memory_gb = (torch.cuda.max_memory_allocated(device) / 2 ** 30
                      if device.type == "cuda" else 0.0)
    summary = snapshot(policy, temperature, policy_coverage, support_coverage)
    if int(budget_violations) != 0:
        raise SystemExit("INVALID_EXPERIMENT: policy sample violated budget")
    final_map = torch.tensor(summary["map_allocation"], device=device)
    if summary["map_rate"] != anchor.rate or summary["support_coverage_min"] == 0:
        raise SystemExit("INVALID_EXPERIMENT: final rate or support coverage failed")
    final_parity = qhard.selfcheck(codec, tail, val, final_map)
    final_orth = frozen.orthogonality_error(codec)
    if final_orth - initial_orth > C.ORTH_TOL:
        raise SystemExit("INVALID_EXPERIMENT: orthogonality drift")

    final_distribution = policy.build(temperature)
    final_record = stage1_validation(
        codec, tail, val, final_distribution, total,
        C.STAGE1_FINAL_SAMPLES)
    validation[-1] = final_record
    book_drift = []
    for initial, quantizer in zip(initial_books, codec.pq.quantizers):
        delta = (quantizer.codebooks.detach() - initial).flatten(1).norm(dim=1)
        base = initial.flatten(1).norm(dim=1).clamp_min(
            torch.finfo(initial.dtype).eps)
        book_drift.append(delta / base)
    book_drift = torch.stack(book_drift, dim=1)
    rotation_drift = float(
        (codec.transform.rotation.detach() - initial_rotation).norm()
        / initial_rotation.norm().clamp_min(
            torch.finfo(initial_rotation.dtype).eps))
    recent = validation[-C.STAGE1_STABILITY_POINTS:]
    stable = (len(recent) == C.STAGE1_STABILITY_POINTS and
              all(item["map_allocation"] == recent[-1]["map_allocation"]
                  for item in recent))
    uniform_modes = torch.full_like(final_map, anchor.uniform_mode)
    stage1 = {
        "contract": "explore exact-budget allocations; prepare every mode; "
                    "finish at a stable, lower-distortion hard MAP allocation",
        "gradient_coverage": gradient_coverage.cpu().tolist(),
        "gradient_coverage_min": int(gradient_coverage.min()),
        "codebook_relative_drift": book_drift.cpu().tolist(),
        "codebook_relative_drift_min": float(book_drift.min()),
        "rotation_relative_drift": rotation_drift,
        "policy_entropy_initial": initial_entropy,
        "policy_entropy_final": final_record["entropy"],
        "entropy_decreased": bool(final_record["entropy"] < initial_entropy),
        "map_is_nonuniform": bool(not torch.equal(final_map, uniform_modes)),
        "map_stable_last_n": bool(stable),
        "stability_points": C.STAGE1_STABILITY_POINTS,
        "map_gain_vs_initial_uniform": float(
            initial_distortion - final_record["map_distortion"]),
        "map_better_than_sample_mean": bool(
            final_record["map_distortion"] <= final_record["sample_mean"]),
    }
    stage1["passed"] = bool(
        stage1["gradient_coverage_min"] > 0
        and stage1["codebook_relative_drift_min"] > 0
        and stage1["rotation_relative_drift"] > 0
        and stage1["entropy_decreased"]
        and stage1["map_is_nonuniform"]
        and stage1["map_stable_last_n"]
        and stage1["map_gain_vs_initial_uniform"] > 0
        and stage1["map_better_than_sample_mean"])

    save_codec_v1(codec, out / "codec.pt")
    torch.save({"policy_state": policy.state_dict(), "groups": policy.groups,
                "bit_costs": bits, "total_bits": anchor.rate,
                "temperature": temperature},
               out / "policy.pt")
    np.save(out / "allocation.npy", final_map.cpu().numpy())
    (out / "policy_summary.json").write_text(json.dumps(summary, indent=2))
    trace = [{"step": item[0], "policy_mean": float(item[1]),
              "support_mean": float(item[2]), "policy_loss": float(item[3]),
              "entropy": float(item[4])} for item in trace]
    payload = {"plan": "v13.1", "anchor": anchor.name, "rate": anchor.rate,
               "run_id": run_id, "init": metadata, "images": len(train_set),
               "val_images": val.count, "batch": batch, "epochs": float(total / per_epoch),
               "steps": total, "steps_per_epoch": per_epoch,
               "schedule_steps": schedule_total,
               "schedule_epochs": int(epochs),
               "lr_U": lr_u, "lr_theta": lr_theta, "policy_lr": policy_lr,
               "book_optimizer": book_optimizer_name,
               "grad_clip": grad_clip,
               "temperature": temperature, "entropy_weight": entropy_weight,
               "coverage_weight": coverage_weight,
               "initial_distortion": initial_distortion,
               "hard_parity_initial": initial_parity,
               "hard_parity_final": final_parity,
               "orthogonality_initial": initial_orth,
               "orthogonality_final": final_orth,
               "validation": validation, "policy": summary,
               "stage1": stage1,
               "trace": trace, "training_seconds": training_seconds,
               "peak_memory_gb": peak_memory_gb,
               "seconds": time.time() - started}
    (out / "train.json").write_text(json.dumps(payload, indent=2))
    log(f"[{anchor.name}] complete in {payload['seconds']:.1f}s")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", required=True, choices=list(C.ANCHOR_BY_NAME))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=C.DEFAULT_EPOCHS)
    parser.add_argument("--steps", type=int, default=None, help="probe override")
    parser.add_argument("--batch", type=int, default=C.DEFAULT_BATCH)
    parser.add_argument("--lr-u", type=float, default=C.DEFAULT_LR_U)
    parser.add_argument("--lr-theta", type=float, default=C.DEFAULT_LR_THETA)
    parser.add_argument("--book-optimizer", choices=("sgd", "adam"),
                        default="adam")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--coverage-weight", type=float,
                        default=C.COVERAGE_WEIGHT)
    parser.add_argument("--policy-lr", type=float, default=C.DEFAULT_POLICY_LR)
    parser.add_argument("--temperature", type=float, default=C.DEFAULT_TEMPERATURE)
    parser.add_argument("--entropy-weight", type=float,
                        default=C.DEFAULT_ENTROPY_WEIGHT)
    parser.add_argument("--images", type=int, default=None, help="probe only")
    args = parser.parse_args(argv)
    result = run(C.ANCHOR_BY_NAME[args.anchor], args.run_id,
                 torch.device(args.device), epochs=args.epochs, steps=args.steps,
                 batch=args.batch, lr_u=args.lr_u, lr_theta=args.lr_theta,
                 policy_lr=args.policy_lr, temperature=args.temperature,
                 entropy_weight=args.entropy_weight, images=args.images,
                 book_optimizer=args.book_optimizer, grad_clip=args.grad_clip,
                 coverage_weight=args.coverage_weight)
    print(json.dumps({"anchor": result["anchor"], "seconds": result["seconds"],
                      "validation": result["validation"],
                      "policy": result["policy"]}, indent=2))


if __name__ == "__main__":
    main()
