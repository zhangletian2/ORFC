"""One continuous joint training trajectory for U, PQ modes, and allocation."""

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
from . import qhard


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


def fixed_rate_neighbors(policy, allocation):
    """All adjacent donor/receiver moves that preserve the exact bit budget."""
    base = np.asarray(torch.as_tensor(allocation).cpu(), dtype=np.int64)
    bits = tuple(int(value) for value in policy.bit_costs)
    candidates, moves = [], []
    for donor, donor_mode in enumerate(base):
        if donor_mode == 0:
            continue
        released = bits[donor_mode] - bits[donor_mode - 1]
        for receiver, receiver_mode in enumerate(base):
            if donor == receiver or receiver_mode + 1 == len(bits):
                continue
            required = bits[receiver_mode + 1] - bits[receiver_mode]
            if released != required:
                continue
            candidate = base.copy()
            candidate[donor] -= 1
            candidate[receiver] += 1
            candidates.append(candidate)
            moves.append((donor, receiver))
    matrix = (np.stack(candidates) if candidates else
              np.empty((0, len(base)), dtype=np.int64))
    return matrix, moves


@torch.no_grad()
def local_neighbor_audit(codec, tail, resident, policy, allocation):
    neighbors, moves = fixed_rate_neighbors(policy, allocation)
    if len(neighbors) == 0:
        return {"neighbor_count": 0,
                "map_distortion": validate(codec, tail, resident, allocation),
                "best_neighbor_distortion": None, "best_neighbor_gain": 0.0,
                "best_move": None, "local_tolerance": 0.0,
                "locally_optimal": True}
    base = np.asarray(torch.as_tensor(allocation).cpu(), dtype=np.int64)[None]
    values = validate_many(codec, tail, resident, np.concatenate((base, neighbors)))
    winner = int(np.argmin(values[1:]))
    gain = float(values[0] - values[winner + 1])
    tolerance = float(abs(values[0]) * C.REPLAY_REL_TOL)
    return {"neighbor_count": int(len(neighbors)),
            "map_distortion": float(values[0]),
            "best_neighbor_distortion": float(values[winner + 1]),
            "best_neighbor_gain": gain,
            "best_move": list(moves[winner]),
            "local_tolerance": tolerance,
            "locally_optimal": bool(gain <= tolerance)}


def joint_validation(codec, tail, resident, distribution, step, samples):
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


def snapshot(policy, temperature, policy_coverage):
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
            "policy_coverage_min": int(policy_coverage.min())}


def run(anchor, run_id, device, epochs=C.DEFAULT_EPOCHS, steps=None,
        batch=C.DEFAULT_BATCH, lr_u=C.DEFAULT_LR_U,
        lr_theta=C.DEFAULT_LR_THETA, policy_lr=C.DEFAULT_POLICY_LR,
        policy_samples=C.POLICY_SAMPLES,
        temperature=C.DEFAULT_TEMPERATURE,
        entropy_weight=C.DEFAULT_ENTROPY_WEIGHT, images=None, log=print,
        book_optimizer="adam", grad_clip=1.0):
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
    policy_samples = int(policy_samples)
    if policy_samples < 2:
        raise ValueError("leave-one-out policy gradient requires >=2 samples")
    train_iterator = iter(train_loader)
    generator = torch.Generator(device=device).manual_seed(C.POLICY_SEED)
    policy_coverage = torch.zeros(
        C.GROUPS, len(bits), dtype=torch.long, device=device)
    gradient_coverage = torch.zeros_like(policy_coverage)
    gradient_coverage_window = torch.zeros_like(policy_coverage)
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
        allocations = distribution.sample(policy_samples, generator=generator)
        budget_violations.add_(
            (policy.actual_rate(allocations) != anchor.rate).sum())
        add_coverage(policy_coverage, allocations)

        set_lrs(rotation_optimizer, book_optimizer, policy_optimizer,
                step - 1, schedule_total, lr_u, lr_theta, policy_lr)
        rotation_optimizer.zero_grad(set_to_none=True)
        book_optimizer.zero_grad(set_to_none=True)
        policy_optimizer.zero_grad(set_to_none=True)

        per_image, labels = qhard.distortions(
            codec, tail, y, mu, std, teacher, allocations)
        per_allocation = per_image.mean(dim=1)
        codec_loss = per_allocation.mean() / scale
        log_probability = distribution.log_prob(allocations, validate=False)
        detached = per_allocation.detach()
        baseline = (detached.sum() - detached) / (len(detached) - 1)
        advantage = (detached - baseline) / scale
        policy_loss = (advantage * log_probability).mean()
        entropy = distribution.entropy()
        entropy_coefficient = C.cosine_lr(
            step, schedule_total, float(entropy_weight), floor_ratio=0.0)
        policy_loss = policy_loss - entropy_coefficient * entropy / C.GROUPS
        (codec_loss + policy_loss).backward()
        book_gradients = [q.codebooks.grad.norm() for q in codec.pq.quantizers
                          if q.codebooks.grad is not None]
        last_gradient_norms = {
            "U": float(codec.transform.rotation.grad.norm()),
            "codebooks": float(torch.stack(book_gradients).norm())
                         if book_gradients else 0.0,
            "policy": float(policy.logits.grad.norm())}

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(codec.parameters(), float(grad_clip))
        with torch.no_grad():
            for mode, quantizer in enumerate(codec.pq.quantizers):
                gradient = quantizer.codebooks.grad
                if gradient is not None:
                    active = gradient.square().sum(dim=(1, 2)).gt(0)
                    gradient_coverage[:, mode].add_(active)
                    gradient_coverage_window[:, mode].add_(active)

        rotation_optimizer.step()
        book_optimizer.step()
        policy_optimizer.step()
        with torch.no_grad():
            policy.logits.sub_(policy.logits.mean(dim=1, keepdim=True))
        if step % C.REVIVE_EVERY == 0:
            for modes, selected in zip(allocations, labels):
                qhard.revive_dead_codewords(codec, y, modes, selected)

        trace.append((step, per_allocation.mean().detach(),
                      policy_loss.detach(), entropy.detach()))
        if step % C.VAL_EVERY == 0 or step == total:
            current = policy.build(temperature)
            record = joint_validation(
                codec, tail, val, current, step, C.STAGE1_VAL_SAMPLES)
            record["gradient_coverage_window_min"] = int(
                gradient_coverage_window.min())
            gradient_coverage_window.zero_()
            previous = next((item for item in reversed(validation)
                             if "map_allocation" in item), None)
            if previous is None:
                record["map_hamming_from_previous"] = None
                record["map_stable_run"] = 1
            else:
                hamming = sum(
                    left != right for left, right in zip(
                        record["map_allocation"], previous["map_allocation"]))
                record["map_hamming_from_previous"] = int(hamming)
                record["map_stable_run"] = (
                    int(previous.get("map_stable_run", 1)) + 1
                    if hamming == 0 else 1)
            validation.append(record)
            log(f"[{anchor.name}] validation step={step} "
                f"MAP={record['map_distortion']:.1f} "
                f"sample_gap={record['map_vs_sample_mean']:.1f} "
                f"H={record['entropy']:.3f} "
                f"pmax={record['marginal_max_mean']:.3f} "
                f"dMAP={record['map_hamming_from_previous']} "
                f"stable={record['map_stable_run']}")
        if step == 1 or step % C.LOG_EVERY == 0:
            log(f"[{anchor.name}] {step}/{total} D={float(per_allocation.mean()):.1f} "
                f"H={float(entropy):.3f} "
                f"coverage={int(policy_coverage.min())}.."
                f"{int(policy_coverage.max())}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.time() - training_started
    peak_memory_gb = (torch.cuda.max_memory_allocated(device) / 2 ** 30
                      if device.type == "cuda" else 0.0)
    summary = snapshot(policy, temperature, policy_coverage)
    if int(budget_violations) != 0:
        raise SystemExit("INVALID_EXPERIMENT: policy sample violated budget")
    final_map = torch.tensor(summary["map_allocation"], device=device)
    if summary["map_rate"] != anchor.rate:
        raise SystemExit("INVALID_EXPERIMENT: final rate failed")
    final_parity = qhard.selfcheck(codec, tail, val, final_map)
    final_orth = frozen.orthogonality_error(codec)
    if final_orth - initial_orth > C.ORTH_TOL:
        raise SystemExit("INVALID_EXPERIMENT: orthogonality drift")

    final_distribution = policy.build(temperature)
    final_record = joint_validation(
        codec, tail, val, final_distribution, total,
        C.STAGE1_FINAL_SAMPLES)
    for key in ("map_hamming_from_previous", "map_stable_run",
                "gradient_coverage_window_min"):
        if key in validation[-1]:
            final_record[key] = validation[-1][key]
    validation[-1] = final_record
    neighbor_audit = local_neighbor_audit(codec, tail, val, policy, final_map)
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
    final_lrs = {"U": rotation_optimizer.param_groups[0]["lr"],
                 "codebooks": book_optimizer.param_groups[0]["lr"],
                 "policy": policy_optimizer.param_groups[0]["lr"]}
    joint = {
        "contract": "one continuous exact-budget trajectory jointly updates "
                    "U, every sampled PQ mode, and the allocation policy",
        "single_continuous_trajectory": True,
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
        "final_learning_rates": final_lrs,
        "lr_floor_ratio": C.LR_FLOOR_RATIO,
        "final_gradient_norms": last_gradient_norms,
        "last_window_gradient_coverage_min": final_record[
            "gradient_coverage_window_min"],
        "all_variables_active_at_end": bool(
            all(value > 0 and math.isfinite(value)
                for value in (*final_lrs.values(),
                              *last_gradient_norms.values()))
            and final_record["gradient_coverage_window_min"] > 0),
        "local_neighbor_audit": neighbor_audit,
    }
    joint["protocol_valid"] = bool(
        joint["gradient_coverage_min"] > 0
        and joint["codebook_relative_drift_min"] > 0
        and joint["rotation_relative_drift"] > 0
        and joint["all_variables_active_at_end"])
    joint["discrete_local_exit_reached"] = bool(
        joint["protocol_valid"]
        and joint["entropy_decreased"]
        and joint["map_is_nonuniform"]
        and joint["map_stable_last_n"]
        and joint["map_gain_vs_initial_uniform"] > 0
        and joint["map_better_than_sample_mean"]
        and neighbor_audit["locally_optimal"])

    save_codec_v1(codec, out / "codec.pt")
    torch.save({"policy_state": policy.state_dict(), "groups": policy.groups,
                "bit_costs": bits, "total_bits": anchor.rate,
                "temperature": temperature},
               out / "policy.pt")
    np.save(out / "allocation.npy", final_map.cpu().numpy())
    (out / "policy_summary.json").write_text(json.dumps(summary, indent=2))
    trace = [{"step": item[0], "policy_mean": float(item[1]),
              "policy_loss": float(item[2]), "entropy": float(item[3])}
             for item in trace]
    payload = {"plan": "continuous_joint_v1", "anchor": anchor.name,
               "rate": anchor.rate,
               "run_id": run_id, "init": metadata, "images": len(train_set),
               "val_images": val.count, "batch": batch, "epochs": float(total / per_epoch),
               "steps": total, "steps_per_epoch": per_epoch,
               "schedule_steps": schedule_total,
               "schedule_epochs": int(epochs),
               "lr_U": lr_u, "lr_theta": lr_theta, "policy_lr": policy_lr,
               "policy_samples": policy_samples,
               "book_optimizer": book_optimizer_name,
               "grad_clip": grad_clip,
               "temperature": temperature, "entropy_weight": entropy_weight,
               "initial_distortion": initial_distortion,
               "hard_parity_initial": initial_parity,
               "hard_parity_final": final_parity,
               "orthogonality_initial": initial_orth,
               "orthogonality_final": final_orth,
               "validation": validation, "policy": summary,
               "joint_training": joint,
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
    parser.add_argument("--policy-lr", type=float, default=C.DEFAULT_POLICY_LR)
    parser.add_argument("--policy-samples", type=int, default=C.POLICY_SAMPLES)
    parser.add_argument("--temperature", type=float, default=C.DEFAULT_TEMPERATURE)
    parser.add_argument("--entropy-weight", type=float,
                        default=C.DEFAULT_ENTROPY_WEIGHT)
    parser.add_argument("--images", type=int, default=None, help="probe only")
    args = parser.parse_args(argv)
    result = run(C.ANCHOR_BY_NAME[args.anchor], args.run_id,
                 torch.device(args.device), epochs=args.epochs, steps=args.steps,
                 batch=args.batch, lr_u=args.lr_u, lr_theta=args.lr_theta,
                 policy_lr=args.policy_lr, policy_samples=args.policy_samples,
                 temperature=args.temperature,
                 entropy_weight=args.entropy_weight, images=args.images,
                 book_optimizer=args.book_optimizer, grad_clip=args.grad_clip)
    print(json.dumps({"anchor": result["anchor"], "seconds": result["seconds"],
                      "validation": result["validation"],
                      "policy": result["policy"],
                      "joint_training": result["joint_training"]}, indent=2))


if __name__ == "__main__":
    main()
