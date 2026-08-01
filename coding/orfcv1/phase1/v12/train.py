"""Joint hard-tail training of U, all PQ modes, and an exact-budget policy."""

import argparse
import json
import math
import time

import numpy as np
import torch

from cayley import CayleySGD
from codec_v1 import save_codec_v1

from . import config as C
from . import init as init_mod
from .allocation_policy import FixedBudgetAllocationPolicy
from .. import engine, frozen
from .. import tail as tail_mod
from ..v11 import qhard
from ..v11.train import BatchStream, MenuStream, gather


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
    rows = torch.as_tensor(allocations).detach().cpu().numpy()
    if rows.ndim == 1:
        rows = rows[None]
    for row in rows:
        table[np.arange(row.size), row] += 1


def hard_distortion(codec, tail, y, mu, std, teacher, modes):
    value, labels = qhard.distortion(codec, tail, y, mu, std, teacher, modes)
    return value.mean(), labels


@torch.no_grad()
def validate(codec, tail, resident, allocation):
    matrix = engine.evaluate_allocations(
        codec, tail, resident, allocation.detach().cpu().numpy()[None],
        image_batch=C.EVAL_IMAGE_BATCH, pair_budget=C.EVAL_PAIR_BUDGET,
        per_image=True)
    return float(matrix[0].mean())


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
            "policy_coverage": policy_coverage.tolist(),
            "policy_coverage_min": int(policy_coverage.min()),
            "support_coverage": support_coverage.tolist(),
            "support_coverage_min": int(support_coverage.min())}


def run(anchor, run_id, device, epochs=C.DEFAULT_EPOCHS, steps=None,
        batch=C.DEFAULT_BATCH, lr_u=C.DEFAULT_LR_U,
        lr_theta=C.DEFAULT_LR_THETA, policy_lr=C.DEFAULT_POLICY_LR,
        temperature=C.DEFAULT_TEMPERATURE,
        entropy_weight=C.DEFAULT_ENTROPY_WEIGHT, images=None, log=print,
        book_optimizer="sgd", grad_clip=0.0,
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
    resident = engine.ResidentSet(*train_paths[:3], device, max_images=images)
    val_paths = C.load_split("train_val")
    val = engine.ResidentSet(*val_paths[:3], device,
                             max_images=None if images is None else min(images, C.N_VAL))
    batch = int(batch)
    per_epoch = max(1, resident.count // batch)
    schedule_total = int(epochs) * per_epoch
    total = int(steps) if steps is not None else schedule_total
    if total < 1:
        raise ValueError("training requires at least one step")
    stream = BatchStream(resident.count, batch, C.TRAIN_SEED)
    generator = torch.Generator(device=device).manual_seed(C.POLICY_SEED)
    support_stream = MenuStream(anchor, seed=C.COVERAGE_SEED)
    policy_coverage = np.zeros((C.GROUPS, len(bits)), dtype=np.int64)
    support_coverage = np.zeros_like(policy_coverage)

    uniform = torch.as_tensor(engine.uniform_allocation(anchor), device=device)
    initial_parity = qhard.selfcheck(codec, tail, val, uniform)
    initial_distortion = validate(codec, tail, val, uniform)
    scale = initial_distortion
    initial_orth = frozen.orthogonality_error(codec)
    trace, validation = [], [{"step": 0, "uniform": initial_distortion}]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    training_started = time.time()

    for step in range(1, total + 1):
        index = stream.next().to(device)
        y, mu, std, teacher = gather(resident, index)
        distribution = policy.build(temperature)
        allocations = distribution.sample(C.POLICY_SAMPLES, generator=generator)
        if not bool((policy.actual_rate(allocations) == anchor.rate).all()):
            raise SystemExit("INVALID_EXPERIMENT: policy sample violated budget")
        _, support_np = support_stream.allocation(step)
        support = torch.as_tensor(support_np, dtype=torch.long, device=device)
        if int(policy.actual_rate(support)) != anchor.rate:
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
        log_probability = distribution.log_prob(allocations)
        advantage = (per_allocation[0] - per_allocation[1]) / scale
        policy_loss = 0.5 * advantage * (log_probability[0] - log_probability[1])
        entropy = distribution.entropy()
        entropy_coefficient = C.cosine_lr(
            step, schedule_total, float(entropy_weight))
        policy_loss = policy_loss - entropy_coefficient * entropy / C.GROUPS
        policy_loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(codec.parameters(), float(grad_clip))

        rotation_optimizer.step()
        book_optimizer.step()
        policy_optimizer.step()
        with torch.no_grad():
            policy.logits.sub_(policy.logits.mean(dim=1, keepdim=True))
        if step % C.REVIVE_EVERY == 0:
            for modes, selected in zip(allocations, labels):
                qhard.revive_dead_codewords(codec, y, modes, selected)
            qhard.revive_dead_codewords(codec, y, support, support_labels)

        trace.append({"step": step, "policy_mean": float(per_allocation.mean()),
                      "support_mean": float(support_mean.detach()),
                      "policy_loss": float(policy_loss.detach()),
                      "entropy": float(entropy.detach())})
        if step % C.VAL_EVERY == 0 or step == total:
            current = policy.build(temperature)
            allocation = current.map_allocation()
            record = {"step": step,
                      "map_distortion": validate(codec, tail, val, allocation),
                      "entropy": float(current.entropy().detach()),
                      "marginal_min": float(current.marginals().min().detach())}
            validation.append(record)
            log(f"[{anchor.name}] validation step={step} "
                f"MAP={record['map_distortion']:.1f} H={record['entropy']:.3f}")
        if step == 1 or step % C.LOG_EVERY == 0:
            log(f"[{anchor.name}] {step}/{total} D={float(per_allocation.mean()):.1f} "
                f"support={float(support_mean):.1f} H={float(entropy):.3f} "
                f"coverage={support_coverage.min()}..{support_coverage.max()}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.time() - training_started
    peak_memory_gb = (torch.cuda.max_memory_allocated(device) / 2 ** 30
                      if device.type == "cuda" else 0.0)
    summary = snapshot(policy, temperature, policy_coverage, support_coverage)
    final_map = torch.tensor(summary["map_allocation"], device=device)
    if summary["map_rate"] != anchor.rate or summary["support_coverage_min"] == 0:
        raise SystemExit("INVALID_EXPERIMENT: final rate or support coverage failed")
    final_parity = qhard.selfcheck(codec, tail, val, final_map)
    final_orth = frozen.orthogonality_error(codec)
    if final_orth - initial_orth > C.ORTH_TOL:
        raise SystemExit("INVALID_EXPERIMENT: orthogonality drift")

    save_codec_v1(codec, out / "codec.pt")
    torch.save({"policy_state": policy.state_dict(), "groups": policy.groups,
                "bit_costs": bits, "total_bits": anchor.rate,
                "temperature": temperature},
               out / "policy.pt")
    np.save(out / "allocation.npy", final_map.cpu().numpy())
    (out / "policy_summary.json").write_text(json.dumps(summary, indent=2))
    payload = {"plan": "v13", "anchor": anchor.name, "rate": anchor.rate,
               "run_id": run_id, "init": metadata, "images": resident.count,
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
                        default="sgd")
    parser.add_argument("--grad-clip", type=float, default=0.0)
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
