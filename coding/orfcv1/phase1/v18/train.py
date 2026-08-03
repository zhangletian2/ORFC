"""Run an unchanged v12-v17 trainer with read-only U-gradient probes.

The probe is installed by wrapping ``joint_validation``.  It neither changes
the training loss nor writes gradients into codec parameters.  At fixed
training steps it reuses the same validation images and the same exact-budget
allocations, and measures allocation-specific gradients with the hard forward
and the codeword surrogate backward used at that point in training.
"""

import argparse
import json

import numpy as np
import torch

from codec_v1 import save_codec_v1

from .. import engine
from ..v12 import config as C
from ..v12 import qhard
from ..v16.grad_conflict import tangent_gradient


def _value(argv, flag, default):
    return argv[argv.index(flag) + 1] if flag in argv else default


def fixed_exchange_set(anchor, count, seed, device):
    """Uniform allocation plus deterministic one-bit exchanges."""
    base = np.asarray(engine.uniform_allocation(anchor), dtype=np.int64)
    if base.min() <= 0 or base.max() >= 2:
        raise ValueError("trajectory probe requires an interior uniform mode")
    rng = np.random.default_rng(seed)
    groups = rng.permutation(len(base))
    rows, moves = [base.copy()], [None]
    for index in range(1, int(count)):
        donor = int(groups[(2 * index - 2) % len(groups)])
        receiver = int(groups[(2 * index - 1) % len(groups)])
        row = base.copy()
        row[donor] -= 1
        row[receiver] += 1
        rows.append(row)
        moves.append([donor, receiver])
    return torch.as_tensor(np.stack(rows), device=device), moves


def checkpoint_steps(total, fractions, explicit):
    if explicit:
        values = {int(item) for item in explicit.split(",")}
    else:
        values = {round(float(item) * total / C.VAL_EVERY) * C.VAL_EVERY
                  for item in fractions.split(",")}
    values = {min(max(value, 0), total) for value in values}
    return sorted(values | {0, total})


def gradient_snapshot(codec, tail, resident, allocations, batches, batch,
                      tau, step):
    """Measure fixed-allocation tangent gradients without touching ``.grad``."""
    all_values, per_pair = [], []
    losses, batch_records = [], []
    for batch_index in range(int(batches)):
        first, last = batch_index * int(batch), (batch_index + 1) * int(batch)
        tensors = resident.slice(first, last)
        gradients, batch_losses = [], []
        for allocation in allocations:
            rotation = codec.transform.get_rotation().detach().requires_grad_(True)
            value, _ = qhard.distortion_sparse(
                codec, tail, *tensors, allocation, rotation=rotation,
                codeword_temperature=float(tau))
            loss = value.mean()
            gradient, = torch.autograd.grad(loss, rotation)
            projected = tangent_gradient(rotation, gradient).flatten()
            gradients.append(projected / projected.norm().clamp_min(1e-30))
            batch_losses.append(float(loss.detach()))
        matrix = torch.stack(gradients) @ torch.stack(gradients).t()
        indices = torch.triu_indices(
            len(gradients), len(gradients), offset=1, device=matrix.device)
        values = matrix[indices[0], indices[1]].detach().cpu().numpy()
        all_values.extend(values.tolist())
        per_pair.append(values)
        losses.append(batch_losses)
        batch_records.append({
            "batch": batch_index,
            "negative_fraction": float((values < 0).mean()),
            "strong_negative_fraction": float((values < -0.1).mean()),
            "cosine_median": float(np.median(values)),
            "cosine_min": float(values.min())})
    values = np.asarray(all_values)
    pair_by_batch = np.stack(per_pair)
    pair_mean = pair_by_batch.mean(axis=0)
    return {
        "step": int(step), "codeword_temperature": float(tau),
        "pair_count": int(values.size),
        "negative_fraction": float((values < 0).mean()),
        "strong_negative_fraction": float((values < -0.1).mean()),
        "cosine_mean": float(values.mean()),
        "cosine_median": float(np.median(values)),
        "cosine_q10": float(np.quantile(values, 0.1)),
        "cosine_min": float(values.min()),
        "pair_mean_negative_fraction": float((pair_mean < 0).mean()),
        "pair_negative_all_batches_fraction": float(
            (pair_by_batch < 0).all(axis=0).mean()),
        "pair_negative_majority_batches_fraction": float(
            ((pair_by_batch < 0).mean(axis=0) > 0.5).mean()),
        "pair_mean_cosine_min": float(pair_mean.min()),
        "allocation_loss_by_batch": losses,
        "batch_records": batch_records}


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gradient-fractions", default="0,0.1,0.5,1.0")
    parser.add_argument("--gradient-steps", default="")
    parser.add_argument("--gradient-batches", type=int, default=8)
    parser.add_argument("--gradient-batch", type=int, default=8)
    parser.add_argument("--gradient-allocations", type=int, default=8)
    parser.add_argument("--gradient-seed", type=int, default=20260803)
    parser.add_argument("--snapshot-every", type=int, default=1000)
    diag, train_argv = parser.parse_known_args(argv)
    anchor = C.ANCHOR_BY_NAME[_value(train_argv, "--anchor", None)]
    run_id = _value(train_argv, "--run-id", None)
    device = torch.device(_value(train_argv, "--device", "cuda"))
    epochs = int(_value(train_argv, "--epochs", C.DEFAULT_EPOCHS))
    images = int(_value(train_argv, "--images", C.N_TRAIN))
    batch = int(_value(train_argv, "--batch", C.DEFAULT_BATCH))
    override = _value(train_argv, "--steps", None)
    total = int(override) if override is not None else epochs * (images // batch)
    per_epoch = images // batch
    start_tau = float(_value(train_argv, "--codeword-temperature", 0.0))
    end_tau = float(_value(
        train_argv, "--codeword-temperature-end", start_tau))
    optimizer_mode = _value(train_argv, "--optimizer-mode", "cayley_sgd")
    policy_temperature = float(_value(
        train_argv, "--temperature", C.DEFAULT_TEMPERATURE))
    selected = checkpoint_steps(
        total, diag.gradient_fractions, diag.gradient_steps)
    allocations, moves = fixed_exchange_set(
        anchor, diag.gradient_allocations, diag.gradient_seed, device)
    if int(diag.gradient_batches) * int(diag.gradient_batch) > C.N_VAL:
        raise ValueError("gradient images exceed the frozen validation split")

    C.PLAN = "v18_training_trajectory_diagnostics"
    C.output_dir = lambda current, current_run: (
        C.PHASE1 / "v18" / str(current_run) / current.name)
    from ..v12 import train
    original = train.joint_validation
    records = []
    measured_steps, saved_steps = set(), set()
    output = C.output_dir(anchor, run_id) / "u_gradient_trajectory.json"

    def save_snapshot(codec, distribution, step, validation):
        checkpoint = output.parent / "checkpoints" / f"step_{int(step):06d}"
        checkpoint.mkdir(parents=True, exist_ok=False)
        save_codec_v1(codec, checkpoint / "codec.pt")
        logits = (distribution.scores.detach() * policy_temperature).cpu()
        torch.save({
            "groups": int(distribution.groups),
            "bit_costs": tuple(anchor.mode_bits),
            "total_bits": int(anchor.rate),
            "temperature": policy_temperature,
            "policy_state": {"logits": logits},
            "step": int(step)}, checkpoint / "policy.pt")
        (checkpoint / "meta.json").write_text(json.dumps({
            "step": int(step),
            "map_allocation": validation["map_allocation"],
            "map_distortion": validation["map_distortion"],
            "entropy": validation["entropy"],
            "validation_split": "v12 train_val, frozen leading rows"}, indent=2))

    def wrapped(codec, tail, resident, distribution, step, samples):
        result = original(codec, tail, resident, distribution, step, samples)
        step = int(step)
        should_save = (step == 0 or step == total or
                       (diag.snapshot_every > 0 and
                        step % int(diag.snapshot_every) == 0))
        if should_save and step not in saved_steps:
            save_snapshot(codec, distribution, step, result)
            saved_steps.add(step)
        if step not in selected or step in measured_steps:
            return result
        if optimizer_mode == "orfc_adam":
            epoch = (max(step, 1) - 1) // per_epoch + 1
            tau = train.codeword_tau(epoch, epochs, start_tau, end_tau)
        else:
            tau = train.codeword_tau(max(int(step), 1), total,
                                     start_tau, end_tau)
        cpu_state = torch.random.get_rng_state()
        cuda_state = (torch.cuda.get_rng_state(device)
                      if device.type == "cuda" else None)
        try:
            with torch.enable_grad():
                record = gradient_snapshot(
                    codec, tail, resident, allocations,
                    diag.gradient_batches, diag.gradient_batch, tau, step)
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, device)
        records.append(record)
        measured_steps.add(step)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "contract": "unchanged training; read-only fixed-image/fixed-allocation probes",
            "gradient": "hard forward, training-temperature codeword surrogate backward, Euclidean dD/dU projected to O(D) tangent",
            "anchor": anchor.name, "run_id": run_id,
            "checkpoint_steps": selected,
            "fixed_validation_rows": list(range(
                int(diag.gradient_batches) * int(diag.gradient_batch))),
            "allocations": allocations.detach().cpu().tolist(), "moves": moves,
            "records": records}
        output.write_text(json.dumps(payload, indent=2))
        print(f"[{anchor.name}] U-gradient probe step={step} "
              f"neg={record['negative_fraction']:.4f} "
              f"median={record['cosine_median']:.4f}", flush=True)
        return result

    train.joint_validation = wrapped
    try:
        train.main(train_argv)
    finally:
        train.joint_validation = original


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
