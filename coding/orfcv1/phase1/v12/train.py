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
from .strict_fair import strict_fair_slate
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
    for parameter in codec.transform.parameters():
        parameter.requires_grad_(True)
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


def set_lr(optimizer, step, total, base):
    value = C.cosine_lr(step, total, base)
    for group in optimizer.param_groups:
        group["lr"] = value


def transform_grad_norm(codec):
    gradients = [p.grad.norm() for p in codec.transform.parameters()
                 if p.grad is not None]
    return float(torch.stack(gradients).norm()) if gradients else 0.0


def codeword_tau(step, total, start, end):
    end = start if end is None else end
    if start == end:
        return float(start)
    if start <= 0 or end <= 0:
        raise ValueError("annealed codeword temperatures must be positive")
    progress = (step - 1) / max(total - 1, 1)
    return float(start * (end / start) ** progress)


def add_coverage(table, allocations):
    rows = torch.as_tensor(allocations, device=table.device, dtype=torch.long)
    if rows.ndim == 1:
        rows = rows[None]
    index = rows.t().contiguous()
    table.scatter_add_(1, index, torch.ones_like(index, dtype=table.dtype))


def fair_policy_weights(distribution, allocations, floor):
    """Policy-relative slate weights with an explicit positive lower bound."""
    count = int(allocations.shape[0])
    floor = float(floor)
    if not 0 <= floor < 1 / count:
        raise ValueError("fair weight floor must lie in [0, 1/slate_size)")
    logp = distribution.log_prob(allocations, validate=False).detach().double()
    weights = torch.softmax(logp, dim=0).to(dtype=torch.float32)
    weights = weights * (1 - count * floor) + floor
    return weights / weights.sum()


def clone_gradient_blocks(codec):
    blocks = [list(codec.transform.parameters())]
    blocks += [[quantizer.codebooks] for quantizer in codec.pq.quantizers]
    return [[None if p.grad is None else p.grad.detach().clone() for p in block]
            for block in blocks]


def restore_norm_matched_gradients(codec, source, target):
    """Restore source directions with each block norm matched to target."""
    blocks = [list(codec.transform.parameters())]
    blocks += [[quantizer.codebooks] for quantizer in codec.pq.quantizers]
    ratios = []
    for parameters, source_block, target_block in zip(blocks, source, target):
        source_norm = torch.sqrt(sum(
            gradient.square().sum() for gradient in source_block
            if gradient is not None))
        target_norm = torch.sqrt(sum(
            gradient.square().sum() for gradient in target_block
            if gradient is not None))
        if not bool(torch.isfinite(source_norm) & torch.isfinite(target_norm)):
            raise RuntimeError("non-finite strict-fair gradient norm")
        if float(source_norm) == 0:
            raise RuntimeError("zero strict-fair gradient cannot be norm matched")
        ratio = target_norm / source_norm
        for parameter, gradient in zip(parameters, source_block):
            parameter.grad = None if gradient is None else gradient * ratio
        ratios.append(float(ratio))
    return {"U": ratios[0], "codebooks": ratios[1:]}


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
        book_optimizer="adam", grad_clip=1.0,
        policy_gradient="reinforce", codeword_temperature=0.0,
        codeword_temperature_end=None, optimizer_mode="cayley_sgd",
        num_workers=C.DATALOADER_WORKERS, coverage_samples=0,
        coverage_weight=0.0, coverage_fraction=1.0,
        strict_fair_codec=False,
        strict_fair_weighting="equal", fair_weight_floor=1e-4,
        stream_allocations=False, image_microbatch=0,
        run_neighbor_audit=True):
    started = time.time()
    run_id = C.ensure_run_id(run_id)
    out = C.output_dir(anchor, run_id)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is non-empty; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    engine.configure_precision(C.ALLOW_TF32)

    parameterization = ("orfc_cayley" if optimizer_mode == "orfc_adam"
                        else "direct")
    codec, metadata = init_mod.load_checked(
        anchor, device, parameterization=parameterization,
        require_full=images is None)
    make_trainable(codec)
    initial_rotation = codec.transform.get_rotation().detach().clone()
    initial_books = [q.codebooks.detach().clone() for q in codec.pq.quantizers]
    book_optimizer_name = book_optimizer
    bits = actual_mode_bits(codec)
    if bits != tuple(anchor.mode_bits):
        raise SystemExit(f"INVALID_EXPERIMENT: menu {bits} != {anchor.mode_bits}")
    policy = FixedBudgetAllocationPolicy(C.GROUPS, bits, anchor.rate).to(device)
    with torch.no_grad():
        policy.logits[:, anchor.uniform_mode].fill_(1e-6)
    policy_optimizer = torch.optim.Adam([policy.logits], lr=float(policy_lr))
    codec_optimizer = None
    if optimizer_mode == "orfc_adam":
        if not hasattr(codec.transform, "triu_params"):
            raise ValueError("orfc_adam requires ORFC Cayley triu parameters")
        if book_optimizer_name != "adam":
            raise ValueError("orfc_adam requires the Adam codebook optimizer")
        if float(lr_u) != float(lr_theta):
            raise ValueError("orfc_adam requires one shared lr for U and codebooks")
        parameters = list(codec.transform.parameters()) + [
            q.codebooks for q in codec.pq.quantizers]
        codec_optimizer = torch.optim.Adam(parameters, lr=float(lr_theta))
        rotation_optimizer = book_optimizer = None
    elif optimizer_mode == "cayley_sgd":
        rotation_optimizer, book_optimizer = build_optimizers(
            codec, lr_u, lr_theta, book_optimizer_name)
    else:
        raise ValueError(f"unknown optimizer mode {optimizer_mode!r}")

    tail = tail_mod.build_tail(C.LAYER, device)
    train_paths = C.load_split("train_fit")
    tail_mod.check_layer(C.LAYER, train_paths[0], train_paths[1])
    train_set = CachedFeatureDataset(
        train_paths[0], train_paths[2], max_images=images)
    # Full teacher table in pinned host RAM.  Fancy indexing would allocate an
    # unpinned temporary and defeat non_blocking H2D, so gathers go through a
    # preallocated pinned staging buffer instead.
    teacher_host = torch.from_numpy(np.load(train_paths[1])).float()
    if device.type == "cuda":
        teacher_host = teacher_host.pin_memory()
    val_paths = C.load_split("train_val")
    val = engine.ResidentSet(*val_paths[:3], device,
                             max_images=None if images is None else min(images, C.N_VAL))
    batch = int(batch)
    num_workers = max(0, int(num_workers))
    teacher_staging = torch.empty(
        (batch, *teacher_host.shape[1:]),
        dtype=teacher_host.dtype,
        pin_memory=(device.type == "cuda"))
    loader_generator = torch.Generator().manual_seed(C.TRAIN_SEED)
    loader_kwargs = dict(
        batch_size=batch, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        generator=loader_generator)
    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(train_set, **loader_kwargs)
    per_epoch = len(train_loader)
    if per_epoch == 0:
        raise ValueError(f"batch {batch} exceeds {len(train_set)} training images")
    schedule_total = int(epochs) * per_epoch
    total = int(steps) if steps is not None else schedule_total
    if total < 1:
        raise ValueError("training requires at least one step")
    policy_samples = int(policy_samples)
    if policy_samples < 2 and policy_gradient == "reinforce":
        raise ValueError("leave-one-out policy gradient requires >=2 samples")
    if policy_gradient not in ("reinforce", "dp_st"):
        raise ValueError("policy_gradient must be reinforce or dp_st")
    if policy_gradient == "dp_st" and not codeword_temperature > 0:
        raise ValueError("dp_st requires a positive codeword temperature")
    coverage_samples = int(coverage_samples)
    coverage_weight, coverage_fraction = map(
        float, (coverage_weight, coverage_fraction))
    legacy_coverage = coverage_samples > 0 or coverage_weight > 0
    strict_fair_codec = bool(strict_fair_codec)
    if strict_fair_weighting not in ("equal", "policy", "norm_match"):
        raise ValueError("unknown strict-fair weighting")
    if strict_fair_weighting != "equal" and not strict_fair_codec:
        raise ValueError("strict-fair weighting requires strict fairness")
    if strict_fair_weighting != "equal" and not stream_allocations:
        raise ValueError("controlled strict-fair weighting requires streaming")
    if strict_fair_codec and legacy_coverage:
        raise ValueError("strict fairness replaces weighted coverage")
    coverage_enabled = legacy_coverage or strict_fair_codec
    if coverage_samples < 0 or not 0 <= coverage_weight <= 1:
        raise ValueError("coverage samples/weight must be nonnegative and weight <= 1")
    if legacy_coverage and (coverage_samples < 1 or coverage_weight <= 0):
        raise ValueError("coverage requires positive samples and weight")
    if coverage_enabled and policy_gradient != "reinforce":
        raise ValueError("exact-budget coverage is separate from DP-ST")
    if stream_allocations and (policy_gradient != "reinforce" or
                               optimizer_mode != "orfc_adam"):
        raise ValueError("streaming currently requires REINFORCE + ORFC Adam")
    if not 0 < coverage_fraction <= 1:
        raise ValueError("coverage_fraction must lie in (0, 1]")
    coverage_until = (total if strict_fair_codec else
                      math.ceil(total * coverage_fraction)
                      if coverage_enabled else 0)
    train_iterator = iter(train_loader)
    generator = torch.Generator(device=device).manual_seed(C.POLICY_SEED)
    policy_coverage = torch.zeros(
        C.GROUPS, len(bits), dtype=torch.long, device=device)
    forced_coverage = torch.zeros_like(policy_coverage)
    coverage_exposure = torch.zeros_like(policy_coverage)
    gradient_coverage = torch.zeros_like(policy_coverage)
    gradient_coverage_window = torch.zeros_like(policy_coverage)
    budget_violations = torch.zeros((), dtype=torch.long, device=device)

    uniform = torch.as_tensor(engine.uniform_allocation(anchor), device=device)
    initial_parity = qhard.selfcheck(codec, tail, val, uniform)
    initial_distortion = validate(codec, tail, val, uniform)
    initial_entropy = float(policy.build(temperature).entropy().detach())
    scale = initial_distortion
    initial_orth = frozen.orthogonality_error(codec)
    trace = []
    validation = [joint_validation(
        codec, tail, val, policy.build(temperature), 0,
        C.STAGE1_VAL_SAMPLES)]
    validation[0]["uniform"] = initial_distortion
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    training_started = time.time()
    last_gradient_norms = {"U": 0.0, "codebooks": 0.0, "policy": 0.0}
    last_gradient_match = None

    for step in range(1, total + 1):
        try:
            rows, x = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            rows, x = next(train_iterator)
        x = x.float().to(device, non_blocking=True)
        # Gather into pinned staging first; teacher_host[rows] would allocate an
        # unpinned temporary and force a synchronous H2D despite non_blocking.
        rows_cpu = rows if rows.device.type == "cpu" else rows.cpu()
        n = int(rows_cpu.shape[0])
        torch.index_select(teacher_host, 0, rows_cpu, out=teacher_staging[:n])
        teacher = teacher_staging[:n].to(device, non_blocking=True)
        with torch.no_grad():
            y, mu, std = batch_normalize_gpu(x, mode=C.NORM_MODE)
        distribution = policy.build(temperature, validate=False)
        policy_allocations = distribution.sample(
            policy_samples, generator=generator)
        coverage_allocations = []
        if strict_fair_codec:
            coverage_allocations = strict_fair_slate(
                C.GROUPS, bits, anchor.rate, device, generator)
            forced_coverage.add_(1)
        elif step <= coverage_until:
            for sample_index in range(coverage_samples):
                target = ((step - 1) * coverage_samples + sample_index)
                target %= C.GROUPS * len(bits)
                group, mode = divmod(target, len(bits))
                coverage_allocations.append(distribution.sample_conditioned(
                    group, mode, generator=generator))
                forced_coverage[group, mode] += 1
        if strict_fair_codec or coverage_allocations:
            if not strict_fair_codec:
                coverage_allocations = torch.cat(coverage_allocations, dim=0)
            add_coverage(coverage_exposure, coverage_allocations)
            allocations = torch.cat(
                (policy_allocations, coverage_allocations), dim=0)
        else:
            coverage_allocations = None
            allocations = policy_allocations
        if policy_gradient == "dp_st":
            marginals = distribution.marginals()
            entropy = distribution.entropy(marginals)
        else:
            marginals = None
            entropy = distribution.entropy()
        budget_violations.add_(
            (policy.actual_rate(allocations) != anchor.rate).sum())
        add_coverage(policy_coverage, policy_allocations)

        if codec_optimizer is None:
            set_lrs(rotation_optimizer, book_optimizer, policy_optimizer,
                    step - 1, schedule_total, lr_u, lr_theta, policy_lr)
            rotation_optimizer.zero_grad(set_to_none=True)
            book_optimizer.zero_grad(set_to_none=True)
        else:
            # Match ORFC exactly: one Adam learning rate, held constant inside
            # an epoch and advanced by CosineAnnealingLR at epoch boundaries.
            epoch_index = (step - 1) // per_epoch
            set_lr(codec_optimizer, epoch_index, int(epochs), lr_theta)
            set_lr(policy_optimizer, step - 1, schedule_total, policy_lr)
            codec_optimizer.zero_grad(set_to_none=True)
        policy_optimizer.zero_grad(set_to_none=True)

        if optimizer_mode == "orfc_adam":
            current_codeword_tau = codeword_tau(
                epoch_index + 1, int(epochs), codeword_temperature,
                codeword_temperature_end)
        else:
            current_codeword_tau = codeword_tau(
                step, total, codeword_temperature, codeword_temperature_end)
        if stream_allocations:
            micro = n if int(image_microbatch) <= 0 else int(image_microbatch)
            count = int(allocations.shape[0])
            n_coverage = count - policy_samples
            if strict_fair_codec:
                equal_fair_weights = torch.full(
                    (n_coverage,), 1 / n_coverage, device=device)
                policy_fair_weights = fair_policy_weights(
                    distribution, coverage_allocations, fair_weight_floor)
                fair_weights = (equal_fair_weights
                                if strict_fair_weighting in ("equal", "norm_match")
                                else policy_fair_weights)
                weights = [0.0] * policy_samples + fair_weights.tolist()
            else:
                weights = ([((1 - coverage_weight) / policy_samples)
                            if n_coverage else (1 / policy_samples)] * policy_samples)
                weights += ([coverage_weight / n_coverage] * n_coverage)
            labels = None

            def stream_pass(pass_allocations, pass_weights):
                values = []
                for allocation, weight in zip(pass_allocations, pass_weights):
                    pieces = []
                    for first in range(0, n, micro):
                        last = min(first + micro, n)
                        value, _ = qhard.distortion_sparse(
                            codec, tail, y[first:last], mu[first:last],
                            std[first:last], teacher[first:last], allocation,
                            codeword_temperature=current_codeword_tau)
                        if float(weight) != 0:
                            (float(weight) * value.mean()
                             * ((last - first) / n)).backward()
                        pieces.append(value.detach())
                    values.append(torch.cat(pieces).mean())
                return torch.stack(values)

            per_allocation = stream_pass(allocations, weights)
            if strict_fair_codec and strict_fair_weighting == "norm_match":
                equal_gradients = clone_gradient_blocks(codec)
                codec_optimizer.zero_grad(set_to_none=True)
                stream_pass(coverage_allocations, policy_fair_weights.tolist())
                policy_gradients = clone_gradient_blocks(codec)
                last_gradient_match = restore_norm_matched_gradients(
                    codec, equal_gradients, policy_gradients)
        else:
            per_image, labels = qhard.distortions(
                codec, tail, y, mu, std, teacher, allocations,
                marginals=marginals,
                codeword_temperature=current_codeword_tau)
            per_allocation = per_image.mean(dim=1)
        policy_distortion = per_allocation[:policy_samples].mean()
        coverage_distortion = (per_allocation[policy_samples:].mean()
                               if coverage_allocations is not None else
                               policy_distortion)
        codec_loss = (coverage_distortion if strict_fair_codec else
                      (1 - coverage_weight) * policy_distortion
                      + coverage_weight * coverage_distortion
                      if coverage_allocations is not None else
                      policy_distortion)
        if optimizer_mode != "orfc_adam":
            codec_loss = codec_loss / scale
        if policy_gradient == "reinforce":
            log_probability = distribution.log_prob(
                policy_allocations, validate=False)
            detached = per_allocation[:policy_samples].detach()
            baseline = (detached.sum() - detached) / (len(detached) - 1)
            advantage = (detached - baseline) / scale
            policy_loss = (advantage * log_probability).mean()
        else:
            policy_loss = codec_loss.new_zeros(())
        entropy_coefficient = C.cosine_lr(
            step, schedule_total, float(entropy_weight), floor_ratio=0.0)
        policy_loss = policy_loss - entropy_coefficient * entropy / C.GROUPS
        if stream_allocations:
            policy_loss.backward()
        else:
            (codec_loss + policy_loss).backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(codec.parameters(), float(grad_clip))
        with torch.no_grad():
            for mode, quantizer in enumerate(codec.pq.quantizers):
                gradient = quantizer.codebooks.grad
                if gradient is not None:
                    active = gradient.square().sum(dim=(1, 2)).gt(0)
                    gradient_coverage[:, mode].add_(active)
                    gradient_coverage_window[:, mode].add_(active)

        if codec_optimizer is None:
            rotation_optimizer.step()
            book_optimizer.step()
        else:
            codec_optimizer.step()
        policy_optimizer.step()
        with torch.no_grad():
            policy.logits.sub_(policy.logits.mean(dim=1, keepdim=True))
        if (optimizer_mode != "orfc_adam" and not stream_allocations
                and step % C.REVIVE_EVERY == 0):
            for modes, selected in zip(allocations, labels):
                qhard.revive_dead_codewords(codec, y, modes, selected)

        trace.append((step, policy_distortion.detach(),
                      coverage_distortion.detach(), policy_loss.detach(),
                      entropy.detach(), coverage_allocations is not None))
        if step % C.VAL_EVERY == 0 or step == total:
            current = policy.build(temperature)
            record = joint_validation(
                codec, tail, val, current, step, C.STAGE1_VAL_SAMPLES)
            record["codeword_temperature"] = current_codeword_tau
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
                f"tau={current_codeword_tau:.5f} "
                f"dMAP={record['map_hamming_from_previous']} "
                f"stable={record['map_stable_run']}")
        if step == 1 or step % C.LOG_EVERY == 0 or step == total:
            # Defer host sync of grad norms to log cadence only.
            book_gradients = [q.codebooks.grad.norm()
                              for q in codec.pq.quantizers
                              if q.codebooks.grad is not None]
            last_gradient_norms = {
                "U": transform_grad_norm(codec),
                "codebooks": float(torch.stack(book_gradients).norm())
                             if book_gradients else 0.0,
                "policy": float(policy.logits.grad.norm())
                          if policy.logits.grad is not None else 0.0}
            log(f"[{anchor.name}] {step}/{total} "
                f"Dpol={float(policy_distortion):.1f} "
                f"Dcov={float(coverage_distortion):.1f} "
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
    neighbor_audit = (local_neighbor_audit(
        codec, tail, val, policy, final_map) if run_neighbor_audit else
        {"skipped": True, "locally_optimal": False})
    book_drift = []
    for initial, quantizer in zip(initial_books, codec.pq.quantizers):
        delta = (quantizer.codebooks.detach() - initial).flatten(1).norm(dim=1)
        base = initial.flatten(1).norm(dim=1).clamp_min(
            torch.finfo(initial.dtype).eps)
        book_drift.append(delta / base)
    book_drift = torch.stack(book_drift, dim=1)
    final_rotation = codec.transform.get_rotation().detach()
    rotation_drift = float(
        (final_rotation - initial_rotation).norm()
        / initial_rotation.norm().clamp_min(
            torch.finfo(initial_rotation.dtype).eps))
    recent = validation[-C.STAGE1_STABILITY_POINTS:]
    stable = (len(recent) == C.STAGE1_STABILITY_POINTS and
              all(item["map_allocation"] == recent[-1]["map_allocation"]
                  for item in recent))
    uniform_modes = torch.full_like(final_map, anchor.uniform_mode)
    if codec_optimizer is None:
        final_lrs = {"U": rotation_optimizer.param_groups[0]["lr"],
                     "codebooks": book_optimizer.param_groups[0]["lr"]}
    else:
        final_lrs = {"U": codec_optimizer.param_groups[0]["lr"],
                     "codebooks": codec_optimizer.param_groups[0]["lr"]}
    final_lrs["policy"] = policy_optimizer.param_groups[0]["lr"]
    joint = {
        "contract": "one continuous exact-budget trajectory jointly updates "
                    "U, every sampled PQ mode, and the allocation policy",
        "policy_gradient": policy_gradient,
        "codeword_temperature": float(codeword_temperature),
        "codeword_temperature_end": float(
            codeword_temperature if codeword_temperature_end is None
            else codeword_temperature_end),
        "optimizer_mode": optimizer_mode,
        "logical_batch": batch,
        "stream_allocations": bool(stream_allocations),
        "image_microbatch": int(image_microbatch),
        "neighbor_audit_enabled": bool(run_neighbor_audit),
        "codec_loss_scale": "raw" if optimizer_mode == "orfc_adam"
                            else "initial_distortion",
        "single_continuous_trajectory": True,
        "exact_budget_coverage": {
            "enabled": coverage_enabled,
            "strict_fair": strict_fair_codec,
            "strict_fair_slate_size": len(bits) if strict_fair_codec else 0,
            "strict_fair_weighting": strict_fair_weighting,
            "fair_weight_floor": float(fair_weight_floor),
            "last_gradient_match": last_gradient_match,
            "samples": coverage_samples,
            "weight": coverage_weight,
            "fraction": coverage_fraction,
            "until_step": coverage_until,
            "forced_exposure": forced_coverage.cpu().tolist(),
            "forced_exposure_min": int(forced_coverage.min()),
            "allocation_exposure": coverage_exposure.cpu().tolist(),
            "allocation_exposure_min": int(coverage_exposure.min()),
            "policy_gradient_uses_coverage": False},
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
              "coverage_mean": float(item[2]),
              "policy_loss": float(item[3]), "entropy": float(item[4]),
              "coverage_active": bool(item[5])}
             for item in trace]
    payload = {"plan": getattr(C, "PLAN", "continuous_joint_v1"),
               "anchor": anchor.name,
               "rate": anchor.rate,
               "run_id": run_id, "init": metadata, "images": len(train_set),
               "val_images": val.count, "batch": batch, "epochs": float(total / per_epoch),
               "steps": total, "steps_per_epoch": per_epoch,
               "schedule_steps": schedule_total,
               "schedule_epochs": int(epochs),
               "lr_U": lr_u, "lr_theta": lr_theta, "policy_lr": policy_lr,
               "policy_samples": policy_samples,
               "coverage_samples": coverage_samples,
               "coverage_weight": coverage_weight,
               "coverage_fraction": coverage_fraction,
               "strict_fair_weighting": strict_fair_weighting,
               "fair_weight_floor": float(fair_weight_floor),
               "optimizer_mode": optimizer_mode,
               "book_optimizer": book_optimizer_name,
               "grad_clip": grad_clip,
               "temperature": temperature, "entropy_weight": entropy_weight,
               "policy_gradient": policy_gradient,
               "codeword_temperature": float(codeword_temperature),
               "codeword_temperature_end": float(
                   codeword_temperature if codeword_temperature_end is None
                   else codeword_temperature_end),
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
    parser.add_argument("--num-workers", type=int, default=C.DATALOADER_WORKERS)
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
    parser.add_argument("--policy-gradient", choices=("reinforce", "dp_st"),
                        default="reinforce")
    parser.add_argument("--codeword-temperature", type=float, default=0.0)
    parser.add_argument("--codeword-temperature-end", type=float, default=None)
    parser.add_argument("--optimizer-mode", choices=("cayley_sgd", "orfc_adam"),
                        default="cayley_sgd")
    parser.add_argument("--coverage-samples", type=int, default=0)
    parser.add_argument("--coverage-weight", type=float, default=0.0)
    parser.add_argument("--coverage-fraction", type=float, default=1.0)
    parser.add_argument("--strict-fair-codec", action="store_true")
    parser.add_argument("--strict-fair-weighting",
                        choices=("equal", "policy", "norm_match"),
                        default="equal")
    parser.add_argument("--fair-weight-floor", type=float, default=1e-4)
    parser.add_argument("--stream-allocations", action="store_true")
    parser.add_argument("--image-microbatch", type=int, default=0)
    parser.add_argument("--skip-neighbor-audit", action="store_true")
    parser.add_argument("--images", type=int, default=None, help="probe only")
    args = parser.parse_args(argv)
    result = run(C.ANCHOR_BY_NAME[args.anchor], args.run_id,
                 torch.device(args.device), epochs=args.epochs, steps=args.steps,
                 batch=args.batch, lr_u=args.lr_u, lr_theta=args.lr_theta,
                 policy_lr=args.policy_lr, policy_samples=args.policy_samples,
                 temperature=args.temperature,
                 entropy_weight=args.entropy_weight, images=args.images,
                 book_optimizer=args.book_optimizer, grad_clip=args.grad_clip,
                 policy_gradient=args.policy_gradient,
                 codeword_temperature=args.codeword_temperature,
                 codeword_temperature_end=args.codeword_temperature_end,
                 optimizer_mode=args.optimizer_mode,
                 num_workers=args.num_workers,
                 coverage_samples=args.coverage_samples,
                 coverage_weight=args.coverage_weight,
                 coverage_fraction=args.coverage_fraction,
                 strict_fair_codec=args.strict_fair_codec,
                 strict_fair_weighting=args.strict_fair_weighting,
                 fair_weight_floor=args.fair_weight_floor,
                 stream_allocations=args.stream_allocations,
                 image_microbatch=args.image_microbatch,
                 run_neighbor_audit=not args.skip_neighbor_audit)
    print(json.dumps({"anchor": result["anchor"], "seconds": result["seconds"],
                      "validation": result["validation"],
                      "policy": result["policy"],
                      "joint_training": result["joint_training"]}, indent=2))


if __name__ == "__main__":
    main()
