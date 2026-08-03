"""Read-only cosine audit of allocation-specific gradients on the shared U."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from codec_v1 import load_codec_v1

from .. import engine
from .. import tail as tail_mod
from ..v12 import config as C
from ..v12 import qhard
from ..v12.allocation_policy import FixedBudgetAllocationPolicy
from ..v12.train import fixed_rate_neighbors


def tangent_gradient(rotation, gradient):
    product = rotation.t() @ gradient
    symmetric = 0.5 * (product + product.t())
    return gradient - rotation @ symmetric


def allocation_set(policy, map_allocation, count, seed):
    neighbors, _ = fixed_rate_neighbors(policy, map_allocation)
    rows = [np.asarray(torch.as_tensor(map_allocation).detach().cpu(),
                       dtype=np.int64)]
    if len(neighbors):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(neighbors))[:max(0, count - 1)]
        rows.extend(neighbors[order])
    if len(rows) < count:
        generator = torch.Generator(device=policy.logits.device).manual_seed(seed)
        distribution = policy.build()
        seen = {tuple(row.tolist()) for row in rows}
        for row in distribution.sample(count * 16, generator=generator).cpu().numpy():
            if tuple(row.tolist()) not in seen:
                rows.append(row)
                seen.add(tuple(row.tolist()))
            if len(rows) == count:
                break
    if len(rows) < 2:
        raise RuntimeError("fewer than two distinct exact-budget allocations")
    return torch.as_tensor(np.stack(rows[:count]), device=policy.logits.device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", required=True, choices=tuple(C.ANCHOR_BY_NAME))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--allocations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args(argv)
    anchor = C.ANCHOR_BY_NAME[args.anchor]
    root = C.PHASE1 / "v16" / args.run_id / anchor.name
    device = torch.device(args.device)
    codec = load_codec_v1(root / "codec.pt", device=device)
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    saved = torch.load(root / "policy.pt", map_location=device)
    policy = FixedBudgetAllocationPolicy(
        saved["groups"], saved["bit_costs"], saved["total_bits"]).to(device)
    policy.load_state_dict(saved["policy_state"])
    map_allocation = torch.from_numpy(np.load(root / "allocation.npy")).to(device)
    allocations = allocation_set(
        policy, map_allocation, args.allocations, args.seed)
    resident = engine.ResidentSet(*C.load_split("train_val")[:3], device)
    tail = tail_mod.build_tail(C.LAYER, device)
    cosines, batch_records = [], []
    for batch_index in range(args.batches):
        first = batch_index * args.batch
        last = first + args.batch
        if last > resident.count:
            raise ValueError("requested diagnostic batches exceed validation set")
        tensors = resident.slice(first, last)
        gradients, losses = [], []
        for allocation in allocations:
            rotation = codec.transform.get_rotation().detach().requires_grad_(True)
            value, _ = qhard.distortion_sparse(
                codec, tail, *tensors, allocation, rotation=rotation)
            loss = value.mean()
            gradient, = torch.autograd.grad(loss, rotation)
            projected = tangent_gradient(rotation, gradient).flatten()
            gradients.append(projected / projected.norm().clamp_min(1e-30))
            losses.append(float(loss.detach()))
        matrix = torch.stack(gradients) @ torch.stack(gradients).t()
        indices = torch.triu_indices(len(gradients), len(gradients), offset=1,
                                     device=device)
        values = matrix[indices[0], indices[1]].detach().cpu().numpy()
        cosines.extend(values.tolist())
        batch_records.append({"batch": batch_index, "losses": losses,
                              "negative_fraction": float((values < 0).mean()),
                              "cosine_median": float(np.median(values))})
    values = np.asarray(cosines)
    result = {
        "anchor": anchor.name, "run_id": args.run_id,
        "gradient": "Euclidean dD/dU projected to the O(D) tangent space",
        "allocation_source": "final MAP plus distinct one-bit exchange neighbors",
        "allocations": allocations.cpu().tolist(),
        "batches": args.batches, "images_per_batch": args.batch,
        "pair_count": int(values.size),
        "negative_fraction": float((values < 0).mean()),
        "strong_negative_fraction": float((values < -0.1).mean()),
        "cosine_mean": float(values.mean()),
        "cosine_median": float(np.median(values)),
        "cosine_q10": float(np.quantile(values, 0.1)),
        "cosine_q25": float(np.quantile(values, 0.25)),
        "cosine_min": float(values.min()),
        "batch_records": batch_records}
    output = root / "u_gradient_conflict.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
