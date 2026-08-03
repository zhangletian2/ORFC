"""Replay three hard-distortion metrics on periodic joint-training states."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

from codec_v1 import load_codec_v1

from .. import engine
from .. import tail as tail_mod
from ..v12 import config as C
from ..v12.allocation_policy import FixedBudgetAllocationPolicy


REQUIRED = ("codec.pt", "policy.pt", "meta.json")


def _step(path, meta):
    if "step" in meta:
        return int(meta["step"])
    match = re.search(r"(\d+)$", path.name)
    if match is None:
        raise ValueError(f"cannot infer step from {path}")
    return int(match.group(1))


def _load_policy(path, device, expected_step):
    saved = torch.load(path, map_location=device)
    missing = {"policy_state", "groups", "bit_costs", "total_bits",
               "temperature", "step"} - set(saved)
    if missing:
        raise ValueError(f"{path} lacks policy fields {sorted(missing)}")
    policy = FixedBudgetAllocationPolicy(
        saved["groups"], tuple(saved["bit_costs"]), saved["total_bits"]
    ).to(device)
    policy.load_state_dict(saved["policy_state"])
    if int(saved["step"]) != int(expected_step):
        raise ValueError(f"{path} step disagrees with meta.json")
    return policy, policy.build(float(saved["temperature"]))


def _row_digest(rows):
    return hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()


def _evaluate(codec, tail, resident, allocations):
    rows = np.asarray(torch.as_tensor(allocations).cpu(), dtype=np.int64)
    unique, inverse = np.unique(rows, axis=0, return_inverse=True)
    values = engine.evaluate_allocations(
        codec, tail, resident, unique, image_batch=C.EVAL_IMAGE_BATCH,
        pair_budget=C.EVAL_PAIR_BUDGET, per_image=True)
    return values[inverse]


def replay(checkpoint_root, final_allocation, anchor, device, samples, seed,
           max_images=None):
    checkpoint_root = Path(checkpoint_root)
    directories = []
    for path in checkpoint_root.iterdir():
        if path.is_dir() and all((path / name).exists() for name in REQUIRED):
            meta = json.loads((path / "meta.json").read_text())
            directories.append((_step(path, meta), path, meta))
    directories.sort(key=lambda item: item[0])
    if not directories:
        raise ValueError(f"no complete checkpoints under {checkpoint_root}")

    final = np.load(final_allocation).astype(np.int64, copy=False)
    if final.shape != (C.GROUPS,):
        raise ValueError(f"final allocation has shape {final.shape}")
    resident = engine.ResidentSet(
        *C.load_split("train_val")[:3], device, max_images=max_images)
    tail = tail_mod.build_tail(C.LAYER, device)
    records = []
    for step, path, meta in directories:
        codec = load_codec_v1(path / "codec.pt", device=device)
        policy, distribution = _load_policy(path / "policy.pt", device, step)
        current_map = distribution.map_allocation()
        if "map_allocation" not in meta:
            raise ValueError(f"{path / 'meta.json'} lacks map_allocation")
        if current_map.cpu().tolist() != meta["map_allocation"]:
            raise ValueError(f"{path} policy MAP disagrees with meta.json")
        generator = torch.Generator(device=device).manual_seed(int(seed))
        draws = distribution.sample(int(samples), generator=generator)
        all_allocations = torch.cat((
            torch.as_tensor(final, device=device)[None],
            current_map[None], draws), dim=0)
        matrix = _evaluate(codec, tail, resident, all_allocations)
        sample_by_image = matrix[2:].mean(axis=0)
        records.append({
            "step": step,
            "checkpoint": str(path),
            "fixed_final_hard_mse": float(matrix[0].mean()),
            "current_map_hard_mse": float(matrix[1].mean()),
            "policy_mean_hard_mse": float(matrix[2:].mean()),
            "map_switch_delta": float(matrix[1].mean() - matrix[0].mean()),
            "policy_vs_map_delta": float(matrix[2:].mean() - matrix[1].mean()),
            "policy_allocation_sem": float(
                matrix[2:].mean(axis=1).std(ddof=1) / np.sqrt(samples)
                if samples > 1 else 0.0),
            "policy_mean_image_sem": float(
                sample_by_image.std(ddof=1) / np.sqrt(len(sample_by_image))),
            "map_hamming_to_final": int(
                (current_map.cpu().numpy() != final).sum()),
            "entropy": float(distribution.entropy().detach()),
            "current_map": current_map.cpu().tolist(),
            "sample_allocations": draws.cpu().tolist(),
            "sample_unique_allocations": int(
                np.unique(draws.cpu().numpy(), axis=0).shape[0]),
            "meta": meta,
        })
        del codec
    return {
        "contract": "phase1.v18.periodic_replay.v1",
        "anchor": anchor.name,
        "rate": anchor.rate,
        "validation_images": resident.count,
        "validation_row_sha256": _row_digest(resident.rows),
        "policy_samples": int(samples),
        "common_random_seed": int(seed),
        "final_allocation": final.tolist(),
        "records": records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", required=True, choices=tuple(C.ANCHOR_BY_NAME))
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--final-allocation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--max-images", type=int, default=None,
                        help="smoke only; formal replay leaves this unset")
    args = parser.parse_args(argv)
    payload = replay(
        args.checkpoint_root, args.final_allocation,
        C.ANCHOR_BY_NAME[args.anchor], torch.device(args.device),
        args.samples, args.seed, args.max_images)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    summary = {key: value for key, value in payload.items() if key != "records"}
    print(json.dumps(summary, indent=2))
    print(f"records={len(payload['records'])} output={output}")


if __name__ == "__main__":
    main()
