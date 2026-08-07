"""Frozen internal audit and held-out paired evaluation for V31."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import engine, tail as tail_mod
from ..v21.config import SPECS, activate
from ..v30 import common, nested
from . import core


def orthogonality(codec):
    rotation = codec.transform.get_rotation().detach().double()
    identity = torch.eye(rotation.shape[0], device=rotation.device,
                         dtype=rotation.dtype)
    return float((rotation.t() @ rotation - identity).norm() /
                 identity.norm())


@torch.no_grad()
def centroid_usage(codec, resident, batch=8):
    result = []
    for mode, size in enumerate(codec.pq.mode_sizes):
        counts = torch.zeros(codec.pq.G, size, dtype=torch.long)
        allocation = torch.full(
            (codec.pq.G,), mode, device=resident.device, dtype=torch.long)
        for first in range(0, resident.count, int(batch)):
            _, labels = nested.reconstruct(
                codec, resident.y[first:first + batch], allocation)
            for group in range(codec.pq.G):
                counts[group] += torch.bincount(
                    labels[group].cpu(), minlength=size)
        result.append(counts)
    dead = sum(int((item == 0).sum()) for item in result)
    total = sum(item.numel() for item in result)
    return {"counts": [item.tolist() for item in result],
            "dead_centroids": dead, "total_centroids": total,
            "dead_fraction": dead / total,
            "minimum_centroid_usage": min(int(item.min()) for item in result)}


def drift(codec, source):
    rows = {}
    for key, value in codec.state_dict().items():
        baseline = source.state_dict()[key]
        if value.is_floating_point() or value.is_complex():
            rows[key] = float((value.detach() - baseline.detach()).norm())
        else:
            rows[key] = 0.0 if torch.equal(value, baseline) else float("inf")
    return {"per_tensor_update_norm": rows,
            "minimum_stage_update_norm": min(
                value for key, value in rows.items() if "pq.stages" in key),
            "transform_update_norm": max(
                value for key, value in rows.items() if "transform" in key)}


def bootstrap_upper(difference, repetitions=10000, seed=20260807):
    generator = np.random.default_rng(seed); means = []
    for _ in range(0, repetitions, 500):
        count = min(500, repetitions - len(means))
        index = generator.integers(0, len(difference),
                                   size=(count, len(difference)))
        means.extend(difference[index].mean(1).tolist())
    return float(np.quantile(np.asarray(means), 0.95))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", default="blk20", choices=tuple(SPECS))
    parser.add_argument("--anchor", required=True, choices=("R64", "R96"))
    parser.add_argument("--bilevel-dir", required=True)
    parser.add_argument("--uniform-dir", required=True)
    parser.add_argument("--source-codec", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-batch", type=int, default=16)
    args = parser.parse_args(argv)
    core.configure_determinism(20260807)
    config = activate(args.block); anchor = config.ANCHOR_BY_NAME[args.anchor]
    device, out = torch.device(args.device), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bilevel_dir, uniform_dir = Path(args.bilevel_dir), Path(args.uniform_dir)
    bilevel_json = json.loads((bilevel_dir / "train.json").read_text())
    uniform_json = json.loads((uniform_dir / "train.json").read_text())
    codec, _ = nested.load_checkpoint(bilevel_dir / "nested_codec.pt", device)
    control, _ = nested.load_checkpoint(uniform_dir / "nested_codec.pt", device)
    source, _ = nested.load_checkpoint(args.source_codec, device)
    allocation = np.load(bilevel_dir / "allocation.npy")
    uniform = np.load(uniform_dir / "allocation.npy")
    bits = common.mode_bits(codec)
    val_paths = config.load_split("train_val")
    tail = tail_mod.build_tail(config.LAYER, device)
    selection = engine.ResidentSet(
        *val_paths[:2], val_paths[2][:500], device)
    select_bilevel = nested.evaluate(
        codec, tail, selection, allocation, args.image_batch)[0]
    select_uniform = nested.evaluate(
        control, tail, selection, uniform, args.image_batch)[0]
    del selection
    exact = (common.nominal_rate(allocation, bits) == anchor.rate and
             common.nominal_rate(uniform, bits) == anchor.rate)
    train_paths = config.load_split("train_fit")
    train = engine.ResidentSet(*train_paths[:2], train_paths[2][:5000], device)
    usage = centroid_usage(codec, train)
    del train
    update = drift(codec, source)
    select_improved = float(select_bilevel.mean()) < float(select_uniform.mean())
    parity_bilevel = abs(float(select_bilevel.mean()) -
                          bilevel_json["final_tail_mse_select500"]) / max(
                              abs(bilevel_json["final_tail_mse_select500"]), 1e-12)
    parity_uniform = abs(float(select_uniform.mean()) -
                         uniform_json["final_tail_mse_select500"]) / max(
                             abs(uniform_json["final_tail_mse_select500"]), 1e-12)
    orth = orthogonality(codec)
    authorization = torch.load(
        bilevel_json["authorization"], map_location="cpu")
    warm_allocations = authorization["prefix"]["allocations"][:300]
    warm_exposure = torch.zeros(codec.pq.G, len(bits), dtype=torch.long)
    groups = torch.arange(codec.pq.G)
    for row in warm_allocations:
        warm_exposure[groups, row] += 1
    events_valid = all(
        not event["accepted"] or (
            event["local_spearman"] >= 0.8 and
            event["winner_strictly_better"] and
            event["winner_realized_as_map"])
        for event in bilevel_json["events"])
    internal = bool(
        exact and bilevel_json["accepted_event_count"] >= 1 and
        int(warm_exposure.min()) >= 100 and events_valid and
        select_improved and parity_bilevel <= 1e-6 and
        parity_uniform <= 1e-6 and orth <= 1e-4 and
        update["minimum_stage_update_norm"] > 0 and
        bilevel_json["codec_committed_update_count"] ==
        uniform_json["codec_committed_update_count"])
    result = {
        "plan": "v31_frozen_audit_and_heldout", "block": args.block,
        "anchor": args.anchor, "exact_nominal_budgets": exact,
        "allocation": allocation.tolist(), "uniform_allocation": uniform.tolist(),
        "accepted_event_count": bilevel_json["accepted_event_count"],
        "warmup_group_mode_exposure_min": int(warm_exposure.min()),
        "accepted_events_contract_valid": events_valid,
        "bilevel_scheduler_count": bilevel_json["codec_committed_update_count"],
        "uniform_scheduler_count": uniform_json["codec_committed_update_count"],
        "orthogonality": orth, "maximum_orthogonality": 1e-4,
        "hard_parity_relative": {"bilevel": parity_bilevel,
                                 "uniform": parity_uniform,
                                 "maximum": 1e-6},
        "parameter_drift": update, "centroid_usage": usage,
        "selection": {"images": 500,
                      "bilevel_mean": float(select_bilevel.mean()),
                      "uniform_mean": float(select_uniform.mean()),
                      "mean_difference": float((select_bilevel-select_uniform).mean()),
                      "bilevel_strictly_better": select_improved},
        "internal_contract_verdict": "PASS" if internal else "FAIL"}
    np.savez_compressed(out / "selection_per_image.npz",
                        selection_bilevel=select_bilevel,
                        selection_uniform=select_uniform)
    if not internal:
        result["heldout"] = {"status": "NOT_OPENED_INTERNAL_FAIL"}
        (out / "evaluation.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        return
    heldout = engine.ResidentSet(
        *val_paths[:2], val_paths[2][500:3000], device)
    heldout_bilevel = nested.evaluate(
        codec, tail, heldout, allocation, args.image_batch)[0]
    heldout_uniform = nested.evaluate(
        control, tail, heldout, uniform, args.image_batch)[0]
    del heldout
    difference = heldout_bilevel - heldout_uniform
    upper = bootstrap_upper(difference)
    result["heldout"] = {
                    "images": 2500, "selection_role": False,
                    "bootstrap_repetitions": 10000,
                    "bootstrap_seed": 20260807,
                    "bilevel_mean": float(heldout_bilevel.mean()),
                    "uniform_mean": float(heldout_uniform.mean()),
                    "paired_mean_difference": float(difference.mean()),
                    "paired_relative_difference": float(
                        difference.mean()/heldout_uniform.mean()),
                    "one_sided_upper_95": upper,
                    "improved_images": int((difference < 0).sum()),
                    "verdict": "PASS" if upper < 0 else "FAIL"}
    np.savez_compressed(out / "heldout_per_image.npz",
                        heldout_bilevel=heldout_bilevel,
                        heldout_uniform=heldout_uniform)
    (out / "evaluation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
