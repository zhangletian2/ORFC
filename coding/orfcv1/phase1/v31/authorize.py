"""V31 authorization gate with warm Adam/scheduler ranking semantics."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .. import engine, tail as tail_mod
from ..v21.config import SPECS, activate
from ..v30 import common, nested
from . import core


def cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return copy.deepcopy(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", default="blk20", choices=tuple(SPECS))
    parser.add_argument("--anchor", required=True, choices=("R64", "R96"))
    parser.add_argument("--nested-codec", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    core.configure_determinism(args.seed)

    config = activate(args.block); anchor = config.ANCHOR_BY_NAME[args.anchor]
    device, out = torch.device(args.device), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is non-empty")
    out.mkdir(parents=True, exist_ok=True)
    codec, source = nested.load_checkpoint(args.nested_codec, device)
    coverage = source.get("extra", {}).get("coverage_export", {})
    capacity = source.get("extra", {}).get("capacity_gate", {})
    if capacity.get("verdict") != "PASS":
        raise SystemExit("source is not backed by a PASS capacity gate")
    if not coverage or coverage.get("smoke", True):
        raise SystemExit("V31 requires a formal coverage checkpoint")
    bits = common.mode_bits(codec)
    if bits != tuple(anchor.mode_bits):
        raise SystemExit("source mode menu does not match anchor")

    paths = config.load_split("train_fit")
    train_n = 160 if args.smoke else 5000
    train = engine.ResidentSet(*paths[:2], paths[2][:train_n], device)
    cost_n = 8 if args.smoke else 512
    cost = core.ResidentView(train, 0, cost_n)
    adapt_first = cost_n
    adapt_n = 128
    if adapt_first + adapt_n > train.count:
        adapt_first = train.count - adapt_n
    adapt = core.ResidentView(train, adapt_first, adapt_n)
    val_paths = config.load_split("train_val")
    val_n = 8 if args.smoke else 500
    validation = engine.ResidentSet(*val_paths[:2], val_paths[2][:val_n], device)
    tail = tail_mod.build_tail(config.LAYER, device)
    uniform = engine.uniform_allocation(anchor, config.GROUPS)
    uniform_mode = int(np.asarray(uniform)[0])
    policy, policy_optimizer = core.build_policy(
        config.GROUPS, bits, anchor.rate, uniform_mode, device)
    codec_optimizer, scheduler = core.build_codec_optimizer(codec)

    steps = 6 if args.smoke else core.AUTHORIZATION_STEPS
    prefix = core.run_authorization_prefix(
        codec, policy, codec_optimizer, scheduler, tail, train,
        config.GROUPS, bits, anchor.rate, args.seed, steps=steps)
    incumbent = policy.build(core.POLICY_TEMPERATURE).map_allocation().cpu().numpy()
    topk, random_count = ((2, 2) if args.smoke else (8, 8))
    pool, predicted, costs = common.candidate_pool(
        codec, tail, cost, incumbent, anchor.rate, topk, random_count,
        16, args.seed + steps)
    shared = nested.evaluate(codec, tail, validation, pool, 16).mean(1)
    batches = core.fixed_adaptation_batches(adapt)
    adapted = []
    for allocation in pool:
        branch, branch_optimizer, branch_scheduler, losses = core.adapt_branch(
            codec, codec_optimizer, scheduler, tail, adapt, allocation, batches)
        adapted.append(float(nested.evaluate(
            branch, tail, validation, allocation, 16).mean()))
        del branch, branch_optimizer, branch_scheduler
    adapted = np.asarray(adapted)
    rho = core.spearman(shared, adapted)
    k = min(8, len(pool))
    shared_top = set(np.argsort(shared, kind="stable")[:k].tolist())
    adapted_top = set(np.argsort(adapted, kind="stable")[:k].tolist())
    best_recalled = int(np.argmin(adapted)) in shared_top
    recall = len(shared_top & adapted_top) / k
    formal_pass = bool(rho >= 0.8 and best_recalled)
    verdict = "SMOKE_COMPLETE" if args.smoke else (
        "PASS" if formal_pass else "FAIL")

    result = {
        "plan": "v31_warm_state_authorization", "block": args.block,
        "anchor": args.anchor, "smoke": args.smoke, "seed": args.seed,
        "source_codec": str(Path(args.nested_codec).resolve()),
        "mode_bits": list(bits), "exact_rate": common.nominal_rate(incumbent, bits),
        "contract": {
            "authorization_state_discarded": True,
            "prefix_steps": steps, "formal_prefix_steps": 500,
            "warmup_steps": 300, "codec_lr": core.CODEC_LR,
            "schedule": "committed_clipped_cosine", "schedule_max": 5580,
            "deterministic_backend": "xformers_off_math_sdpa",
            "deterministic_algorithms": True,
            "cost_train": cost.count, "formal_cost_train": 512,
            "adapt_train": adapt.count, "adapt_steps": 8,
            "adapt_batch": 16, "validation_select": validation.count,
            "same_adaptation_batches": True,
            "warm_codec_adam": True, "warm_scheduler": True,
            "average_rank_spearman": True,
            "candidate_requested_including_incumbent": 1 + topk + random_count,
            "candidate_actual_after_dedup": len(pool)},
        "incumbent": incumbent.tolist(), "allocations": pool.tolist(),
        "predicted_scores": predicted.tolist(), "shared_scores": shared.tolist(),
        "adapted_scores": adapted.tolist(), "spearman": rho,
        "shared_top8_adapted_top8_recall": recall,
        "adapted_best_in_shared_top8": best_recalled,
        "minimum_spearman": 0.8, "verdict": verdict}
    warm = {
        "format": "v31_authorization_warm_state_v1", "result": result,
        "codec": cpu_state(codec.state_dict()),
        "codec_optimizer": cpu_state(codec_optimizer.state_dict()),
        "scheduler": scheduler.state_dict(),
        "policy": cpu_state(policy.state_dict()),
        "policy_optimizer": cpu_state(policy_optimizer.state_dict()),
        "prefix": {key: cpu_state(value) for key, value in prefix.items()
                   if key not in {"batch_generator", "allocation_generator", "fair"}},
        "candidate_costs": costs.tolist()}
    torch.save(warm, out / "authorization_state.pt")
    (out / "authorization.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
