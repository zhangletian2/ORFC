"""Matched fixed-uniform control for a completed formal V31 run."""

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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", default="blk20", choices=tuple(SPECS))
    parser.add_argument("--anchor", required=True, choices=("R64", "R96"))
    parser.add_argument("--nested-codec", required=True)
    parser.add_argument("--bilevel-result", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--image-batch", type=int, default=16)
    args = parser.parse_args(argv)
    core.configure_determinism(args.seed)
    bilevel = json.loads(Path(args.bilevel_result).read_text())
    if (bilevel.get("plan") != "v31_warm_state_bilevel" or
            bilevel.get("smoke") or bilevel.get("block") != args.block or
            bilevel.get("anchor") != args.anchor):
        raise SystemExit("a matching formal V31 bilevel result is required")
    accepted_steps = set(map(int, bilevel["accepted_event_steps"]))
    config = activate(args.block); anchor = config.ANCHOR_BY_NAME[args.anchor]
    device, out = torch.device(args.device), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is non-empty")
    out.mkdir(parents=True, exist_ok=True)
    codec, source = nested.load_checkpoint(args.nested_codec, device)
    bits = common.mode_bits(codec)
    paths = config.load_split("train_fit")
    train = engine.ResidentSet(*paths[:2], paths[2][:5000], device)
    adapt = core.ResidentView(train, 512, 128)
    adapt_batches = core.fixed_adaptation_batches(adapt)
    val_paths = config.load_split("train_val")
    validation = engine.ResidentSet(*val_paths[:2], val_paths[2][:500], device)
    tail = tail_mod.build_tail(config.LAYER, device)
    uniform = engine.uniform_allocation(anchor, config.GROUPS)
    policy, _ = core.build_policy(
        config.GROUPS, bits, anchor.rate, int(uniform[0]), device)
    optimizer, scheduler = core.build_codec_optimizer(codec)
    initial = float(nested.evaluate(
        codec, tail, validation, uniform, args.image_batch).mean())
    batch_generator, allocation_generator = core.generators(device, args.seed)
    fair = None; exposure = torch.zeros(
        config.GROUPS, len(bits), dtype=torch.long, device=device)
    for step in range(1, 5001):
        if step <= 300:
            allocation, fair = core.inner_allocation(
                step, policy, fair, config.GROUPS, bits, anchor.rate,
                device, allocation_generator)
        else:
            allocation = torch.as_tensor(uniform, device=device)
        index = torch.randint(
            train.count, (32,), generator=batch_generator, device=device)
        exposure[torch.arange(config.GROUPS, device=device), allocation] += 1
        core.codec_step(codec, optimizer, scheduler, tail, train,
                        allocation, index)
        if step in accepted_steps:
            for adapt_index in adapt_batches:
                core.codec_step(codec, optimizer, scheduler, tail, adapt,
                                uniform, adapt_index)
    for _ in range(500):
        index = torch.randint(
            train.count, (32,), generator=batch_generator, device=device)
        core.codec_step(codec, optimizer, scheduler, tail, train,
                        uniform, index)
    final = float(nested.evaluate(
        codec, tail, validation, uniform, args.image_batch).mean())
    exact = common.nominal_rate(uniform, bits) == anchor.rate
    result = {
        "plan": "v31_matched_fixed_uniform", "block": args.block,
        "anchor": args.anchor, "source_codec": str(Path(args.nested_codec).resolve()),
        "matched_bilevel_result": str(Path(args.bilevel_result).resolve()),
        "accepted_event_steps_compensated": sorted(accepted_steps),
        "deterministic_backend": "xformers_off_math_sdpa",
        "compensated_updates": 8 * len(accepted_steps),
        "same_first300_strict_fair": True, "main_uniform_steps": 4700,
        "final_uniform_steps": 500, "codec_committed_update_count": scheduler.count,
        "initial_tail_mse_select500": initial,
        "final_tail_mse_select500": final,
        "relative_change": final / initial - 1.0,
        "final_allocation": np.asarray(uniform).tolist(),
        "exact_nominal_budget": exact,
        "group_mode_exposure": exposure.cpu().tolist(),
        "group_mode_exposure_min_first300_by_contract": 100,
        "internal_verdict": "PASS" if exact and final < initial else "FAIL"}
    nested.save_checkpoint(codec, source["source_codec"],
                           out / "nested_codec.pt",
                           extra={"v31_uniform_control": result})
    np.save(out / "allocation.npy", uniform)
    (out / "train.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
