"""Formal V31 warm-state bilevel training and matched uniform control."""

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


def require_authorization(path, block, anchor):
    payload = torch.load(path, map_location="cpu")
    result = payload.get("result", {})
    if (payload.get("format") != "v31_authorization_warm_state_v1" or
            result.get("verdict") != "PASS" or result.get("smoke") or
            result.get("block") != block or result.get("anchor") != anchor):
        raise SystemExit("formal matching V31 authorization PASS is required")
    return payload


def clone_policy(policy, optimizer):
    branch = copy.deepcopy(policy)
    branch_optimizer = torch.optim.Adam(
        branch.parameters(), lr=core.POLICY_LR)
    branch_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    return branch, branch_optimizer


def realize(policy, optimizer, target, maximum=250):
    target = torch.as_tensor(target, device=policy.logits.device)
    for iteration in range(1, int(maximum) + 1):
        loss = -policy.build(core.POLICY_TEMPERATURE).log_prob(target)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        with torch.no_grad():
            policy.logits.sub_(policy.logits.mean(1, keepdim=True))
        if torch.equal(policy.build(
                core.POLICY_TEMPERATURE).map_allocation(), target):
            return True, iteration
    return False, int(maximum)


def outer_event(codec, codec_optimizer, scheduler, policy, policy_optimizer,
                tail, cost, adapt, validation, rate, step, seed, image_batch):
    incumbent = policy.build(core.POLICY_TEMPERATURE).map_allocation().cpu().numpy()
    pool, predicted, costs = common.candidate_pool(
        codec, tail, cost, incumbent, rate, 8, 8,
        image_batch, seed + step)
    shared = nested.evaluate(codec, tail, validation, pool, image_batch).mean(1)
    incumbent_index = next(index for index, row in enumerate(pool)
                           if np.array_equal(row, incumbent))
    ranked = [index for index in np.argsort(shared, kind="stable").tolist()
              if index != incumbent_index]
    keep = np.asarray([incumbent_index] + ranked[:3], dtype=np.int64)
    kept = pool[keep]
    batches = core.fixed_adaptation_batches(adapt)
    branches, adapted = [], []
    for allocation in kept:
        branch = core.adapt_branch(
            codec, codec_optimizer, scheduler, tail, adapt, allocation, batches)
        score = float(nested.evaluate(
            branch[0], tail, validation, allocation, image_batch).mean())
        branches.append(branch); adapted.append(score)
    adapted = np.asarray(adapted)
    rho = core.spearman(shared[keep], adapted)
    winner_local = int(np.argmin(adapted)); winner = kept[winner_local]
    incumbent_score = float(adapted[0]); winner_score = float(adapted[winner_local])
    strictly_better = bool(
        winner_local != 0 and winner_score < incumbent_score)
    eligible = bool(rho >= 0.8 and strictly_better)
    realized, policy_iterations = False, 0
    if eligible:
        policy_branch, policy_optimizer_branch = clone_policy(
            policy, policy_optimizer)
        realized, policy_iterations = realize(
            policy_branch, policy_optimizer_branch, winner)
    accepted = bool(eligible and realized)
    if accepted:
        selected = branches[winner_local]
        codec, codec_optimizer, scheduler = selected[:3]
        policy, policy_optimizer = policy_branch, policy_optimizer_branch
    event = {
        "step": int(step), "candidate_count": len(pool),
        "candidate_actual_after_dedup": len(pool), "incumbent_forced_top4": True,
        "kept_indices": keep.tolist(), "incumbent": incumbent.tolist(),
        "winner": winner.tolist(), "shared_scores": shared.tolist(),
        "predicted_scores": predicted.tolist(), "adapted_kept_scores": adapted.tolist(),
        "local_spearman": rho, "minimum_spearman": 0.8,
        "winner_strictly_better": strictly_better,
        "winner_realized_as_map": realized, "policy_iterations": policy_iterations,
        "accepted": accepted, "incumbent_adapted_score": incumbent_score,
        "winner_adapted_score": winner_score,
        "scheduler_count_after": scheduler.count,
        "map_after": policy.build(core.POLICY_TEMPERATURE).map_allocation().tolist(),
        "candidate_costs": costs.tolist()}
    return (codec, codec_optimizer, scheduler, policy, policy_optimizer,
            event)


def parity(prefix, authorization, codec, codec_optimizer, scheduler,
           policy, policy_optimizer, pool, shared):
    expected = authorization["prefix"]
    discrete = (torch.equal(prefix["allocations"], expected["allocations"])
                and torch.equal(prefix["batches"], expected["batches"])
                and torch.equal(prefix["exposure"], expected["exposure"]))
    if not discrete:
        raise SystemExit("authorization replay discrete parity failed")
    errors = {
        "codec": core.max_state_error(codec.state_dict(), authorization["codec"]),
        "codec_optimizer": core.max_state_error(
            codec_optimizer.state_dict(), authorization["codec_optimizer"]),
        "scheduler": core.max_state_error(
            scheduler.state_dict(), authorization["scheduler"]),
        "policy": core.max_state_error(policy.state_dict(), authorization["policy"]),
        "policy_optimizer": core.max_state_error(
            policy_optimizer.state_dict(), authorization["policy_optimizer"])}
    float_error = max(errors.values())
    expected_pool = np.asarray(authorization["result"]["allocations"])
    if not np.array_equal(pool, expected_pool):
        raise SystemExit("authorization replay candidate parity failed")
    expected_shared = np.asarray(authorization["result"]["shared_scores"])
    score_error = float(np.max(np.abs(shared - expected_shared) /
                               np.maximum(np.abs(expected_shared), 1e-12)))
    if float_error > 1e-7 or score_error > 1e-6:
        raise SystemExit("authorization replay floating parity failed")
    return {"discrete_exact": True, "state_errors": errors,
            "maximum_state_relative_error": float_error,
            "shared_score_max_relative_error": score_error,
            "pass": True}


def save(codec, source, policy, codec_optimizer, policy_optimizer, scheduler,
         allocation, result, out):
    nested.save_checkpoint(codec, source, out / "nested_codec.pt",
                           extra={"v31_training": result})
    torch.save(policy.state_dict(), out / "policy.pt")
    torch.save({
        "format": "v31_atomic_training_state_v1",
        "codec": codec.state_dict(),
        "codec_optimizer": codec_optimizer.state_dict(),
        "policy": policy.state_dict(),
        "policy_optimizer": policy_optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "allocation": np.asarray(allocation)}, out / "training_state.pt")
    np.save(out / "allocation.npy", allocation)
    (out / "train.json").write_text(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", default="blk20", choices=tuple(SPECS))
    parser.add_argument("--anchor", required=True, choices=("R64", "R96"))
    parser.add_argument("--nested-codec", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--image-batch", type=int, default=16)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    core.configure_determinism(args.seed)
    config = activate(args.block); anchor = config.ANCHOR_BY_NAME[args.anchor]
    authorization = require_authorization(
        args.authorization, args.block, args.anchor) if not args.smoke else None
    device, out = torch.device(args.device), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is non-empty")
    out.mkdir(parents=True, exist_ok=True)
    codec, source = nested.load_checkpoint(args.nested_codec, device)
    bits = common.mode_bits(codec)
    paths = config.load_split("train_fit")
    train_n = 160 if args.smoke else 5000
    train = engine.ResidentSet(*paths[:2], paths[2][:train_n], device)
    cost_n = 8 if args.smoke else 512
    cost = core.ResidentView(train, 0, cost_n)
    adapt_first = cost_n; adapt_n = 128
    if adapt_first + adapt_n > train.count:
        adapt_first = train.count - adapt_n
    adapt = core.ResidentView(train, adapt_first, adapt_n)
    val_paths = config.load_split("train_val")
    val_n = 8 if args.smoke else 500
    validation = engine.ResidentSet(*val_paths[:2], val_paths[2][:val_n], device)
    tail = tail_mod.build_tail(config.LAYER, device)
    uniform = engine.uniform_allocation(anchor, config.GROUPS)
    policy, policy_optimizer = core.build_policy(
        config.GROUPS, bits, anchor.rate, int(uniform[0]), device)
    codec_optimizer, scheduler = core.build_codec_optimizer(codec)
    initial = float(nested.evaluate(
        codec, tail, validation, uniform, args.image_batch).mean())
    prefix_steps = 6 if args.smoke else 500
    prefix = core.run_authorization_prefix(
        codec, policy, codec_optimizer, scheduler, tail, train,
        config.GROUPS, bits, anchor.rate, args.seed, steps=prefix_steps)
    incumbent = policy.build(core.POLICY_TEMPERATURE).map_allocation().cpu().numpy()
    topk, random_count = ((2, 2) if args.smoke else (8, 8))
    replay_pool, _, _ = common.candidate_pool(
        codec, tail, cost, incumbent, anchor.rate, topk, random_count,
        args.image_batch, args.seed + prefix_steps)
    replay_shared = nested.evaluate(
        codec, tail, validation, replay_pool, args.image_batch).mean(1)
    replay = ({"smoke": True} if args.smoke else parity(
        prefix, authorization, codec, codec_optimizer, scheduler,
        policy, policy_optimizer, replay_pool, replay_shared))
    batch_generator = prefix["batch_generator"]
    allocation_generator = prefix["allocation_generator"]
    fair, events = prefix["fair"], []
    total_steps = 10 if args.smoke else 5000
    outer_every = 5 if args.smoke else 500
    # Event 500 is executed after replay parity; smoke uses its last prefix step.
    event_steps = {prefix_steps} | set(range(
        ((prefix_steps // outer_every) + 1) * outer_every,
        total_steps + 1, outer_every))
    for step in range(prefix_steps, total_steps + 1):
        if step > prefix_steps:
            allocation, fair = core.inner_allocation(
                step, policy, fair, config.GROUPS, bits, anchor.rate, device,
                allocation_generator)
            index = torch.randint(
                train.count, (32,), generator=batch_generator, device=device)
            prefix["exposure"][torch.arange(config.GROUPS), allocation.cpu()] += 1
            core.codec_step(codec, codec_optimizer, scheduler, tail, train,
                            allocation, index)
        if step not in event_steps:
            continue
        (codec, codec_optimizer, scheduler, policy, policy_optimizer,
         event) = outer_event(
            codec, codec_optimizer, scheduler, policy, policy_optimizer,
            tail, cost, adapt, validation, anchor.rate, step, args.seed,
            args.image_batch)
        events.append(event)
        print(f"step={step} rho={event['local_spearman']:.4f} "
              f"accepted={event['accepted']}", flush=True)
    final_allocation = policy.build(
        core.POLICY_TEMPERATURE).map_allocation().cpu().numpy()
    finetune = 2 if args.smoke else 500
    for _ in range(finetune):
        index = torch.randint(
            train.count, (32,), generator=batch_generator, device=device)
        core.codec_step(codec, codec_optimizer, scheduler, tail, train,
                        final_allocation, index)
    final = float(nested.evaluate(
        codec, tail, validation, final_allocation, args.image_batch).mean())
    exact = common.nominal_rate(final_allocation, bits) == anchor.rate
    result = {
        "plan": "v31_warm_state_bilevel", "block": args.block,
        "anchor": args.anchor, "smoke": args.smoke,
        "source_codec": str(Path(args.nested_codec).resolve()),
        "authorization": str(Path(args.authorization).resolve()),
        "deterministic_backend": "xformers_off_math_sdpa",
        "replay_parity": replay, "mode_bits": list(bits),
        "initial_tail_mse_select500": initial,
        "final_tail_mse_select500": final,
        "relative_change": final / initial - 1.0,
        "final_allocation": final_allocation.tolist(),
        "exact_nominal_budget": exact, "events": events,
        "accepted_event_steps": [e["step"] for e in events if e["accepted"]],
        "accepted_event_count": sum(e["accepted"] for e in events),
        "codec_committed_update_count": scheduler.count,
        "group_mode_exposure": prefix["exposure"].tolist(),
        "group_mode_exposure_min": int(prefix["exposure"].min()),
        "finetune_steps": finetune,
        "internal_verdict": ("PASS" if exact and final < initial and
                             any(e["accepted"] for e in events) else "FAIL")}
    save(codec, source["source_codec"], policy, codec_optimizer,
         policy_optimizer, scheduler, final_allocation, result, out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
