"""Gate 3: single-path inner training and slow exact-budget outer updates.

The outer step uses successive halving: shared-weight screening followed by
independent short adaptation.  It updates only the exact-budget allocation
policy.  The final MAP allocation is hardened before a codec-only finetune.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import engine, tail as tail_mod
from ..v12.allocation_policy import FixedBudgetAllocationPolicy
from ..v12.strict_fair import strict_fair_slate
from ..v21.config import SPECS, activate
from . import common, nested


def require_gate(path, name):
    payload = json.loads(Path(path).read_text())
    if payload.get("verdict") != "PASS":
        raise SystemExit(f"{name} did not pass: {path}")
    return payload


def train_step(codec, optimizer, tail, resident, allocation, batch,
               temperature, generator):
    index = torch.randint(resident.count, (int(batch),),
                          generator=generator, device=resident.device)
    value, _ = nested.distortion(
        codec, tail,
        resident.y.index_select(0, index),
        resident.mu.index_select(0, index),
        resident.std.index_select(0, index),
        resident.teacher.index_select(0, index), allocation,
        temperature=temperature)
    loss = value.mean(); optimizer.zero_grad(set_to_none=True)
    loss.backward(); torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", default="blk20", choices=tuple(SPECS))
    parser.add_argument("--anchor", default="R64")
    parser.add_argument("--nested-codec", required=True)
    parser.add_argument("--capacity-gate", required=True)
    parser.add_argument("--ranking-gate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--outer-every", type=int, default=500)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--policy-lr", type=float, default=1e-2)
    parser.add_argument("--outer-policy-steps", type=int, default=250)
    parser.add_argument("--outer-topk", type=int, default=8)
    parser.add_argument("--outer-random", type=int, default=8)
    parser.add_argument("--halving-keep", type=int, default=4)
    parser.add_argument("--outer-adapt-steps", type=int, default=8)
    parser.add_argument("--outer-adapt-batch", type=int, default=16)
    parser.add_argument("--min-spearman", type=float, default=0.8)
    parser.add_argument("--finetune-steps", type=int, default=500)
    parser.add_argument("--train-images", type=int, default=5000)
    parser.add_argument("--outer-cal-images", type=int, default=512)
    parser.add_argument("--val-images", type=int, default=500)
    parser.add_argument("--image-batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    require_gate(args.capacity_gate, "capacity gate")
    require_gate(args.ranking_gate, "ranking gate")
    config = activate(args.block); anchor = config.ANCHOR_BY_NAME[args.anchor]
    device, out = torch.device(args.device), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is non-empty")
    out.mkdir(parents=True, exist_ok=True)
    codec, checkpoint = nested.load_checkpoint(args.nested_codec, device)
    bits = common.mode_bits(codec)
    if bits != tuple(anchor.mode_bits):
        raise SystemExit("nested menu does not match the anchor")
    train_n = min(args.train_images, 16) if args.smoke else args.train_images
    cal_n = min(args.outer_cal_images, 4) if args.smoke \
        else args.outer_cal_images
    val_n = min(args.val_images, 4) if args.smoke else args.val_images
    steps = min(args.steps, 3) if args.smoke else args.steps
    warmup = min(args.warmup_steps, 1) if args.smoke else args.warmup_steps
    outer_every = 2 if args.smoke else args.outer_every
    train_paths = config.load_split("train_fit")
    val_paths = config.load_split("train_val")
    train = engine.ResidentSet(
        *train_paths[:2], train_paths[2][:train_n], device)
    cal = engine.ResidentSet(
        *train_paths[:2], train_paths[2][:cal_n], device)
    validation = engine.ResidentSet(
        *val_paths[:2], val_paths[2][:val_n], device)
    tail = tail_mod.build_tail(config.LAYER, device)
    base = engine.uniform_allocation(anchor, config.GROUPS)
    logits = torch.zeros(config.GROUPS, len(bits))
    logits[torch.arange(config.GROUPS), torch.as_tensor(base)] = 2.0
    policy = FixedBudgetAllocationPolicy(
        config.GROUPS, bits, anchor.rate, init_logits=logits).to(device)
    codec_optimizer = torch.optim.Adam(codec.parameters(), lr=args.lr)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial = float(nested.evaluate(
        codec, tail, validation, base, args.image_batch).mean())
    events, losses, exposures = [], [], torch.zeros(
        config.GROUPS, len(bits), dtype=torch.long, device=device)
    fair = None
    for step in range(1, steps + 1):
        if step <= warmup:
            if fair is None or (step - 1) % len(bits) == 0:
                fair = strict_fair_slate(
                    config.GROUPS, bits, anchor.rate, device, generator)
            allocation = fair[(step - 1) % len(bits)]
        else:
            allocation = policy.build(args.policy_temperature).sample(
                1, generator=generator)[0]
        exposures[torch.arange(config.GROUPS, device=device), allocation] += 1
        losses.append(train_step(
            codec, codec_optimizer, tail, train, allocation, args.batch,
            args.temperature, generator))
        if step % outer_every or step <= warmup:
            continue
        distribution = policy.build(args.policy_temperature)
        current = distribution.map_allocation().cpu().numpy()
        pool, _, _ = common.candidate_pool(
            codec, tail, cal, current, anchor.rate,
            min(args.outer_topk, 2) if args.smoke else args.outer_topk,
            min(args.outer_random, 1) if args.smoke else args.outer_random,
            args.image_batch, args.seed + step)
        shared = nested.evaluate(
            codec, tail, validation, pool, args.image_batch).mean(1)
        keep_count = min(args.halving_keep, len(pool))
        # Candidate zero is the incumbent.  It may never be removed by the
        # shared-weight screening stage.
        ranked_non_incumbent = [index for index in np.argsort(
            shared, kind="stable").tolist() if index != 0]
        keep = np.asarray(
            [0] + ranked_non_incumbent[:max(0, keep_count - 1)],
            dtype=np.int64)
        kept = pool[keep]
        adapted = common.adapted_scores(
            codec, tail, cal, validation, kept,
            min(args.outer_adapt_steps, 1) if args.smoke
            else args.outer_adapt_steps,
            args.outer_adapt_batch, args.lr, args.temperature,
            args.image_batch, args.seed + step, train_u=True)
        rho = common.spearman(shared[keep], adapted)
        winner_index = int(np.argmin(adapted))
        winner = kept[winner_index]
        incumbent_score = float(adapted[0])
        winner_score = float(adapted[winner_index])
        improved = bool(winner_index != 0 and winner_score < incumbent_score)
        eligible = bool(rho >= args.min_spearman and improved)
        before = current.tolist()
        old_logits = policy.logits.detach().clone()
        if eligible:
            target = torch.as_tensor(winner, device=device)
            for _ in range(args.outer_policy_steps):
                objective = -policy.build(
                    args.policy_temperature).log_prob(target)
                policy_optimizer.zero_grad(set_to_none=True)
                objective.backward(); policy_optimizer.step()
                with torch.no_grad():
                    policy.logits.sub_(policy.logits.mean(1, keepdim=True))
                if torch.equal(policy.build(
                        args.policy_temperature).map_allocation(), target):
                    break
        after = policy.build(args.policy_temperature).map_allocation().tolist()
        realized = bool(after == winner.tolist())
        accepted = bool(eligible and realized)
        if eligible and not realized:
            with torch.no_grad():
                policy.logits.copy_(old_logits)
            after = policy.build(
                args.policy_temperature).map_allocation().tolist()
        events.append({
            "step": step, "candidate_count": len(pool),
            "halving_keep": keep_count, "spearman": rho,
            "incumbent_forced": True, "incumbent_adapted_score":
                incumbent_score, "winner_adapted_score": winner_score,
            "winner_strictly_better": improved,
            "eligible_before_policy_update": eligible,
            "winner_realized_as_map": realized,
            "accepted": accepted, "map_before": before,
            "winner": winner.tolist(), "map_after": after,
            "shared_scores": shared.tolist(),
            "adapted_kept_scores": adapted.tolist()})
        print(f"step={step} rho={rho:.3f} accepted={accepted} "
              f"map_changed={before != after}", flush=True)
    final_allocation = policy.build(
        args.policy_temperature).map_allocation().cpu().numpy()
    finetune_steps = min(args.finetune_steps, 1) if args.smoke \
        else args.finetune_steps
    common.train_fixed(
        codec, tail, train, final_allocation, finetune_steps, args.batch,
        args.lr, args.temperature, args.seed + 999, train_u=True)
    final = float(nested.evaluate(
        codec, tail, validation, final_allocation, args.image_batch).mean())
    exact = common.nominal_rate(final_allocation, bits) == anchor.rate
    result = {
        "plan": "v30_two_timescale_bilevel_gate", "block": args.block,
        "anchor": args.anchor, "mode_bits": list(bits),
        "steps": steps, "warmup_steps": warmup,
        "outer_every": outer_every, "single_path_inner": True,
        "single_path_pq": True,
        "validation_select_images": validation.count,
        "initial_tail_mse": initial, "final_tail_mse": final,
        "relative_change": final / initial - 1.0,
        "final_allocation": final_allocation.tolist(),
        "exact_nominal_budget": exact, "events": events,
        "group_mode_exposure_min": int(exposures.min()),
        "group_mode_exposure": exposures.cpu().tolist(),
        "mean_inner_loss": float(np.mean(losses)),
        "finetune_steps": finetune_steps,
        "verdict": ("PASS" if exact and final < initial and
                    any(event["accepted"] for event in events) else "FAIL")}
    nested.save_checkpoint(
        codec, checkpoint["source_codec"], out / "nested_codec.pt",
        extra={"bilevel_gate": result})
    torch.save(policy.state_dict(), out / "policy.pt")
    np.save(out / "allocation.npy", final_allocation)
    (out / "train.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
