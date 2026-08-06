"""Gate 2: shared-weight candidate ranking versus independent adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import engine, tail as tail_mod
from ..v21.config import SPECS, activate
from . import common, nested


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", default="blk20", choices=tuple(SPECS))
    parser.add_argument("--anchor", default="R64")
    parser.add_argument("--nested-codec", required=True)
    parser.add_argument("--allocation")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cal-images", type=int, default=512)
    parser.add_argument("--adapt-images", type=int, default=128)
    parser.add_argument("--val-images", type=int, default=500)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--random", type=int, default=8)
    parser.add_argument("--adapt-steps", type=int, default=8)
    parser.add_argument("--adapt-batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--image-batch", type=int, default=16)
    parser.add_argument("--min-spearman", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    config = activate(args.block); anchor = config.ANCHOR_BY_NAME[args.anchor]
    device, out = torch.device(args.device), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    codec, payload = nested.load_checkpoint(args.nested_codec, device)
    capacity_gate = payload.get("extra", {}).get("capacity_gate", {})
    if capacity_gate.get("verdict") != "PASS":
        raise SystemExit("nested checkpoint is not backed by a PASS capacity gate")
    coverage_export = payload.get("extra", {}).get("coverage_export", {})
    if not coverage_export or coverage_export.get("smoke", True):
        raise SystemExit("ranking gate requires a formal coverage export")
    bits = common.mode_bits(codec)
    base = (np.load(args.allocation).astype(np.int64)
            if args.allocation else engine.uniform_allocation(
                anchor, config.GROUPS))
    if common.nominal_rate(base, bits) != anchor.rate:
        raise SystemExit("base allocation violates the exact budget")
    scale = 8 if args.smoke else 1
    counts = [max(4, value // scale) for value in (
        args.cal_images, args.adapt_images, args.val_images)]
    train_paths = config.load_split("train_fit")
    val_paths = config.load_split("train_val")
    cal = engine.ResidentSet(
        *train_paths[:2], train_paths[2][:counts[0]], device)
    adapt = engine.ResidentSet(
        *train_paths[:2],
        train_paths[2][counts[0]:counts[0] + counts[1]], device)
    validation = engine.ResidentSet(
        *val_paths[:2], val_paths[2][:counts[2]], device)
    tail = tail_mod.build_tail(config.LAYER, device)
    pool, predicted, _ = common.candidate_pool(
        codec, tail, cal, base, anchor.rate,
        min(args.topk, 2) if args.smoke else args.topk,
        min(args.random, 2) if args.smoke else args.random,
        args.image_batch, args.seed)
    # Before/after rankings use the identical fixed validation-select set.
    shared = nested.evaluate(
        codec, tail, validation, pool, args.image_batch).mean(1)
    adapted = common.adapted_scores(
        codec, tail, adapt, validation, pool,
        min(args.adapt_steps, 1) if args.smoke else args.adapt_steps,
        args.adapt_batch, args.lr, args.temperature, args.image_batch,
        args.seed, train_u=True)
    rho = common.spearman(shared, adapted)
    k = min(args.topk, len(pool))
    shared_top = set(np.argsort(shared)[:k].tolist())
    adapted_top = set(np.argsort(adapted)[:k].tolist())
    best_recalled = int(np.argmin(adapted)) in shared_top
    recall = len(shared_top & adapted_top) / k
    result = {
        "plan": "v30_shared_weight_ranking_gate", "block": args.block,
        "anchor": args.anchor, "mode_bits": list(bits),
        "capacity_gate_verdict": capacity_gate["verdict"],
        "candidate_count": len(pool), "topk": k,
        "data": {"cost_table_train": cal.count,
                 "adapt_train": adapt.count,
                 "validation_select": validation.count,
                 "same_validation_before_after": True,
                 "identical_adaptation_order": True},
        "shared_scores": shared.tolist(), "adapted_scores": adapted.tolist(),
        "allocations": pool.tolist(), "spearman": rho,
        "topk_recall": recall, "adapted_best_in_shared_topk": best_recalled,
        "min_spearman": args.min_spearman,
        "verdict": ("PASS" if rho >= args.min_spearman and best_recalled
                    else "FAIL")}
    (out / "ranking_gate.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
