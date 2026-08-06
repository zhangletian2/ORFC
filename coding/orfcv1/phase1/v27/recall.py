"""Audit DP Top-K recall after equal fixed-U codebook adaptation."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from codec_v1 import load_codec_v1
from phase1 import engine, tail as tail_mod
from phase1.v12 import qhard
from phase1.v12 import train as joint
from phase1.v21.config import activate
from phase1.v22.allocation_dp import topk_allocations
from phase1.v22.probe import evaluate, local_table, split_resident
from phase1.v22.train import validate


def adapt(codec, tail, resident, allocation, steps, batch, lr, tau, rate_lambda):
    branch = copy.deepcopy(codec)
    for parameter in branch.transform.parameters():
        parameter.requires_grad_(False)
    parameters = []
    for quantizer in branch.pq.quantizers:
        for parameter in quantizer.parameters():
            parameter.requires_grad_(True)
            parameters.append(parameter)
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for step in range(steps):
        first = (step * batch) % max(1, resident.count - batch + 1)
        values = resident.slice(first, min(first + batch, resident.count))
        distortion, _, rate = qhard.distortion_sparse(
            branch, tail, *values, allocation, codeword_temperature=tau,
            return_rate=True)
        loss = (rate * resident.tokens + rate_lambda * distortion).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
    return branch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--allocation")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--pool", type=int, default=64)
    parser.add_argument("--adapt-steps", type=int, default=4)
    parser.add_argument("--adapt-batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--rate-lambda", type=float, default=0.5)
    args = parser.parse_args(argv)

    config = activate("blk20")
    anchor = config.ANCHOR_BY_NAME["R64"]
    device = torch.device(args.device)
    engine.configure_precision(config.ALLOW_TF32)
    codec = load_codec_v1(Path(args.codec), device=device).eval()
    joint.make_trainable(codec)
    bits = joint.actual_mode_bits(codec)
    base = (np.load(args.allocation).astype(np.int64)
            if args.allocation else
            engine.uniform_allocation(anchor, config.GROUPS))
    rows = config.load_split("train_val")[2]
    cal = split_resident(config, rows, 0, 64, device)
    select = split_resident(config, rows, 64, 64, device)
    report = split_resident(config, rows, 128, 128, device)
    tail = tail_mod.build_tail(config.LAYER, device)

    probes, keys = local_table(base, len(bits))
    _, _, objective = evaluate(codec, tail, cal, probes, 16, args.rate_lambda)
    base_value = float(objective[0].mean())
    costs = np.zeros((config.GROUPS, len(bits)), dtype=np.float64)
    for index, (group, mode) in enumerate(keys[1:], 1):
        costs[group, mode] = float(objective[index].mean() - base_value)
    dp = [np.asarray(row, dtype=np.int64) for _, row in topk_allocations(
        costs, bits, anchor.rate, args.topk)]

    neighbors = []
    for donor in range(config.GROUPS):
        if base[donor] == 0:
            continue
        for receiver in range(config.GROUPS):
            if donor == receiver or base[receiver] == len(bits) - 1:
                continue
            row = base.copy(); row[donor] -= 1; row[receiver] += 1
            neighbors.append(row)
    neighbors = np.unique(np.asarray(neighbors, dtype=np.int64), axis=0)
    _, _, raw_objective = evaluate(
        codec, tail, cal, neighbors, 16, args.rate_lambda)
    raw_order = np.argsort(raw_objective.mean(1), kind="stable")
    pool, seen = [], set()
    for row in dp + [neighbors[index] for index in raw_order]:
        key = tuple(row.tolist())
        if key not in seen:
            pool.append(row.copy()); seen.add(key)
        if len(pool) >= args.pool:
            break

    records = []
    dp_set = {tuple(row.tolist()) for row in dp}
    for row in pool:
        branch = adapt(
            codec, tail, cal, row, args.adapt_steps, args.adapt_batch,
            args.lr, args.tau, args.rate_lambda)
        selected = validate(branch, tail, select, row, 16, args.rate_lambda)
        reported = validate(branch, tail, report, row, 16, args.rate_lambda)
        records.append({"allocation": row.tolist(), "in_dp_topk":
                        tuple(row.tolist()) in dp_set, "select": selected,
                        "report": reported})
        del branch
    winner = int(np.argmin([item["select"]["objective"] for item in records]))
    report_winner = int(np.argmin([
        item["report"]["objective"] for item in records]))
    result = {
        "plan": "v27_dp_recall_after_equal_fixed_u_adaptation",
        "neighbors": int(len(neighbors)), "pool": int(len(pool)),
        "dp_topk": int(args.topk), "adapt_steps": int(args.adapt_steps),
        "adapt_batch": int(args.adapt_batch), "records": records,
        "select_winner": winner, "select_recall_at_k": bool(
            records[winner]["in_dp_topk"]),
        "report_winner": report_winner, "report_recall_at_k": bool(
            records[report_winner]["in_dp_topk"]),
        "select_report_same_winner": bool(winner == report_winner)}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in (
        "neighbors", "pool", "dp_topk", "select_winner",
        "select_recall_at_k", "report_winner", "report_recall_at_k",
        "select_report_same_winner")}, indent=2))


if __name__ == "__main__":
    main()
