"""Fixed-identity, exact-budget DP beam training with full Tail MSE."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from codec_v1 import load_codec_v1, save_codec_v1
from .. import engine, frozen, tail as tail_mod
from ..v12 import qhard
from ..v12 import train as joint
from ..v21.config import SPECS, activate
from ..v28.train import candidates, hard_distortion, val_record


def train_codebooks(parent, state, tail, train, allocation, permutations,
                    batch, lr, taus):
    branch = copy.deepcopy(parent)
    for parameter in branch.transform.parameters():
        parameter.requires_grad_(False)
    parameters = [parameter for quantizer in branch.pq.quantizers
                  for parameter in quantizer.parameters()]
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=lr)
    if state is not None:
        optimizer.load_state_dict(copy.deepcopy(state))
    rotation_before = branch.transform.get_rotation().detach().clone()
    books_before = [q.codebooks.detach().clone()
                    for q in branch.pq.quantizers]
    losses = []
    for permutation, tau in zip(permutations, taus):
        for first in range(0, len(permutation), batch):
            index = permutation[first:first + batch]
            if len(index) != batch:
                continue
            distortion, _ = qhard.distortion_sparse(
                branch, tail,
                train.y.index_select(0, index),
                train.mu.index_select(0, index),
                train.std.index_select(0, index),
                train.teacher.index_select(0, index),
                allocation, codeword_temperature=tau)
            loss = distortion.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    selected = torch.as_tensor(allocation, device=train.device)
    mask = torch.zeros(
        len(allocation), len(branch.pq.quantizers), dtype=torch.bool,
        device=train.device)
    mask[torch.arange(len(allocation), device=train.device), selected] = True
    drift = torch.stack([
        (q.codebooks.detach() - old).flatten(1).norm(dim=1)
        for q, old in zip(branch.pq.quantizers, books_before)], dim=1)
    u_drift = float((branch.transform.get_rotation().detach()
                     - rotation_before).abs().max())
    return {
        "codec": branch, "optimizer_state": optimizer.state_dict(),
        "train_loss": float(np.mean(losses)), "u_drift": u_drift,
        "selected_book_drift_min": float(drift[mask].min()),
        "unselected_book_drift_max": float(drift[~mask].max())}


def save_state(state, out, cycle):
    save_codec_v1(state["codec"], out / "codec.pt")
    np.save(out / "allocation.npy", state["allocation"])
    torch.save(state["optimizer_state"], out / "optimizer.pt")
    return {"cycle": cycle, "hard_tail_mse": state["score"],
            "allocation": state["allocation"].tolist()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", default="blk20", choices=tuple(SPECS))
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--source-codec", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--branch-epochs", type=int, default=5)
    parser.add_argument("--dp-topk", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tau-start", type=float, default=0.5)
    parser.add_argument("--tau-end", type=float, default=0.005)
    parser.add_argument("--image-batch", type=int, default=20)
    parser.add_argument("--alloc-chunk", type=int, default=2)
    parser.add_argument("--train-images", type=int, default=5000)
    parser.add_argument("--cost-images", type=int, default=512)
    parser.add_argument("--val-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.beam_width < 1 or args.dp_topk < 1:
        raise SystemExit("beam width and DP top-k must be positive")
    started = time.time()
    config = activate(args.block)
    anchor = config.ANCHOR_BY_NAME[args.anchor]
    device, out = torch.device(args.device), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is non-empty")
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    val_rows = np.asarray(
        manifest["validation_select"]["indices"][:args.val_images])
    if len(val_rows) != args.val_images or len(np.unique(val_rows)) != len(val_rows):
        raise SystemExit("invalid validation manifest")
    engine.configure_precision(config.ALLOW_TF32)
    source = load_codec_v1(Path(args.source_codec), device=device)
    identity = torch.eye(
        config.GROUPS * config.DIM, device=device,
        dtype=source.transform.get_rotation().dtype)
    if float((source.transform.get_rotation() - identity).abs().max()) != 0.0:
        raise SystemExit("source codec is not exact identity")
    if any(getattr(q, "use_rate", False) for q in source.pq.quantizers):
        raise SystemExit("V29 uses Tail MSE without an entropy prior")
    bits = tuple(int(round(math.log2(q.codebooks.shape[1])))
                 for q in source.pq.quantizers)
    allocation = engine.uniform_allocation(anchor, config.GROUPS)
    train_paths = config.load_split("train_fit")
    train_rows = np.asarray(train_paths[2][:args.train_images])
    if args.cost_images < 1 or args.cost_images > len(train_rows):
        raise SystemExit("cost-images must lie within the training split")
    train = engine.ResidentSet(
        train_paths[0], train_paths[1], train_rows, device)
    cost_order = np.random.default_rng(args.seed).permutation(len(train_rows))
    cost_rows = train_rows[cost_order[:args.cost_images]]
    cost = engine.ResidentSet(
        train_paths[0], train_paths[1], cost_rows, device)
    val_paths = config.load_split("train_val")
    val = engine.ResidentSet(val_paths[0], val_paths[1], val_rows, device)
    tail = tail_mod.build_tail(config.LAYER, device)
    initial = val_record(source, tail, val, allocation, args.image_batch)
    beam = [{"codec": source, "allocation": allocation,
             "optimizer_state": None,
             "score": initial["hard_tail_mse"], "origin": "initial"}]
    best_payload = save_state(beam[0], out, 0)
    events, generator = [], torch.Generator(device=device).manual_seed(args.seed)
    cycles = 1 if args.smoke else args.cycles
    total_epochs = cycles * args.branch_epochs
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for cycle in range(cycles):
        permutations = [torch.randperm(
            train.count, generator=generator, device=device)
            for _ in range(args.branch_epochs)]
        taus = [joint.codeword_tau(
            cycle * args.branch_epochs + epoch + 1, total_epochs,
            args.tau_start, args.tau_end)
            for epoch in range(args.branch_epochs)]
        pool, parent_records = [], []
        for parent_index, parent in enumerate(beam):
            # Keep the unmodified parent: retained validation best is monotone.
            pool.append(parent)
            rows, table = candidates(
                parent["codec"], tail, cost, parent["allocation"], bits,
                anchor.rate, args.dp_topk, args.image_batch,
                args.alloc_chunk, cycle == 0)
            records = []
            for row in rows:
                trained = train_codebooks(
                    parent["codec"], parent["optimizer_state"], tail, train,
                    row, permutations, args.batch, args.lr, taus)
                measured = val_record(
                    trained["codec"], tail, val, row, args.image_batch)
                state = {**trained, "allocation": row.copy(),
                         "score": measured["hard_tail_mse"],
                         "origin": f"cycle{cycle + 1}/parent{parent_index}"}
                pool.append(state)
                records.append({
                    **measured, "allocation": row.tolist(),
                    "train_loss": trained["train_loss"],
                    "u_drift": trained["u_drift"],
                    "selected_book_drift_min":
                        trained["selected_book_drift_min"],
                    "unselected_book_drift_max":
                        trained["unselected_book_drift_max"]})
            parent_records.append({"parent": parent_index, "table": table,
                                   "branches": records})
        pool.sort(key=lambda state: (
            state["score"], tuple(state["allocation"].tolist()),
            state["origin"]))
        beam, seen = [], set()
        for state in pool:
            key = tuple(state["allocation"].tolist())
            if key in seen:
                continue
            beam.append(state)
            seen.add(key)
            if len(beam) == args.beam_width:
                break
        if beam[0]["score"] < best_payload["hard_tail_mse"]:
            best_payload = save_state(beam[0], out, cycle + 1)
        events.append({"cycle": cycle + 1, "parents": parent_records,
                       "retained": [{"score": state["score"],
                                     "allocation": state["allocation"].tolist(),
                                     "origin": state["origin"]}
                                    for state in beam]})
        print(f"cycle={cycle + 1}/{cycles} beam=" + ",".join(
            f"{state['score']:.3f}:{state['allocation'].tolist()}"
            for state in beam), flush=True)
    codec = load_codec_v1(out / "codec.pt", device=device).eval()
    allocation = np.load(out / "allocation.npy")
    final = val_record(codec, tail, val, allocation, args.image_batch)
    parity = qhard.selfcheck(codec, tail, val, allocation, args.image_batch)
    u_final_gap = float((codec.transform.get_rotation() - identity).abs().max())
    if engine.nominal_rate(allocation, anchor) != anchor.rate:
        raise SystemExit("INVALID_EXPERIMENT: nominal rate")
    if parity["rel_gap"] > parity["tolerance"]:
        raise SystemExit("INVALID_EXPERIMENT: hard parity")
    if u_final_gap != 0.0:
        raise SystemExit("INVALID_EXPERIMENT: identity transform drift")
    payload = {
        "plan": "v29_fixed_identity_exact_budget_dp_beam",
        "block": args.block, "anchor": args.anchor,
        "nominal_rate": anchor.rate, "manifest": str(Path(args.manifest).resolve()),
        "data": {"train": train.count, "cost_table": cost.count,
                 "cost_rows": cost_rows.tolist(),
                 "validation_select": val.count},
        "cycles": cycles, "branch_epochs": args.branch_epochs,
        "total_path_epochs": total_epochs, "dp_topk": args.dp_topk,
        "beam_width": args.beam_width, "batch": args.batch, "lr": args.lr,
        "tau_start": args.tau_start, "tau_end": args.tau_end,
        "initial": initial, "best": best_payload, "final": final,
        "allocation": allocation.tolist(), "events": events,
        "hard_parity": parity, "identity_drift_max": u_final_gap,
        "peak_memory_gb": (torch.cuda.max_memory_allocated(device) / 2**30
                           if device.type == "cuda" else 0.0),
        "seconds": time.time() - started}
    (out / "train.json").write_text(json.dumps(payload, indent=2))
    (out / "training_complete.json").write_text(json.dumps({
        "best": best_payload, "allocation": allocation.tolist(),
        "hard_parity": parity, "identity_drift_max": u_final_gap}, indent=2))
    print(json.dumps({key: payload[key] for key in (
        "initial", "best", "final", "allocation", "identity_drift_max",
        "peak_memory_gb", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
