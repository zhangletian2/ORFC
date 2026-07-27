#!/usr/bin/env python3
"""Offline P1 coverage/generalisation checks and one-batch U gradcheck."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _moments(values):
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "p05": float(np.quantile(x, 0.05)),
        "p50": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
    }


def _best_pair(mask, score):
    ids = np.flatnonzero(mask)
    return ids[int(np.argmax(score[ids]))]


def analyse_one(path, args, seed):
    z = np.load(path, allow_pickle=False)
    alloc, rates = z["allocations"], z["rates"]
    phi, distortion = z["phi"], z["distortion_per_image"]
    rng = np.random.default_rng(seed)
    order = rng.permutation(distortion.shape[1])
    cut = min(args.select_images, len(order) - 1)
    select, held = order[:cut], order[cut:]
    e_select = distortion[:, select].mean(1) - phi
    e_held = distortion[:, held].mean(1) - phi

    left, right = np.triu_indices(len(alloc), 1)
    steps = np.abs(rates[left] - rates[right]).sum(1) / 2.0
    signed_select = e_select[left] - e_select[right]
    signed_held = e_held[left] - e_held[right]
    select_score, held_score = abs(signed_select), abs(signed_held)
    edge, multi = np.isclose(steps, 1.0), steps > 1.0 + 1e-9
    oracle = float(held_score.max())

    random_cov, hybrid_cov = [], []
    edge_ids, multi_ids = np.flatnonzero(edge), np.flatnonzero(multi)
    for _ in range(args.trials):
        m = min(args.pairs, len(multi_ids))
        random_ids = rng.choice(multi_ids, m, replace=False)
        e = min(args.pairs // 2, len(edge_ids))
        h_edge = rng.choice(edge_ids, e, replace=False)
        h_multi = rng.choice(multi_ids, min(args.pairs - e, len(multi_ids)),
                             replace=False)
        random_cov.append(float(held_score[random_ids].max() / oracle))
        hybrid_cov.append(float(held_score[np.r_[h_edge, h_multi]].max()
                                / oracle))

    top = np.argsort(select_score)[-min(args.topk, len(select_score)):]
    split_rows = {"mine": [], "heldout_allocations": []}
    n_mine = max(2, int(len(alloc) * (1.0 - args.allocation_holdout)))
    for _ in range(args.trials):
        nodes = rng.permutation(len(alloc))
        mine_nodes, held_nodes = nodes[:n_mine], nodes[n_mine:]
        mine_mask = np.isin(left, mine_nodes) & np.isin(right, mine_nodes)
        held_mask = np.isin(left, held_nodes) & np.isin(right, held_nodes)
        for name, mask in (("mine", mine_mask),
                           ("heldout_allocations", held_mask)):
            if not mask.any():
                continue
            chosen = _best_pair(mask, select_score)
            oracle_local = held_score[mask].max()
            split_rows[name].append({
                "retention": float(held_score[chosen]
                                   / max(select_score[chosen], 1e-12)),
                "coverage": float(held_score[chosen]
                                  / max(oracle_local, 1e-12)),
                "sign": float(np.sign(signed_select[chosen])
                              == np.sign(signed_held[chosen])),
            })

    result = {
        "n_allocations": int(len(alloc)),
        "n_images": {"selection": int(len(select)), "heldout": int(len(held))},
        "n_pairs": {"all": int(len(left)), "edge": int(edge.sum()),
                    "multi": int(multi.sum())},
        "heldout_omega": oracle,
        "coverage": {
            "all_edges": float(held_score[edge].max() / oracle),
            "random_multi": _moments(random_cov),
            "hybrid": _moments(hybrid_cov),
        },
        "hard_pair_image_generalisation": {
            "all_pair_score_pearson": float(
                np.corrcoef(select_score, held_score)[0, 1]),
            "topk_heldout_coverage": float(held_score[top].max() / oracle),
            "topk_sign_agreement": float(
                np.mean(np.sign(signed_select[top])
                        == np.sign(signed_held[top]))),
        },
        "allocation_split_generalisation": {},
    }
    for name, rows in split_rows.items():
        result["allocation_split_generalisation"][name] = {
            key: _moments([row[key] for row in rows])
            for key in ("retention", "coverage", "sign")
        }
    return result


def command_analyse(args):
    root, output = Path(args.run_dir), Path(args.output)
    report = {"run_dir": str(root), "results": []}
    for arm_i, arm in enumerate(args.arms.split(",")):
        for budget in map(int, args.budgets.split(",")):
            path = root / arm / f"measurement_R{budget}.npz"
            row = analyse_one(path, args, args.seed + 100 * arm_i + budget)
            row.update({"arm": arm, "budget": budget})
            report["results"].append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


def _soft_reconstruct(codec, y, modes, tau, rotation):
    z = y.reshape(-1, y.shape[-1]) @ rotation
    zg = z.reshape(-1, codec.pq.G, codec.pq.d).permute(1, 0, 2)
    groups = []
    for g, mode in enumerate(modes.tolist()):
        q, c = codec.pq.quantizers[mode], codec.pq.quantizers[mode].codebooks[g]
        cost = (zg[g, :, None] - c[None]).square().sum(-1)
        if q.use_rate:
            log2_p = -torch.log_softmax(q.log_prior[g], 0) / np.log(2.0)
            cost = cost + log2_p[None] / q.lmbda
        groups.append(torch.softmax(-cost / tau, -1) @ c)
    z_hat = torch.stack(groups, 1).reshape_as(z)
    return (z_hat @ rotation.t()).reshape_as(y)


def command_gradcheck(args):
    from codec_v1 import load_codec_v1
    from opq import batch_inv_normalize_gpu, batch_normalize_gpu
    from p1_fixed_rate import build_tail

    device = torch.device(args.device)
    codec = load_codec_v1(args.codec, device).train()
    tail = build_tail(args.layer, device)
    measured = np.load(args.measurement, allow_pickle=False)
    n_select = min(args.select_images,
                   measured["distortion_per_image"].shape[1])
    e = measured["distortion_per_image"][:, :n_select].mean(1) - measured["phi"]
    ia, ib = int(e.argmax()), int(e.argmin())
    modes_a, modes_b = [
        torch.as_tensor(measured["allocations"][i], device=device)
        for i in (ia, ib)
    ]
    phi_delta = float(measured["phi"][ia] - measured["phi"][ib])

    features = np.load(args.features, mmap_mode="r")
    teachers = np.load(args.teachers, mmap_mode="r")
    sl = slice(args.image_offset, args.image_offset + args.images)
    x = torch.from_numpy(np.array(features[sl], copy=True)).float().to(device)
    teacher = torch.from_numpy(
        np.array(teachers[sl], copy=True)).float().to(device)
    y, mu, std = batch_normalize_gpu(x, mode=args.norm_mode)

    def objective():
        rotation = codec.transform.get_rotation()
        losses = []
        for modes in (modes_a, modes_b):
            y_hat = _soft_reconstruct(
                codec, y, modes, args.temperature, rotation)
            output = tail(batch_inv_normalize_gpu(y_hat, mu, std))
            losses.append((output - teacher).square().reshape(
                len(x), -1).sum(1).mean())
        delta = losses[0] - losses[1] - phi_delta
        return torch.sqrt(delta.square() + args.smooth_abs ** 2), delta

    param = codec.transform.triu_params
    codec.zero_grad(set_to_none=True)
    loss, delta = objective()
    loss.backward()
    grad = param.grad.detach().clone()
    if not torch.isfinite(grad).all() or grad.norm() == 0:
        raise RuntimeError("non-finite or zero U gradient")
    original = param.detach().clone()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    random = torch.randn(
        param.shape, generator=generator, device=device, dtype=param.dtype)
    mixed = grad / grad.norm() + random / random.norm()
    coordinate = torch.zeros_like(grad)
    coordinate[grad.abs().argmax()] = 1.0
    directions = {
        "largest_coordinate": coordinate,
        "gradient": grad / grad.norm(),
        "mixed_random": mixed / mixed.norm(),
    }
    rows = {}
    try:
        for name, direction in directions.items():
            ad = float(torch.dot(grad, direction))
            scans = []
            for eps in map(float, args.eps.split(",")):
                with torch.no_grad():
                    param.copy_(original + eps * direction)
                    plus = float(objective()[0])
                    param.copy_(original - eps * direction)
                    minus = float(objective()[0])
                fd = (plus - minus) / (2.0 * eps)
                scans.append({
                    "eps": eps, "autograd": ad, "finite_difference": fd,
                    "relative_error": abs(fd - ad)
                    / max(abs(fd), abs(ad), 1e-12),
                    "sign_match": bool(np.sign(fd) == np.sign(ad)),
                })
            rows[name] = scans
    finally:
        with torch.no_grad():
            param.copy_(original)
    rotation = codec.transform.get_rotation().detach()
    orth = float((rotation.t() @ rotation
                  - torch.eye(rotation.shape[0], device=device)).norm())
    best = min(rows["largest_coordinate"],
               key=lambda row: row["relative_error"])
    output = {
        "pair_indices": [ia, ib], "soft_loss": float(loss.detach()),
        "residual_delta": float(delta.detach()), "grad_norm": float(grad.norm()),
        "orthogonality_frobenius": orth,
        "largest_coordinate_pass": bool(
            best["sign_match"] and best["relative_error"] < 0.05),
        "directions": rows,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(path)


def parser():
    main = argparse.ArgumentParser()
    sub = main.add_subparsers(dest="command", required=True)
    p = sub.add_parser("analyse")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--arms", default="identity,opq,orfc,response")
    p.add_argument("--budgets", default="128,192")
    p.add_argument("--select-images", type=int, default=150)
    p.add_argument("--pairs", type=int, default=512)
    p.add_argument("--topk", type=int, default=16)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--allocation-holdout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("gradcheck")
    p.add_argument("--codec", required=True)
    p.add_argument("--measurement", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--teachers", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--norm-mode", default="per_image")
    p.add_argument("--select-images", type=int, default=150)
    p.add_argument("--image-offset", type=int, default=300)
    p.add_argument("--images", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.01)
    p.add_argument("--smooth-abs", type=float, default=1e-6)
    p.add_argument("--eps", default="0.0000001,0.0000003,0.000001,"
                                     "0.000003,0.00001")
    p.add_argument("--seed", type=int, default=42)
    return main


if __name__ == "__main__":
    args = parser().parse_args()
    globals()[f"command_{args.command}"](args)
