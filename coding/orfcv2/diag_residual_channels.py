#!/usr/bin/env python
"""Diagnose shareR + K2 residual: codebook collapse and leave-one-group-out.

Two complementary probes, both in the shared ORFC coordinates (G=32 groups):

  collapse
      Per-group K=2 codebook geometry + hard-assignment usage + residual energy
      vs reconstruction MSE.  Does not load the ViT tail.

  loco  (leave-one-channel-out)
      Other groups keep the *original* residual; only group g is replaced by
      its residual codebook.  Reports Acc / CE / teacher-KL so a single
      harmful codebook can show up even when Acc is saturated.
      Also reports the complementary drop-g (group g original, others
      quantized) to see whether removing one group recovers Acc.

Usage:
    python diag_residual_channels.py --mode collapse
    python diag_residual_channels.py --mode loco --layer blk15 --gpu 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_fn

ORFCV2_ROOT = os.path.dirname(os.path.abspath(__file__))
CODING_ROOT = os.path.dirname(ORFCV2_ROOT)
ORFC_ROOT = os.path.join(CODING_ROOT, "orfc")
PROJECT_ROOT = os.path.dirname(CODING_ROOT)
if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)

from opq import batch_inv_normalize_gpu, batch_normalize_gpu
from run_multilayer_calibrator import load_gt, preload_features, set_seed
from backbone.wrapper import Dinov2Wrapper
from residual_pq import ClsTaskTail, load_residual_codec


SKIP_STEM_SUBSTR = ("K4_emb32_bt1024_ws_tau0.5_lr0.0003_ep300",)


def _base_k(ckpt_path: str) -> int:
    return int(os.path.basename(ckpt_path).split("_")[1][1:])


def _is_pure_kd(cfg: dict) -> bool:
    return float(cfg.get("ce_weight", 1.0)) == 0.0 and float(
        cfg.get("lmbda_kd", 0.0)) > 0.0


def discover_shareR_jsons(result_dir: Path, layer=None):
    out = []
    for fp in sorted(result_dir.glob("*shareR*.json")):
        if any(s in fp.stem for s in SKIP_STEM_SUBSTR):
            continue
        d = json.loads(fp.read_text())
        cfg = d["config"]
        if not cfg.get("share_base_transform"):
            continue
        if not _is_pure_kd(cfg):
            continue
        if layer is not None and cfg.get("layer") != layer:
            continue
        if not d.get("res_ckpt") or not os.path.isfile(d["res_ckpt"]):
            continue
        out.append((fp, d))
    return out


def _kd_kl(student, teacher, T=2.0):
    log_s = F_fn.log_softmax(student / T, dim=-1)
    p_t = F_fn.softmax(teacher / T, dim=-1)
    return F_fn.kl_div(log_s, p_t, reduction="batchmean") * (T * T)


@torch.no_grad()
def codebook_geometry(pq):
    C = pq.codebooks.detach().float()  # [G, K, d]
    G, K, d = C.shape
    nrm = C.norm(dim=-1)               # [G, K]
    pair = {}
    if K >= 2:
        diff = C[:, 0] - C[:, 1]
        pair_l2 = diff.norm(dim=-1)
        pair_cos = F_fn.cosine_similarity(C[:, 0], C[:, 1], dim=-1)
        pair = {
            "pair_l2": pair_l2.cpu().numpy(),
            "pair_cos": pair_cos.cpu().numpy(),
        }
    return {
        "G": G, "K": K, "d": d,
        "code_norm": nrm.cpu().numpy(),
        **pair,
    }


@torch.no_grad()
def encode_residual_parts(Y, codec):
    """Return base recon + residual in the residual-R frame, grouped [B,T,G,d]."""
    Y_base = codec.base_forward(Y)
    R_y = Y - Y_base
    B, T, D = Y.shape
    Rr = codec.res.transform.get_rotation()
    Z = R_y.reshape(B * T, D) @ Rr
    Z_hat, usage = codec.res.pq._quantise(Z)
    G, d = codec.res.pq.G, codec.res.pq.d
    Zg = Z.view(B, T, G, d)
    Zhg = Z_hat.view(B, T, G, d)
    return Y_base, Zg, Zhg, Rr, usage


@torch.no_grad()
def decode_residual_groups(Zg, Rr):
    B, T, G, d = Zg.shape
    flat = Zg.reshape(B * T, G * d)
    return (flat @ Rr.t()).view(B, T, G * d)


@torch.no_grad()
def scan_codec(codec, features, device, norm_mode="per_image", batch_size=32):
    """Codebook geometry + per-group usage / energy / MSE on `features`."""
    codec.eval()
    pq = codec.res.pq
    geo = codebook_geometry(pq)
    G, K, d = pq.G, pq.K, pq.d
    usage = torch.zeros(G, K, device=device)
    energy = torch.zeros(G, device=device)
    mse = torch.zeros(G, device=device)
    n_tok = 0
    r_share = None
    if (codec.base.transform is not None
            and hasattr(codec.base.transform, "get_rotation")
            and hasattr(codec.res.transform, "get_rotation")):
        rb = codec.base.transform.get_rotation()
        rr = codec.res.transform.get_rotation()
        r_share = float((rb - rr).norm().item())

    for s in range(0, len(features), batch_size):
        e = min(s + batch_size, len(features))
        X = torch.from_numpy(np.stack(features[s:e])).float().to(device)
        Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
        _, Zg, Zhg, _, use = encode_residual_parts(Y, codec)
        usage += use
        err = (Zg - Zhg)
        energy += (Zg.pow(2).mean(dim=(0, 1, 3)) * (e - s))
        mse += (err.pow(2).mean(dim=(0, 1, 3)) * (e - s))
        n_tok += Zg.shape[0] * Zg.shape[1]
        del X, Y, Zg, Zhg, err, use
    n_img = len(features)
    energy = (energy / n_img).cpu().numpy()
    mse = (mse / n_img).cpu().numpy()
    use_np = usage.cpu().numpy()
    p = use_np / np.clip(use_np.sum(-1, keepdims=True), 1e-12, None)
    ppl = np.exp(-(p * np.log(p + 1e-30)).sum(-1))
    dead = int((use_np == 0).sum())
    pair_l2 = geo.get("pair_l2")
    pair_cos = geo.get("pair_cos")
    rms = np.sqrt(np.clip(energy, 0, None))
    collapse_ratio = (pair_l2 / np.clip(rms, 1e-8, None)
                      if pair_l2 is not None else None)

    groups = []
    for g in range(G):
        rec = {
            "g": g,
            "energy": float(energy[g]),
            "rms": float(rms[g]),
            "mse": float(mse[g]),
            "snr_db": float(10 * np.log10((energy[g] + 1e-12) / (mse[g] + 1e-12))),
            "ppl": float(ppl[g]),
            "usage": [float(x) for x in p[g]],
            "dead": int((use_np[g] == 0).sum()),
            "code_norm": [float(x) for x in geo["code_norm"][g]],
        }
        if pair_l2 is not None:
            rec["pair_l2"] = float(pair_l2[g])
            rec["pair_cos"] = float(pair_cos[g])
            rec["collapse_ratio"] = float(collapse_ratio[g])
        groups.append(rec)

    summary = {
        "G": G, "K": K, "d": d, "n_tokens": int(n_tok),
        "dead_entries": dead,
        "mean_ppl": float(ppl.mean()),
        "min_ppl": float(ppl.min()),
        "mean_pair_l2": float(pair_l2.mean()) if pair_l2 is not None else None,
        "min_pair_l2": float(pair_l2.min()) if pair_l2 is not None else None,
        "n_cos_gt_0.99": (int((pair_cos > 0.99).sum())
                          if pair_cos is not None else None),
        "n_cos_gt_0.90": (int((pair_cos > 0.90).sum())
                          if pair_cos is not None else None),
        "n_imbalance_gt_0.9": int((p.max(-1) > 0.9).sum()),
        "n_collapse_ratio_lt_0.2": (int((collapse_ratio < 0.2).sum())
                                    if collapse_ratio is not None else None),
        "n_collapse_ratio_lt_0.5": (int((collapse_ratio < 0.5).sum())
                                    if collapse_ratio is not None else None),
        "R_share_L2": r_share,
        "mean_energy": float(energy.mean()),
        "mean_mse": float(mse.mean()),
        "mean_snr_db": float(np.mean([g["snr_db"] for g in groups])),
    }
    return {"summary": summary, "groups": groups}


@torch.no_grad()
def eval_logits(task_tail, X_hat, labels, teacher):
    logits = task_tail.forward_nograd(X_hat)
    acc = float((logits.argmax(1) == labels).float().mean().item())
    ce = float(F_fn.cross_entropy(logits, labels).item())
    kl = float(_kd_kl(logits, teacher).item())
    return acc, ce, kl, int((logits.argmax(1) == labels).sum().item())


@torch.no_grad()
def loco_eval(codec, features, labels, task_tail, device,
              norm_mode="per_image", batch_size=16):
    """Leave-one-group-out classification metrics on the test set.

    only[g]: quantize residual group g, keep other groups at the original
             residual (i.e. uncompressed in those coordinates).
    drop[g]: original residual on group g, quantized residual on the rest.
    """
    codec.eval()
    G = codec.res.pq.G
    n = len(features)
    keys = (["original", "base", "full"]
            + [f"only_{g}" for g in range(G)]
            + [f"drop_{g}" for g in range(G)])
    tot = {k: dict(correct=0, ce=0.0, kl=0.0, n=0) for k in keys}

    def accu(name, acc, ce, kl, n_ok, bsz):
        tot[name]["correct"] += n_ok
        tot[name]["ce"] += ce * bsz
        tot[name]["kl"] += kl * bsz
        tot[name]["n"] += bsz

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        X = torch.from_numpy(np.stack(features[s:e])).float().to(device)
        y = torch.from_numpy(labels[s:e]).long().to(device)
        B = X.shape[0]
        Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
        teacher = task_tail.forward_nograd(X)

        Y_base, Zg, Zhg, Rr, _ = encode_residual_parts(Y, codec)
        err_g = Zhg - Zg                                 # [B,T,G,d]
        E = torch.zeros(G, B, X.shape[1], X.shape[2], device=device)
        for g in range(G):
            slot = torch.zeros_like(err_g)
            slot[:, :, g] = err_g[:, :, g]
            E[g] = decode_residual_groups(slot, Rr)

        Y_full = Y_base + decode_residual_groups(Zhg, Rr)
        variants = {
            "original": X,
            "base": batch_inv_normalize_gpu(Y_base, Mu, Std),
            "full": batch_inv_normalize_gpu(Y_full, Mu, Std),
        }
        for name, Xh in variants.items():
            acc, ce, kl, n_ok = eval_logits(task_tail, Xh, y, teacher)
            accu(name, acc, ce, kl, n_ok, B)
            if name != "original":
                del Xh

        for g in range(G):
            X_only = batch_inv_normalize_gpu(Y + E[g], Mu, Std)
            acc, ce, kl, n_ok = eval_logits(task_tail, X_only, y, teacher)
            accu(f"only_{g}", acc, ce, kl, n_ok, B)
            del X_only
            X_drop = batch_inv_normalize_gpu(Y_full - E[g], Mu, Std)
            acc, ce, kl, n_ok = eval_logits(task_tail, X_drop, y, teacher)
            accu(f"drop_{g}", acc, ce, kl, n_ok, B)
            del X_drop

        del X, Y, Mu, Std, Y_base, Zg, Zhg, err_g, E, Y_full, teacher, y
        if (s // batch_size) % 4 == 0:
            torch.cuda.empty_cache()

    def pack(name):
        t = tot[name]
        return {
            "acc": t["correct"] / t["n"],
            "ce": t["ce"] / t["n"],
            "kl": t["kl"] / t["n"],
            "n": t["n"],
        }

    only, drop = [], []
    orig, base, full = pack("original"), pack("base"), pack("full")
    for g in range(G):
        og = pack(f"only_{g}")
        dg = pack(f"drop_{g}")
        og["g"] = g
        dg["g"] = g
        og["d_acc_vs_orig"] = og["acc"] - orig["acc"]
        og["d_kl_vs_orig"] = og["kl"] - orig["kl"]
        dg["d_acc_vs_full"] = dg["acc"] - full["acc"]
        dg["d_kl_vs_full"] = dg["kl"] - full["kl"]
        only.append(og)
        drop.append(dg)
    return {
        "original": orig, "base": base, "full": full,
        "only": only, "drop": drop,
    }


def load_test(layer, feat_root, gt_path, max_images):
    test_dir = Path(feat_root) / "test" / "dinov2_vitl14" / layer
    files = sorted(test_dir.glob("*.npy"))
    if max_images > 0:
        files = files[:max_images]
    features, basenames = preload_features(files, num_workers=4, verbose=True)
    gt = load_gt(gt_path)
    keep, labels = [], []
    feats = []
    names = []
    for x, b in zip(features, basenames):
        if b in gt:
            feats.append(x)
            names.append(b)
            labels.append(gt[b])
    return feats, np.asarray(labels, dtype=np.int64), names


def load_tail(layer, device):
    layer_idx = int(layer[-2:])
    wrapper = Dinov2Wrapper(head_layers=1, model_name="dinov2_vitl14",
                            device=device)
    tail_blocks = list(wrapper.backbone.blocks[layer_idx + 1:])
    task_tail = ClsTaskTail(tail_blocks, wrapper.backbone.norm,
                            wrapper.head, device=device)
    print(f"  tail: {len(tail_blocks)} blocks after {layer}")
    return wrapper, task_tail


def run_one(json_path, payload, features, labels, device, args, task_tail=None):
    cfg = payload["config"]
    layer = cfg["layer"]
    k = _base_k(cfg["base_ckpt"])
    tag = f"{layer}_K{k}"
    print(f"\n=== {tag}  {Path(json_path).name[:80]} ===")
    print(f"  json Acc  base={payload['base_acc']:.4f}  "
          f"full={payload['full_acc']:.4f}  Δ={payload['delta_acc']:+.4f}")
    codec = load_residual_codec(payload["res_ckpt"], device=device,
                                base_ckpt_path=cfg["base_ckpt"])
    collapse = scan_codec(codec, features, device,
                          norm_mode=cfg.get("norm_mode", "per_image"),
                          batch_size=args.batch_size)
    s = collapse["summary"]
    print(f"  codebook  dead={s['dead_entries']}  mean_ppl={s['mean_ppl']:.3f}  "
          f"min_ppl={s['min_ppl']:.3f}  ||Rres-Rbase||={s['R_share_L2']:.2e}")
    print(f"  pair_l2   mean={s['mean_pair_l2']:.4f}  min={s['min_pair_l2']:.4f}  "
          f"cos>0.99={s['n_cos_gt_0.99']}  collapse_ratio<0.5="
          f"{s['n_collapse_ratio_lt_0.5']}/32")
    print(f"  residual  energy={s['mean_energy']:.4f}  mse={s['mean_mse']:.4f}  "
          f"SNR={s['mean_snr_db']:.2f} dB  imbalance>0.9={s['n_imbalance_gt_0.9']}")

    loco = None
    if args.mode in ("loco", "both") and task_tail is not None:
        print("  running leave-one-group-out …")
        loco = loco_eval(
            codec, features, labels, task_tail, device,
            norm_mode=cfg.get("norm_mode", "per_image"),
            batch_size=args.loco_batch,
        )
        print(f"  Acc  orig={loco['original']['acc']:.4f}  "
              f"base={loco['base']['acc']:.4f}  full={loco['full']['acc']:.4f}")
        print(f"  KL   orig={loco['original']['kl']:.4f}  "
              f"base={loco['base']['kl']:.4f}  full={loco['full']['kl']:.4f}")
        worst = min(loco["only"], key=lambda r: r["d_acc_vs_orig"])
        best_help = max(loco["drop"], key=lambda r: r["d_acc_vs_full"])
        print(f"  only  worst g={worst['g']}  ΔAcc={worst['d_acc_vs_orig']:+.4f}  "
              f"ΔKL={worst['d_kl_vs_orig']:+.4f}")
        print(f"  drop  best recover g={best_help['g']}  "
              f"ΔAcc={best_help['d_acc_vs_full']:+.4f}")

    del codec
    torch.cuda.empty_cache()
    return {
        "tag": tag,
        "layer": layer,
        "base_K": k,
        "json_path": str(json_path),
        "res_ckpt": payload["res_ckpt"],
        "json_base_acc": payload["base_acc"],
        "json_full_acc": payload["full_acc"],
        "json_delta_acc": payload["delta_acc"],
        "collapse": collapse,
        "loco": loco,
    }


def main():
    here = Path(ORFCV2_ROOT)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["collapse", "loco", "both"],
                   default="collapse")
    p.add_argument("--result_dir",
                   default=str(here / "results" / "dinov2_vitl14"))
    p.add_argument("--out", default="")
    p.add_argument("--layer", default="",
                   help="Restrict to one layer, e.g. blk15")
    p.add_argument("--base_k", type=int, nargs="*", default=None,
                   help="Restrict to these base K values")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--loco_batch", type=int, default=16)
    p.add_argument("--max_images", type=int, default=500)
    p.add_argument("--feat_root",
                   default=os.path.join(PROJECT_ROOT, "features"))
    p.add_argument("--gt_path",
                   default=os.path.join(PROJECT_ROOT, "utils",
                                        "imagenet_selected_label500.txt"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Honour a pre-set CUDA_VISIBLE_DEVICES (launcher); otherwise pick --gpu.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(
            f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"device={device}  CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

    result_dir = Path(args.result_dir)
    jobs = discover_shareR_jsons(result_dir, args.layer or None)
    if args.base_k:
        want = set(args.base_k)
        jobs = [(fp, d) for fp, d in jobs
                if _base_k(d["config"]["base_ckpt"]) in want]
    if not jobs:
        raise SystemExit("no matching shareR pure-KD results")
    print(f"jobs: {len(jobs)}")

    by_layer = {}
    for fp, d in jobs:
        by_layer.setdefault(d["config"]["layer"], []).append((fp, d))

    reports = []
    for layer, layer_jobs in by_layer.items():
        print(f"\n######## {layer}  n={len(layer_jobs)} ########")
        features, labels, _ = load_test(
            layer, args.feat_root, args.gt_path, args.max_images)
        print(f"  test images: {len(features)}")
        task_tail = None
        wrapper = None
        if args.mode in ("loco", "both"):
            wrapper, task_tail = load_tail(layer, device)
        for fp, d in layer_jobs:
            reports.append(run_one(
                fp, d, features, labels, device, args, task_tail=task_tail))
        if wrapper is not None:
            del wrapper, task_tail
            torch.cuda.empty_cache()
        del features, labels

    out = args.out or str(
        Path(args.result_dir) / "diag" /
        f"channels_{args.mode}_{args.layer or 'all'}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "mode": args.mode,
        "n_images": args.max_images,
        "reports": reports,
    }, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
