#!/usr/bin/env python
"""Evaluate rANS BPFP on ImageNet 500 test features for all best configs.
   Compare with CSV values to verify consistency."""
import os, sys, math, json
import numpy as np
import torch
from pathlib import Path

ORFC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(ORFC_ROOT, "..", ".."))
if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)

from run_multilayer_calibrator import preload_features
from opq import batch_normalize_gpu
from soft_pq import load_codec
from compressai._CXX import pmf_to_quantized_cdf
from compressai import ans

CONFIGS = [
    # (layer, K, emb, lmbda, lr, epochs, desc, csv_bpfp)
    ("blk05",   4, 32, 0.5, 3e-4, 100, "K=4,e32",           0.0610),
    ("blk05",   8, 32, 0.5, 3e-4, 100, "K=8,e32",           0.0922),
    ("blk05",  16, 32, 0.5, 3e-4, 100, "K=16,e32",          0.1230),
    ("blk05",  64, 32, 0.5, 5e-4, 100, "K=64,e32,lr5e4",    0.1849),
    ("blk05", 256, 32, 0.5, 3e-4, 100, "K=256,e32",         0.2471),
    ("blk05",  64, 16, 0.5, 3e-4, 100, "K=64,e16",          0.3707),
    ("blk05", 256, 16, 0.2, 3e-4, 100, "K=256,e16,lm0.2",   0.4476),
    ("blk05", 256, 16, 0.5, 3e-4, 100, "K=256,e16",         0.4880),
    ("blk10",   4, 32, 0.5, 3e-4, 100, "K=4,e32",           0.0602),
    ("blk10",   8, 32, 0.5, 3e-4, 100, "K=8,e32",           0.0913),
    ("blk10",  16, 32, 0.5, 3e-4, 100, "K=16,e32",          0.1228),
    ("blk10",  64, 32, 0.0, 3e-4, 100, "K=64,e32,lm0",      0.1849),
    ("blk10", 256, 32, 0.0, 5e-4, 100, "K=256,e32,lm0,lr5e4",0.2460),
    ("blk10", 256, 32, 0.5, 3e-4, 100, "K=256,e32",         0.2467),
    ("blk10",  64, 16, 0.5, 3e-4, 100, "K=64,e16",          0.3687),
    ("blk10", 256, 16, 0.5, 3e-4, 100, "K=256,e16",         0.4844),
    ("blk15",   4, 32, 0.5, 3e-4, 100, "K=4,e32",           0.0610),
    ("blk15",   8, 32, 0.5, 3e-4, 100, "K=8,e32",           0.0922),
    ("blk15",  16, 32, 0.5, 3e-4, 100, "K=16,e32",          0.1230),
    ("blk15",  64, 32, 0.5, 5e-4, 100, "K=64,e32,lr5e4",    0.1849),
    ("blk15", 256, 32, 0.5, 3e-4, 100, "K=256,e32",         0.2471),
    ("blk15",  64, 16, 0.5, 3e-4, 100, "K=64,e16",          0.3707),
    ("blk15", 256, 16, 0.2, 3e-4, 100, "K=256,e16,lm0.2",   0.4476),
    ("blk15", 256, 16, 0.5, 3e-4, 100, "K=256,e16",         0.4880),
    ("blk20",   4, 32, 0.0, 1e-3, 300, "K=4,e32,lm0,ep300", 0.0613),
    ("blk20",   8, 32, 0.5, 3e-4, 100, "K=8,e32",           0.0928),
    ("blk20",  16, 32, 0.5, 5e-4, 100, "K=16,e32,lr5e4",    0.1239),
    ("blk20",  32, 32, 0.5, 3e-4, 100, "K=32,e32",          0.1538),
    ("blk20",  64, 32, 0.0, 3e-4, 100, "K=64,e32,lm0",      0.1842),
    ("blk20", 256, 32, 0.0, 5e-4, 100, "K=256,e32,lm0,lr5e4",0.2469),
    ("blk20", 256, 16, 0.2, 3e-4, 100, "K=256,e16,lm0.2",   0.4100),
    ("blk20", 256, 16, 0.5, 3e-4, 100, "K=256,e16",         0.4883),
]

D = 1024
FEAT_ROOT = os.path.join(PROJECT_ROOT, "features")


def get_ckpt(layer, K, emb, lmbda, lr, epochs):
    rt = f"_lmbda{lmbda}" if lmbda > 0 else ""
    return os.path.join(ORFC_ROOT, "checkpoints", "dinov2_vitl14",
        f"{layer}_K{K}_emb{emb}_bt1024_ws{rt}_tau0.5_lr{lr}_ep{epochs}_n5000_s42.pt")


def get_labels_batch(feats, codec, device, batch_size=32):
    """feats: list of [T, D] arrays (same T). Returns [G, N_total]."""
    codec.eval()
    pq = codec.pq
    all_lab = []
    with torch.no_grad():
        for i in range(0, len(feats), batch_size):
            batch = feats[i:i+batch_size]
            X = torch.from_numpy(np.stack(batch)).float().to(device)
            Y, _, _ = batch_normalize_gpu(X, mode="per_image")
            codec(Y)
            all_lab.append(pq._last_labels.cpu())
            del X, Y
    return torch.cat(all_lab, dim=1).numpy()


def hist_pmf(labels, G, K):
    pmfs = []
    for g in range(G):
        c = np.zeros(K, dtype=np.float64)
        np.add.at(c, labels[g], 1)
        c += 1.0
        pmfs.append(c / c.sum())
    return pmfs


def rans_bpt(labels, pmf_list, G, K):
    enc = ans.RansEncoder()
    N = labels.shape[1]
    cdfs, sizes = [], []
    for g in range(G):
        p = torch.from_numpy(pmf_list[g]).float()
        p = torch.cat([p, (1.0 - p.sum()).clamp_min(0).unsqueeze(0)])
        cdfs.append(pmf_to_quantized_cdf(p.tolist(), 16))
        sizes.append(K + 2)
    sym, idx = [], []
    for n in range(N):
        for g in range(G):
            sym.append(int(labels[g, n]))
            idx.append(g)
    bs = enc.encode_with_indexes(sym, idx, cdfs, sizes, [0]*G)
    return len(bs) * 8 / N


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    feat_cache = {}
    results = []

    for layer, K, emb, lmbda, lr, epochs, desc, csv_bpfp in CONFIGS:
        G = D // emb
        cp = get_ckpt(layer, K, emb, lmbda, lr, epochs)
        if not os.path.exists(cp):
            print(f"  SKIP {layer} {desc}: ckpt missing")
            continue

        if layer not in feat_cache:
            print(f"\n  Loading ImageNet test features for {layer}...")
            test_dir = Path(FEAT_ROOT) / "test" / "dinov2_vitl14" / layer
            files = sorted(test_dir.glob("*.npy"))
            feat_cache[layer], _ = preload_features(files, num_workers=8)
            print(f"    {len(feat_cache[layer])} images, T={feat_cache[layer][0].shape[0]}")

        codec = load_codec(cp, device=device)
        labels = get_labels_batch(feat_cache[layer], codec, device)

        if codec.pq.use_rate:
            pmf = [codec.pq.get_prior_pmf()[g] for g in range(G)]
        else:
            pmf = hist_pmf(labels, G, K)

        bpt = rans_bpt(labels, pmf, G, K)
        bpfp = bpt / D
        diff = bpfp - csv_bpfp

        results.append({"layer": layer, "desc": desc, "K": K, "emb": emb,
                         "lmbda": lmbda, "bpfp": bpfp, "csv_bpfp": csv_bpfp,
                         "diff": diff})
        tag = "OK" if abs(diff) < 0.005 else "DIFF!"
        print(f"  {layer} {desc:30s}  BPFP={bpfp:.4f}  CSV={csv_bpfp:.4f}  Δ={diff:+.4f}  {tag}")

        del codec; torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print(f"{'Layer':>6} {'Config':>30} {'recomputed':>10} {'CSV':>8} {'Δ':>8} {'status':>6}")
    print("-"*80)
    for r in results:
        tag = "OK" if abs(r['diff']) < 0.005 else "DIFF!"
        print(f"{r['layer']:>6} {r['desc']:>30} {r['bpfp']:>10.4f} {r['csv_bpfp']:>8.4f} "
              f"{r['diff']:>+8.4f} {tag:>6}")

    with open(os.path.join(ORFC_ROOT, "cls_rate_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: cls_rate_results.json")


if __name__ == "__main__":
    main()
