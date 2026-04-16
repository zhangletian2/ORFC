#!/usr/bin/env python
"""
Recompute rANS rate on cls (ImageNet 500) and seg (VOC 100) features
for ALL existing checkpoints, then append results to corresponding JSONs.

Adds two new fields to each JSON:
  - cls_rate_recomputed: {rans_bpt, bpfp, xent_bpt, H_emp_bpt, max_bpt}
  - voc_rate_info:       {rans_bpt, bpfp, xent_bpt, H_emp_bpt, max_bpt, n_tokens}
"""
import os, sys, math, json, re, time
import numpy as np
import torch
from pathlib import Path
from collections import OrderedDict

ORFC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(ORFC_ROOT, "..", ".."))
if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)

from run_multilayer_calibrator import preload_features
from opq import batch_normalize_gpu
from soft_pq import load_codec
from compressai._CXX import pmf_to_quantized_cdf
from compressai import ans

D = 1024
FEAT_ROOT = os.path.join(PROJECT_ROOT, "features")
SEG_ROOT  = os.path.join(PROJECT_ROOT, "features", "voc2012_100")
IMG_LIST  = os.path.join(PROJECT_ROOT, "utils", "voc2012_val_100.txt")
CKPT_DIR  = os.path.join(ORFC_ROOT, "checkpoints", "dinov2_vitl14")
JSON_DIR  = os.path.join(ORFC_ROOT, "results", "soft_pq", "dinov2_vitl14")


def get_labels_batch(feats, codec, device, batch_size=32):
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


def get_labels_variable(feats, codec, device):
    """For variable-length features (VOC slides)."""
    codec.eval()
    pq = codec.pq
    all_lab = []
    with torch.no_grad():
        for f in feats:
            X = torch.from_numpy(f).float().unsqueeze(0).to(device)
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


def compute_rate(labels, pmf_list, G, K):
    N = labels.shape[1]
    xent = 0.0
    for g in range(G):
        log2_p = np.log2(np.array(pmf_list[g]) + 1e-30)
        xent += -log2_p[labels[g]].sum()
    xent_bpt = xent / N

    test_pmf = hist_pmf(labels, G, K)
    H_emp = 0.0
    for g in range(G):
        pg = np.array(test_pmf[g])
        pg = pg[pg > 0]
        H_emp += -np.sum(pg * np.log2(pg))

    enc = ans.RansEncoder()
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
    rans_bpt = len(bs) * 8 / N

    return {
        "rans_bpt": round(rans_bpt, 4),
        "bpfp": round(rans_bpt / D, 6),
        "xent_bpt": round(xent_bpt, 4),
        "H_emp_bpt": round(H_emp, 4),
        "max_bpt": round(G * math.log2(K), 4),
        "n_tokens": int(N),
    }


def load_voc(layer):
    feat_dir = os.path.join(SEG_ROOT, "dinov2_vitl14", layer)
    with open(IMG_LIST) as f:
        names = [l.strip() for l in f if l.strip()]
    out = []
    for n in names:
        fp = os.path.join(feat_dir, f"{n}.npy")
        if os.path.exists(fp):
            d = np.load(fp)
            for s in range(d.shape[0]):
                out.append(d[s])
    return out


def parse_layer(tag):
    m = re.match(r'(blk\d+)_', tag)
    return m.group(1) if m else None


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    ckpts = sorted(Path(CKPT_DIR).glob("*.pt"))
    print(f"Found {len(ckpts)} checkpoints")

    # Group by layer
    layer_ckpts = OrderedDict()
    for cp in ckpts:
        tag = cp.stem
        layer = parse_layer(tag)
        if layer not in layer_ckpts:
            layer_ckpts[layer] = []
        layer_ckpts[layer].append(cp)

    cls_cache = {}
    voc_cache = {}
    total = len(ckpts)
    done = 0
    t0 = time.time()

    for layer, cps in layer_ckpts.items():
        print(f"\n{'='*60}")
        print(f"  Layer: {layer}  ({len(cps)} ckpts)")
        print(f"{'='*60}")

        if layer not in cls_cache:
            test_dir = Path(FEAT_ROOT) / "test" / "dinov2_vitl14" / layer
            files = sorted(test_dir.glob("*.npy"))
            cls_cache[layer], _ = preload_features(files, num_workers=8)
            print(f"  CLS: {len(cls_cache[layer])} images")

        if layer not in voc_cache:
            voc_cache[layer] = load_voc(layer)
            print(f"  VOC: {len(voc_cache[layer])} slides")

        for cp in cps:
            tag = cp.stem
            json_path = os.path.join(JSON_DIR, f"{tag}.json")

            if not os.path.exists(json_path):
                print(f"  [SKIP] {tag}: no JSON")
                done += 1
                continue

            codec = load_codec(str(cp), device=device)
            pq = codec.pq
            G, K = pq.G, pq.K

            if pq.use_rate:
                pmf = [pq.get_prior_pmf()[g] for g in range(G)]
            else:
                cls_labels_tmp = get_labels_batch(cls_cache[layer], codec, device)
                pmf = hist_pmf(cls_labels_tmp, G, K)

            # CLS rate
            cls_labels = get_labels_batch(cls_cache[layer], codec, device)
            cls_rate = compute_rate(cls_labels, pmf, G, K)

            # VOC rate
            voc_labels = get_labels_variable(voc_cache[layer], codec, device)
            voc_rate = compute_rate(voc_labels, pmf, G, K)

            # Update JSON
            with open(json_path) as f:
                data = json.load(f)
            data["cls_rate_recomputed"] = cls_rate
            data["voc_rate_info"] = voc_rate
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"  [{done}/{total}] {tag}")
            print(f"    CLS: BPFP={cls_rate['bpfp']:.4f}  rANS={cls_rate['rans_bpt']:.2f}")
            print(f"    VOC: BPFP={voc_rate['bpfp']:.4f}  rANS={voc_rate['rans_bpt']:.2f}")
            print(f"    ETA: {eta:.0f}s")

            del codec
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"  All done. {done}/{total} JSONs updated in {time.time()-t0:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
