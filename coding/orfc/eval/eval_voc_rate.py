#!/usr/bin/env python
"""Evaluate rANS BPFP on VOC2012 seg features, write seg_rate_info into result JSONs.

Usage:
  # Retroactively add VOC rate to ALL result JSONs (skips those already done)
  python eval_voc_rate.py --gpu 0

  # Only process blk20 ablation results
  python eval_voc_rate.py --gpu 0 --filter blk20

  # Dry run (show what would be done)
  python eval_voc_rate.py --gpu 0 --dry_run

  # Force re-compute even if seg_rate_info exists
  python eval_voc_rate.py --gpu 0 --force
"""
import os, sys, math, json, glob
import numpy as np
import torch

ORFC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(ORFC_ROOT, "..", ".."))
if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)

from opq import batch_normalize_gpu
from soft_pq import load_codec

try:
    from compressai._CXX import pmf_to_quantized_cdf
    from compressai import ans
    _HAS_ANS = True
except (ImportError, ModuleNotFoundError):
    _HAS_ANS = False

SEG_ROOT = os.path.join(PROJECT_ROOT, "features", "voc2012_100")
IMG_LIST = os.path.join(PROJECT_ROOT, "utils", "voc2012_val_100.txt")


def load_voc(layer, backbone="dinov2_vitl14"):
    feat_dir = os.path.join(SEG_ROOT, backbone, layer)
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


def get_labels(feats, codec, norm_mode, device):
    codec.eval()
    pq = codec.pq
    all_lab = []
    with torch.no_grad():
        for f in feats:
            X = torch.from_numpy(f).float().unsqueeze(0).to(device)
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
            codec(Y)
            all_lab.append(pq._last_labels.cpu())
            del X, Y
    return torch.cat(all_lab, dim=1).numpy()  # [G, N]


def hist_pmf(labels, G, K, smoothing=1.0):
    pmfs = []
    for g in range(G):
        c = np.zeros(K, dtype=np.float64)
        np.add.at(c, labels[g], 1)
        c += smoothing
        pmfs.append(c / c.sum())
    return pmfs


def rans_encode_bpt(labels, pmf_list, G, K, precision=16):
    if not _HAS_ANS:
        return None
    enc = ans.RansEncoder()
    N = labels.shape[1]
    cdfs, sizes = [], []
    for g in range(G):
        p = torch.from_numpy(pmf_list[g]).float()
        p = torch.cat([p, (1.0 - p.sum()).clamp_min(0).unsqueeze(0)])
        cdfs.append(pmf_to_quantized_cdf(p.tolist(), precision))
        sizes.append(K + 2)
    sym, idx = [], []
    for n in range(N):
        for g in range(G):
            sym.append(int(labels[g, n]))
            idx.append(g)
    bs = enc.encode_with_indexes(sym, idx, cdfs, sizes, [0] * G)
    return len(bs) * 8 / N


def compute_seg_rate_info(voc_feats, codec, norm_mode, device, D):
    """Compute seg_rate_info dict matching run_soft_pq.py evaluate_rate format."""
    G = codec.pq.G
    K = codec.pq.K
    labels = get_labels(voc_feats, codec, norm_mode, device)
    N = labels.shape[1]

    if codec.pq.use_rate:
        primary_pmf_arr = codec.pq.get_prior_pmf()   # [G, K] numpy
        primary_pmf = [primary_pmf_arr[g] for g in range(G)]
    else:
        primary_pmf = hist_pmf(labels, G, K)

    voc_pmf = hist_pmf(labels, G, K)

    xent_primary = sum(
        -np.log2(primary_pmf[g][labels[g]] + 1e-30).sum() for g in range(G)
    ) / N
    xent_voc = sum(
        -np.log2(voc_pmf[g][labels[g]] + 1e-30).sum() for g in range(G)
    ) / N

    emp_ent = 0.0
    test_pmf = hist_pmf(labels, G, K, smoothing=0)
    for g in range(G):
        p = test_pmf[g]
        p = p[p > 0]
        emp_ent += -np.sum(p * np.log2(p))

    rans_primary = rans_encode_bpt(labels, primary_pmf, G, K)
    rans_voc = rans_encode_bpt(labels, voc_pmf, G, K)

    result = {
        'xent_rate_bpt': float(xent_primary),
        'empirical_entropy_bpt': float(emp_ent),
        'max_rate_bpt': float(G * math.log2(K)),
    }
    if abs(xent_voc - xent_primary) > 1e-6:
        result['xent_train_bpt'] = float(xent_voc)
    if rans_primary is not None:
        result['rans_bpt'] = float(rans_primary)
    if rans_voc is not None and abs((rans_voc or 0) - (rans_primary or 0)) > 1e-6:
        result['rans_train_bpt'] = float(rans_voc)
    return result


def ckpt_path_from_config(cfg):
    """Construct checkpoint path, trying new naming (with fz_tag) then old."""
    layer = cfg['layer']
    K = cfg['K']
    emb = cfg['embedding_dim']
    bt = cfg.get('bottleneck_dim', 1024)
    ws = cfg.get('warm_start_opq', True)
    mse = cfg.get('mse_loss', False)
    lmbda = cfg.get('lmbda', 0.0)
    tau_start = cfg.get('tau_start', 0.5)
    freeze_transform = cfg.get('freeze_transform', False)
    freeze_codebooks = cfg.get('freeze_codebooks', False)
    init_rotation = cfg.get('init_rotation', 'opq')
    prior_floor = cfg.get('prior_floor', 0.0)
    lr = cfg['lr']
    epochs = cfg['epochs']
    n = cfg.get('max_train_images', 5000)
    seed = cfg.get('seed', 42)
    backbone = cfg.get('backbone', 'dinov2_vitl14')

    bt_tag = f"bt{bt}" if bt > 0 else "noBt"
    ws_tag = "ws" if ws else "km"
    mse_tag = "_mse" if mse else ""
    rate_tag = f"_lmbda{lmbda}" if lmbda > 0 else ""
    tau_tag = f"_tau{tau_start}" if tau_start > 0 else ""
    fz_tag = ""
    if freeze_transform:
        fz_tag += "_fzR"
    if freeze_codebooks:
        fz_tag += "_fzC"
    rot_tag = f"_rot{init_rotation}" if init_rotation != 'opq' else ""
    pfloor_tag = f"_pf{prior_floor}" if prior_floor > 0 else ""

    ckpt_dir = os.path.join(ORFC_ROOT, "checkpoints", backbone)

    new_name = (f"{layer}_K{K}_emb{emb}_{bt_tag}_{ws_tag}{mse_tag}{rate_tag}"
                f"{fz_tag}{rot_tag}{pfloor_tag}{tau_tag}"
                f"_lr{lr}_ep{epochs}_n{n}_s{seed}.pt")
    new_path = os.path.join(ckpt_dir, new_name)
    if os.path.exists(new_path):
        return new_path

    old_name = (f"{layer}_K{K}_emb{emb}_{bt_tag}_{ws_tag}{mse_tag}{rate_tag}"
                f"{tau_tag}_lr{lr}_ep{epochs}_n{n}_s{seed}.pt")
    old_path = os.path.join(ckpt_dir, old_name)
    if os.path.exists(old_path):
        return old_path

    return None


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Retroactively compute VOC seg rate for result JSONs")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--results_dir", type=str, default=None)
    p.add_argument("--filter", type=str, default="",
                   help="Only process JSONs whose filename contains this string")
    p.add_argument("--force", action="store_true",
                   help="Re-compute even if seg_rate_info already exists")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    results_dir = args.results_dir or os.path.join(
        ORFC_ROOT, "results", "soft_pq", "dinov2_vitl14")
    json_files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    if args.filter:
        json_files = [f for f in json_files if args.filter in os.path.basename(f)]

    print(f"  Found {len(json_files)} result JSONs"
          f" (filter={args.filter!r})")

    voc_cache = {}
    updated, skipped, no_ckpt = 0, 0, 0

    for jf in json_files:
        bn = os.path.basename(jf)
        with open(jf) as f:
            data = json.load(f)

        if 'seg_rate_info' in data and not args.force:
            skipped += 1
            continue

        cfg = data.get('config', {})
        layer = cfg.get('layer', '')
        if not layer:
            print(f"  SKIP (no config): {bn}")
            skipped += 1
            continue

        ckpt = ckpt_path_from_config(cfg)
        if ckpt is None:
            print(f"  NO CKPT: {bn}")
            no_ckpt += 1
            continue

        if args.dry_run:
            print(f"  WOULD: {bn}  <-  {os.path.basename(ckpt)}")
            continue

        if layer not in voc_cache:
            print(f"\n  Loading VOC features for {layer}...")
            voc_cache[layer] = load_voc(layer)
            n_tok = sum(f.shape[0] for f in voc_cache[layer])
            print(f"    {len(voc_cache[layer])} slides, {n_tok} tokens")

        codec = load_codec(ckpt, device=device)
        norm_mode = cfg.get('norm_mode', 'per_image')
        D = voc_cache[layer][0].shape[-1]

        seg_rate_info = compute_seg_rate_info(
            voc_cache[layer], codec, norm_mode, device, D)
        data['seg_rate_info'] = seg_rate_info

        with open(jf, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        rans = seg_rate_info.get('rans_bpt')
        bpfp = rans / D if rans else None
        print(f"  OK {bn}  rANS={rans:.2f}  BPFP={bpfp:.4f}"
              if rans else f"  OK {bn}  rANS=N/A")

        del codec
        torch.cuda.empty_cache()
        updated += 1

    print(f"\n{'='*60}")
    print(f"  Done: updated={updated}  skipped={skipped}  no_ckpt={no_ckpt}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
