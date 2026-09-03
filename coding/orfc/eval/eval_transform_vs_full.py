#!/usr/bin/env python
"""Compare R-only transform vs Full ORFC codec: time & GPU memory.

Fixed codebook config: K=64, emb=32.
Models: dinov2_vitl14 (blk05, D=1024) and dinov2_vitg14 (blk09, D=1536).

R-only pipeline (transform only, no quantisation):
  Encode: normalize → rotate   (flat @ R)
  Decode: inv-rotate → denorm  (Z @ R^T, then denorm)

Full ORFC pipeline:
  Encode: normalize → rotate → PQ quantize → rANS encode
  Decode: rANS decode → lookup → inv-rotate → denormalize

Usage:
  python eval_transform_vs_full.py --gpu 0
"""
import os, sys, time, math
import numpy as np
import torch
import torch.nn.functional as F_fn
from pathlib import Path

ORFC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(ORFC_ROOT, "..", ".."))
if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)

from opq import batch_normalize_gpu, batch_inv_normalize_gpu
from soft_pq import load_codec
from compressai._CXX import pmf_to_quantized_cdf
from compressai import ans

FEAT_ROOT = os.path.join(PROJECT_ROOT, "features")

# (backbone, layer, D, K, emb, lmbda, lr, epochs)
MODEL_SPECS = [
    ("dinov2_vitl14", "blk05", 1024, 64, 32, 0.5, 5e-4, 100),
    ("dinov2_vitg14", "blk09", 1536, 64, 32, 0.5, 3e-4, 100),
]


def preload_features(feat_files, num_workers=1):
    from concurrent.futures import ThreadPoolExecutor
    feat_files = list(feat_files)
    def _load(f):
        return np.load(f).astype(np.float32), Path(f).stem
    if num_workers <= 1:
        results = [_load(f) for f in feat_files]
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            results = list(pool.map(_load, feat_files))
    return [r[0] for r in results], [r[1] for r in results]


def get_ckpt(backbone, layer, D, K, emb, lmbda, lr, epochs):
    rt = f"_lmbda{lmbda}" if lmbda > 0 else ""
    return os.path.join(
        ORFC_ROOT, "checkpoints", backbone,
        f"{layer}_K{K}_emb{emb}_bt{D}_ws{rt}_tau0.5_lr{lr}_ep{epochs}_n5000_s42.pt")


# ---- PMF / CDF helpers ----

def get_all_labels_concat(feats, codec, device):
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


def prepare_cdfs(pmf_list, G, K, precision=16):
    cdfs, sizes = [], []
    for g in range(G):
        p = torch.from_numpy(pmf_list[g]).float()
        p = torch.cat([p, (1.0 - p.sum()).clamp_min(0).unsqueeze(0)])
        cdfs.append(pmf_to_quantized_cdf(p.tolist(), precision))
        sizes.append(K + 2)
    return cdfs, sizes


# ---- R-only measurement ----

def measure_r_only(feats, R, device, D, n_warmup=10):
    """Measure per-feature time & peak GPU memory for rotation-only pipeline.

    Encode = normalize + rotate
    Decode = inv-rotate + denormalize
    """
    enc_times, dec_times, token_counts = [], [], []
    enc_mem_peaks, dec_mem_peaks = [], []

    for i, feat in enumerate(feats):
        N = feat.shape[0]

        # ---- R-only ENCODE ----
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
            Y, Mu, Std = batch_normalize_gpu(X, mode="per_image")
            flat = Y.reshape(-1, D)
            Z = flat @ R

        torch.cuda.synchronize()
        t_enc = time.perf_counter() - t0
        enc_peak = torch.cuda.max_memory_allocated(device)

        # ---- R-only DECODE ----
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            Y_hat = Z @ R.t()
            X_hat = batch_inv_normalize_gpu(
                Y_hat.reshape(1, N, D), Mu, Std)

        torch.cuda.synchronize()
        t_dec = time.perf_counter() - t0
        dec_peak = torch.cuda.max_memory_allocated(device)

        if i >= n_warmup:
            enc_times.append(t_enc)
            dec_times.append(t_dec)
            token_counts.append(N)
            enc_mem_peaks.append(enc_peak)
            dec_mem_peaks.append(dec_peak)

        del X, Y, Mu, Std, flat, Z, Y_hat, X_hat

    return enc_times, dec_times, token_counts, enc_mem_peaks, dec_mem_peaks


# ---- Full ORFC measurement ----

def measure_full_orfc(feats, codec, R, device, cdfs, sizes, G, K, d, D,
                      n_warmup=10):
    """Measure per-feature time & peak GPU memory for full ORFC pipeline.

    Encode = normalize + rotate + PQ-quantize + rANS-encode
    Decode = rANS-decode + codebook-lookup + inv-rotate + denormalize
    """
    codec.eval()
    pq = codec.pq
    C = pq.codebooks
    offsets = [0] * G

    log2_pmf_cost = None
    if pq.use_rate:
        with torch.no_grad():
            log_p = F_fn.log_softmax(pq.log_prior, dim=-1)
            pf = pq.prior_floor
            if pf > 0:
                p_val = log_p.exp()
                p_val = (1.0 - pf) * p_val + pf / K
                log2_pmf_cost = -(p_val + 1e-30).log() / math.log(2)
            else:
                log2_pmf_cost = -log_p / math.log(2)

    enc_times, dec_times, token_counts = [], [], []
    enc_mem_peaks, dec_mem_peaks = [], []

    for i, feat in enumerate(feats):
        N = feat.shape[0]

        # ---- Full ENCODE ----
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
            Y, Mu, Std = batch_normalize_gpu(X, mode="per_image")
            flat = Y.reshape(-1, D)
            Z = flat @ R
            sub_g = Z.reshape(N, G, d).permute(1, 0, 2)
            dists_sq = torch.cdist(sub_g, C).pow(2)
            if log2_pmf_cost is not None:
                cost = dists_sq + log2_pmf_cost.unsqueeze(1) / pq.lmbda
            else:
                cost = dists_sq
            labels = cost.argmin(dim=-1)
            labels_np = labels.cpu().numpy()

        torch.cuda.synchronize()

        sym = labels_np.T.ravel().tolist()
        idx_list = np.tile(np.arange(G, dtype=np.int32), N).tolist()
        enc_obj = ans.RansEncoder()
        bitstream = enc_obj.encode_with_indexes(
            sym, idx_list, cdfs, sizes, offsets)

        t_enc = time.perf_counter() - t0
        enc_peak = torch.cuda.max_memory_allocated(device)

        # ---- Full DECODE ----
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        dec_obj = ans.RansDecoder()
        decoded_sym = dec_obj.decode_with_indexes(
            bitstream, idx_list, cdfs, sizes, offsets)

        with torch.no_grad():
            dec_np = np.array(decoded_sym, dtype=np.int64).reshape(N, G).T
            dec_labels = torch.from_numpy(dec_np).to(device)

            Z_hat_g = torch.gather(
                C.unsqueeze(1).expand(-1, N, -1, -1), 2,
                dec_labels.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, d),
            ).squeeze(2)
            Z_hat = Z_hat_g.permute(1, 0, 2).reshape(N, D)
            Y_hat = Z_hat @ R.t()
            X_hat = batch_inv_normalize_gpu(
                Y_hat.unsqueeze(0), Mu, Std)

        torch.cuda.synchronize()
        t_dec = time.perf_counter() - t0
        dec_peak = torch.cuda.max_memory_allocated(device)

        if i >= n_warmup:
            enc_times.append(t_enc)
            dec_times.append(t_dec)
            token_counts.append(N)
            enc_mem_peaks.append(enc_peak)
            dec_mem_peaks.append(dec_peak)

        del X, Y, Mu, Std, X_hat

    return enc_times, dec_times, token_counts, enc_mem_peaks, dec_mem_peaks


def summarise(label, enc_t, dec_t, tok_n, enc_mem, dec_mem):
    return {
        'label': label,
        'enc_ms': np.mean(enc_t) * 1000,
        'enc_std': np.std(enc_t) * 1000,
        'dec_ms': np.mean(dec_t) * 1000,
        'dec_std': np.std(dec_t) * 1000,
        'avg_tok': np.mean(tok_n),
        'n': len(enc_t),
        'enc_mem_avg_MB': np.mean(enc_mem) / (1024 ** 2),
        'enc_mem_peak_MB': np.max(enc_mem) / (1024 ** 2),
        'dec_mem_avg_MB': np.mean(dec_mem) / (1024 ** 2),
        'dec_mem_peak_MB': np.max(dec_mem) / (1024 ** 2),
    }


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="R-only vs Full ORFC timing & memory comparison")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_warmup", type=int, default=10)
    args = p.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    all_rows = []

    for backbone, layer, D, K, emb, lmbda, lr, epochs in MODEL_SPECS:
        G = D // emb
        d = emb

        print(f"\n{'#'*70}")
        print(f"# {backbone}  layer={layer}  D={D}  K={K}  G={G}  d={d}")
        print(f"{'#'*70}")

        # Load features
        test_dir = Path(FEAT_ROOT) / "test" / backbone / layer
        files = sorted(test_dir.glob("*.npy"))
        if not files:
            print(f"  SKIP: no features in {test_dir}")
            continue
        feats, _ = preload_features(files, num_workers=8)
        print(f"  {len(feats)} test images, shape={feats[0].shape}")

        # Load codec
        cp = get_ckpt(backbone, layer, D, K, emb, lmbda, lr, epochs)
        if not os.path.exists(cp):
            print(f"  SKIP: checkpoint missing ({os.path.basename(cp)})")
            continue
        codec = load_codec(cp, device=device)

        # Pre-compute R
        R = None
        if codec.transform is not None and hasattr(codec.transform, 'get_rotation'):
            with torch.no_grad():
                R = codec.transform.get_rotation()
        if R is None:
            print(f"  SKIP: no OrthogonalTransform found in codec")
            del codec
            continue

        # PMF / CDF for rANS
        print(f"  computing labels for PMF...")
        all_labels = get_all_labels_concat(feats, codec, device)
        if codec.pq.use_rate:
            pmf = [codec.pq.get_prior_pmf()[g] for g in range(G)]
        else:
            from eval_timing import hist_pmf
            pmf = hist_pmf(all_labels, G, K)
        cdfs, sizes = prepare_cdfs(pmf, G, K)

        # ---- Measure R-only ----
        print(f"  [R-only]    timing {len(feats)} features (warmup={args.n_warmup})...")
        r_enc, r_dec, r_tok, r_emem, r_dmem = measure_r_only(
            feats, R, device, D, n_warmup=args.n_warmup)
        r_stats = summarise("R-only", r_enc, r_dec, r_tok, r_emem, r_dmem)

        torch.cuda.empty_cache()

        # ---- Measure Full ORFC ----
        print(f"  [Full ORFC] timing {len(feats)} features (warmup={args.n_warmup})...")
        f_enc, f_dec, f_tok, f_emem, f_dmem = measure_full_orfc(
            feats, codec, R, device, cdfs, sizes, G, K, d, D,
            n_warmup=args.n_warmup)
        f_stats = summarise("Full ORFC", f_enc, f_dec, f_tok, f_emem, f_dmem)

        # Print per-model results
        for s in (r_stats, f_stats):
            print(f"    {s['label']:>10}  "
                  f"enc {s['enc_ms']:.3f}±{s['enc_std']:.3f} ms  "
                  f"dec {s['dec_ms']:.3f}±{s['dec_std']:.3f} ms  "
                  f"enc_mem {s['enc_mem_avg_MB']:.1f} (peak {s['enc_mem_peak_MB']:.1f}) MB  "
                  f"dec_mem {s['dec_mem_avg_MB']:.1f} (peak {s['dec_mem_peak_MB']:.1f}) MB")

        all_rows.append((backbone, layer, D, r_stats, f_stats))

        del codec, R
        torch.cuda.empty_cache()

    # ---- Summary table ----
    print(f"\n{'='*140}")
    print(f"{'Model':>16} {'Layer':>6} {'Mode':>10} | "
          f"{'Enc(ms)':>16} {'Dec(ms)':>16} | "
          f"{'EncMem(MB)':>12} {'DecMem(MB)':>12} | "
          f"{'Tokens':>7} {'N':>5}")
    print(f"{'-'*140}")
    for backbone, layer, D, r_s, f_s in all_rows:
        for s in (r_s, f_s):
            print(f"{backbone:>16} {layer:>6} {s['label']:>10} | "
                  f"{s['enc_ms']:>7.3f}±{s['enc_std']:<7.3f}"
                  f"{s['dec_ms']:>7.3f}±{s['dec_std']:<7.3f} | "
                  f"{s['enc_mem_peak_MB']:>12.1f} {s['dec_mem_peak_MB']:>12.1f} | "
                  f"{s['avg_tok']:>7.0f} {s['n']:>5}")

    # R-only fraction
    print(f"\n{'='*80}")
    print(f"{'Model':>16} {'Layer':>6} | {'R/Full Enc%':>12} {'R/Full Dec%':>12} | "
          f"{'R/Full EncMem%':>14} {'R/Full DecMem%':>14}")
    print(f"{'-'*80}")
    for backbone, layer, D, r_s, f_s in all_rows:
        enc_pct = r_s['enc_ms'] / f_s['enc_ms'] * 100 if f_s['enc_ms'] > 0 else 0
        dec_pct = r_s['dec_ms'] / f_s['dec_ms'] * 100 if f_s['dec_ms'] > 0 else 0
        emem_pct = r_s['enc_mem_peak_MB'] / f_s['enc_mem_peak_MB'] * 100 if f_s['enc_mem_peak_MB'] > 0 else 0
        dmem_pct = r_s['dec_mem_peak_MB'] / f_s['dec_mem_peak_MB'] * 100 if f_s['dec_mem_peak_MB'] > 0 else 0
        print(f"{backbone:>16} {layer:>6} | "
              f"{enc_pct:>11.1f}% {dec_pct:>11.1f}% | "
              f"{emem_pct:>13.1f}% {dmem_pct:>13.1f}%")


if __name__ == "__main__":
    main()
