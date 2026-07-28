#!/usr/bin/env python
"""Measure per-feature encode/decode time and GPU memory for Soft-PQ codec.

Full pipeline timed:
  Encode: normalize → rotate → PQ quantize → rANS entropy encode
  Decode: rANS entropy decode → codebook lookup → inverse rotate → denormalize

Reports average time (ms) and peak GPU memory (MB) per feature.

Usage:
  python eval_timing.py --gpu 0
  python eval_timing.py --gpu 0 --n_warmup 20
"""
import os, sys, time, json, math
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

# (backbone, layer, D, configs)
# configs: list of (K, emb, lmbda, lr, epochs, desc)
MODEL_SPECS = [
    ("dinov2_vitl14", "blk05", 1024, [
        (4,   32, 0.5, 3e-4, 100, "K=4,e32"),
        (16,  32, 0.5, 3e-4, 100, "K=16,e32"),
        (64,  32, 0.5, 5e-4, 100, "K=64,e32"),
        (256, 32, 0.5, 3e-4, 100, "K=256,e32"),
    ]),
    ("dinov2_vitg14", "blk09", 1536, [
        (4,   32, 0.5, 3e-4, 100, "K=4,e32"),
        (16,  32, 0.5, 3e-4, 100, "K=16,e32"),
        (64,  32, 0.5, 3e-4, 100, "K=64,e32"),
        (256, 32, 0.5, 3e-4, 100, "K=256,e32"),
    ]),
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


def hist_pmf(labels, G, K):
    pmfs = []
    for g in range(G):
        c = np.zeros(K, dtype=np.float64)
        np.add.at(c, labels[g], 1)
        c += 1.0
        pmfs.append(c / c.sum())
    return pmfs


def prepare_cdfs(pmf_list, G, K, precision=16):
    cdfs, sizes = [], []
    for g in range(G):
        p = torch.from_numpy(pmf_list[g]).float()
        p = torch.cat([p, (1.0 - p.sum()).clamp_min(0).unsqueeze(0)])
        cdfs.append(pmf_to_quantized_cdf(p.tolist(), precision))
        sizes.append(K + 2)
    return cdfs, sizes


def get_all_labels_concat(feats, codec, device):
    """Get quantization labels for all features (for PMF estimation)."""
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


def measure_timing(feats, codec, device, cdfs, sizes, G, K, d, D,
                   n_warmup=10):
    """Measure per-feature encode/decode time and peak GPU memory.

    Encode = normalize + rotate + PQ-quantize + rANS-encode
    Decode = rANS-decode + codebook-lookup + inv-rotate + denormalize

    First n_warmup features are discarded from timing statistics.
    Returns (enc_times, dec_times, token_counts,
             enc_mem_peaks, dec_mem_peaks) — times in seconds, mem in bytes.
    """
    codec.eval()
    pq = codec.pq
    C = pq.codebooks                                      # [G, K, d]
    offsets = [0] * G

    R = None
    if codec.transform is not None and hasattr(codec.transform, 'get_rotation'):
        with torch.no_grad():
            R = codec.transform.get_rotation()

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
        N = feat.shape[0]                                 # tokens

        # ==================== ENCODE ====================
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
            Y, Mu, Std = batch_normalize_gpu(X, mode="per_image")
            flat = Y.reshape(-1, D)
            Z = flat @ R if R is not None else flat
            sub_g = Z.reshape(N, G, d).permute(1, 0, 2)   # [G, N, d]
            dists_sq = torch.cdist(sub_g, C).pow(2)        # [G, N, K]
            if log2_pmf_cost is not None:
                cost = dists_sq + log2_pmf_cost.unsqueeze(1) / pq.lmbda
            else:
                cost = dists_sq
            labels = cost.argmin(dim=-1)                   # [G, N]
            labels_np = labels.cpu().numpy()

        torch.cuda.synchronize()

        sym = labels_np.T.ravel().tolist()
        idx_list = np.tile(np.arange(G, dtype=np.int32), N).tolist()
        enc_obj = ans.RansEncoder()
        bitstream = enc_obj.encode_with_indexes(
            sym, idx_list, cdfs, sizes, offsets)

        t_enc = time.perf_counter() - t0
        enc_peak = torch.cuda.max_memory_allocated(device)

        # ==================== DECODE ====================
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
            ).squeeze(2)                                   # [G, N, d]
            Z_hat = Z_hat_g.permute(1, 0, 2).reshape(N, D)
            Y_hat = (Z_hat @ R.t()) if R is not None else Z_hat
            X_hat = batch_inv_normalize_gpu(
                Y_hat.unsqueeze(0), Mu, Std)

        torch.cuda.synchronize()
        t_dec = time.perf_counter() - t0
        dec_peak = torch.cuda.max_memory_allocated(device)

        if i == 0:
            assert np.array_equal(labels_np, dec_np), \
                "rANS encode/decode mismatch!"

        if i >= n_warmup:
            enc_times.append(t_enc)
            dec_times.append(t_dec)
            token_counts.append(N)
            enc_mem_peaks.append(enc_peak)
            dec_mem_peaks.append(dec_peak)

        del X, Y, Mu, Std, X_hat

    return enc_times, dec_times, token_counts, enc_mem_peaks, dec_mem_peaks


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Per-feature encode/decode timing & memory for Soft-PQ codec")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_warmup", type=int, default=10,
                   help="Warmup features excluded from timing")
    args = p.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    all_results = []

    for backbone, layer, D, configs in MODEL_SPECS:
        print(f"\n{'#'*70}")
        print(f"# {backbone}  layer={layer}  D={D}")
        print(f"{'#'*70}")

        # ---- Load cls features ----
        test_dir = Path(FEAT_ROOT) / "test" / backbone / layer
        files = sorted(test_dir.glob("*.npy"))
        if not files:
            print(f"  SKIP: no features in {test_dir}")
            continue
        cls_feats, _ = preload_features(files, num_workers=8)
        print(f"  {len(cls_feats)} test images, "
              f"shape={cls_feats[0].shape}")

        for K, emb, lmbda, lr, epochs, desc in configs:
            G = D // emb
            cp = get_ckpt(backbone, layer, D, K, emb, lmbda, lr, epochs)
            if not os.path.exists(cp):
                print(f"\n  SKIP {desc}: checkpoint missing ({os.path.basename(cp)})")
                continue

            print(f"\n{'='*70}")
            print(f"  [{backbone}/{layer}] {desc}  "
                  f"(K={K}, G={G}, d={emb}, lmbda={lmbda})")
            codec = load_codec(cp, device=device)

            # PMF / CDF
            print(f"    computing labels for PMF...")
            all_labels = get_all_labels_concat(cls_feats, codec, device)
            if codec.pq.use_rate:
                pmf = [codec.pq.get_prior_pmf()[g] for g in range(G)]
            else:
                pmf = hist_pmf(all_labels, G, K)
            cdfs, sizes = prepare_cdfs(pmf, G, K)

            # Timing + Memory
            print(f"    timing {len(cls_feats)} features "
                  f"(warmup={args.n_warmup})...")
            enc_t, dec_t, tok_n, enc_mem, dec_mem = measure_timing(
                cls_feats, codec, device, cdfs, sizes, G, K, emb, D,
                n_warmup=args.n_warmup)

            enc_avg = np.mean(enc_t) * 1000
            dec_avg = np.mean(dec_t) * 1000
            enc_std = np.std(enc_t)  * 1000
            dec_std = np.std(dec_t)  * 1000
            tok_avg = np.mean(tok_n)
            enc_mem_avg = np.mean(enc_mem) / (1024 ** 2)
            dec_mem_avg = np.mean(dec_mem) / (1024 ** 2)
            enc_mem_max = np.max(enc_mem) / (1024 ** 2)
            dec_mem_max = np.max(dec_mem) / (1024 ** 2)

            print(f"    enc {enc_avg:.3f}±{enc_std:.3f} ms  "
                  f"dec {dec_avg:.3f}±{dec_std:.3f} ms  "
                  f"(avg {tok_avg:.0f} tok, n={len(enc_t)})")
            print(f"    enc_mem {enc_mem_avg:.1f} MB (peak {enc_mem_max:.1f})  "
                  f"dec_mem {dec_mem_avg:.1f} MB (peak {dec_mem_max:.1f})")

            all_results.append({
                'backbone': backbone, 'layer': layer, 'D': D,
                'desc': desc, 'K': K, 'emb': emb, 'G': G, 'lmbda': lmbda,
                'enc_ms_mean': round(enc_avg, 4),
                'enc_ms_std':  round(enc_std, 4),
                'dec_ms_mean': round(dec_avg, 4),
                'dec_ms_std':  round(dec_std, 4),
                'avg_tokens':  round(float(tok_avg), 1),
                'n_measured':  len(enc_t),
                'enc_mem_avg_MB': round(float(enc_mem_avg), 2),
                'enc_mem_peak_MB': round(float(enc_mem_max), 2),
                'dec_mem_avg_MB': round(float(dec_mem_avg), 2),
                'dec_mem_peak_MB': round(float(dec_mem_max), 2),
            })

            del codec
            torch.cuda.empty_cache()

    # ---- Summary ----
    print(f"\n{'='*120}")
    print(f"{'Model':>16} {'Layer':>6} {'Config':>10} | "
          f"{'Enc(ms)':>14} {'Dec(ms)':>14} | "
          f"{'EncMem(MB)':>11} {'DecMem(MB)':>11} | "
          f"{'Tokens':>7} {'N':>5}")
    print(f"{'-'*120}")
    for r in all_results:
        print(f"{r['backbone']:>16} {r['layer']:>6} {r['desc']:>10} | "
              f"{r['enc_ms_mean']:>7.3f}±{r['enc_ms_std']:<5.3f}"
              f"{r['dec_ms_mean']:>7.3f}±{r['dec_ms_std']:<5.3f} | "
              f"{r['enc_mem_peak_MB']:>11.1f} {r['dec_mem_peak_MB']:>11.1f} | "
              f"{r['avg_tokens']:>7.0f} {r['n_measured']:>5}")

    out_path = os.path.join(ORFC_ROOT, "timing_multilayer_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
