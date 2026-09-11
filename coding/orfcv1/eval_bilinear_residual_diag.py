#!/usr/bin/env python
"""Bilinear residual diagnostics: codec timing vs ORFC, and skip-Q_G Lref.

1. Timing (same protocol as ``eval_haar_orfc_codec_timing.py``):
   VOC val-100 + NYU test-80, per-image encode/decode, rANS on the ORFC
   stream, 2-bit packing on guidance.  No backbone / task head.

2. Fixed-basis no-quant Lref: freeze bilinear main ORFC and ``B0``, drop
   ``Q_G``, evaluate ImageNet Lref once.  Compare to main-only and
   ``B0``+``Q_G`` to split subspace vs quantization.

Usage:
    CUDA_VISIBLE_DEVICES=4 python -u eval_bilinear_residual_diag.py
    CUDA_VISIBLE_DEVICES=4 python -u eval_bilinear_residual_diag.py --jobs timing
    CUDA_VISIBLE_DEVICES=4 python -u eval_bilinear_residual_diag.py --jobs noquant
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ORFC = HERE.parent / "orfc"
PROJECT = HERE.parents[1]
for p in (str(ORFC), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from opq import batch_inv_normalize_gpu, batch_normalize_gpu  # noqa: E402
from soft_pq import FrozenTail, load_codec  # noqa: E402

from bilinear_residual import (  # noqa: E402
    BilinearSpatialCodec, ResidualLinearCodec, freeze_module,
    load_residual_init, pack_two_bit_indices, reconstruct_main,
    residual_from_main, unpack_two_bit_indices, combine_residual,
)
from eval_haar_orfc_codec_timing import (  # noqa: E402
    codebook_decode, find_orfc_ckpt, flatten_slides, load_task_images,
    log2_pmf_cost, measure_orfc, pmf_from_codec, pq_labels, prepare_cdfs,
    rans_pack, rans_unpack, rotation, summarise_image_times,
)
from run_bilinear_residual import (  # noqa: E402
    eval_cascade, init_path, load_features, main_ckpt_path, make_split,
)
from run_multilayer_calibrator import load_gt  # noqa: E402
from backbone.wrapper import Dinov2Wrapper  # noqa: E402


def parse_diag_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--jobs", default="timing,noquant",
                   help="comma: timing, noquant")
    p.add_argument("--layer", default="blk05")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--c", type=int, default=4)
    p.add_argument("--backbone", default="dinov2_vitl14")
    p.add_argument("--n_warmup", type=int, default=5)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--tasks", default="seg,depth")
    p.add_argument("--result_dir", default=str(
        HERE / "results" / "bilinear_residual"))
    p.add_argument("--seg_feat_root",
                   default=str(PROJECT / "features" / "voc2012_100"))
    p.add_argument("--nyu_feat_root",
                   default=str(PROJECT / "features" / "nyu_depth_80"
                               / "dinov2_vitl14"))
    return p.parse_args()


def train_ns(diag):
    return SimpleNamespace(
        layer=diag.layer, K=diag.K, c=diag.c, backbone=diag.backbone,
        embedding_dim=32, bottleneck_dim=1024, lmbda=0.5, tau_start=2.0,
        tau_end=2.0, tau_schedule="constant", lr=3e-4, residual_lr=3e-4,
        epochs=100, residual_epochs=100, max_train_images=5000, n_val=200,
        seed=42, warm_start_opq=True, ckpt_dir=str(HERE / "checkpoints"),
        result_dir=str(HERE / "results" / "bilinear_residual"),
        main_ckpt="", init_ckpt="", residual_ckpt="", scale=2,
        feat_root=str(PROJECT / "features"), train_subset="train",
        test_subset="test",
        gt_path=str(PROJECT / "utils" / "imagenet_selected_label500.txt"),
        batch_size=32, norm_mode="per_image", skip_test_acc=False,
    )


def load_bili_stack(ns, device, residual=True):
    main_ckpt = main_ckpt_path(ns)
    if not Path(main_ckpt).is_file():
        raise FileNotFoundError(main_ckpt)
    orfc = freeze_module(load_codec(str(main_ckpt), device=device))
    D = int(getattr(orfc.transform, "D", 1024))
    spatial = freeze_module(
        BilinearSpatialCodec(D, n_prefix=1, scale=ns.scale).to(device))
    res = None
    init_ckpt = None
    if residual:
        init_ckpt = init_path(ns)
        B0, scale, _ = load_residual_init(str(init_ckpt), device="cpu")
        res = ResidualLinearCodec(D, ns.c, B0=B0, scale=scale).to(device)
        res.quant.set_scale(scale.to(device))
        res = freeze_module(res)
    return spatial, orfc, res, main_ckpt, init_ckpt


def guidance_hat_from_q(q, scale):
    q_t = torch.from_numpy(np.asarray(q, dtype=np.int64)).to(scale.device)
    return scale.to(dtype=torch.float32) * (q_t.float() - 1.5)


@torch.no_grad()
def measure_bilinear(slides, owners, spatial, orfc, residual, device,
                     cdfs, sizes, n_warmup_images):
    pq = orfc.pq
    C, G, K, d = pq.codebooks, pq.G, pq.K, pq.d
    D = C.shape[0] * d
    Rmat = rotation(orfc)
    cost = log2_pmf_cost(pq)
    lmbda = pq.lmbda if pq.use_rate else None
    n_prefix = spatial.n_prefix
    use_res = residual is not None
    enc_parts = {"gpu": [], "rans": []}
    dec_parts = {"rans": [], "gpu": []}
    if use_res:
        enc_parts["guidance"] = []
        dec_parts["guidance"] = []
    enc = {"t": [], "tok": [], "parts": enc_parts}
    dec = {"t": [], "parts": dec_parts}
    checked = False
    scale = residual.quant.scale if use_res else None
    for feat in slides:
        N = feat.shape[0]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
        Y, Mu, Std = batch_normalize_gpu(X, mode="per_image")
        seq, aux = spatial.encode(Y)
        Tm = seq.shape[1]
        flat = seq.reshape(-1, D)
        Z = flat if Rmat is None else flat @ Rmat
        labels = pq_labels(Z, C, d, G, cost, lmbda)
        labels_np = labels.cpu().numpy()
        seq_hat = codebook_decode(labels, C, d, Rmat, D).view(1, Tm, D)
        Y0 = spatial.decode(seq_hat, aux)
        q_np = None
        if use_res:
            R = residual_from_main(Y, Y0, n_prefix=n_prefix)
            Gcoeff = residual.encode_residual(R)
            _, q = residual.quant(Gcoeff)
            q_np = q.cpu().numpy().astype(np.uint8)
        torch.cuda.synchronize()
        t_gpu = time.perf_counter() - t0
        bitstream, idx_list = rans_pack(labels_np, cdfs, sizes, G)
        t_rans = time.perf_counter()
        gbuf = pack_two_bit_indices(q_np) if use_res else b""
        t_enc = time.perf_counter() - t0
        t1 = time.perf_counter()
        dec_np = rans_unpack(bitstream, idx_list, cdfs, sizes, G, Tm)
        t_rans_dec = time.perf_counter() - t1
        q_rec = unpack_two_bit_indices(gbuf, q_np.shape) if use_res else None
        t_gdec = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        dec_labels = torch.from_numpy(dec_np).to(device)
        seq_hat_d = codebook_decode(dec_labels, C, d, Rmat, D).view(1, Tm, D)
        Y0_d = spatial.decode(seq_hat_d, aux)
        if use_res:
            G_hat = guidance_hat_from_q(q_rec, scale)
            R_hat = residual.decode_residual(G_hat, Y0_d[:, n_prefix:])
            Y_hat = combine_residual(Y0_d, R_hat, n_prefix=n_prefix)
        else:
            Y_hat = Y0_d
        X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        torch.cuda.synchronize()
        t_gpu_dec = time.perf_counter() - t2
        t_dec = time.perf_counter() - t1
        if not checked:
            if not np.array_equal(labels_np, dec_np):
                raise RuntimeError("bilinear ORFC rANS round-trip mismatch")
            if use_res and not np.array_equal(q_np, q_rec):
                raise RuntimeError("guidance 2-bit round-trip mismatch")
            checked = True
        enc["t"].append(t_enc)
        enc["tok"].append(Tm)
        enc["parts"]["gpu"].append(t_gpu)
        enc["parts"]["rans"].append(t_rans - t0 - t_gpu)
        if use_res:
            enc["parts"]["guidance"].append(t_enc - (t_rans - t0))
        dec["t"].append(t_dec)
        dec["parts"]["rans"].append(t_rans_dec)
        if use_res:
            dec["parts"]["guidance"].append(t_gdec - t1 - t_rans_dec)
            dec["parts"]["gpu"].append(t_gpu_dec)
        else:
            dec["parts"]["gpu"].append(t_gpu_dec)
        del X, Y, Mu, Std, X_hat, seq, Y0
    out = summarise_image_times(enc, dec, owners, n_warmup_images)
    out["avg_coded_tokens_per_image"] = out.pop("avg_tokens_per_image")
    keep_src = [sl.shape[0] for sl, own in zip(slides, owners)
                if own >= n_warmup_images]
    out["avg_src_tokens_per_image"] = round(
        float(np.mean(keep_src) * (out["n_slides_per_image"] or 1)), 1)
    out["guidance_bits_per_src_patch"] = 8 if use_res else 0
    return out


def run_timing(diag, device):
    ns = train_ns(diag)
    spatial, orfc_b, residual, main_ckpt, init_ckpt = load_bili_stack(
        ns, device, residual=True)
    orfc_plain_path = find_orfc_ckpt(diag.layer, diag.K, SimpleNamespace(
        orfc_ckpt_dir=str(ORFC / "checkpoints"), backbone=diag.backbone))
    if orfc_plain_path is None:
        raise FileNotFoundError("original ORFC checkpoint")
    orfc_plain = freeze_module(load_codec(str(orfc_plain_path), device=device))

    pmf_b = pmf_from_codec(orfc_b)
    if pmf_b is None:
        raise RuntimeError("bilinear ORFC has no learned prior")
    cdfs_b, sizes_b = prepare_cdfs(pmf_b, orfc_b.pq.G, orfc_b.pq.K)
    pmf_o = pmf_from_codec(orfc_plain)
    if pmf_o is None:
        raise RuntimeError("original ORFC has no learned prior")
    cdfs_o, sizes_o = prepare_cdfs(pmf_o, orfc_plain.pq.G, orfc_plain.pq.K)

    tasks = [x.strip() for x in diag.tasks.split(",") if x.strip()]
    rows = []
    print(f"\n{'#' * 70}")
    print(f"# bilinear vs ORFC timing  layer={diag.layer}  K={diag.K}")
    print(f"# warmup_images={diag.n_warmup}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}", flush=True)

    codec_specs = [
        ("orfc", "ORFC 257"),
        ("bili_main", "bilinear main"),
        ("bili_fixed", "bilinear+B0+QG"),
    ]
    for task in tasks:
        images = load_task_images(task, diag.layer, diag)
        slides, owners = flatten_slides(images)
        print(f"\n  {diag.layer} {task}: {len(images)} images, "
              f"{len(slides)} slides, shape={slides[0].shape}", flush=True)
        for codec_id, label in codec_specs:
            t0 = time.time()
            if codec_id == "orfc":
                stats = measure_orfc(
                    slides, owners, orfc_plain, device, cdfs_o, sizes_o,
                    diag.n_warmup)
            elif codec_id == "bili_main":
                stats = measure_bilinear(
                    slides, owners, spatial, orfc_b, None, device,
                    cdfs_b, sizes_b, diag.n_warmup)
            else:
                stats = measure_bilinear(
                    slides, owners, spatial, orfc_b, residual, device,
                    cdfs_b, sizes_b, diag.n_warmup)
            stats.update({
                "codec": codec_id, "label": label,
                "layer": diag.layer, "K": diag.K, "task": task,
                "orfc_ckpt": (str(orfc_plain_path) if codec_id == "orfc"
                              else str(main_ckpt)),
                "residual_init": None if codec_id != "bili_fixed" else str(init_ckpt),
                "elapsed_s": round(time.time() - t0, 2),
            })
            rows.append(stats)
            coded = stats.get("avg_coded_tokens_per_image",
                              stats.get("avg_tokens_per_image"))
            print(f"    {label:18s}  enc {stats['enc_ms_mean']:.2f}±"
                  f"{stats['enc_ms_std']:.2f} ms  dec {stats['dec_ms_mean']:.2f}±"
                  f"{stats['dec_ms_std']:.2f} ms  coded {coded}",
                  flush=True)

    out_dir = Path(diag.result_dir) / "timing"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{diag.layer}_K{diag.K}_c{diag.c}_timing.json"
    payload = {
        "layer": diag.layer, "K": diag.K, "c": diag.c,
        "n_warmup_images": diag.n_warmup,
        "unit": "ms per image (sum of slides); ORFC rANS included; "
                "guidance 2-bit pack included; backbone/task excluded",
        "rows": rows,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out_path}", flush=True)
    del spatial, orfc_b, residual, orfc_plain
    torch.cuda.empty_cache()
    return payload


def run_noquant(diag, device):
    ns = train_ns(diag)
    spatial, orfc, residual, main_ckpt, init_ckpt = load_bili_stack(
        ns, device, residual=True)
    features_train, features_test, basenames_test = load_features(ns)
    gt_test = load_gt(ns.gt_path)
    train_feat, val_feat, _, _ = make_split(features_train, ns)
    layer_idx = int(diag.layer[-2:])
    print(f"\n{'#' * 70}")
    print(f"# skip-Q_G Lref  layer={diag.layer}  freeze main + B0")
    print(f"# main {main_ckpt}")
    print(f"# B0   {init_ckpt}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}", flush=True)

    print(f"\nLoading {ns.backbone}...", flush=True)
    wrapper = Dinov2Wrapper(head_layers=1, model_name=ns.backbone, device=device)
    tail = FrozenTail(
        list(wrapper.backbone.blocks[layer_idx + 1:]),
        wrapper.backbone.norm, device=device)

    specs = [
        ("main", None, True, False),
        ("B0_no_QG", residual, False, False),
        ("B0_QG", residual, True, False),
    ]
    results = {
        "layer": diag.layer,
        "main_ckpt": str(main_ckpt),
        "init_ckpt": str(init_ckpt),
        "n_val": len(val_feat),
        "n_test": len(features_test),
        "rows": {},
    }
    for name, res, quantize, _ in specs:
        print(f"\n  --- {name}  quantize={quantize} ---", flush=True)
        row = {}
        row["val"] = eval_cascade(
            f"{name} val", val_feat, spatial, orfc, res, tail,
            ns.norm_mode, device, ns.batch_size, quantize=quantize)
        row["test"] = eval_cascade(
            f"{name} test", features_test, spatial, orfc, res, tail,
            ns.norm_mode, device, ns.batch_size,
            basenames=basenames_test, gt=gt_test, wrapper=wrapper,
            layer_idx=layer_idx, quantize=quantize)
        results["rows"][name] = row

    main_t = results["rows"]["main"]["test"]["delta_l"]
    noq_t = results["rows"]["B0_no_QG"]["test"]["delta_l"]
    qg_t = results["rows"]["B0_QG"]["test"]["delta_l"]
    results["delta"] = {
        "test_B0_minus_main": noq_t - main_t,
        "test_QG_minus_B0": qg_t - noq_t,
        "test_QG_minus_main": qg_t - main_t,
        "val_B0_minus_main": (
            results["rows"]["B0_no_QG"]["val"]["delta_l"]
            - results["rows"]["main"]["val"]["delta_l"]),
        "val_QG_minus_B0": (
            results["rows"]["B0_QG"]["val"]["delta_l"]
            - results["rows"]["B0_no_QG"]["val"]["delta_l"]),
    }
    print("\n  split:  ΔL(B0 no QG) − ΔL(main)  = subspace", flush=True)
    print("          ΔL(B0+QG) − ΔL(B0 no QG) = Q_G", flush=True)
    print(f"  test subspace {results['delta']['test_B0_minus_main']:+.1f}  "
          f"Q_G {results['delta']['test_QG_minus_B0']:+.1f}  "
          f"total {results['delta']['test_QG_minus_main']:+.1f}", flush=True)

    out_dir = Path(diag.result_dir) / ns.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{diag.layer}_bili65_K{diag.K}_c{diag.c}_noquant_lref.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {out_path}", flush=True)
    del wrapper, tail
    torch.cuda.empty_cache()
    return results


def main():
    diag = parse_diag_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jobs = [x.strip() for x in diag.jobs.split(",") if x.strip()]
    if "timing" in jobs:
        run_timing(diag, device)
    if "noquant" in jobs:
        run_noquant(diag, device)


if __name__ == "__main__":
    main()
