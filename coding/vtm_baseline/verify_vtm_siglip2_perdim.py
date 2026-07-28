"""
VTM + SigLIP2 分类验证 — Per-Dim Quantization

用 per-dim（每维度独立 min/max）量化替换全局线性量化，
验证能否解决 blk15/23 的 row-176 极端值导致的分类崩溃问题。

Side-info 开销: 1152 dims × 2 (min/max) × 16 bit (fp16) = 0.125 BPFP

Usage:
  conda activate siglip_codec
  cd /data4/workspace/zlt/featcodec/ORFC/coding/vtm_baseline
  python verify_vtm_siglip2_perdim.py
"""

import os, sys, subprocess, shutil, threading
import numpy as np
from pathlib import Path
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

# ─────────────── 配置 ───────────────
FEAT_ROOT = PROJECT_ROOT / "features" / "test" / "siglip2_so400m"
LABEL_FILE = Path("/data4/workspace/zlt/featcodec/utils/imagenet_selected_label500.txt")
CLASSNAMES_FILE = PROJECT_ROOT / "utils" / "classnames.txt"
MODEL_ID = "google/siglip2-so400m-patch14-224"

VTM_ENCODER = str(SCRIPT_DIR / "EncoderAppStatic")
VTM_DECODER = str(SCRIPT_DIR / "DecoderAppStatic")
VTM_CFG = str(SCRIPT_DIR / "encoder_intra_vtm.cfg")
BIT_DEPTH = 10

LAYERS = ["blk07", "blk15", "blk23"]
QPS_MAP = {
    "blk07": [0, 5, 10, 12, 17, 22, 25, 27],
    "blk15": [0, 5, 12, 22, 32, 42],
    "blk23": [0, 2, 5, 7, 10, 12, 17],
}
N_SAMPLES = 10
DEVICE = os.environ.get("CUDA_DEVICE", "cuda")
VTM_WORKERS = 10
SEQ_LEN = 256  # (224/14)^2


# ─────────────── Per-Dim 量化 ───────────────

def quantize_per_dim(x, bit_depth):
    """每列（维度）独立 min/max 线性量化。
    x: [H, W]  →  q: [H, W] uint16, meta dict
    """
    x = x.astype(np.float32)
    H, W = x.shape
    max_val = (1 << bit_depth) - 1

    col_min = x.min(axis=0)   # [W]
    col_max = x.max(axis=0)   # [W]
    col_range = col_max - col_min
    col_range[col_range <= 0] = 1.0
    scale = max_val / col_range  # [W]

    q = np.round((x - col_min[np.newaxis, :]) * scale[np.newaxis, :])
    q = np.clip(q, 0, max_val).astype(np.uint16 if bit_depth > 8 else np.uint8)

    meta = {
        "col_min": col_min.astype(np.float16),
        "col_max": col_max.astype(np.float16),
        "bit_depth": bit_depth,
    }
    return q, meta


def dequantize_per_dim(q, meta):
    """Per-dim 反量化。"""
    q = q.astype(np.float32)
    B = int(meta["bit_depth"])
    max_val = (1 << B) - 1
    col_min = meta["col_min"].astype(np.float32)
    col_max = meta["col_max"].astype(np.float32)
    col_range = col_max - col_min
    col_range[col_range <= 0] = 1.0
    scale = max_val / col_range

    return (q / scale[np.newaxis, :] + col_min[np.newaxis, :]).astype(np.float32)


def per_dim_side_info_bits(W):
    return W * 2 * 16


# ─────────────── VTM 编解码 ───────────────

def vtm_encode_decode(feat_2d, qp, tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)
    H, W = feat_2d.shape
    q, meta = quantize_per_dim(feat_2d, BIT_DEPTH)

    uid = f"vpdcls_{os.getpid()}_{threading.get_ident()}_{qp}"
    yuv_path = os.path.join(tmp_dir, f"{uid}.y")
    bitstream = os.path.join(tmp_dir, f"{uid}.vvc")
    dec_yuv = os.path.join(tmp_dir, f"{uid}.dec.y")
    enc_log = os.path.join(tmp_dir, f"{uid}.enc.log")
    dec_log = os.path.join(tmp_dir, f"{uid}.dec.log")

    q.tofile(yuv_path)

    cmd_enc = [
        VTM_ENCODER, "-c", VTM_CFG,
        "-i", yuv_path, "-b", bitstream,
        f"--SourceWidth={W}", f"--SourceHeight={H}",
        "--FramesToBeEncoded=1", "--FrameRate=1",
        "--InputChromaFormat=400", "--ConformanceWindowMode=1",
        f"--InternalBitDepth={BIT_DEPTH}",
        f"--InputBitDepth={BIT_DEPTH}",
        f"--OutputBitDepth={BIT_DEPTH}",
        f"--QP={qp}",
    ]
    with open(enc_log, "w") as f:
        subprocess.run(cmd_enc, stdout=f, stderr=subprocess.STDOUT, check=True)

    cmd_dec = [VTM_DECODER, "-b", bitstream, "-o", dec_yuv]
    with open(dec_log, "w") as f:
        subprocess.run(cmd_dec, stdout=f, stderr=subprocess.STDOUT, check=True)

    dt = np.uint16 if BIT_DEPTH > 8 else np.uint8
    q_rec = np.fromfile(dec_yuv, dtype=dt).reshape(H, W)
    feat_rec = dequantize_per_dim(q_rec, meta)

    vtm_bytes = os.path.getsize(bitstream)
    side_bits = per_dim_side_info_bits(W)
    total_bits = vtm_bytes * 8 + side_bits

    for p in [yuv_path, bitstream, dec_yuv, enc_log, dec_log]:
        try:
            os.unlink(p)
        except OSError:
            pass

    return feat_rec, vtm_bytes, side_bits


# ─────────────── 并行 VTM 任务 ───────────────

def vtm_task(args_tuple):
    layer, qp, idx, name, feat_path, tmp_dir = args_tuple
    feat_orig = np.load(feat_path)
    if feat_orig.ndim == 3 and feat_orig.shape[0] == 1:
        feat_orig = feat_orig[0]
    H, W = feat_orig.shape
    feat_rec, vtm_bytes, side_bits = vtm_encode_decode(feat_orig, qp, tmp_dir)
    total_bits = vtm_bytes * 8 + side_bits
    bpfp = total_bits / (H * W)
    bpfp_vtm = (vtm_bytes * 8) / (H * W)
    bpfp_side = side_bits / (H * W)
    feat_mse = float(np.mean((feat_orig - feat_rec) ** 2))
    cos_sim = float(np.dot(feat_orig.flatten(), feat_rec.flatten()) /
                    (np.linalg.norm(feat_orig) * np.linalg.norm(feat_rec) + 1e-12))
    rec_path = os.path.join(tmp_dir, f"{layer}_{qp}_{name}.npy")
    np.save(rec_path, feat_rec)
    return (layer, qp, idx, name, bpfp, bpfp_vtm, bpfp_side,
            feat_mse, cos_sim, rec_path)


# ─────────────── 主流程 ───────────────

def main():
    print("=" * 90)
    print("  VTM + SigLIP2 分类 — Per-Dim Quantization 验证")
    print(f"  每层 {N_SAMPLES} 个样本 × 所有 QP")
    print(f"  模型: {MODEL_ID}")
    print(f"  BitDepth={BIT_DEPTH}, 量化方式: per-dim (每维独立 min/max, side-info fp16)")
    print("=" * 90)

    labels = {}
    with open(LABEL_FILE) as f:
        for ln in f:
            b, idx = ln.strip().split()
            labels[b] = int(idx)

    sample_names = []
    ref_layer = LAYERS[-1]
    feat_files = sorted((FEAT_ROOT / ref_layer).glob("*.npy"))
    for fp in feat_files:
        name = fp.stem
        if name in labels:
            sample_names.append(name)
        if len(sample_names) >= N_SAMPLES:
            break
    print(f"  样本: {sample_names}")

    side_bpfp = per_dim_side_info_bits(1152) / (256 * 1152)
    print(f"  Side-info BPFP 开销: {side_bpfp:.4f}")

    # 对比: per-dim vs 全局量化的有效精度
    print("\n  ─── Per-dim vs 全局量化精度对比 ───")
    for layer in LAYERS:
        feat = np.load(str(FEAT_ROOT / layer / f"{sample_names[0]}.npy"))
        flat = feat.flatten()
        col_ranges = feat.max(axis=0) - feat.min(axis=0)
        global_range = flat.max() - flat.min()
        global_step = global_range / 1023
        perdim_steps = col_ranges / 1023
        print(f"  {layer}:")
        print(f"    全局: range={global_range:.2f}, step={global_step:.4f}")
        print(f"    per-dim: range median={np.median(col_ranges):.2f}, "
              f"step median={np.median(perdim_steps):.6f}, "
              f"step max={perdim_steps.max():.6f}")
        print(f"    精度提升: step 缩小 {global_step / np.median(perdim_steps):.1f}x")

    tmp_dir = str(SCRIPT_DIR / "_vtm_verify_perdim_cls_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # ================================================================
    #  Phase 1: 并行 VTM 编解码
    # ================================================================
    print(f"\n[1/3] VTM 编解码 (workers={VTM_WORKERS})...")

    tasks = []
    for layer in LAYERS:
        qps = QPS_MAP[layer]
        for qp in qps:
            for i, name in enumerate(sample_names):
                feat_path = str(FEAT_ROOT / layer / f"{name}.npy")
                tasks.append((layer, qp, i, name, feat_path, tmp_dir))

    print(f"  总任务数: {len(tasks)} ({len(LAYERS)} layers × QPs × {N_SAMPLES} samples)")
    t0_all = perf_counter()

    vtm_results = {}
    completed = 0
    with ProcessPoolExecutor(max_workers=VTM_WORKERS) as executor:
        futures = {executor.submit(vtm_task, t): t for t in tasks}
        for future in as_completed(futures):
            completed += 1
            try:
                (layer, qp, idx, name, bpfp, bpfp_vtm, bpfp_side,
                 feat_mse, cos_sim, rec_path) = future.result()
                vtm_results[(layer, qp, idx)] = {
                    'name': name, 'bpfp': bpfp,
                    'bpfp_vtm': bpfp_vtm, 'bpfp_side': bpfp_side,
                    'feat_mse': feat_mse,
                    'cos_sim': cos_sim, 'rec_path': rec_path,
                }
                if completed % 10 == 0 or completed == len(tasks):
                    print(f"  VTM 进度: {completed}/{len(tasks)}")
            except Exception as e:
                task_info = futures[future]
                print(f"  [ERR] {task_info[0]}/QP{task_info[1]}/{task_info[3]}: {e}")

    vtm_elapsed = perf_counter() - t0_all
    print(f"  VTM 编解码完成: {vtm_elapsed:.1f}s")

    # ================================================================
    #  Phase 2: 加载模型，Anchor + 回放
    # ================================================================
    print(f"\n[2/3] 加载 SigLIP2 模型，Anchor + 回放评测...")

    import torch
    from siglip2_feat_pipeline import (
        load_siglip2, build_text_emb, continue_from_tokens, compute_logits,
    )

    model, processor = load_siglip2(MODEL_ID, DEVICE)
    model.float()
    vision_model = model.vision_model

    text_emb = build_text_emb(model, processor, str(CLASSNAMES_FILE), DEVICE)

    def classify_one(img_emb):
        logits = compute_logits(model, img_emb, text_emb)
        top5_idx = torch.topk(logits, k=5, dim=-1).indices.squeeze(0).tolist()
        return top5_idx

    # Anchor
    print("\n  计算 Anchor（无压缩回放）...")
    anchor_results = {}
    for layer in LAYERS:
        layer_idx = int(layer.replace("blk", ""))
        correct1 = correct5 = 0
        for name in sample_names:
            feat = np.load(str(FEAT_ROOT / layer / f"{name}.npy"))
            tok = torch.from_numpy(feat).unsqueeze(0).to(DEVICE, dtype=torch.float32)
            with torch.no_grad():
                img_emb = continue_from_tokens(vision_model, tok, layer_idx)
                top5 = classify_one(img_emb)
            gt = labels[name]
            if top5[0] == gt:
                correct1 += 1
            if gt in top5:
                correct5 += 1
        anchor_results[layer] = {
            'acc1': correct1 / len(sample_names) * 100,
            'acc5': correct5 / len(sample_names) * 100,
        }
        print(f"    {layer}: Acc@1={anchor_results[layer]['acc1']:.1f}%, "
              f"Acc@5={anchor_results[layer]['acc5']:.1f}%")

    # 纯量化（无 VTM）基线
    print("\n  纯 per-dim 量化（无 VTM）→ 回放分类...")
    qonly_results = {}
    for layer in LAYERS:
        layer_idx = int(layer.replace("blk", ""))
        correct1 = correct5 = 0
        cos_list = []
        for name in sample_names:
            feat = np.load(str(FEAT_ROOT / layer / f"{name}.npy"))
            q, meta = quantize_per_dim(feat, BIT_DEPTH)
            feat_q = dequantize_per_dim(q, meta)
            cos = np.dot(feat.flatten(), feat_q.flatten()) / (
                np.linalg.norm(feat) * np.linalg.norm(feat_q) + 1e-12)
            cos_list.append(cos)

            tok = torch.from_numpy(feat_q).unsqueeze(0).to(DEVICE, dtype=torch.float32)
            with torch.no_grad():
                img_emb = continue_from_tokens(vision_model, tok, layer_idx)
                top5 = classify_one(img_emb)
            gt = labels[name]
            if top5[0] == gt:
                correct1 += 1
            if gt in top5:
                correct5 += 1
        qonly_results[layer] = {
            'acc1': correct1 / len(sample_names) * 100,
            'acc5': correct5 / len(sample_names) * 100,
            'cos': np.mean(cos_list),
        }
        anc = anchor_results[layer]
        print(f"    {layer}: Acc@1={qonly_results[layer]['acc1']:.1f}% "
              f"(Δ={qonly_results[layer]['acc1']-anc['acc1']:+.1f}%), "
              f"Acc@5={qonly_results[layer]['acc5']:.1f}%, "
              f"CosSim={np.mean(cos_list):.8f}")

    # VTM 重建回放
    print("\n  回放 VTM 重建特征...")
    results = {}

    for layer in LAYERS:
        layer_idx = int(layer.replace("blk", ""))
        qps = QPS_MAP[layer]

        print(f"\n  ─── {layer} (layer_idx={layer_idx}) ───")

        for qp in qps:
            correct1 = correct5 = 0
            bpfp_list, bpfp_vtm_list, mse_list, cos_list = [], [], [], []

            for i, name in enumerate(sample_names):
                key = (layer, qp, i)
                if key not in vtm_results:
                    continue
                r = vtm_results[key]
                bpfp_list.append(r['bpfp'])
                bpfp_vtm_list.append(r['bpfp_vtm'])
                mse_list.append(r['feat_mse'])
                cos_list.append(r['cos_sim'])

                feat_rec = np.load(r['rec_path'])
                tok = torch.from_numpy(feat_rec).unsqueeze(0).to(DEVICE, dtype=torch.float32)
                with torch.no_grad():
                    img_emb = continue_from_tokens(vision_model, tok, layer_idx)
                    top5 = classify_one(img_emb)

                gt = labels[name]
                if top5[0] == gt:
                    correct1 += 1
                if gt in top5:
                    correct5 += 1

            n = len(sample_names)
            results[(layer, qp)] = {
                'bpfp': np.mean(bpfp_list),
                'bpfp_vtm': np.mean(bpfp_vtm_list),
                'feat_mse': np.mean(mse_list),
                'cos_sim': np.mean(cos_list),
                'acc1': correct1 / n * 100,
                'acc5': correct5 / n * 100,
            }
            print(f"    QP={qp:>2}: BPFP={np.mean(bpfp_list):.4f} "
                  f"(vtm={np.mean(bpfp_vtm_list):.4f}), "
                  f"CosSim={np.mean(cos_list):.6f}, "
                  f"Acc@1={correct1/n*100:.1f}%, Acc@5={correct5/n*100:.1f}%")

    del model, vision_model
    torch.cuda.empty_cache()

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ================================================================
    #  Phase 3: 汇总报告
    # ================================================================
    print("\n" + "=" * 100)
    print("[3/3] Per-Dim Quantization 分类验证汇总")
    print("=" * 100)

    side_bpfp = per_dim_side_info_bits(1152) / (256 * 1152)

    print(f"\n{'Layer':<8} {'QP':>4} {'BPFP':>8} {'VTM':>8} {'Side':>6} "
          f"{'FeatMSE':>10} {'CosSim':>8} "
          f"{'Acc@1':>8} {'Acc@5':>8} {'ΔAcc@1':>8}")
    print("-" * 88)

    for layer in LAYERS:
        qps = QPS_MAP[layer]
        anc = anchor_results[layer]
        qo = qonly_results[layer]

        print(f"  {layer} Anchor:  Acc@1={anc['acc1']:.1f}%, Acc@5={anc['acc5']:.1f}%")
        print(f"  {layer} QOnly:   Acc@1={qo['acc1']:.1f}%, Acc@5={qo['acc5']:.1f}%, "
              f"CosSim={qo['cos']:.8f}")

        for qp in qps:
            r = results[(layer, qp)]
            d_acc1 = r['acc1'] - anc['acc1']
            print(f"{layer:<8} {qp:>4} {r['bpfp']:>8.4f} {r['bpfp_vtm']:>8.4f} "
                  f"{r['bpfp']-r['bpfp_vtm']:>6.3f} "
                  f"{r['feat_mse']:>10.6f} {r['cos_sim']:>8.6f} "
                  f"{r['acc1']:>7.1f}% {r['acc5']:>7.1f}% "
                  f"{d_acc1:>+7.1f}%")
        print()

    # 验证结论
    print("─── 验证结论 ───")
    print(f"  Per-dim side-info 开销: {side_bpfp:.4f} BPFP\n")

    print("  1. Per-dim 量化是否解决分类崩溃:")
    for layer in LAYERS:
        anc = anchor_results[layer]
        qo = qonly_results[layer]
        qp0 = results.get((layer, 0))
        q_drop = qo['acc1'] - anc['acc1']
        vtm_drop = qp0['acc1'] - anc['acc1'] if qp0 else float('nan')
        q_ok = "PASS" if abs(q_drop) < 5 else "FAIL"
        v_ok = "PASS" if abs(vtm_drop) < 5 else "FAIL"
        print(f"     {layer}: 纯量化 Δ={q_drop:+.1f}% [{q_ok}]  "
              f"VTM QP=0 Δ={vtm_drop:+.1f}% [{v_ok}]")

    print("\n  2. 各层 BPFP 覆盖 (目标: 0.05 ~ lossless):")
    for layer in LAYERS:
        anc = anchor_results[layer]
        qps = QPS_MAP[layer]
        layer_data = [(q, results[(layer, q)]['bpfp'], results[(layer, q)]['acc1']) for q in qps]
        lossless_bpfp = None
        for q, bp, a1 in layer_data:
            if a1 >= anc['acc1']:
                lossless_bpfp = bp
                break
        bpfps = [bp for _, bp, _ in layer_data]
        if lossless_bpfp is None:
            print(f"     {layer}: [{min(bpfps):.4f}, {max(bpfps):.4f}]  "
                  f"⚠ 无 QP 达到无损 (Anchor={anc['acc1']:.1f}%)")
        else:
            cov = "OK" if min(bpfps) <= 0.06 else "最小BPFP > 0.05"
            print(f"     {layer}: [{min(bpfps):.4f}, {max(bpfps):.4f}]  "
                  f"lossless@BPFP={lossless_bpfp:.4f}  {cov}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
