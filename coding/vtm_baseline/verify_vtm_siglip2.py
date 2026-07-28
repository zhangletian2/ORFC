"""
VTM + SigLIP2 分类验证脚本

每层取 2 个样本，在所有 QP 下做完整 编码→解码→回放→零样本分类，用于：
  1. 检验 QP 取值是否产生合理 BPFP 范围
  2. 验证 VTM + replay 流水线端到端正确性

Usage:
  conda activate siglip_codec
  cd /data4/workspace/zlt/featcodec/ORFC/coding/vtm_baseline
  python verify_vtm_siglip2.py
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
IMAGENET_ROOT = PROJECT_ROOT / "data" / "imagenet" / "images" / "val"
LIST_FILE = Path("/data4/workspace/zlt/featcodec/utils/imagenet_selected_pathname500.txt")
MODEL_ID = "google/siglip2-so400m-patch14-224"

VTM_ENCODER = str(SCRIPT_DIR / "EncoderAppStatic")
VTM_DECODER = str(SCRIPT_DIR / "DecoderAppStatic")
VTM_CFG = str(SCRIPT_DIR / "encoder_intra_vtm.cfg")
BIT_DEPTH = 10

LAYERS = ["blk08", "blk16", "blk24"]
QPS_MAP = {
    "blk08": [0, 5, 10, 12, 17, 22, 25, 27],
    "blk16": [0, 5, 12, 22, 32, 42],
    "blk24": [0, 2, 5, 7, 10],
}
N_SAMPLES = 10
DEVICE = os.environ.get("CUDA_DEVICE", "cuda")
VTM_WORKERS = 10
SEQ_LEN = 256  # (224/14)^2


# ─────────────── VTM 编解码 ───────────────

def quantize_linear(x, bit_depth):
    x = x.astype(np.float32)
    xmin, xmax = float(x.min()), float(x.max())
    if xmax <= xmin:
        scale = 1.0
    else:
        scale = ((1 << bit_depth) - 1) / (xmax - xmin)
    q = np.round((x - xmin) * scale).astype(np.uint16 if bit_depth > 8 else np.uint8)
    return q, {"min": xmin, "max": xmax, "bit_depth": bit_depth}


def dequantize_linear(q, meta):
    q = q.astype(np.float32)
    B = int(meta["bit_depth"])
    xmin, xmax = float(meta["min"]), float(meta["max"])
    if xmax <= xmin:
        return np.full_like(q, xmin, dtype=np.float32)
    scale = ((1 << B) - 1) / (xmax - xmin)
    return (q / scale + xmin).astype(np.float32)


def vtm_encode_decode(feat_2d, qp, tmp_dir):
    H, W = feat_2d.shape
    q, meta = quantize_linear(feat_2d, BIT_DEPTH)

    uid = f"verify_{os.getpid()}_{threading.get_ident()}_{qp}"
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
    feat_rec = dequantize_linear(q_rec, meta)
    bs_bytes = os.path.getsize(bitstream)

    for p in [yuv_path, bitstream, dec_yuv, enc_log, dec_log]:
        try:
            os.unlink(p)
        except OSError:
            pass

    return feat_rec, bs_bytes


# ─────────────── 并行 VTM 任务 ───────────────

def vtm_task(args_tuple):
    layer, qp, idx, name, feat_path, tmp_dir = args_tuple
    feat_orig = np.load(feat_path)
    if feat_orig.ndim == 3 and feat_orig.shape[0] == 1:
        feat_orig = feat_orig[0]
    H, W = feat_orig.shape
    feat_rec, bs_bytes = vtm_encode_decode(feat_orig, qp, tmp_dir)
    bpfp = (bs_bytes * 8) / (H * W)
    feat_mse = float(np.mean((feat_orig - feat_rec) ** 2))
    cos_sim = float(np.dot(feat_orig.flatten(), feat_rec.flatten()) /
                    (np.linalg.norm(feat_orig) * np.linalg.norm(feat_rec) + 1e-12))
    rec_path = os.path.join(tmp_dir, f"{layer}_{qp}_{name}.npy")
    np.save(rec_path, feat_rec)
    return (layer, qp, idx, name, bpfp, feat_mse, cos_sim, rec_path)


# ─────────────── 主流程 ───────────────

def main():
    print("=" * 90)
    print("  VTM + SigLIP2 分类 验证脚本")
    print(f"  每层 {N_SAMPLES} 个样本 × 所有 QP")
    print(f"  模型: {MODEL_ID}")
    print("=" * 90)

    # 加载 label
    labels = {}
    with open(LABEL_FILE) as f:
        for ln in f:
            b, idx = ln.strip().split()
            labels[b] = int(idx)

    # 每层取前 N_SAMPLES 个有标签的样本
    sample_names = []
    feat_files = sorted((FEAT_ROOT / "blk24").glob("*.npy"))
    for fp in feat_files:
        name = fp.stem
        if name in labels:
            sample_names.append(name)
        if len(sample_names) >= N_SAMPLES:
            break
    print(f"  样本: {sample_names}")

    tmp_dir = str(SCRIPT_DIR / "_vtm_verify_siglip2_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # ================================================================
    #  Phase 1: 并行 VTM 编解码（纯 CPU）
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
                layer, qp, idx, name, bpfp, feat_mse, cos_sim, rec_path = future.result()
                vtm_results[(layer, qp, idx)] = {
                    'name': name, 'bpfp': bpfp, 'feat_mse': feat_mse,
                    'cos_sim': cos_sim, 'rec_path': rec_path,
                }
                if completed % 5 == 0 or completed == len(tasks):
                    print(f"  VTM 进度: {completed}/{len(tasks)}")
            except Exception as e:
                task_info = futures[future]
                print(f"  [ERR] {task_info[0]}/QP{task_info[1]}/{task_info[3]}: {e}")

    vtm_elapsed = perf_counter() - t0_all
    print(f"  VTM 编解码完成: {vtm_elapsed:.1f}s")

    # ================================================================
    #  Phase 2: 加载 SigLIP2 模型，Anchor + 回放
    # ================================================================
    print(f"\n[2/3] 加载 SigLIP2 模型，Anchor + 回放评测...")

    import torch
    from siglip2_feat_pipeline import (
        load_siglip2, build_text_emb, continue_from_tokens, compute_logits,
        load_classnames_wnid_format,
    )

    model, processor = load_siglip2(MODEL_ID, DEVICE)
    model.float()
    vision_model = model.vision_model

    text_emb = build_text_emb(model, processor, str(CLASSNAMES_FILE), DEVICE)

    def classify_one(img_emb):
        logits = compute_logits(model, img_emb, text_emb)
        top5_idx = torch.topk(logits, k=5, dim=-1).indices.squeeze(0).tolist()
        return top5_idx

    # Anchor: 原始特征（无压缩）回放
    print("\n  计算 Anchor（无压缩回放）...")
    anchor_results = {}
    for layer in LAYERS:
        layer_idx = int(layer[-2:])
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

    # 回放重建特征
    print("\n  回放 VTM 重建特征...")
    results = {}

    for layer in LAYERS:
        layer_idx = int(layer[-2:])
        qps = QPS_MAP[layer]

        print(f"\n  ─── {layer} (layer_idx={layer_idx}) ───")

        for qp in qps:
            correct1 = correct5 = 0
            bpfp_list, mse_list, cos_list = [], [], []

            for i, name in enumerate(sample_names):
                key = (layer, qp, i)
                if key not in vtm_results:
                    continue
                r = vtm_results[key]
                bpfp_list.append(r['bpfp'])
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
                'feat_mse': np.mean(mse_list),
                'cos_sim': np.mean(cos_list),
                'acc1': correct1 / n * 100,
                'acc5': correct5 / n * 100,
            }
            print(f"    QP={qp:>2}: BPFP={np.mean(bpfp_list):.4f}, "
                  f"CosSim={np.mean(cos_list):.4f}, "
                  f"Acc@1={correct1/n*100:.1f}%, Acc@5={correct5/n*100:.1f}%")

    del model, vision_model
    torch.cuda.empty_cache()

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ================================================================
    #  Phase 3: 汇总报告
    # ================================================================
    print("\n" + "=" * 90)
    print("[3/3] 验证汇总报告")
    print("=" * 90)

    print(f"\n{'Layer':<8} {'QP':>4} {'BPFP':>8} {'FeatMSE':>12} {'CosSim':>8} "
          f"{'Acc@1':>8} {'Acc@5':>8} {'ΔAcc@1':>8}")
    print("-" * 72)

    for layer in LAYERS:
        qps = QPS_MAP[layer]
        anc = anchor_results[layer]
        for qp in qps:
            r = results[(layer, qp)]
            d_acc1 = r['acc1'] - anc['acc1']
            print(f"{layer:<8} {qp:>4} {r['bpfp']:>8.4f} {r['feat_mse']:>12.4f} "
                  f"{r['cos_sim']:>8.4f} {r['acc1']:>7.1f}% {r['acc5']:>7.1f}% "
                  f"{d_acc1:>+7.1f}%")
        print()

    print("-" * 72)
    for layer in LAYERS:
        anc = anchor_results[layer]
        print(f"  Anchor {layer}: Acc@1={anc['acc1']:.1f}%, Acc@5={anc['acc5']:.1f}%")
    print()

    # 判定
    print("─── 验证结论 ───")
    all_bpfps = [results[(l, q)]['bpfp'] for l in LAYERS for q in QPS_MAP[l]]
    bpfp_min, bpfp_max = min(all_bpfps), max(all_bpfps)
    print(f"  1. BPFP 范围: [{bpfp_min:.4f}, {bpfp_max:.4f}]", end="")
    if bpfp_min < 0.1 and bpfp_max > 0.5:
        print("  ✓ 覆盖充分")
    elif bpfp_max > bpfp_min * 2:
        print("  ~ 有一定梯度，可能需微调")
    else:
        print("  ✗ 范围不足，需调整 QP")

    all_cos = [results[(l, q)]['cos_sim'] for l in LAYERS for q in QPS_MAP[l]]
    cos_min = min(all_cos)
    print(f"  2. 最低 CosSim: {cos_min:.4f}", end="")
    if cos_min > 0.99:
        print("  ✓ 高保真（可能 QP 范围偏低）")
    elif cos_min > 0.8:
        print("  ✓ 合理范围")
    else:
        print("  ⚠ 失真较大")

    # 检查各层是否能覆盖目标 BPFP 范围 [0.05, min(1.0, lossless)]
    print(f"  3. 各层 BPFP 覆盖情况 (目标: 0.05 ~ lossless):")
    for layer in LAYERS:
        anc = anchor_results[layer]
        qps = QPS_MAP[layer]
        layer_bpfps = [(q, results[(layer, q)]['bpfp'], results[(layer, q)]['acc1']) for q in qps]
        lossless_bpfp = None
        for q, bp, a1 in layer_bpfps:
            if a1 >= anc['acc1']:
                lossless_bpfp = bp
                break
        bpfp_range = [bp for _, bp, _ in layer_bpfps]
        if lossless_bpfp is None:
            print(f"     {layer}: [{min(bpfp_range):.4f}, {max(bpfp_range):.4f}] "
                  f"⚠ 无 QP 达到无损 (Anchor={anc['acc1']:.1f}%)")
        elif min(bpfp_range) > 0.05:
            print(f"     {layer}: [{min(bpfp_range):.4f}, {max(bpfp_range):.4f}] "
                  f"lossless@BPFP={lossless_bpfp:.4f}  ⚠ 最小BPFP > 0.05")
        else:
            print(f"     {layer}: [{min(bpfp_range):.4f}, {max(bpfp_range):.4f}] "
                  f"lossless@BPFP={lossless_bpfp:.4f}  ✓ 可覆盖")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
