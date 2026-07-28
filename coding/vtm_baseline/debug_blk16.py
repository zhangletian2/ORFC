"""
blk16 VTM 失败根因排查：逐环节隔离诊断

流水线环节：
  原始特征 ──①量化──> 整数 ──②VTM编码/解码──> 重建整数 ──③反量化──> 重建浮点 ──④回放──> 分类

本脚本对每个环节独立测量误差，定位问题出在哪里。
"""

import os, sys, subprocess, threading
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

FEAT_ROOT = PROJECT_ROOT / "features" / "test" / "siglip2_so400m"
LABEL_FILE = Path("/data4/workspace/zlt/featcodec/utils/imagenet_selected_label500.txt")
CLASSNAMES_FILE = PROJECT_ROOT / "utils" / "classnames.txt"
MODEL_ID = "google/siglip2-so400m-patch14-224"

VTM_ENCODER = str(SCRIPT_DIR / "EncoderAppStatic")
VTM_DECODER = str(SCRIPT_DIR / "DecoderAppStatic")
VTM_CFG = str(SCRIPT_DIR / "encoder_intra_vtm.cfg")
BIT_DEPTH = 10

LAYERS = ["blk07", "blk15", "blk23"]
N_SAMPLES = 10
DEVICE = "cuda"


def load_labels():
    labels = {}
    with open(LABEL_FILE) as f:
        for ln in f:
            b, idx = ln.strip().split()
            labels[b] = int(idx)
    return labels


def get_sample_names(labels, n=N_SAMPLES):
    names = []
    for fp in sorted((FEAT_ROOT / "blk23").glob("*.npy")):
        if fp.stem in labels:
            names.append(fp.stem)
        if len(names) >= n:
            break
    return names


# ═══════════════════════════════════════════════════════
#  第1步：特征分布分析
# ═══════════════════════════════════════════════════════

def analyze_distribution(sample_names):
    print("\n" + "=" * 90)
    print("  第1步：特征分布分析")
    print("=" * 90)

    for layer in LAYERS:
        print(f"\n  ─── {layer} ───")
        all_feats = []
        for name in sample_names:
            feat = np.load(str(FEAT_ROOT / layer / f"{name}.npy"))
            all_feats.append(feat)

        row_max_per_sample = []
        row_std_per_sample = []
        for feat in all_feats:
            row_max_per_sample.append(np.max(np.abs(feat), axis=1))
            row_std_per_sample.append(np.std(feat, axis=1))

        stacked = np.stack(all_feats)  # [N_SAMPLES, 256, 1152]
        flat = stacked.flatten()

        print(f"    全局统计:")
        print(f"      shape per sample: {all_feats[0].shape}")
        print(f"      min={flat.min():.4f}, max={flat.max():.4f}, range={flat.max()-flat.min():.4f}")
        print(f"      mean={flat.mean():.4f}, std={flat.std():.4f}")
        print(f"      P0.01={np.percentile(flat,0.01):.4f}, P1={np.percentile(flat,1):.4f}, "
              f"P50={np.percentile(flat,50):.4f}, P99={np.percentile(flat,99):.4f}, "
              f"P99.99={np.percentile(flat,99.99):.4f}")

        row_absmax = np.stack(row_max_per_sample)  # [N_SAMPLES, 256]
        mean_row_absmax = row_absmax.mean(axis=0)  # [256]
        top5_rows = np.argsort(mean_row_absmax)[-5:][::-1]
        print(f"    逐行 |max| 最大的5行 (平均across samples):")
        for r in top5_rows:
            vals = row_absmax[:, r]
            print(f"      row {r:>3}: |max| = {vals.mean():.2f} ± {vals.std():.2f}  "
                  f"(range: {vals.min():.2f} ~ {vals.max():.2f})")

        normal_mask = np.ones(256, dtype=bool)
        normal_mask[top5_rows[0]] = False
        flat_excl_top = stacked[:, normal_mask, :].flatten()
        flat_top = stacked[:, top5_rows[0], :].flatten()
        print(f"    排除 row {top5_rows[0]} 后:")
        print(f"      其余: min={flat_excl_top.min():.4f}, max={flat_excl_top.max():.4f}, "
              f"range={flat_excl_top.max()-flat_excl_top.min():.4f}")
        print(f"      row {top5_rows[0]}: min={flat_top.min():.4f}, max={flat_top.max():.4f}, "
              f"range={flat_top.max()-flat_top.min():.4f}")

        print(f"    10-bit 量化有效级数分析:")
        for feat in all_feats[:1]:
            vmin, vmax = feat.min(), feat.max()
            step = (vmax - vmin) / 1023
            print(f"      全局 step = ({vmax:.2f} - {vmin:.2f}) / 1023 = {step:.6f}")
            excl = feat[normal_mask]
            excl_range = excl.max() - excl.min()
            effective_levels = excl_range / step if step > 0 else 0
            print(f"      排除 row {top5_rows[0]} 后: range={excl_range:.4f}, "
                  f"有效级数={effective_levels:.1f} / 1023 ({effective_levels/1023*100:.1f}%)")
            print(f"      等价有效 bit = {np.log2(max(effective_levels, 1)):.2f}")

        top_row = top5_rows[0]
        cross_sample_cos = []
        ref = stacked[0, top_row, :]
        for s in range(1, len(stacked)):
            v = stacked[s, top_row, :]
            cos = np.dot(ref, v) / (np.linalg.norm(ref) * np.linalg.norm(v) + 1e-12)
            cross_sample_cos.append(cos)
        if cross_sample_cos:
            print(f"    row {top_row} 跨样本余弦相似度: "
                  f"mean={np.mean(cross_sample_cos):.8f}, "
                  f"min={np.min(cross_sample_cos):.8f}")


# ═══════════════════════════════════════════════════════
#  第2步：纯量化（无VTM）精度损失
# ═══════════════════════════════════════════════════════

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


def test_quantize_only(sample_names, labels):
    print("\n" + "=" * 90)
    print("  第2步：纯量化/反量化 → 回放分类（不经过 VTM）")
    print("=" * 90)

    import torch
    from siglip2_feat_pipeline import (
        load_siglip2, build_text_emb, continue_from_tokens, compute_logits,
    )

    model, processor = load_siglip2(MODEL_ID, DEVICE)
    model.float()
    vision_model = model.vision_model
    text_emb = build_text_emb(model, processor, str(CLASSNAMES_FILE), DEVICE)

    def classify(img_emb):
        logits = compute_logits(model, img_emb, text_emb)
        return torch.topk(logits, k=5, dim=-1).indices.squeeze(0).tolist()

    for layer in LAYERS:
        layer_idx = int(layer[-2:])
        print(f"\n  ─── {layer} ───")

        orig_c1 = orig_c5 = 0
        qonly_c1 = qonly_c5 = 0
        cos_list, mse_list = [], []

        for name in sample_names:
            gt = labels[name]
            feat = np.load(str(FEAT_ROOT / layer / f"{name}.npy"))

            # A: 原始特征直接回放
            tok_orig = torch.from_numpy(feat).unsqueeze(0).to(DEVICE, dtype=torch.float32)
            with torch.no_grad():
                emb_orig = continue_from_tokens(vision_model, tok_orig, layer_idx)
                top5_orig = classify(emb_orig)
            if top5_orig[0] == gt: orig_c1 += 1
            if gt in top5_orig: orig_c5 += 1

            # B: 量化 → 反量化 → 回放（不经过 VTM）
            q, meta = quantize_linear(feat, BIT_DEPTH)
            feat_q = dequantize_linear(q, meta)

            cos = np.dot(feat.flatten(), feat_q.flatten()) / (
                np.linalg.norm(feat) * np.linalg.norm(feat_q) + 1e-12)
            mse = np.mean((feat - feat_q) ** 2)
            cos_list.append(cos)
            mse_list.append(mse)

            tok_q = torch.from_numpy(feat_q).unsqueeze(0).to(DEVICE, dtype=torch.float32)
            with torch.no_grad():
                emb_q = continue_from_tokens(vision_model, tok_q, layer_idx)
                top5_q = classify(emb_q)
            if top5_q[0] == gt: qonly_c1 += 1
            if gt in top5_q: qonly_c5 += 1

        n = len(sample_names)
        print(f"    原始特征回放:     Acc@1={orig_c1/n*100:.1f}%, Acc@5={orig_c5/n*100:.1f}%")
        print(f"    纯量化后回放:     Acc@1={qonly_c1/n*100:.1f}%, Acc@5={qonly_c5/n*100:.1f}%")
        print(f"    量化 CosSim:      {np.mean(cos_list):.8f}")
        print(f"    量化 MSE:         {np.mean(mse_list):.6f}")
        print(f"    Acc@1 drop:       {(qonly_c1-orig_c1)/n*100:+.1f}%")

    del model, vision_model
    import torch; torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════
#  第3步：VTM QP=0 无损测试
# ═══════════════════════════════════════════════════════

def vtm_encode_decode(feat_2d, qp, tmp_dir, sample_id=""):
    os.makedirs(tmp_dir, exist_ok=True)
    H, W = feat_2d.shape
    q, meta = quantize_linear(feat_2d, BIT_DEPTH)

    uid = f"debug_{sample_id}_{qp}_{os.getpid()}"
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
        try: os.unlink(p)
        except OSError: pass

    return q, q_rec, meta, feat_rec, bs_bytes


def test_vtm_lossless(sample_names):
    print("\n" + "=" * 90)
    print("  第3步：VTM QP=0 无损性验证（量化整数层面）")
    print("=" * 90)

    tmp_dir = str(SCRIPT_DIR / "_debug_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    for layer in LAYERS:
        print(f"\n  ─── {layer} ───")
        for name in sample_names[:2]:
            feat = np.load(str(FEAT_ROOT / layer / f"{name}.npy"))
            try:
                q_orig, q_rec, meta, feat_rec, bs_bytes = vtm_encode_decode(
                    feat, 0, tmp_dir, sample_id=f"{layer}_{name}")
            except Exception as e:
                print(f"    {name}: VTM 编解码失败: {e}")
                continue

            q_diff = (q_orig.astype(np.int32) - q_rec.astype(np.int32))
            q_exact = np.all(q_diff == 0)
            bpfp = bs_bytes * 8 / (feat.shape[0] * feat.shape[1])

            print(f"    {name}: 量化整数完全匹配={q_exact}, "
                  f"max|q_diff|={np.max(np.abs(q_diff))}, BPFP={bpfp:.4f}")
            if not q_exact:
                diff_pos = np.argwhere(q_diff != 0)
                print(f"      不匹配位置数: {len(diff_pos)} / {q_orig.size}")
                for pos in diff_pos[:5]:
                    r, c = pos
                    print(f"        [{r},{c}]: orig_q={q_orig[r,c]}, rec_q={q_rec[r,c]}, diff={q_diff[r,c]}")

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════
#  第4步：量化误差的逐行分析
# ═══════════════════════════════════════════════════════

def analyze_quantization_error_by_row(sample_names):
    print("\n" + "=" * 90)
    print("  第4步：量化误差逐行分析（哪些行受损最严重）")
    print("=" * 90)

    for layer in LAYERS:
        print(f"\n  ─── {layer} ───")
        feat = np.load(str(FEAT_ROOT / layer / f"{sample_names[0]}.npy"))
        q, meta = quantize_linear(feat, BIT_DEPTH)
        feat_q = dequantize_linear(q, meta)

        row_mse = np.mean((feat - feat_q) ** 2, axis=1)   # [256]
        row_cos = []
        for r in range(256):
            c = np.dot(feat[r], feat_q[r]) / (np.linalg.norm(feat[r]) * np.linalg.norm(feat_q[r]) + 1e-12)
            row_cos.append(c)
        row_cos = np.array(row_cos)

        worst_mse = np.argsort(row_mse)[-5:][::-1]
        worst_cos = np.argsort(row_cos)[:5]

        print(f"    量化 MSE 最大的5行:")
        for r in worst_mse:
            print(f"      row {r:>3}: MSE={row_mse[r]:.6f}, CosSim={row_cos[r]:.8f}, "
                  f"orig_range=[{feat[r].min():.2f}, {feat[r].max():.2f}]")

        print(f"    量化 CosSim 最低的5行:")
        for r in worst_cos:
            print(f"      row {r:>3}: CosSim={row_cos[r]:.8f}, MSE={row_mse[r]:.6f}, "
                  f"orig_range=[{feat[r].min():.2f}, {feat[r].max():.2f}]")

        print(f"    全局统计: mean_row_MSE={row_mse.mean():.6f}, "
              f"median_row_CosSim={np.median(row_cos):.8f}, "
              f"min_row_CosSim={row_cos.min():.8f}")

        q_unique = np.unique(q)
        print(f"    量化后使用的唯一整数值: {len(q_unique)} / 1024")

        excl_row = worst_mse[0] if row_mse[worst_mse[0]] > row_mse.mean() * 10 else None
        if excl_row is not None:
            mask = np.ones(256, dtype=bool)
            mask[excl_row] = False
            q_excl = np.unique(q[mask])
            print(f"    排除 row {excl_row} 后使用的唯一整数值: {len(q_excl)} / 1024")

            q_hist, _ = np.histogram(q[mask].flatten(), bins=50)
            occupied_bins = np.sum(q_hist > 0)
            print(f"    排除 row {excl_row} 后的值分布: 50-bin 中有 {occupied_bins} bin 被占用")


# ═══════════════════════════════════════════════════════
#  第5步：逐层 VTM 有损 + 回放分类 vs 纯量化回放分类
# ═══════════════════════════════════════════════════════

def test_vtm_vs_quantize(sample_names, labels):
    print("\n" + "=" * 90)
    print("  第5步：VTM QP=0 重建 vs 纯量化反量化 → 回放分类对比")
    print("=" * 90)

    import torch
    from siglip2_feat_pipeline import (
        load_siglip2, build_text_emb, continue_from_tokens, compute_logits,
    )

    model, processor = load_siglip2(MODEL_ID, DEVICE)
    model.float()
    vision_model = model.vision_model
    text_emb = build_text_emb(model, processor, str(CLASSNAMES_FILE), DEVICE)

    def classify(img_emb):
        logits = compute_logits(model, img_emb, text_emb)
        return torch.topk(logits, k=5, dim=-1).indices.squeeze(0).tolist()

    tmp_dir = str(SCRIPT_DIR / "_debug_tmp2")
    os.makedirs(tmp_dir, exist_ok=True)

    for layer in LAYERS:
        layer_idx = int(layer[-2:])
        print(f"\n  ─── {layer} ───")

        orig_c1 = qonly_c1 = vtm0_c1 = 0

        for name in sample_names:
            gt = labels[name]
            feat = np.load(str(FEAT_ROOT / layer / f"{name}.npy"))

            tok = torch.from_numpy(feat).unsqueeze(0).to(DEVICE, dtype=torch.float32)
            with torch.no_grad():
                top5 = classify(continue_from_tokens(vision_model, tok, layer_idx))
            if top5[0] == gt: orig_c1 += 1

            q, meta = quantize_linear(feat, BIT_DEPTH)
            feat_q = dequantize_linear(q, meta)
            tok_q = torch.from_numpy(feat_q).unsqueeze(0).to(DEVICE, dtype=torch.float32)
            with torch.no_grad():
                top5 = classify(continue_from_tokens(vision_model, tok_q, layer_idx))
            if top5[0] == gt: qonly_c1 += 1

            _, _, _, feat_vtm0, _ = vtm_encode_decode(
                feat, 0, tmp_dir, sample_id=f"{layer}_{name}")
            tok_v = torch.from_numpy(feat_vtm0).unsqueeze(0).to(DEVICE, dtype=torch.float32)
            with torch.no_grad():
                top5 = classify(continue_from_tokens(vision_model, tok_v, layer_idx))
            if top5[0] == gt: vtm0_c1 += 1

        n = len(sample_names)
        print(f"    原始特征:        Acc@1 = {orig_c1/n*100:.1f}%")
        print(f"    纯量化反量化:    Acc@1 = {qonly_c1/n*100:.1f}%")
        print(f"    VTM QP=0 重建:   Acc@1 = {vtm0_c1/n*100:.1f}%")
        print(f"    量化损失:        {(qonly_c1-orig_c1)/n*100:+.1f}%")
        print(f"    VTM附加损失:     {(vtm0_c1-qonly_c1)/n*100:+.1f}%")

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    del model, vision_model
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════

def main():
    print("=" * 90)
    print("  blk15 VTM 失败根因排查")
    print("  流水线: 原始 →①量化→ 整数 →②VTM→ 重建整数 →③反量化→ 浮点 →④回放→ 分类")
    print("=" * 90)

    labels = load_labels()
    sample_names = get_sample_names(labels)
    print(f"  样本数: {len(sample_names)}")
    print(f"  样本: {sample_names}")

    analyze_distribution(sample_names)
    analyze_quantization_error_by_row(sample_names)
    test_vtm_lossless(sample_names)
    test_quantize_only(sample_names, labels)
    test_vtm_vs_quantize(sample_names, labels)

    print("\n" + "=" * 90)
    print("  排查完成")
    print("=" * 90)


if __name__ == "__main__":
    main()
