"""
VTM Depth Pipeline 验证脚本

每层取 2 个样本，在所有 QP 下做完整 编码→解码→回放→评测，用于：
  1. 检验 QP 取值是否产生合理 BPFP 范围
  2. 不同压缩强度下任务精度差异是否可观测
  3. 验证 VTM + replay 流水线端到端正确性

Usage:
  conda activate featcodec2
  cd /data4/workspace/zlt/featcodec/ORFC/coding/vtm_baseline
  python verify_vtm_depth.py
"""

import os, sys, subprocess, shutil
import numpy as np
from pathlib import Path
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))
sys.path.insert(0, str(PROJECT_ROOT / "backbone" / "dinov2"))

os.environ['USE_XFORMERS'] = '0'

import torch
from dinov2_depth_pipeline import (
    preprocess_image, parse_split_file, load_depth_gt,
    compute_depth_metrics, _load_backbone, _load_depth_head,
    decode_depth, direct_inference, MODEL_REGISTRY, PATCH_SIZE
)

# ─────────────── 配置 ───────────────
FEAT_ROOT = PROJECT_ROOT / "features" / "nyu_depth_80" / "dinov2_vitl14"
DATA_ROOT = Path("/data4/workspace/zlt/LaMoFC/Data_example/dinov2/dpt/source/NYU_Test80")
SPLIT_FILE = Path("/data4/workspace/zlt/LaMoFC/Data_example/dinov2/dpt/source/nyu_test_80.txt")

VTM_ENCODER = str(SCRIPT_DIR / "EncoderAppStatic")
VTM_DECODER = str(SCRIPT_DIR / "DecoderAppStatic")
VTM_CFG = str(SCRIPT_DIR / "encoder_intra_vtm.cfg")
BIT_DEPTH = 10

LAYERS = ["blk05", "blk10", "blk15", "blk20"]
QPS_SHALLOW = [25, 27, 30, 32, 35]
QPS_DEEP = [0, 2, 5, 7, 10, 12]
N_SAMPLES = 2
DEVICE = os.environ.get("CUDA_DEVICE", "cuda")
VTM_WORKERS = 20


# ─────────────── VTM 编解码单帧 ───────────────

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
    """对 2D 特征执行 VTM 编解码，返回 (重建特征, bitstream_bytes)"""
    import threading
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


# ─────────────── 并行 VTM 任务（模块级，可 pickle） ───────────────

def vtm_task(args_tuple):
    layer, qp, idx, name, feat_path, _, tmp_dir = args_tuple
    feat_orig = np.load(feat_path)
    if feat_orig.ndim == 3 and feat_orig.shape[0] == 1:
        feat_2d = feat_orig[0]
    else:
        feat_2d = feat_orig
    H, W = feat_2d.shape
    feat_rec, bs_bytes = vtm_encode_decode(feat_2d, qp, tmp_dir)
    bpfp = (bs_bytes * 8) / (H * W)
    feat_mse = float(np.mean((feat_2d - feat_rec) ** 2))
    rec_path = os.path.join(tmp_dir, f"{layer}_{qp}_{name}.npy")
    np.save(rec_path, feat_rec)
    return (layer, qp, idx, bpfp, feat_mse, rec_path)


# ─────────────── 主流程 ───────────────

def main():
    print("=" * 90)
    print("  VTM + DINOv2 Depth 验证脚本")
    print(f"  每层 {N_SAMPLES} 个样本 × 所有 QP")
    print("=" * 90)

    # 解析数据列表，取前 N_SAMPLES 个
    samples = parse_split_file(str(SPLIT_FILE))[:N_SAMPLES]
    print(f"  样本: {[s[0] for s in samples]}")

    # 临时目录（放在脚本同级目录下）
    tmp_dir = str(SCRIPT_DIR / "_vtm_verify_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"  临时目录: {tmp_dir}")

    # ================================================================
    #  Phase 1: 并行 VTM 编解码（纯 CPU，不占显存）
    # ================================================================
    print(f"\n[1/3] VTM 编解码 (workers={VTM_WORKERS})，无 GPU 占用...")

    tasks = []
    for layer in LAYERS:
        qps = QPS_DEEP if layer == "blk20" else QPS_SHALLOW
        for qp in qps:
            for i, (rgb_rel, depth_rel, _) in enumerate(samples):
                name = rgb_rel.replace('/', '_').rsplit('.', 1)[0]
                feat_path = str(FEAT_ROOT / layer / f"{name}.npy")
                tasks.append((layer, qp, i, name, feat_path, depth_rel, tmp_dir))

    print(f"  总任务数: {len(tasks)} (4 layers × QPs × {N_SAMPLES} samples)")
    t0_all = perf_counter()

    vtm_results = {}
    completed = 0
    with ProcessPoolExecutor(max_workers=VTM_WORKERS) as executor:
        futures = {executor.submit(vtm_task, t): t for t in tasks}
        for future in as_completed(futures):
            completed += 1
            try:
                layer, qp, idx, bpfp, feat_mse, rec_path = future.result()
                vtm_results[(layer, qp, idx)] = (bpfp, feat_mse, rec_path)
                if completed % 10 == 0 or completed == len(tasks):
                    print(f"  VTM 编解码进度: {completed}/{len(tasks)}")
            except Exception as e:
                task_info = futures[future]
                print(f"  [ERR] {task_info[0]}/QP{task_info[1]}/{task_info[3]}: {e}")

    vtm_elapsed = perf_counter() - t0_all
    print(f"  VTM 编解码完成: {vtm_elapsed:.1f}s (并行 {VTM_WORKERS} workers)")

    # ================================================================
    #  Phase 2: 加载模型到 GPU，计算 Anchor + 回放全部重建特征
    # ================================================================
    print(f"\n[2/3] 加载模型到 GPU，计算 Anchor + 回放评测...")

    class Args:
        model = "vitl14"
        weights_root = "/data4/workspace/zlt/cache/torch/hub/checkpoints"
        device = DEVICE
    args = Args()

    backbone, reg = _load_backbone(args)
    head = _load_depth_head(args)

    # Anchor（无压缩直接推理）
    print("  计算 Anchor（无压缩直接推理）...")
    anchor_metrics = []
    sample_info = []
    for rgb_rel, depth_rel, _ in samples:
        img_path = str(DATA_ROOT / rgb_rel)
        img_tensor, ori_shape, pad_shape = preprocess_image(img_path)
        depth_direct = direct_inference(backbone, head, img_tensor, pad_shape, ori_shape, DEVICE)
        gt_path = str(DATA_ROOT / depth_rel)
        depth_gt = load_depth_gt(gt_path)
        m = compute_depth_metrics(depth_direct, depth_gt)
        anchor_metrics.append(m)
        sample_info.append((rgb_rel, ori_shape, pad_shape))
    anchor_avg = {k: np.mean([m[k] for m in anchor_metrics]) for k in anchor_metrics[0]}
    print(f"  Anchor: RMSE={anchor_avg['rmse']:.4f}")

    # 回放全部重建特征
    print("\n  回放重建特征...")
    results = {}

    for layer in LAYERS:
        layer_idx = int(layer[-2:])
        qps = QPS_DEEP if layer == "blk20" else QPS_SHALLOW

        print(f"\n  ─── {layer} (layer_idx={layer_idx}) ───")

        for qp in qps:
            qp_metrics = []
            qp_bpfp_list = []
            qp_mse_list = []

            for i, (rgb_rel, depth_rel, _) in enumerate(samples):
                key = (layer, qp, i)
                if key not in vtm_results:
                    continue
                bpfp, feat_mse, rec_path = vtm_results[key]
                qp_bpfp_list.append(bpfp)
                qp_mse_list.append(feat_mse)

                feat_rec = np.load(rec_path)
                feat_tensor = torch.from_numpy(feat_rec[np.newaxis, ...]).to(DEVICE)
                ori_shape = sample_info[i][1]
                pad_shape = sample_info[i][2]
                depth_pred = decode_depth(
                    backbone, head, feat_tensor, layer_idx, pad_shape, ori_shape, DEVICE
                )

                gt_path = str(DATA_ROOT / depth_rel)
                depth_gt = load_depth_gt(gt_path)
                m = compute_depth_metrics(depth_pred, depth_gt)
                qp_metrics.append(m)

            if not qp_metrics:
                continue
            avg_bpfp = np.mean(qp_bpfp_list)
            avg_mse = np.mean(qp_mse_list)
            avg_m = {k: np.mean([m[k] for m in qp_metrics]) for k in qp_metrics[0]}

            results[(layer, qp)] = {
                'bpfp': avg_bpfp, 'feat_mse': avg_mse,
                'rmse': avg_m['rmse'],
            }
            print(f"    QP={qp:>2}: BPFP={avg_bpfp:.4f}, FeatMSE={avg_mse:.6f}, "
                  f"RMSE={avg_m['rmse']:.4f}")

    # 释放 GPU
    del backbone, head
    torch.cuda.empty_cache()

    # 清理临时文件
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # 汇总报告
    print("\n" + "=" * 90)
    print("[3/3] 验证汇总报告")
    print("=" * 90)

    print(f"\n{'Layer':<8} {'QP':>4} {'BPFP':>8} {'FeatMSE':>12} {'RMSE':>8} {'ΔRMSE':>8}")
    print("-" * 56)

    for layer in LAYERS:
        qps = QPS_DEEP if layer == "blk20" else QPS_SHALLOW
        for qp in qps:
            r = results[(layer, qp)]
            d_rmse = r['rmse'] - anchor_avg['rmse']
            print(f"{layer:<8} {qp:>4} {r['bpfp']:>8.4f} {r['feat_mse']:>12.6f} "
                  f"{r['rmse']:>8.4f} {d_rmse:>+8.4f}")
        print()

    print("-" * 56)
    print(f"  Anchor (无压缩): RMSE={anchor_avg['rmse']:.4f}")
    print()

    # 判定
    print("─── 验证结论 ───")
    all_bpfps = [results[(l, q)]['bpfp'] for l in LAYERS
                 for q in (QPS_DEEP if l == "blk20" else QPS_SHALLOW)]
    bpfp_min, bpfp_max = min(all_bpfps), max(all_bpfps)
    print(f"  1. BPFP 范围: [{bpfp_min:.4f}, {bpfp_max:.4f}]", end="")
    if bpfp_min < 0.1 and bpfp_max > 0.5:
        print("  ✓ 覆盖充分")
    else:
        print("  ✗ 范围不足，需调整 QP")

    shallow_rmse = [results[("blk05", q)]['rmse'] for q in QPS_SHALLOW]
    rmse_range = max(shallow_rmse) - min(shallow_rmse)
    print(f"  2. blk05 RMSE 变化幅度: {rmse_range:.4f}", end="")
    if rmse_range > 0.01:
        print("  ✓ 差异可观测")
    else:
        print("  ≈ 差异较小（可接受）")

    max_qp_layer = "blk05"
    max_qp = QPS_SHALLOW[-1]
    r_max = results[(max_qp_layer, max_qp)]
    print(f"  3. 最高 QP ({max_qp_layer}/QP{max_qp}) vs Anchor: "
          f"ΔRMSE={r_max['rmse'] - anchor_avg['rmse']:+.4f}", end="")
    if abs(r_max['rmse'] - anchor_avg['rmse']) < 5.0:
        print("  ✓ 流水线正确（非异常值）")
    else:
        print("  ✗ 异常偏大，需排查")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
