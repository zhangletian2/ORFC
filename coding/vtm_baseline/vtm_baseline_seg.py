"""
VTM baseline for segmentation features.

分割特征形状为 (num_slides, 1+N, D)，例如 (2, 1370, 1024)。
编码时沿最后一维 concat 为 (1370, 2048)，解码后还原为 (2, 1370, 1024) 保存。
BPFP 和 MSE 按整个原始特征 (2×1370×1024) 计算。

目录结构：
  {feat_root}/{model}/{layer}/*.npy                     # 原始特征 (2,1370,1024)
  {feat_root}/{model}/decoded/vtm/{qp}/{layer}/*.npy    # 解码特征 (2,1370,1024)
  {feat_root}/{model}/decoded/vtm/{qp}/_stats.csv       # 统计CSV
"""

import os, glob, csv, argparse, subprocess as sp
import numpy as np
from pathlib import Path
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict


# ─────────────── 量化 / 反量化 ───────────────

def quantize_linear(x: np.ndarray, bit_depth: int):
    x = x.astype(np.float32)
    xmin, xmax = float(x.min()), float(x.max())
    if xmax <= xmin:
        scale = 1.0
    else:
        scale = ((1 << bit_depth) - 1) / (xmax - xmin)
    q = np.round((x - xmin) * scale).astype(np.uint16 if bit_depth > 8 else np.uint8)
    meta = {"min": xmin, "max": xmax, "bit_depth": bit_depth}
    return q, meta


def dequantize_linear(q: np.ndarray, meta):
    q = q.astype(np.float32)
    B = int(meta["bit_depth"])
    xmin, xmax = float(meta["min"]), float(meta["max"])
    if xmax <= xmin:
        return np.full_like(q, xmin, dtype=np.float32)
    scale = ((1 << B) - 1) / (xmax - xmin)
    x = q / scale + xmin
    return x.astype(np.float32)


# ─────────────── YUV I/O ───────────────

def write_y400(raw: np.ndarray, path: Path):
    raw.tofile(str(path))


def read_y400(path: Path, shape, bit_depth: int):
    dt = np.uint16 if bit_depth > 8 else np.uint8
    arr = np.fromfile(str(path), dtype=dt)
    return arr.reshape(shape)


# ─────────────── VTM 编解码 ───────────────

def run_vtm_encode(enc_bin, cfg, in_yuv, w, h, bit_depth, qp, out_bitstream, log_path):
    cmd = [
        enc_bin, "-c", cfg,
        "-i", str(in_yuv),
        "-b", str(out_bitstream),
        f"--SourceWidth={w}",
        f"--SourceHeight={h}",
        "--FramesToBeEncoded=1",
        "--FrameRate=1",
        "--InputChromaFormat=400",
        "--ConformanceWindowMode=1",
        f"--InternalBitDepth={bit_depth}",
        f"--InputBitDepth={bit_depth}",
        f"--OutputBitDepth={bit_depth}",
        f"--QP={qp}",
    ]
    with open(log_path, "w") as f:
        sp.run(cmd, stdout=f, stderr=sp.STDOUT, check=True)


def run_vtm_decode(dec_bin, bitstream, out_yuv, log_path):
    cmd = [dec_bin, "-b", str(bitstream), "-o", str(out_yuv)]
    with open(log_path, "w") as f:
        sp.run(cmd, stdout=f, stderr=sp.STDOUT, check=True)


# ─────────────── 核心处理 ───────────────

def process_one_file(npy_path: Path, out_dir: Path, tmp_dir: Path,
                     enc_bin: str, dec_bin: str, cfg: str, qp: int, bit_depth: int,
                     model: str = "", layer: str = ""):
    """
    处理单个分割特征文件。
    输入: (S, T, D) 例如 (2, 1370, 1024)
    编码: concat 为 (T, S*D) 即 (1370, 2048)
    解码后还原为 (S, T, D) 保存
    """
    feat = np.load(npy_path)  # (S, T, D)
    if feat.ndim != 3:
        raise ValueError(f"expect 3D feature (S,T,D), got {feat.ndim}D shape {feat.shape} @ {npy_path}")

    S, T, D = feat.shape
    orig_total = S * T * D  # 原始特征标量总数

    # concat: (S, T, D) -> (T, S*D)
    feat_2d = np.concatenate([feat[i] for i in range(S)], axis=-1)  # (T, S*D)
    H, W = feat_2d.shape

    q, meta = quantize_linear(feat_2d, bit_depth)

    stem = npy_path.stem
    unique_id = f"{model}_{layer}_qp{qp}_{stem}"
    yuv_path = tmp_dir / f"{unique_id}.y"
    write_y400(q, yuv_path)

    # 编码
    bitstream = tmp_dir / f"{unique_id}.vvc"
    enc_log = tmp_dir / f"{unique_id}.enc.log"
    t0 = perf_counter()
    run_vtm_encode(enc_bin, cfg, yuv_path, W, H, bit_depth, qp, bitstream, enc_log)
    t1 = perf_counter()
    encode_s = t1 - t0

    # 解码
    dec_yuv = tmp_dir / f"{unique_id}.dec.y"
    dec_log = tmp_dir / f"{unique_id}.dec.log"
    t2 = perf_counter()
    run_vtm_decode(dec_bin, bitstream, dec_yuv, dec_log)
    t3 = perf_counter()
    decode_s = t3 - t2
    total_s = t3 - t0

    # 反量化
    q_rec = read_y400(dec_yuv, (H, W), bit_depth)
    feat_2d_rec = dequantize_linear(q_rec, meta)  # (T, S*D)

    # 还原: (T, S*D) -> (S, T, D)
    feat_rec = np.stack(np.split(feat_2d_rec, S, axis=-1), axis=0)  # (S, T, D)

    # 保存（保持原始形状）
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{stem}.npy", feat_rec.astype(np.float32))

    # 统计（BPFP 按原始特征总标量数计算）
    bs_bytes = os.path.getsize(bitstream)
    bits = bs_bytes * 8
    bpfp = bits / orig_total

    # 清理临时文件
    for f in [yuv_path, bitstream, enc_log, dec_yuv, dec_log]:
        try:
            f.unlink()
        except:
            pass

    return {
        "filename": stem,
        "model": model,
        "layer": layer,
        "qp": qp,
        "orig_shape": f"{S}x{T}x{D}",
        "encode_shape": f"{H}x{W}",
        "bitstream_bytes": bs_bytes,
        "bits": bits,
        "bpfp": bpfp,
        "encode_s": encode_s,
        "decode_s": decode_s,
        "total_s": total_s,
    }


def process_one_file_wrapper(task):
    """包装函数用于并行处理"""
    npy_path, out_dir, tmp_dir, enc_bin, dec_bin, cfg, qp, bit_depth, model, layer = task
    try:
        return process_one_file(
            Path(npy_path), Path(out_dir), Path(tmp_dir),
            enc_bin, dec_bin, cfg, qp, bit_depth, model, layer
        )
    except Exception as e:
        return {"error": str(e), "filename": Path(npy_path).stem,
                "model": model, "layer": layer, "qp": qp}


# ─────────────── Main ───────────────

def main():
    ap = argparse.ArgumentParser(description="VTM baseline for segmentation features")
    _project_root = str(Path(__file__).resolve().parents[2])
    ap.add_argument("--feat_root", default=os.path.join(_project_root, "features", "voc2012_100"))
    ap.add_argument("--models", nargs="+", default=["dinov2_vitl14"])
    ap.add_argument("--layers", nargs="+", default=["blk05", "blk10", "blk15", "blk20"])
    ap.add_argument("--qps", nargs="+", type=int, default=[0, 12, 22, 32, 42])
    ap.add_argument("--bit_depth", type=int, default=10)
    ap.add_argument("--vtm_encoder", default="EncoderAppStatic")
    ap.add_argument("--vtm_decoder", default="DecoderAppStatic")
    ap.add_argument("--vtm_cfg",    default="encoder_intra_vtm.cfg")
    ap.add_argument("--tmp_dir", default="./_vtm_tmp_seg")
    ap.add_argument("--workers", type=int, default=1, help="并行worker数量，默认1表示串行")
    args = ap.parse_args()

    feat_root = Path(args.feat_root)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有任务
    all_tasks = []
    skipped_stems = defaultdict(list)

    for model in args.models:
        for qp in args.qps:
            for layer in args.layers:
                in_dir  = feat_root / model / layer
                out_dir = feat_root / model / "decoded" / "vtm" / str(qp) / layer
                npy_list = sorted(glob.glob(str(in_dir / "*.npy")))
                if not npy_list:
                    print(f"[WARN] empty: {in_dir}")
                    continue

                skipped = 0
                for p in npy_list:
                    stem = Path(p).stem
                    out_npy = out_dir / f"{stem}.npy"
                    if out_npy.exists():
                        skipped_stems[(model, layer, qp)].append(stem)
                        skipped += 1
                        continue
                    task = (
                        p,                    # npy_path
                        str(out_dir),         # out_dir
                        str(tmp_dir),         # tmp_dir
                        args.vtm_encoder,     # enc_bin
                        args.vtm_decoder,     # dec_bin
                        args.vtm_cfg,         # cfg
                        qp,                   # qp
                        args.bit_depth,       # bit_depth
                        model,                # model
                        layer,                # layer
                    )
                    all_tasks.append(task)
                if skipped:
                    print(f"[INFO] {model}/{layer}/QP{qp}: skipped {skipped} existing, "
                          f"remaining {len(npy_list) - skipped}")

    if not all_tasks and not skipped_stems:
        print("[WARN] 没有找到任何任务")
        return

    print(f"\n[INFO] 总任务数: {len(all_tasks)}, 跳过: {sum(len(v) for v in skipped_stems.values())}, "
          f"并行workers: {args.workers}")

    # 执行任务
    results = []
    if not all_tasks:
        print("[INFO] 所有文件已存在，仅更新统计")
    elif args.workers <= 1:
        for idx, task in enumerate(all_tasks, 1):
            rec = process_one_file_wrapper(task)
            results.append(rec)
            if "error" in rec:
                print(f"  [{idx:04d}/{len(all_tasks)}] [ERR] {rec['filename']}: {rec['error']}")
            else:
                print(f"  [{idx:04d}/{len(all_tasks)}] {rec['model']}/{rec['layer']}/QP{rec['qp']} {rec['filename']}: "
                      f"Enc {rec['encode_s']:.3f}s | Dec {rec['decode_s']:.3f}s | "
                      f"Total {rec['total_s']:.3f}s | BPFP={rec['bpfp']:.4f}")
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one_file_wrapper, task): task for task in all_tasks}
            for future in as_completed(futures):
                completed += 1
                rec = future.result()
                results.append(rec)
                if "error" in rec:
                    print(f"  [{completed:04d}/{len(all_tasks)}] [ERR] {rec['filename']}: {rec['error']}")
                else:
                    print(f"  [{completed:04d}/{len(all_tasks)}] {rec['model']}/{rec['layer']}/QP{rec['qp']} {rec['filename']}: "
                          f"Enc {rec['encode_s']:.3f}s | Dec {rec['decode_s']:.3f}s | "
                          f"Total {rec['total_s']:.3f}s | BPFP={rec['bpfp']:.4f}")

    # 按 (model, layer, qp) 分组统计并写入CSV
    grouped = defaultdict(list)
    for rec in results:
        if "error" not in rec:
            key = (rec["model"], rec["layer"], rec["qp"])
            grouped[key].append(rec)

    csv_header = ["filename","model","layer","qp","orig_shape","encode_shape",
                  "bitstream_bytes","bits","bpfp","encode_s","decode_s","total_s"]

    all_groups = set(grouped.keys()) | set(skipped_stems.keys())
    for (model, layer, qp) in all_groups:
        stats_csv = feat_root / model / "decoded" / "vtm" / str(qp) / "_stats.csv"
        stats_csv.parent.mkdir(parents=True, exist_ok=True)

        # 读取已有 csv，保留其他 layer 的行，捞回本 layer 被跳过文件的旧行
        other_rows = []
        carried_rows = []
        skip_set = set(skipped_stems.get((model, layer, qp), []))
        if stats_csv.exists():
            with open(stats_csv, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) < 4:
                        continue
                    if row[2] == layer and str(row[3]) == str(qp):
                        if row[0] in skip_set:
                            carried_rows.append(row)
                    else:
                        other_rows.append(row)

        # 构建本次新处理的行
        new_rows = []
        for rec in sorted(grouped.get((model, layer, qp), []), key=lambda x: x["filename"]):
            new_rows.append([
                rec["filename"], model, layer, qp,
                rec["orig_shape"], rec["encode_shape"],
                rec["bitstream_bytes"], rec["bits"], f"{rec['bpfp']:.6f}",
                f"{rec['encode_s']:.6f}", f"{rec['decode_s']:.6f}", f"{rec['total_s']:.6f}"
            ])

        # 整体重写
        with open(stats_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(csv_header)
            w.writerows(other_rows)
            w.writerows(sorted(carried_rows, key=lambda r: r[0]))
            w.writerows(new_rows)

        total_rows = len(carried_rows) + len(new_rows)
        if total_rows:
            all_bpfp = [float(r[8]) for r in carried_rows] + [float(r[8]) for r in new_rows]
            avg_bpfp = sum(all_bpfp) / len(all_bpfp)
            print(f"[DONE] {model}/{layer}/QP{qp}: "
                  f"{len(carried_rows)} carried + {len(new_rows)} new = {total_rows} total, "
                  f"avg BPFP={avg_bpfp:.4f}")


if __name__ == "__main__":
    main()
