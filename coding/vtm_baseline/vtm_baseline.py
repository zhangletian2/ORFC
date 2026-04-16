import os, glob, csv, argparse, subprocess as sp
import numpy as np
from pathlib import Path
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

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

def write_y400(raw: np.ndarray, path: Path):
    raw.tofile(str(path))

def read_y400(path: Path, shape, bit_depth: int):
    dt = np.uint16 if bit_depth > 8 else np.uint8
    arr = np.fromfile(str(path), dtype=dt)
    return arr.reshape(shape)

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

def process_one_file(npy_path: Path, out_dir: Path, tmp_dir: Path,
                     enc_bin: str, dec_bin: str, cfg: str, qp: int, bit_depth: int,
                     model: str = "", layer: str = ""):
    """处理单个文件，支持并行调用，支持任意形状的2D特征"""
    feat = np.load(npy_path)
    if feat.ndim == 3 and feat.shape[0] == 1:
        feat = feat[0]
    # 支持任意形状的2D特征 (H, W)
    if feat.ndim != 2:
        raise ValueError(f"expect 2D feature, got {feat.ndim}D with shape {feat.shape} @ {npy_path}")

    H, W = feat.shape
    q, meta = quantize_linear(feat, bit_depth)

    stem = npy_path.stem
    # 为并行处理添加唯一标识，避免文件冲突
    unique_id = f"{model}_{layer}_qp{qp}_{stem}"
    yuv_path = tmp_dir / f"{unique_id}.y"
    write_y400(q, yuv_path)

    bitstream = tmp_dir / f"{unique_id}.vvc"
    enc_log = tmp_dir / f"{unique_id}.enc.log"
    t0 = perf_counter()
    run_vtm_encode(enc_bin, cfg, yuv_path, W, H, bit_depth, qp, bitstream, enc_log)
    t1 = perf_counter()
    encode_s = t1 - t0

    dec_yuv = tmp_dir / f"{unique_id}.dec.y"
    dec_log = tmp_dir / f"{unique_id}.dec.log"
    t2 = perf_counter()
    run_vtm_decode(dec_bin, bitstream, dec_yuv, dec_log)
    t3 = perf_counter()
    decode_s = t3 - t2
    total_s = t3 - t0

    q_rec = read_y400(dec_yuv, (H, W), bit_depth)
    feat_rec = dequantize_linear(q_rec, meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{stem}.npy", feat_rec.astype(np.float32))

    bs_bytes = os.path.getsize(bitstream)
    bits = bs_bytes * 8
    bpfp = bits / (H * W)

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
        return {"error": str(e), "filename": Path(npy_path).stem, "model": model, "layer": layer, "qp": qp}

def main():
    ap = argparse.ArgumentParser()
    _script_dir = str(Path(__file__).resolve().parent)
    _project_root = str(Path(__file__).resolve().parents[2])
    ap.add_argument("--feat_root", default=os.path.join(_project_root, "features", "test"))
    ap.add_argument("--models", nargs="+", default=["dinov2_vitl14", "clip_vitl14"])
    ap.add_argument("--layers", nargs="+", default=["blk05", "blk11", "blk17", "blk23"])
    ap.add_argument("--qps", nargs="+", type=int, default=[0, 12, 22, 32, 42])
    ap.add_argument("--bit_depth", type=int, default=10)
    ap.add_argument("--vtm_encoder", default="EncoderAppStatic")
    ap.add_argument("--vtm_decoder", default="DecoderAppStatic")
    ap.add_argument("--vtm_cfg",    default="encoder_intra_vtm.cfg")
    ap.add_argument("--tmp_dir", default="./_vtm_tmp")
    ap.add_argument("--workers", type=int, default=1, help="并行worker数量，默认1表示串行")
    args = ap.parse_args()

    feat_root = Path(args.feat_root)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有任务
    all_tasks = []
    task_info = {}  # 用于按 (model, layer, qp) 分组记录任务
    
    for model in args.models:
        for qp in args.qps:
            for layer in args.layers:
                in_dir  = feat_root / model / layer
                out_dir = feat_root / model / "decoded" / "vtm" / str(qp) / layer
                npy_list = sorted(glob.glob(str(in_dir / "*.npy")))
                if not npy_list:
                    print(f"[WARN] empty: {in_dir}")
                    continue

                key = (model, layer, qp)
                task_info[key] = {
                    "out_dir": out_dir,
                    "total_files": len(npy_list),
                }
                
                for p in npy_list:
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

    if not all_tasks:
        print("[WARN] 没有找到任何任务")
        return

    print(f"\n[INFO] 总任务数: {len(all_tasks)}, 并行workers: {args.workers}")
    
    # 执行任务
    results = []
    if args.workers <= 1:
        # 串行处理
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
        # 并行处理
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
    from collections import defaultdict
    grouped = defaultdict(list)
    for rec in results:
        if "error" not in rec:
            key = (rec["model"], rec["layer"], rec["qp"])
            grouped[key].append(rec)

    csv_header = ["filename","model","layer","qp","bitstream_bytes","bits","bpfp","encode_s","decode_s","total_s"]
    
    for (model, layer, qp), recs in grouped.items():
        out_dir = feat_root / model / "decoded" / "vtm" / str(qp) / layer
        stats_csv = out_dir.parent / "_stats.csv"
        stats_csv.parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for rec in sorted(recs, key=lambda x: x["filename"]):
            rows.append([
                rec["filename"], model, layer, qp,
                rec["bitstream_bytes"], rec["bits"], f"{rec['bpfp']:.6f}",
                f"{rec['encode_s']:.6f}", f"{rec['decode_s']:.6f}", f"{rec['total_s']:.6f}"
            ])
        
        write_header = not stats_csv.exists()
        with open(stats_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(csv_header)
            w.writerows(rows)
        
        if rows:
            avg_bpfp = sum(float(r[6]) for r in rows) / len(rows)
            avg_enc  = sum(float(r[7]) for r in rows) / len(rows)
            avg_dec  = sum(float(r[8]) for r in rows) / len(rows)
            print(f"[DONE] {model}/{layer}/QP{qp}: "
                  f"avg BPFP={avg_bpfp:.4f} bits/scalar, "
                  f"enc={avg_enc:.3f}s, dec={avg_dec:.3f}s")

if __name__ == "__main__":
    main()
