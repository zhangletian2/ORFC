#!/usr/bin/env python
"""重新处理QP=7缺失的文件，并写入_stats.csv"""
import os, glob, csv, argparse, subprocess as sp
import numpy as np
from pathlib import Path
from time import perf_counter

CSV_HEADER = ["filename","model","layer","qp","bitstream_bytes","bits","bpfp","encode_s","decode_s","total_s"]

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
                     enc_bin: str, dec_bin: str, cfg: str, qp: int, bit_depth: int):
    feat = np.load(npy_path)
    if feat.ndim == 3 and feat.shape[0] == 1:
        feat = feat[0]
    assert feat.shape == (257, 1024), f"expect (257,1024), got {feat.shape} @ {npy_path}"

    H, W = feat.shape
    q, meta = quantize_linear(feat, bit_depth)

    stem = npy_path.stem
    yuv_path = tmp_dir / f"{stem}.y"
    write_y400(q, yuv_path)

    bitstream = tmp_dir / f"{stem}.qp{qp}.vvc"
    enc_log = tmp_dir / f"{stem}.qp{qp}.enc.log"
    t0 = perf_counter()
    run_vtm_encode(enc_bin, cfg, yuv_path, W, H, bit_depth, qp, bitstream, enc_log)
    t1 = perf_counter()
    encode_s = t1 - t0

    dec_yuv = tmp_dir / f"{stem}.qp{qp}.dec.y"
    dec_log = tmp_dir / f"{stem}.qp{qp}.dec.log"
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

    return {
        "filename": stem,
        "bitstream_bytes": bs_bytes,
        "bits": bits,
        "bpfp": bpfp,
        "encode_s": encode_s,
        "decode_s": decode_s,
        "total_s": total_s,
    }

# 缺失的文件列表
MISSING_FILES = {
    ("clip_vitl14", "blk11"): [
        "ILSVRC2012_val_00000003", "ILSVRC2012_val_00000007", "ILSVRC2012_val_00000008",
        "ILSVRC2012_val_00000011", "ILSVRC2012_val_00000092", "ILSVRC2012_val_00000093",
        "ILSVRC2012_val_00000098", "ILSVRC2012_val_00000105", "ILSVRC2012_val_00000111",
        "ILSVRC2012_val_00000122", "ILSVRC2012_val_00000125", "ILSVRC2012_val_00000127",
        "ILSVRC2012_val_00000130", "ILSVRC2012_val_00000137", "ILSVRC2012_val_00000164",
        "ILSVRC2012_val_00000172", "ILSVRC2012_val_00000175", "ILSVRC2012_val_00000181",
        "ILSVRC2012_val_00000191", "ILSVRC2012_val_00000195", "ILSVRC2012_val_00000198",
        "ILSVRC2012_val_00000199", "ILSVRC2012_val_00000207", "ILSVRC2012_val_00000218",
        "ILSVRC2012_val_00000225", "ILSVRC2012_val_00000227", "ILSVRC2012_val_00000231",
        "ILSVRC2012_val_00000238", "ILSVRC2012_val_00000249", "ILSVRC2012_val_00000253",
        "ILSVRC2012_val_00000256", "ILSVRC2012_val_00000301",
    ],
    ("clip_vitl14", "blk17"): [
        "ILSVRC2012_val_00000074",
    ],
    ("dinov2_vitl14", "blk17"): [
        "ILSVRC2012_val_00000153", "ILSVRC2012_val_00000272", "ILSVRC2012_val_00000287",
        "ILSVRC2012_val_00000288", "ILSVRC2012_val_00000289", "ILSVRC2012_val_00000297",
        "ILSVRC2012_val_00000298", "ILSVRC2012_val_00000312", "ILSVRC2012_val_00000318",
        "ILSVRC2012_val_00000319", "ILSVRC2012_val_00000330", "ILSVRC2012_val_00000342",
        "ILSVRC2012_val_00000346", "ILSVRC2012_val_00001066", "ILSVRC2012_val_00001067",
    ],
}

def main():
    _script_dir = Path(__file__).resolve().parent
    _project_root = (_script_dir / ".." / "..").resolve()
    feat_root = _project_root / "features" / "test"
    tmp_dir = _script_dir / "_vtm_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    enc_bin = str(_script_dir / "EncoderAppStatic")
    dec_bin = str(_script_dir / "DecoderAppStatic")
    cfg = str(_script_dir / "encoder_intra_vtm.cfg")
    bit_depth = 10
    qp = 7

    for (model, layer), files in MISSING_FILES.items():
        in_dir  = feat_root / model / layer
        out_dir = feat_root / model / "decoded" / "vtm" / str(qp) / layer
        
        # CSV 文件路径（与原始 vtm_baseline.py 保持一致）
        stats_csv = out_dir.parent / "_stats.csv"
        stats_csv.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[RUN] model={model} layer={layer} qp={qp} files={len(files)}")
        
        rows = []
        for idx, stem in enumerate(files, 1):
            npy_path = in_dir / f"{stem}.npy"
            if not npy_path.exists():
                print(f"  [SKIP] {npy_path} 不存在")
                continue
            
            try:
                rec = process_one_file(
                    npy_path, out_dir, tmp_dir,
                    enc_bin, dec_bin, cfg, qp, bit_depth
                )
                # 收集CSV行数据
                rows.append([
                    rec["filename"], model, layer, qp,
                    rec["bitstream_bytes"], rec["bits"], f"{rec['bpfp']:.6f}",
                    f"{rec['encode_s']:.6f}", f"{rec['decode_s']:.6f}", f"{rec['total_s']:.6f}"
                ])
                print(f"  [{idx:04d}/{len(files)}] {rec['filename']}: "
                      f"Enc {rec['encode_s']:.3f}s | Dec {rec['decode_s']:.3f}s | "
                      f"Total {rec['total_s']:.3f}s | BPFP={rec['bpfp']:.4f}")
            except Exception as e:
                print(f"  [ERR] {stem}: {e}")
        
        # 追加写入 _stats.csv
        if rows:
            write_header = not stats_csv.exists()
            with open(stats_csv, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(CSV_HEADER)
                w.writerows(rows)
            print(f"  [CSV] 已追加 {len(rows)} 条记录到 {stats_csv}")

    print("\n[完成] 缺失文件补全完成")

if __name__ == "__main__":
    main()
