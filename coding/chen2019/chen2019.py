#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import math
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from subprocess import Popen, PIPE, CalledProcessError
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))

def pad_to_multiple(x: int, m: int = 8) -> int:
    r = x % m
    return x if r == 0 else x + (m - r)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_yuv400_u8(path: Path, arr_u8_2d: np.ndarray):
    with open(path, "wb") as f:
        f.write(arr_u8_2d.astype(np.uint8).tobytes(order="C"))

def read_yuv400_u8(path: Path, h: int, w: int) -> np.ndarray:
    with open(path, "rb") as f:
        raw = f.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    assert arr.size == h * w, f"YUV size mismatch: {arr.size} vs {h*w}"
    return arr.reshape((h, w))

def linear_quantize(feat_2d: np.ndarray):
    fmin = float(np.min(feat_2d))
    fmax = float(np.max(feat_2d))
    if math.isclose(fmax, fmin):
        q = np.zeros_like(feat_2d, dtype=np.uint8)
    else:
        q = np.clip(np.rint((feat_2d - fmin) * 255.0 / (fmax - fmin)), 0, 255).astype(np.uint8)

    H, W = feat_2d.shape
    Hpad, Wpad = pad_to_multiple(H, 8), pad_to_multiple(W, 8)
    pad_h, pad_w = Hpad - H, Wpad - W
    if pad_h > 0 or pad_w > 0:
        q = np.pad(q, ((0, pad_h), (0, pad_w)), mode="edge")

    meta = {
        "mode": "linear",
        "fmin": fmin, "fmax": fmax,
        "pad_h": pad_h, "pad_w": pad_w,
        "orig_h": H, "orig_w": W,
    }
    return q, meta

def linear_dequantize(q_crop: np.ndarray, meta: dict) -> np.ndarray:
    fmin, fmax = meta["fmin"], meta["fmax"]
    if math.isclose(fmax, fmin):
        return np.full_like(q_crop, fmin, dtype=np.float32)
    return q_crop.astype(np.float32) * (fmax - fmin) / 255.0 + fmin

def run_cmd(cmd: list, cwd: Path=None):
    t0 = time.time()
    p = Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=str(cwd) if cwd else None, text=True)
    out, err = p.communicate()
    dt = time.time() - t0
    return p.returncode, out, err, dt

def encode_with_hm(enc_bin: str, cfg_path: str, in_yuv: Path, out_bit: Path,
                   rec_yuv: Path, width: int, height: int, qp: int, bitdepth: int = 8):
    cmd = [
        enc_bin,
        "-c", cfg_path,
        "-i", str(in_yuv),
        "-b", str(out_bit),
        "-o", str(rec_yuv),
        "--InputChromaFormat=400",
        # === 位深强制为 8（关键修复点）===
        f"--InputBitDepth={bitdepth}",
        f"--InputBitDepthC={bitdepth}",
        f"--InternalBitDepth={bitdepth}",
        f"--InternalBitDepthC={bitdepth}",
        f"--OutputBitDepth={bitdepth}",
        f"--OutputBitDepthC={bitdepth}",
        "--FrameRate=30",
        "-f", str(1),
        "--InputChromaFormat=400",
        f"--SourceWidth={width}",
        f"--SourceHeight={height}",
        f"--QP={qp}"
    ]
    code, out, err, dt = run_cmd(cmd)
    if code != 0:
        print("[STDERR]\n", err)
        import pdb
        pdb.set_trace()
        raise CalledProcessError(code, cmd, output=out, stderr=err)
    return dt, out

def decode_with_hm(dec_bin: str, bitstream: Path, out_yuv: Path):
    cmd = [dec_bin, "-b", str(bitstream), "-o", str(out_yuv)]
    code, out, err, dt = run_cmd(cmd)
    if code != 0:
        raise CalledProcessError(code, cmd, output=out, stderr=err)
    return dt, out

def mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = (a.astype(np.float64) - b.astype(np.float64)).ravel()
    return float(np.mean(diff * diff)) if diff.size else 0.0

def collect_bits(bit_path: Path) -> int:
    return bit_path.stat().st_size * 8

def list_feature_files(layer_dir: Path):
    return sorted(layer_dir.glob("*.npy"))

def main():
    parser = argparse.ArgumentParser(description="HM codec on ViT middle-layer features (257x1024)")
    parser.add_argument("--feat_root", type=str, default=os.path.join(_PROJECT_ROOT, "features", "test"))
    parser.add_argument("--models", type=str, nargs="+", default=["dinov2_vitl14", "clip_vitl14"])
    parser.add_argument("--layers", type=str, nargs="+", default=["blk05","blk11","blk17","blk23"])
    parser.add_argument("--qps", type=int, nargs="+", default=[0,12,22,32,42])
    parser.add_argument("--hm_root", type=str, default=os.path.join(_SCRIPT_DIR, "HM-16.21"))
    parser.add_argument("--hm_cfg", type=str, default="encoder_intra_main_rext.cfg")
    parser.add_argument("--tmpdir", type=str, default="./hm_tmp")
    args = parser.parse_args()

    feat_root = Path(args.feat_root)
    tmpdir = Path(args.tmpdir); ensure_dir(tmpdir)
    enc_bin = str(Path(args.hm_root) / "bin" / "TAppEncoderStatic")
    dec_bin = str(Path(args.hm_root) / "bin" / "TAppDecoderStatic")
    cfg_path = str(Path(args.hm_root) / "cfg" / args.hm_cfg)

    for model in args.models:
        for qp in args.qps:
            # stats.csv 放到 {feat_root}/{model}/decoded/chen/{qp}/_stats.csv
            qp_root = feat_root / model / "decoded" / "chen" / str(qp)
            ensure_dir(qp_root)
            stats_csv = qp_root / "_stats.csv"
            if not stats_csv.exists():
                pd.DataFrame(columns=[
                    "model","layer","qp","filename",
                    "points","bits","bpfp","enc_time_s","dec_time_s","mse"
                ]).to_csv(stats_csv, index=False)

            print(f"\n=== Model: {model} | QP: {qp} ===")

            for layer in args.layers:
                in_dir = feat_root / model / layer
                files = list_feature_files(in_dir)
                if not files:
                    print(f"[WARN] no npy files under {in_dir}")
                    continue

                # 重建特征保存到 {feat_root}/{model}/decoded/chen/{qp}/{layer}
                out_dir = qp_root / layer
                ensure_dir(out_dir)

                for npy_path in tqdm(files, desc=f"{model}/QP{qp}/{layer}", unit="feat", leave=False):
                    feat = np.load(npy_path)          # (257, 1024)
                    if feat.ndim != 2 or feat.shape != (257,1024):
                        raise ValueError(f"Expect (257,1024), got {feat.shape} @ {npy_path}")

                    q, meta = linear_quantize(feat)
                    Hpad, Wpad = q.shape
                    H, W = meta["orig_h"], meta["orig_w"]

                    stem = npy_path.stem
                    yuv_in  = tmpdir / f"{stem}_in.yuv"
                    yuv_rec = tmpdir / f"{stem}_rec.yuv"
                    bitf    = tmpdir / f"{stem}.bin"
                    write_yuv400_u8(yuv_in, q)

                    # encode
                    try:
                        enc_time, _ = encode_with_hm(enc_bin, cfg_path, yuv_in, bitf, yuv_rec,
                                                     width=Wpad, height=Hpad, qp=qp, bitdepth=8)
                    except Exception as e:
                        print(f"[ENC_ERR] {npy_path.name}: {e}")
                        # 清理局部临时文件后继续
                        for p in [yuv_in, yuv_rec, bitf]:
                            if Path(p).exists():
                                try: os.remove(p)
                                except: pass
                        continue

                    # decode
                    try:
                        dec_out = tmpdir / f"{stem}_dec.yuv"
                        dec_time, _ = decode_with_hm(dec_bin, bitf, dec_out)
                        rec_src = dec_out if dec_out.exists() else yuv_rec
                        rec_u8 = read_yuv400_u8(rec_src, Hpad, Wpad)
                        if dec_out.exists():
                            os.remove(dec_out)
                    except Exception as e:
                        print(f"[DEC_ERR] {npy_path.name}: {e}")
                        rec_u8 = read_yuv400_u8(yuv_rec, Hpad, Wpad)
                        dec_time = 0.0

                    # crop & dequant
                    rec_crop = rec_u8[:H, :W]
                    rec_feat = linear_dequantize(rec_crop, meta)

                    bits = collect_bits(bitf)
                    points = H * W
                    bpfp = bits / points
                    err_mse = mse(feat, rec_feat)

                    # save recon feature
                    np.save(out_dir / f"{stem}.npy", rec_feat.astype(np.float32))

                    # append to qp-level _stats.csv
                    pd.DataFrame([{
                        "model": model, "layer": layer, "qp": qp,
                        "filename": npy_path.name,
                        "points": points, "bits": bits, "bpfp": bpfp,
                        "enc_time_s": enc_time, "dec_time_s": dec_time, "mse": err_mse
                    }]).to_csv(stats_csv, mode="a", header=False, index=False)

                    # realtime print
                    print(f"[{model}][{layer}][QP{qp}] {npy_path.name}  "
                          f"enc:{enc_time:.3f}s  dec:{dec_time:.3f}s  "
                          f"BPFP:{bpfp:.4f}  MSE:{err_mse:.6f}")

                    # cleanup
                    for p in [yuv_in, yuv_rec, bitf]:
                        try:
                            if Path(p).exists(): os.remove(p)
                        except:
                            pass

    print("\nAll done.")

if __name__ == "__main__":
    main()
