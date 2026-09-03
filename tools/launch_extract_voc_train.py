#!/usr/bin/env python
"""Shard voc2012_all_5000.txt across GPUs and extract flattened slide features.

Each worker:
  dinov2_seg_pipeline.py extract --blocks 5,10,15,20 --flatten_slides

Output:
  features/train/dinov2_vitl14_voc/blk{05,10,15,20}/{name}_sXX.npy
  shape [1370, 1024] per slide (1 CLS + 37x37)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORFC = HERE.parent
FEATCODEC = ORFC.parent
PY = "/home/user/anaconda3/envs/featcodec2/bin/python"
PIPELINE = HERE / "dinov2_seg_pipeline.py"
IMAGE_LIST = ORFC / "utils" / "voc2012_all_5000.txt"
OUT_ROOT = FEATCODEC / "features" / "train" / "dinov2_vitl14_voc"
LOGDIR = ORFC / "coding" / "orfcv2" / "logs"
DEFAULT_GPUS = "0,1,2,3,4,5"


def parse_gpus(s: str):
    return [int(x) for x in s.split(",") if x.strip() != ""]


def ensure_out_dirs(out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)
    for k in (5, 10, 15, 20):
        (out_root / f"blk{k:02d}").mkdir(parents=True, exist_ok=True)
    orfc_train = ORFC / "features" / "train"
    orfc_link = orfc_train / "dinov2_vitl14_voc"
    if orfc_train.is_dir() and orfc_link.resolve() != out_root.resolve():
        if orfc_link.is_symlink() or orfc_link.exists():
            return
        try:
            orfc_link.symlink_to(out_root)
            print(f"symlink {orfc_link} -> {out_root}")
        except OSError as e:
            print(f"warn: could not link {orfc_link}: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", default=os.environ.get("GPUS", DEFAULT_GPUS))
    ap.add_argument("--image_list", default=str(IMAGE_LIST))
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    ap.add_argument("--blocks", default="5,10,15,20")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    gpus = parse_gpus(args.gpus)
    nsh = len(gpus)
    if nsh < 1:
        raise SystemExit("no GPUs given")
    if not Path(args.image_list).is_file():
        raise SystemExit(f"missing list {args.image_list}")
    if not PIPELINE.is_file():
        raise SystemExit(f"missing {PIPELINE}")

    out_root = Path(args.out_root)
    ensure_out_dirs(out_root)
    LOGDIR.mkdir(parents=True, exist_ok=True)

    env_base = os.environ.copy()
    env_base["TORCH_HOME"] = str(ORFC / "pretrained")
    env_base.setdefault("USE_XFORMERS", "0")

    procs = []
    print(f"extract VOC train  list={args.image_list}")
    print(f"  out={out_root}  blocks={args.blocks}  flatten_slides  {nsh} workers")
    for sid, gpu in enumerate(gpus):
        log = LOGDIR / f"extract_voc_train_w{sid}_gpu{gpu}.log"
        cmd = [
            PY, "-u", str(PIPELINE), "extract",
            "--model", "vitl14",
            "--voc_root", str(ORFC / "data" / "VOCdevkit" / "VOC2012"),
            "--weights_root", str(ORFC / "pretrained"),
            "--out_root", str(out_root),
            "--blocks", args.blocks,
            "--image_list", args.image_list,
            "--batch_size", str(args.batch_size),
            "--device", "cuda",
            "--flatten_slides",
            "--skip_existing",
            "--shard_id", str(sid),
            "--num_shards", str(nsh),
        ]
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        proc = subprocess.Popen(
            cmd, cwd=str(HERE), env=env,
            stdout=open(log, "w"), stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        procs.append((sid, gpu, proc, log))
        print(f"  worker{sid}  GPU{gpu}  pid={proc.pid}  log={log.name}", flush=True)

    t0 = time.time()
    failed = []
    for sid, gpu, proc, log in procs:
        rc = proc.wait()
        print(f"  worker{sid} GPU{gpu} exit={rc}  ({time.time()-t0:.0f}s)  {log.name}",
              flush=True)
        if rc != 0:
            failed.append((sid, gpu, rc, log))

    print(f"all workers finished in {time.time()-t0:.0f}s")
    if failed:
        for sid, gpu, rc, log in failed:
            print(f"FAIL worker{sid} GPU{gpu} rc={rc} see {log}")
        raise SystemExit(1)

    # quick sanity: count npy and print one shape
    import numpy as np
    blk = out_root / "blk05"
    npy = sorted(blk.glob("*.npy"))
    print(f"blk05 npy count={len(npy)}")
    if npy:
        arr = np.load(npy[0], mmap_mode="r")
        print(f"  sample {npy[0].name} shape={tuple(arr.shape)} dtype={arr.dtype}")


if __name__ == "__main__":
    main()
