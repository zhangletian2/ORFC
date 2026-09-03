#!/usr/bin/env python3
"""Sweep residual K2 (share R, RAEv2 L1) over a fixed set of DINOv3 ORFC bases.

Per layer, 5 bases (one GPU each, one round):
  K=4/8/16  → lambda 0.5
  K=64/256  → lambda 1.0

  python launch_residual_raev2.py --gpus 2,4,5,6,7
  python launch_residual_raev2.py --gpus 2 --max_images 8 --epochs 2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ORFCV2 = Path(__file__).resolve().parent
ORFC = ORFCV2.parent / "orfc"
if str(ORFC / "eval") not in sys.path:
    sys.path.insert(0, str(ORFC / "eval"))

from dinov3_eval_common import (  # noqa: E402
    CKPT_DIR, COFAI_ROOT, IMAGENET_TRAIN_FEAT_STAGE2, IMAGENET_TRAIN_LIST_STAGE2,
    list_orfc_ckpts, parse_ckpt_name,
)

PY = "/data4/workspace/zlt/featcodec/CoFAI/.venv/bin/python"
TRAIN_PY = ORFCV2 / "run_residual_raev2.py"
LOG_DIR = ORFCV2 / "logs" / "dinov3_raev2_res"
OUT_DIR = ORFCV2 / "checkpoints" / "dinov3_vitl16_raev2_res"

LAYERS = ("blk05", "blk10", "blk15", "blk20")
# (K1, base-training λ)
BASE_SPECS = (
    (4, 0.5),
    (8, 0.5),
    (16, 0.5),
    (64, 1.0),
    (256, 1.0),
)


def _env(gpu: str) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["PROJECT_ROOT"] = str(COFAI_ROOT)
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
              "ALL_PROXY", "all_proxy"):
        env.pop(k, None)
    return env


def ckpt_done(base: Path, k2: int, epochs: int, lmbda: float) -> bool:
    stem_part = f"{base.stem}_resK{k2}_shareR_raev2l1_lmbda{lmbda}"
    if not OUT_DIR.is_dir():
        return False
    return any(
        f"_ep{epochs}_" in p.name
        for p in OUT_DIR.glob(f"{stem_part}_*.pt")
    )


def _match_ckpt(index: dict, layer: str, k: int, lmbda: float) -> Path | None:
    for (ly, kk, ll), path in index.items():
        if ly == layer and kk == k and abs(float(ll) - lmbda) < 1e-6:
            return path
    return None


def select_base_ckpts(ckpt_dir: Path) -> list[Path]:
    index = {}
    for p in list_orfc_ckpts(ckpt_dir):
        info = parse_ckpt_name(p)
        index[(info["layer"], info["K"], info["lmbda"])] = p
    selected, missing = [], []
    for layer in LAYERS:
        for k, lam in BASE_SPECS:
            p = _match_ckpt(index, layer, k, lam)
            if p is None:
                missing.append(f"{layer} K{k} lmbda{lam}")
            else:
                selected.append(p)
    if missing:
        raise SystemExit("missing base ckpts:\n  " + "\n  ".join(missing))
    return selected


def make_jobs(ckpt_dir: Path, extra: list[str], k2: int, epochs: int,
             batch_size: int, res_lmbda: float):
    out = []
    for ckpt in select_base_ckpts(ckpt_dir):
        info = parse_ckpt_name(ckpt)
        tag = f"{ckpt.stem}_resK{k2}_lmbda{res_lmbda}"
        cmd = [
            PY, "-u", str(TRAIN_PY),
            "--base_ckpt", str(ckpt),
            "--K", str(k2),
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--device", "cuda:0",
            "--out_dir", str(OUT_DIR),
            "--pathname_list", str(IMAGENET_TRAIN_LIST_STAGE2),
            "--feat_cache_dir", str(IMAGENET_TRAIN_FEAT_STAGE2),
            *extra,
        ]
        out.append({
            "tag": tag,
            "ckpt": ckpt,
            "layer": info["layer"],
            "K": info["K"],
            "base_lmbda": info["lmbda"],
            "res_lmbda": res_lmbda,
            "cmd": cmd,
            "log": LOG_DIR / f"train_{tag}.log",
            "done": ckpt_done(ckpt, k2, epochs, res_lmbda),
            "batch_size": batch_size,
        })
    return out


def launch_round(gpus: list[str], jobs: list[dict]) -> int:
    """One layer = 5 configs on 5 GPUs. Wait until the round finishes."""
    if len(jobs) != len(gpus):
        print(f"  warn: {len(jobs)} jobs vs {len(gpus)} gpus", flush=True)
    running = []
    fail = 0
    for job, gpu in zip(jobs, gpus):
        job["log"].parent.mkdir(parents=True, exist_ok=True)
        print(
            f"  [gpu{gpu}] B={job['batch_size']} "
            f"{job['layer']} K{job['K']} baseλ{job['base_lmbda']} "
            f"resλ{job['res_lmbda']}  {job['log'].name}",
            flush=True,
        )
        proc = subprocess.Popen(
            job["cmd"],
            cwd=str(ORFCV2),
            env=_env(gpu),
            stdout=open(job["log"], "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        running.append((proc, job, gpu))
    leftover = jobs[len(gpus):]
    free = []
    idx = 0
    while running or leftover:
        still = []
        for proc, job, gpu in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, job, gpu))
            else:
                if rc != 0:
                    print(f"  [FAIL] {job['tag']} rc={rc}", flush=True)
                    fail += 1
                else:
                    print(f"  [done] {job['tag']}", flush=True)
                free.append(gpu)
        running = still
        while leftover and free:
            gpu = free.pop(0)
            job = leftover.pop(0)
            job["log"].parent.mkdir(parents=True, exist_ok=True)
            print(
                f"  [gpu{gpu}] B={job['batch_size']} "
                f"{job['layer']} K{job['K']} baseλ{job['base_lmbda']} "
                f"resλ{job['res_lmbda']}  {job['log'].name}",
                flush=True,
            )
            proc = subprocess.Popen(
                job["cmd"],
                cwd=str(ORFCV2),
                env=_env(gpu),
                stdout=open(job["log"], "w"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running.append((proc, job, gpu))
        if running:
            time.sleep(15)
    return fail


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default="2,4,5,6,7")
    p.add_argument("--ckpt_dir", default=str(CKPT_DIR))
    p.add_argument("--K", "--K2", dest="K", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lmbda", type=float, default=0.0,
                   help="Residual-training rate λ (R+λD; 0 = distortion only)")
    return p.parse_args()


def main():
    args = parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    extra = [
        "--lr", str(args.lr),
        "--lmbda", str(args.lmbda),
    ]
    if args.max_images:
        extra += ["--max_images", str(args.max_images)]
    all_jobs = make_jobs(
        Path(args.ckpt_dir), extra, args.K, args.epochs, args.batch_size, args.lmbda,
    )
    pending = [j for j in all_jobs if not j["done"]]
    skipped = len(all_jobs) - len(pending)
    print(f"GPUS={gpus}  selected={len(all_jobs)}  skip={skipped}  pending={len(pending)}")
    print(f"  batch_size={args.batch_size}  res_lmbda={args.lmbda}")
    print(f"  data={IMAGENET_TRAIN_FEAT_STAGE2}")
    print(f"  list={IMAGENET_TRAIN_LIST_STAGE2}")
    for j in all_jobs:
        mark = "skip" if j["done"] else "run"
        print(f"  [{mark}] {j['layer']} K{j['K']:>3} baseλ{j['base_lmbda']} resλ{j['res_lmbda']}")
    fail = 0
    by_layer = {ly: [j for j in pending if j["layer"] == ly] for ly in LAYERS}
    for ly in LAYERS:
        jobs = by_layer[ly]
        if not jobs:
            continue
        print(f"=== round {ly}  n={len(jobs)} ===", flush=True)
        fail += launch_round(gpus, jobs)
    print(f"fail={fail}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
