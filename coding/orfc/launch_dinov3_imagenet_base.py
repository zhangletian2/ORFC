#!/usr/bin/env python
"""Launch DINOv3 ImageNet Soft-PQ base codecs: 4 layers × 4 K.

Round order: blk20 → blk15 → blk10 → blk05
Each round: K4/8/16/64 queued on GPUs 0,1,3,7.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = "/data4/workspace/zlt/featcodec/CoFAI/.venv/bin/python"
SCRIPT = HERE / "run_soft_pq_dinov3.py"
LOGDIR = HERE / "logs"
CKPT_DIR = HERE / "checkpoints" / "dinov3_vitl16"
LOGDIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

GPUS = [0, 1, 3, 7]
LAYERS = ["blk20", "blk15", "blk10", "blk05"]
KS = [4, 8, 16, 64]
LMBDA = 0.5
LR = 3e-4
EPOCHS = 100
EMB = 32
SEED = 42
N_TRAIN = 5000
NORM_MODE = "split_reg_cls_patch"
N_PREFIX = 5


def ckpt_path(layer, K):
    norm_tag = "" if NORM_MODE == "per_image" else f"_{NORM_MODE}"
    stem = (
        f"{layer}_K{K}_emb{EMB}_bt1024_ws_lmbda{LMBDA}{norm_tag}"
        f"_tau0.5_lr{LR}_ep{EPOCHS}_n{N_TRAIN}_s{SEED}"
    )
    return CKPT_DIR / f"{stem}.pt"


def tag_of(layer, K):
    return f"{layer}_K{K}_e{EMB}_lmbda{LMBDA}_{NORM_MODE}_ep{EPOCHS}"


def launch(gpu, layer, K):
    tag = tag_of(layer, K)
    log = LOGDIR / f"train_{tag}.log"
    cmd = [
        PY, "-u", str(SCRIPT),
        "--layer", layer,
        "--K", str(K),
        "--embedding_dim", str(EMB),
        "--epochs", str(EPOCHS),
        "--lr", str(LR),
        "--lmbda", str(LMBDA),
        "--norm_mode", NORM_MODE,
        "--n_prefix", str(N_PREFIX),
        "--batch_size", "32",
        "--gpu", "0",
        "--seed", str(SEED),
        "--max_train_images", str(N_TRAIN),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["PROJECT_ROOT"] = str(HERE.parents[2] / "CoFAI")
    proc = subprocess.Popen(
        cmd, cwd=str(HERE), env=env,
        stdout=open(log, "w"), stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"  GPU{gpu}  pid={proc.pid}  {tag}  log={log.name}", flush=True)
    return proc, tag, log


def wait_jobs(pending):
    """Run ``pending`` (gpu, layer, K) with at most one job per GPU."""
    queue = list(pending)
    running = {}
    finished = []
    while queue or running:
        busy = set(running)
        for gpu in GPUS:
            if gpu in busy or not queue:
                continue
            item = queue.pop(0)
            _, ly, K = item
            running[gpu] = launch(gpu, ly, K)
            time.sleep(1.0)
        for gpu, (proc, tag, log) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                print(f"  DONE  GPU{gpu}  {tag}  exit={rc}  {log.name}", flush=True)
                finished.append((tag, rc))
                del running[gpu]
        if running:
            time.sleep(20)
    return finished


def main():
    print(f"DINOv3 ImageNet Soft-PQ  {len(LAYERS) * len(KS)} jobs  GPUs={GPUS}", flush=True)
    print(
        f"  order={LAYERS}  K={KS}  λ={LMBDA}  ep={EPOCHS}  lr={LR}  "
        f"norm={NORM_MODE} n_prefix={N_PREFIX}",
        flush=True,
    )
    print(f"  ckpt={CKPT_DIR}", flush=True)
    all_rc = []
    for layer in LAYERS:
        jobs = []
        for K in KS:
            ckpt = ckpt_path(layer, K)
            if ckpt.is_file():
                print(f"  SKIP  {ckpt.name} exists", flush=True)
                continue
            jobs.append((None, layer, K))
        if not jobs:
            print(f"  round {layer}: all checkpoints present", flush=True)
            continue
        print(f"\n=== round {layer}  ({len(jobs)} jobs) ===", flush=True)
        all_rc.extend(wait_jobs(jobs))
        time.sleep(5)

    nfail = sum(1 for _, rc in all_rc if rc != 0)
    print(f"\nAll rounds done. jobs={len(all_rc)} fail={nfail}", flush=True)
    for tag, rc in all_rc:
        if rc != 0:
            print(f"  FAIL {tag} exit={rc}", flush=True)
    raise SystemExit(1 if nfail else 0)


if __name__ == "__main__":
    main()
