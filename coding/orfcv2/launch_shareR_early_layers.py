#!/usr/bin/env python
"""Queue shareR + freeze R + K2 pure-KD jobs for blk05/10/15 e32 bases."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CK = Path("/data4/workspace/zlt/featcodec/ORFC/coding/orfc/checkpoints/dinov2_vitl14")
PY = "/home/user/anaconda3/envs/featcodec2/bin/python"
LOGDIR = HERE / "logs"
LOGDIR.mkdir(exist_ok=True)

# Prefer empty cards; 0/1/3 already have a few GB occupied.
GPUS = [2, 4, 5, 6, 7, 0, 1, 3]


def ckpt_name(layer, K, lmbda, lr, epochs):
    rt = f"_lmbda{lmbda}" if lmbda > 0 else ""
    return f"{layer}_K{K}_emb32_bt1024_ws{rt}_tau0.5_lr{lr}_ep{epochs}_n5000_s42.pt"


def tag_of(layer, K, lmbda, lr):
    extra = ""
    if lmbda == 0.0 and lr == 5e-4:
        extra = "_l0_lr5e4"
    elif lmbda == 0.0:
        extra = "_l0"
    elif lr == 5e-4:
        extra = "_lr5e4"
    return f"{layer}_baseK{K}{extra}_res-K2_shareR_clsKD1_ep50"


# CSV e32 rows for blk05 / blk10 / blk15.
JOBS = [
    ("blk05",   4, 0.5, 3e-4, 100),
    ("blk05",   8, 0.5, 3e-4, 100),
    ("blk05",  16, 0.5, 3e-4, 100),
    ("blk05",  64, 0.5, 5e-4, 100),
    ("blk05", 256, 0.5, 3e-4, 100),
    ("blk10",   4, 0.5, 3e-4, 100),
    ("blk10",   8, 0.5, 3e-4, 100),
    ("blk10",  16, 0.5, 3e-4, 100),
    ("blk10",  64, 0.0, 3e-4, 100),
    ("blk10", 256, 0.0, 5e-4, 100),
    ("blk10", 256, 0.5, 3e-4, 100),
    ("blk15",   4, 0.5, 3e-4, 100),
    ("blk15",   8, 0.5, 3e-4, 100),
    ("blk15",  16, 0.5, 3e-4, 100),
    ("blk15",  64, 0.5, 5e-4, 100),
    ("blk15", 256, 0.5, 3e-4, 100),
]


def launch(gpu, layer, K, lmbda, lr, ep):
    ckpt = CK / ckpt_name(layer, K, lmbda, lr, ep)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    tag = tag_of(layer, K, lmbda, lr)
    log = LOGDIR / f"{tag}.log"
    cmd = [
        PY, "-u", str(HERE / "run_residual_pq.py"),
        "--base_ckpt", str(ckpt),
        "--layer", layer,
        "--task", "cls",
        "--K", "2",
        "--embedding_dim", "32",
        "--res_loss", "task",
        "--ce_weight", "0",
        "--lmbda_kd", "1.0",
        "--share_base_transform",
        "--epochs", "50",
        "--lr", "3e-4",
        "--batch_size", "32",
        "--max_train_images", "5000",
        "--n_val", "200",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    proc = subprocess.Popen(
        cmd, cwd=str(HERE), env=env,
        stdout=open(log, "w"), stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"GPU{gpu}  pid={proc.pid}  {tag}", flush=True)
    return proc, tag


def main():
    queue = list(JOBS)
    running = {}  # gpu -> (Popen, tag)
    finished = []
    while queue or running:
        for gpu, (proc, tag) in list(running.items()):
            if proc.poll() is not None:
                print(f"DONE  GPU{gpu}  {tag}  exit={proc.returncode}", flush=True)
                finished.append(tag)
                del running[gpu]
        free = [g for g in GPUS if g not in running]
        while queue and free:
            gpu = free.pop(0)
            layer, K, lmbda, lr, ep = queue.pop(0)
            proc, tag = launch(gpu, layer, K, lmbda, lr, ep)
            running[gpu] = (proc, tag)
        if queue or running:
            time.sleep(20)
    print(f"all done: {len(finished)} jobs", flush=True)


if __name__ == "__main__":
    main()
