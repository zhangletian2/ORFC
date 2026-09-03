#!/usr/bin/env python
"""Objective ablation: shareR residual K=4, CE vs mse_tail (ΔL_ref).

KD results already exist from launch_shareR_cls_stage2_k4.py.
Compare each to the matched-rate single-stage base:
  blk05 K16+resK4  vs  blk05 K64
  blk10 K16+resK4  vs  blk10 K64
  blk20 K8+resK4   vs  blk20 K32

GPUs 4/5/6, one process per card, 2 rounds of 3.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CK = Path("/data4/workspace/zlt/featcodec/ORFC/coding/orfc/checkpoints/dinov2_vitl14")
TRAIN_FEAT = PROJECT / "features" / "train" / "dinov2_vitl14_stage2"
TRAIN_GT = Path("/data4/workspace/zlt/featcodec/utils/imagenet_selected_label5000_stage2.txt")
PY = "/home/user/anaconda3/envs/featcodec2/bin/python"
LOGDIR = HERE / "logs"
LOGDIR.mkdir(exist_ok=True)

GPUS = [4, 5, 6]


def ckpt_name(layer, K, lmbda, lr, epochs):
    rt = f"_lmbda{lmbda}" if lmbda > 0 else ""
    return f"{layer}_K{K}_emb32_bt1024_ws{rt}_tau0.5_lr{lr}_ep{epochs}_n5000_s42.pt"


# (layer, base_K, lmbda, lr, base_ep, objective)
JOBS = [
    ("blk05", 16, 0.5, 3e-4, 100, "ce"),
    ("blk05", 16, 0.5, 3e-4, 100, "mse_tail"),
    ("blk20",  8, 0.5, 3e-4, 100, "ce"),
    ("blk20",  8, 0.5, 3e-4, 100, "mse_tail"),
    ("blk10", 16, 0.5, 3e-4, 100, "ce"),
    ("blk10", 16, 0.5, 3e-4, 100, "mse_tail"),
]


def tag_of(layer, K, obj):
    return f"{layer}_baseK{K}_res-K4_shareR_{obj}_ep50_stage2"


def launch(gpu, layer, K, lmbda, lr, ep, obj):
    ckpt = CK / ckpt_name(layer, K, lmbda, lr, ep)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    tag = tag_of(layer, K, obj)
    log = LOGDIR / f"{tag}.log"
    cmd = [
        PY, "-u", str(HERE / "run_residual_pq.py"),
        "--base_ckpt", str(ckpt),
        "--layer", layer,
        "--task", "cls",
        "--K", "4",
        "--embedding_dim", "32",
        "--share_base_transform",
        "--train_feat_root", str(TRAIN_FEAT),
        "--train_gt_path", str(TRAIN_GT),
        "--epochs", "50",
        "--lr", "3e-4",
        "--batch_size", "32",
        "--max_train_images", "5000",
        "--n_val", "200",
        "--result_suffix", "stage2",
    ]
    if obj == "ce":
        cmd += ["--res_loss", "task", "--ce_weight", "1", "--lmbda_kd", "0"]
    elif obj == "mse_tail":
        cmd += ["--res_loss", "mse_tail", "--ce_weight", "0", "--lmbda_kd", "0"]
    else:
        raise ValueError(obj)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    proc = subprocess.Popen(
        cmd, cwd=str(HERE), env=env,
        stdout=open(log, "w"), stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"  GPU{gpu}  pid={proc.pid}  {tag}", flush=True)
    return proc, tag


def wait_round(running):
    finished = []
    pending = dict(running)
    while pending:
        for gpu, (proc, tag) in list(pending.items()):
            rc = proc.poll()
            if rc is not None:
                print(f"  DONE  GPU{gpu}  {tag}  exit={rc}", flush=True)
                finished.append((tag, rc))
                del pending[gpu]
        if pending:
            time.sleep(20)
    return finished


def wait_gpus_idle(timeout_s=120):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-compute-apps=pid,process_name",
                 "--format=csv,noheader"],
                text=True,
            )
        except subprocess.CalledProcessError:
            break
        leftover = [
            line for line in out.strip().splitlines()
            if line.strip() and "run_residual_pq" in line
        ]
        if not leftover:
            time.sleep(8)
            return
        time.sleep(3)
    print("  WARN: residual GPU processes still listed after wait", flush=True)


def main():
    for layer, K, lmbda, lr, ep, obj in JOBS:
        ckpt = CK / ckpt_name(layer, K, lmbda, lr, ep)
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
    if not TRAIN_GT.is_file():
        raise FileNotFoundError(TRAIN_GT)

    all_finished = []
    n_rounds = (len(JOBS) + len(GPUS) - 1) // len(GPUS)
    for r in range(n_rounds):
        part = JOBS[r * len(GPUS):(r + 1) * len(GPUS)]
        gpus = GPUS[:len(part)]
        print(
            f"\n======== round {r+1}/{n_rounds}  n={len(part)}  "
            f"GPUs={gpus} ========",
            flush=True,
        )
        running = {}
        for gpu, job in zip(gpus, part):
            proc, tag = launch(gpu, *job)
            running[gpu] = (proc, tag)
        all_finished.extend(wait_round(running))
        wait_gpus_idle()

    n_fail = sum(1 for _, rc in all_finished if rc != 0)
    print(f"all done: {len(all_finished)} jobs, {n_fail} failed", flush=True)


if __name__ == "__main__":
    main()
