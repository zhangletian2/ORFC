#!/usr/bin/env python
"""ShareR + freeze R + K2 pure-KD classification on stage2 ImageNet 5k.

Train:  features/train/dinov2_vitl14_stage2  (disjoint from ORFC 5k)
Eval:   features/test/dinov2_vitl14 + label500
Base:   CSV e32 ORFC checkpoints for blk05/10/15/20

8 rounds, GPUs 0/1/6, one process per card:
  blk05  5 jobs  (3+2)
  blk10  6 jobs  (3+3)
  blk15  5 jobs  (3+2)
  blk20  6 jobs  (3+3)
Residual: K=2 e32, 50 epochs, batch_size=32, shareR, freeze R2, pure KD.
"""
from __future__ import annotations

import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CK = Path("/data4/workspace/zlt/featcodec/ORFC/coding/orfc/checkpoints/dinov2_vitl14")
TRAIN_FEAT = PROJECT / "features" / "train" / "dinov2_vitl14_stage2"
TRAIN_GT = Path("/data4/workspace/zlt/featcodec/utils/imagenet_selected_label5000_stage2.txt")
PY = "/home/user/anaconda3/envs/featcodec2/bin/python"
LOGDIR = HERE / "logs"
LOGDIR.mkdir(exist_ok=True)

GPUS = [0, 1, 6]
ROUND_ORDER = ["blk05", "blk10", "blk15", "blk20"]
CHUNK = 3


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
    return f"{layer}_baseK{K}{extra}_res-K2_shareR_clsKD1_ep50_stage2"


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
    ("blk20",   4, 0.0, 5e-4, 300),
    ("blk20",   8, 0.5, 3e-4, 100),
    ("blk20",  16, 0.5, 5e-4, 100),
    ("blk20",  32, 0.5, 3e-4, 100),
    ("blk20",  64, 0.0, 3e-4, 100),
    ("blk20", 256, 0.0, 5e-4, 100),
]


def chunks(jobs, n=CHUNK):
    for i in range(0, len(jobs), n):
        yield jobs[i:i + n]


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
        "--train_feat_root", str(TRAIN_FEAT),
        "--train_gt_path", str(TRAIN_GT),
        "--epochs", "50",
        "--lr", "3e-4",
        "--batch_size", "32",
        "--max_train_images", "5000",
        "--n_val", "200",
        "--result_suffix", "stage2",
    ]
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
    missing = []
    for layer, K, lmbda, lr, ep in JOBS:
        ckpt = CK / ckpt_name(layer, K, lmbda, lr, ep)
        if not ckpt.is_file():
            missing.append(str(ckpt))
    if missing:
        raise FileNotFoundError("missing checkpoints:\n" + "\n".join(missing))
    if not TRAIN_FEAT.is_dir():
        raise FileNotFoundError(TRAIN_FEAT)
    for layer in ROUND_ORDER:
        n = len(list((TRAIN_FEAT / layer).glob("*.npy")))
        if n < 5000:
            raise FileNotFoundError(
                f"{TRAIN_FEAT / layer} has {n} npy, expected 5000")

    by_layer = defaultdict(list)
    for job in JOBS:
        by_layer[job[0]].append(job)

    all_finished = []
    round_i = 0
    for layer in ROUND_ORDER:
        jobs = by_layer[layer]
        parts = list(chunks(jobs))
        if len(parts) != 2:
            raise ValueError(
                f"{layer}: expected 2 rounds, got {len(parts)} from {len(jobs)} jobs")
        for part in parts:
            round_i += 1
            gpus = GPUS[:len(part)]
            print(
                f"\n======== round {round_i}/8  {layer}  "
                f"n={len(part)}  B=32  GPUs={gpus} ========",
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
