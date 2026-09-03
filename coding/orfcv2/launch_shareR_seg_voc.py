#!/usr/bin/env python
"""ShareR + freeze R + K2 pure-KD segmentation on VOC slide features.

Train:  features/train/dinov2_vitl14_voc  (flattened slides, path-mode)
Eval:   voc2012_100 mIoU (--eval_seg implied by --task seg)
Base:   CSV e32 ORFC checkpoints for blk05/10/15/20

GPUs 2,4,5,6,7 for 5-job layers; blk10/blk20 add GPU 0 as the 6th
job in the same wave (Isaac Sim already sits on 0, ~2.5GB).
Residual: K=2 e32, 20 epochs, shareR, freeze R2, pure KD.
Batch: blk05=4, others=8 (1370 tokens, same ballpark as ADE).
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
TRAIN_FEAT = Path("/data4/workspace/zlt/featcodec/features/train/dinov2_vitl14_voc")
PY = "/home/user/anaconda3/envs/featcodec2/bin/python"
LOGDIR = HERE / "logs"
LOGDIR.mkdir(exist_ok=True)

GPUS_5 = [2, 4, 5, 6, 7]
GPUS_6 = [2, 4, 5, 6, 7, 0]
ROUND_ORDER = ["blk05", "blk10", "blk15", "blk20"]


def gpus_for(n_jobs):
    if n_jobs == 5:
        return list(GPUS_5)
    if n_jobs == 6:
        return list(GPUS_6)
    raise ValueError(f"expected 5 or 6 jobs, got {n_jobs}")


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
    return f"{layer}_baseK{K}{extra}_res-K2_shareR_segKD1_ep20_voc"


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


def batch_size_for(layer):
    return 4 if layer == "blk05" else 8


def residual_procs_for_layer(layer):
    """PIDs of live run_residual_pq.py jobs for this layer."""
    try:
        out = subprocess.check_output(["pgrep", "-af", "run_residual_pq.py"], text=True)
    except subprocess.CalledProcessError:
        return []
    pids = []
    for line in out.splitlines():
        if f"--layer {layer}" in line and "launch_shareR_seg_voc" not in line:
            pids.append(int(line.split()[0]))
    return pids


def wait_existing_layer(layer):
    pids = residual_procs_for_layer(layer)
    if not pids:
        return []
    print(f"  wait existing {layer} pids={pids}", flush=True)
    pending = set(pids)
    finished = []
    while pending:
        still = set(residual_procs_for_layer(layer))
        for pid in list(pending):
            if pid not in still:
                print(f"  DONE  existing pid={pid}  {layer}", flush=True)
                finished.append((f"{layer}_pid{pid}", 0))
                pending.discard(pid)
        if pending:
            time.sleep(20)
    return finished


def launch(gpu, layer, K, lmbda, lr, ep):
    ckpt = CK / ckpt_name(layer, K, lmbda, lr, ep)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    tag = tag_of(layer, K, lmbda, lr)
    bs = batch_size_for(layer)
    log = LOGDIR / f"{tag}.log"
    cmd = [
        PY, "-u", str(HERE / "run_residual_pq.py"),
        "--base_ckpt", str(ckpt),
        "--layer", layer,
        "--task", "seg",
        "--K", "2",
        "--embedding_dim", "32",
        "--res_loss", "task",
        "--ce_weight", "0",
        "--lmbda_kd", "1.0",
        "--share_base_transform",
        "--train_feat_root", str(TRAIN_FEAT),
        "--path_mode",
        "--token_hw", "37,37",
        "--eval_seg",
        "--epochs", "20",
        "--lr", "3e-4",
        "--batch_size", str(bs),
        "--max_train_images", "10000",
        "--n_val", "50",
        "--result_suffix", "voc_ep20",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    proc = subprocess.Popen(
        cmd, cwd=str(HERE), env=env,
        stdout=open(log, "w"), stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"  GPU{gpu}  pid={proc.pid}  B={bs}  {tag}", flush=True)
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
                 "--query-compute-apps=gpu_bus_id,pid,used_memory,process_name",
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

    by_layer = defaultdict(list)
    for job in JOBS:
        by_layer[job[0]].append(job)

    all_finished = []
    for round_i, layer in enumerate(ROUND_ORDER, start=1):
        jobs = by_layer[layer]
        gpus = gpus_for(len(jobs))
        bs = batch_size_for(layer)
        print(
            f"\n======== round {round_i}/4  {layer}  "
            f"n={len(jobs)}  B={bs}  GPUs={gpus} ========",
            flush=True,
        )
        existing = residual_procs_for_layer(layer)
        if existing:
            all_finished.extend(wait_existing_layer(layer))
        else:
            running = {}
            for gpu, job in zip(gpus, jobs):
                proc, tag = launch(gpu, *job)
                running[gpu] = (proc, tag)
            all_finished.extend(wait_round(running))
        wait_gpus_idle()

    n_fail = sum(1 for _, rc in all_finished if rc != 0)
    print(f"all done: {len(all_finished)} jobs, {n_fail} failed", flush=True)


if __name__ == "__main__":
    main()
