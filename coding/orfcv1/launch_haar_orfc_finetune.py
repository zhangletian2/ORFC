#!/usr/bin/env python
"""Launch blk05 Haar+ORFC joint fine-tune: 3 Ks × 2 temperatures on 6 GPUs.

    GPU 1  K=4    τ=0.5→0.005
    GPU 2  K=4    τ=1.5→0.005
    GPU 4  K=16   τ=0.5→0.005
    GPU 5  K=16   τ=1.5→0.005
    GPU 6  K=256  τ=0.5→0.005
    GPU 7  K=256  τ=1.5→0.005

Usage:
    python -u launch_haar_orfc_finetune.py --dry_run
    python -u launch_haar_orfc_finetune.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent

JOBS = [
    (1, 4, 0.5),
    (2, 4, 1.5),
    (4, 16, 0.5),
    (5, 16, 1.5),
    (6, 256, 0.5),
    (7, 256, 1.5),
]


def log_stem(K, tau, args):
    return (f"blk05_haar65_jointft_K{K}_lmbda{args.lmbda}"
            f"_tau{tau}_lr{args.lr}_ep{args.epochs}")


def ckpt_path(K, tau, args):
    tau_tag = f"_tau{tau}" if tau > 0 else ""
    tau_end_tag = (
        f"_te{args.tau_end}"
        if tau > 0 and args.tau_end != 0.005
        else ""
    )
    name = (
        f"blk05_haar65_jointft_K{K}_emb32_bt1024_ws_lmbda{args.lmbda}"
        f"{tau_tag}{tau_end_tag}_lr{args.lr}_ep{args.epochs}"
        f"_n{args.max_train_images}_s{args.seed}.pt"
    )
    return Path(args.ckpt_dir) / args.backbone / name


def build_cmd(K, tau, args):
    cmd = [
        args.python, "-u", str(HERE / "run_haar_orfc_finetune.py"),
        "--layer", "blk05",
        "--K", str(K),
        "--lmbda", str(args.lmbda),
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--tau_start", str(tau),
        "--tau_end", str(args.tau_end),
        "--batch_size", str(args.batch_size),
        "--max_train_images", str(args.max_train_images),
        "--n_val", str(args.n_val),
        "--seed", str(args.seed),
        "--backbone", args.backbone,
        "--feat_root", args.feat_root,
        "--gt_path", args.gt_path,
        "--ckpt_dir", args.ckpt_dir,
        "--result_dir", args.result_dir,
        "--orfc_train_epochs", str(args.orfc_train_epochs),
    ]
    if args.skip_baselines:
        cmd.append("--skip_baselines")
    return cmd


def run_one(gpu, K, tau, args):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TORCH_HOME"] = args.torch_home
    env["PYTHONUNBUFFERED"] = "1"
    tag = log_stem(K, tau, args)
    ckpt = ckpt_path(K, tau, args)
    if args.skip_existing and ckpt.is_file():
        print(f"[GPU {gpu}] skip existing {tag} -> {ckpt.name}", flush=True)
        return None
    cmd = build_cmd(K, tau, args)
    log_path = Path(args.log_dir) / f"{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[GPU {gpu}] start {tag}", flush=True)
    with open(log_path, "w") as logf:
        logf.write(" ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(HERE), env=env,
            stdout=logf, stderr=subprocess.STDOUT)
        rc = proc.wait()
    if rc != 0:
        print(f"[GPU {gpu}] FAILED rc={rc} {tag}  log={log_path}", flush=True)
        return tag
    print(f"[GPU {gpu}] done {tag}", flush=True)
    return None


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--python", default="/home/user/anaconda3/envs/featcodec2/bin/python")
    p.add_argument("--torch_home",
                   default="/data4/workspace/zlt/featcodec/ORFC/pretrained")
    p.add_argument("--lmbda", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--tau_end", type=float, default=0.005)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_train_images", type=int, default=5000)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", default="dinov2_vitl14")
    p.add_argument("--orfc_train_epochs", type=int, default=100)
    p.add_argument("--feat_root",
                   default=str(HERE.parents[1] / "features"))
    p.add_argument("--gt_path",
                   default=str(HERE.parents[1] / "utils"
                               / "imagenet_selected_label500.txt"))
    p.add_argument("--ckpt_dir", default=str(HERE / "checkpoints"))
    p.add_argument("--result_dir",
                   default=str(HERE / "results" / "haar_orfc_jointft"))
    p.add_argument("--log_dir",
                   default=str(HERE / "logs" / "haar_orfc_jointft"))
    p.add_argument("--skip_existing", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--skip_baselines", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 72)
    print("blk05 Haar+ORFC joint fine-tune")
    print(f"  ep={args.epochs}  lr={args.lr}  λ={args.lmbda}  "
          f"bs={args.batch_size}")
    print("=" * 72)
    for gpu, K, tau in JOBS:
        ckpt = ckpt_path(K, tau, args)
        exists = "exists" if ckpt.is_file() else "train"
        print(f"  GPU {gpu}  K={K:<3d}  τ={tau}→{args.tau_end}  {exists}")
    print("=" * 72)
    if args.dry_run:
        return

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    failures = []
    with ThreadPoolExecutor(max_workers=len(JOBS)) as pool:
        futs = {
            pool.submit(run_one, gpu, K, tau, args): (gpu, K, tau)
            for gpu, K, tau in JOBS
        }
        for fut in as_completed(futs):
            tag = fut.result()
            if tag:
                failures.append(tag)
    if failures:
        raise SystemExit(f"{len(failures)} job(s) failed: {failures}")
    print("all jobs finished", flush=True)


if __name__ == "__main__":
    main()
