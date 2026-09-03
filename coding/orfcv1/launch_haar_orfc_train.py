#!/usr/bin/env python
"""Launch Haar-65 + ORFC training aligned with ``orfc/best config.csv``.

One round per layer; the five Ks run in parallel on GPUs 2/4/5/6/7:

    GPU 2 → K=4     GPU 4 → K=8     GPU 5 → K=16
    GPU 6 → K=64    GPU 7 → K=256

Rounds: blk05 → blk10 → blk15 → blk20.  The only 300-epoch job is
blk20 K=4, so it sits in the last round and does not block anything after.

Only K ∈ {4, 8, 16, 64, 256} and e32 are taken.  Duplicate (layer, K)
rows keep the first CSV occurrence (blk10 K=256 e32 appears twice).

Usage:
    python -u launch_haar_orfc_train.py --dry_run
    python -u launch_haar_orfc_train.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORFC_DIR = HERE.parent / "orfc"
EVAL_DIR = ORFC_DIR / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from run_best_config_eval_only import load_best_configs  # noqa: E402

LAYERS = ["blk05", "blk10", "blk15", "blk20"]
KEEP_K = {4, 8, 16, 64, 256}
KEEP_EMB = 32
K_ORDER = (4, 8, 16, 64, 256)


def codec_stem(cfg, args):
    bt_tag = f"bt{args.bottleneck_dim}" if args.bottleneck_dim > 0 else "noBt"
    ws_tag = "ws" if args.warm_start_opq else "km"
    rate_tag = f"_lmbda{cfg.lmbda}"
    tau_tag = f"_tau{args.tau_start}" if args.tau_start > 0 else ""
    tau_end_tag = (
        f"_te{args.tau_end}"
        if args.tau_start > 0 and args.tau_end != 0.005
        else ""
    )
    tau_sched_tag = (
        f"_ts{args.tau_schedule[:3]}"
        if args.tau_schedule != "exponential"
        else ""
    )
    return (
        f"{cfg.layer}_haar65_K{cfg.K}_emb{cfg.embedding_dim}"
        f"_{bt_tag}_{ws_tag}{rate_tag}{tau_tag}{tau_end_tag}{tau_sched_tag}"
        f"_lr{cfg.lr}_ep{cfg.epochs}_n{args.max_train_images}_s{args.seed}"
    )


def ckpt_path(cfg, args):
    return Path(args.ckpt_dir) / args.backbone / f"{codec_stem(cfg, args)}.pt"


def select_configs(csv_path, default_lmbda, default_lr, default_epochs):
    raw = load_best_configs(
        csv_path,
        default_lmbda=default_lmbda,
        default_lr=default_lr,
        default_epochs=default_epochs,
    )
    seen = set()
    selected = []
    for cfg in raw:
        if cfg.K not in KEEP_K or cfg.embedding_dim != KEEP_EMB:
            continue
        key = (cfg.layer, cfg.K)
        if key in seen:
            continue
        seen.add(key)
        selected.append(cfg)
    return selected


def jobs_by_layer(configs):
    grouped = defaultdict(list)
    for cfg in configs:
        grouped[cfg.layer].append(cfg)
    for layer in grouped:
        grouped[layer].sort(key=lambda c: K_ORDER.index(c.K) if c.K in K_ORDER else c.K)
    return grouped


def round_assignments(layer_jobs, gpus):
    """Pair this layer's Ks with GPUs. Extra jobs spill into a second wave."""
    waves = []
    remaining = list(layer_jobs)
    while remaining:
        wave = remaining[:len(gpus)]
        remaining = remaining[len(gpus):]
        waves.append(list(zip(gpus[:len(wave)], wave)))
    return waves


def haar_ckpt(cfg, args):
    return (
        HERE / "results" / "global_residual" / args.backbone
        / f"{cfg.layer}_global_haar_joint_D_lr0.0003_ep30_s42.pt"
    )


def build_cmd(cfg, args):
    cmd = [
        args.python, "-u", str(HERE / "run_haar_orfc_train.py"),
        "--layer", cfg.layer,
        "--K", str(cfg.K),
        "--embedding_dim", str(cfg.embedding_dim),
        "--bottleneck_dim", str(args.bottleneck_dim),
        "--lmbda", str(cfg.lmbda),
        "--lr", str(cfg.lr),
        "--epochs", str(cfg.epochs),
        "--tau_start", str(args.tau_start),
        "--tau_end", str(args.tau_end),
        "--tau_schedule", args.tau_schedule,
        "--max_train_images", str(args.max_train_images),
        "--batch_size", str(args.batch_size),
        "--seed", str(args.seed),
        "--backbone", args.backbone,
        "--feat_root", args.feat_root,
        "--gt_path", args.gt_path,
        "--haar_ckpt", str(haar_ckpt(cfg, args)),
        "--ckpt_dir", args.ckpt_dir,
        "--result_dir", args.result_dir,
    ]
    if args.warm_start_opq:
        cmd.append("--warm_start_opq")
    else:
        cmd.append("--no-warm_start_opq")
    if args.skip_baselines:
        cmd.append("--skip_baselines")
    return cmd


def log_stem(cfg):
    return (f"{cfg.layer}_haar65_K{cfg.K}_emb{cfg.embedding_dim}"
            f"_lmbda{cfg.lmbda}_lr{cfg.lr}_ep{cfg.epochs}")


def run_one(gpu, cfg, args):
    """Run a single job on ``gpu``.  Returns failure stem or None."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TORCH_HOME"] = args.torch_home
    env["PYTHONUNBUFFERED"] = "1"
    ckpt = ckpt_path(cfg, args)
    tag = log_stem(cfg)
    if args.skip_existing and ckpt.is_file():
        print(f"[GPU {gpu}] skip existing {tag} -> {ckpt.name}", flush=True)
        return None
    haar = haar_ckpt(cfg, args)
    if not haar.is_file():
        print(f"[GPU {gpu}] MISSING HAAR {haar}", flush=True)
        return tag
    cmd = build_cmd(cfg, args)
    log_path = Path(args.log_dir) / f"{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[GPU {gpu}] start {tag}", flush=True)
    with open(log_path, "w") as logf:
        logf.write(" ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(HERE), env=env,
            stdout=logf, stderr=subprocess.STDOUT,
        )
        rc = proc.wait()
    if rc != 0:
        print(f"[GPU {gpu}] FAILED rc={rc} {tag}  log={log_path}", flush=True)
        return tag
    print(f"[GPU {gpu}] done {tag}", flush=True)
    return None


def print_plan(grouped, gpus, args):
    print("=" * 72)
    print("Haar-65 ORFC train schedule  (one layer per round)")
    print(f"  csv   : {args.best_csv}")
    print(f"  python: {args.python}")
    print(f"  GPUs  : {gpus}  (K={list(K_ORDER)})")
    print(f"  rounds: {' → '.join(LAYERS)}")
    print("=" * 72)
    n = 0
    for r, layer in enumerate(LAYERS, 1):
        jobs = grouped.get(layer, [])
        waves = round_assignments(jobs, gpus)
        print(f"\nRound {r}/{len(LAYERS)}  {layer}  "
              f"{len(jobs)} job(s)  {len(waves)} wave(s)")
        for w, wave in enumerate(waves, 1):
            if len(waves) > 1:
                print(f"  wave {w}:")
            for gpu, cfg in wave:
                ckpt = ckpt_path(cfg, args)
                exists = "exists" if ckpt.is_file() else "train"
                print(f"  GPU {gpu}  K={cfg.K:<3d}  e{cfg.embedding_dim}  "
                      f"λ={cfg.lmbda:g}  lr={cfg.lr:g}  ep={cfg.epochs:<3d}  "
                      f"{exists:6s}  {cfg.config_text}")
                n += 1
    print(f"\nTotal jobs: {n}")
    print("=" * 72)


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--best_csv", default=str(ORFC_DIR / "best config.csv"))
    p.add_argument("--python", default="/home/user/anaconda3/envs/featcodec2/bin/python")
    p.add_argument("--torch_home",
                   default="/data4/workspace/zlt/featcodec/ORFC/pretrained")
    p.add_argument("--gpus", default="2,4,5,6,7",
                   help="Physical GPU ids, one per K in this round")
    p.add_argument("--default_lmbda", type=float, default=0.5)
    p.add_argument("--default_lr", type=float, default=3e-4)
    p.add_argument("--default_epochs", type=int, default=100)
    p.add_argument("--bottleneck_dim", type=int, default=1024)
    p.add_argument("--warm_start_opq", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--tau_start", type=float, default=0.5)
    p.add_argument("--tau_end", type=float, default=0.005)
    p.add_argument("--tau_schedule", default="exponential")
    p.add_argument("--max_train_images", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", default="dinov2_vitl14")
    p.add_argument("--feat_root",
                   default=str(HERE.parents[1] / "features"))
    p.add_argument("--gt_path",
                   default=str(HERE.parents[1] / "utils"
                               / "imagenet_selected_label500.txt"))
    p.add_argument("--ckpt_dir", default=str(HERE / "checkpoints"))
    p.add_argument("--result_dir", default=str(HERE / "results" / "haar_orfc"))
    p.add_argument("--log_dir", default=str(HERE / "logs" / "haar_orfc"))
    p.add_argument("--skip_existing", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--skip_baselines", action="store_true")
    p.add_argument("--strict", action="store_true",
                   help="Stop the remaining rounds after a failure")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def run_wave(wave, args):
    failures = []
    with ThreadPoolExecutor(max_workers=len(wave)) as pool:
        futs = {
            pool.submit(run_one, gpu, cfg, args): (gpu, cfg)
            for gpu, cfg in wave
        }
        for fut in as_completed(futs):
            gpu, cfg = futs[fut]
            tag = fut.result()
            if tag:
                failures.append(tag)
    return failures


def main():
    args = parse_args()
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise SystemExit("--gpus is empty")

    configs = select_configs(
        args.best_csv, args.default_lmbda, args.default_lr, args.default_epochs)
    grouped = jobs_by_layer(configs)
    print_plan(grouped, gpus, args)
    if args.dry_run:
        return

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    failures = []
    for r, layer in enumerate(LAYERS, 1):
        jobs = grouped.get(layer, [])
        if not jobs:
            continue
        waves = round_assignments(jobs, gpus)
        print(f"\n=== round {r}/{len(LAYERS)} {layer} ===", flush=True)
        for w, wave in enumerate(waves, 1):
            if len(waves) > 1:
                print(f"  wave {w}/{len(waves)}", flush=True)
            fails = run_wave(wave, args)
            failures.extend(fails)
            if fails and args.strict:
                raise SystemExit(f"strict stop after {fails}")
        print(f"=== round {r} {layer} finished ===", flush=True)

    if failures:
        raise SystemExit(f"{len(failures)} job(s) failed: {failures}")
    print("all rounds finished", flush=True)


if __name__ == "__main__":
    main()
