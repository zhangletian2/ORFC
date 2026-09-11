#!/usr/bin/env python
"""Stage-1 conv2 E/U residual ablations (no ORFC / no PQ).

Three rounds, three jobs per layer in parallel:

    GPU 2 → main     hat X = U(E(X))
    GPU 4 → recon0   hat X = X0 + F_φ(X0, 0)
    GPU 5 → full     hat X = X0 + F_φ(X0, G)

    round 1 blk05 → round 2 blk10 → round 3 blk15 → round 4 blk20

Usage:
    python -u launch_bilinear_stage1_ablations.py --dry_run
    python -u launch_bilinear_stage1_ablations.py
    python -u launch_bilinear_stage1_ablations.py --gpus 2,4,5
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]

from run_bilinear_residual import residual_ckpt_path  # noqa: E402

LAYERS = ["blk05", "blk10", "blk15", "blk20"]
ABLATIONS = ("main", "recon0", "full")


def job_args(layer, ablation, args):
    return SimpleNamespace(
        layer=layer,
        K=4,
        c=4,
        residual_lr=args.lr,
        residual_epochs=args.epochs,
        max_train_images=args.max_train_images,
        n_val=args.n_val,
        seed=args.seed,
        backbone=args.backbone,
        result_dir=args.result_dir,
        residual_quantize=False,
        residual_orfc=False,
        residual_decoder="conv",
        residual_ablation=ablation,
        spatial_down="conv2",
        spatial_up="conv2",
        latent_channels=args.latent_channels,
    )


def ckpt_path(layer, ablation, args):
    return residual_ckpt_path(job_args(layer, ablation, args), "both")


def log_stem(layer, ablation):
    tag = "" if ablation == "full" else f"_ablation_{ablation}"
    return (f"{layer}_bili65_conv2_uconv2_K4_c4_wa_conv_both_pre_noq{tag}")


def build_cmd(layer, ablation, args):
    return [
        args.python, "-u", str(HERE / "run_bilinear_residual.py"),
        "--stage", "residual",
        "--residual_mode", "both",
        "--layer", layer,
        "--spatial_down", "conv2",
        "--spatial_up", "conv2",
        "--residual_decoder", "conv",
        "--residual_ablation", ablation,
        "--latent_channels", str(args.latent_channels),
        "--no-residual_orfc",
        "--no-residual_quantize",
        "--residual_epochs", str(args.epochs),
        "--residual_lr", str(args.lr),
        "--n_val", str(args.n_val),
        "--max_train_images", str(args.max_train_images),
        "--seed", str(args.seed),
        "--backbone", args.backbone,
        "--feat_root", args.feat_root,
        "--result_dir", args.result_dir,
    ]


def run_one(gpu, layer, ablation, args):
    tag = log_stem(layer, ablation)
    ckpt = ckpt_path(layer, ablation, args)
    if args.skip_existing and ckpt.is_file():
        print(f"[GPU {gpu}] skip existing {tag} -> {ckpt.name}", flush=True)
        return None
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TORCH_HOME"] = args.torch_home
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("MKL_NUM_THREADS", "8")
    cmd = build_cmd(layer, ablation, args)
    log_path = Path(args.log_dir) / f"{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[GPU {gpu}] start {tag}", flush=True)
    t0 = time.time()
    with open(log_path, "w") as logf:
        logf.write(" ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(HERE), env=env,
            stdout=logf, stderr=subprocess.STDOUT)
        rc = proc.wait()
    dt = time.time() - t0
    if rc != 0:
        print(f"[GPU {gpu}] FAILED rc={rc} {tag}  {dt:.0f}s  log={log_path}",
              flush=True)
        return tag
    print(f"[GPU {gpu}] done {tag}  {dt:.0f}s", flush=True)
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default="2,4,5")
    p.add_argument("--layers", default=",".join(LAYERS))
    p.add_argument("--ablations", default=",".join(ABLATIONS))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--latent_channels", type=int, default=0,
                   help="Latent channel width C (0 = same as D)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_train_images", type=int, default=5000)
    p.add_argument("--n_val", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", default="dinov2_vitl14")
    p.add_argument("--feat_root", default=str(PROJECT_ROOT / "features"))
    p.add_argument("--result_dir", default=str(HERE / "results" / "bilinear_residual"))
    p.add_argument("--torch_home", default=str(PROJECT_ROOT / "pretrained"))
    p.add_argument("--log_dir", default=str(HERE / "logs" / "bilinear_residual"))
    p.add_argument("--python",
                   default="/home/user/anaconda3/envs/featcodec2/bin/python")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--no_skip_existing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.no_skip_existing:
        args.skip_existing = False
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    layers = [x.strip() for x in args.layers.split(",") if x.strip()]
    ablations = [x.strip() for x in args.ablations.split(",") if x.strip()]
    if len(gpus) < len(ablations):
        raise SystemExit(f"need {len(ablations)} GPUs, got {gpus}")
    wave = list(zip(gpus[:len(ablations)], ablations))

    print("=" * 72, flush=True)
    print("stage-1 conv2 E/U  ablations  (no ORFC / no PQ)", flush=True)
    print(f"  ep={args.epochs}  lr={args.lr}  n={args.max_train_images}  "
          f"n_val={args.n_val}", flush=True)
    print(f"  GPUs: {[g for g, _ in wave]}  ablations={ablations}", flush=True)
    print(f"  rounds: {layers}", flush=True)
    print("=" * 72, flush=True)

    n_train = n_skip = 0
    for r, layer in enumerate(layers, 1):
        print(f"  round {r}  {layer}", flush=True)
        for gpu, ab in wave:
            ckpt = ckpt_path(layer, ab, args)
            if args.skip_existing and ckpt.is_file():
                status = "skip"
                n_skip += 1
            else:
                status = "train"
                n_train += 1
            print(f"    GPU {gpu}  {ab:6s}  {status:5s}  {ckpt.name}", flush=True)
    print(f"Jobs: train={n_train}  skip={n_skip}", flush=True)
    print("=" * 72, flush=True)
    if args.dry_run:
        return

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    failures = []
    for r, layer in enumerate(layers, 1):
        print(f"\n===== ROUND {r}/{len(layers)}  {layer} =====", flush=True)
        jobs = []
        for gpu, ab in wave:
            ckpt = ckpt_path(layer, ab, args)
            if args.skip_existing and ckpt.is_file():
                print(f"[GPU {gpu}] skip existing {log_stem(layer, ab)}",
                      flush=True)
                continue
            jobs.append((gpu, ab))
        if not jobs:
            print(f"  round {r} all skipped", flush=True)
            continue
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futs = {
                pool.submit(run_one, gpu, layer, ab, args): (gpu, ab)
                for gpu, ab in jobs
            }
            for fut in as_completed(futs):
                fail = fut.result()
                if fail:
                    failures.append(fail)
        print(f"===== ROUND {r} done  {layer} =====", flush=True)

    print("\n" + "=" * 72, flush=True)
    if failures:
        print(f"FAILED ({len(failures)}): {failures}", flush=True)
        raise SystemExit(1)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
