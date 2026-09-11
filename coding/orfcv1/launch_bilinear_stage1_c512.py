#!/usr/bin/env python
"""Stage-1 conv2 E(D->C)/U(C->D) spatial pretraining with channel bottleneck.

Trains BilinearSpatialCodec(D=1024, C=512, conv2/conv2) + residual F_phi
for each ablation on each layer, one job per GPU.

Aligns with ``launch_bilinear_stage1_ablations.py``:
    ep=30, lr=3e-4, n=5000, seed=42, no ORFC / no PQ.

Usage:
    python -u launch_bilinear_stage1_c512.py --dry_run
    python -u launch_bilinear_stage1_c512.py
    python -u launch_bilinear_stage1_c512.py --gpus 2,4,5 --layers blk05,blk20
    python -u launch_bilinear_stage1_c512.py --ablations main --gpus 2
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
LATENT_C = 512


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
        cls_mode=args.cls_mode,
        ortho_init=bool(args.ortho_init),
    )


def ckpt_path(layer, ablation, args):
    return residual_ckpt_path(job_args(layer, ablation, args), "both")


def log_stem(layer, ablation, args):
    tag = "" if ablation == "full" else f"_ablation_{ablation}"
    cls = "_clsid" if args.cls_mode == "identity" else ""
    ortho = "_ortho" if args.ortho_init else ""
    return (f"{layer}_bili65_conv2_uconv2_C{args.latent_channels}{ortho}{cls}"
            f"_K4_c4_wa_conv_both_pre_noq{tag}")


def build_cmd(layer, ablation, args):
    cmd = [
        args.python, "-u", str(HERE / "run_bilinear_residual.py"),
        "--stage", "residual",
        "--residual_mode", "both",
        "--layer", layer,
        "--spatial_down", "conv2",
        "--spatial_up", "conv2",
        "--residual_decoder", "conv",
        "--residual_ablation", ablation,
        "--latent_channels", str(args.latent_channels),
        "--cls_mode", args.cls_mode,
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
    if args.ortho_init:
        cmd.append("--ortho_init")
    if not args.skip_existing:
        cmd.append("--no-skip_existing")
    return cmd


def run_one(gpu, layer, ablation, args):
    tag = log_stem(layer, ablation, args)
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
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--gpus", default="2,4,5")
    p.add_argument("--layers", default=",".join(LAYERS))
    p.add_argument("--ablations", default=",".join(ABLATIONS))
    p.add_argument("--latent_channels", type=int, default=LATENT_C)
    p.add_argument("--cls_mode", default="learned",
                   choices=["learned", "identity"])
    p.add_argument("--ortho_init", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_train_images", type=int, default=5000)
    p.add_argument("--n_val", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", default="dinov2_vitl14")
    p.add_argument("--feat_root", default=str(PROJECT_ROOT / "features"))
    p.add_argument("--result_dir",
                   default=str(HERE / "results" / "bilinear_residual"))
    p.add_argument("--torch_home",
                   default=str(PROJECT_ROOT / "pretrained"))
    p.add_argument("--log_dir",
                   default=str(HERE / "logs" / "bilinear_residual"))
    p.add_argument("--python",
                   default="/home/user/anaconda3/envs/featcodec2/bin/python")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_existing", action=argparse.BooleanOptionalAction,
                   default=True)
    return p.parse_args()


def main():
    args = parse_args()
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    layers = [x.strip() for x in args.layers.split(",") if x.strip()]
    ablations = [x.strip() for x in args.ablations.split(",") if x.strip()]
    if len(gpus) < len(ablations):
        raise SystemExit(f"need {len(ablations)} GPUs, got {gpus}")
    wave = list(zip(gpus[:len(ablations)], ablations))

    print("=" * 72, flush=True)
    print(f"stage-1 conv2 E(D->C={args.latent_channels})/U(C->D)  "
          f"ablations  (no ORFC / no PQ)", flush=True)
    print(f"  ep={args.epochs}  lr={args.lr}  n={args.max_train_images}  "
          f"n_val={args.n_val}  C={args.latent_channels}", flush=True)
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
            print(f"    GPU {gpu}  {ab:6s}  {status:5s}  {ckpt.name}",
                  flush=True)
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
                print(f"[GPU {gpu}] skip existing "
                      f"{log_stem(layer, ab, args)}", flush=True)
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
