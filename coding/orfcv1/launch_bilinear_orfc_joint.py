#!/usr/bin/env python
"""Joint conv2 spatial + ORFC from OPQ.  Aligns with linear4+ORFC.

Stage-1: existing unquantized ep30 (no retrain).  Default ablation is
recon0 (``hat X = X0 + F_φ(X0, 0)``).  Pass ``--residual_ablation main``
for ``hat X = U(ORFC(E(X)))``; ckpt tag becomes ``conv2_main_jointopq``.

Joint: λ=0.5, ORFC lr=3e-4, E/U(/F_φ) lr=3e-5, τ=2.0 fixed, 100 epoch,
val-best spatial+ORFC restore.

One K per GPU in a single wave when 6 GPUs are given:

    GPU 1 → K=4      GPU 2 → K=8      GPU 4 → K=16
    GPU 5 → K=64     GPU 6 → K=256    GPU 7 → K=512

    round 1 blk05 → round 2 blk10 → round 3 blk15 → round 4 blk20

Usage:
    python -u launch_bilinear_orfc_joint.py --dry_run
    python -u launch_bilinear_orfc_joint.py
    python -u launch_bilinear_orfc_joint.py --residual_ablation main --gpus 1,2,4,5,6,7 --layers blk05,blk10,blk15,blk20
"""

from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent

from run_bilinear_orfc_joint import (  # noqa: E402
    default_spatial_ckpt, joint_stem, orfc_ckpt_path, spatial_tag,
)

K_ORDER = (4, 8, 16, 64, 256, 512)
SPATIAL_LR_SCALE = 0.1
TAU = 2.0
LAYERS = ["blk05", "blk10", "blk15", "blk20"]


def job_args(layer, K, args):
    ns = argparse.Namespace(**vars(args))
    ns.layer = layer
    ns.K = K
    ns.embedding_dim = 32
    ns.bottleneck_dim = 1024
    ns.tau_start = TAU
    ns.tau_end = TAU
    ns.tau_schedule = "constant"
    ns.spatial_lr_scale = args.spatial_lr_scale
    ns.latent_channels = args.latent_channels
    return ns


def log_stem(layer, K, args):
    slr = args.spatial_lr_scale
    slr_tag = "" if abs(slr - 1.0) < 1e-12 else f"_hlr{slr:g}"
    return (f"{layer}_{spatial_tag(args)}_jointopq_K{K}_lmbda{args.lmbda}"
            f"_tau{TAU}_te{TAU}_tscon{slr_tag}_bv_lr{args.lr}_ep{args.epochs}")


def ckpt_path(layer, K, args):
    return orfc_ckpt_path(job_args(layer, K, args))


def stage1_ckpt(layer, args):
    if args.residual_ckpt:
        return Path(args.residual_ckpt)
    return default_spatial_ckpt(
        layer, args.backbone, args.residual_dir,
        ablation=args.residual_ablation)


def build_cmd(layer, K, args):
    cmd = [
        args.python, "-u", str(HERE / "run_bilinear_orfc_joint.py"),
        "--layer", layer,
        "--K", str(K),
        "--lmbda", str(args.lmbda),
        "--lr", str(args.lr),
        "--spatial_lr_scale", str(args.spatial_lr_scale),
        "--epochs", str(args.epochs),
        "--residual_ckpt", str(stage1_ckpt(layer, args)),
        "--residual_ablation", args.residual_ablation,
        "--tau_start", str(TAU),
        "--tau_end", str(TAU),
        "--tau_schedule", "constant",
        "--batch_size", str(args.batch_size),
        "--latent_channels", str(args.latent_channels),
        "--max_train_images", str(args.max_train_images),
        "--n_val", str(args.n_val),
        "--seed", str(args.seed),
        "--backbone", args.backbone,
        "--feat_root", args.feat_root,
        "--gt_path", args.gt_path,
        "--ckpt_dir", args.ckpt_dir,
        "--result_dir", args.result_dir,
        "--residual_dir", args.residual_dir,
    ]
    if args.skip_baselines:
        cmd.append("--skip_baselines")
    return cmd


def run_one(gpu, layer, K, args):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TORCH_HOME"] = args.torch_home
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("MKL_NUM_THREADS", "8")
    tag = log_stem(layer, K, args)
    ckpt = ckpt_path(layer, K, args)
    lock = ckpt.with_suffix(ckpt.suffix + ".training")
    if args.skip_existing and (ckpt.is_file() or lock.is_file()):
        why = "lock" if lock.is_file() else "ckpt"
        print(f"[GPU {gpu}] skip existing {tag} ({why}) -> {ckpt.name}",
              flush=True)
        return None
    spat = stage1_ckpt(layer, args)
    if not spat.is_file():
        print(f"[GPU {gpu}] MISSING stage-1 {args.residual_ablation} {spat}",
              flush=True)
        return tag
    cmd = build_cmd(layer, K, args)
    log_path = Path(args.log_dir) / f"{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    print(f"[GPU {gpu}] start {tag}", flush=True)
    lock.touch()
    try:
        with open(log_path, "w") as logf:
            logf.write(" ".join(cmd) + "\n\n")
            logf.flush()
            proc = subprocess.Popen(
                cmd, cwd=str(HERE), env=env,
                stdout=logf, stderr=subprocess.STDOUT)
            rc = proc.wait()
    finally:
        if lock.is_file():
            lock.unlink()
    if rc != 0:
        print(f"[GPU {gpu}] FAILED rc={rc} {tag}  log={log_path}", flush=True)
        return tag
    print(f"[GPU {gpu}] done {tag}", flush=True)
    return None


def k_waves(gpus, ks):
    if not gpus:
        raise SystemExit("need at least 1 GPU")
    waves = []
    for i in range(0, len(ks), len(gpus)):
        chunk = ks[i:i + len(gpus)]
        waves.append(list(zip(gpus[:len(chunk)], chunk)))
    return waves


def print_plan(gpus, layers, args):
    waves = k_waves(gpus, K_ORDER)
    print("=" * 72)
    print(f"joint conv2 {args.residual_ablation}+ORFC  "
          f"(reuse stage-1 ep30, spatial lr = ORFC×{args.spatial_lr_scale:g})")
    spat_lr = args.lr * args.spatial_lr_scale
    extra = "" if args.residual_ablation == "main" else "/F_φ"
    print(f"  ep={args.epochs}  ORFC lr={args.lr}  E/U{extra} lr={spat_lr:g}  "
          f"λ={args.lmbda}")
    print(f"  τ={TAU} fixed  val={args.n_val}/{args.max_train_images}")
    print(f"  GPUs: {gpus}  K={list(K_ORDER)}")
    print(f"  layers: {' → '.join(layers)}")
    print("=" * 72)
    n_train = n_skip = n_miss = 0
    for r, layer in enumerate(layers, 1):
        spat = stage1_ckpt(layer, args)
        spat_st = "exists" if spat.is_file() else "MISSING"
        print(f"\nLayer {r}/{len(layers)}  {layer}  "
              f"stage1({args.residual_ablation})={spat.name}  {spat_st}")
        if not spat.is_file():
            n_miss += 1
        for w, wave in enumerate(waves, 1):
            print(f"  wave {w}:")
            for gpu, K in wave:
                ckpt = ckpt_path(layer, K, args)
                if args.skip_existing and ckpt.is_file():
                    exists = "skip"
                    n_skip += 1
                else:
                    exists = "train"
                    n_train += 1
                print(f"    GPU {gpu}  K={K:<3d}  {exists:5s}  {ckpt.name}")
    print(f"\nJobs: train={n_train}  skip={n_skip}  "
          f"stage1_missing={n_miss}")
    print("=" * 72)
    return n_miss


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--python",
                   default="/home/user/anaconda3/envs/featcodec2/bin/python")
    p.add_argument("--torch_home",
                   default="/data4/workspace/zlt/featcodec/ORFC/pretrained")
    p.add_argument("--gpus", default="1,2,4,5,6,7")
    p.add_argument("--layers", default=",".join(LAYERS))
    p.add_argument("--residual_ckpt", default="",
                   help="Override stage-1 ckpt; default is ep30 per layer/ablation")
    p.add_argument("--residual_ablation", default="recon0",
                   choices=["main", "recon0", "full"],
                   help="Must match the stage-1 ckpt.  main does not train F_φ.")
    p.add_argument("--spatial_lr_scale", type=float, default=SPATIAL_LR_SCALE)
    p.add_argument("--lmbda", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--latent_channels", type=int, default=0,
                   help="Latent channel width C (0 = same as D)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_train_images", type=int, default=5000)
    p.add_argument("--n_val", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", default="dinov2_vitl14")
    p.add_argument("--feat_root",
                   default=str(HERE.parents[1] / "features"))
    p.add_argument("--gt_path",
                   default=str(HERE.parents[1] / "utils"
                               / "imagenet_selected_label500.txt"))
    p.add_argument("--ckpt_dir", default=str(HERE / "checkpoints"))
    p.add_argument("--result_dir",
                   default=str(HERE / "results" / "bilinear_orfc_jointopq"))
    p.add_argument("--residual_dir",
                   default=str(HERE / "results" / "bilinear_residual"))
    p.add_argument("--log_dir",
                   default=str(HERE / "logs" / "bilinear_orfc_jointopq"))
    p.add_argument("--skip_existing", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--skip_baselines", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def run_wave(layer, wave, args):
    failures = []
    with ThreadPoolExecutor(max_workers=max(len(wave), 1)) as pool:
        futs = {
            pool.submit(run_one, gpu, layer, K, args): (gpu, K)
            for gpu, K in wave
        }
        for fut in as_completed(futs):
            tag = fut.result()
            if tag:
                failures.append(tag)
    return failures


def main():
    args = parse_args()
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    layers = [x.strip() for x in args.layers.split(",") if x.strip()]
    n_miss = print_plan(gpus, layers, args)
    if n_miss:
        raise SystemExit(
            f"{n_miss} stage-1 {args.residual_ablation} ckpt(s) missing")
    if args.dry_run:
        return

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    waves = k_waves(gpus, K_ORDER)
    failures = []
    for r, layer in enumerate(layers, 1):
        print(f"\n=== layer {r}/{len(layers)} {layer} ===", flush=True)
        for w, wave in enumerate(waves, 1):
            print(f"=== {layer} wave {w}/{len(waves)} ===", flush=True)
            fails = run_wave(layer, wave, args)
            failures.extend(fails)
            if fails and args.strict:
                raise SystemExit(f"strict stop after {fails}")
        print(f"=== layer {r} {layer} finished ===", flush=True)

    if failures:
        raise SystemExit(f"{len(failures)} job(s) failed: {failures}")
    print("all jobs finished", flush=True)


if __name__ == "__main__":
    main()
