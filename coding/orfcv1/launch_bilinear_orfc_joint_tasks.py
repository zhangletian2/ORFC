#!/usr/bin/env python
"""VOC/NYU + rANS timing for recon0+ORFC joint vs original ORFC.

One K per GPU per layer.  Sequential layers.

    python -u launch_bilinear_orfc_joint_tasks.py --dry_run
    python -u launch_bilinear_orfc_joint_tasks.py --gpus 1,2,4,5,6,7
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ORFC = HERE.parent / "orfc"

from run_bilinear_orfc_joint import (  # noqa: E402
    joint_stem, orfc_ckpt_path, spatial_out_path, spatial_tag,
)

LAYERS = ["blk05", "blk10", "blk15", "blk20"]
K_ORDER = (4, 8, 16, 64, 256, 512)


def job_ns(layer, K, args):
    return SimpleNamespace(
        layer=layer, K=K, embedding_dim=32, bottleneck_dim=1024,
        lmbda=args.lmbda, lr=args.lr, spatial_lr_scale=args.spatial_lr_scale,
        epochs=args.epochs, max_train_images=5000, n_val=200, seed=42,
        backbone=args.backbone, ckpt_dir=args.ckpt_dir,
        tau_start=2.0, tau_end=2.0, tau_schedule="constant",
        residual_ablation=getattr(args, "residual_ablation", "recon0"),
    )


def log_stem(layer, K, args):
    return (f"{layer}_{spatial_tag(args)}_jointopq_K{K}_lmbda{args.lmbda}"
            f"_tau2.0_te2.0_tscon_hlr{args.spatial_lr_scale:g}"
            f"_bv_lr{args.lr}_ep{args.epochs}_tasks")


def ckpt_pair(layer, K, args):
    ns = job_ns(layer, K, args)
    return orfc_ckpt_path(ns), spatial_out_path(ns)


def out_path(layer, K, args):
    orfc, _ = ckpt_pair(layer, K, args)
    return Path(args.result_dir) / f"{orfc.stem}_tasks.json"


def build_cmd(layer, K, args):
    return [
        args.python, "-u", str(HERE / "eval_bilinear_orfc_joint_tasks.py"),
        "--layer", layer,
        "--K", str(K),
        "--backbone", args.backbone,
        "--tasks", args.tasks,
        "--ckpt_dir", args.ckpt_dir,
        "--orfc_ckpt_dir", args.orfc_ckpt_dir,
        "--result_dir", args.result_dir,
        "--linear4_task_dir", args.linear4_task_dir,
        "--residual_ablation", args.residual_ablation,
    ]


def run_one(gpu, layer, K, args):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TORCH_HOME"] = args.torch_home
    env["PYTHONUNBUFFERED"] = "1"
    tag = log_stem(layer, K, args)
    orfc, spat = ckpt_pair(layer, K, args)
    out = out_path(layer, K, args)
    if args.skip_existing and out.is_file():
        print(f"[GPU {gpu}] skip existing {tag} -> {out.name}", flush=True)
        return None
    if not orfc.is_file() or not spat.is_file():
        print(f"[GPU {gpu}] MISSING {orfc.name}", flush=True)
        return tag
    cmd = build_cmd(layer, K, args)
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


def round_assignments(gpus):
    if len(gpus) < len(K_ORDER):
        raise SystemExit(f"need {len(K_ORDER)} GPUs, got {gpus}")
    return list(zip(gpus[:len(K_ORDER)], K_ORDER))


def print_plan(gpus, layers, args):
    wave = round_assignments(gpus)
    print("=" * 72)
    print(f"conv2 {args.residual_ablation}+ORFC joint  VOC mIoU / NYU RMSE / rANS / timing")
    print(f"  vs original ORFC  tasks={args.tasks}")
    print(f"  GPUs: {gpus}  K={list(K_ORDER)}")
    print(f"  layers: {' → '.join(layers)}")
    print("=" * 72)
    n_eval = n_skip = n_miss = 0
    for r, layer in enumerate(layers, 1):
        print(f"\nRound {r}/{len(layers)}  {layer}")
        for gpu, K in wave:
            orfc, spat = ckpt_pair(layer, K, args)
            out = out_path(layer, K, args)
            if not orfc.is_file() or not spat.is_file():
                st = "MISSING"
                n_miss += 1
            elif args.skip_existing and out.is_file():
                st = "skip"
                n_skip += 1
            else:
                st = "eval"
                n_eval += 1
            print(f"  GPU {gpu}  K={K:<3d}  {st:7s}  {orfc.name}")
    print(f"\nJobs: eval={n_eval}  skip={n_skip}  missing={n_miss}")
    print("=" * 72)
    return n_miss


def run_wave(layer, wave, args):
    failures = []
    with ThreadPoolExecutor(max_workers=len(wave)) as pool:
        futs = {
            pool.submit(run_one, gpu, layer, K, args): (gpu, K)
            for gpu, K in wave
        }
        for fut in as_completed(futs):
            tag = fut.result()
            if tag:
                failures.append(tag)
    return failures


def _cell(obj, *keys, fmt=None):
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return ""
        cur = cur[k]
    if cur is None:
        return ""
    if fmt:
        return fmt.format(cur)
    return str(cur)


def write_summary(layers, args):
    rows = []
    for layer in layers:
        for K in K_ORDER:
            path = out_path(layer, K, args)
            if not path.is_file():
                continue
            d = json.loads(path.read_text())
            b = d.get("bilinear") or {}
            o = d.get("orfc") or {}
            l4 = d.get("linear4") or {}
            rows.append({
                "layer": layer, "K": K,
                "bili_miou": _cell(b, "seg", "miou"),
                "bili_rmse": _cell(b, "depth", "rmse"),
                "bili_bits_seg": _cell(b, "timing", "seg", "bits_per_image"),
                "bili_enc_seg": _cell(b, "timing", "seg", "enc_ms_mean"),
                "bili_dec_seg": _cell(b, "timing", "seg", "dec_ms_mean"),
                "bili_bits_dep": _cell(b, "timing", "depth", "bits_per_image"),
                "bili_enc_dep": _cell(b, "timing", "depth", "enc_ms_mean"),
                "bili_dec_dep": _cell(b, "timing", "depth", "dec_ms_mean"),
                "orfc_miou": _cell(o, "seg", "miou"),
                "orfc_rmse": _cell(o, "depth", "rmse"),
                "orfc_bits_seg": _cell(o, "timing", "seg", "bits_per_image"),
                "orfc_enc_seg": _cell(o, "timing", "seg", "enc_ms_mean"),
                "orfc_dec_seg": _cell(o, "timing", "seg", "dec_ms_mean"),
                "orfc_bits_dep": _cell(o, "timing", "depth", "bits_per_image"),
                "orfc_enc_dep": _cell(o, "timing", "depth", "enc_ms_mean"),
                "orfc_dec_dep": _cell(o, "timing", "depth", "dec_ms_mean"),
                "l4_miou": _cell(l4, "seg", "miou"),
                "l4_rmse": _cell(l4, "depth", "rmse"),
                "path": str(path),
            })
    out = Path(args.result_dir) / "summary.json"
    with open(out, "w") as f:
        json.dump({"rows": rows, "unit": {
            "bits": "rANS bits/image + 32 μ/σ; no grouping/guidance",
            "time": "ms/image, rANS included, backbone/head excluded",
        }}, f, indent=2)
    print(f"\n{'layer':<6} {'K':>4}  "
          f"{'mIoU bili':>10} {'ORFC':>10} {'L4':>10}  "
          f"{'RMSE bili':>10} {'ORFC':>10} {'L4':>10}  "
          f"{'bits bili':>10} {'ORFC':>10}  "
          f"{'enc bili':>8} {'ORFC':>8}  "
          f"{'dec bili':>8} {'ORFC':>8}")
    for r in rows:
        def f4(x):
            try:
                return f"{float(x):10.4f}"
            except (TypeError, ValueError):
                return f"{'':>10}"

        def f1(x):
            try:
                return f"{float(x):10.1f}"
            except (TypeError, ValueError):
                return f"{'':>10}"

        def fms(x):
            try:
                return f"{float(x):8.2f}"
            except (TypeError, ValueError):
                return f"{'':>8}"

        print(f"{r['layer']:<6} {r['K']:>4}  "
              f"{f4(r['bili_miou'])} {f4(r['orfc_miou'])} {f4(r['l4_miou'])}  "
              f"{f4(r['bili_rmse'])} {f4(r['orfc_rmse'])} {f4(r['l4_rmse'])}  "
              f"{f1(r['bili_bits_seg'])} {f1(r['orfc_bits_seg'])}  "
              f"{fms(r['bili_enc_seg'])} {fms(r['orfc_enc_seg'])}  "
              f"{fms(r['bili_dec_seg'])} {fms(r['orfc_dec_seg'])}")
    print(f"\nSaved {out}", flush=True)
    return out


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--python",
                   default="/home/user/anaconda3/envs/featcodec2/bin/python")
    p.add_argument("--torch_home",
                   default="/data4/workspace/zlt/featcodec/ORFC/pretrained")
    p.add_argument("--gpus", default="1,2,4,5,6,7")
    p.add_argument("--layers", default=",".join(LAYERS))
    p.add_argument("--spatial_lr_scale", type=float, default=0.1)
    p.add_argument("--lmbda", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--backbone", default="dinov2_vitl14")
    p.add_argument("--ckpt_dir", default=str(HERE / "checkpoints"))
    p.add_argument("--orfc_ckpt_dir", default=str(ORFC / "checkpoints"))
    p.add_argument("--result_dir",
                   default=str(HERE / "results" / "bilinear_orfc_jointopq" / "tasks"))
    p.add_argument("--linear4_task_dir",
                   default=str(HERE / "results" / "linear4_orfc_jointopq" / "tasks"))
    p.add_argument("--log_dir",
                   default=str(HERE / "logs" / "bilinear_orfc_jointopq"))
    p.add_argument("--tasks", default="seg,depth")
    p.add_argument("--residual_ablation", default="recon0",
                   choices=["main", "recon0", "full"])
    p.add_argument("--skip_existing", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    layers = [x.strip() for x in args.layers.split(",") if x.strip()]
    n_miss = print_plan(gpus, layers, args)
    if args.dry_run:
        return
    if n_miss:
        raise SystemExit(f"{n_miss} checkpoint(s) missing")

    wave = round_assignments(gpus)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    all_failures = []
    for r, layer in enumerate(layers, 1):
        print(f"\n{'#' * 72}")
        print(f"# Round {r}/{len(layers)}: {layer}")
        print(f"{'#' * 72}", flush=True)
        fails = run_wave(layer, wave, args)
        if fails:
            all_failures.extend(fails)
            print(f"Round {layer} failed on: {fails}", flush=True)
            if args.strict:
                raise SystemExit(f"aborted by --strict after {layer} failures")
        else:
            print(f"Round {layer} finished cleanly", flush=True)

    write_summary(layers, args)
    print("\n" + "=" * 72)
    if all_failures:
        print(f"FAILED ({len(all_failures)}): {all_failures}")
        raise SystemExit(1)
    print("ALL JOINT TASK EVALS COMPLETED")
    print("=" * 72)


if __name__ == "__main__":
    main()
