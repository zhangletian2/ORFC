#!/usr/bin/env python
"""Eval-only retest for SoftPQ best-config checkpoints.

This driver parses "best config.csv", resolves the matching saved codec
checkpoints, and invokes run_soft_pq.py with --eval_only on ImageNet test2 and
VOC2012-500 segmentation features.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]


@dataclass(frozen=True)
class BestConfig:
    layer: str
    config_text: str
    K: int
    embedding_dim: int
    lmbda: float
    lr: float
    epochs: int

    @property
    def log_stem(self) -> str:
        return (
            f"{self.layer}_K{self.K}_emb{self.embedding_dim}"
            f"_lmbda{self.lmbda}_lr{self.lr}_ep{self.epochs}"
        )


def parse_best_config(text, *, default_lmbda, default_lr, default_epochs):
    k_match = re.search(r"K\s*=\s*(\d+)", text)
    emb_match = re.search(r"e\s*(\d+)", text)
    if not k_match or not emb_match:
        raise ValueError(f"Cannot parse config: {text!r}")

    lmbda = default_lmbda
    lr = default_lr
    epochs = default_epochs

    match = re.search(r"λ\s*=\s*([0-9.]+)", text)
    if match:
        lmbda = float(match.group(1))

    match = re.search(r"lr\s*=\s*([0-9.eE+-]+)", text)
    if match:
        lr = float(match.group(1))

    match = re.search(r"ep\s*=\s*(\d+)", text)
    if match:
        epochs = int(match.group(1))

    return int(k_match.group(1)), int(emb_match.group(1)), lmbda, lr, epochs


def load_best_configs(csv_path, *, default_lmbda, default_lr, default_epochs):
    rows = list(csv.reader(open(csv_path, encoding="utf-8-sig")))
    # best config.csv stores layer groups as: 5, 10, 15, 20.
    layer_columns = [("blk05", 0), ("blk10", 4), ("blk15", 8), ("blk20", 12)]
    configs = []

    for row in rows[3:]:
        for layer, col in layer_columns:
            if col >= len(row) or not row[col].strip():
                continue
            text = row[col].strip().strip('"')
            K, emb, lmbda, lr, epochs = parse_best_config(
                text,
                default_lmbda=default_lmbda,
                default_lr=default_lr,
                default_epochs=default_epochs,
            )
            configs.append(BestConfig(layer, text, K, emb, lmbda, lr, epochs))

    return configs


def checkpoint_path(cfg, args):
    ckpt_dir = Path(args.ckpt_dir)
    bt_tag = f"bt{args.bottleneck_dim}" if args.bottleneck_dim > 0 else "noBt"
    ws_tag = "ws" if args.warm_start_opq else "km"
    rate_tag = f"_lmbda{cfg.lmbda}" if cfg.lmbda > 0 else ""
    tau_tag = f"_tau{args.tau_start}" if args.tau_start > 0 else ""
    tau_end_tag = (
        f"_te{args.tau_end}"
        if args.tau_start > 0 and args.tau_end != 0.005
        else ""
    )
    name = (
        f"{cfg.layer}_K{cfg.K}_emb{cfg.embedding_dim}_{bt_tag}_{ws_tag}"
        f"{rate_tag}{tau_tag}{tau_end_tag}"
        f"_lr{cfg.lr}_ep{cfg.epochs}_n{args.max_train_images}_s{args.seed}.pt"
    )
    return ckpt_dir / name


def build_command(cfg, ckpt, args):
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_soft_pq.py"),
        "--eval_only",
        "--ckpt_path",
        str(ckpt),
        "--layer",
        cfg.layer,
        "--K",
        str(cfg.K),
        "--embedding_dim",
        str(cfg.embedding_dim),
        "--bottleneck_dim",
        str(args.bottleneck_dim),
        "--lr",
        str(cfg.lr),
        "--epochs",
        str(cfg.epochs),
        "--lmbda",
        str(cfg.lmbda),
        "--tau_start",
        str(args.tau_start),
        "--tau_end",
        str(args.tau_end),
        "--max_train_images",
        str(args.max_train_images),
        "--batch_size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--feat_root",
        str(args.feat_root),
        "--train_subset",
        args.train_subset,
        "--test_subset",
        args.test_subset,
        "--backbone",
        args.backbone,
        "--gt_path",
        str(args.gt_path),
        "--result_suffix",
        args.result_suffix,
    ]
    if args.warm_start_opq:
        cmd.append("--warm_start_opq")
    if args.eval_seg:
        cmd.extend(
            [
                "--eval_seg",
                "--seg_feat_root",
                str(args.seg_feat_root),
                "--voc_root",
                str(args.voc_root),
                "--seg_image_list",
                str(args.seg_image_list),
            ]
        )
    return cmd


def print_inventory(configs, present, missing):
    print("=" * 72)
    print("Best-config eval-only inventory")
    print(f"  expected: {len(configs)}")
    print(f"  present : {len(present)}")
    print(f"  missing : {len(missing)}")
    if missing:
        print("\nMissing checkpoints:")
        for cfg, ckpt in missing:
            print(f"  {cfg.layer:5s} {cfg.config_text:28s} -> {ckpt.name}")
    print("=" * 72)


def run_jobs(jobs, args):
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        gpus = [None]

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    active = []
    failures = []
    next_job = 0

    while next_job < len(jobs) or active:
        busy_gpus = {item[4] for item in active}
        available_gpus = [gpu for gpu in gpus if gpu not in busy_gpus]
        while next_job < len(jobs) and available_gpus:
            gpu = available_gpus.pop(0)
            cfg, cmd = jobs[next_job]
            next_job += 1

            log_path = log_dir / f"{cfg.log_stem}.log"
            env = os.environ.copy()
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu

            print(f"[RUN] gpu={gpu or '-'} {cfg.log_stem}")
            log_fh = open(log_path, "w")
            proc = subprocess.Popen(
                cmd,
                cwd=str(SCRIPT_DIR),
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
            active.append((proc, log_fh, log_path, cfg, gpu))

        time.sleep(args.poll_interval)

        still_active = []
        for proc, log_fh, log_path, cfg, gpu in active:
            ret = proc.poll()
            if ret is None:
                still_active.append((proc, log_fh, log_path, cfg, gpu))
                continue
            log_fh.close()
            if ret == 0:
                print(f"[OK]  gpu={gpu or '-'} {cfg.log_stem}")
            else:
                print(f"[ERR] gpu={gpu or '-'} {cfg.log_stem} exit={ret} log={log_path}")
                failures.append((cfg, ret, log_path))
        active = still_active

    return failures


def main():
    parser = argparse.ArgumentParser(
        description="Retest SoftPQ best-config checkpoints in eval-only mode"
    )
    parser.add_argument("--best_csv", type=Path,
                        default=SCRIPT_DIR / "best config.csv")
    parser.add_argument("--ckpt_dir", type=Path,
                        default=SCRIPT_DIR / "checkpoints" / "dinov2_vitl14")
    parser.add_argument("--backbone", default="dinov2_vitl14")
    parser.add_argument("--feat_root", type=Path,
                        default=PROJECT_ROOT / "features")
    parser.add_argument("--train_subset", default="train")
    parser.add_argument("--test_subset", default="test2")
    parser.add_argument("--gt_path", type=Path,
                        default=PROJECT_ROOT / "utils" / "imagenet_selected_label2000.txt")
    parser.add_argument("--eval_seg", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--seg_feat_root", type=Path,
                        default=PROJECT_ROOT / "features" / "voc2012_500")
    parser.add_argument("--voc_root", type=Path,
                        default=PROJECT_ROOT / "data" / "VOCdevkit" / "VOC2012")
    parser.add_argument("--seg_image_list", type=Path,
                        default=PROJECT_ROOT / "utils" / "voc2012_val_500.txt")
    parser.add_argument("--result_suffix", default="retest_test2_voc500")
    parser.add_argument("--log_dir", type=Path,
                        default=SCRIPT_DIR / "logs" / "best_config_eval_only")
    parser.add_argument("--gpus", default=os.environ.get("GPUS", "0"),
                        help="Comma-separated GPU ids; jobs run in parallel across these ids")
    parser.add_argument("--only_layers", default="",
                        help="Comma-separated layer filter, e.g. blk05,blk20")
    parser.add_argument("--limit", type=int, default=0,
                        help="Run only the first N present configs")
    parser.add_argument("--strict", action="store_true",
                        help="Fail instead of skipping missing checkpoints")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--poll_interval", type=float, default=5.0)

    parser.add_argument("--bottleneck_dim", type=int, default=1024)
    parser.add_argument("--warm_start_opq", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--default_lmbda", type=float, default=0.5)
    parser.add_argument("--default_lr", type=float, default=3e-4)
    parser.add_argument("--default_epochs", type=int, default=100)
    parser.add_argument("--tau_start", type=float, default=0.5)
    parser.add_argument("--tau_end", type=float, default=0.005)
    parser.add_argument("--max_train_images", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configs = load_best_configs(
        args.best_csv,
        default_lmbda=args.default_lmbda,
        default_lr=args.default_lr,
        default_epochs=args.default_epochs,
    )

    if args.only_layers:
        keep = {x.strip() for x in args.only_layers.split(",") if x.strip()}
        configs = [cfg for cfg in configs if cfg.layer in keep]

    present = []
    missing = []
    for cfg in configs:
        ckpt = checkpoint_path(cfg, args)
        if ckpt.exists():
            present.append((cfg, ckpt))
        else:
            missing.append((cfg, ckpt))

    print_inventory(configs, present, missing)
    if missing and args.strict:
        return 2

    if args.limit > 0:
        present = present[:args.limit]

    jobs = [(cfg, build_command(cfg, ckpt, args)) for cfg, ckpt in present]
    if args.dry_run:
        for cfg, cmd in jobs:
            print(f"\n# {cfg.config_text}")
            print(" ".join(cmd))
        return 0

    failures = run_jobs(jobs, args)
    if failures:
        print("\nFailed jobs:")
        for cfg, ret, log_path in failures:
            print(f"  {cfg.log_stem}: exit={ret}, log={log_path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
