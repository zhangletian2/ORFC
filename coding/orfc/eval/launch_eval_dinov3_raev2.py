#!/usr/bin/env python3
"""Sweep DINOv3 ORFC ckpts on RAEv2 reconstruction (ImageNet-500).

  python launch_eval_dinov3_raev2.py --gpus 2,4,5,6,7
  python launch_eval_dinov3_raev2.py --summary-only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent
_ORFC = _EVAL_DIR.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from dinov3_eval_common import (  # noqa: E402
    CKPT_DIR,
    COFAI_ROOT,
    LAYERS,
    RAEV2_LOG_DIR,
    RAEV2_RESULTS_DIR,
    list_orfc_ckpts,
    parse_ckpt_name,
)

PY = "/data4/workspace/zlt/featcodec/CoFAI/.venv/bin/python"
EVAL_PY = _EVAL_DIR / "eval_dinov3_raev2.py"


def _env(gpu: str) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["PROJECT_ROOT"] = str(COFAI_ROOT)
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)
    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    env.pop("ALL_PROXY", None)
    env.pop("all_proxy", None)
    return env


def _run(cmd, gpu, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [gpu{gpu}] {log_path.name}", flush=True)
    return subprocess.Popen(
        cmd,
        cwd=str(_ORFC),
        env=_env(gpu),
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def result_ok(tag: str) -> bool:
    p = RAEV2_RESULTS_DIR / f"recon_{tag}.json"
    if not p.is_file():
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        return int(d.get("n_samples", 0)) >= 400
    except Exception:
        return False


def eval_jobs(include_bypass: bool, include_last: bool, ckpt_dir: Path, extra: list[str]):
    jobs = []
    if include_last:
        jobs.append(
            {
                "tag": "bypass_blk23",
                "cmd": [
                    PY, "-u", str(EVAL_PY),
                    "--bypass", "--layer", "blk23",
                    "--device", "cuda:0",
                    "--vis_dir", str(RAEV2_RESULTS_DIR / "vis"),
                    *extra,
                ],
                "log": RAEV2_LOG_DIR / "eval_bypass_blk23.log",
            }
        )
    if include_bypass:
        for lyr in LAYERS:
            jobs.append(
                {
                    "tag": f"bypass_{lyr}",
                    "cmd": [
                        PY, "-u", str(EVAL_PY),
                        "--bypass", "--layer", lyr,
                        "--device", "cuda:0", *extra,
                    ],
                    "log": RAEV2_LOG_DIR / f"eval_bypass_{lyr}.log",
                }
            )
    vis_once = True
    for ckpt in list_orfc_ckpts(ckpt_dir):
        tag = ckpt.stem
        cmd = [
            PY, "-u", str(EVAL_PY),
            "--ckpt_path", str(ckpt),
            "--device", "cuda:0", *extra,
        ]
        if vis_once:
            cmd += ["--vis_dir", str(RAEV2_RESULTS_DIR / "vis")]
            vis_once = False
        jobs.append(
            {
                "tag": tag,
                "cmd": cmd,
                "log": RAEV2_LOG_DIR / f"eval_{tag}.log",
            }
        )
    return jobs


def launch(gpus: list[str], jobs: list[dict]):
    pending = []
    for j in jobs:
        if result_ok(j["tag"]):
            print(f"  [skip] {j['tag']}")
        else:
            pending.append(j)
    print(f"jobs: {len(pending)} pending / {len(jobs)} total  gpus={gpus}")
    if not pending:
        return 0
    free = list(gpus)
    running = []
    fail = 0
    idx = 0
    while idx < len(pending) or running:
        while idx < len(pending) and free:
            gpu = free.pop(0)
            job = pending[idx]
            idx += 1
            proc = _run(job["cmd"], gpu, job["log"])
            running.append((proc, job["tag"], gpu))
        still = []
        for proc, tag, gpu in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, tag, gpu))
            else:
                if rc != 0:
                    print(f"  [FAIL] {tag} rc={rc}", flush=True)
                    fail += 1
                else:
                    print(f"  [done] {tag}", flush=True)
                free.append(gpu)
        running = still
        if running and (idx >= len(pending) or not free):
            time.sleep(8)
    return fail


def print_summary():
    rows = []
    for p in sorted(RAEV2_RESULTS_DIR.glob("recon_*.json")):
        with open(p) as f:
            d = json.load(f)
        rows.append(d)
    if not rows:
        print("  (no result json)")
        return
    print(
        f"{'tag':72s}  {'layer':6s}  {'mode':6s}  "
        f"{'PSNR':>7s}  {'MS-SSIM':>7s}  {'LPIPS':>7s}  {'bpp':>7s}"
    )
    print("-" * 130)
    for d in rows:
        m = d.get("metrics") or {}
        mb = d.get("metrics_bypass") or {}
        rate = d.get("rate") or {}
        tag = d.get("tag", "")
        bpp = rate.get("bpp")
        bpp_s = f"{bpp:.4f}" if bpp is not None else "-"
        print(
            f"{tag:72s}  {str(d.get('layer')):6s}  {str(d.get('mode')):6s}  "
            f"{(m.get('psnr') or 0):7.3f}  {(m.get('ms_ssim') or 0):7.4f}  "
            f"{(m.get('lpips') if m.get('lpips') is not None else float('nan')):7.4f}  {bpp_s:>7s}"
        )
        if d.get("mode") == "orfc" and mb.get("psnr") is not None:
            print(
                f"{'  ↳ bypass':72s}  {str(d.get('layer')):6s}  {'bp':6s}  "
                f"{mb['psnr']:7.3f}  {(mb.get('ms_ssim') or 0):7.4f}  "
                f"{(mb.get('lpips') if mb.get('lpips') is not None else float('nan')):7.4f}  {'-':>7s}"
            )

    csv_path = RAEV2_RESULTS_DIR / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("tag,layer,mode,K,lmbda,psnr,ms_ssim,lpips,bpp,bpfp,psnr_bypass\n")
        for d in rows:
            info = parse_ckpt_name(d.get("ckpt") or d.get("tag") or "")
            m = d.get("metrics") or {}
            mb = d.get("metrics_bypass") or {}
            rate = d.get("rate") or {}
            f.write(
                f"{d.get('tag')},{d.get('layer')},{d.get('mode')},"
                f"{info.get('K') or rate.get('K') or ''},"
                f"{info.get('lmbda') if info.get('lmbda') is not None else ''},"
                f"{m.get('psnr')},{m.get('ms_ssim')},{m.get('lpips')},"
                f"{rate.get('bpp')},{rate.get('bpfp')},{mb.get('psnr')}\n"
            )
    print(f"\ncsv → {csv_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default="2,4,5,6,7")
    p.add_argument("--ckpt_dir", default=str(CKPT_DIR))
    p.add_argument("--no-bypass", action="store_true")
    p.add_argument("--no-last", action="store_true", help="Skip blk23 last-layer RAEv2 k1 bypass")
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--force", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    RAEV2_LOG_DIR.mkdir(parents=True, exist_ok=True)
    RAEV2_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    extra = ["--batch_size", str(args.batch_size)]
    if args.max_images:
        extra += ["--max_images", str(args.max_images)]
    if args.force:
        extra.append("--force")

    if args.summary_only:
        print_summary()
        return

    jobs = eval_jobs(
        include_bypass=not args.no_bypass,
        include_last=not args.no_last,
        ckpt_dir=Path(args.ckpt_dir),
        extra=extra,
    )
    print(f"GPUS={gpus}  ckpts={len(list_orfc_ckpts(Path(args.ckpt_dir)))}  jobs={len(jobs)}")
    fail = launch(gpus, jobs)
    print("\n========== SUMMARY ==========")
    print_summary()
    print(f"fail={fail}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
