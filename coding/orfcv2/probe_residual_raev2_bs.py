#!/usr/bin/env python3
"""Probe max batch_size (no gradient checkpoint) on blk05/10/15/20.

Tries B in 16,8,4,2,1 high-to-low per layer. First success is the max.

  python probe_residual_raev2_bs.py --gpus 2,4,5,6
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ORFCV2 = Path(__file__).resolve().parent
ORFC = ORFCV2.parent / "orfc"
if str(ORFC / "eval") not in sys.path:
    sys.path.insert(0, str(ORFC / "eval"))

from dinov3_eval_common import CKPT_DIR, COFAI_ROOT, LAYERS, list_orfc_ckpts  # noqa: E402

PY = "/data4/workspace/zlt/featcodec/CoFAI/.venv/bin/python"
TRAIN_PY = ORFCV2 / "run_residual_raev2.py"
LOG_DIR = ORFCV2 / "logs" / "dinov3_raev2_res_probe"
CANDIDATES = (16, 8, 4, 2, 1)


def _env(gpu: str) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["PROJECT_ROOT"] = str(COFAI_ROOT)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
              "ALL_PROXY", "all_proxy"):
        env.pop(k, None)
    return env


def pick_ckpt(ckpt_dir: Path, layer: str) -> Path:
    cks = [p for p in list_orfc_ckpts(ckpt_dir) if p.name.startswith(layer + "_")]
    if not cks:
        raise SystemExit(f"no ckpt for {layer} in {ckpt_dir}")
    for p in cks:
        if "_K16_" in p.name and "lmbda0.5" in p.name:
            return p
    return cks[0]


def run_one(gpu: str, ckpt: Path, layer: str, batch: int, max_images: int) -> tuple[int, Path]:
    log = LOG_DIR / f"probe_{layer}_B{batch}.log"
    cmd = [
        PY, "-u", str(TRAIN_PY),
        "--base_ckpt", str(ckpt),
        "--layer", layer,
        "--K", "2",
        "--epochs", "1",
        "--batch_size", str(batch),
        "--max_images", str(max_images),
        "--kmeans_max_samples", "80000",
        "--log_every", "1",
        "--device", "cuda:0",
        "--no_save",
    ]
    print(f"  [gpu{gpu}] {layer} B={batch}  {log.name}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ORFCV2),
        env=_env(gpu),
        stdout=open(log, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    rc = proc.wait()
    return rc, log


def probe_layer(gpu: str, ckpt: Path, layer: str, candidates, max_images: int) -> int | None:
    print(f"\n== {layer}  base={ckpt.name}  gpu={gpu} ==", flush=True)
    best = None
    for b in candidates:
        if b > max_images:
            continue
        rc, log = run_one(gpu, ckpt, layer, b, max_images)
        tail = ""
        try:
            lines = log.read_text(errors="replace").splitlines()
            tail = " | ".join(lines[-3:])[:240]
        except Exception:
            pass
        if rc == 0:
            print(f"  OK  {layer} B={b}  {tail}", flush=True)
            best = b
            break
        oom = "out of memory" in tail.lower() or "CUDA out of memory" in tail
        print(f"  FAIL rc={rc} {layer} B={b}  oom={oom}  {tail}", flush=True)
    return best


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default="2,4,5,6")
    p.add_argument("--ckpt_dir", default=str(CKPT_DIR))
    p.add_argument("--max_images", type=int, default=16)
    p.add_argument(
        "--candidates", default="16,8,4,2,1",
        help="high-to-low batch sizes to try",
    )
    return p.parse_args()


def main():
    args = parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    candidates = tuple(int(x) for x in args.candidates.split(",") if x.strip())
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)

    jobs = []
    for i, layer in enumerate(LAYERS):
        gpu = gpus[i % len(gpus)]
        ckpt = pick_ckpt(ckpt_dir, layer)
        jobs.append((gpu, ckpt, layer))

    # Run layers in parallel (one GPU each). Each layer tries B sequentially.
    results = {layer: None for _, _, layer in jobs}
    if len(gpus) >= len(jobs):
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as ex:
            futs = {
                ex.submit(
                    probe_layer, gpu, ckpt, layer, candidates, args.max_images
                ): layer
                for gpu, ckpt, layer in jobs
            }
            for fut in concurrent.futures.as_completed(futs):
                layer = futs[fut]
                results[layer] = fut.result()
    else:
        for gpu, ckpt, layer in jobs:
            results[layer] = probe_layer(
                gpu, ckpt, layer, candidates, args.max_images
            )

    print("\n======== max batch_size (no grad checkpoint) ========")
    for layer in LAYERS:
        print(f"  {layer}:  B={results[layer]}")
    print("=====================================================")
    missing = [k for k, v in results.items() if v is None]
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
