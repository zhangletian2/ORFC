#!/usr/bin/env python3
"""Launch DINOv3 ORFC task eval: extract (if needed) → probes → sweep ckpts.

Usage:
  python launch_eval_dinov3.py --gpus 2,4,5,6,7
  python launch_eval_dinov3.py --skip-extract --skip-probe --gpus 2,4,5,6,7
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
    ADE_FEAT_ROOT,
    CKPT_DIR,
    COFAI_ROOT,
    IMAGENET_TEST_FEAT,
    LAYERS,
    LOG_DIR,
    NYU_FEAT_ROOT,
    PROBE_DIR,
    RESULTS_DIR,
    list_orfc_ckpts,
    parse_ckpt_name,
)

PY = "/data4/workspace/zlt/featcodec/CoFAI/.venv/bin/python"
EXTRACT_PY = _EVAL_DIR / "extract_dinov3_eval_features.py"
PROBE_PY = _EVAL_DIR / "train_dinov3_cls_probe.py"
EVAL_PY = _EVAL_DIR / "eval_dinov3_tasks.py"


def _env(gpu: str) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["PROJECT_ROOT"] = str(COFAI_ROOT)
    return env


def _run(cmd, gpu, log_path: Path, cwd=_ORFC):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [gpu{gpu}] {' '.join(str(c) for c in cmd[-8:])}  → {log_path.name}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=_env(gpu),
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc


def count_npy(d: Path) -> int:
    if not d.is_dir():
        return 0
    return sum(1 for _ in d.glob("*.npy"))


def extract_ready() -> bool:
    ok = True
    for lyr in LAYERS:
        n = count_npy(IMAGENET_TEST_FEAT / lyr)
        print(f"  imagenet {lyr}: {n}/500")
        if n < 500:
            ok = False
        n = count_npy(ADE_FEAT_ROOT / lyr)
        print(f"  ade      {lyr}: {n}/2000")
        if n < 2000:
            ok = False
        n = count_npy(NYU_FEAT_ROOT / lyr)
        print(f"  nyu      {lyr}: {n}/654")
        if n < 650:
            ok = False
    return ok


def probes_ready() -> bool:
    ok = True
    for lyr in LAYERS:
        pt = PROBE_DIR / f"{lyr}_linear_cls.pt"
        print(f"  probe {lyr}: {'ok' if pt.is_file() else 'MISSING'}")
        if not pt.is_file():
            ok = False
    return ok


def wait_procs(procs: list[tuple], timeout=None):
    fail = 0
    t0 = time.time()
    while procs:
        still = []
        for proc, tag in procs:
            rc = proc.poll()
            if rc is None:
                still.append((proc, tag))
            elif rc != 0:
                print(f"  [FAIL] {tag} rc={rc}", flush=True)
                fail += 1
            else:
                print(f"  [done] {tag}  ({time.time() - t0:.0f}s)", flush=True)
        procs = still
        if procs:
            time.sleep(5)
    return fail


def launch_extract(gpus: list[str]):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("imagenet", gpus[0 % len(gpus)], ["--dataset", "imagenet", "--batch_size", "16"]),
        ("ade", gpus[1 % len(gpus)], ["--dataset", "ade", "--batch_size", "4"]),
        ("nyu", gpus[2 % len(gpus)], ["--dataset", "nyu", "--batch_size", "4"]),
    ]
    procs = []
    for name, gpu, extra in jobs:
        cmd = [PY, "-u", str(EXTRACT_PY), "--device", "cuda:0", *extra]
        proc = _run(cmd, gpu, LOG_DIR / f"extract_{name}.log", cwd=_EVAL_DIR)
        procs.append((proc, f"extract-{name}"))
    return wait_procs(procs)


def launch_probes(gpu: str, layers=None):
    layers = list(layers or LAYERS)
    todo = [lyr for lyr in layers if not (PROBE_DIR / f"{lyr}_linear_cls.pt").is_file()]
    if not todo:
        print("  probes already exist")
        return 0
    cmd = [
        PY, "-u", str(PROBE_PY),
        "--device", "cuda:0",
        "--layers", *todo,
    ]
    proc = _run(cmd, gpu, LOG_DIR / "train_cls_probe.log", cwd=_EVAL_DIR)
    return wait_procs([(proc, "train-probe")])


def eval_jobs(include_bypass: bool, tasks: str, ckpt_dir: Path):
    jobs = []
    if include_bypass:
        for lyr in LAYERS:
            jobs.append(
                {
                    "tag": f"bypass_{lyr}",
                    "cmd": [
                        PY, "-u", str(EVAL_PY),
                        "--bypass", "--layer", lyr,
                        "--tasks", tasks,
                        "--device", "cuda:0",
                    ],
                    "log": LOG_DIR / f"eval_bypass_{lyr}.log",
                }
            )
    for ckpt in list_orfc_ckpts(ckpt_dir):
        info = parse_ckpt_name(ckpt)
        tag = ckpt.name[:-3] if ckpt.suffix == ".pt" else ckpt.stem
        jobs.append(
            {
                "tag": tag,
                "cmd": [
                    PY, "-u", str(EVAL_PY),
                    "--ckpt_path", str(ckpt),
                    "--tasks", tasks,
                    "--device", "cuda:0",
                ],
                "log": LOG_DIR / f"eval_{tag}.log",
                "layer": info.get("layer"),
            }
        )
    return jobs


_MIN_SAMPLES = {"cls": 500, "semseg": 2000, "depth": 650}


def result_complete(task: str, tag: str) -> bool:
    path = RESULTS_DIR / f"{task}_{tag}.json"
    if not path.is_file():
        return False
    try:
        with open(path) as f:
            d = json.load(f)
        return int(d.get("n_samples", 0)) >= _MIN_SAMPLES.get(task, 1)
    except Exception:
        return False


def all_task_json_exist(tag: str, tasks: list[str]) -> bool:
    return all(result_complete(t, tag) for t in tasks)


def launch_eval(gpus: list[str], include_bypass: bool, tasks: str, ckpt_dir: Path):
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    jobs = eval_jobs(include_bypass, tasks, ckpt_dir)
    pending = []
    for j in jobs:
        if all_task_json_exist(j["tag"], task_list):
            print(f"  [skip] {j['tag']}")
        else:
            pending.append(j)
    print(f"eval jobs: {len(pending)} pending / {len(jobs)} total  gpus={gpus}")
    if not pending:
        return 0

    free = list(gpus)
    running: list[tuple] = []
    fail = 0
    idx = 0
    while idx < len(pending) or running:
        while idx < len(pending) and free:
            gpu = free.pop(0)
            job = pending[idx]
            idx += 1
            proc = _run(job["cmd"], gpu, job["log"], cwd=_ORFC)
            running.append((proc, job["tag"], gpu))
        still = []
        for proc, tag, gpu in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, tag, gpu))
            else:
                if rc != 0:
                    print(f"  [FAIL] {tag} rc={rc}  log={tag}", flush=True)
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
    for p in sorted(RESULTS_DIR.glob("*.json")):
        with open(p) as f:
            d = json.load(f)
        m = d.get("metrics") or {}
        rows.append((p.name, d.get("task"), d.get("layer"), d.get("mode"), m, d.get("avg_mse")))
    if not rows:
        print("  (no result json)")
        return
    print(f"\n{'file':62s}  {'metric':28s}  mse")
    print("-" * 110)
    for name, task, layer, mode, m, mse in rows:
        if "mIoU" in m:
            metric = f"mIoU={m['mIoU']:.4f}"
        elif "top-1" in m:
            metric = f"top1={m['top-1']:.2f} top5={m.get('top-5', 0):.2f}"
        elif "rmse" in m:
            metric = f"rmse={m['rmse']:.4f} abs_rel={m.get('abs_rel', 0):.4f} a1={m.get('a1', 0):.4f}"
        else:
            metric = str(m)
        mse_s = f"{mse:.6f}" if mse is not None else "-"
        print(f"{name:62s}  {metric:28s}  {mse_s}")

    # compact table by layer/K/lmbda
    by = {}
    for name, task, layer, mode, m, mse in rows:
        key = name.replace(".json", "")
        by.setdefault(key, {})[task] = m
    print("\n==== compact ====")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default="2,4,5,6,7")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-probe", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--no-bypass", action="store_true")
    p.add_argument("--tasks", default="cls,semseg,depth")
    p.add_argument("--ckpt_dir", default=str(CKPT_DIR))
    p.add_argument("--summary-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"GPUS={gpus}")
    print(f"ckpts={len(list_orfc_ckpts(Path(args.ckpt_dir)))}")

    if args.summary_only:
        print_summary()
        return

    fail = 0
    if not args.skip_extract:
        if extract_ready():
            print("features already complete, skip extract")
        else:
            print("=== extract ===")
            fail += launch_extract(gpus)
            if not extract_ready():
                print("[WARN] extract counts incomplete")
    else:
        extract_ready()

    if not args.skip_probe:
        print("=== probes ===")
        fail += launch_probes(gpus[0])
        probes_ready()
    else:
        probes_ready()

    if not args.skip_eval:
        print("=== eval sweep ===")
        fail += launch_eval(
            gpus,
            include_bypass=not args.no_bypass,
            tasks=args.tasks,
            ckpt_dir=Path(args.ckpt_dir),
        )

    print("\n========== SUMMARY ==========")
    print_summary()
    print(f"fail={fail}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
