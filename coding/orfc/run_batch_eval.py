#!/usr/bin/env python
"""
Batch evaluation: regenerate JSON results from existing checkpoints.

Scans checkpoint directories, finds checkpoints without corresponding JSONs,
parses CLI args from filenames, and runs eval-only mode in parallel across GPUs.

Usage:
    python run_batch_eval.py --backbone dinov2_vitl14 --gpus 0 1 2 3 4 5
    python run_batch_eval.py --backbone dinov2_vitl14 --force   # overwrite existing JSONs
    python run_batch_eval.py --dry_run                          # print plan only
"""

import argparse, os, re, subprocess, sys, time, heapq, json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

PYTHON = sys.executable
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_soft_pq.py")
V34_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(V34_ROOT, "logs", "batch_eval")

# Layer → estimated eval time (minutes). Longer tail = slower.
EST_MINUTES = {
    "dinov2_vitl14": {"blk05": 12, "blk10": 10, "blk15": 8, "blk20": 6},
    "dinov2_vitg14": {"blk09": 16, "blk19": 12, "blk29": 10},
    "clip_vitl14":   {"blk05": 11, "blk10": 9,  "blk15": 7, "blk20": 5},
}

# Bottleneck dim per backbone
BT_DIM = {"dinov2_vitl14": 1024, "dinov2_vitg14": 1536, "clip_vitl14": 1024}

# ================================================================
#  Parse checkpoint filename → CLI args dict
# ================================================================

_CKPT_RE = re.compile(
    r'^(?P<layer>blk\d+)'
    r'_K(?P<K>\d+)'
    r'_emb(?P<emb>\d+)'
    r'_bt(?P<bt>\d+)'
    r'_(?P<ws>ws|km)'
    r'(?P<mse>_mse)?'
    r'(?:_lmbda(?P<lmbda>[\d.]+))?'
    r'(?P<fzR>_fzR)?'
    r'(?P<fzC>_fzC)?'
    r'(?:_rot(?P<rot>[a-z]+))?'
    r'(?:_pf(?P<pf>[\d.]+))?'
    r'(?:_tau(?P<tau>[\d.]+))?'
    r'(?:_te(?P<te>[\d.eE+-]+))?'
    r'(?:_ts(?P<ts>[a-z]+))?'
    r'_lr(?P<lr>[\d.eE+-]+)'
    r'_ep(?P<ep>\d+)'
    r'_n(?P<n>\d+)'
    r'_s(?P<seed>\d+)$'
)

# Map short rot tags back to CLI choices
_ROT_MAP = {"identity": "identity", "randomorth": "random_orth",
            "pca": "pca", "randorth": "random_orth"}


def parse_ckpt_name(stem, backbone):
    """Parse checkpoint stem into a dict of CLI args for run_soft_pq.py."""
    m = _CKPT_RE.match(stem)
    if not m:
        return None
    d = m.groupdict()

    rot_raw = d.get("rot")
    rot_cli = _ROT_MAP.get(rot_raw, "opq") if rot_raw else "opq"

    tau_sched_raw = d.get("ts")
    tau_sched_map = {"lin": "linear", "exp": "exponential", "con": "exponential"}
    tau_sched = tau_sched_map.get(tau_sched_raw, "exponential") if tau_sched_raw else "exponential"

    args = {
        "backbone": backbone,
        "layer": d["layer"],
        "K": int(d["K"]),
        "embedding_dim": int(d["emb"]),
        "bottleneck_dim": int(d["bt"]),
        "lr": float(d["lr"]),
        "epochs": int(d["ep"]),
        "max_train_images": int(d["n"]),
        "seed": int(d["seed"]),
        "warm_start_opq": d["ws"] == "ws",
        "mse_loss": d["mse"] is not None,
        "lmbda": float(d["lmbda"]) if d["lmbda"] else 0.0,
        "freeze_transform": d["fzR"] is not None,
        "freeze_codebooks": d["fzC"] is not None,
        "init_rotation": rot_cli,
        "prior_floor": float(d["pf"]) if d["pf"] else 0.0,
        "tau_start": float(d["tau"]) if d["tau"] else 0.0,
        "tau_end": float(d["te"]) if d["te"] else 0.005,
        "tau_schedule": tau_sched,
        "eval_seg": backbone != "clip_vitl14",
    }
    return args


def build_eval_cmd(cfg, ckpt_path):
    """Build CLI command for eval-only mode."""
    cmd = [PYTHON, SCRIPT, "--eval_only", "--ckpt_path", ckpt_path]
    for k, v in cfg.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.append(flag)
            cmd.append(str(v))
    return cmd


# ================================================================
#  Scheduling
# ================================================================

def schedule_jobs(jobs, gpu_ids):
    """Greedy longest-first scheduling. Returns {gpu_id: [(tag, cmd, est), ...]}."""
    queues = {g: [] for g in gpu_ids}
    heap = [(0, g) for g in gpu_ids]
    heapq.heapify(heap)
    for tag, cmd, est in sorted(jobs, key=lambda j: -j[2]):
        total, gpu = heapq.heappop(heap)
        queues[gpu].append((tag, cmd, est))
        heapq.heappush(heap, (total + est, gpu))
    return queues


def run_queue(gpu_id, queue, dry_run=False):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    for tag, cmd, est in queue:
        log_path = os.path.join(LOG_DIR, f"eval_{tag}.log")
        print(f"  [GPU {gpu_id}] START  {tag}  (~{est} min)")
        if dry_run:
            continue
        t0 = time.time()
        with open(log_path, "w") as lf:
            lf.write(f"# {' '.join(cmd)}\n# started: {datetime.now()}\n\n")
            lf.flush()
            proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            elapsed = (time.time() - t0) / 60
            status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
            lf.write(f"\n# {status}  elapsed={elapsed:.1f}min\n")
        print(f"  [GPU {gpu_id}] {status} {tag}  ({elapsed:.1f} min)")


# ================================================================
#  Main
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, nargs="+",
                        default=["dinov2_vitl14"],
                        help="Backbone(s) to eval")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate even if JSON already exists")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    jobs = []
    skipped = 0

    for backbone in args.backbone:
        ckpt_dir = os.path.join(V34_ROOT, "checkpoints", backbone)
        json_dir = os.path.join(V34_ROOT, "results", "soft_pq", backbone)
        os.makedirs(json_dir, exist_ok=True)

        if not os.path.isdir(ckpt_dir):
            print(f"  [WARN] No checkpoint dir: {ckpt_dir}")
            continue

        ckpt_files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
        print(f"\n  {backbone}: {len(ckpt_files)} checkpoints in {ckpt_dir}")

        est_map = EST_MINUTES.get(backbone, {})

        for ckpt_file in ckpt_files:
            stem = ckpt_file[:-3]
            json_path = os.path.join(json_dir, f"{stem}.json")

            if os.path.isfile(json_path) and not args.force:
                skipped += 1
                continue

            cfg = parse_ckpt_name(stem, backbone)
            if cfg is None:
                print(f"    [SKIP] Cannot parse: {ckpt_file}")
                continue

            ckpt_path = os.path.join(ckpt_dir, ckpt_file)
            cmd = build_eval_cmd(cfg, ckpt_path)
            est = est_map.get(cfg["layer"], 8)
            tag = f"{backbone}_{stem}"
            jobs.append((tag, cmd, est))

    print(f"\n{'=' * 70}")
    print(f"  Batch Eval Scheduler")
    print(f"  Jobs to run: {len(jobs)}   Skipped (JSON exists): {skipped}")
    print(f"  GPUs: {args.gpus}")
    print(f"{'=' * 70}")

    if not jobs:
        print("  Nothing to do.")
        return

    queues = schedule_jobs(jobs, args.gpus)

    max_est = 0
    total_est = 0
    for gpu in sorted(queues):
        q = queues[gpu]
        gpu_total = sum(e for _, _, e in q)
        total_est += gpu_total
        max_est = max(max_est, gpu_total)
        print(f"\n  GPU {gpu}  ({len(q)} jobs, ~{gpu_total} min):")
        for tag, _, est in q:
            short = tag.split("_", 1)[1] if "_" in tag else tag
            print(f"    {est:3d} min  {short}")

    print(f"\n  Total GPU-minutes: {total_est}")
    print(f"  Estimated wall-clock: ~{max_est} min ({max_est/60:.1f} h)")

    if args.dry_run:
        print("\n  [DRY RUN] — no jobs launched.")
        return

    print(f"\n  Launching at {datetime.now()} ...")
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = {}
        for gpu in sorted(queues):
            if queues[gpu]:
                f = pool.submit(run_queue, gpu, queues[gpu])
                futures[f] = gpu
        for f in as_completed(futures):
            gpu = futures[f]
            try:
                f.result()
                print(f"  GPU {gpu} — all eval jobs done.")
            except Exception as e:
                print(f"  GPU {gpu} — ERROR: {e}")

    print(f"\n  All done at {datetime.now()}")


if __name__ == "__main__":
    main()
