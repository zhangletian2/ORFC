#!/usr/bin/env python
"""
Batch training scheduler for all supplementary experiments.

Groups:
  A. CLIP ViT-L/14 checkpoints       (Experiment 1 – Sensitivity)
  B. G / K / d sensitivity            (Experiment 3)
  C. Temperature schedule             (Experiment 4)
  D. Initialization                   (Experiment 5)
  E. Multi-seed                       (Experiment 6)

Greedy longest-first scheduling on 6 GPUs (configurable).

Usage:
    python run_batch_train.py                       # run all
    python run_batch_train.py --groups A B           # run subset
    python run_batch_train.py --dry_run              # print schedule only
    python run_batch_train.py --gpus 0 1 2 3         # use 4 GPUs
"""

import argparse, os, subprocess, sys, time, heapq, json
from datetime import datetime

PYTHON = sys.executable
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_soft_pq.py")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "batch_all")


# ================================================================
#  Job definitions
# ================================================================
# Each job: (group, tag, estimated_minutes, dict of CLI args)
# CLI args are merged with per-backbone defaults.

DEFAULTS_DINO_L = dict(
    backbone="dinov2_vitl14", bottleneck_dim=1024, batch_size=32,
    max_train_images=5000, lr=0.0003, epochs=100, seed=42,
    warm_start_opq=True, eval_seg=True,
    tau_start=0.5, tau_end=0.005, tau_schedule="exponential",
)
DEFAULTS_DINO_G = dict(
    backbone="dinov2_vitg14", bottleneck_dim=1536, batch_size=16,
    max_train_images=5000, lr=0.0003, epochs=100, seed=42,
    warm_start_opq=True, eval_seg=True,
    tau_start=0.5, tau_end=0.005, tau_schedule="exponential",
)
DEFAULTS_CLIP = dict(
    backbone="clip_vitl14", bottleneck_dim=1024, batch_size=32,
    max_train_images=5000, lr=0.0003, epochs=100, seed=42,
    warm_start_opq=True, eval_seg=False,
    tau_start=0.5, tau_end=0.005, tau_schedule="exponential",
)

# Estimated minutes by (backbone, layer)
# blk smaller → longer tail → slower training
TIME_DINO_L = {"blk05": 60, "blk10": 50, "blk15": 40, "blk20": 28}
TIME_DINO_G = {"blk09": 70, "blk19": 55, "blk29": 45}
TIME_CLIP   = {"blk05": 55, "blk10": 45, "blk15": 35, "blk20": 25}


def _t(backbone, layer):
    if backbone == "dinov2_vitl14":
        return TIME_DINO_L.get(layer, 30)
    if backbone == "dinov2_vitg14":
        return TIME_DINO_G.get(layer, 50)
    return TIME_CLIP.get(layer, 30)


def _defaults(backbone):
    if backbone == "dinov2_vitg14":
        return DEFAULTS_DINO_G
    if backbone == "clip_vitl14":
        return DEFAULTS_CLIP
    return DEFAULTS_DINO_L


JOBS = []


# ────────── A. CLIP ViT-L/14 (for Experiment 1 Sensitivity) ──────────
for layer in ["blk05", "blk10", "blk15", "blk20"]:
    JOBS.append(("A", f"clip_{layer}_K16", _t("clip_vitl14", layer),
                 dict(layer=layer, K=16, embedding_dim=32, lmbda=0.5,
                      **_defaults("clip_vitl14"))))


# ────────── B. G/K/d sensitivity (dinov2_vitl14 blk20) ──────────
_B = dict(layer="blk20", **_defaults("dinov2_vitl14"))
for K, d, tag in [
    (4,   16, "K4_d16"),
    (16,  16, "K16_d16"),
    (16,   8, "K16_d8"),
    (128, 32, "K128_d32"),
    (512, 16, "K512_d16"),
    (1024,16, "K1024_d16"),
]:
    JOBS.append(("B", f"gkd_{tag}", _t("dinov2_vitl14", "blk20"),
                 dict(K=K, embedding_dim=d, lmbda=0.5, **_B)))


# ────────── C. Temperature schedule (dinov2_vitl14 blk20 K=16 e=32) ──
_C_base = {k: v for k, v in _defaults("dinov2_vitl14").items()
           if k not in ("tau_start", "tau_end", "tau_schedule")}
for ts, te, sched, tag in [
    (0.1,  0.005, "exponential", "tau0.1_exp"),
    (1.0,  0.005, "exponential", "tau1.0_exp"),
    (2.0,  0.005, "exponential", "tau2.0_exp"),
    (0.5,  0.05,  "exponential", "tau0.5_te0.05"),
    (0.5,  0.0005,"exponential", "tau0.5_te0.0005"),
    (0.5,  0.005, "linear",      "tau0.5_linear"),
    (0.5,  0.5,   "exponential", "tau0.5_const"),   # constant τ
]:
    JOBS.append(("C", f"temp_{tag}", _t("dinov2_vitl14", "blk20"),
                 dict(layer="blk20", K=16, embedding_dim=32, lmbda=0.5,
                      tau_start=ts, tau_end=te, tau_schedule=sched,
                      **_C_base)))


# ────────── D. Initialization (dinov2_vitl14 blk20) ──────────
for K, d, kd_tag in [(16, 32, "K16"), (256, 16, "K256")]:
    for rot, rot_tag in [
        ("opq",         "km"),           # k-means only, no OPQ warm-start
        ("identity",    "identity"),
        ("random_orth", "randorth"),
        ("pca",         "pca"),
    ]:
        JOBS.append(("D", f"init_{kd_tag}_{rot_tag}",
                     _t("dinov2_vitl14", "blk20"),
                     dict(layer="blk20", K=K, embedding_dim=d, lmbda=0.5,
                          warm_start_opq=False, init_rotation=rot,
                          eval_seg=True,
                          **{k: v for k, v in _defaults("dinov2_vitl14").items()
                             if k not in ("warm_start_opq", "eval_seg")})))


# ────────── E. Multi-seed (seeds 43, 44) ──────────
# dinov2_vitl14 blk20 – 5 rate points × 2 seeds
for seed in [43, 44]:
    for K, d in [(4, 32), (8, 32), (16, 32), (64, 16), (256, 16)]:
        JOBS.append(("E", f"ms_L14_blk20_K{K}_e{d}_s{seed}",
                     _t("dinov2_vitl14", "blk20"),
                     dict(layer="blk20", K=K, embedding_dim=d, lmbda=0.5,
                          seed=seed, **{k: v for k, v in
                          _defaults("dinov2_vitl14").items()
                          if k != "seed"})))

# dinov2_vitl14 blk10 – 3 rate points × 2 seeds
for seed in [43, 44]:
    for K, d in [(4, 32), (16, 32), (256, 16)]:
        JOBS.append(("E", f"ms_L14_blk10_K{K}_e{d}_s{seed}",
                     _t("dinov2_vitl14", "blk10"),
                     dict(layer="blk10", K=K, embedding_dim=d, lmbda=0.5,
                          seed=seed, **{k: v for k, v in
                          _defaults("dinov2_vitl14").items()
                          if k != "seed"})))

# dinov2_vitg14 blk29 – 2 rate points × 2 seeds
for seed in [43, 44]:
    for K, d in [(16, 32), (64, 32)]:
        JOBS.append(("E", f"ms_G14_blk29_K{K}_e{d}_s{seed}",
                     _t("dinov2_vitg14", "blk29"),
                     dict(layer="blk29", K=K, embedding_dim=d, lmbda=0.5,
                          seed=seed, **{k: v for k, v in
                          _defaults("dinov2_vitg14").items()
                          if k != "seed"})))


# ================================================================
#  Predict the exact checkpoint / JSON filename from a config dict
#  (mirrors the naming logic in run_soft_pq.py)
# ================================================================

EXP_GROUP_NAMES = {
    "A": "clip_sensitivity",
    "B": "gkd_sensitivity",
    "C": "temperature",
    "D": "initialization",
    "E": "multi_seed",
}


def predict_filenames(cfg):
    """Return (ckpt_relpath, json_relpath) that run_soft_pq.py will produce."""
    backbone   = cfg["backbone"]
    bt_dim     = cfg.get("bottleneck_dim", 0)
    bt_tag     = f"bt{bt_dim}" if bt_dim > 0 else "noBt"
    ws_tag     = "ws" if cfg.get("warm_start_opq", False) else "km"
    mse_tag    = "_mse" if cfg.get("mse_loss", False) else ""
    lmbda      = cfg.get("lmbda", 0.0)
    rate_tag   = f"_lmbda{lmbda}" if lmbda > 0 else ""

    fz_tag = ""
    if cfg.get("freeze_transform", False):
        fz_tag += "_fzR"
    if cfg.get("freeze_codebooks", False):
        fz_tag += "_fzC"

    init_rot  = cfg.get("init_rotation", "opq")
    rot_short = init_rot.replace("_", "")
    rot_tag   = f"_rot{rot_short}" if init_rot != "opq" else ""

    pfloor    = cfg.get("prior_floor", 0.0)
    pfloor_tag = f"_pf{pfloor}" if pfloor > 0 else ""

    tau_s = cfg.get("tau_start", 0.0)
    tau_e = cfg.get("tau_end", 0.005)
    tau_sched = cfg.get("tau_schedule", "exponential")
    tau_tag      = f"_tau{tau_s}" if tau_s > 0 else ""
    tau_end_tag  = f"_te{tau_e}" if tau_e != 0.005 and tau_s > 0 else ""
    tau_sched_tag = f"_ts{tau_sched[:3]}" if tau_sched != "exponential" else ""

    stem = (f"{cfg['layer']}_K{cfg['K']}_emb{cfg['embedding_dim']}"
            f"_{bt_tag}_{ws_tag}{mse_tag}{rate_tag}"
            f"{fz_tag}{rot_tag}{pfloor_tag}{tau_tag}{tau_end_tag}{tau_sched_tag}"
            f"_lr{cfg['lr']}_ep{cfg['epochs']}_n{cfg['max_train_images']}_s{cfg['seed']}")

    ckpt = f"checkpoints/{backbone}/{stem}.pt"
    json_p = f"results/soft_pq/{backbone}/{stem}.json"
    return ckpt, json_p


# ================================================================
#  Build CLI command from job config
# ================================================================

def build_cmd(cfg):
    args = []
    for k, v in cfg.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                args.append(flag)
        else:
            args.append(flag)
            args.append(str(v))
    return [PYTHON, SCRIPT] + args


# ================================================================
#  Greedy scheduler – assign longest jobs first
# ================================================================

def schedule(jobs, gpu_ids):
    """Return list of (gpu_id, [(tag, cmd, est_min), ...]) per GPU."""
    gpu_queues = {g: [] for g in gpu_ids}
    # min-heap: (total_est_minutes, gpu_id)
    heap = [(0, g) for g in gpu_ids]
    heapq.heapify(heap)

    # Sort jobs longest first for best bin packing
    sorted_jobs = sorted(jobs, key=lambda j: -j[2])

    for group, tag, est_min, cfg in sorted_jobs:
        total_min, gpu = heapq.heappop(heap)
        gpu_queues[gpu].append((tag, build_cmd(cfg), est_min))
        heapq.heappush(heap, (total_min + est_min, gpu))

    return gpu_queues


# ================================================================
#  Runner
# ================================================================

def run_gpu_queue(gpu_id, queue, dry_run=False):
    """Sequentially run all jobs assigned to one GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    for tag, cmd, est_min in queue:
        log_path = os.path.join(LOG_DIR, f"{tag}.log")
        print(f"  [GPU {gpu_id}] START  {tag}  (~{est_min} min)")
        if dry_run:
            continue
        t0 = time.time()
        with open(log_path, "w") as lf:
            lf.write(f"# {' '.join(cmd)}\n")
            lf.write(f"# started: {datetime.now()}\n\n")
            lf.flush()
            proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            elapsed = (time.time() - t0) / 60
            status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
            lf.write(f"\n# {status}  elapsed={elapsed:.1f}min\n")
        print(f"  [GPU {gpu_id}] {status} {tag}  ({elapsed:.1f} min)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--groups", type=str, nargs="+", default=None,
                        help="Run only these groups (A/B/C/D/E)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print schedule without running")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    # Filter jobs by group
    jobs = JOBS
    if args.groups:
        groups = set(g.upper() for g in args.groups)
        jobs = [j for j in JOBS if j[0] in groups]

    print(f"\n{'=' * 70}")
    print(f"  Batch Training Scheduler")
    print(f"  Total jobs: {len(jobs)}   GPUs: {args.gpus}")
    print(f"  Groups: {sorted(set(j[0] for j in jobs))}")
    print(f"{'=' * 70}")

    # Schedule
    gpu_queues = schedule(jobs, args.gpus)

    # Print schedule
    total_est = 0
    max_gpu_est = 0
    for gpu in sorted(gpu_queues):
        queue = gpu_queues[gpu]
        gpu_total = sum(e for _, _, e in queue)
        total_est += gpu_total
        max_gpu_est = max(max_gpu_est, gpu_total)
        print(f"\n  GPU {gpu}  ({len(queue)} jobs, ~{gpu_total} min):")
        for tag, cmd, est in queue:
            print(f"    {est:3d} min  {tag}")

    print(f"\n  Total GPU-minutes: {total_est}")
    print(f"  Estimated wall-clock: ~{max_gpu_est} min ({max_gpu_est/60:.1f} h)")

    # ── Write experiment manifest ──
    base_dir = os.path.dirname(os.path.abspath(__file__))
    manifest = {}
    for group, tag, est_min, cfg in jobs:
        ckpt_rel, json_rel = predict_filenames(cfg)
        manifest[tag] = {
            "group": group,
            "experiment": EXP_GROUP_NAMES.get(group, group),
            "checkpoint": ckpt_rel,
            "json_result": json_rel,
            "est_minutes": est_min,
            "config": {k: v for k, v in cfg.items()
                       if k not in ("eval_seg",)},
        }
    manifest_path = os.path.join(base_dir, "experiment_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest saved: {manifest_path}")

    if args.dry_run:
        print("\n  [DRY RUN] — no jobs launched.")
        return

    print(f"\n  Launching at {datetime.now()} ...")
    print(f"  Logs → {LOG_DIR}/")

    # Launch one subprocess per GPU, each running its queue sequentially
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = {}
        for gpu in sorted(gpu_queues):
            if gpu_queues[gpu]:
                f = pool.submit(run_gpu_queue, gpu, gpu_queues[gpu])
                futures[f] = gpu
        for f in as_completed(futures):
            gpu = futures[f]
            try:
                f.result()
                print(f"  GPU {gpu} — all jobs done.")
            except Exception as e:
                print(f"  GPU {gpu} — ERROR: {e}")

    print(f"\n  All done at {datetime.now()}")


if __name__ == "__main__":
    main()
