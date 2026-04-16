#!/usr/bin/env python
"""
Exp 1 — Batch sensitivity ablation scheduler.

Generates and optionally runs compute_sensitivity.py for all
(backbone, layer) combinations, using available checkpoints.

Usage:
    python run_exp1_batch.py --dry_run            # preview commands
    python run_exp1_batch.py --gpus 0 1 2 3       # run on 4 GPUs
"""

import os, sys, json, argparse, glob, subprocess, time
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))

BACKBONE_LAYERS = {
    'dinov2_vitl14': ['blk05', 'blk10', 'blk15', 'blk20'],
    'dinov2_vitg14': ['blk09', 'blk19', 'blk29'],
}

BACKBONE_DEFAULTS = {
    'dinov2_vitl14': dict(K=16, embedding_dim=32, D=1024),
    'dinov2_vitg14': dict(K=16, embedding_dim=32, D=1536),
}


def find_checkpoint(backbone, layer, K, emb):
    ckpt_dir = os.path.join(ROOT, 'checkpoints', backbone)
    pattern = f'{layer}_K{K}_emb{emb}_*_ws_*_s42.pt'
    matches = glob.glob(os.path.join(ckpt_dir, pattern))
    preferred = [m for m in matches
                 if '_mse' not in m and '_fzR' not in m
                 and 'lmbda0.5' in m and '_km_' not in m
                 and '_rot' not in m and '_te' not in m
                 and '_tslin' not in m]
    if preferred:
        return sorted(preferred)[0]
    fallback = [m for m in matches
                if '_mse' not in m and '_fzR' not in m]
    return sorted(fallback)[0] if fallback else None


def build_jobs():
    jobs = []
    for backbone, layers in BACKBONE_LAYERS.items():
        cfg = BACKBONE_DEFAULTS[backbone]
        for layer in layers:
            ckpt = find_checkpoint(backbone, layer, cfg['K'],
                                   cfg['embedding_dim'])
            cmd_parts = [
                sys.executable, os.path.join(ROOT, 'compute_sensitivity.py'),
                '--backbone', backbone,
                '--layer', layer,
                '--K', str(cfg['K']),
                '--embedding_dim', str(cfg['embedding_dim']),
                '--n_diag', '200',
                '--max_train_images', '5000',
            ]
            if ckpt:
                cmd_parts += ['--codec_path', ckpt]
            layer_idx = int(layer[-2:])
            est_min = 15 + layer_idx * 2
            jobs.append({
                'backbone': backbone,
                'layer': layer,
                'ckpt': ckpt,
                'cmd': cmd_parts,
                'est_min': est_min,
            })
    return jobs


def schedule(jobs, gpu_ids):
    """Greedy longest-first scheduling."""
    jobs_sorted = sorted(jobs, key=lambda j: -j['est_min'])
    queues = {g: [] for g in gpu_ids}
    loads = {g: 0 for g in gpu_ids}
    for j in jobs_sorted:
        g = min(loads, key=loads.get)
        j['gpu'] = g
        queues[g].append(j)
        loads[g] += j['est_min']
    return queues, loads


def run_gpu_queue(gpu_id, queue, dry_run=False):
    for j in queue:
        cmd = j['cmd'] + ['--gpu', str(gpu_id)]
        cmd_str = ' '.join(cmd)
        print(f"[GPU {gpu_id}] {j['backbone']} {j['layer']} "
              f"(~{j['est_min']}min)")
        if dry_run:
            print(f"  CMD: {cmd_str}")
            continue
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=ROOT)
        elapsed = (time.time() - t0) / 60
        status = 'OK' if proc.returncode == 0 else f'FAIL({proc.returncode})'
        print(f"  [{status}] {elapsed:.1f}min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    jobs = build_jobs()
    print(f"Total jobs: {len(jobs)}")
    for j in jobs:
        ckpt_tag = 'OK' if j['ckpt'] else 'NO_CKPT (Identity+OPQ only)'
        print(f"  {j['backbone']:15s} {j['layer']:6s}  ~{j['est_min']:3d}min  "
              f"ckpt={ckpt_tag}")

    queues, loads = schedule(jobs, args.gpus)
    print(f"\nSchedule ({len(args.gpus)} GPUs):")
    for g in sorted(queues):
        tasks = ', '.join(f"{j['layer']}" for j in queues[g])
        print(f"  GPU {g}: {tasks}  ({loads[g]}min)")

    if args.dry_run:
        print("\n[DRY RUN] Commands:")
        for g in sorted(queues):
            for j in queues[g]:
                print(f"  CUDA_VISIBLE_DEVICES={g} {' '.join(j['cmd'])} --gpu {g}")
        return

    import multiprocessing
    procs = []
    for g in sorted(queues):
        p = multiprocessing.Process(
            target=run_gpu_queue, args=(g, queues[g]))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    print("\nDone. Check results/sensitivity/ for output JSONs.")


if __name__ == '__main__':
    main()
