#!/usr/bin/env python
"""
Exp 2 — Batch correlation analysis scheduler.

Generates commands to run analysis_intro_v2.py --mode correlation
for all (backbone, layer) combinations.

Usage:
    python run_exp2_batch.py --dry_run
    python run_exp2_batch.py --gpus 0 1 2
"""

import os, sys, argparse, subprocess, time

ROOT = os.path.dirname(os.path.abspath(__file__))

BACKBONE_LAYERS = {
    'dinov2_vitl14': ['blk05', 'blk10', 'blk15', 'blk20'],
    'dinov2_vitg14': ['blk09', 'blk19', 'blk29'],
}

LAYER_EST_MIN = {
    'blk05': 90, 'blk10': 70, 'blk15': 60, 'blk20': 50,
    'blk09': 100, 'blk19': 80, 'blk29': 70,
}


def build_jobs():
    jobs = []
    for backbone, layers in BACKBONE_LAYERS.items():
        for layer in layers:
            cmd = [
                sys.executable,
                os.path.join(ROOT, 'analysis_intro_v2.py'),
                '--mode', 'correlation',
                '--backbone', backbone,
                '--layer', layer,
                '--eval_seg',
            ]
            jobs.append({
                'backbone': backbone,
                'layer': layer,
                'cmd': cmd,
                'est_min': LAYER_EST_MIN.get(layer, 60),
            })
    return jobs


def schedule(jobs, gpu_ids):
    jobs_sorted = sorted(jobs, key=lambda j: -j['est_min'])
    queues = {g: [] for g in gpu_ids}
    loads = {g: 0 for g in gpu_ids}
    for j in jobs_sorted:
        g = min(loads, key=loads.get)
        j['gpu'] = g
        queues[g].append(j)
        loads[g] += j['est_min']
    return queues, loads


def run_gpu_queue(gpu_id, queue):
    for j in queue:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"[GPU {gpu_id}] {j['backbone']} {j['layer']} "
              f"(~{j['est_min']}min)", flush=True)
        t0 = time.time()
        proc = subprocess.run(j['cmd'], cwd=ROOT, env=env)
        elapsed = (time.time() - t0) / 60
        status = 'OK' if proc.returncode == 0 else f'FAIL({proc.returncode})'
        print(f"  [{status}] {elapsed:.1f}min", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    jobs = build_jobs()
    print(f"Total jobs: {len(jobs)}")
    for j in jobs:
        print(f"  {j['backbone']:15s} {j['layer']:6s}  ~{j['est_min']:3d}min")

    queues, loads = schedule(jobs, args.gpus)
    print(f"\nSchedule ({len(args.gpus)} GPUs):")
    for g in sorted(queues):
        tasks = ', '.join(f"{j['backbone'].split('_')[0]}_{j['layer']}"
                          for j in queues[g])
        print(f"  GPU {g}: {tasks}  ({loads[g]}min)")

    if args.dry_run:
        print("\n[DRY RUN] Commands:")
        for g in sorted(queues):
            for j in queues[g]:
                print(f"  CUDA_VISIBLE_DEVICES={g} {' '.join(j['cmd'])}")
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

    print("\nDone. Check results/analysis_intro_v2/correlation_*.json")


if __name__ == '__main__':
    main()
