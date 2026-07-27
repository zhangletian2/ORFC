#!/usr/bin/env python
"""Select the final Phase-B/C configuration on seed-42 validation only."""

import argparse
import glob
import json
import os
import tempfile


def _atomic_json(path, value):
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix='.json.tmp')
    os.close(fd)
    try:
        with open(tmp, 'w') as f:
            json.dump(value, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--rho-d', type=float, default=0.05)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, '*.json'))):
        name = os.path.basename(path)
        if '_phaseB_' not in name and '_phaseC_' not in name:
            continue
        with open(path) as f:
            r = json.load(f)
        cfg = r.get('config', {})
        hist = r.get('history', [])
        if len(hist) != int(cfg.get('epochs', -1)):
            continue
        if not r.get('reconstruction_audit', {}).get('passed', False):
            continue
        val_d0 = hist[-1].get('val_D0')
        val = r.get('heldout_metrics', {}).get('val')
        if val_d0 is None:
            continue
        span = None if val is None else val.get('eps_g', {}).get('global_span')
        rows.append({
            'path': path,
            'beta': float(cfg.get('beta', 0.0)),
            'val_D0': float(val_d0),
            'val_eps_span': None if span is None else float(span),
            'step_mode': cfg.get('step_mode', 'joint'),
            'alt_u_steps': int(cfg.get('alt_u_steps', 1)),
            'alt_c_steps': int(cfg.get('alt_c_steps', 1)),
        })

    baseline = [
        x for x in rows
        if x['beta'] == 0 and x['step_mode'] == 'joint'
        and '_phaseB_' in os.path.basename(x['path'])
    ]
    if len(baseline) != 1:
        raise SystemExit(f'expected one Phase-B beta=0 baseline, got {len(baseline)}')
    candidates = [
        x for x in rows
        if x['beta'] > 0 and x['val_eps_span'] is not None
    ]
    if not candidates:
        raise SystemExit('no valid positive-beta Phase-B/C candidates')

    limit = baseline[0]['val_D0'] * (1 + args.rho_d)
    eligible = [x for x in candidates if x['val_D0'] <= limit]
    pool = eligible or candidates
    chosen = min(pool, key=lambda x: (x['val_eps_span'], x['val_D0']))
    decision = {
        'rho_d': args.rho_d,
        'baseline': baseline[0],
        'd0_limit': limit,
        'status': ('constraint_satisfied' if eligible
                   else 'constraint_violation'),
        'selected': chosen,
        'candidates': rows,
    }
    _atomic_json(args.output, decision)
    print(
        f"{chosen['step_mode']} {chosen['beta']} "
        f"{chosen['alt_u_steps']} {chosen['alt_c_steps']}")


if __name__ == '__main__':
    main()
