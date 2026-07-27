#!/usr/bin/env python
"""Select the Phase-C beta using only the current run's Phase-B validation.

Rule: among positive-beta candidates whose validation D0 is within rho_d of
the beta=0 Phase-B baseline, minimise held-out epsilon span, then validation
D0. If no positive candidate satisfies the constraint, select the positive
candidate with the smallest D0 and mark the decision as a constraint
violation so the result cannot be mistaken for a certified choice.
"""

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


def _load_candidates(results_dir):
    candidates = []
    for path in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
        if '_phaseB_' not in os.path.basename(path):
            continue
        with open(path) as f:
            result = json.load(f)
        cfg = result.get('config', {})
        history = result.get('history', [])
        epochs = int(cfg.get('epochs', -1))
        audit = result.get('reconstruction_audit', {})
        if epochs <= 0 or len(history) != epochs:
            continue
        if not audit.get('passed', False):
            continue
        val_d0 = history[-1].get('val_D0')
        if val_d0 is None:
            continue
        beta = float(cfg.get('beta', 0.0))
        val_metrics = result.get('heldout_metrics', {}).get('val')
        span = None
        if val_metrics is not None:
            span = val_metrics.get('eps_g', {}).get('global_span')
        candidates.append({
            'path': path,
            'beta': beta,
            'val_D0': float(val_d0),
            'val_eps_span': None if span is None else float(span),
        })
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--rho-d', type=float, default=0.05)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    candidates = _load_candidates(args.results_dir)
    baselines = [c for c in candidates if c['beta'] == 0.0]
    positive = [
        c for c in candidates
        if c['beta'] > 0 and c['val_eps_span'] is not None
    ]
    if len(baselines) != 1:
        raise SystemExit(
            f'expected exactly one complete beta=0 Phase-B result, '
            f'found {len(baselines)}')
    if not positive:
        raise SystemExit('no complete positive-beta Phase-B candidates')

    baseline = baselines[0]
    d0_limit = baseline['val_D0'] * (1.0 + args.rho_d)
    eligible = [c for c in positive if c['val_D0'] <= d0_limit]
    if eligible:
        chosen = min(
            eligible, key=lambda c: (c['val_eps_span'], c['val_D0']))
        status = 'constraint_satisfied'
    else:
        chosen = min(positive, key=lambda c: c['val_D0'])
        status = 'constraint_violation'

    decision = {
        'rule': (
            'positive beta; val_D0 <= beta0*(1+rho_d); '
            'min val epsilon span, tie-break by val_D0'),
        'rho_d': args.rho_d,
        'baseline': baseline,
        'd0_limit': d0_limit,
        'status': status,
        'selected': chosen,
        'candidates': candidates,
    }
    _atomic_json(args.output, decision)
    print(chosen['beta'])


if __name__ == '__main__':
    main()
