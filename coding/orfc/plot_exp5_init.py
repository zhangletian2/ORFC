#!/usr/bin/env python
"""
Exp 5 — Initialization Impact Analysis

Reads D-group JSONs (init variants with full history) + OPQ warm-start
baselines, then generates a 2x2 figure:
  (a) K=16 convergence: epoch vs L_ref (val_loss)
  (b) K=256 convergence: same
  (c) Rate convergence: epoch vs rate_bits for both K configs
  (d) Terminal performance: grouped bar chart Acc / mIoU
"""

import json, os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'font.family': 'sans-serif',
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
})

ROOT = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(ROOT, 'results', 'soft_pq', 'dinov2_vitl14')
FIG_DIR = os.path.join(ROOT, 'figures')

INIT_LABELS = {
    'identity': r'$R\!=\!I$',
    'pca': 'PCA',
    'random_orth': 'Rand SO(D)',
}
INIT_COLORS = {
    'identity': '#E24A33',
    'pca': '#55A868',
    'random_orth': '#FFA500',
}
SKIP_ROTS = {'opq'}
WS_COLOR = '#B983CD'


def _load_manifest_group(group):
    mf = os.path.join(ROOT, 'experiment_manifest.json')
    m = json.load(open(mf))
    return {k: v for k, v in m.items() if v['group'] == group}


def load_exp5_data():
    d_jobs = _load_manifest_group('D')

    records = []
    for tag, v in sorted(d_jobs.items()):
        jf = v['json_result']
        if not os.path.exists(jf):
            continue
        with open(jf) as fh:
            j = json.load(fh)
        cfg = j['config']
        records.append(dict(
            tag=tag,
            K=cfg['K'], d=cfg['embedding_dim'],
            rot=cfg.get('init_rotation', 'opq'),
            ws=cfg.get('warm_start_opq', True),
            acc=j['soft_pq_acc'],
            miou=j.get('soft_pq_miou', 0),
            dl=j.get('soft_pq_delta_l', 0),
            bpt=j['rate_info']['rans_bpt'],
            history=j.get('history', []),
        ))

    for K, d in [(16, 32), (256, 16)]:
        path = os.path.join(
            JSON_DIR,
            f'blk20_K{K}_emb{d}_bt1024_ws_lmbda0.5_tau0.5_'
            f'lr0.0003_ep100_n5000_s42.json')
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            j = json.load(fh)
        hist = j.get('history', [])
        if not hist:
            hist = _parse_log_history(K, d)
        records.append(dict(
            tag=f'ws_K{K}',
            K=K, d=d,
            rot='opq', ws=True,
            acc=j['soft_pq_acc'],
            miou=j.get('soft_pq_miou', 0),
            dl=j.get('soft_pq_delta_l', 0),
            bpt=j['rate_info']['rans_bpt'],
            history=hist,
        ))

    return records


_LOG_EP_RE = re.compile(
    r'ep\s+(\d+)/\d+\s+D=([\d.]+)\s+ppl=([\d.]+)\s+R=([\d.]+)b/t'
    r'.*?val=([\d.]+)')

def _parse_log_history(K, d):
    """Parse training history from batch_all log files (seed 42 or 43)."""
    log_dir = os.path.join(ROOT, 'logs', 'batch_all')
    candidates = [
        os.path.join(log_dir, f'ms_L14_blk20_K{K}_e{d}_s42.log'),
        os.path.join(log_dir, f'ms_L14_blk20_K{K}_e{d}_s43.log'),
    ]
    for lp in candidates:
        if not os.path.exists(lp):
            continue
        with open(lp) as f:
            text = f.read()
        if f'\u03bb={0.5}' not in text and 'λ=0.5' not in text:
            continue
        hist = []
        for m in _LOG_EP_RE.finditer(text):
            hist.append({
                'epoch': int(m.group(1)),
                'val_loss': float(m.group(5)),
                'rate_bits': float(m.group(4)),
                'perplexity': float(m.group(3)),
                'dead_entries': 0,
                'temperature': 0,
            })
        if hist:
            print(f"  [log] parsed {len(hist)} epochs from {os.path.basename(lp)}")
            return hist
    return []


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {FIG_DIR}/{name}.[pdf|png]")


def _init_label(r):
    if r['ws']:
        return 'OPQ warm-start'
    return INIT_LABELS.get(r['rot'], r['rot'])


def _init_color(r):
    if r['ws']:
        return WS_COLOR
    return INIT_COLORS.get(r['rot'], '#888')


def _keep(r):
    if r['K'] != 16:
        return False
    if r['ws']:
        return True
    return r['rot'] not in SKIP_ROTS


def plot_exp5(records):
    init_order = ['identity', 'pca', 'random_orth']

    def _clean(ax):
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

    subset = [r for r in records if _keep(r)]

    # ── (a) Convergence ──
    fig_a, ax = plt.subplots(figsize=(3.8, 2.6))
    for r in sorted(subset, key=lambda r: (r['ws'],
                    init_order.index(r['rot']) if r['rot'] in init_order else 99)):
        h = r['history']
        if not h:
            ax.axhline(r['dl'], color=_init_color(r), ls=':', lw=0.8,
                       alpha=.7, label=_init_label(r) + ' (end)')
            continue
        epochs = [e['epoch'] for e in h]
        vl = [e['val_loss'] for e in h]
        ax.plot(epochs, vl, color=_init_color(r), lw=1.0, alpha=.9,
                label=_init_label(r))
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$L_{\rm ref}$ (val)')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.legend(fontsize=6, loc='upper right', framealpha=.9, handlelength=1.2)
    _clean(ax)
    fig_a.tight_layout(pad=0.4)
    _save(fig_a, 'exp5_convergence')

    # ── (b) Rate convergence ──
    fig_b, ax = plt.subplots(figsize=(3.8, 2.6))
    for r in sorted(subset, key=lambda r: (r['ws'],
                    init_order.index(r['rot']) if r['rot'] in init_order else 99)):
        h = r['history']
        if not h:
            ax.axhline(r['bpt'], color=_init_color(r), ls=':', lw=0.8,
                       alpha=.7, label=_init_label(r) + ' (end)')
            continue
        epochs = [e['epoch'] for e in h]
        rbits = [e['rate_bits'] for e in h]
        ax.plot(epochs, rbits, '-', color=_init_color(r), lw=1.0,
                alpha=.9, label=_init_label(r))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BPFP (bits / feature point)')
    ax.legend(fontsize=6, loc='upper right', framealpha=.9, handlelength=1.2)
    _clean(ax)
    fig_b.tight_layout(pad=0.4)
    _save(fig_b, 'exp5_rate')

    # ── (c) Terminal bars ──
    methods = ['identity', 'pca', 'random_orth', 'ws']
    method_labels = [r'$R\!=\!I$', 'PCA', 'Rand SO', 'Warm-start']
    x = np.arange(len(methods))
    w = 0.3

    fig_c, ax = plt.subplots(figsize=(3.5, 2.8))
    accs, mious = [], []
    for meth in methods:
        found = None
        for r in records:
            if r['K'] == 16:
                if meth == 'ws' and r['ws']:
                    found = r
                    break
                elif not r['ws'] and r['rot'] == meth:
                    found = r
                    break
        accs.append(found['acc'] * 100 if found else 0)
        mious.append(found['miou'] * 100 if found else 0)

    ax.bar(x - w / 2, accs, w, label='Acc@1', color='#B983CD', alpha=.8)
    ax.bar(x + w / 2, mious, w, label='mIoU', color='#286CA0', alpha=.8)

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=7)
    ax.set_ylabel('Performance (%)')
    ax.legend(fontsize=6, loc='lower right', framealpha=.9, handlelength=1.0)
    _clean(ax)
    fig_c.tight_layout(pad=0.4)
    _save(fig_c, 'exp5_terminal')


def print_summary(records):
    W = 80
    print(f"\n{'=' * W}")
    print("Exp 5 Summary — Initialization Comparison (blk20)")
    print(f"{'=' * W}")
    print(f"{'K':>4} {'d':>3} {'init':>15s}  {'ws':>3}  {'acc':>6}  {'miou':>6}  {'bpt':>8}")
    print('-' * W)
    for r in sorted(records, key=lambda r: (r['K'], r['ws'], r['rot'])):
        lbl = _init_label(r)
        print(f"{r['K']:4d} {r['d']:3d} {lbl:>15s}  {str(r['ws']):>3}  "
              f"{r['acc']*100:5.1f}%  {r['miou']*100:5.1f}%  {r['bpt']:8.1f}")


def main():
    records = load_exp5_data()
    print(f"Loaded {len(records)} init configs\n")
    plot_exp5(records)
    print_summary(records)


if __name__ == '__main__':
    main()
