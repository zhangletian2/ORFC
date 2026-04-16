#!/usr/bin/env python
"""
Exp 4 — Temperature Schedule Analysis

Reads C-group JSONs (with training history) and the default baseline,
then generates a 2x2 figure:
  (a) Convergence: epoch vs val_loss (L_ref)
  (b) Tau schedule: epoch vs effective temperature
  (c) Codeword diversity: epoch vs perplexity & dead entries
  (d) Terminal comparison: bar chart of Acc / mIoU / perplexity
"""

import json, os, glob
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

PALETTE = ['#E24A33', '#286CA0', '#B983CD', '#55A868',
           '#FFA500', '#8B0000', '#4682B4', '#999999']


def _load_manifest_group(group):
    mf = os.path.join(ROOT, 'experiment_manifest.json')
    m = json.load(open(mf))
    return {k: v for k, v in m.items() if v['group'] == group}


def _make_label(cfg):
    ts = cfg.get('tau_start', 0.5)
    te = cfg.get('tau_end', 0.005)
    sched = cfg.get('tau_schedule', 'exponential')

    if te == ts:
        return f'const τ={ts}'
    if sched == 'linear':
        return f'τ {ts}→{te} lin'
    if te != 0.005:
        return f'τ {ts}→{te}'
    return f'τ₀={ts}'


def load_exp4_data():
    """Load C-group JSONs + baseline into a list of dicts."""
    c_jobs = _load_manifest_group('C')

    records = []
    for tag, v in sorted(c_jobs.items()):
        jf = v['json_result']
        if not os.path.exists(jf):
            continue
        with open(jf) as fh:
            j = json.load(fh)
        cfg = j['config']
        records.append(dict(
            tag=tag,
            label=_make_label(cfg),
            tau_s=cfg.get('tau_start', 0.5),
            tau_e=cfg.get('tau_end', 0.005),
            sched=cfg.get('tau_schedule', 'exponential'),
            acc=j['soft_pq_acc'],
            miou=j.get('soft_pq_miou', 0),
            dl=j.get('soft_pq_delta_l', 0),
            bpt=j['rate_info']['rans_bpt'],
            history=j.get('history', []),
        ))

    baseline_path = os.path.join(
        JSON_DIR,
        'blk20_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.json')
    if os.path.exists(baseline_path):
        with open(baseline_path) as fh:
            j = json.load(fh)
        records.append(dict(
            tag='baseline',
            label='τ₀=0.5 (base)',
            tau_s=0.5, tau_e=0.005, sched='exponential',
            acc=j['soft_pq_acc'],
            miou=j.get('soft_pq_miou', 0),
            dl=j.get('soft_pq_delta_l', 0),
            bpt=j['rate_info']['rans_bpt'],
            history=j.get('history', []),
        ))

    return records


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {FIG_DIR}/{name}.[pdf|png]")


def plot_exp4(records):
    has_hist = [r for r in records if len(r['history']) > 0]
    all_recs = records

    def _clean(ax):
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

    # ── (a) Convergence ──
    fig_a, ax = plt.subplots(figsize=(3.8, 2.6))
    for i, r in enumerate(has_hist):
        epochs = [h['epoch'] for h in r['history']]
        vl = [h['val_loss'] for h in r['history']]
        ax.plot(epochs, vl, color=PALETTE[i % len(PALETTE)],
                lw=1.0, alpha=.9, label=r['label'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$L_{\rm ref}$ (val)')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.legend(fontsize=6.5, loc='upper right', ncol=2,
              framealpha=.9, handlelength=1.2)
    _clean(ax)
    fig_a.tight_layout(pad=0.4)
    _save(fig_a, 'exp4_convergence')

    # ── (b) Temperature schedule ──
    fig_b, ax = plt.subplots(figsize=(3.8, 2.6))
    for i, r in enumerate(has_hist):
        epochs = [h['epoch'] for h in r['history']]
        taus = [h['temperature'] for h in r['history']]
        ax.plot(epochs, taus, color=PALETTE[i % len(PALETTE)],
                lw=1.2, alpha=.9, label=r['label'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$\tau$')
    ax.set_yscale('log')
    ax.legend(fontsize=6.5, loc='lower left', ncol=2,
              framealpha=.9, handlelength=1.2)
    _clean(ax)
    fig_b.tight_layout(pad=0.4)
    _save(fig_b, 'exp4_tau_schedule')

    # ── (c) Codeword diversity ──
    fig_c, ax = plt.subplots(figsize=(3.8, 2.6))
    for i, r in enumerate(has_hist):
        epochs = [h['epoch'] for h in r['history']]
        perp = [h['perplexity'] for h in r['history']]
        ax.plot(epochs, perp, color=PALETTE[i % len(PALETTE)],
                lw=1.0, alpha=.9, label=r['label'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Perplexity')
    ax.legend(fontsize=6.5, loc='lower right', ncol=2,
              framealpha=.9, handlelength=1.2)
    _clean(ax)

    ax_dead = ax.twinx()
    for i, r in enumerate(has_hist):
        epochs = [h['epoch'] for h in r['history']]
        dead = [h['dead_entries'] for h in r['history']]
        if max(dead) > 0:
            ax_dead.plot(epochs, dead, '--', color=PALETTE[i % len(PALETTE)],
                         lw=0.7, alpha=.5)
    ax_dead.set_ylabel('Dead entries (dashed)', fontsize=7, color='#888')
    ax_dead.tick_params(axis='y', labelcolor='#888', labelsize=6)
    fig_c.tight_layout(pad=0.4)
    _save(fig_c, 'exp4_diversity')

    # ── (d) Terminal comparison bars ──
    fig_d, ax_bar = plt.subplots(figsize=(4.2, 3.2))
    labels = [r['label'] for r in all_recs]
    x = np.arange(len(all_recs))
    w = 0.25

    accs = [r['acc'] * 100 for r in all_recs]
    mious = [r['miou'] * 100 for r in all_recs]
    perps = []
    for r in all_recs:
        if r['history']:
            perps.append(r['history'][-1]['perplexity'])
        else:
            perps.append(np.nan)

    bars_a = ax_bar.bar(x - w, accs, w, color='#B983CD', alpha=.8)
    bars_m = ax_bar.bar(x, mious, w, color='#286CA0', alpha=.8)

    ax_bar2 = ax_bar.twinx()
    bars_p = ax_bar2.bar(x + w, perps, w, color='#55A868', alpha=.6)
    ax_bar2.set_ylabel('Perplexity', fontsize=7, color='#55A868')
    ax_bar2.tick_params(axis='y', labelcolor='#55A868', labelsize=6)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    ax_bar.set_ylabel('Performance (%)')
    _clean(ax_bar)

    handles = [bars_a, bars_m, bars_p]
    leg_labels = ['Acc@1', 'mIoU', 'Perplexity']
    ax_bar.legend(handles, leg_labels, fontsize=6,
                  loc='lower center', bbox_to_anchor=(0.5, 1.0),
                  ncol=3, framealpha=.9, handlelength=1.0,
                  borderaxespad=0.3)
    fig_d.tight_layout(pad=0.4, rect=[0, 0, 1, 0.95])
    _save(fig_d, 'exp4_terminal')


def print_summary(records):
    W = 85
    print(f"\n{'=' * W}")
    print("Exp 4 Summary — Temperature Schedule Comparison (blk20, K=16, d=32)")
    print(f"{'=' * W}")
    print(f"{'label':>22s}  {'acc':>6s}  {'miou':>6s}  {'dl':>10s}  {'bpt':>8s}  {'perp':>6s}")
    print('-' * W)
    for r in records:
        perp = r['history'][-1]['perplexity'] if r['history'] else float('nan')
        print(f"{r['label']:>22s}  {r['acc']*100:5.1f}%  {r['miou']*100:5.1f}%  "
              f"{r['dl']:10.0f}  {r['bpt']:8.1f}  {perp:6.2f}")


def main():
    records = load_exp4_data()
    print(f"Loaded {len(records)} temperature configs\n")
    plot_exp4(records)
    print_summary(records)


if __name__ == '__main__':
    main()
