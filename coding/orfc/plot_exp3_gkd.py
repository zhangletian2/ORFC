#!/usr/bin/env python
"""
Exp 3 — G/K/d Sensitivity Analysis

Reads blk20 JSONs from results/soft_pq/dinov2_vitl14/ and generates:
  (a) K sweep at d=32 & d=16: Acc/mIoU vs K with OPQ baseline
  (b) d sweep at K=16: Acc/mIoU vs d (G=1024/d)
  (c) R-D curves: rate vs Acc/mIoU for Ours vs OPQ at different (K,d)
  (d) Ours gain heatmap over OPQ
  + summary table printed to stdout
"""

import json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

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

C_Ours = '#B983CD'
C_OPQ = '#286CA0'
C_Ours2 = '#E24A33'
C_OPQ2 = '#999999'


def load_blk20():
    out = []
    for f in sorted(glob.glob(os.path.join(JSON_DIR, 'blk20_*.json'))):
        with open(f) as fh:
            j = json.load(fh)
        c = j['config']
        out.append(dict(
            K=c['K'], d=c['embedding_dim'],
            G=c['bottleneck_dim'] // c['embedding_dim'],
            lm=c.get('lmbda', 0),
            seed=c.get('seed', 42),
            ws=c.get('warm_start_opq', True),
            mse=c.get('mse_loss', False),
            fzR=c.get('freeze_transform', False),
            tau_s=c.get('tau_start', 0.5),
            tau_e=c.get('tau_end', 0.005),
            ts=c.get('tau_schedule', 'exponential'),
            rot=c.get('init_rotation', 'opq'),
            lr=c.get('lr', 0.0003),
            ep=c.get('epochs', 100),
            acc=j['soft_pq_acc'],
            opq_acc=j.get('std_opq_acc', 0),
            miou=j.get('soft_pq_miou', 0),
            opq_miou=j.get('std_opq_miou', 0),
            dl=j.get('soft_pq_delta_l', 0),
            opq_dl=j.get('std_opq_delta_l', 0),
            bpt=j['rate_info']['rans_bpt'],
            file=os.path.basename(f),
        ))
    return out


def _is_std(r):
    """Standard hyperparams: seed 42, ws, default tau/rot/lr/ep, no mse/fzR."""
    return (r['seed'] == 42 and r['ws'] and not r['mse'] and not r['fzR']
            and r['tau_s'] == 0.5 and r['tau_e'] == 0.005
            and r['ts'] == 'exponential' and r['rot'] == 'opq'
            and r['lr'] == 0.0003 and r['ep'] == 100)


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {FIG_DIR}/{name}.[pdf|png]")


# ── (a) K sweep ──────────────────────────────────────────────────────
def plot_k_sweep(data):
    std = [r for r in data if _is_std(r) and r['lm'] == 0.5]

    configs = [
        (32, C_Ours, C_OPQ,  r'$d\!=\!32$  ($G\!=\!32$)', 'd32'),
        (16, C_Ours2, C_OPQ2, r'$d\!=\!16$  ($G\!=\!64$)', 'd16'),
    ]
    for dv, co, cb, title, tag in configs:
        pts = sorted([r for r in std if r['d'] == dv], key=lambda r: r['K'])
        if not pts:
            continue
        x = np.log2([r['K'] for r in pts])

        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        ax.plot(x, [r['acc'] * 100 for r in pts], '-^', color=co,
                ms=5, lw=1.2, label='Ours Acc')
        ax.plot(x, [r['opq_acc'] * 100 for r in pts], '--s', color=cb,
                ms=4, lw=1.0, alpha=.7, label='OPQ Acc')
        ax.plot(x, [r['miou'] * 100 for r in pts], '-v', color=co,
                ms=5, lw=1.2, alpha=.7, label='Ours mIoU')
        ax.plot(x, [r['opq_miou'] * 100 for r in pts], '--D', color=cb,
                ms=4, lw=1.0, alpha=.5, label='OPQ mIoU')

        ax.set_xticks(x)
        ax.set_xticklabels([str(r['K']) for r in pts])
        ax.set_xlabel(r'Codebook size $K$')
        ax.set_ylabel('Performance (%)')
        ax.set_title(title, pad=4)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=6, loc='lower right', framealpha=.9,
                  handlelength=1.5)
        fig.tight_layout(pad=0.4)
        _save(fig, f'exp3_gkd_k_sweep_{tag}')


# ── (b) d sweep at K=16 ─────────────────────────────────────────────
def plot_d_sweep(data):
    std = [r for r in data
           if _is_std(r) and r['lm'] == 0.5 and r['K'] == 16]
    pts = sorted(std, key=lambda r: r['d'])
    if not pts:
        print("  [skip] no d-sweep data for K=16")
        return

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    ds = [r['d'] for r in pts]
    Gs = [1024 // d for d in ds]

    ax.plot(ds, [r['acc'] * 100 for r in pts], '-^', color=C_Ours,
            ms=6, lw=1.3, label='Ours Acc')
    ax.plot(ds, [r['opq_acc'] * 100 for r in pts], '--s', color=C_OPQ,
            ms=5, lw=1.0, alpha=.7, label='OPQ Acc')
    ax.plot(ds, [r['miou'] * 100 for r in pts], '-v', color=C_Ours,
            ms=6, lw=1.3, alpha=.7, label='Ours mIoU')
    ax.plot(ds, [r['opq_miou'] * 100 for r in pts], '--D', color=C_OPQ,
            ms=5, lw=1.0, alpha=.5, label='OPQ mIoU')

    ax.set_xlabel(r'Subvector dimension $d$')
    ax.set_ylabel('Performance (%)')
    ax.set_xticks(ds)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ds)
    ax2.set_xticklabels([f'G={g}' for g in Gs], fontsize=6)
    ax2.tick_params(length=0)
    ax2.spines['top'].set_visible(False)

    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=6.5, loc='lower left', framealpha=.9)
    fig.tight_layout()
    _save(fig, 'exp3_gkd_d_sweep')


# ── (c) R-D curves ──────────────────────────────────────────────────
def _annotate_k(ax, pts, xs, color, y_key, y_off_sign=1):
    """Annotate K values with alternating two-tier offsets."""
    for i, (r, x) in enumerate(zip(pts, xs)):
        y_off = y_off_sign * (9 if i % 2 == 0 else 17)
        ax.annotate(f'N={r["K"]}', (x, r[y_key] * 100),
                    textcoords='offset points', xytext=(0, y_off),
                    fontsize=5, color=color, alpha=.85, ha='center')


def plot_rd(data):
    std = [r for r in data if _is_std(r) and r['lm'] == 0.5]

    line_cfgs = [
        (32, '^', C_Ours, C_OPQ),
        (16, 'D', C_Ours2, C_OPQ2),
    ]

    rd_plots = [
        ('acc', 'opq_acc', 'Acc@1 (%)', 'exp3_gkd_rd_acc'),
        ('miou', 'opq_miou', 'mIoU (%)', 'exp3_gkd_rd_miou'),
    ]

    for y_key, y_opq_key, ylabel, fname in rd_plots:
        fig, ax = plt.subplots(figsize=(3.5, 2.8))

        for ci, (dv, mk, co, cb) in enumerate(line_cfgs):
            pts = sorted([r for r in std if r['d'] == dv],
                         key=lambda r: r['bpt'])
            if not pts:
                continue
            rates = [r['bpt'] / 1024 for r in pts]
            lbl = f'd={dv}'

            ax.plot(rates, [r[y_key] * 100 for r in pts],
                    f'-{mk}', color=co, ms=5, lw=1.2, label=f'Ours {lbl}')
            ax.plot(rates, [r[y_opq_key] * 100 for r in pts],
                    '--s', color=cb, ms=4, lw=1.0, alpha=.6,
                    label=f'OPQ {lbl}')

            sign = 1 if ci == 0 else -1
            _annotate_k(ax, pts, rates, co, y_key, y_off_sign=sign)

        ax.set_xlabel('BPFP')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=6, loc='lower right', framealpha=.9)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        fig.tight_layout(pad=0.4)
        _save(fig, fname)


# ── (d) gain heatmap ────────────────────────────────────────────────
def plot_gain(data):
    std = [r for r in data if _is_std(r) and r['lm'] == 0.5]

    ga, gm = {}, {}
    for r in std:
        k = (r['K'], r['d'])
        ga[k] = (r['acc'] - r['opq_acc']) * 100
        gm[k] = (r['miou'] - r['opq_miou']) * 100

    Ks = sorted({k for k, _ in ga})
    ds = sorted({d for _, d in ga})
    ma = np.full((len(Ks), len(ds)), np.nan)
    mm = np.full((len(Ks), len(ds)), np.nan)
    for i, K in enumerate(Ks):
        for j, d in enumerate(ds):
            if (K, d) in ga:
                ma[i, j] = ga[(K, d)]
                mm[i, j] = gm[(K, d)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.5))

    for ax, mat, title in [
        (ax1, ma, r'$\Delta$Acc  (pp)'),
        (ax2, mm, r'$\Delta$mIoU  (pp)'),
    ]:
        vabs = max(np.nanmax(np.abs(mat)), 1e-6)
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        im = ax.imshow(mat, cmap='RdYlGn', norm=norm, aspect='auto')

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                col = 'white' if abs(v) > vabs * 0.55 else 'black'
                ax.text(j, i, f'{v:+.1f}', ha='center', va='center',
                        fontsize=7, fontweight='bold', color=col)

        ax.set_xticks(range(len(ds)))
        ax.set_xticklabels([str(d) for d in ds])
        ax.set_yticks(range(len(Ks)))
        ax.set_yticklabels([str(K) for K in Ks])
        ax.set_xlabel(r'Subvector dim $d$')
        ax.set_ylabel(r'Codebook size $K$')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=.8, pad=.04)

    fig.tight_layout()
    _save(fig, 'exp3_gkd_gain')


# ── summary table ────────────────────────────────────────────────────
def summary_table(data):
    std = sorted([r for r in data if _is_std(r) and r['lm'] == 0.5],
                 key=lambda r: (r['d'], r['K']))
    W = 90
    print(f"\n{'=' * W}")
    print("Exp 3 Summary — Ours vs OPQ  (blk20, lm=0.5, standard hparams)")
    print(f"{'=' * W}")
    print(f"{'K':>5} {'d':>3} {'G':>4} {'rate':>8}  "
          f"{'Ours_acc':>8} {'OPQ_acc':>8} {'da':>6}  "
          f"{'Ours_mIoU':>9} {'OPQ_mIoU':>9} {'dm':>6}")
    print('-' * W)
    for r in std:
        da = (r['acc'] - r['opq_acc']) * 100
        dm = (r['miou'] - r['opq_miou']) * 100
        print(f"{r['K']:5d} {r['d']:3d} {r['G']:4d} {r['bpt']:8.1f}  "
              f"{r['acc']*100:7.1f}% {r['opq_acc']*100:7.1f}% {da:+5.1f}  "
              f"{r['miou']*100:8.1f}% {r['opq_miou']*100:8.1f}% {dm:+5.1f}")


# ── main ─────────────────────────────────────────────────────────────
def main():
    data = load_blk20()
    print(f"Loaded {len(data)} blk20 JSONs\n")

    # print("(a) K sweep ...")
    # plot_k_sweep(data)
    # print("(b) d sweep ...")
    # plot_d_sweep(data)
    print("(c) R-D curves ...")
    plot_rd(data)
    # print("(d) gain heatmap ...")
    # plot_gain(data)
    # summary_table(data)


if __name__ == '__main__':
    main()
