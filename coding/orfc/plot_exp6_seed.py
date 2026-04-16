#!/usr/bin/env python
"""
Exp 6 — Multi-seed Randomness Analysis

Collects E-group JSONs (seeds 43, 44) + seed-42 baselines, then:
  (a) Overlays R-D curves per seed for each (backbone, layer)
  (b) Computes Bjontegaard Delta Rate (BD-rate) between seeds
  (c) Reports mean +/- std of Acc, mIoU, delta_l across seeds
"""

import json, os, glob
from collections import defaultdict
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
FIG_DIR = os.path.join(ROOT, 'figures')

SEED_STYLES = {42: ('-', 'o'), 43: ('--', 's'), 44: (':', '^')}
SEED_COLORS = {42: '#B983CD', 43: '#286CA0', 44: '#E24A33'}


# ── BD-rate ──────────────────────────────────────────────────────────

def bd_rate(R1, Q1, R2, Q2):
    """Bjontegaard Delta Rate: percentage rate difference between two R-D
    curves at equal quality.  Positive = curve 2 uses more bits.

    Fits cubic polynomial lR = p(Q) in log-rate / quality space.
    Returns NaN when the overlap region is empty or data is insufficient.
    """
    R1, Q1, R2, Q2 = map(np.asarray, (R1, Q1, R2, Q2))
    if len(R1) < 2 or len(R2) < 2:
        return float('nan')

    lR1, lR2 = np.log(R1), np.log(R2)
    deg = min(3, len(R1) - 1, len(R2) - 1)
    p1 = np.polyfit(Q1, lR1, deg)
    p2 = np.polyfit(Q2, lR2, deg)

    q_lo = max(Q1.min(), Q2.min())
    q_hi = min(Q1.max(), Q2.max())
    if q_lo >= q_hi:
        return float('nan')

    p_diff = np.polysub(p2, p1)
    p_int = np.polyint(p_diff)
    integral = np.polyval(p_int, q_hi) - np.polyval(p_int, q_lo)
    return (np.exp(integral / (q_hi - q_lo)) - 1) * 100


# ── data loading ─────────────────────────────────────────────────────

def _is_standard(cfg):
    return (cfg.get('warm_start_opq', True)
            and not cfg.get('mse_loss', False)
            and not cfg.get('freeze_transform', False)
            and cfg.get('tau_start', 0.5) == 0.5
            and cfg.get('tau_end', 0.005) == 0.005
            and cfg.get('tau_schedule', 'exponential') == 'exponential'
            and cfg.get('init_rotation', 'opq') == 'opq'
            and cfg.get('lmbda', 0) == 0.5
            and cfg.get('lr', 0.0003) == 0.0003
            and cfg.get('epochs', 100) == 100)


def load_all_seed_data():
    """Load E-group JSONs + seed-42 baselines, grouped by
    (backbone, layer) -> {seed -> [(rate, acc, miou, dl, K, d)]}."""

    manifest = os.path.join(ROOT, 'experiment_manifest.json')
    m = json.load(open(manifest))
    e_jobs = {k: v for k, v in m.items() if v['group'] == 'E'}

    needed_keys = set()
    for tag, v in e_jobs.items():
        cfg = v['config']
        needed_keys.add((cfg['backbone'], cfg['layer'],
                         cfg['K'], cfg['embedding_dim']))

    groups = defaultdict(lambda: defaultdict(list))

    for tag, v in e_jobs.items():
        jf = v['json_result']
        if not os.path.exists(jf):
            continue
        with open(jf) as fh:
            j = json.load(fh)
        cfg = j['config']
        seed = cfg.get('seed', 42)
        bb, layer = cfg['backbone'], cfg['layer']
        groups[(bb, layer)][seed].append(dict(
            K=cfg['K'], d=cfg['embedding_dim'],
            acc=j['soft_pq_acc'], miou=j.get('soft_pq_miou', 0),
            dl=j.get('soft_pq_delta_l', 0), bpt=j['rate_info']['rans_bpt'],
        ))

    json_dirs = {
        'dinov2_vitl14': os.path.join(ROOT, 'results', 'soft_pq', 'dinov2_vitl14'),
        'dinov2_vitg14': os.path.join(ROOT, 'results', 'soft_pq', 'dinov2_vitg14'),
        'clip_vitl14': os.path.join(ROOT, 'results', 'soft_pq', 'clip_vitl14'),
    }

    for bb, layer, K, d in needed_keys:
        jdir = json_dirs.get(bb)
        if not jdir:
            continue
        for f in glob.glob(os.path.join(jdir, f'{layer}_K{K}_emb{d}_*_s42.json')):
            with open(f) as fh:
                j = json.load(fh)
            cfg = j['config']
            if cfg.get('seed', 42) != 42 or not _is_standard(cfg):
                continue
            groups[(bb, layer)][42].append(dict(
                K=cfg['K'], d=cfg['embedding_dim'],
                acc=j['soft_pq_acc'], miou=j.get('soft_pq_miou', 0),
                dl=j.get('soft_pq_delta_l', 0), bpt=j['rate_info']['rans_bpt'],
            ))
            break

    return groups


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {FIG_DIR}/{name}.[pdf|png]")


# ── plotting ─────────────────────────────────────────────────────────

def _pretty_short(bb, layer):
    blk_num = int(layer[-2:])
    if 'vitg' in bb:
        blk_num += 1
        return f'D2_vitg14_blk{blk_num:02d}'
    elif 'vitl' in bb:
        return f'D2_vitl14_{layer}'
    elif 'clip' in bb:
        return f'CLIP_vitl14_{layer}'
    return f'{bb}_{layer}'


def plot_rd_overlay(groups):
    bl_list = sorted(groups.keys())

    for bb, layer in bl_list:
        seed_data = groups[(bb, layer)]
        tag = _pretty_short(bb, layer)

        # ── Acc figure ──
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        for seed in sorted(seed_data.keys()):
            pts = sorted(seed_data[seed], key=lambda p: p['bpt'])
            rates = [p['bpt'] / 1024 for p in pts]
            accs = [p['acc'] * 100 for p in pts]
            ls, mk = SEED_STYLES.get(seed, ('-', 'x'))
            col = SEED_COLORS.get(seed, '#888')
            ax.plot(rates, accs, ls, marker=mk, color=col,
                    ms=4, lw=1.1, alpha=.85, label=f'seed {seed}')
        ax.set_xlabel('BPFP (bits / feature point)')
        ax.set_ylabel('Acc@1 (%)')
        ax.legend(fontsize=6, loc='lower right', framealpha=.9)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        _save(fig, f'exp6_seed_rd_acc_{tag}')

        # ── mIoU figure ──
        fig2, ax2 = plt.subplots(figsize=(3.5, 2.8))
        for seed in sorted(seed_data.keys()):
            pts = sorted(seed_data[seed], key=lambda p: p['bpt'])
            rates = [p['bpt'] / 1024 for p in pts]
            mious = [p['miou'] * 100 for p in pts]
            ls, mk = SEED_STYLES.get(seed, ('-', 'x'))
            col = SEED_COLORS.get(seed, '#888')
            ax2.plot(rates, mious, ls, marker=mk, color=col,
                     ms=4, lw=1.1, alpha=.85, label=f'seed {seed}')
        ax2.set_xlabel('BPFP (bits / feature point)')
        ax2.set_ylabel('mIoU (%)')
        ax2.legend(fontsize=6, loc='lower right', framealpha=.9)
        for sp in ('top', 'right'):
            ax2.spines[sp].set_visible(False)
        fig2.tight_layout()
        _save(fig2, f'exp6_seed_rd_miou_{tag}')


# ── statistics ───────────────────────────────────────────────────────

def compute_stats(groups):
    W = 100
    print(f"\n{'=' * W}")
    print("Exp 6 — Per-point mean +/- std across seeds")
    print(f"{'=' * W}")
    print(f"{'backbone':>15s} {'layer':>6s} {'K':>5s} {'d':>3s}  "
          f"{'acc_mean':>8s} {'acc_std':>8s}  "
          f"{'miou_mean':>9s} {'miou_std':>9s}  "
          f"{'bpt_mean':>9s} {'bpt_std':>8s}")
    print('-' * W)

    for (bb, layer) in sorted(groups.keys()):
        seed_data = groups[(bb, layer)]
        all_Kd = set()
        for pts in seed_data.values():
            for p in pts:
                all_Kd.add((p['K'], p['d']))

        for K, d in sorted(all_Kd):
            vals = defaultdict(list)
            for seed, pts in seed_data.items():
                for p in pts:
                    if p['K'] == K and p['d'] == d:
                        vals['acc'].append(p['acc'] * 100)
                        vals['miou'].append(p['miou'] * 100)
                        vals['bpt'].append(p['bpt'])
                        vals['dl'].append(p['dl'])

            n = len(vals['acc'])
            if n < 2:
                continue
            a_m, a_s = np.mean(vals['acc']), np.std(vals['acc'])
            m_m, m_s = np.mean(vals['miou']), np.std(vals['miou'])
            b_m, b_s = np.mean(vals['bpt']), np.std(vals['bpt'])
            print(f"{bb:>15s} {layer:>6s} {K:5d} {d:3d}  "
                  f"{a_m:7.2f}% {a_s:7.3f}%  "
                  f"{m_m:8.2f}% {m_s:8.3f}%  "
                  f"{b_m:9.1f} {b_s:7.2f}")

    print(f"\n{'=' * W}")
    print("Exp 6 — BD-rate (seed 42 as anchor, %; + = test uses more bits)")
    print(f"{'=' * W}")
    print(f"{'backbone':>15s} {'layer':>6s}  {'s43_cls':>8s} {'s44_cls':>8s}  "
          f"{'s43_seg':>8s} {'s44_seg':>8s}")
    print('-' * W)

    for (bb, layer) in sorted(groups.keys()):
        sd = groups[(bb, layer)]
        if 42 not in sd:
            continue
        ref = sorted(sd[42], key=lambda p: p['bpt'])
        R_ref = [p['bpt'] for p in ref]
        A_ref = [p['acc'] * 100 for p in ref]
        M_ref = [p['miou'] * 100 for p in ref]

        bd_vals = {}
        for seed in [43, 44]:
            if seed not in sd:
                bd_vals[seed] = (float('nan'), float('nan'))
                continue
            test = sorted(sd[seed], key=lambda p: p['bpt'])
            R_t = [p['bpt'] for p in test]
            A_t = [p['acc'] * 100 for p in test]
            M_t = [p['miou'] * 100 for p in test]
            bd_cls = bd_rate(R_ref, A_ref, R_t, A_t)
            bd_seg = bd_rate(R_ref, M_ref, R_t, M_t)
            bd_vals[seed] = (bd_cls, bd_seg)

        c43, s43 = bd_vals.get(43, (np.nan, np.nan))
        c44, s44 = bd_vals.get(44, (np.nan, np.nan))
        print(f"{bb:>15s} {layer:>6s}  {c43:+7.2f}% {c44:+7.2f}%  "
              f"{s43:+7.2f}% {s44:+7.2f}%")


def main():
    groups = load_all_seed_data()
    total = sum(sum(len(pts) for pts in sd.values())
                for sd in groups.values())
    print(f"Loaded {total} data points across "
          f"{len(groups)} (backbone, layer) groups\n")

    print("Plotting R-D overlays ...")
    plot_rd_overlay(groups)
    compute_stats(groups)


if __name__ == '__main__':
    main()
