#!/usr/bin/env python
"""
Exp 1 — Per-group Sensitivity Visualization

Reads sensitivity JSONs from results/sensitivity/ and generates:
  (a) Per-group sensitivity heatmap for each (backbone, layer)
  (b) Gini coefficient comparison across conditions / layers
  (c) CV (coefficient of variation) summary
"""

import json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

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
SENS_DIR = os.path.join(ROOT, 'results', 'sensitivity')
LEGACY_DIR = os.path.join(ROOT, 'results', 'analysis_intro_v2')
FIG_DIR = os.path.join(ROOT, 'figures')

COND_COLORS = {'identity': '#777', 'opq': '#286CA0', 'trained': '#B983CD'}
COND_LABELS = {'identity': 'Identity', 'opq': 'OPQ',
               'trained': r'Ours ($L_{\rm ref}$)'}

ABLATION_TYPES = ['ablation_lref', 'ablation_cls']
ABLATION_LABELS = {
    'ablation_lref': r'$\Delta L_{\rm ref}$',
    'ablation_cls': r'$\Delta L_{\rm cls}$',
}


def _pretty_title(backbone, layer):
    """Format backbone+layer into a readable title.
    G/14 block indices are displayed 1-indexed (blk09->Block 10)."""
    blk_num = int(layer[-2:])
    if 'vitg' in backbone:
        blk_num += 1
        bb_short = 'DINOv2 ViT-G/14'
    elif 'vitl' in backbone:
        bb_short = 'DINOv2 ViT-L/14'
    elif 'clip' in backbone:
        bb_short = 'CLIP ViT-L/14'
    else:
        bb_short = backbone
    return f'{bb_short}  Block {blk_num}'


def _pretty_short(backbone, layer):
    """Short label for bar chart x-axis."""
    blk_num = int(layer[-2:])
    if 'vitg' in backbone:
        blk_num += 1
        prefix = 'G/14'
    elif 'vitl' in backbone:
        prefix = 'L/14'
    else:
        prefix = backbone.split('_')[0]
    return f'{prefix}\nBlk {blk_num}'


def _gini(arr):
    a = np.sort(np.abs(np.asarray(arr, dtype=float)))
    n = len(a)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * a) / (n * np.sum(a) + 1e-30))
                 - (n + 1) / n)


def _ensure_gini(abl_dict):
    """Always recompute gini from values to use the corrected formula."""
    vals = abl_dict.get('values', [])
    abl_dict['gini'] = _gini(vals) if vals else 0.0
    return abl_dict


def load_sensitivity_data():
    """Load all sensitivity JSONs from results/sensitivity/ and
    legacy results/analysis_intro_v2/."""
    records = []

    for f in sorted(glob.glob(os.path.join(SENS_DIR, 'sens_*.json'))):
        with open(f) as fh:
            d = json.load(fh)
        for ck in d.get('conditions', {}):
            for ak in ABLATION_TYPES:
                if ak in d['conditions'][ck]:
                    _ensure_gini(d['conditions'][ck][ak])
        records.append(d)

    for f in sorted(glob.glob(os.path.join(
            LEGACY_DIR, 'cls_sensitivity_*.json'))):
        with open(f) as fh:
            d = json.load(fh)
        cfg = d['config']
        if 'backbone' not in cfg:
            cfg['backbone'] = 'dinov2_vitl14'
        for ck in d.get('conditions', {}):
            for ak in ABLATION_TYPES:
                if ak in d['conditions'][ck]:
                    _ensure_gini(d['conditions'][ck][ak])
        records.append(d)

    return records


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {FIG_DIR}/{name}.[pdf|png]")


def plot_per_group_lines(records):
    """One figure per (backbone, layer) showing per-group sensitivity
    for each condition."""
    for rec in records:
        cfg = rec['config']
        bb = cfg.get('backbone', 'dinov2_vitl14')
        layer = cfg['layer']
        conds = rec['conditions']

        for abl_key, abl_label in ABLATION_LABELS.items():
            available = [k for k in conds
                         if abl_key in conds[k]]
            if not available:
                continue

            fig, ax = plt.subplots(figsize=(4.0, 2.4))
            for key in ['identity', 'opq', 'trained']:
                if key not in available:
                    continue
                vals = np.array(conds[key][abl_key]['values'])
                xs = np.arange(len(vals))
                label = COND_LABELS.get(key, key)
                ax.plot(xs, vals, '-o', color=COND_COLORS.get(key, '#888'),
                        ms=3, lw=1.0, alpha=.9, label=label)

            ax.set_xlabel('Group index')
            ax.set_ylabel(abl_label)
            ax.set_xlim(-0.5, xs[-1] + 0.5)
            ax.set_xticks(np.arange(0, len(vals), max(1, len(vals) // 8)))
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            ax.legend(fontsize=6, loc='lower center',
                      bbox_to_anchor=(0.5, 1.0), ncol=3,
                      framealpha=.9, handlelength=1.5,
                      borderaxespad=0.3)
            ax.set_title(_pretty_title(bb, layer), fontsize=9, pad=22)

            short_bb = bb.replace('dinov2_', 'D2_').replace('clip_', 'CL_')
            abl_short = abl_key.replace('ablation_', '')
            fig.tight_layout(pad=0.4, rect=[0, 0, 1, 0.95])
            _save(fig, f'exp1_sens_{short_bb}_{layer}_{abl_short}')


def plot_gini_summary(records):
    """Bar chart comparing Gini across conditions and layers."""
    groups = {}
    for rec in records:
        cfg = rec['config']
        bb = cfg.get('backbone', 'dinov2_vitl14')
        layer = cfg['layer']
        key = (bb, layer)
        groups[key] = rec['conditions']

    if not groups:
        print("  [skip] no data for Gini summary")
        return

    for abl_key, abl_label in ABLATION_LABELS.items():
        data_points = []
        for (bb, layer), conds in sorted(groups.items()):
            for cond_key in ['identity', 'opq', 'trained']:
                if cond_key not in conds or abl_key not in conds[cond_key]:
                    continue
                g = conds[cond_key][abl_key].get('gini', 0)
                data_points.append((bb, layer, cond_key, g))

        if not data_points:
            continue

        labels_set = sorted(set((bb, layer) for bb, layer, _, _ in data_points))
        cond_keys = ['identity', 'opq', 'trained']
        x = np.arange(len(labels_set))
        w = 0.25

        fig, ax = plt.subplots(figsize=(max(4, len(labels_set) * 1.2), 2.8))
        for ci, ck in enumerate(cond_keys):
            ginis = []
            for bb, layer in labels_set:
                found = [g for b, l, c, g in data_points
                         if b == bb and l == layer and c == ck]
                ginis.append(found[0] if found else 0)
            ax.bar(x + (ci - 1) * w, ginis, w,
                   label=COND_LABELS.get(ck, ck),
                   color=COND_COLORS.get(ck, '#888'), alpha=.8)

        ax.set_xticks(x)
        short_labels = [_pretty_short(bb, layer)
                        for bb, layer in labels_set]
        ax.set_xticklabels(short_labels, fontsize=6)
        ax.set_ylabel(f'Gini coefficient ({abl_label})')
        ax.legend(fontsize=6.5, loc='upper right', framealpha=.9)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

        fig.tight_layout()
        abl_short = abl_key.replace('ablation_', '')
        _save(fig, f'exp1_gini_{abl_short}')


def print_summary(records):
    W = 90
    print(f"\n{'=' * W}")
    print("Exp 1 — Sensitivity Summary")
    print(f"{'=' * W}")
    print(f"{'backbone':>15s} {'layer':>6s} {'cond':>10s}  "
          f"{'CV_lref':>8s} {'Gini_lref':>10s}  "
          f"{'CV_cls':>8s} {'Gini_cls':>10s}")
    print('-' * W)
    for rec in records:
        cfg = rec['config']
        bb = cfg.get('backbone', 'dinov2_vitl14')
        layer = cfg['layer']
        for ck in ['identity', 'opq', 'trained']:
            if ck not in rec['conditions']:
                continue
            c = rec['conditions'][ck]
            lr = c.get('ablation_lref', {})
            cl = c.get('ablation_cls', {})
            cv_lr = lr.get('cv', float('nan'))
            gi_lr = lr.get('gini', float('nan'))
            cv_cl = cl.get('cv', float('nan'))
            gi_cl = cl.get('gini', float('nan'))
            print(f"{bb:>15s} {layer:>6s} {ck:>10s}  "
                  f"{cv_lr:>8.4f} {gi_lr:>10.4f}  "
                  f"{cv_cl:>8.4f} {gi_cl:>10.4f}")


def main():
    records = load_sensitivity_data()
    print(f"Loaded {len(records)} sensitivity records\n")

    if not records:
        print("No sensitivity data found. Run compute_sensitivity.py first.")
        return

    print("Per-group sensitivity lines ...")
    plot_per_group_lines(records)
    print_summary(records)


if __name__ == '__main__':
    main()
