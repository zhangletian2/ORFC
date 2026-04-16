#!/usr/bin/env python
"""
Figure 4 — Rate-Task: Ours vs baseline (VTM or OPQ).

DINOv2 ViT-L/14, selectable blocks, BPFP < 0.4.
Solid lines for Acc, dashed for mIoU.

Usage:
    python plot_fig4.py                          # default: VTM, blk 5 10
    python plot_fig4.py --baseline opq
    python plot_fig4.py --baseline vtm --blocks 5 10 15 20
    python plot_fig4.py --baseline opq --blocks 20
"""

import os, sys, argparse, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

COL_BASE = '#999999'
COL_OURS = '#B983CD'

root = os.path.dirname(os.path.abspath(__file__))

BLOCK_COL_OFFSETS = {5: 0, 10: 3, 15: 6, 20: 9}
ACC0  = {5: 98.0, 10: 98.0, 15: 98.0, 20: 98.0}
MIOU0 = {5: 82.0, 10: 82.0, 15: 82.0, 20: 82.0}


def _parse_pct(s):
    """'93.60%' -> 93.6,  '0.0625' -> 0.0625"""
    s = s.strip()
    if s.endswith('%'):
        return float(s[:-1])
    return float(s)


def parse_csv(path):
    """Parse the multi-section CSV into {method: {block: ([bpfp], [metric])}}."""
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        rows = list(csv.reader(f))

    result = {}
    i = 0
    while i < len(rows):
        first = rows[i][0].strip() if rows[i] else ''
        if first in ('LaMoFC', 'VTM', 'OPQ', 'Ours'):
            method = first
            i += 2  # skip block-number row
            i += 1  # skip column-header row
            block_data = {b: ([], []) for b in BLOCK_COL_OFFSETS}
            while i < len(rows):
                r = rows[i]
                if not r or not r[0].strip():
                    if all(not c.strip() for c in r):
                        i += 1
                        continue
                is_next_section = (r[0].strip() in
                                   ('LaMoFC', 'VTM', 'OPQ', 'Ours'))
                if is_next_section:
                    break
                for blk, offset in BLOCK_COL_OFFSETS.items():
                    bpfp_col = offset + 1
                    metric_col = offset + 2
                    if metric_col < len(r) and r[bpfp_col].strip():
                        try:
                            bpfp = float(r[bpfp_col].strip())
                            metric = _parse_pct(r[metric_col])
                            block_data[blk][0].append(bpfp)
                            block_data[blk][1].append(metric)
                        except (ValueError, IndexError):
                            pass
                i += 1
            result[method] = block_data
        else:
            i += 1
    return result


def plot_block(blk_id, base_acc, base_seg, ours_acc, ours_seg,
               baseline_label, out_suffix, bpfp_max=0.4):
    br, bp = base_acc
    br, bp = zip(*[(r, p) for r, p in zip(br, bp) if r < bpfp_max]) if br else ([], [])
    bsr, bsp = base_seg
    bsr, bsp = zip(*[(r, p) for r, p in zip(bsr, bsp) if r < bpfp_max]) if bsr else ([], [])

    or_, op = ours_acc
    or_, op = zip(*[(r, p) for r, p in zip(or_, op) if r < bpfp_max]) if or_ else ([], [])
    osr, osp = ours_seg
    osr, osp = zip(*[(r, p) for r, p in zip(osr, osp) if r < bpfp_max]) if osr else ([], [])

    fig, ax = plt.subplots(figsize=(3.2, 2.2))

    ax.plot(br, bp, '-s', color=COL_BASE, ms=4, lw=1.2,
            label=f'{baseline_label} Acc')
    ax.plot(bsr, bsp, '-D', color=COL_BASE, ms=3.5, lw=1.2,
            label=f'{baseline_label} mIoU')
    ax.plot(or_, op, '-^', color=COL_OURS, ms=4, lw=1.2,
            label='Ours Acc')
    ax.plot(osr, osp, '-o', color=COL_OURS, ms=3.5, lw=1.2,
            label='Ours mIoU')

    acc0 = ACC0.get(blk_id, 98.0)
    miou0 = MIOU0.get(blk_id, 82.0)
    ax.axhline(acc0, color='#ccc', ls=':', lw=0.6)
    ax.axhline(miou0, color='#ccc', ls=':', lw=0.6)
    xmax = bpfp_max - 0.04
    ax.text(xmax, acc0 + 1, r'Acc$_0$', fontsize=6, color='#999', va='bottom')
    ax.text(xmax, miou0 + 1, r'mIoU$_0$', fontsize=6, color='#999', va='bottom')

    ax.set_xlabel('BPFP')
    ax.set_ylabel('Task performance')
    ax.set_xlim(-0.01, bpfp_max)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v)}%'))
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    ax.legend(fontsize=6.5, loc='lower right', framealpha=0.9,
              borderpad=0.4, handlelength=1.5, ncol=2, columnspacing=1.0)

    if len(br) > 0:
        ax.annotate('Low-bitrate\ncollapse',
                    xy=(br[0], bp[0] + 12),
                    xytext=(0.0, 50),
                    fontsize=7, color='#444',
                    arrowprops=dict(arrowstyle='->', color='#444', lw=1.0),
                    va='center', ha='left')

    fig.tight_layout()

    fig_dir = os.path.join(root, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(fig_dir, f'fig4_rd_{out_suffix}.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {fig_dir}/fig4_rd_{out_suffix}.[pdf|png]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', choices=['vtm', 'opq'], default='vtm')
    parser.add_argument('--blocks', type=int, nargs='+', default=[5, 10])
    parser.add_argument('--bpfp_max', type=float, default=0.4)
    parser.add_argument('--cls_csv', default=os.path.join(
        root, 'results', 'dinov2_vitl14_cls.csv'))
    parser.add_argument('--seg_csv', default=os.path.join(
        root, 'results', 'dinov2_vitl14_seg.csv'))
    args = parser.parse_args()

    cls_data = parse_csv(args.cls_csv)
    seg_data = parse_csv(args.seg_csv)

    baseline_key = args.baseline.upper()
    baseline_label = baseline_key

    if baseline_key not in cls_data:
        print(f"ERROR: '{baseline_key}' not found in {args.cls_csv}")
        print(f"  Available sections: {list(cls_data.keys())}")
        sys.exit(1)

    for blk in args.blocks:
        print(f"\n=== Block {blk} — Ours vs {baseline_label} ===")
        base_acc = cls_data[baseline_key].get(blk, ([], []))
        base_seg = seg_data[baseline_key].get(blk, ([], []))
        ours_acc = cls_data['Ours'].get(blk, ([], []))
        ours_seg = seg_data['Ours'].get(blk, ([], []))

        suffix = f'blk{blk}_{args.baseline}'
        plot_block(blk, base_acc, base_seg, ours_acc, ours_seg,
                   baseline_label, suffix, bpfp_max=args.bpfp_max)


if __name__ == '__main__':
    main()
