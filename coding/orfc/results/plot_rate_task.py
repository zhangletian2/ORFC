#!/usr/bin/env python3
"""Plot 12 Rate-Task curves: 3 feature/task settings × 4 blocks.

Data has been cleaned:
  - Non-Pareto-optimal points removed (non-monotonic BPFP→perf)
  - Duplicate BPFP entries consolidated (keep best)
  - LaMoFC marked as 'failed' where max perf < ~5%
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')

os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────
font_size = 18
legend_size = 16
plt.rcParams.update({
    'font.size': font_size,
    'axes.labelsize': font_size,
    'axes.titlesize': font_size,
    'legend.fontsize': legend_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'font.family': 'sans-serif',
})
COL_OURS = '#E74C3C' # #8F5BB0 # #AD6CC4 # #E74C3C
COL_OPQ = '#286CA0' # #3498DB' # #286CA0'
COL_VTM = '#8F5BB0' # #EF937D' # #ED846B # #55A868
COL_LAMOFC = '#666'
STYLES = {
    'Ours': dict(color=COL_OURS, marker='o', ls='-',  lw=2.4, ms=7, zorder=10),
    'OPQ':         dict(color=COL_OPQ, marker='^', ls='--', lw=1.9, ms=6, zorder=5),
    'VTM':         dict(color=COL_VTM, marker='s', ls='-',  lw=1.9, ms=6, zorder=4),
    'LaMoFC':      dict(color=COL_LAMOFC, marker='D', ls=':',  lw=1.9, ms=6, zorder=3),
}
DRAW_ORDER = ['LaMoFC', 'VTM', 'OPQ', 'Ours']


# ── Plotting ─────────────────────────────────────────────────────
def plot_one(block_data, title, ylabel, save_path, xlim=1.0, is_seg=False, is_clip=False):
    fig, ax = plt.subplots(figsize=(7, 5))

    lamofc_failed = block_data.get('lamofc_failed', False)
    has_lamofc_curve = False

    for name in DRAW_ORDER:
        if name == 'LaMoFC' and lamofc_failed:
            continue

        bpfp, perf = block_data.get(name, ([], []))
        if not bpfp:
            continue

        if name == 'LaMoFC':
            has_lamofc_curve = True

        s = STYLES[name]
        ax.plot(bpfp, perf, label=name,
                color=s['color'], marker=s['marker'], linestyle=s['ls'],
                linewidth=s['lw'], markersize=s['ms'], zorder=s['zorder'],
                markeredgecolor='white', markeredgewidth=0.5,
                clip_on=True)

    ax.set_xlim(0, xlim)
    ax.set_ylim(0, 102)

    from matplotlib.ticker import FixedLocator, FixedFormatter
    xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    yticks = [0, 20, 40, 60, 80, 100]
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.xaxis.set_major_formatter(FixedFormatter(
        ['0'] + [f'{v:g}' for v in xticks[1:]]))
    ax.yaxis.set_major_formatter(FixedFormatter(
        [''] + [f'{int(v)}' for v in yticks[1:]]))

    ax.set_xlabel('BPFP')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')

    if is_seg or is_clip:
        leg1 = ax.legend(loc='center', bbox_to_anchor=(0.5, 92.0 / 102.0),
                         ncol=4, framealpha=0.92, edgecolor='#cccccc',
                         fancybox=True, columnspacing=1.0, handletextpad=0.4,
                         fontsize=legend_size)
    else:
        leg1 = ax.legend(loc='center right', framealpha=0.92, edgecolor='#cccccc',
                         fancybox=True, fontsize=legend_size)

    if lamofc_failed:
        from matplotlib.lines import Line2D
        peak = block_data.get('lamofc_max', 0)
        handle = Line2D([0], [0], marker='D', color=COL_LAMOFC,
                        linestyle=':', linewidth=1.9, markersize=5)
        leg2 = ax.legend(handles=[handle], labels=[f'LaMoFC (peak {peak:.1f}%)'],
                         loc='lower right', framealpha=0.92, edgecolor='#cccccc',
                         fancybox=True, fontsize=legend_size)
        ax.add_artist(leg1)

    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=200)
    png_path = save_path.replace('.pdf', '.png')
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  -> {os.path.basename(save_path)}')


def plot_combined_cls():
    """Plot 3×3 combined figure for classification task."""
    from matplotlib.ticker import FixedLocator, FixedFormatter
    from matplotlib.lines import Line2D

    rows = [
        ('DINOv2 ViT-G/14', dinog_cls, [10, 20, 30]),
        ('DINOv2 ViT-L/14', dino_cls,  [10, 15, 20]),
        ('CLIP ViT-L/14',   clip_cls,  [10, 15, 20]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    yticks = [0, 20, 40, 60, 80, 100]

    for ri, (backbone, data_dict, blocks) in enumerate(rows):
        for ci, blk in enumerate(blocks):
            ax = axes[ri, ci]
            bd = data_dict[blk]
            failed = bd.get('lamofc_failed', False)

            for name in DRAW_ORDER:
                if name == 'LaMoFC' and failed:
                    continue
                bpfp, perf = bd.get(name, ([], []))
                if not bpfp:
                    continue
                s = STYLES[name]
                ax.plot(bpfp, perf,
                        color=s['color'], marker=s['marker'], linestyle=s['ls'],
                        linewidth=s['lw'], markersize=s['ms'], zorder=s['zorder'],
                        markeredgecolor='white', markeredgewidth=0.5,
                        clip_on=True)

            ax.set_xlim(0, 1.0)
            ax.set_ylim(0, 102)
            ax.xaxis.set_major_locator(FixedLocator(xticks))
            ax.yaxis.set_major_locator(FixedLocator(yticks))
            ax.xaxis.set_major_formatter(FixedFormatter(
                ['0'] + [f'{v:g}' for v in xticks[1:]]))
            ax.yaxis.set_major_formatter(FixedFormatter(
                [''] + [f'{int(v)}' for v in yticks[1:]]))

            if ri == 2:
                ax.set_xlabel('BPFP')
            if ci == 0:
                ax.set_ylabel('Acc@1 (%)')

            if failed:
                peak = bd.get('lamofc_max', 0)
                ax.annotate(f'LaMoFC (peak {peak:.1f}%)',
                            xy=(0.97, 0.05), xycoords='axes fraction',
                            ha='right', va='bottom',
                            fontsize=legend_size - 2, color=COL_LAMOFC,
                            bbox=dict(boxstyle='round,pad=0.3',
                                      fc='white', ec='#cccccc', alpha=0.92))

            ax.set_title(f'{backbone}, Block {blk}', fontweight='bold')
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    handles = []
    for name in DRAW_ORDER:
        s = STYLES[name]
        handles.append(Line2D([0], [0], color=s['color'], marker=s['marker'],
                              linestyle=s['ls'], linewidth=s['lw'],
                              markersize=s['ms'],
                              markeredgecolor='white', markeredgewidth=0.5,
                              label=name))

    fig.legend(handles=handles, loc='upper center',
               ncol=len(DRAW_ORDER), framealpha=0.92, edgecolor='#cccccc',
               fancybox=True, fontsize=legend_size, columnspacing=2.0,
               handletextpad=0.5, bbox_to_anchor=(0.5, 1.0))

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf_path = os.path.join(OUT_DIR, 'combined_cls.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', dpi=200)
    png_path = os.path.join(OUT_DIR, 'combined_cls.png')
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  -> {os.path.basename(pdf_path)}')
    print(f'  -> {os.path.basename(png_path)}')


def plot_combined_seg():
    """Plot 2×3 combined figure for segmentation task."""
    from matplotlib.ticker import FixedLocator, FixedFormatter
    from matplotlib.lines import Line2D

    rows = [
        ('DINOv2 ViT-G/14', dinog_seg, [10, 20, 30]),
        ('DINOv2 ViT-L/14', dino_seg,  [10, 15, 20]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))

    xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    yticks = [0, 20, 40, 60, 80, 100]

    for ri, (backbone, data_dict, blocks) in enumerate(rows):
        for ci, blk in enumerate(blocks):
            ax = axes[ri, ci]
            bd = data_dict[blk]
            failed = bd.get('lamofc_failed', False)

            for name in DRAW_ORDER:
                if name == 'LaMoFC' and failed:
                    continue
                bpfp, perf = bd.get(name, ([], []))
                if not bpfp:
                    continue
                s = STYLES[name]
                ax.plot(bpfp, perf,
                        color=s['color'], marker=s['marker'], linestyle=s['ls'],
                        linewidth=s['lw'], markersize=s['ms'], zorder=s['zorder'],
                        markeredgecolor='white', markeredgewidth=0.5,
                        clip_on=True)

            ax.set_xlim(0, 1.0)
            ax.set_ylim(0, 102)
            ax.xaxis.set_major_locator(FixedLocator(xticks))
            ax.yaxis.set_major_locator(FixedLocator(yticks))
            ax.xaxis.set_major_formatter(FixedFormatter(
                ['0'] + [f'{v:g}' for v in xticks[1:]]))
            ax.yaxis.set_major_formatter(FixedFormatter(
                [''] + [f'{int(v)}' for v in yticks[1:]]))

            if ri == 1:
                ax.set_xlabel('BPFP')
            if ci == 0:
                ax.set_ylabel('mIoU (%)')

            if failed:
                peak = bd.get('lamofc_max', 0)
                ax.annotate(f'LaMoFC (peak {peak:.1f}%)',
                            xy=(0.97, 0.05), xycoords='axes fraction',
                            ha='right', va='bottom',
                            fontsize=legend_size - 2, color=COL_LAMOFC,
                            bbox=dict(boxstyle='round,pad=0.3',
                                      fc='white', ec='#cccccc', alpha=0.92))

            ax.set_title(f'{backbone}, Block {blk}', fontweight='bold')
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    handles = []
    for name in DRAW_ORDER:
        s = STYLES[name]
        handles.append(Line2D([0], [0], color=s['color'], marker=s['marker'],
                              linestyle=s['ls'], linewidth=s['lw'],
                              markersize=s['ms'],
                              markeredgecolor='white', markeredgewidth=0.5,
                              label=name))

    fig.legend(handles=handles, loc='upper center',
               ncol=len(DRAW_ORDER), framealpha=0.92, edgecolor='#cccccc',
               fancybox=True, fontsize=legend_size, columnspacing=2.0,
               handletextpad=0.5, bbox_to_anchor=(0.5, 1.0))

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    pdf_path = os.path.join(OUT_DIR, 'combined_seg.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', dpi=200)
    png_path = os.path.join(OUT_DIR, 'combined_seg.png')
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  -> {os.path.basename(pdf_path)}')
    print(f'  -> {os.path.basename(png_path)}')


def plot_combined_det():
    """Plot 1×3 combined figure for detection task."""
    from matplotlib.ticker import FixedLocator, FixedFormatter
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    yticks = [0, 20, 40, 60, 80, 100]

    for ci, blk in enumerate([3, 6, 9]):
        ax = axes[ci]
        bd = vitdet_det[blk]
        failed = bd.get('lamofc_failed', False)

        for name in DRAW_ORDER:
            if name == 'LaMoFC' and failed:
                continue
            bpfp, perf = bd.get(name, ([], []))
            if not bpfp:
                continue
            s = STYLES[name]
            ax.plot(bpfp, perf,
                    color=s['color'], marker=s['marker'], linestyle=s['ls'],
                    linewidth=s['lw'], markersize=s['ms'], zorder=s['zorder'],
                    markeredgecolor='white', markeredgewidth=0.5,
                    clip_on=True)

        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 102)
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.xaxis.set_major_formatter(FixedFormatter(
            ['0'] + [f'{v:g}' for v in xticks[1:]]))
        ax.yaxis.set_major_formatter(FixedFormatter(
            [''] + [f'{int(v)}' for v in yticks[1:]]))

        ax.set_xlabel('BPFP')
        if ci == 0:
            ax.set_ylabel('mAP@0.5:0.95 (%)')

        if failed:
            peak = bd.get('lamofc_max', 0)
            ax.annotate(f'LaMoFC (peak {peak:.1f}%)',
                        xy=(0.97, 0.05), xycoords='axes fraction',
                        ha='right', va='bottom',
                        fontsize=legend_size - 2, color=COL_LAMOFC,
                        bbox=dict(boxstyle='round,pad=0.3',
                                  fc='white', ec='#cccccc', alpha=0.92))

        ax.set_title(f'Block {blk}', fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    handles = []
    for name in DRAW_ORDER:
        s = STYLES[name]
        handles.append(Line2D([0], [0], color=s['color'], marker=s['marker'],
                              linestyle=s['ls'], linewidth=s['lw'],
                              markersize=s['ms'],
                              markeredgecolor='white', markeredgewidth=0.5,
                              label=name))

    fig.legend(handles=handles, loc='upper center',
               ncol=len(DRAW_ORDER), framealpha=0.92, edgecolor='#cccccc',
               fancybox=True, fontsize=legend_size, columnspacing=2.0,
               handletextpad=0.5, bbox_to_anchor=(0.5, 1.0))

    fig.tight_layout(rect=[0, 0, 1, 0.88])

    pdf_path = os.path.join(OUT_DIR, 'combined_det.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', dpi=200)
    png_path = os.path.join(OUT_DIR, 'combined_det.png')
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  -> {os.path.basename(pdf_path)}')
    print(f'  -> {os.path.basename(png_path)}')


def plot_combined_ret():
    """Plot 1×3 combined figure for retrieval task."""
    from matplotlib.ticker import FixedLocator, FixedFormatter
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    yticks = [0, 20, 40, 60, 80, 100]

    for ci, blk in enumerate([8, 16, 24]):
        ax = axes[ci]
        bd = siglip_ret[blk]
        failed = bd.get('lamofc_failed', False)
        vtm_failed = bd.get('vtm_failed', False)

        for name in DRAW_ORDER:
            if name == 'LaMoFC' and failed:
                continue
            if name == 'VTM' and vtm_failed:
                continue
            bpfp, perf = bd.get(name, ([], []))
            if not bpfp:
                continue
            s = STYLES[name]
            ax.plot(bpfp, perf,
                    color=s['color'], marker=s['marker'], linestyle=s['ls'],
                    linewidth=s['lw'], markersize=s['ms'], zorder=s['zorder'],
                    markeredgecolor='white', markeredgewidth=0.5,
                    clip_on=True)

        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 102)
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.xaxis.set_major_formatter(FixedFormatter(
            ['0'] + [f'{v:g}' for v in xticks[1:]]))
        ax.yaxis.set_major_formatter(FixedFormatter(
            [''] + [f'{int(v)}' for v in yticks[1:]]))

        ax.set_xlabel('BPFP')
        if ci == 0:
            ax.set_ylabel('Recall@1 (%)')

        if failed:
            peak = bd.get('lamofc_max', 0)
            ax.annotate(f'LaMoFC (peak {peak:.1f}%)',
                        xy=(0.97, 0.05), xycoords='axes fraction',
                        ha='right', va='bottom',
                        fontsize=legend_size - 2, color=COL_LAMOFC,
                        bbox=dict(boxstyle='round,pad=0.3',
                                  fc='white', ec='#cccccc', alpha=0.92))

        if vtm_failed:
            peak = bd.get('vtm_max', 0)
            ax.annotate(f'VTM (peak {peak:.1f}%)',
                        xy=(0.97, 0.50), xycoords='axes fraction',
                        ha='right', va='center',
                        fontsize=legend_size - 2, color=COL_VTM,
                        bbox=dict(boxstyle='round,pad=0.3',
                                  fc='white', ec='#cccccc', alpha=0.92))

        ax.set_title(f'Block {blk}', fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    handles = []
    for name in DRAW_ORDER:
        s = STYLES[name]
        handles.append(Line2D([0], [0], color=s['color'], marker=s['marker'],
                              linestyle=s['ls'], linewidth=s['lw'],
                              markersize=s['ms'],
                              markeredgecolor='white', markeredgewidth=0.5,
                              label=name))

    fig.legend(handles=handles, loc='upper center',
               ncol=len(DRAW_ORDER), framealpha=0.92, edgecolor='#cccccc',
               fancybox=True, fontsize=legend_size, columnspacing=2.0,
               handletextpad=0.5, bbox_to_anchor=(0.5, 1.0))

    fig.tight_layout(rect=[0, 0, 1, 0.88])

    pdf_path = os.path.join(OUT_DIR, 'combined_ret.pdf')
    fig.savefig(pdf_path, bbox_inches='tight', dpi=200)
    png_path = os.path.join(OUT_DIR, 'combined_ret.png')
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  -> {os.path.basename(pdf_path)}')
    print(f'  -> {os.path.basename(png_path)}')


# ================================================================
#  DATA — CLIP ViT-L/14  Classification  (Acc@1 %)
# ================================================================
clip_cls = {}

# ---- Block 5 ----
clip_cls[5] = {
    'lamofc_failed': True,
    'lamofc_max': 12.8,
    'VTM':         ([0.013, 0.178, 0.328, 0.977, 1.3822],
                    [0.20, 31.20, 56.60, 77.40, 79.20]),
    'OPQ':         ([0.0625, 0.0938, 0.125, 0.1875, 0.25, 0.375, 0.5],
                    [15.00, 54.80, 65.20, 73.40, 76.00, 78.60, 80.20]),
    'Ours': ([0.059, 0.091, 0.122, 0.246, 0.368, 0.492],
                    [72.60, 77.60, 79.80, 80.20, 80.60, 81.20]),
}

# ---- Block 10 ----
clip_cls[10] = {
    'LaMoFC':      ([0.2029, 0.4265, 0.5613, 1.3437],
                    [0.20, 0.21, 3.27, 24.17]),
    'VTM':         ([0.0196, 0.0351, 0.0841, 0.1481, 0.4301, 1.4415],
                    [0.00, 0.60, 9.60, 31.00, 75.20, 79.40]),
    'OPQ':         ([0.0625, 0.0938, 0.125, 0.1875, 0.25, 0.375, 0.5],
                    [2.00, 45.60, 62.00, 72.20, 77.40, 78.00, 78.40]),
    'Ours': ([0.06, 0.091, 0.122, 0.183, 0.47],
                    [71.80, 77.00, 77.40, 79.40, 79.60]),
}

# ---- Block 15 ----
clip_cls[15] = {
    'lamofc_failed': True,
    'lamofc_max': 2.8,
    'VTM':         ([0.006, 0.0214, 0.037, 0.0847, 0.1563, 0.4596, 0.8256],
                    [0.00, 1.00, 7.60, 34.00, 51.60, 71.00, 76.00]),
    'OPQ':         ([0.0625, 0.0938, 0.125, 0.1875, 0.2813, 0.5, 0.5625],
                    [20.20, 40.60, 50.40, 64.00, 72.60, 77.20, 79.80]),
    'Ours': ([0.059, 0.091, 0.12, 0.17, 0.193, 0.28, 0.494],
                    [63.40, 69.20, 73.20, 73.40, 74.80, 75.60, 77.80]),
}

# ---- Block 20 ----
clip_cls[20] = {
    'LaMoFC':      ([0.1913, 0.6627, 0.8825, 1.1091],
                    [0.00, 1.58, 13.84, 15.31]),
    'VTM':         ([0.00863, 0.02872, 0.1219, 0.251, 0.6935, 1.0987],
                    [0.60, 12.40, 40.80, 62.40, 78.00, 81.80]),
    'OPQ':         ([0.0625, 0.0938, 0.125, 0.1875, 0.2813, 0.5625, 1.0],
                    [12.20, 24.60, 36.00, 53.60, 63.40, 74.20, 78.40]),
    'Ours': ([0.06, 0.091, 0.12, 0.177, 0.25, 0.45],
                    [32.20, 46.00, 55.40, 65.80, 70.40, 75.20]),
}
    # 'Ours': ([0.06, 0.091, 0.12, 0.177, 0.25, 0.45, 0.648, 0.996],
    #                 [32.20, 46.00, 55.40, 65.80, 70.40, 75.20, 75.60, 78.80]),


# ================================================================
#  DATA — DINOv2 ViT-L/14  Classification  (Acc@1 %)
# ================================================================
dino_cls = {}

# ---- Block 5 ----
dino_cls[5] = {
    'lamofc_failed': True,
    'lamofc_max': 0.2,
    'VTM':         ([0.056, 0.139, 0.24, 0.533, 0.852, 1.407],
                    [0.20, 18.20, 65.00, 91.00, 96.40, 98.20]),
    'OPQ':         ([0.0625, 0.0938, 0.125, 0.1875, 0.25, 0.375, 0.5],
                    [2.00, 22.40, 60.00, 87.00, 90.00, 93.60, 96.60]),
    'Ours': ([0.0569, 0.087, 0.1179, 0.1796, 0.2361, 0.3534, 0.4607],
                    [93.80, 95.00, 96.00, 96.80, 97.00, 97.40, 97.80]),
}

# ---- Block 10 ----
dino_cls[10] = {
    'lamofc_failed': True,
    'lamofc_max': 6.2,
    'VTM':         ([0.04327, 0.10489, 0.19656, 0.50256, 0.89639, 1.51030],
                    [0.20, 19.80, 73.20, 95.40, 97.40, 98.00]),
    'OPQ':         ([0.0625, 0.0938, 0.125, 0.1875, 0.25, 0.375, 0.5],
                    [34.20, 78.00, 91.00, 95.80, 97.00, 97.80, 98.20]),
    'Ours': ([0.0602, 0.0913, 0.1228, 0.1849, 0.246],
                    [96.00, 97.00, 97.80, 98.00, 98.00]),
}

# ---- Block 15 ----
dino_cls[15] = {
    'lamofc_failed': True,
    'lamofc_max': 0.5,
    'VTM':         ([0.04066, 0.10042, 0.20325, 0.58657, 1.00030, 1.57484],
                    [0.20, 8.00, 71.60, 96.60, 97.00, 98.20]),
    'OPQ':         ([0.0625, 0.0938, 0.125, 0.1875, 0.25, 0.375, 0.5],
                    [7.60, 45.20, 73.60, 91.00, 94.80, 95.80, 97.60]),
    'Ours': ([0.061, 0.0922, 0.123, 0.1849, 0.2471, 0.3707, 0.4476, 0.488],
                    [93.80, 95.00, 96.00, 96.80, 97.00, 97.40, 97.60, 97.80]),
}

# ---- Block 20 ----
dino_cls[20] = {
    'LaMoFC':      ([0.2131, 0.3497, 0.5675, 0.6658, 1.1238],
                    [0.00, 4.66, 10.32, 10.93, 11.66]),
    'VTM':         ([0.00521, 0.02611, 0.04605, 0.11007, 0.22129, 0.61696, 1.02814],
                    [0.00, 7.40, 19.40, 48.40, 73.60, 91.80, 95.60]),
    'OPQ':         ([0.0625, 0.0938, 0.125, 0.1563, 0.1875, 0.25, 0.5],
                    [3.60, 34.40, 54.60, 69.40, 78.80, 87.20, 93.40]),
    'Ours': ([0.0613, 0.0928, 0.1239, 0.1538, 0.1842, 0.2469, 0.41, 0.4883],
                    [82.60, 87.00, 92.40, 93.00, 94.00, 95.60, 96.20, 97.40]),
}


# ================================================================
#  DATA — DINOv2 ViT-L/14  Segmentation  (mIoU %)
# ================================================================
dino_seg = {}

# ---- Block 5 ----
dino_seg[5] = {
    'LaMoFC':      ([0.0615, 0.0808, 0.1676, 0.2979, 0.4368, 0.7517],
                    [1.85, 3.07, 4.25, 9.39, 19.27, 54.44]),
    'VTM':         ([0.003794, 0.034781, 0.081366, 0.140561, 0.291468, 0.452644, 0.829022],
                    [2.93, 3.43, 19.50, 60.71, 78.67, 80.97, 81.96]),
    'OPQ':         ([0.0625, 0.0938, 0.125, 0.1875, 0.25, 0.375, 0.5],
                    [3.36, 36.34, 72.70, 79.59, 81.22, 81.60, 81.85]),
    'Ours': ([0.0565, 0.0864, 0.1172, 0.1788, 0.2339, 0.3497],
                    [78.30, 78.90, 79.40, 79.70, 80.00, 80.40]),
}

# ---- Block 10 ----
dino_seg[10] = {
    'LaMoFC':      ([0.0098, 0.2896, 0.4983, 0.9582, 1.2669, 1.4653],
                    [3.36, 19.80, 46.70, 69.33, 74.48, 81.41]),
    'VTM':         ([0.00257, 0.066635, 0.122617, 0.28553, 0.482946],
                    [3.66, 21.93, 55.88, 79.61, 81.87]),
    'OPQ':         ([0.0625, 0.0938, 0.125, 0.1875, 0.25, 0.375, 0.5],
                    [47.77, 71.88, 78.12, 80.19, 80.55, 80.91, 81.15]),
    'Ours': ([0.0597, 0.091, 0.1803, 0.2389, 0.3678],
                    [78.40, 80.10, 80.40, 80.90, 81.20]),
}

# ---- Block 15 ----
dino_seg[15] = {
    'LaMoFC':      ([0.0083, 0.3209, 0.6231, 1.0883],
                    [2.50, 19.45, 27.47, 72.43]),
    'VTM':         ([0.001747, 0.023936, 0.074254, 0.149925, 0.350076, 0.59617, 1.128721],
                    [3.11, 3.38, 19.13, 61.37, 79.69, 81.52, 81.75]),
    'OPQ':         ([0.0625, 0.0938, 0.125, 0.1875, 0.25, 0.375, 0.5],
                    [8.05, 36.12, 60.85, 76.73, 78.88, 79.14, 79.88]),
    'Ours': ([0.0607, 0.092, 0.1227, 0.1849, 0.2472, 0.3708],
                    [78.30, 78.90, 79.40, 79.70, 80.00, 80.40]),
}

# ---- Block 20 ----
dino_seg[20] = {
    'LaMoFC':      ([0.0677, 0.2196, 0.9477],
                    [6.84, 33.05, 50.13]),
    'VTM':         ([0.000532, 0.001061, 0.013066, 0.027212, 0.091487, 0.180139,
                     0.470134, 0.795831],
                    [0.23, 2.80, 15.46, 28.83, 55.50, 66.77, 75.98, 79.36]),
    'OPQ':         ([0.0625, 0.0938, 0.125, 0.1563, 0.1875, 0.25, 0.5],
                    [8.33, 24.96, 38.72, 49.34, 57.85, 63.88, 68.86]),
    'Ours': ([0.0596, 0.0942, 0.1244, 0.177, 0.4083, 0.485],
                    [68.31, 68.60, 71.00, 72.80, 74.60, 75.50]),
}


# ================================================================
#  DATA — DINOv2 ViT-G/14  Classification  (Acc@1 %)
# ================================================================
dinog_cls = {}

# ---- Block 9 ----
dinog_cls[10] = {
    'lamofc_failed': True,
    'lamofc_max': 0.2,
    'VTM':  ([0.02, 0.038, 0.099, 0.182, 0.477, 1.435, 2.339],
             [0.00, 6.80, 65.40, 85.60, 96.80, 99.60, 99.80]),
    'OPQ':  ([0.0625, 0.0938, 0.125, 0.1875, 0.25, 0.375],
             [89.60, 93.80, 95.80, 97.80, 98.20, 99.20]),
    'Ours': ([0.0603, 0.0902, 0.1207, 0.1808, 0.239],
             [97.40, 97.80, 99.00, 99.20, 99.20]),
}

# ---- Block 19 ----
dinog_cls[20] = {
    'lamofc_failed': True,
    'lamofc_max': 0.23,
    'VTM':  ([0.003, 0.058, 0.111, 0.329, 0.663, 1.287],
             [0.00, 11.80, 54.80, 94.80, 97.60, 98.80]),
    'OPQ':  ([0.0625, 0.0938, 0.125, 0.1875, 0.25, 0.375],
             [90.20, 95.80, 97.00, 98.80, 99.20, 99.80]),
    'Ours': ([0.0605, 0.092, 0.1231, 0.1849, 0.3585],
             [97.60, 98.80, 99.40, 99.40, 99.60]),
}

# ---- Block 29 ----
dinog_cls[30] = {
    'LaMoFC': ([0.1398, 0.8735, 0.9617, 0.9769, 1.5409],
               [0.00, 4.98, 21.00, 28.85, 47.90]),
    'VTM':  ([0.004, 0.021, 0.041, 0.115, 0.241, 0.674, 1.122],
             [0.20, 10.60, 24.60, 60.40, 81.00, 96.60, 98.20]),
    'OPQ':  ([0.0625, 0.0938, 0.125, 0.1875, 0.25, 0.375],
             [28.80, 81.40, 91.00, 95.80, 97.40, 97.60]),
    'Ours': ([0.0616, 0.0929, 0.1241, 0.1853, 0.2462],
             [95.60, 96.20, 97.80, 98.00, 98.80]),
}


# ================================================================
#  DATA — DINOv2 ViT-G/14  Segmentation  (mIoU %)
# ================================================================
dinog_seg = {}

# ---- Block 9 ----
dinog_seg[10] = {
    'LaMoFC': ([0.0001, 0.1677, 0.2549, 0.5541, 0.9746, 1.234, 2.6542],
               [3.36, 3.43, 11.66, 34.18, 65.27, 65.96, 84.31]),
    'VTM':  ([0.0131, 0.0248, 0.0616, 0.1111, 0.2627, 0.9702, 1.9135],
             [4.72, 9.49, 48.23, 70.32, 81.59, 83.21, 83.39]),
    'OPQ':  ([0.0625, 0.0938, 0.125, 0.1875, 0.25, 0.375],
             [78.55, 81.24, 82.09, 82.25, 82.54, 82.60]),
    'Ours': ([0.0599, 0.09, 0.1206, 0.1798],
             [82.22, 82.92, 83.06, 83.25]),
}

# ---- Block 19 ----
dinog_seg[20] = {
    'LaMoFC': ([0.0001, 0.1513, 0.3467, 0.6095, 1.2178],
               [2.60, 26.93, 48.89, 66.42, 80.12]),
    'VTM':  ([0.0012, 0.0403, 0.0855, 0.2363, 0.4361, 0.9417],
             [3.28, 38.21, 65.45, 81.68, 83.23, 83.54]),
    'OPQ':  ([0.0625, 0.0938, 0.125, 0.1875, 0.25, 0.375],
             [79.42, 81.83, 82.20, 82.60, 82.71, 82.78]),
    'Ours': ([0.0602, 0.0918, 0.2448],
             [83.13, 83.43, 83.48]),
}

# ---- Block 29 ----
dinog_seg[30] = {
    'LaMoFC': ([0.0249, 0.243, 0.2643, 0.4819, 1.1004, 1.4536],
               [3.19, 47.33, 51.16, 55.88, 71.31, 76.34]),
    'VTM':  ([0.0013, 0.0167, 0.0362, 0.1226, 0.2356, 0.5806, 0.9563],
             [2.97, 35.90, 52.91, 72.52, 78.45, 82.31, 83.16]),
    'OPQ':  ([0.0625, 0.0938, 0.125, 0.1875, 0.25, 0.375],
             [24.96, 57.45, 72.14, 77.17, 80.50, 80.92]),
    'Ours': ([0.0616, 0.0928, 0.1242, 0.184, 0.2442],
             [79.75, 80.80, 81.38, 81.93, 82.50]),
}


# ================================================================
#  DATA — ViTDet ViT-B/16  Detection  (bbox mAP %)
# ================================================================
vitdet_det = {}

# ---- Block 3 ----
vitdet_det[3] = {
    'VTM':         ([0.0051, 0.014, 0.0511, 0.1588, 0.4531, 1.0436],
                    [2.3, 19.4, 49.5, 76.1, 92.8, 99.2]),
    'LaMoFC':      ([0.0417, 0.1512, 0.2944, 0.4717, 0.6443, 1.0056],
                    [5.7, 29.0, 58.9, 78.0, 89.4, 94.5]),
    'OPQ':         ([0.0625, 0.0938, 0.125, 0.1875, 0.375, 0.5],
                    [36.7, 65.9, 81.7, 88.7, 93.7, 95.6]),
    'Ours':        ([0.0521, 0.1138, 0.1761, 0.3499, 0.4508],
                    [91.6, 94.7, 94.7, 95.0, 96.2]),
}

# ---- Block 6 ----
vitdet_det[6] = {
    'VTM':         ([0.0069, 0.018, 0.0622, 0.2155, 0.6201, 1.2497],
                    [3.5, 10.4, 39.2, 75.8, 93.1, 98.7]),
    'LaMoFC':      ([0.0622, 0.2099, 0.4073, 0.5753, 0.7553, 1.2079],
                    [13.6, 53.1, 79.8, 89.5, 92.3, 97.3]),
    'OPQ':         ([0.0625, 0.0938, 0.125, 0.1875, 0.375, 0.5],
                    [13.9, 34.0, 64.6, 77.7, 90.5, 95.0]),
    'Ours':        ([0.0574, 0.088, 0.1805, 0.3652, 0.4778],
                    [88.5, 92.3, 94.0, 95.1, 95.5]),
}

# ---- Block 9 ----
vitdet_det[9] = {
    'VTM':         ([0.0118, 0.0343, 0.117, 0.3771, 0.8865, 1.5212],
                    [24.0, 55.0, 79.9, 92.2, 97.5, 99.2]),
    'LaMoFC':      ([0.0917, 0.152, 0.5716, 0.8216, 0.9902, 1.1702],
                    [68.3, 75.3, 93.6, 95.7, 96.5, 97.2]),
    'OPQ':         ([0.0625, 0.0938, 0.125, 0.1875, 0.375, 0.5],
                    [1.2, 4.6, 13.4, 54.0, 72.7, 83.0]),
    'Ours':        ([0.0599, 0.0891, 0.1803, 0.3642, 0.4818],
                    [79.7, 85.0, 90.9, 93.3, 95.1]),
}


# ================================================================
#  DATA — SigLIP2 SO400M/14  Retrieval  (avg Recall@1 %)
# ================================================================
siglip_ret = {}

# ---- Block 8 ----
siglip_ret[8] = {
    'VTM':         ([0.0508, 0.0857, 0.1779, 0.3162, 0.8616],
                    [0.50, 2.10, 13.36, 51.70, 80.73]),
    'LaMoFC':      ([0.084, 0.6242, 1.0055, 1.2062, 1.4108],
                    [33.05, 41.65, 74.87, 75.23, 76.53]),
    'OPQ':         ([0.0625, 0.0938, 0.125, 0.1875, 0.375, 0.5],
                    [12.96, 46.50, 64.96, 77.77, 80.25, 81.15]),
    'Ours':        ([0.0597, 0.0908, 0.1218, 0.1839],
                    [78.39, 80.53, 80.57, 81.01]),
}

# ---- Block 16 ----
siglip_ret[16] = {
    'vtm_failed': True,
    'vtm_max': 2.78,
    'LaMoFC':      ([0.1279, 0.2196, 1.4024],
                    [0.64, 14.10, 16.96]),
    'OPQ':         ([0.0625, 0.0938, 0.125, 0.1875, 0.375, 0.5],
                    [26.27, 43.40, 54.98, 71.29, 76.83, 78.95]),
    'Ours':        ([0.0599, 0.0919, 0.1226, 0.1835, 0.3696, 0.4855],
                    [76.13, 77.77, 78.93, 79.21, 79.93, 79.97]),
}

# ---- Block 24 ----
siglip_ret[24] = {
    'VTM':         ([0.0069, 0.0147, 0.0212, 0.0396, 0.0664],
                    [0.24, 0.74, 3.30, 19.68, 36.75]),
    'LaMoFC':      ([0.1055, 0.2293, 0.3768, 0.9061, 1.1622, 1.8605],
                    [3.80, 37.47, 47.69, 69.60, 72.76, 76.73]),
    'OPQ':         ([0.0625, 0.0938, 0.125, 0.1875, 0.375, 0.5],
                    [30.97, 52.54, 63.64, 71.49, 75.99, 78.95]),
    'Ours':        ([0.0604, 0.0915, 0.1224, 0.1816, 0.3718, 0.4956],
                    [70.89, 76.03, 76.73, 78.99, 79.71, 80.15]),
}


# ================================================================
#  Generate all plots
# ================================================================
CONFIGS = [
    ('clip_vitl14_cls',   clip_cls,   'CLIP ViT-L/14 Classification',   'Acc@1 (%)', [5, 10, 15, 20]),
    ('dinov2_vitl14_cls', dino_cls,   'DINOv2 ViT-L/14 Classification', 'Acc@1 (%)', [5, 10, 15, 20]),
    ('dinov2_vitl14_seg', dino_seg,   'DINOv2 ViT-L/14 Segmentation',   'mIoU (%)',  [5, 10, 15, 20]),
    ('dinov2_vitg14_cls', dinog_cls,  'DINOv2 ViT-G/14 Classification', 'Acc@1 (%)', [10, 20, 30]),
    ('dinov2_vitg14_seg', dinog_seg,  'DINOv2 ViT-G/14 Segmentation',   'mIoU (%)',  [10, 20, 30]),
]

print(f'Generating Rate-Task curves ...')
for csv_key, data_dict, title_prefix, ylabel, blocks in CONFIGS:
    print(f'\n[{title_prefix}]')
    for block in blocks:
        block_data = data_dict[block]
        title = f'Block {block}'

        fname = f'{csv_key}_blk{block:02d}.pdf'
        plot_one(block_data, title, ylabel, os.path.join(OUT_DIR, fname),
                 is_seg=('seg' in csv_key),
                 is_clip=csv_key.startswith('clip'))

print(f'\n[Combined Classification]')
plot_combined_cls()

print(f'\n[Combined Segmentation]')
plot_combined_seg()

print(f'\n[Combined Detection]')
plot_combined_det()

print(f'\n[Combined Retrieval]')
plot_combined_ret()

print(f'\nDone!  All figures saved to {OUT_DIR}/')
