#!/usr/bin/env python
"""
Exp 2 — MSE vs L_ref Correlation Visualization

Reads correlation JSONs from results/analysis_intro_v2/ and
result JSONs from results/soft_pq/ to produce:
  (a) Scatter plots: MSE vs Acc and L_ref vs Acc (from correlation JSONs)
  (b) Correlation heatmap: Pearson / Spearman across layers and tasks
  (c) Summary table with p-values

Data sources:
  - results/analysis_intro_v2/correlation_*.json  (OPQ sweep with MSE+L_ref)
  - results/soft_pq/*/blk*_*.json  (DOPQ with L_ref only, for enrichment)
"""

import json, os, glob
import numpy as np
from scipy import stats as sp_stats
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
CORR_DIR = os.path.join(ROOT, 'results', 'analysis_intro_v2')
FIG_DIR = os.path.join(ROOT, 'figures')

C_MSE = '#286CA0'
C_LREF = '#B983CD'


def _pretty_title(backbone, layer):
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


def _backbone_from_filename(path):
    """Infer backbone from filename like correlation_dinov2_vitg14_blk09.json."""
    base = os.path.basename(path).replace('correlation_', '').replace('.json', '')
    for bb in ('dinov2_vitg14', 'dinov2_vitl14', 'clip_vitl14'):
        if base.startswith(bb):
            return bb
    return None


def load_correlation_jsons():
    """Load all correlation_*.json files."""
    records = {}
    for f in sorted(glob.glob(os.path.join(CORR_DIR, 'correlation_*.json'))):
        with open(f) as fh:
            d = json.load(fh)
        layer = d['config']['layer']
        backbone = d['config'].get('backbone')
        if not backbone:
            backbone = _backbone_from_filename(f) or 'dinov2_vitl14'
        key = (backbone, layer)
        records[key] = d
    return records


def load_dopq_correlations():
    """Load DOPQ result JSONs and extract (L_ref, Acc, mIoU) per
    (backbone, layer) as supplementary correlation data."""
    result_dirs = {
        'dinov2_vitl14': os.path.join(ROOT, 'results', 'soft_pq', 'dinov2_vitl14'),
        'dinov2_vitg14': os.path.join(ROOT, 'results', 'soft_pq', 'dinov2_vitg14'),
    }

    groups = {}
    for bb, jdir in result_dirs.items():
        if not os.path.isdir(jdir):
            continue
        for f in sorted(glob.glob(os.path.join(jdir, '*.json'))):
            with open(f) as fh:
                j = json.load(fh)
            cfg = j['config']
            layer = cfg['layer']
            if cfg.get('seed', 42) != 42:
                continue
            if cfg.get('mse_loss', False) or cfg.get('freeze_transform', False):
                continue

            key = (bb, layer)
            if key not in groups:
                groups[key] = []
            groups[key].append({
                'dl': j.get('soft_pq_delta_l', 0),
                'acc': j['soft_pq_acc'],
                'miou': j.get('soft_pq_miou'),
                'bpt': j['rate_info']['rans_bpt'],
                'K': cfg['K'], 'd': cfg['embedding_dim'],
            })

    return groups


def compute_correlations(x, y):
    """Compute Pearson and Spearman correlations with p-values."""
    x, y = np.asarray(x), np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {'pearson_r': np.nan, 'pearson_p': np.nan,
                'spearman_r': np.nan, 'spearman_p': np.nan, 'n': len(x)}
    pr, pp = sp_stats.pearsonr(x, y)
    sr, sp_ = sp_stats.spearmanr(x, y)
    return {'pearson_r': float(pr), 'pearson_p': float(pp),
            'spearman_r': float(sr), 'spearman_p': float(sp_), 'n': len(x)}


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {FIG_DIR}/{name}.[pdf|png]")


# ── (a) Scatter plots (one figure per block) ────────────────────────

def plot_scatter(corr_data):
    """One figure per (backbone, layer): overlaid normalised MSE and
    L_ref vs Acc drop, showing L_ref's superiority as predictor."""
    for (bb, layer), d in sorted(corr_data.items()):
        opq_pts = [p for p in d['points'] if p['method'] == 'opq']
        if len(opq_pts) < 3:
            continue

        all_mse = np.array([p['mse'] for p in opq_pts])
        all_lref = np.array([p['l_ref'] for p in opq_pts])
        all_acc = np.array([p['acc'] for p in opq_pts]) * 100

        acc_orig = all_acc.max()
        acc_drop = acc_orig - all_acc

        mse_rng = all_mse.max() - all_mse.min()
        lref_rng = all_lref.max() - all_lref.min()
        if mse_rng < 1e-12 or lref_rng < 1e-12:
            continue
        mse_norm = (all_mse - all_mse.min()) / mse_rng
        lref_norm = (all_lref - all_lref.min()) / lref_rng

        r_mse = compute_correlations(mse_norm, acc_drop)['pearson_r']
        r_lref = compute_correlations(lref_norm, acc_drop)['pearson_r']

        fig, ax = plt.subplots(figsize=(3.2, 2.4))

        ax.scatter(mse_norm, acc_drop, s=22, c=C_MSE, alpha=0.6,
                   edgecolors=C_MSE, linewidths=0.6, zorder=2,
                   label=f'MSE (r={r_mse:.2f})')
        ax.scatter(lref_norm, acc_drop, s=22, c=C_LREF, alpha=0.6,
                   edgecolors=C_LREF, linewidths=0.6, zorder=3,
                   label=rf'$\mathcal{{L}}_{{\rm ref}}$ (r={r_lref:.2f})')

        x_gt = np.linspace(0, 1, 50)
        y_max = acc_drop.max() if acc_drop.max() > 0 else 1.0
        ax.plot(x_gt, x_gt * y_max, ls='--', lw=1.0, color='#333',
                alpha=0.4, zorder=1, label='Ideal')

        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel('Normalised distortion')
        ax.set_ylabel('Acc drop (%)')
        ax.set_xlim(-0.05, 1.05)
        ax.set_title(_pretty_title(bb, layer), fontsize=9, pad=4)
        ax.legend(fontsize=6.5, loc='upper left', framealpha=1.0,
                  borderpad=0.3, handlelength=1.0)

        fig.tight_layout(pad=0.4)
        short_bb = bb.replace('dinov2_', 'D2_').replace('clip_', 'CL_')
        _save(fig, f'exp2_corr_scatter_{short_bb}_{layer}')


# ── (b) Correlation heatmap ──────────────────────────────────────────

def plot_heatmap(corr_data, dopq_data):
    """Heatmap: Pearson/Spearman of (MSE,Acc), (L_ref,Acc),
    (MSE,mIoU), (L_ref,mIoU) across layers."""
    metrics = [
        ('MSE', 'Acc', 'pearson'),
        (r'$L_{\rm ref}$', 'Acc', 'pearson'),
        ('MSE', 'mIoU', 'pearson'),
        (r'$L_{\rm ref}$', 'mIoU', 'pearson'),
        ('MSE', 'Acc', 'spearman'),
        (r'$L_{\rm ref}$', 'Acc', 'spearman'),
    ]

    layer_keys = sorted(set(list(corr_data.keys()) + list(dopq_data.keys())))
    if not layer_keys:
        print("  [skip] no data for heatmap")
        return

    all_corr_data = corr_data

    col_labels = [f'{m[2][:4]} r({m[0]},{m[1]})' for m in metrics]
    row_labels = [f'{bb.split("_")[0]} {layer}' for bb, layer in layer_keys]

    mat = np.full((len(layer_keys), len(metrics)), np.nan)

    for ri, key in enumerate(layer_keys):
        if key not in all_corr_data:
            continue
        opq_pts = [p for p in all_corr_data[key]['points']
                   if p['method'] == 'opq']
        if len(opq_pts) < 3:
            continue

        mse = np.array([p['mse'] for p in opq_pts])
        lref = np.array([p['l_ref'] for p in opq_pts])
        acc = np.array([p['acc'] for p in opq_pts])
        miou_vals = [p.get('miou') for p in opq_pts]
        has_miou = all(v is not None for v in miou_vals)
        miou = np.array(miou_vals) if has_miou else None

        for ci, (dist_name, task_name, corr_type) in enumerate(metrics):
            dist = mse if 'MSE' in dist_name else lref
            task = acc if task_name == 'Acc' else (miou if miou is not None
                                                    else None)
            if task is None:
                continue
            cr = compute_correlations(dist, task)
            key_name = f'{corr_type}_r'
            mat[ri, ci] = cr[key_name]

    fig, ax = plt.subplots(figsize=(max(5, len(col_labels) * 0.9),
                                     max(2, len(row_labels) * 0.5)))
    vabs = max(np.nanmax(np.abs(mat[np.isfinite(mat)])), 0.01) if np.any(np.isfinite(mat)) else 1.0
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    im = ax.imshow(mat, cmap='RdBu_r', norm=norm, aspect='auto')

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            col = 'white' if abs(v) > 0.5 else 'black'
            ax.text(j, i, f'{v:+.3f}', ha='center', va='center',
                    fontsize=7, fontweight='bold', color=col)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha='right', fontsize=6)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    plt.colorbar(im, ax=ax, shrink=.8, pad=.04)

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(mat.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(mat.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', length=0)

    fig.tight_layout()
    _save(fig, 'exp2_corr_heatmap')


# ── DOPQ L_ref correlation ──────────────────────────────────────────

def plot_dopq_lref_heatmap(dopq_data):
    """Heatmap of Pearson r(L_ref, task) from DOPQ JSONs across layers."""
    if not dopq_data:
        print("  [skip] no DOPQ data")
        return

    layer_keys = sorted(dopq_data.keys())
    metrics = [
        (r'$L_{\rm ref}$', 'Acc', 'pearson'),
        (r'$L_{\rm ref}$', 'mIoU', 'pearson'),
        (r'$L_{\rm ref}$', 'Acc', 'spearman'),
        (r'$L_{\rm ref}$', 'mIoU', 'spearman'),
    ]
    col_labels = [f'{m[2][:4]} r({m[0]},{m[1]})' for m in metrics]
    row_labels = [f'{bb.split("_")[0]} {layer}' for bb, layer in layer_keys]

    mat = np.full((len(layer_keys), len(metrics)), np.nan)

    for ri, key in enumerate(layer_keys):
        pts = dopq_data[key]
        if len(pts) < 4:
            continue
        dl = np.array([p['dl'] for p in pts])
        acc = np.array([p['acc'] for p in pts])
        miou_vals = [p.get('miou') for p in pts]
        has_miou = all(v is not None for v in miou_vals)
        miou = np.array(miou_vals) if has_miou else None

        for ci, (_, task_name, corr_type) in enumerate(metrics):
            task = acc if task_name == 'Acc' else (miou if has_miou else None)
            if task is None:
                continue
            cr = compute_correlations(dl, task)
            mat[ri, ci] = cr[f'{corr_type}_r']

    fig, ax = plt.subplots(figsize=(max(4, len(col_labels) * 0.9),
                                     max(2, len(row_labels) * 0.45)))
    vabs = max(np.nanmax(np.abs(mat[np.isfinite(mat)])), 0.01) if np.any(np.isfinite(mat)) else 1.0
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    im = ax.imshow(mat, cmap='RdBu_r', norm=norm, aspect='auto')

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            col = 'white' if abs(v) > 0.5 else 'black'
            ax.text(j, i, f'{v:+.3f}', ha='center', va='center',
                    fontsize=7, fontweight='bold', color=col)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha='right', fontsize=6)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    plt.colorbar(im, ax=ax, shrink=.8, pad=.04)

    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(mat.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(mat.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', length=0)

    fig.tight_layout()
    _save(fig, 'exp2_dopq_lref_heatmap')


# ── summary ──────────────────────────────────────────────────────────

def print_summary(corr_data, dopq_data):
    W = 95
    print(f"\n{'=' * W}")
    print("Exp 2 — Correlation Summary (OPQ sweep: MSE vs L_ref as Acc predictor)")
    print(f"{'=' * W}")
    print(f"{'backbone':>15s} {'layer':>6s}  {'n':>3s}  "
          f"{'pear(MSE,Acc)':>13s} {'pear(Lref,Acc)':>14s} "
          f"{'spea(MSE,Acc)':>13s} {'spea(Lref,Acc)':>14s}")
    print('-' * W)

    for key in sorted(corr_data.keys()):
        bb, layer = key
        d = corr_data[key]
        opq_pts = [p for p in d['points'] if p['method'] == 'opq']
        if len(opq_pts) < 3:
            continue
        mse = np.array([p['mse'] for p in opq_pts])
        lref = np.array([p['l_ref'] for p in opq_pts])
        acc = np.array([p['acc'] for p in opq_pts])

        c_ma = compute_correlations(mse, acc)
        c_la = compute_correlations(lref, acc)

        print(f"{bb:>15s} {layer:>6s}  {len(opq_pts):3d}  "
              f"{c_ma['pearson_r']:+.3f} p={c_ma['pearson_p']:.1e}  "
              f"{c_la['pearson_r']:+.3f} p={c_la['pearson_p']:.1e}  "
              f"{c_ma['spearman_r']:+.3f}             "
              f"{c_la['spearman_r']:+.3f}")

    print(f"\n{'=' * W}")
    print("Exp 2 — DOPQ L_ref correlation (from result JSONs)")
    print(f"{'=' * W}")
    print(f"{'backbone':>15s} {'layer':>6s}  {'n':>3s}  "
          f"{'pear(Lref,Acc)':>14s} {'pear(Lref,mIoU)':>15s}")
    print('-' * W)
    for key in sorted(dopq_data.keys()):
        bb, layer = key
        pts = dopq_data[key]
        if len(pts) < 4:
            continue
        dl = np.array([p['dl'] for p in pts])
        acc = np.array([p['acc'] for p in pts])
        miou_vals = [p.get('miou') for p in pts]
        has_miou = all(v is not None for v in miou_vals)
        miou = np.array(miou_vals) if has_miou else None

        c_a = compute_correlations(dl, acc)
        c_m = compute_correlations(dl, miou) if has_miou else {
            'pearson_r': np.nan}
        print(f"{bb:>15s} {layer:>6s}  {len(pts):3d}  "
              f"{c_a['pearson_r']:+.3f}              "
              f"{c_m['pearson_r']:+.3f}")


def main():
    corr_data = load_correlation_jsons()
    dopq_data = load_dopq_correlations()
    print(f"Loaded {len(corr_data)} correlation JSONs, "
          f"{len(dopq_data)} DOPQ layer groups\n")

    if corr_data:
        print("Scatter plots (OPQ MSE vs L_ref) ...")
        plot_scatter(corr_data)
        print("Correlation heatmap (OPQ) ...")
        plot_heatmap(corr_data, dopq_data)

    if dopq_data:
        print("DOPQ L_ref heatmap ...")
        plot_dopq_lref_heatmap(dopq_data)

    print_summary(corr_data, dopq_data)


if __name__ == '__main__':
    main()
