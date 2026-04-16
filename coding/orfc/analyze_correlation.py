#!/usr/bin/env python
"""
Correlation analysis between tail distortion metrics and downstream task metrics.

Loads JSON results produced by run_distortion_correlation.py, computes
Spearman / Kendall correlations (with bootstrap CI and partial correlations
controlling for bitrate), per-block error profiling, and decomposition
visualisation.

Usage:
    python analyze_correlation.py
    python analyze_correlation.py --results_dir results/distortion_correlation
"""

import os
import sys
import argparse
import json
import glob
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V34_ROOT = os.path.dirname(os.path.abspath(__file__))


# ================================================================
#                    Data loading
# ================================================================

def load_results(results_dir):
    """Load all JSON experiment results into a DataFrame.

    Returns:
        df: DataFrame with scalar metrics (one row per experiment).
        details: list of (config_key, detail_dict) from each experiment.
        g_diag_stats_list: list of (config_key, g_diag_stats_dict).
    """
    json_files = sorted(glob.glob(os.path.join(results_dir, '*.json')))
    json_files = [f for f in json_files if not f.endswith('summary.json')]

    rows = []
    details = []
    g_diag_stats_list = []
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)

        cfg = data['config']
        row = {
            'layer': cfg['layer'],
            'layer_idx': cfg['layer_idx'],
            'K': cfg['K'],
            'embedding_dim': cfg['embedding_dim'],
            'num_groups': cfg['num_groups'],
            'bits_per_token': cfg['bits_per_token'],
            'seed': cfg.get('seed', 42),
        }

        for mk, mv in data.get('distortion_metrics', {}).items():
            row[mk] = mv

        for dk, dv in data.get('downstream', {}).items():
            row[dk] = dv

        rows.append(row)

        config_key = (cfg['layer'], cfg['K'], cfg['embedding_dim'],
                      cfg.get('seed', 42))
        if 'detail' in data and data['detail']:
            details.append((config_key, data['detail']))
        if 'g_diag_stats' in data and data['g_diag_stats']:
            g_diag_stats_list.append((config_key, data['g_diag_stats']))

    if not rows:
        print(f"[warn] No JSON results found in {results_dir}")
        return pd.DataFrame(), [], []

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} experiment points from {results_dir}")
    return df, details, g_diag_stats_list


# ================================================================
#                    Correlation computation
# ================================================================

DISTORTION_COLS = [
    'feature_mse',
    'attn_kl', 'attn_js',
    'cls_attn_kl', 'cls_attn_js', 'cls_attn_l2',
    'attn_output_mse',
    'rollout_cls_l2', 'rollout_cls_kl', 'rollout_patch_fro',
    'rollout_watt_cls_l2', 'rollout_watt_cls_kl', 'rollout_watt_patch_fro',
    'block_output_mse_uniform', 'block_output_mse_exp',
    'quad_sensitivity',
    'quad_rayleigh', 'quad_norm_trace',
    'decomp_dA_V_mse', 'decomp_A_dV_mse',
]

DOWNSTREAM_COLS = ['acc_at_1', 'seg_miou']


def _bootstrap_ci(x, y, stat_fn, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence interval for a bivariate statistic."""
    n = len(x)
    rng = np.random.default_rng(42)
    stats_boot = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        val = stat_fn(x[idx], y[idx])
        stats_boot.append(val)
    lo = np.percentile(stats_boot, (1 - ci) / 2 * 100)
    hi = np.percentile(stats_boot, (1 + ci) / 2 * 100)
    return lo, hi


def compute_correlations(df, distortion_cols=None, downstream_cols=None,
                         n_bootstrap=1000):
    """Compute Spearman and Kendall correlations per layer and globally,
    with bootstrap 95% CIs.

    Returns:
        corr_df: DataFrame with columns
          [layer, metric, downstream, spearman_rho, spearman_p,
           spearman_ci_lo, spearman_ci_hi,
           kendall_tau, kendall_p, kendall_ci_lo, kendall_ci_hi, n]
    """
    if distortion_cols is None:
        distortion_cols = [c for c in DISTORTION_COLS if c in df.columns]
    if downstream_cols is None:
        downstream_cols = [c for c in DOWNSTREAM_COLS if c in df.columns]

    rows = []
    layers = sorted(df['layer'].unique().tolist()) + ['ALL']

    for layer in layers:
        sub = df if layer == 'ALL' else df[df['layer'] == layer]
        if len(sub) < 3:
            continue
        for dm in distortion_cols:
            if dm not in sub.columns:
                continue
            for ds in downstream_cols:
                if ds not in sub.columns:
                    continue
                x = sub[dm].values
                y = sub[ds].values
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 3:
                    continue
                xm, ym = x[mask], y[mask]
                sr, sp = stats.spearmanr(xm, ym)
                kt, kp = stats.kendalltau(xm, ym)

                s_lo, s_hi = _bootstrap_ci(
                    xm, ym,
                    lambda a, b: stats.spearmanr(a, b)[0],
                    n_bootstrap=n_bootstrap)
                k_lo, k_hi = _bootstrap_ci(
                    xm, ym,
                    lambda a, b: stats.kendalltau(a, b)[0],
                    n_bootstrap=n_bootstrap)

                rows.append({
                    'layer': layer,
                    'metric': dm,
                    'downstream': ds,
                    'spearman_rho': sr,
                    'spearman_p': sp,
                    'spearman_ci_lo': s_lo,
                    'spearman_ci_hi': s_hi,
                    'kendall_tau': kt,
                    'kendall_p': kp,
                    'kendall_ci_lo': k_lo,
                    'kendall_ci_hi': k_hi,
                    'n': int(mask.sum()),
                })

    return pd.DataFrame(rows)


def _partial_corr_residual(x, y, z):
    """Partial correlation of x and y controlling for z (residual method)."""
    if len(x) < 4:
        return np.nan, np.nan
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return stats.spearmanr(rx, ry)


def compute_partial_correlations(df, control='bits_per_token',
                                 distortion_cols=None, downstream_cols=None):
    """Partial Spearman correlation controlling for bitrate."""
    if distortion_cols is None:
        distortion_cols = [c for c in DISTORTION_COLS if c in df.columns]
    if downstream_cols is None:
        downstream_cols = [c for c in DOWNSTREAM_COLS if c in df.columns]

    rows = []
    layers = sorted(df['layer'].unique().tolist()) + ['ALL']

    for layer in layers:
        sub = df if layer == 'ALL' else df[df['layer'] == layer]
        if len(sub) < 4:
            continue
        z = sub[control].values
        for dm in distortion_cols:
            if dm not in sub.columns:
                continue
            for ds in downstream_cols:
                if ds not in sub.columns:
                    continue
                x = sub[dm].values
                y = sub[ds].values
                mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                if mask.sum() < 4:
                    continue
                rho, p = _partial_corr_residual(x[mask], y[mask], z[mask])
                rows.append({
                    'layer': layer,
                    'metric': dm,
                    'downstream': ds,
                    'partial_spearman_rho': rho,
                    'partial_spearman_p': p,
                    'control': control,
                    'n': int(mask.sum()),
                })

    return pd.DataFrame(rows)


# ================================================================
#                    Per-block profile analysis  (B)
# ================================================================

def analyze_per_block_profile(details, df, out_dir):
    """For each (block_index, metric_type), compute Spearman vs downstream.

    Produces per_block_profile.csv and a profile heatmap.
    """
    if not details:
        print("  [per_block_profile] No detail data found, skipping.")
        return

    detail_metric_keys = [
        'per_block_output_mse', 'per_block_cls_mse', 'per_block_patch_mse',
        'per_block_attn_output_mse', 'per_block_dA_V', 'per_block_A_dV',
    ]

    profile_rows = []
    for layer in sorted(df['layer'].unique()):
        layer_details = [(k, d) for k, d in details if k[0] == layer]
        if len(layer_details) < 3:
            continue
        layer_df = df[df['layer'] == layer].copy()
        layer_df['_detail_key'] = list(zip(
            layer_df['layer'], layer_df['K'],
            layer_df['embedding_dim'], layer_df['seed']))

        for metric_key in detail_metric_keys:
            sample_detail = layer_details[0][1]
            if metric_key not in sample_detail:
                continue
            n_blocks = len(sample_detail[metric_key])

            for blk_idx in range(n_blocks):
                blk_vals = {}
                for cfg_key, det in layer_details:
                    blk_vals[cfg_key] = det[metric_key][blk_idx]

                vals, acc_vals, miou_vals = [], [], []
                for _, row in layer_df.iterrows():
                    dk = row['_detail_key']
                    if dk in blk_vals:
                        vals.append(blk_vals[dk])
                        acc_vals.append(row.get('acc_at_1', np.nan))
                        miou_vals.append(row.get('seg_miou', np.nan))

                vals = np.array(vals)
                for ds_name, ds_vals in [('acc_at_1', acc_vals),
                                          ('seg_miou', miou_vals)]:
                    ds_arr = np.array(ds_vals)
                    mask = np.isfinite(vals) & np.isfinite(ds_arr)
                    if mask.sum() < 3:
                        continue
                    rho, p = stats.spearmanr(vals[mask], ds_arr[mask])
                    profile_rows.append({
                        'layer': layer,
                        'metric_key': metric_key,
                        'block_idx': blk_idx,
                        'downstream': ds_name,
                        'spearman_rho': rho,
                        'spearman_p': p,
                        'n': int(mask.sum()),
                    })

    if not profile_rows:
        print("  [per_block_profile] Not enough data for profile analysis.")
        return

    profile_df = pd.DataFrame(profile_rows)
    csv_path = os.path.join(out_dir, 'per_block_profile.csv')
    profile_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"  Per-block profile saved: {csv_path}")

    # Plot heatmap per (layer, downstream)
    for ds in profile_df['downstream'].unique():
        for layer in profile_df['layer'].unique():
            sub = profile_df[(profile_df['downstream'] == ds) &
                             (profile_df['layer'] == layer)]
            if sub.empty:
                continue
            pivot = sub.pivot_table(
                index='metric_key', columns='block_idx',
                values='spearman_rho', aggfunc='first')
            if pivot.empty:
                continue

            fig, ax = plt.subplots(
                figsize=(max(4, len(pivot.columns) * 0.8),
                         max(2.5, len(pivot.index) * 0.5)))
            im = ax.imshow(pivot.values, aspect='auto', cmap='RdBu_r',
                           vmin=-1, vmax=1)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f'blk{c}' for c in pivot.columns],
                               rotation=45, ha='right', fontsize=7)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=7)
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if np.isfinite(val):
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                                fontsize=6,
                                color='white' if abs(val) > 0.6 else 'black')
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(f'Per-block ρ vs {ds} ({layer})', fontsize=10)
            plt.tight_layout()
            fname = f'per_block_profile_{layer}_{ds}.png'
            fig.savefig(os.path.join(out_dir, fname), dpi=150,
                        bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved {fname}")


# ================================================================
#                    Decomposition visualisation  (C)
# ================================================================

def plot_decomposition(df, out_dir):
    """Compare decomp_dA_V_mse vs decomp_A_dV_mse and their correlation
    with downstream metrics."""
    cols_needed = ['decomp_dA_V_mse', 'decomp_A_dV_mse']
    if not all(c in df.columns for c in cols_needed):
        print("  [decomposition] Missing decomposition columns, skipping.")
        return

    downstream_cols = [c for c in DOWNSTREAM_COLS if c in df.columns]
    layers = sorted(df['layer'].unique())
    cmap = plt.cm.get_cmap('tab10', max(len(layers), 1))
    layer_colors = {l: cmap(i) for i, l in enumerate(layers)}

    # Ratio bar chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, layer in enumerate(layers):
        sub = df[df['layer'] == layer]
        dA_V_mean = sub['decomp_dA_V_mse'].mean()
        A_dV_mean = sub['decomp_A_dV_mse'].mean()
        total = dA_V_mean + A_dV_mean
        if total > 0:
            axes[0].bar(i, dA_V_mean / total, color='steelblue',
                        label='(δA)V' if i == 0 else '')
            axes[0].bar(i, A_dV_mean / total, bottom=dA_V_mean / total,
                        color='coral', label='A(δV)' if i == 0 else '')
    axes[0].set_xticks(range(len(layers)))
    axes[0].set_xticklabels(layers, fontsize=8)
    axes[0].set_ylabel('Fraction of ‖δ(AV)‖²')
    axes[0].set_title('Decomposition ratio per layer')
    axes[0].legend(fontsize=8)

    # Scatter: each component vs downstream
    for ds in downstream_cols[:1]:
        for comp, color, label in [('decomp_dA_V_mse', 'steelblue', '(δA)V'),
                                    ('decomp_A_dV_mse', 'coral', 'A(δV)')]:
            for layer in layers:
                sub = df[df['layer'] == layer]
                axes[1].scatter(sub[comp], sub[ds], c=[layer_colors[layer]],
                                marker='o' if 'dA' in comp else 's',
                                s=30, alpha=0.7, edgecolors='k',
                                linewidths=0.3)
        axes[1].set_xlabel('Component MSE', fontsize=9)
        axes[1].set_ylabel(ds, fontsize=9)
        axes[1].set_title(f'Decomposition components vs {ds}')

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   label='(δA)V', markersize=7),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                   label='A(δV)', markersize=7),
        ] + [Line2D([0], [0], marker='o', color='w',
                     markerfacecolor=layer_colors[l],
                     label=l, markersize=7) for l in layers]
        axes[1].legend(handles=legend_elements, fontsize=7, loc='best')

    plt.tight_layout()
    fname = 'decomposition_analysis.png'
    fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


# ================================================================
#                    Visualisation
# ================================================================

def plot_correlation_heatmap(corr_df, out_dir, stat_col='spearman_rho',
                             title_suffix=''):
    """Heatmap: metric (rows) x layer (cols), coloured by correlation."""
    for ds in corr_df['downstream'].unique():
        sub = corr_df[(corr_df['downstream'] == ds) &
                      (corr_df['layer'] != 'ALL')]
        if sub.empty:
            continue
        pivot = sub.pivot_table(
            index='metric', columns='layer', values=stat_col, aggfunc='first'
        )
        if pivot.empty:
            continue

        fig, ax = plt.subplots(
            figsize=(max(4, len(pivot.columns) * 1.4),
                     max(3, len(pivot.index) * 0.45))
        )
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdBu_r',
                        vmin=-1, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=7,
                            color='white' if abs(val) > 0.6 else 'black')

        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f'{stat_col} vs {ds}{title_suffix}')
        plt.tight_layout()

        fname = f'heatmap_{stat_col}_{ds}{title_suffix}.png'
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_scatter(df, out_dir):
    """Scatter plots: each distortion metric vs each downstream metric,
    coloured by layer."""
    distortion_cols = [c for c in DISTORTION_COLS if c in df.columns]
    downstream_cols = [c for c in DOWNSTREAM_COLS if c in df.columns]

    layers = sorted(df['layer'].unique())
    cmap = plt.cm.get_cmap('tab10', max(len(layers), 1))
    layer_colors = {l: cmap(i) for i, l in enumerate(layers)}

    for ds in downstream_cols:
        n_met = len(distortion_cols)
        ncols = min(4, n_met)
        nrows = (n_met + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.5 * ncols, 3.5 * nrows))
        axes = np.atleast_2d(axes)

        for idx, dm in enumerate(distortion_cols):
            ax = axes[idx // ncols, idx % ncols]
            for layer in layers:
                sub = df[df['layer'] == layer]
                ax.scatter(sub[dm], sub[ds], c=[layer_colors[layer]],
                           label=layer, s=40, alpha=0.8, edgecolors='k',
                           linewidths=0.4)
            ax.set_xlabel(dm, fontsize=8)
            ax.set_ylabel(ds, fontsize=8)
            ax.tick_params(labelsize=7)
            valid = df[[dm, ds]].dropna()
            if len(valid) >= 3:
                rho, _ = stats.spearmanr(valid[dm], valid[ds])
                ax.set_title(f'ρ={rho:.3f}', fontsize=9)

        for idx in range(n_met, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center',
                   ncol=len(layers), fontsize=8)
        fig.suptitle(f'Distortion metrics vs {ds}', fontsize=12, y=1.02)
        plt.tight_layout()

        fname = f'scatter_{ds}.png'
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_bitrate_controlled(df, out_dir):
    """For each unique bitrate, plot distortion metric vs downstream."""
    downstream_cols = [c for c in DOWNSTREAM_COLS if c in df.columns]
    distortion_cols = [c for c in DISTORTION_COLS if c in df.columns]

    bitrates = sorted(df['bits_per_token'].unique())
    if len(bitrates) < 2:
        return

    layers = sorted(df['layer'].unique())
    cmap = plt.cm.get_cmap('tab10', max(len(layers), 1))
    layer_colors = {l: cmap(i) for i, l in enumerate(layers)}

    for ds in downstream_cols:
        top_metrics = distortion_cols[:6]
        n_met = len(top_metrics)
        fig, axes = plt.subplots(1, n_met, figsize=(4 * n_met, 3.5))
        if n_met == 1:
            axes = [axes]

        for idx, dm in enumerate(top_metrics):
            ax = axes[idx]
            for layer in layers:
                sub = df[df['layer'] == layer]
                ax.scatter(sub[dm], sub[ds], c=[layer_colors[layer]],
                           label=layer, s=40, alpha=0.8, edgecolors='k',
                           linewidths=0.4)
                for _, row in sub.iterrows():
                    ax.annotate(f"{row['bits_per_token']:.0f}b",
                                (row[dm], row[ds]),
                                fontsize=5, alpha=0.6)
            ax.set_xlabel(dm, fontsize=8)
            ax.set_ylabel(ds, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_title(dm, fontsize=9)

        handles, labels = axes[0].get_legend_handles_labels()
        seen = set()
        unique_h, unique_l = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                unique_h.append(h)
                unique_l.append(l)
                seen.add(l)
        fig.legend(unique_h, unique_l, loc='upper center',
                   ncol=len(layers), fontsize=8)
        fig.suptitle(f'Distortion vs {ds} (annotated by bitrate)', y=1.04,
                     fontsize=11)
        plt.tight_layout()

        fname = f'bitrate_controlled_{ds}.png'
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {fname}")


# ================================================================
#                    Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Correlation analysis for distortion-downstream experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_dir", type=str,
                        default=os.path.join(V34_ROOT, 'results',
                                             'distortion_correlation'))
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir for plots. Defaults to results_dir.")
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="Bootstrap resamples for CI estimation")

    args = parser.parse_args()
    out_dir = args.output_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load
    result = load_results(args.results_dir)
    if isinstance(result, tuple):
        df, details, g_diag_stats_list = result
    else:
        df = result
        details, g_diag_stats_list = [], []

    if df.empty:
        print("No data to analyse. Run experiments first.")
        return

    csv_path = os.path.join(out_dir, 'summary.csv')
    df.to_csv(csv_path, index=False, float_format='%.8f')
    print(f"Summary saved: {csv_path}")

    # 2. Correlations (with bootstrap CI)
    print("\n=== Spearman / Kendall correlations (with bootstrap CI) ===")
    corr_df = compute_correlations(df, n_bootstrap=args.n_bootstrap)
    if not corr_df.empty:
        corr_path = os.path.join(out_dir, 'correlations.csv')
        corr_df.to_csv(corr_path, index=False, float_format='%.4f')
        print(f"Correlations saved: {corr_path}")

        all_corr = corr_df[corr_df['layer'] == 'ALL'].sort_values(
            'spearman_rho', key=abs, ascending=False
        )
        if not all_corr.empty:
            print("\nGlobal correlations (sorted by |rho|):")
            for _, r in all_corr.iterrows():
                print(f"  {r['metric']:30s} vs {r['downstream']:10s}: "
                      f"ρ={r['spearman_rho']:+.4f} "
                      f"[{r['spearman_ci_lo']:+.4f}, {r['spearman_ci_hi']:+.4f}] "
                      f"(p={r['spearman_p']:.4f}), "
                      f"τ={r['kendall_tau']:+.4f} "
                      f"[{r['kendall_ci_lo']:+.4f}, {r['kendall_ci_hi']:+.4f}] "
                      f"(p={r['kendall_p']:.4f})")

    # 3. Partial correlations
    print("\n=== Partial correlations (controlling for bits_per_token) ===")
    pcorr_df = compute_partial_correlations(df)
    if not pcorr_df.empty:
        pcorr_path = os.path.join(out_dir, 'partial_correlations.csv')
        pcorr_df.to_csv(pcorr_path, index=False, float_format='%.4f')
        print(f"Partial correlations saved: {pcorr_path}")

        all_pcorr = pcorr_df[pcorr_df['layer'] == 'ALL'].sort_values(
            'partial_spearman_rho', key=abs, ascending=False
        )
        if not all_pcorr.empty:
            print("\nGlobal partial correlations (sorted by |rho|):")
            for _, r in all_pcorr.iterrows():
                print(f"  {r['metric']:30s} vs {r['downstream']:10s}: "
                      f"ρ_partial={r['partial_spearman_rho']:+.4f} "
                      f"(p={r['partial_spearman_p']:.4f})")

    # 4. Per-block profile analysis (B)
    print("\n=== Per-block error propagation profile ===")
    analyze_per_block_profile(details, df, out_dir)

    # 5. Plots
    print("\n=== Generating plots ===")
    if not corr_df.empty:
        plot_correlation_heatmap(corr_df, out_dir, 'spearman_rho')
        plot_correlation_heatmap(corr_df, out_dir, 'kendall_tau')
    if not pcorr_df.empty:
        plot_correlation_heatmap(pcorr_df, out_dir, 'partial_spearman_rho',
                                 title_suffix='_partial')

    plot_scatter(df, out_dir)
    plot_bitrate_controlled(df, out_dir)

    # 6. Decomposition visualisation (C)
    print("\n=== Decomposition analysis ===")
    plot_decomposition(df, out_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
