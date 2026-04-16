#!/usr/bin/env python
"""
Figure 1 — Empirical observation: MSE does not predict downstream
task accuracy; L_ref does.

Single scatter plot for DINOv2 ViT-L/14 Block 20.
x-axis: normalised distortion (0 = best, 1 = worst).
y-axis: Acc drop (higher = worse).
Blue dots = MSE as proxy, purple dots = L_ref as proxy.
"""

import json, os
import numpy as np
from scipy import stats as sp_stats
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

C_MSE  = '#286CA0'
C_LREF = '#B983CD'

root = os.path.dirname(os.path.abspath(__file__))

# ── Load correlation JSON ────────────────────────────────
corr_path = os.path.join(root, 'results', 'analysis_intro_v2',
                         'correlation_dinov2_vitl14_blk20.json')
if not os.path.exists(corr_path):
    corr_path = os.path.join(root, 'results', 'analysis_intro_v2',
                             'correlation_blk20.json')

with open(corr_path) as f:
    corr_data = json.load(f)

opq_pts = [p for p in corr_data['points'] if p['method'] == 'opq']

all_mse  = np.array([p['mse'] for p in opq_pts])
all_lref = np.array([p['l_ref'] for p in opq_pts])
all_acc  = np.array([p['acc'] for p in opq_pts]) * 100

acc_orig = all_acc.max()
acc_drop = acc_orig - all_acc

mse_norm  = (all_mse  - all_mse.min())  / (all_mse.max()  - all_mse.min())
lref_norm = (all_lref - all_lref.min()) / (all_lref.max() - all_lref.min())

r_mse  = sp_stats.pearsonr(mse_norm, acc_drop)[0]
r_lref = sp_stats.pearsonr(lref_norm, acc_drop)[0]

print(f"OPQ points: {len(opq_pts)}")
print(f"Pearson r(MSE,  Acc drop) = {r_mse:.3f}")
print(f"Pearson r(Lref, Acc drop) = {r_lref:.3f}")

# ── Figure ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.2, 2.4))

ax.scatter(mse_norm, acc_drop, s=22, c=C_MSE, alpha=0.6,
           edgecolors=C_MSE, linewidths=0.6, zorder=2,
           label=f'MSE (r={r_mse:.2f})')
ax.scatter(lref_norm, acc_drop, s=22, c=C_LREF, alpha=0.6,
           edgecolors=C_LREF, linewidths=0.6, zorder=3,
           label=rf'$\mathcal{{L}}_{{\rm ref}}$ (r={r_lref:.2f})')

x_gt = np.linspace(0, 1, 50)
y_max = acc_drop.max() if acc_drop.max() > 0 else 1.0
ax.plot(x_gt, x_gt * y_max, ls='--', lw=1.0, color='#333', alpha=0.4,
        zorder=1, label='Ideal')

for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)

ax.set_xlabel('Normalised distortion')
ax.set_ylabel('Acc drop (%)')
ax.set_xlim(-0.05, 1.05)
ax.set_title('DINOv2 ViT-L/14  Block 20', fontsize=9, pad=4)
ax.legend(fontsize=6.5, loc='upper left', framealpha=1.0,
          borderpad=0.3, handlelength=1.0)

fig.tight_layout(pad=0.4)

fig_dir = os.path.join(root, 'figures')
os.makedirs(fig_dir, exist_ok=True)
for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(fig_dir, f'fig1_correlation.{ext}'),
                dpi=300, bbox_inches='tight')
print(f"Saved: {fig_dir}/fig1_correlation.[pdf|png]")
