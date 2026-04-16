#!/usr/bin/env python
"""
Figure 3 — Signal divergence: optimisation direction conflict
            and MSE-ablation decoupling after training.

3×2 heatmap:
  Col 1: r(S_lref, S_mse)  — gradient direction conflict
  Col 2: r(MSE, Ablation)  — high-MSE groups ≠ high task impact
  Rows:  Identity / MSE-opt (OPQ) / L_ref-opt (Trained)
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'font.family': 'sans-serif',
    'axes.linewidth': 0.6,
})

root = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(root, 'results', 'analysis_intro_v2',
                       'sensitivity_blk20_K16_emb32_ndiag200_ckpt.json')) as f:
    data = json.load(f)

rows = ['Identity', 'MSE-opt', r'$L_{\rm ref}$-opt']
cols = [r'$r(S_{L_{\rm ref}},\, S_{\rm MSE})$',
        r'$r({\rm MSE},\, {\rm Ablation})$']

keys = ['identity', 'opq', 'trained']
mat = np.array([
    [data['conditions'][k]['correlations']['sens_lref_vs_sens_mse'],
     data['conditions'][k]['correlations']['mse_vs_ablation']]
    for k in keys
])

print("Heatmap values:")
for i, r in enumerate(rows):
    print(f"  {r:>12s}: {mat[i,0]:+.3f}  {mat[i,1]:+.3f}")

# ── Figure ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(2.6, 2.0))

norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
cmap = plt.cm.RdBu_r

im = ax.imshow(mat, cmap=cmap, norm=norm, aspect='auto')

for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        color = 'white' if abs(v) > 0.5 else 'black'
        ax.text(j, i, f'{v:+.3f}', ha='center', va='center',
                fontsize=10, fontweight='bold', color=color)

ax.set_xticks(np.arange(len(cols)))
ax.set_xticklabels(cols)
ax.set_yticks(np.arange(len(rows)))
ax.set_yticklabels(rows)

ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False,
               length=0)

for sp in ax.spines.values():
    sp.set_visible(False)

ax.set_xticks(np.arange(mat.shape[1] + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(mat.shape[0] + 1) - 0.5, minor=True)
ax.grid(which='minor', color='white', linewidth=2)
ax.tick_params(which='minor', length=0)

fig.tight_layout()

fig_dir = os.path.join(root, 'figures')
os.makedirs(fig_dir, exist_ok=True)
for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(fig_dir, f'fig3_divergence.{ext}'),
                dpi=300, bbox_inches='tight')
print(f"Saved: {fig_dir}/fig3_divergence.[pdf|png]")
