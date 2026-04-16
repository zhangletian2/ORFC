#!/usr/bin/env python
"""
Figure 2 (cls) — Restore-one-group ablation for L_cls.

Reads JSON produced by compute_cls_sensitivity.py and plots per-group
ΔL_cls (classification-loss drop when restoring group g)
for Identity / OPQ / Trained.  Same visual style as plot_fig2.py.
"""

import json, os, sys
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

COL_ID  = '#777' # #E24A33
COL_OPQ = '#286CA0' # #999999
COL_TR  = '#B983CD' # #55A868

root = os.path.dirname(os.path.abspath(__file__))

json_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    root, 'results', 'analysis_intro_v2',
    'cls_sensitivity_blk20_K16_emb32_ndiag200_ckpt.json')

with open(json_path) as f:
    data = json.load(f)

conds = data['conditions']
KEY = 'ablation_cls'

s_id  = np.array(conds['identity'][KEY]['values'])
s_opq = np.array(conds['opq'][KEY]['values'])
s_tr  = np.array(conds['trained'][KEY]['values'])

cv_id  = conds['identity'][KEY]['cv']
cv_opq = conds['opq'][KEY]['cv']
cv_tr  = conds['trained'][KEY]['cv']

print(f"CV — Identity: {cv_id:.3f}  OPQ: {cv_opq:.3f}  Trained: {cv_tr:.3f}")

# ── Figure ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.2, 2.4))

xs = np.arange(len(s_id))
ax.plot(xs, s_id,  '-o', color=COL_ID,  ms=3, lw=1.0, alpha=0.9,
        label='Identity')
ax.plot(xs, s_opq, '-s', color=COL_OPQ, ms=3, lw=1.0, alpha=0.9,
        label='MSE')
ax.plot(xs, s_tr,  '-^', color=COL_TR,  ms=3, lw=1.0, alpha=0.9,
        label=r'${\mathcal{L}}_{\rm ref}$')

ax.set_xlabel('Group index')
ax.set_ylabel(r'$\Delta {\rm Acc}$ (%)')
ax.set_xlim(-0.5, len(s_id) - 0.5)
ax.set_xticks(np.arange(0, len(s_id), 4))
ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
ax.yaxis.get_offset_text().set_fontsize(7)

for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)

ax.legend(fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, 1.18),
          ncol=3, framealpha=0.9, borderpad=0.3,
          handlelength=1.5, columnspacing=1.0)

fig.tight_layout()

fig_dir = os.path.join(root, 'figures')
os.makedirs(fig_dir, exist_ok=True)
for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(fig_dir, f'fig2_cls_ablation.{ext}'),
                dpi=300, bbox_inches='tight')
print(f"Saved: {fig_dir}/fig2_cls_ablation.[pdf|png]")
