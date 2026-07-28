# ORFC fixed-rate allocation status

Branch: `dev`

Current pipeline:

1. `p1_fixed_rate.py` prepares multi-mode PQ menus and measures fixed-total-rate
   distortion.
2. `fixed_rate_remainder.py` measures
   \(D=\Phi+M+C+N\) on hard allocations.
3. `allocation_train.py repair` freezes the original 6-bit ORFC codebook,
   expands it into higher-rate modes, and trains the remaining codebooks under
   an operational monotonic-RD constraint.
4. `allocation_train.py short` jointly updates the orthogonal representation
   and non-anchor codebooks using candidate distortion, candidate Top-K
   distortion and a sampled remainder-range proxy.
5. `eval_v1_1_all.py` evaluates a frozen non-uniform allocation on downstream
   classification and segmentation tasks.

Stable artifacts:

- `artifacts/dinov2_vitl14/p1_menus/{identity,opq,orfc,response}.pt`
- `results/dinov2_vitl14/p1_fixed_rate/formal_p1_corrected_20260727T090628Z/`

The historical response-equalisation and P2 training pipelines were removed
after their conclusions were recorded in the local project node documents.
