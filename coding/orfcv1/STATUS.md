# ORFC fixed-rate allocation status

Branch: `dev`

Current pipeline:

1. `p1_fixed_rate.py` prepares multi-mode PQ menus and measures fixed-total-rate
   distortion.
2. `fixed_rate_remainder.py` measures
   \(D=\Phi+M+C+N\) on hard allocations.
3. `allocation_train.py repair` keeps the original 6-bit ORFC codebook as an
   initialization anchor, builds nested lower modes and data-driven centroid
   splits for higher modes, then enforces an operational monotonic-RD gate.
4. `allocation_train.py short` jointly updates the orthogonal representation
   and every mode codebook. It periodically refreshes \(c_g\), the analytic
   minimizer set, its external gap and hard candidate set.
5. `run_allocation_full.sh` extends the accepted short configuration to the
   original 5k-scale protocol (4.5k optimisation plus 500 held-out validation)
   with 100 epochs, temperature annealing and a fixed hard-validation trace.
   It always returns the final training state instead of rolling back to an
   earlier checkpoint.
6. `eval_v1_1_all.py` evaluates a frozen non-uniform allocation on downstream
   classification and segmentation tasks.

Stable artifacts:

- `artifacts/dinov2_vitl14/p1_menus/{identity,opq,orfc,response}.pt`
- `results/dinov2_vitl14/p1_fixed_rate/formal_p1_corrected_20260727T090628Z/`

The historical response-equalisation and P2 training pipelines were removed
after their conclusions were recorded in the local project node documents.

The former short run confirmed that the sampled remainder range is
differentiable, but also exposed three invalid assumptions: freezing K64 while
updating \(U\), treating one arbitrary `argmin` as a unique analytic solution,
and duplicating anchor centroids for K128/K256. The current code repairs these
interfaces; a new formal run has not yet been started.
