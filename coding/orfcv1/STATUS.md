# ORFC fixed-rate allocation status

Branch: `dev`

Current pipeline:

1. `p1_fixed_rate.py` learns an OPQ rotation or starts from identity, stores
   the effective matrix directly, then runs every mode's K-means in those
   coordinates.
2. `fixed_rate_remainder.py` measures
   \(D=\Phi+E\) on hard fixed-rate allocations.  The coefficient \(c_g\) is
   calibrated independently with a central JVP along each realised group
   quantisation error.  The 6/7/8-bit normalized coefficients are retained to
   test the common-shape assumption instead of fitting it from \(D\).
   When that assumption fails, calibration stores the directly measured
   per-group/per-mode JVP table and the evaluator uses its separable sum as
   \(\Phi\).
3. `allocation_train.py warmup` freezes the clean OPQ/identity rotation and
   task-aligns every independently initialized mode codebook.
4. `allocation_train.py short` jointly updates the rotation and all codebooks.
   With `--outer-refresh`, each outer block recomputes the discrete JVP table,
   exact Top-K ideal allocation set and hard fixed-rate competitors on reserved
   training images.  The inner block minimizes the best ideal-set distortion
   and the empirical violation between the best set member and best sampled
   outside competitor, using the active pair's subgradient. Codebooks use
   Adam; the rotation uses Cayley-SGD directly on the orthogonal manifold.
   The recovery auxiliary updates both the rotation and every active codebook;
   conflicting gradients are projected against the primary distortion
   gradient, with the rotation gradients first mapped to the tangent space.
   Full tail distortion never refits the ideal table inside an inner block.
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

The discarded anchor pipeline loaded the task-trained ORFC rotation and K64
codebook, mixed them with independently initialized modes and continued after a
failed menu gate. Its results are invalid for testing the new objective and
have been removed. A clean OPQ-versus-scratch run has not yet been started.

Measurement contract:

- `ideal_gap` is the exact best-versus-runner-up gap over every feasible
  finite-menu allocation, computed by Top-2 dynamic programming.
- `ideal_set_gap` is the exact ideal-cost separation between the best
  allocation and the first allocation outside the Top-K ideal set.
- `omega_sampled` is the observed range of \(E=D-\Phi\) over the saved
  allocation pool. It is a lower bound on the global remainder range, so the
  fixed audit pool is empirical even when Top-K ideal members are exact.
- The recovery condition can only be certified after a valid upper bound on
  the global remainder range is available.
- Every exact Top-K member and the first ideal-cost allocation outside that
  set are always included in the sampled pool.

Outer/inner smoke:

- zero-step refresh preserved every codec parameter exactly;
- two update steps produced non-zero gradients and updates for the rotation
  and all six mode codebooks while preserving menu monotonicity;
- the 4-image held-out empirical margin moved from -12426.8 to -12409.2.
  This is a gradient-path smoke only, not an effectiveness result.

The 10-step single-target formal pair reduced the fixed-pool remainder range
only with the recovery auxiliary, but its held-out target distortion was
7.37 higher than primary-only (95% CI [1.07, 13.68]).  The near-zero
best-versus-runner-up ideal gap made the single target unstable; this motivated
the Top-K set objective.
