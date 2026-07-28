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
3. `allocation_train.py warmup` freezes the clean OPQ/identity rotation and
   task-aligns every independently initialized mode codebook.
4. `allocation_train.py short` jointly updates the rotation and all codebooks
   using exact tail distortion at the fixed independently calibrated ideal
   target. Codebooks use
   Adam; the rotation uses Cayley-SGD directly on the orthogonal manifold.
   The recovery auxiliary acts only on the rotation, and conflicts are removed
   after both rotation gradients are projected to the tangent space.
   Full tail distortion never refits \(c_g\), the ideal target or its exact
   finite-menu gap.
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
- `omega_sampled` is the observed range of \(E=D-\Phi\) over the saved
  allocation pool and is therefore a lower bound on the global remainder
  range.
- The recovery condition can only be certified after a valid upper bound on
  the global remainder range is available.
