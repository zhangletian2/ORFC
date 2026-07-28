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
   training images. The outer pass evaluates the complete ideal set; the inner
   pass keeps a small active subset containing the lowest-distortion and nominal
   best members. It minimizes the active best distortion and its violation
   against the best sampled outside competitor. Codebooks use
   Adam; the rotation uses Cayley-SGD directly on the orthogonal manifold.
   The recovery auxiliary updates both the rotation and every active codebook;
   conflicting gradients are projected against the primary distortion
   gradient, with the rotation gradients first mapped to the tangent space.
   Full tail distortion never refits the ideal table inside an inner block.
5. `run_allocation_full.sh` is retained as a stale historical driver. It does
   not yet inherit the accepted Top-256/active-16 objective, 300-image outer
   mining, fixed outer operational anchor or disjoint data slices, and must not
   be used for the next full experiment.
6. `eval_v1_1_all.py` evaluates a frozen non-uniform allocation on downstream
   classification and segmentation tasks.
7. `ideal_set_statistics.py` measures bootstrap stability of the ideal
   allocation and draws exact uniform samples from the complete fixed-budget
   allocation space. Both outputs explicitly carry empirical/probabilistic
   scope and never claim a worst-case certificate.

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

With 64 calibration images, 5000 nonparametric bootstrap resamples produced
4172 different ideal optima; a 95% frequency-ranked set contained 3922
allocations. With 300 images these numbers fell to 588 and 338, respectively,
but the 64-image and 300-image nominal Top-16 sets were disjoint. Outer
calibration therefore requires more than 64 images before another training
claim is made.

The final short gate is
`statset256_mining300_20260728T180555Z`. It uses disjoint 300-image ideal
calibration, 300-image hard mining, 512-image inner optimisation and
300-image independent validation. The exact ideal Top-256 set is used outside
and 16 active members inside. Both arms selected uniform 6-bit as the outer
operational anchor.

On 300 independent images and 843 common allocations, recovery minus
primary-only changed:

- uniform 6-bit distortion by -2.37, CI95 [-6.25, 1.23];
- exact-uniform-512 mean distortion by -2.89, CI95 [-4.28, -1.60];
- common-candidate mean distortion by -5.14, CI95 [-7.90, -2.50];
- Top-256 best distortion by -8.53, CI95 [-14.29, -3.31];
- empirical inside-versus-outside margin by +6.17, CI95 [0.87, 11.47];
- common-candidate remainder range by -3.42, CI95 [-10.65, 1.93].

The short gate therefore passes as an incremental empirical objective: it
improves the ideal-set candidates and sampled recovery margin without a
detectable regression at the operational best allocation. It does not certify
recovery. Uniform 6-bit remains outside the nominal Top-256 and is the
empirical optimum; the sampled remainder-range-to-ideal-gap ratio is about
772. The complete feasible space contains
84,225,312,014,367,853,059,837 allocations, so no finite pool used here is a
strict search-space reduction.
