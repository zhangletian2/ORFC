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
   \(\Phi\). Remainder measurement and the \(M/C/N\) decomposition now call
   the same ideal-model function. Allocation extrema are reselected inside
   every bootstrap replicate.
3. `allocation_train.py warmup` freezes the clean OPQ/identity rotation and
   task-aligns every independently initialized mode codebook.
4. `allocation_train.py short` jointly updates the rotation and all codebooks.
   With `--outer-refresh`, each outer block recomputes the discrete JVP table,
   exact Top-K ideal allocation set and hard fixed-rate competitors on reserved
   training images. The outer pass evaluates the complete ideal set; the inner
   pass keeps a small active subset containing the lowest-distortion and nominal
   best members. The auxiliary is selected explicitly as either the smooth
   sampled remainder range or the worst sampled recovery violation. Codebooks use
   Adam; the rotation uses Cayley-SGD directly on the orthogonal manifold.
   The recovery auxiliary updates both the rotation and every active codebook;
   conflicting gradients are projected against the primary distortion
   gradient, with the rotation gradients first mapped to the tangent space.
   Full tail distortion never refits the ideal table inside an inner block.
   A prepared outer state can now be serialized once and loaded by both paired
   arms, and the recovery weight is recalibrated after every later refresh.
5. `run_allocation_full.sh` now inherits the disjoint 300/300/3900 split,
   Top-256/active-16 objective, fixed outer operational anchor and independent
   300-image audit. It remains gated on the revised three-arm medium experiment
   and must not be launched before that gate passes.
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
- Training outputs report a fixed-audit bootstrap interval for the sampled
  remainder range and its paired change from initialization. Max/min
  allocations are reselected in each resample, so the interval includes
  empirical extreme-selection variability.

Implementation repair on 2026-07-29:

- remainder-range and recovery objectives have separate command-line contracts;
- recovery aggregates the worst active violation by default;
- outer refreshes occur before a trainable inner block, with no untrained final
  refresh;
- auxiliary gradient calibration defaults to 32 images;
- optional per-block LR restart and per-mode gradient coverage are recorded;
- the medium driver compares primary-only, recovery and remainder-range arms
  on the same initial outer state and 300-image independent audit.

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

The fixed-outer 50-step gate is
`statset256_medium_fixedouter_20260728T234817Z`. On 300 independent images and
905 common allocations, recovery minus primary-only changed common-candidate
mean distortion by -25.81, CI95 [-31.13, -20.66], uniform 6-bit distortion by
-3.66, CI95 [-8.13, 0.28], Top-256 best distortion by -43.31, CI95
[-53.69, -32.75], and the empirical set margin by +39.66, CI95
[28.75, 49.94]. The sampled remainder range changed by only -5.16, CI95
[-22.09, 2.38]. Both arms significantly improved overall distortion relative
to the common initialization, but the recovery condition remains false.

`run_multi_outer_medium.sh` implements the next gate without duplicating the
trainer. It prepares one initial outer state for both arms, then performs five
10-step inner blocks by default. Every later arm-specific outer refresh
recomputes the ideal table, Top-256 set, hard competitors and operational
anchor, followed by recovery-weight recalibration. A two-step smoke verified
identical initial audit allocations, nonzero joint gradients, changing
recovery weights, finite Cayley updates and monotonic six-mode menus.

The formal multi-outer run is
`statset256_multiouter_medium_20260729T030153Z`. All five refreshes completed;
recovery weights stayed in [0.265, 0.289], uniform 6-bit remained the
operational anchor, and menu/orthogonality gates passed. On 300 independent
images and 910 common allocations, recovery minus primary-only changed common
mean distortion by -25.03, CI95 [-30.53, -19.72], uniform 6-bit distortion by
-2.41, CI95 [-7.03, 1.81], and sampled remainder range by -4.63, CI95
[-19.58, 1.81]. Ideal-set best and empirical-margin intervals also crossed
zero. The multi-outer arm is statistically indistinguishable from the accepted
fixed-outer recovery arm, so refreshing the outer state adds no measured
benefit yet.

The remaining mismatch is inside the inner objective. Outer 300-image mining
keeps the ideal-set-versus-uniform margin negative, while the four-image
training hinge is usually inactive because its local ordering flips. Only
three of eleven logged recovery terms were nonzero. The next gate must train a
fixed outer-mined inside/outside pair with an unhinged paired expectation, or
estimate that expectation by accumulating multiple inner batches.

The outer-gated working-set path is now implemented behind
`--outer-gated-recovery`. Each outer pass selects up to `--recovery-pairs`
violating outside allocations on the mining population; the inner block keeps
those pairs fixed and optimizes their unhinged mean paired difference. Previous
selected constraints are retained in a persistent workset and re-scored after
later refreshes. The legacy minibatch-hinge path remains the default.

`workset_outergate_smoke2_20260729T1205Z` completed two joint update steps with
one refresh per step. The outer gate and four fixed pairs remained active at
both steps, including the first minibatch where the local margin had the
opposite sign and the legacy hinge would have been zero. Rotation and all six
codebooks had nonzero gradients; the workset grew from 16 to 19 allocations,
the menu remained monotone, and orthogonality error was \(1.09\times10^{-5}\).
This is an implementation smoke only. The next effectiveness gate must use the
accepted 300/300 outer split and independent 300-image audit.
