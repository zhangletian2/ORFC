# Current delivery gates

## Menu repair

- Preserve the original 6-bit ORFC codebook exactly during menu initialization.
- Use nested lower-rate subsets and small data-driven centroid splits above K64.
- Require non-increasing validation tail distortion and a positive K256 gain
  over the 3--8 bit menu.
- Report the high-rate gain over the 6-bit anchor.

## Candidate-bound short training

- Calibrate \(c_g\) on data disjoint from held-out distortion evaluation with
  central JVPs at 6/7/8 bits. Report the across-rate CV and span before treating
  \(c_g2^{-2r_g/d}\) as a common ideal curve.
- Compute an exact Top-K ideal set and its separation from the first outside
  allocation with dynamic programming.
- Keep the discrete ideal table and Top-K set fixed inside each inner block.
  At outer boundaries, recompute both from reserved calibration images without
  fitting full nonlinear distortion.
- Audit Cayley-SGD against the former skew-Cayley/Adam path: require finite
  gradients, stable orthogonality, lower rotation-step time and no regression
  in the exact initial distortion before short training is accepted.
- Compare candidate-only and candidate-plus-remainder from the same repaired
  checkpoint, seed and train/validation split.
- Jointly update \(U\) and all mode codebooks. Protect the primary distortion
  gradient when adding the empirical recovery gradient for every parameter.
- Refresh the hard candidate set around every ideal-set member; retain
  low-distortion outside competitors and extreme measured remainders.
- Optimize the active minimum inside the ideal set and its margin against the
  best sampled outside competitor, while retaining the uniform 6-bit reference.
- Accept the proxy only if held-out hard-PQ candidate distortion improves
  without breaking the menu monotonicity gate.
- Before claiming recovery, derive or validate an upper bound for the
  remainder range over all feasible allocations. A sampled range alone cannot
  certify the theorem.
- Label the fixed audit pool and all adaptively mined pools as empirical.
  Report their size against the exact number of feasible allocations; do not
  treat absence of a sampled counterexample as a certificate.
- Select the ideal-set size from an independent bootstrap stability audit.
  Report nominal Top-K coverage of bootstrap optima and fail the calibration
  gate when the requested probability mass cannot fit the trainable set.
- The paired Top-256/active-16 short gate is complete. Its final accepted run
  uses 300 calibration images, 300 disjoint hard-mining images, a later
  512-image optimisation slice and 300 independent validation images.
- Preserve the accepted outer operational anchor: recompute it only on the
  300-image hard-mining slice, then hold it fixed inside the following inner
  block. Keep the primary gradient protected from the recovery auxiliary.
- Treat Top-256 as an 84.38%-coverage empirical set. Before a stronger recovery
  claim, compare uncertainty-aware Top-338/Top-512 or a bootstrap union while
  keeping the exact feasible-space count and `strict_recovery_certified=false`.

## Next stage before full optimisation

- Do not run the current `run_allocation_full.sh`; it still implements the old
  single-target/no-outer-refresh protocol.
- The fixed-outer 50-step gate is complete: overall and ideal-set distortion
  improve, while the sampled remainder-range reduction is not significant.
- Run `run_multi_outer_medium.sh` next. It inherits the accepted
  Top-256/active-16 objective, shared 300-image initial outer state,
  300-image later hard mining, fixed within-block operational anchors and
  mutually disjoint inner and validation slices.
- Recalibrate the recovery weight after every outer refresh; verify the saved
  `remainder_weight_history` contains one finite positive entry per recovery
  block and zero entries for primary-only.
- Refresh the outer table and operational anchor periodically rather than on
  every inner epoch; record every anchor switch, auxiliary weight and
  primary/recovery gradient cosine.
- Accept the medium gate only if the independent operational-best distortion
  is non-inferior, the ideal-set best and sampled margin improve, menu
  monotonicity holds and no result is presented as a global certificate.

## Full optimisation after the medium gate

- Use the original 5k-scale pool while reserving disjoint calibration,
  hard-mining and held-out validation subsets; test remains isolated.
- Match ORFC's batch 32, 100 epochs, `3e-4` learning rate, gradient clipping
  and epoch-wise `0.5 -> 0.005` PQ-temperature/LR schedules.
- Use exact Top-K sets plus independent exact-uniform allocation samples for
  empirical audit. Do not retain the historical fixed 261-pool contract.
- Record the complete hard validation objective at each outer refresh and
  deliver the final training state. Accept only when final operational-best
  distortion, ideal-set distortion and empirical recovery margin meet the
  paired validation gates.

## Final acceptance

- Select the non-uniform allocation using validation only.
- Freeze the final checkpoint and validation-selected allocation before test.
- Report nominal/actual rate, tail distortion, classification accuracy and
  segmentation mIoU with paired confidence intervals.
