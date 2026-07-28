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
- Run the paired `primary_only` versus `primary_recovery` short experiment from
  `run_outer_inner_short.sh`. Do not start full optimisation until held-out
  target distortion is non-inferior and the empirical recovery margin improves.

## Full optimisation

- Use the original 5k-scale pool as 4.5k optimisation plus 500 held-out
  validation images; test remains isolated.
- Match ORFC's batch 32, 100 epochs, `3e-4` learning rate, gradient clipping
  and epoch-wise `0.5 -> 0.005` PQ-temperature/LR schedules.
- Keep the 261 allocations fixed as an audit set. Refresh a separate training
  allocation pool once per epoch.
- Record the complete hard validation objective once per epoch, but always
  deliver the final training state. Accept the optimisation only when its
  final candidate distortion and remainder range improve over initialization.

## Final acceptance

- Select the non-uniform allocation using validation only.
- Freeze the final checkpoint and validation-selected allocation before test.
- Report nominal/actual rate, tail distortion, classification accuracy and
  segmentation mIoU with paired confidence intervals.
