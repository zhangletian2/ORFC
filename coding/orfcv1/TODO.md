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
- Compute the ideal allocation and its exact global finite-menu gap with
  Top-2 dynamic programming.
- Keep \(c_g\), the ideal target and this gap fixed during one training run.
  Full nonlinear tail distortion must only enter \(E=D-\Phi\); it must not be
  used to refit the ideal model.
- Audit Cayley-SGD against the former skew-Cayley/Adam path: require finite
  gradients, stable orthogonality, lower rotation-step time and no regression
  in the exact initial distortion before short training is accepted.
- Compare candidate-only and candidate-plus-remainder from the same repaired
  checkpoint, seed and train/validation split.
- Jointly update \(U\) and all mode codebooks for reconstruction; route the
  remainder gradient to \(U\) so its contribution can be identified.
- Refresh the hard candidate set around the fixed ideal solution; retain
  difficult allocations with extreme measured remainder.
- Bind the loss to the complete minimizer set, candidate Top-K distortion,
  the uniform 6-bit reference and the sampled recovery violation.
- Accept the proxy only if held-out hard-PQ candidate distortion improves
  without breaking the menu monotonicity gate.
- Before claiming recovery, derive or validate an upper bound for the
  remainder range over all feasible allocations. A sampled range alone cannot
  certify the theorem.

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
