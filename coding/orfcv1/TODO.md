# Current delivery gates

## Menu repair

- Preserve the original 6-bit ORFC codebook exactly during menu initialization.
- Use nested lower-rate subsets and small data-driven centroid splits above K64.
- Require non-increasing validation tail distortion and a positive K256 gain
  over the 3--8 bit menu.
- Report the high-rate gain over the 6-bit anchor.

## Candidate-bound short training

- Compare candidate-only and candidate-plus-remainder from the same repaired
  checkpoint, seed and train/validation split.
- Jointly update \(U\) and all mode codebooks, including K64.
- Refresh \(c_g\), the analytic minimizer set, its external gap and the hard
  candidate set during training.
- Bind the loss to the complete minimizer set, candidate Top-K distortion,
  the uniform 6-bit reference and the sampled recovery violation.
- Accept the proxy only if held-out hard-PQ candidate distortion improves
  without breaking the menu monotonicity gate.

## Full optimisation

- Use the original 5k-scale pool as 4.5k optimisation plus 500 held-out
  validation images; test remains isolated.
- Match ORFC's 100 epochs, `3e-4` learning rate, gradient clipping and
  `0.5 -> 0.005` PQ-temperature schedule.
- Refresh coefficients/candidates and select checkpoints once per epoch using
  validation only.

## Final acceptance

- Select the non-uniform allocation using validation only.
- Freeze checkpoint and allocation before test.
- Report nominal/actual rate, tail distortion, classification accuracy and
  segmentation mIoU with paired confidence intervals.
