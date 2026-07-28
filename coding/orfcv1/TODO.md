# Current delivery gates

## Menu repair

- Preserve the original 6-bit ORFC codebook exactly.
- Require non-increasing validation tail distortion over the 3--8 bit menu.
- Report the high-rate gain over the 6-bit anchor.

## Candidate-bound short training

- Compare candidate-only and candidate-plus-remainder from the same repaired
  checkpoint, seed and train/validation split.
- Refresh \(c_g\), the ideal allocation and the hard allocation set during
  training.
- Bind the loss to the current ideal candidate, candidate Top-K distortion and
  the original uniform 6-bit reference.
- Accept the proxy only if held-out hard-PQ candidate distortion improves
  without breaking the menu monotonicity gate.

## Final acceptance

- Select the non-uniform allocation using validation only.
- Freeze checkpoint and allocation before test.
- Report nominal/actual rate, tail distortion, classification accuracy and
  segmentation mIoU with paired confidence intervals.
