# Periodic replay checkpoint contract

Each immutable `step_NNNNNN/` directory contains states captured after the
same optimizer step:

- `codec.pt`, written by `save_codec_v1`;
- `policy.pt`, with `policy_state`, `groups`, `bit_costs`, `total_bits`,
  `temperature`, and `step`;
- `meta.json`, with `step`, `map_allocation`, and a validation-split identifier.

The run root separately owns the final hard `allocation.npy`. Optimizer states
are outside this read-only replay contract. A producer writes a temporary
directory and renames it only after all three files exist.

`replay_metrics.py` evaluates every checkpoint on the same frozen v12
validation rows and reports:

1. hard MSE at the run's final allocation (codec trajectory only);
2. hard MSE at that checkpoint's policy MAP (adds MAP switching);
3. Monte-Carlo hard MSE under that checkpoint's policy (adds policy spread).

The Monte-Carlo seed and draw count are fixed across checkpoints. Actual draws
are stored in the output, making the comparison exactly reproducible.
