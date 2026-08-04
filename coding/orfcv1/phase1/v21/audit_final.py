"""V21 hard ECVQ neighborhood, rate, and centroid audit."""

import argparse
import json

import numpy as np
import torch

from codec_v1 import load_codec_v1

from .config import SPECS, activate
from .. import engine
from .. import tail as tail_mod
from ..v12 import qhard
from ..v15.audit_final import centroid_usage, full_neighbors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", required=True, choices=tuple(SPECS))
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    C = activate(args.block)
    anchor = C.ANCHOR_BY_NAME[args.anchor]
    root = C.output_dir(anchor, args.run_id)
    device = torch.device(args.device)
    codec = load_codec_v1(root / "codec.pt", device=device)
    allocation = np.load(root / "allocation.npy")
    candidates, moves = full_neighbors(allocation, anchor.mode_bits)
    val_paths = C.load_split("train_val")
    resident = engine.ResidentSet(*val_paths[:3], device)
    tail = tail_mod.build_tail(C.LAYER, device)
    matrix = engine.evaluate_allocations(
        codec, tail, resident, np.concatenate((allocation[None], candidates)),
        image_batch=C.EVAL_IMAGE_BATCH, pair_budget=C.EVAL_PAIR_BUDGET)
    means = matrix.mean(1)
    winner = int(np.argmin(means[1:]))
    rate_chunks = []
    for first in range(0, resident.count, C.EVAL_IMAGE_BATCH):
        rate_chunks.append(qhard.rates(
            codec, resident.y[first:first + C.EVAL_IMAGE_BATCH], allocation))
    result = {
        "block": args.block, "anchor": anchor.name,
        "nominal_rate": int(sum(anchor.mode_bits[m] for m in allocation)),
        "cross_entropy_rate_bpt": float(torch.cat(rate_chunks, 1).mean()),
        "allocation": allocation.tolist(),
        "full_two_group_neighborhood": {
            "candidate_count": len(candidates),
            "base_distortion": float(means[0]),
            "best_distortion": float(means[winner + 1]),
            "best_gain": float(means[0] - means[winner + 1]),
            "best_move": moves[winner],
            "locally_optimal": bool(means[winner + 1] >= means[0])}}
    del tail, resident, matrix
    torch.cuda.empty_cache()
    result["centroid_usage_train5k"] = centroid_usage(
        codec, C.TRAIN_FEATURES, C.N_TRAIN, device, C.NORM_MODE, 16)
    (root / "final_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
