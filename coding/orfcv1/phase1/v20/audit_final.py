"""V20 full fixed-rate neighborhood and train-5k centroid audit."""

import argparse
import json

import numpy as np
import torch

from codec_v1 import load_codec_v1
from .config import SPECS, activate
from .. import engine, tail as tail_mod
from ..v15.audit_final import centroid_usage, full_neighbors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=tuple(SPECS))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    C, anchor = activate(args.profile)
    root, device = C.output_dir(anchor, args.run_id), torch.device(args.device)
    codec = load_codec_v1(root / "codec.pt", device=device)
    allocation = np.load(root / "allocation.npy")
    candidates, moves = full_neighbors(allocation, anchor.mode_bits)
    resident = engine.ResidentSet(*C.load_split("train_val")[:3], device)
    tail = tail_mod.build_tail(C.LAYER, device)
    values = engine.evaluate_allocations(
        codec, tail, resident, np.concatenate((allocation[None], candidates)),
        image_batch=C.EVAL_IMAGE_BATCH, pair_budget=C.EVAL_PAIR_BUDGET).mean(1)
    winner = int(np.argmin(values[1:]))
    neighborhood = {
        "candidate_count": len(candidates), "base_distortion": float(values[0]),
        "best_distortion": float(values[winner + 1]),
        "best_gain": float(values[0] - values[winner + 1]),
        "best_move": moves[winner],
        "locally_optimal": bool(values[winner + 1] >= values[0])}
    del tail, resident, values
    torch.cuda.empty_cache()
    usage = centroid_usage(codec, C.TRAIN_FEATURES, C.N_TRAIN, device,
                           C.NORM_MODE, 4 if max(anchor.mode_sizes) >= 1024 else 16)
    result = {"profile": args.profile, "rate": anchor.rate,
              "allocation": allocation.tolist(),
              "full_two_group_neighborhood": neighborhood,
              "centroid_usage_train5k": usage}
    (root / "final_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
