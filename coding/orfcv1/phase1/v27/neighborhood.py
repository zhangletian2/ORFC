"""Held-out exact-rate two-group neighborhood audit for an arbitrary codec."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from codec_v1 import load_codec_v1
from phase1 import engine, tail as tail_mod
from phase1.v15.audit_final import full_neighbors
from phase1.v21.config import activate
from phase1.v22.probe import evaluate, split_resident


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", default="blk20")
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--allocation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offset", type=int, default=256)
    parser.add_argument("--images", type=int, default=256)
    parser.add_argument("--image-batch", type=int, default=16)
    parser.add_argument("--rate-lambda", type=float, default=0.5)
    args = parser.parse_args(argv)

    config = activate(args.block)
    anchor = config.ANCHOR_BY_NAME[args.anchor]
    device = torch.device(args.device)
    engine.configure_precision(config.ALLOW_TF32)
    codec = load_codec_v1(Path(args.codec), device=device).eval()
    allocation = np.load(args.allocation).astype(np.int64)
    neighbors, moves = full_neighbors(allocation, anchor.mode_bits)
    rows = config.load_split("train_val")[2]
    resident = split_resident(
        config, rows, args.offset, args.images, device)
    tail = tail_mod.build_tail(config.LAYER, device)
    candidates = np.concatenate((allocation[None], neighbors), axis=0)
    distortion, rate, objective = evaluate(
        codec, tail, resident, candidates, args.image_batch,
        args.rate_lambda)
    result = {"definition": "all exact-rate two-group transfers",
              "offset": args.offset, "images": args.images,
              "candidate_count": int(len(neighbors)),
              "allocation": allocation.tolist()}
    for name, values in (("distortion", distortion), ("rate", rate),
                         ("objective", objective)):
        means = values.mean(1)
        winner = int(np.argmin(means[1:]))
        result[name] = {"base": float(means[0]),
                        "best": float(means[winner + 1]),
                        "gain": float(means[0] - means[winner + 1]),
                        "best_move": moves[winner],
                        "locally_optimal": bool(
                            means[winner + 1] >= means[0])}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
