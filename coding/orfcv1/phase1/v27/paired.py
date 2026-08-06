"""Paired held-out comparison of a block-allocation codec and its control."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from codec_v1 import load_codec_v1
from phase1 import engine, tail as tail_mod
from phase1.v21.config import activate
from phase1.v22.probe import evaluate, split_resident


def summary(control, block, seed):
    delta = np.asarray(block) - np.asarray(control)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(20):
        index = rng.integers(0, len(delta), size=(1000, len(delta)))
        draws.append(delta[index].mean(1))
    draws = np.concatenate(draws)
    return {"mean_delta": float(delta.mean()),
            "relative_delta": float(delta.mean() / np.mean(control)),
            "bootstrap_ci95": np.quantile(draws, (0.025, 0.975)).tolist(),
            "block_better_count": int((delta < 0).sum()),
            "images": int(len(delta))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offset", type=int, default=256)
    parser.add_argument("--images", type=int, default=240)
    args = parser.parse_args(argv)
    config = activate("blk20")
    device = torch.device(args.device)
    rows = config.load_split("train_val")[2]
    resident = split_resident(
        config, rows, args.offset, args.images, device)
    tail = tail_mod.build_tail(config.LAYER, device)
    values = {}
    for name, root in (("control", Path(args.control)),
                       ("block", Path(args.block))):
        codec = load_codec_v1(root / "codec.pt", device=device).eval()
        allocation = np.load(root / "allocation.npy")[None]
        values[name] = evaluate(
            codec, tail, resident, allocation, 16, 0.5)
        del codec
    result = {"anchor": args.anchor, "offset": args.offset,
              "images": args.images}
    for index, name in enumerate(("distortion", "rate", "objective")):
        result[name] = summary(
            values["control"][index][0], values["block"][index][0],
            20260806 + index)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
