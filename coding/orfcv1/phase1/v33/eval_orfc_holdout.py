"""Score a converted ORFC ``codec_v1`` on the val-cache hold-out rows.

Companion to :mod:`phase1.v33.eval_alloc`, which only speaks V33 checkpoints.
Both walk the same rows through the same tail, so the numbers are comparable.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from codec_v1 import load_codec_v1
from opq import batch_inv_normalize_gpu

from .. import engine, tail as tail_mod
from ..v12 import config as C
from ..v21.config import SPECS, activate

from . import valset


@torch.no_grad()
def score_rows(codec, tail, rows, allocation, device, chunk=500,
               image_batch=16):
    from ..v12 import qhard

    pieces = []
    for first in range(0, len(rows), int(chunk)):
        resident = valset.resident_from_rows(
            device, rows[first:first + int(chunk)])
        for start in range(0, resident.count, int(image_batch)):
            y, mu, std, teacher = resident.slice(start, start + image_batch)
            decoded = qhard.quantise(codec, y, allocation[None])[0]
            output = tail(batch_inv_normalize_gpu(decoded, mu, std))
            pieces.append((output - teacher).square()
                          .reshape(y.shape[0], -1).sum(-1).cpu().numpy())
        del resident
        torch.cuda.empty_cache()
    return np.concatenate(pieces)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", default="blk20", choices=tuple(SPECS))
    parser.add_argument("--anchor", default="R64")
    parser.add_argument("--codec", required=True, help="codec_v1 .pt path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--chunk", type=int, default=500)
    parser.add_argument("--image-batch", type=int, default=16)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    activate(args.block)
    engine.configure_precision(C.ALLOW_TF32)
    device = torch.device(args.device)
    anchor = C.ANCHOR_BY_NAME[args.anchor]
    tail = tail_mod.build_tail(C.LAYER, device)
    codec = load_codec_v1(args.codec, device=device).eval()

    rows = (valset.holdout_rows() if args.holdout
            else valset.fixed_val_rows())
    allocation = torch.as_tensor(
        engine.uniform_allocation(anchor, C.GROUPS), device=device)
    values = score_rows(
        codec, tail, rows, allocation, device,
        chunk=args.chunk, image_batch=args.image_batch)

    report = {
        "codec": args.codec,
        "split": (f"val_cache[{int(rows[0])}:{int(rows[-1]) + 1}] (holdout)"
                  if args.holdout else "train_val"),
        "images": int(values.size),
        "tail_mse_mean": float(values.mean()),
        "tail_mse_sem": float(values.std(ddof=1) / np.sqrt(values.size)),
    }
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
