"""Score checkpoints under several allocations on the val list or the hold-out.

The 500-image ``train_val`` list is the set search selects on, so a gain read
off it is a selection estimate.  ``--holdout`` scores the untouched remainder
of the same feature cache instead, which is the honest number.

Arms may mix codec families so a V33 delivery and its ORFC baseline can be
compared image-by-image on one resident set (see :func:`parse_arm`).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from codec_v1 import load_codec_v1
from opq import batch_inv_normalize_gpu
from soft_pq import load_codec as load_orfc

from .. import engine, tail as tail_mod
from ..v12 import config as C
from ..v21.config import SPECS, activate

from . import checkpoint as ckpt
from . import distortion, valset

KINDS = ("v33", "cv1", "orfc")


def parse_arm(spec):
    """``NAME=[KIND:]CHECKPOINT[:ALLOCATION]`` -> ``(name, kind, path, alloc)``.

    ``KIND`` defaults to ``v33``.  ``orfc`` is a single-codebook ``soft_pq``
    checkpoint and takes no allocation; ``cv1`` is a ``codec_v1`` payload.
    """
    name, _, target = spec.partition("=")
    if not name or not target:
        raise ValueError(f"arm {spec!r} must look like NAME=CHECKPOINT")
    kind = "v33"
    for candidate in KINDS:
        if target.startswith(candidate + ":"):
            kind, target = candidate, target[len(candidate) + 1:]
            break
    checkpoint, _, allocation = target.partition(":")
    if kind == "orfc" and allocation:
        raise ValueError(f"arm {name}: an orfc codec has no per-group modes")
    return name, kind, checkpoint, (allocation or None)


def load_arm(kind, checkpoint, allocation, anchor, device):
    """Return ``(codec, allocation_tensor_or_None, rate)``."""
    if kind == "v33":
        codec, _ = ckpt.load_checkpoint(checkpoint, device=device)
        mode_bits = codec.pq.mode_bits
    elif kind == "cv1":
        codec = load_codec_v1(checkpoint, device=device)
        mode_bits = tuple(int(round(np.log2(k))) for k in codec.pq.mode_sizes)
    else:
        codec = load_orfc(checkpoint, device=device).eval()
        return codec, None, int(codec.pq.G * round(np.log2(codec.pq.K)))

    modes = (ckpt.load_allocation(allocation, groups=C.GROUPS, device=device)
             if allocation else
             torch.as_tensor(engine.uniform_allocation(anchor, C.GROUPS),
                             device=device))
    rate = int(sum(mode_bits[int(m)] for m in modes.tolist()))
    return codec, modes, rate


def _row_chunks(rows, chunk):
    for first in range(0, len(rows), int(chunk)):
        yield rows[first:first + int(chunk)]


@torch.no_grad()
def _score_resident(codec, kind, tail, resident, allocation, image_batch):
    if kind == "v33":
        return distortion.evaluate(
            codec, tail, resident, allocation, image_batch=image_batch)[0]
    modes = None if allocation is None else allocation.tolist()
    pieces = []
    for start in range(0, resident.count, int(image_batch)):
        y, mu, std, teacher = resident.slice(start, start + image_batch)
        decoded, _ = (codec(y) if modes is None else codec(y, modes=modes))
        output = tail(batch_inv_normalize_gpu(decoded, mu, std))
        pieces.append((output - teacher).square()
                      .reshape(y.shape[0], -1).sum(-1).cpu().numpy())
    return np.concatenate(pieces)


@torch.no_grad()
def score_rows(codec, tail, rows, allocation, device, chunk=500,
               image_batch=16, kind="v33"):
    """Per-image tail MSE over arbitrary cache rows, loaded chunk by chunk."""
    pieces = []
    for block in _row_chunks(rows, chunk):
        resident = valset.resident_from_rows(device, block)
        pieces.append(_score_resident(
            codec, kind, tail, resident, allocation, image_batch))
        del resident
        torch.cuda.empty_cache()
    return np.concatenate(pieces)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", default="blk20", choices=tuple(SPECS))
    parser.add_argument("--anchor", default="R64")
    parser.add_argument("--arm", action="append", default=[], required=True,
                        help="repeatable NAME=[KIND:]CHECKPOINT[:ALLOCATION] "
                             f"with KIND in {KINDS} (default v33); a bare "
                             "checkpoint uses the uniform allocation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--holdout", action="store_true",
                        help="score val-cache rows past N_VAL instead")
    parser.add_argument("--images", type=int, default=None,
                        help="cap the number of scored images")
    parser.add_argument("--chunk", type=int, default=500)
    parser.add_argument("--image-batch", type=int, default=16)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    activate(args.block)
    engine.configure_precision(C.ALLOW_TF32)
    device = torch.device(args.device)
    anchor = C.ANCHOR_BY_NAME[args.anchor]
    tail = tail_mod.build_tail(C.LAYER, device)

    if args.holdout:
        rows = valset.holdout_rows()
        split = f"val_cache[{int(rows[0])}:{int(rows[-1]) + 1}] (holdout)"
    else:
        rows = valset.fixed_val_rows()
        split = "train_val (search selection set)"
    if args.images is not None:
        rows = rows[:int(args.images)]

    results, per_image = [], {}
    for spec in args.arm:
        name, kind, checkpoint, allocation_path = parse_arm(spec)
        codec, allocation, rate = load_arm(
            kind, checkpoint, allocation_path, anchor, device)
        if rate != anchor.rate:
            raise SystemExit(f"{name}: allocation rate {rate} != {anchor.rate}")
        values = score_rows(
            codec, tail, rows, allocation, device, kind=kind,
            chunk=args.chunk, image_batch=args.image_batch)
        per_image[name] = values
        results.append({
            "name": name,
            "kind": kind,
            "checkpoint": checkpoint,
            "allocation": (None if allocation is None
                           else [int(x) for x in allocation.tolist()]),
            "uses_L": bool(getattr(codec, "uses_L", False)),
            "rate": rate,
            "tail_mse_mean": float(values.mean()),
            "tail_mse_sem": float(values.std(ddof=1) / np.sqrt(values.size)),
        })
        del codec
        torch.cuda.empty_cache()

    # Paired stats against the first arm: image-level noise cancels, so this
    # is the interval that actually decides whether an arm is better.
    reference = results[0]["name"]
    for row in results[1:]:
        difference = per_image[row["name"]] - per_image[reference]
        row["paired_delta"] = float(difference.mean())
        row["paired_sem"] = float(
            difference.std(ddof=1) / np.sqrt(difference.size))
        row["paired_percent"] = 100.0 * row["paired_delta"] / float(
            per_image[reference].mean())
        row["paired_t"] = (row["paired_delta"] / row["paired_sem"]
                           if row["paired_sem"] > 0 else float("inf"))

    report = {
        "split": split,
        "images": int(len(rows)),
        "anchor": args.anchor,
        "reference": reference,
        "rows": results,
    }
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
