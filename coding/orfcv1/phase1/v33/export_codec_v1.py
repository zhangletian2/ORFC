"""Export a V33 checkpoint to ``codec_v1`` so downstream tooling can read it.

``phase1.v12.eval_downstream`` (the Acc / mIoU path used by v21 / v28) only
loads ``codec_v1``.  The conversion is exact rather than approximate: the
nested tree is a *parameterisation* of each mode's centroids, and
``NestedMultiModePQ.composed_codebook(m)`` already materialises the ``[G, K, d]``
tensor that ``MultiModeSoftPQ.quantizers[m].codebooks`` holds directly.  Group
layout, rotation, and hard-argmin assignment are identical on both sides, so
the exported codec reconstructs bit-for-bit.

Refuses to export a codec that still carries the block-diagonal ``L`` bank —
``codec_v1`` has nowhere to put it.  Fold it first with
``V33Codec.fold_L_into_codebooks(allocation)``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cayley import AnchoredCayleyTransform, DirectOrthogonalTransform
from codec_v1 import FeatureCodecV1, save_codec_v1
from multimode_pq import MultiModeSoftPQ

from . import checkpoint as ckpt


@torch.no_grad()
def to_codec_v1(codec):
    """Build an equivalent :class:`FeatureCodecV1` from a ``V33Codec``."""
    if codec.uses_L:
        raise ValueError(
            "codec still has a block-diagonal L bank; call "
            "fold_L_into_codebooks(allocation) before exporting")
    pq = codec.pq
    multimode = MultiModeSoftPQ(pq.G, pq.mode_sizes, pq.d)
    for mode, quantizer in enumerate(multimode.quantizers):
        quantizer.codebooks.data.copy_(pq.composed_codebook(mode).detach())

    source = codec.transform
    name = type(source).__name__
    if name == "AnchoredCayleyTransform":
        transform = AnchoredCayleyTransform(source.D)
    elif name == "DirectOrthogonalTransform":
        transform = DirectOrthogonalTransform(source.D)
    else:
        raise ValueError(f"unsupported transform {name!r} for codec_v1 export")
    transform.load_state_dict(source.state_dict())

    exported = FeatureCodecV1(multimode, transform)
    return exported.to(next(codec.parameters()).device).eval()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="v33_codec_v1 path")
    parser.add_argument("--out", required=True, help="codec_v1 .pt path")
    parser.add_argument("--allocation", default=None,
                        help="written alongside --out; defaults to the "
                             "allocation stored in the checkpoint meta")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    codec, payload = ckpt.load_checkpoint(args.checkpoint, device=device)
    exported = to_codec_v1(codec)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_codec_v1(exported, str(out))

    allocation = (ckpt.load_allocation(args.allocation, groups=codec.pq.G)
                  if args.allocation else ckpt.meta_allocation(payload))
    if allocation is None:
        raise SystemExit(
            "no allocation found; pass --allocation for the downstream call")
    allocation = np.asarray(
        torch.as_tensor(allocation).reshape(-1).tolist(), dtype=np.int64)
    np.save(out.with_name("allocation.npy"), allocation)

    report = {
        "source": str(Path(args.checkpoint).resolve()),
        "codec_v1": str(out.resolve()),
        "allocation_npy": str(out.with_name("allocation.npy").resolve()),
        "allocation": allocation.tolist(),
        "mode_sizes": list(codec.pq.mode_sizes),
        "nominal_rate": int(sum(
            int(codec.pq.mode_bits[m]) for m in allocation.tolist())),
    }
    out.with_suffix(".export.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
