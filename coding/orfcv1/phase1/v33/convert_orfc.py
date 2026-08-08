"""Convert an ORFC SoftPQ ``.pt`` into a v1 MultiMode codec for orfcv1 training.

Preserves the Cayley chart: ``AnchoredCayleyTransform`` with identity base and
the ORFC ``triu_params``, so ``orfc_adam`` can continue from the same chart.
The uniform-mode codebook is copied from ORFC; other modes are left at zero
(never selected under ``--uniform``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cayley import AnchoredCayleyTransform
from codec_v1 import FeatureCodecV1, save_codec_v1
from multimode_pq import MultiModeSoftPQ

from ..v12 import config as C
from ..v12 import train as joint
from .. import engine, frozen
from .. import tail as tail_mod


def convert(pt_path, anchor, out_dir, device="cuda"):
    pt_path = Path(pt_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    state = blob.get("state_dict", blob)
    meta = {k: v for k, v in blob.items() if k != "state_dict"} if isinstance(
        blob, dict) else {}

    books = state["pq.codebooks"].detach().float()
    triu = state["transform.triu_params"].detach().float()
    g, k, d = books.shape
    if (g, d) != (C.GROUPS, C.DIM):
        raise SystemExit(
            f"INVALID: codebooks shape {(g, k, d)} != "
            f"({C.GROUPS}, K, {C.DIM})")
    expected_k = int(anchor.mode_sizes[anchor.uniform_mode])
    if k != expected_k:
        raise SystemExit(
            f"INVALID: ORFC K={k} != uniform K={expected_k}")

    pq = MultiModeSoftPQ(
        C.GROUPS, anchor.mode_sizes, C.DIM,
        lmbda=float(meta.get("lmbda", 0.0)),
        prior_floor=float(meta.get("prior_floor", 0.0)))
    for mode, quantizer in enumerate(pq.quantizers):
        if mode == anchor.uniform_mode:
            quantizer.codebooks.data.copy_(books)
        else:
            quantizer.codebooks.data.zero_()

    transform = AnchoredCayleyTransform(C.GROUPS * C.DIM)
    # Identity base + ORFC triu => same Cayley chart as OrthogonalTransform.
    transform.base_rotation.copy_(torch.eye(transform.D))
    transform.triu_params.data.copy_(triu)

    codec = FeatureCodecV1(pq, transform).to(device).eval()
    rotation = codec.transform.get_rotation().detach()
    identity = torch.eye(rotation.shape[0], device=rotation.device)
    orth = float((rotation.t() @ rotation - identity).norm())

    allocation = torch.as_tensor(
        engine.uniform_allocation(anchor), dtype=torch.long)
    save_codec_v1(codec, out_dir / "codec.pt")
    np.save(out_dir / "allocation.npy", allocation.cpu().numpy())

    report = {
        "source_pt": str(pt_path.resolve()),
        "anchor": anchor.name,
        "uniform_mode": int(anchor.uniform_mode),
        "orfc_meta": {str(a): (float(b) if isinstance(b, (int, float)) else str(b))
                      for a, b in meta.items()
                      if a not in ("state_dict",)},
        "orthogonality_error": orth,
        "allocation": allocation.tolist(),
        "transform": "AnchoredCayleyTransform(identity_base)+orfc_triu",
    }
    (out_dir / "convert.json").write_text(json.dumps(report, indent=2))
    return codec, allocation, report


@torch.no_grad()
def evaluate_matched500(codec, allocation, device="cuda"):
    tail = tail_mod.build_tail(C.LAYER, device)
    val_paths = C.load_split("train_val")
    val = engine.ResidentSet(*val_paths[:3], device)
    hard_mse = joint.validate(codec, tail, val, allocation.to(device))
    orth = frozen.orthogonality_error(codec)
    return {
        "images": int(val.count),
        "tail_mse_mean": float(hard_mse),
        "orthogonality_error": float(orth),
        "split": "train_val",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt", required=True)
    parser.add_argument("--anchor", default="R64", choices=list(C.ANCHOR_BY_NAME))
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval", action="store_true",
                        help="also score matched500 via orfcv1 validate")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    anchor = C.ANCHOR_BY_NAME[args.anchor]
    codec, allocation, report = convert(args.pt, anchor, args.out, device)
    if args.eval:
        scores = evaluate_matched500(codec, allocation, device)
        report["matched500"] = scores
        Path(args.out, "matched500.json").write_text(
            json.dumps({
                "kind": "orfc_converted",
                "checkpoint": report["source_pt"],
                "images": scores["images"],
                "tail_mse_mean": scores["tail_mse_mean"],
                "orthogonality_error": scores["orthogonality_error"],
            }, indent=2))
        print(json.dumps(scores, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
