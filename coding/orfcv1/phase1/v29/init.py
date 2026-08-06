"""Build identity-transform, independently k-means-fitted multi-mode PQ."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from cayley import AnchoredCayleyTransform
from codec_v1 import FeatureCodecV1, save_codec_v1
from multimode_pq import MultiModeSoftPQ
from ..v15.init import _audit, _fit
from ..v21.config import SPECS, activate


@torch.no_grad()
def build(block, anchor_name, out, device, images=None):
    started = time.time()
    config = activate(block)
    anchor = config.ANCHOR_BY_NAME[anchor_name]
    from ..v12 import init as shared
    y, count = shared._vectors(device, max_images=images)
    width = config.GROUPS * config.DIM
    identity = torch.eye(width, device=device)
    transform = AnchoredCayleyTransform(width).to(device)
    transform.init_from_opq(identity)
    for parameter in transform.parameters():
        parameter.requires_grad_(False)
    pq = MultiModeSoftPQ(
        config.GROUPS, anchor.mode_sizes, config.DIM).to(device)
    modes = []
    for mode, (bits, size) in enumerate(zip(
            anchor.mode_bits, anchor.mode_sizes)):
        sub, book = _fit(
            identity, y, config.GROUPS, config.DIM, size,
            config.CODEBOOK_KMEANS_ITERS,
            config.CODEBOOK_SEED + mode, device)
        mse, dead = _audit(sub, book)
        pq.quantizers[mode].codebooks.copy_(book)
        modes.append({"bits": bits, "K": size, "feature_mse": mse,
                      "dead_fraction_max": dead})
        del sub, book
    if any(modes[i + 1]["feature_mse"] > modes[i]["feature_mse"]
           for i in range(len(modes) - 1)):
        raise SystemExit("INVALID_EXPERIMENT: non-monotone k-means menu")
    codec = FeatureCodecV1(pq, transform).eval()
    target = Path(out)
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"{target} is non-empty")
    target.mkdir(parents=True, exist_ok=True)
    save_codec_v1(codec, target / "codec.pt")
    payload = {
        "plan": "v29_fixed_identity_all_modes_kmeans",
        "block": block, "anchor": anchor.name, "nominal_rate": anchor.rate,
        "mode_bits": list(anchor.mode_bits), "images": count,
        "all_modes_kmeans": True, "rotation": "fixed_identity",
        "rotation_max_abs_from_identity": float(
            (codec.transform.get_rotation() - identity).abs().max()),
        "modes": modes, "seconds": time.time() - started}
    (target / "init.json").write_text(json.dumps(payload, indent=2))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", default="blk20", choices=tuple(SPECS))
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--images", type=int)
    args = parser.parse_args(argv)
    print(json.dumps(build(
        args.block, args.anchor, args.out, torch.device(args.device),
        args.images), indent=2))


if __name__ == "__main__":
    main()
