"""Convert the frozen OPQ/k-means initialisation to an ORFC-rate codec."""

import argparse
import json
import time

import numpy as np
import torch

from codec_v1 import load_codec_v1, save_codec_v1
from multimode_pq import MultiModeSoftPQ
from opq import batch_normalize_gpu

from .config import FAIR_LOSS_ALPHA, PRIOR_FLOOR, RATE_LAMBDA, SPECS, activate


@torch.no_grad()
def build(block, anchor, device, batch=16):
    started = time.time()
    C = activate(block)
    source_plan = C.PHASE1 / ("v15/blk05" if block == "blk05" else "v12")
    source = source_plan / "init_orfc_adam" / anchor.name / "codec.pt"
    old = load_codec_v1(source, device=device)
    pq = MultiModeSoftPQ(C.GROUPS, anchor.mode_sizes, C.DIM,
                         lmbda=RATE_LAMBDA, prior_floor=PRIOR_FLOOR).to(device)
    for target, original in zip(pq.quantizers, old.pq.quantizers):
        target.codebooks.copy_(original.codebooks)
    codec = type(old)(pq, old.transform).to(device).eval()

    counts = [torch.zeros(q.G, q.K, device=device) for q in pq.quantizers]
    source_features = np.load(C.TRAIN_FEATURES, mmap_mode="r")
    rotation = codec.transform.get_rotation()
    for first in range(0, C.N_TRAIN, int(batch)):
        x = torch.from_numpy(np.array(
            source_features[first:first + int(batch)], copy=True)).float().to(device)
        y, _, _ = batch_normalize_gpu(x, mode=C.NORM_MODE)
        z = y.reshape(-1, y.shape[-1]) @ rotation
        sub = z.reshape(-1, C.GROUPS, C.DIM).permute(1, 0, 2)
        for table, quantizer in zip(counts, pq.quantizers):
            labels = torch.cdist(sub, quantizer.codebooks).square().argmin(-1)
            table.scatter_add_(1, labels, torch.ones_like(labels, dtype=table.dtype))
    for quantizer, table in zip(pq.quantizers, counts):
        quantizer.init_prior_from_freq(table, smoothing=1.0)

    root = C.init_dir(anchor, "orfc_cayley")
    root.mkdir(parents=True, exist_ok=True)
    save_codec_v1(codec, root / "codec.pt")
    arrays = {"U0": codec.transform.get_rotation().cpu().numpy()}
    arrays.update({f"codebook_{m}": q.codebooks.cpu().numpy()
                   for m, q in enumerate(pq.quantizers)})
    arrays.update({f"log_prior_{m}": q.log_prior.cpu().numpy()
                   for m, q in enumerate(pq.quantizers)})
    np.savez(root / "codec_ref.npz", **arrays)
    payload = {
        "plan": C.PLAN, "stage": "reuse_opq_kmeans_init_plus_empirical_priors",
        "transform_parameterization": "orfc_cayley", "block": block,
        "anchor": anchor.name, "rate": anchor.rate,
        "mode_bits": list(anchor.mode_bits), "images": C.N_TRAIN,
        "all_modes_kmeans": True, "source_codec": str(source),
        "lmbda": RATE_LAMBDA, "prior_floor": PRIOR_FLOOR,
        "fair_loss_alpha": FAIR_LOSS_ALPHA,
        "prior_smoothing": 1.0,
        "prior_dead_entries": [int((table == 0).sum()) for table in counts],
        "seconds": time.time() - started}
    (root / "init.json").write_text(json.dumps(payload, indent=2))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", required=True, choices=tuple(SPECS))
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args(argv)
    C = activate(args.block)
    if args.anchor not in C.ANCHOR_BY_NAME:
        parser.error(f"unknown anchor {args.anchor!r}")
    target = C.init_dir(C.ANCHOR_BY_NAME[args.anchor], "orfc_cayley") / "codec.pt"
    if target.exists():
        raise SystemExit(f"{target} exists; refusing to overwrite")
    print(json.dumps(build(args.block, C.ANCHOR_BY_NAME[args.anchor],
                           torch.device(args.device), args.batch), indent=2))


if __name__ == "__main__":
    main()
