"""Build V15 OPQ initialisations; high-K paths remain chunked on GPU."""

import argparse
import json
import time

import numpy as np
import torch

from cayley import AnchoredCayleyTransform, DirectOrthogonalTransform
from codec_v1 import FeatureCodecV1, save_codec_v1
from multimode_pq import MultiModeSoftPQ
from opq import batched_kmeans

from .config import SPECS, activate


def _fit(rotation, y, groups, dim, k, iterations, seed, device):
    torch.manual_seed(int(seed))
    z = y @ rotation
    sub = z.reshape(-1, groups, dim).permute(1, 0, 2).contiguous()
    del z
    book = batched_kmeans(
        sub, k, max_iter=iterations, device=device, verbose=False)
    return sub, book


@torch.no_grad()
def _audit(sub, book, chunk=8192):
    groups, total, _ = sub.shape
    counts = torch.zeros(groups, book.shape[1], device=sub.device)
    error = 0.0
    for first in range(0, total, chunk):
        block = sub[:, first:first + chunk]
        distance = torch.cdist(block, book).square()
        labels = distance.argmin(-1)
        error += float(distance.amin(-1).sum())
        counts.scatter_add_(1, labels, torch.ones_like(labels, dtype=counts.dtype))
    return error / total, float((counts == 0).float().mean(1).max())


@torch.no_grad()
def _opq(y, groups, dim, k, outer, inner, seed, device, log):
    width = groups * dim
    rotation = torch.eye(width, device=device)
    history, best_mse, best_rotation = [], float("inf"), rotation.clone()
    for step in range(int(outer)):
        sub, book = _fit(
            rotation, y, groups, dim, k, inner, seed + step, device)
        moment = torch.zeros(width, width, dtype=torch.float64, device=device)
        error = 0.0
        for first in range(0, y.shape[0], 8192):
            block = sub[:, first:first + 8192]
            distance = torch.cdist(block, book).square()
            labels = distance.argmin(-1)
            error += float(distance.amin(-1).sum())
            hat = torch.gather(
                book, 1, labels.unsqueeze(-1).expand(-1, -1, dim))
            hat = hat.permute(1, 0, 2).reshape(-1, width)
            moment += (y[first:first + len(hat)].t() @ hat).double()
        left, _, right = torch.linalg.svd(moment)
        rotation = (left @ right).float()
        mse = error / y.shape[0]
        history.append(mse)
        if mse < best_mse:
            best_mse, best_rotation = mse, rotation.clone()
        log(f"  OPQ {step + 1}/{outer}: mse={mse:.6f}")
        del sub, book, moment
    return best_rotation, history


def build(anchor, device, max_images=None, parameterization="orfc_cayley", log=print):
    C = activate("blk20")
    from ..v12 import init as old
    started = time.time()
    y, images = old._vectors(device, max_images=max_images)
    rotation, history = _opq(
        y, C.GROUPS, C.DIM, 2 ** anchor.uniform_bits,
        C.OPQ_ITERS, C.OPQ_KMEANS_ITERS, C.OPQ_SEED, device, log)
    identity = torch.eye(C.GROUPS * C.DIM, dtype=torch.float64, device=device)
    opq_orth = float((rotation.double().t() @ rotation.double() - identity).norm())
    if opq_orth > C.OPQ_ORTH_TOL:
        raise SystemExit(f"INVALID_EXPERIMENT: OPQ orthogonality {opq_orth:.3e}")
    if parameterization == "orfc_cayley":
        transform = AnchoredCayleyTransform(C.GROUPS * C.DIM).to(device)
    elif parameterization == "direct":
        transform = DirectOrthogonalTransform(C.GROUPS * C.DIM).to(device)
    else:
        raise ValueError(parameterization)
    transform.init_from_opq(rotation)
    fit_rotation = transform.get_rotation().detach()
    effective_orth = float(
        (fit_rotation.double().t() @ fit_rotation.double() - identity).norm())
    tolerance = (C.CAYLEY_ORTH_ABS_TOL if parameterization == "orfc_cayley"
                 else C.OPQ_ORTH_TOL)
    if effective_orth > tolerance:
        raise SystemExit(
            f"INVALID_EXPERIMENT: effective orthogonality {effective_orth:.3e}")
    pq = MultiModeSoftPQ(C.GROUPS, anchor.mode_sizes, C.DIM).to(device)
    modes = []
    for mode, (bits, size) in enumerate(zip(anchor.mode_bits, anchor.mode_sizes)):
        sub, book = _fit(
            fit_rotation, y, C.GROUPS, C.DIM, size,
            C.CODEBOOK_KMEANS_ITERS, C.CODEBOOK_SEED + mode, device)
        mse, dead = _audit(sub, book)
        pq.quantizers[mode].codebooks.data.copy_(book)
        modes.append({"bits": bits, "K": size, "rotated_mse": mse,
                      "dead_fraction_max": dead})
        log(f"[{anchor.name}] mode={bits}b K={size} mse={mse:.6f} dead={dead:.4f}")
        del sub, book
    if any(modes[i + 1]["rotated_mse"] > modes[i]["rotated_mse"]
           for i in range(len(modes) - 1)):
        raise SystemExit("INVALID_EXPERIMENT: non-monotone initial PQ MSE")
    codec = FeatureCodecV1(pq, transform).eval()
    root = C.init_dir(anchor, parameterization)
    root.mkdir(parents=True, exist_ok=True)
    save_codec_v1(codec, root / "codec.pt")
    arrays = {"U0": codec.transform.get_rotation().detach().cpu().numpy()}
    arrays.update({f"codebook_{m}": q.codebooks.detach().cpu().numpy()
                   for m, q in enumerate(codec.pq.quantizers)})
    np.savez(root / "codec_ref.npz", **arrays)
    meta = {"plan": "v15", "stage": "opq_u0_all_modes_batched_kmeans",
            "transform_parameterization": parameterization,
            "anchor": anchor.name, "rate": anchor.rate,
            "mode_bits": list(anchor.mode_bits), "images": images,
            "vectors": int(y.shape[0]), "opq_warmup_K": 2 ** anchor.uniform_bits,
            "opq_iters": C.OPQ_ITERS, "opq_history": history,
            "opq_codebooks_reused": False, "all_modes_kmeans": True,
            "orthogonality": effective_orth,
            "opq_orthogonality": opq_orth,
            "effective_orthogonality": effective_orth,
            "modes": modes, "seconds": time.time() - started}
    (root / "init.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", required=True, choices=tuple(SPECS))
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--images", type=int, default=None)
    parser.add_argument("--parameterization", choices=("direct", "orfc_cayley"),
                        default="orfc_cayley")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    C = activate(args.block)
    if args.anchor not in C.ANCHOR_BY_NAME:
        parser.error(f"{args.anchor} is not defined for {args.block}")
    anchor = C.ANCHOR_BY_NAME[args.anchor]
    target = C.init_dir(anchor, args.parameterization) / "codec.pt"
    if target.exists() and not args.force:
        raise SystemExit(f"{target} exists; pass --force to replace")
    if args.block == "blk05":
        from ..v12 import init as old
        result = old.build(anchor, torch.device(args.device), args.images,
                           args.parameterization)
        result["plan"] = "v15"
    else:
        result = build(anchor, torch.device(args.device), args.images,
                       args.parameterization)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
