"""Build v12's OPQ rotation and independently k-means-fit every menu mode."""

import argparse
import json
import time

import numpy as np
import torch

from cayley import DirectOrthogonalTransform
from codec_v1 import FeatureCodecV1, load_codec_v1, save_codec_v1
from multimode_pq import MultiModeSoftPQ
from opq import batch_normalize_gpu
from soft_pq import OrthogonalTransform

from . import config as C
from .. import kmeans


def _vectors(device, max_images=None, batch=64):
    count = C.N_TRAIN if max_images is None else min(int(max_images), C.N_TRAIN)
    source = np.load(C.TRAIN_FEATURES, mmap_mode="r")
    chunks = []
    for start in range(0, count, batch):
        x = torch.from_numpy(np.asarray(source[start:start + batch])).float().to(device)
        y, _, _ = batch_normalize_gpu(x, mode=C.NORM_MODE)
        chunks.append(y.reshape(-1, y.shape[-1]))
    return torch.cat(chunks), count


def build(anchor, device, max_images=None, parameterization="direct", log=print):
    started = time.time()
    y, images = _vectors(device, max_images=max_images)
    generator = torch.Generator(device=device).manual_seed(C.OPQ_SEED)
    rotation, history = kmeans.opq_warmup(
        y, C.GROUPS, C.DIM, 2 ** anchor.uniform_bits,
        C.OPQ_ITERS, C.OPQ_KMEANS_ITERS, generator, device, log=log)
    identity = torch.eye(C.GROUPS * C.DIM, device=device, dtype=torch.float64)
    orth = float((rotation.double().t() @ rotation.double() - identity).norm())
    if orth > C.OPQ_ORTH_TOL:
        raise SystemExit(f"INVALID_EXPERIMENT: OPQ orthogonality {orth:.3e}")

    if parameterization == "orfc_cayley":
        transform = OrthogonalTransform(C.GROUPS * C.DIM).to(device)
        transform.init_from_opq(rotation.detach().cpu().numpy())
    elif parameterization == "direct":
        transform = DirectOrthogonalTransform(C.GROUPS * C.DIM).to(device)
        transform.init_from_opq(rotation)
    else:
        raise ValueError(f"unknown transform parameterization {parameterization!r}")
    fit_rotation = transform.get_rotation().detach()
    effective_orth = float(
        (fit_rotation.double().t() @ fit_rotation.double() - identity).norm())
    effective_tol = (C.CAYLEY_ORTH_ABS_TOL
                     if parameterization == "orfc_cayley"
                     else C.OPQ_ORTH_TOL)
    if effective_orth > effective_tol:
        raise SystemExit(
            f"INVALID_EXPERIMENT: effective orthogonality {effective_orth:.3e}")
    rotation_conversion_rel = float(
        (fit_rotation - rotation).norm() / rotation.norm().clamp_min(1e-12))
    pq = MultiModeSoftPQ(C.GROUPS, anchor.mode_sizes, C.DIM).to(device)
    modes = []
    for mode, (bits, size) in enumerate(zip(anchor.mode_bits, anchor.mode_sizes)):
        seeded = torch.Generator(device=device).manual_seed(C.CODEBOOK_SEED + mode)
        book = kmeans.kmeans_plusplus(
            y, fit_rotation, C.GROUPS, C.DIM, size, seeded, device)
        book = kmeans.lloyd(
            y, fit_rotation, book, C.CODEBOOK_KMEANS_ITERS, C.GROUPS, C.DIM)
        pq.quantizers[mode].codebooks.data.copy_(book)
        modes.append({"bits": bits, "K": size,
                      "rotated_mse": kmeans.quantisation_mse(
                          y, fit_rotation, book, C.GROUPS, C.DIM),
                      "dead_fraction_max": float(kmeans.dead_fraction(
                          y, fit_rotation, book, C.GROUPS, C.DIM).max())})
        log(f"[{anchor.name}] k-means mode={bits}b K={size}")

    codec = FeatureCodecV1(pq, transform).eval()
    root = C.init_dir(anchor, parameterization)
    root.mkdir(parents=True, exist_ok=True)
    save_codec_v1(codec, root / "codec.pt")
    arrays = {"U0": codec.transform.get_rotation().detach().cpu().numpy()}
    arrays.update({f"codebook_{m}": q.codebooks.detach().cpu().numpy()
                   for m, q in enumerate(codec.pq.quantizers)})
    np.savez(root / "codec_ref.npz", **arrays)
    meta = {"plan": "v12", "stage": "opq_u0_all_modes_kmeans",
            "transform_parameterization": parameterization,
            "opq_to_effective_rotation_relative": rotation_conversion_rel,
            "anchor": anchor.name, "rate": anchor.rate,
            "mode_bits": list(anchor.mode_bits), "images": images,
            "vectors": int(y.shape[0]), "opq_warmup_K": 2 ** anchor.uniform_bits,
            "opq_iters": C.OPQ_ITERS, "opq_codebooks_reused": False,
            "all_modes_kmeans": True, "orthogonality": effective_orth,
            "opq_orthogonality": orth,
            "effective_orthogonality": effective_orth,
            "modes": modes, "seconds": time.time() - started}
    (root / "init.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_checked(anchor, device, parameterization="direct", require_full=True):
    root = C.init_dir(anchor, parameterization)
    meta = json.loads((root / "init.json").read_text())
    codec = load_codec_v1(root / "codec.pt", device=device)
    reference = np.load(root / "codec_ref.npz")
    tensors = {"U0": codec.transform.get_rotation().detach().cpu().numpy()}
    tensors.update({f"codebook_{m}": q.codebooks.detach().cpu().numpy()
                    for m, q in enumerate(codec.pq.quantizers)})
    for name, value in tensors.items():
        if not np.array_equal(value, reference[name]):
            raise SystemExit(f"INVALID_EXPERIMENT: v12 init tensor {name} changed")
    if meta.get("transform_parameterization", "direct") != parameterization:
        raise SystemExit("INVALID_EXPERIMENT: transform parameterization mismatch")
    if require_full and (meta["images"] != C.N_TRAIN
                         or not meta["all_modes_kmeans"]):
        raise SystemExit("INVALID_EXPERIMENT: v12 formal init is not full-5k all-kmeans")
    return codec, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", required=True, choices=list(C.ANCHOR_BY_NAME))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--images", type=int, default=None, help="debug only")
    parser.add_argument("--parameterization", choices=("direct", "orfc_cayley"),
                        default="direct")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    anchor = C.ANCHOR_BY_NAME[args.anchor]
    target = C.init_dir(anchor, args.parameterization) / "codec.pt"
    if target.exists() and not args.force:
        raise SystemExit(f"{target} exists; pass --force to replace")
    payload = build(anchor, torch.device(args.device), max_images=args.images,
                    parameterization=args.parameterization)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
