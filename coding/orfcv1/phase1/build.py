"""Build one anchor's frozen measurement object ``(U_0, Theta_0)`` (v9 section 3).

Two steps, in this order and no other:

1. OPQ warm-up on train-core at ``K = 2 ** anchor.uniform_bits`` -- the rotation
   is warmed up for the point the experiment perturbs around -- giving `U_0`.
   Orthogonality is asserted in float64, not hoped for.
2. Each of the anchor's three modes is initialised by an *independent*
   k-means++ / Lloyd run on the `U_0`-rotated train-core features.  The warm-up's
   own codebooks are discarded: reusing them would hand the uniform mode a
   maturity the down- and up-modes do not have, and every candidate in this
   round is precisely a trade between those three modes.

Nothing here reads cal, dev or measure.  After this script the object is frozen;
`cal_sweep.py` and `unseal.py` only load it.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from cayley import DirectOrthogonalTransform
from codec_v1 import FeatureCodecV1, save_codec_v1
from multimode_pq import MultiModeSoftPQ

from . import config as C
from . import frozen
from . import kmeans


def build(anchor, device, images=None, log=print):
    started = time.time()
    y, names, feature_path = kmeans.load_training_vectors(
        device=device, max_images=images or C.OPQ_IMAGES)
    log(f"[{anchor.name}] train-core vectors {tuple(y.shape)} "
        f"from {len(names)} images")

    warm_k = 2 ** anchor.uniform_bits
    generator = torch.Generator(device=device).manual_seed(C.OPQ_SEED)
    rotation, history = kmeans.opq_warmup(
        y, C.GROUPS, C.DIM, warm_k, C.OPQ_ITERS, C.OPQ_KMEANS_ITERS,
        generator, device, log=log)

    width = C.GROUPS * C.DIM
    identity = torch.eye(width, device=device, dtype=torch.float64)
    wide = rotation.double()
    orth_error = float((wide.t() @ wide - identity).norm())
    if orth_error > C.OPQ_ORTH_TOL:
        raise SystemExit(
            f"INVALID_EXPERIMENT: [{anchor.name}] ||U^T U - I||_F = "
            f"{orth_error:.3e} exceeds {C.OPQ_ORTH_TOL:.1e}")

    transform = DirectOrthogonalTransform(width).to(device)
    transform.init_from_opq(rotation)

    pq = MultiModeSoftPQ(C.GROUPS, anchor.mode_sizes, C.DIM).to(device)
    per_mode = []
    for mode, (bits, size) in enumerate(zip(anchor.mode_bits,
                                            anchor.mode_sizes)):
        seeded = torch.Generator(device=device).manual_seed(
            C.CODEBOOK_SEED + mode)
        centroids = kmeans.kmeans_plusplus(
            y, rotation, C.GROUPS, C.DIM, size, seeded, device)
        centroids = kmeans.lloyd(y, rotation, centroids,
                                 C.CODEBOOK_KMEANS_ITERS, C.GROUPS, C.DIM)
        pq.quantizers[mode].codebooks.data.copy_(centroids)
        mse = kmeans.quantisation_mse(y, rotation, centroids, C.GROUPS, C.DIM)
        dead = kmeans.dead_fraction(y, rotation, centroids, C.GROUPS, C.DIM)
        per_mode.append({"bits": bits, "K": size, "rotated_mse": mse,
                         "dead_fraction_max": float(dead.max()),
                         "dead_fraction_mean": float(dead.mean())})
        log(f"[{anchor.name}] mode {bits}b (K={size:3d})  rotated mse "
            f"{mse:.6f}  dead max {dead.max():.4f}")

    codec = FeatureCodecV1(pq, transform).to(device)
    anchor.root.mkdir(parents=True, exist_ok=True)
    output = anchor.root / "codec.pt"
    save_codec_v1(codec.eval(), output)
    frozen.save_reference(anchor, codec)

    meta = {
        "plan": "v9", "stage": "frozen_measurement_object",
        "anchor": anchor.name, "rate": anchor.rate,
        "uniform_bits": anchor.uniform_bits,
        "mode_bits": list(anchor.mode_bits),
        "mode_sizes": list(anchor.mode_sizes),
        "checkpoint_id": f"{anchor.name}/{output.name}",
        "block": C.BLOCK, "layer": C.LAYER, "norm_mode": C.NORM_MODE,
        "split": "train_core", "images": len(names),
        "vectors": int(y.shape[0]), "feature_cache": str(feature_path),
        "opq": {"warmup_K": warm_k, "iters": C.OPQ_ITERS,
                "kmeans_iters": C.OPQ_KMEANS_ITERS, "seed": C.OPQ_SEED,
                "init": "kmeans++", "mse_history": history},
        "orth_error_float64": orth_error,
        "codebooks": {"seed": C.CODEBOOK_SEED, "init": "kmeans++",
                      "kmeans_iters": C.CODEBOOK_KMEANS_ITERS,
                      "reused_opq_codebooks": False,
                      "reused_orfc_codebooks": False,
                      "task_tail_pretraining": False,
                      "per_mode": per_mode},
        "allow_tf32": C.ALLOW_TF32,
        "seconds": time.time() - started,
    }
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    log(f"[{anchor.name}] saved {output}  ({meta['seconds'] / 60:.1f} min)")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True, choices=list(C.ANCHOR_BY_NAME))
    ap.add_argument("--images", type=int, default=None,
                    help="debug only; the protocol uses all of train-core")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    anchor = C.ANCHOR_BY_NAME[args.anchor]
    output = anchor.root / "codec.pt"
    if output.exists() and not args.force:
        raise SystemExit(f"{output} exists; pass --force to overwrite")

    torch.backends.cuda.matmul.allow_tf32 = C.ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = C.ALLOW_TF32
    meta = build(anchor, torch.device("cuda"), images=args.images)
    print(json.dumps({k: v for k, v in meta.items() if k != "opq"}, indent=2))


if __name__ == "__main__":
    main()
