"""N15: rebuild the three-mode menu on A0's rotation.

Plan v11 section 4.  A0 gives one codebook -- the uniform mode's -- and one
rotation.  The down- and up-modes do not exist in an ORFC checkpoint at all, so
they are fitted here, on ``train-fit`` features rotated by A0's own ``U``.

Two asymmetries are deliberate and are the opposite of v9's choice, for a reason
that is specific to warm starting:

*The uniform mode's codebook is copied, not refitted.*  v9 discarded its OPQ
warm-up's codebooks and refitted all three modes independently, so that no mode
began with a maturity the others lacked.  That was right for v9, where the point
was a symmetric comparison among three modes of one object.  It is wrong here:
v11's claim is "better than this ORFC checkpoint", and a refitted uniform mode
would no longer *be* that checkpoint.  W-I2 requires ``array_equal`` against
A0's codebooks, so the copy is the invariant, not a shortcut.

*The side modes are therefore weaker than the uniform mode at step 0.*  They get
a k-means fit on 3500 images; the uniform mode got ORFC's full training.  This is
recorded, not corrected.  Correcting it -- by pretraining the side modes against
the tail -- would be handing the non-uniform arm an advantage the uniform arm
never received, and A1 vs A2 would stop being a clean contrast.  What keeps the
side modes usable is the menu-coverage auxiliary during training (``qhard``), not
a better initialisation here.

The consequence to keep in view when reading N16: at step 0 the outer search is
being offered side modes that are worse-than-ORFC quality.  A one-bit swap that
would pay off under matched training may not pay off here.  N16 measures that
rather than assuming either way.
"""

import argparse
import json
import time

import numpy as np
import torch

from opq import batch_normalize_gpu
from codec_v1 import FeatureCodecV1, save_codec_v1
from cayley import DirectOrthogonalTransform
from multimode_pq import MultiModeSoftPQ

from . import config as C
from . import qhard
from .. import kmeans


def load_train_fit_vectors(device, batch=64, max_images=None, log=print):
    """Normalised train-fit features flattened to ``[N, D]`` on the GPU.

    Mirrors ``kmeans.load_training_vectors`` but goes through v11's five-way
    split, so it cannot reach cal, dev or holdout: ``config.load_split`` refuses
    holdout outright and this function names train_fit explicitly.
    """
    feature_path, _, rows, names = C.load_split("train_fit")
    if max_images is not None:
        rows, names = rows[:max_images], names[:max_images]
    array = np.load(feature_path, mmap_mode="r")
    out = []
    for start in range(0, len(rows), batch):
        block = np.asarray(array[rows[start:start + batch]])
        x = torch.from_numpy(block).float().to(device)
        y, _, _ = batch_normalize_gpu(x, mode=C.NORM_MODE)
        out.append(y.reshape(-1, y.shape[-1]))
        del x
    return torch.cat(out), names, feature_path


def build(anchor, device, images=None, log=print):
    started = time.time()
    baseline = json.loads(C.baseline_path(anchor).read_text())
    reference = np.load(C.anchor_dir(anchor) / "a0.npz")
    rotation = np.ascontiguousarray(reference["R"], dtype=np.float32)
    a0_codebooks = np.ascontiguousarray(reference["codebooks"],
                                        dtype=np.float32)
    log(f"[{anchor.name}] A0 = {baseline['A0']['stem']}  "
        f"dev {baseline['A0']['dev_mean']:.2f}")

    width = C.GROUPS * C.DIM
    transform = DirectOrthogonalTransform(width).to(device)
    transform.rotation.data.copy_(torch.from_numpy(rotation).to(device))
    rotation_t = transform.get_rotation().detach()

    y, names, feature_path = load_train_fit_vectors(
        device, max_images=images, log=log)
    log(f"[{anchor.name}] train-fit vectors {tuple(y.shape)} from "
        f"{len(names)} images")

    pq = MultiModeSoftPQ(C.GROUPS, anchor.mode_sizes, C.DIM).to(device)
    per_mode = []
    for mode, (bits, size) in enumerate(zip(anchor.mode_bits,
                                            anchor.mode_sizes)):
        if mode == anchor.uniform_mode:
            centroids = torch.from_numpy(a0_codebooks).to(device)
            origin = "copied from A0 (W-I2)"
        else:
            generator = torch.Generator(device=device).manual_seed(
                C.CODEBOOK_SEED + mode)
            centroids = kmeans.kmeans_plusplus(
                y, rotation_t, C.GROUPS, C.DIM, size, generator, device)
            centroids = kmeans.lloyd(y, rotation_t, centroids,
                                     C.CODEBOOK_KMEANS_ITERS, C.GROUPS, C.DIM)
            origin = "k-means++/Lloyd on A0-rotated train-fit"
        pq.quantizers[mode].codebooks.data.copy_(centroids)
        if hasattr(pq.quantizers[mode], "log_prior"):
            pq.quantizers[mode].log_prior.data.zero_()

        mse = kmeans.quantisation_mse(y, rotation_t, centroids, C.GROUPS,
                                      C.DIM)
        dead = kmeans.dead_fraction(y, rotation_t, centroids, C.GROUPS, C.DIM)
        per_mode.append({"bits": bits, "K": size, "origin": origin,
                         "rotated_mse": mse,
                         "dead_fraction_max": float(dead.max()),
                         "dead_fraction_mean": float(dead.mean())})
        log(f"[{anchor.name}] mode {bits}b (K={size:3d})  rotated mse "
            f"{mse:.6f}  dead max {dead.max():.4f}   [{origin}]")

    codec = FeatureCodecV1(pq, transform).to(device)

    # W-I1 / W-I2 at the point of construction, before anything is written.
    installed_R = codec.transform.get_rotation().detach().cpu().numpy()
    if not np.array_equal(installed_R, rotation):
        raise SystemExit(
            f"INVALID_EXPERIMENT: [{anchor.name}] W-I1 failed at build time: "
            f"the installed rotation differs from A0's by "
            f"{np.abs(installed_R - rotation).max():.3e}")
    installed_book = codec.pq.quantizers[
        anchor.uniform_mode].codebooks.detach().cpu().numpy()
    if not np.array_equal(installed_book, a0_codebooks):
        raise SystemExit(
            f"INVALID_EXPERIMENT: [{anchor.name}] W-I2 failed at build time: "
            f"the uniform-mode codebook differs from A0's by "
            f"{np.abs(installed_book - a0_codebooks).max():.3e}")

    root = C.menu_dir(anchor)
    root.mkdir(parents=True, exist_ok=True)
    output = root / "codec.pt"
    save_codec_v1(codec.eval(), output)
    save_reference(anchor, codec)

    identity = torch.eye(width, device=device, dtype=rotation_t.dtype)
    orth = float((rotation_t.t() @ rotation_t - identity).norm())

    rng = np.random.default_rng(C.MENU_PERMUTATION_SEED)
    permutation = rng.permutation(C.GROUPS)
    cycle = C.menu_cycle_allocations(anchor, permutation)
    counts = qhard.coverage_counts(torch.from_numpy(cycle))

    meta = {
        "plan": "v11", "node": "N15", "anchor": anchor.name,
        "rate": anchor.rate, "uniform_bits": anchor.uniform_bits,
        "mode_bits": list(anchor.mode_bits),
        "mode_sizes": list(anchor.mode_sizes),
        "uniform_mode": anchor.uniform_mode,
        "checkpoint_id": f"v11/{anchor.name}/menu/{output.name}",
        "A0": {"stem": baseline["A0"]["stem"],
               "dev_mean": baseline["A0"]["dev_mean"],
               "reference": baseline["A0"]["reference"]},
        "block": C.BLOCK, "layer": C.LAYER, "norm_mode": C.NORM_MODE,
        "split": "train_fit", "images": len(names),
        "vectors": int(y.shape[0]), "feature_cache": str(feature_path),
        "codebooks": {"seed": C.CODEBOOK_SEED, "init": "kmeans++",
                      "kmeans_iters": C.CODEBOOK_KMEANS_ITERS,
                      "uniform_mode_copied_from_A0": True,
                      "side_modes_task_pretrained": False,
                      "per_mode": per_mode},
        "orthogonality": {
            "error": orth,
            "per_direction": orth / float(np.sqrt(width)),
            "per_direction_tolerance": C.ORFC_ORTH_PER_DIRECTION_TOL,
            "training_drift_tolerance": C.ORTH_TOL,
            "note": "W-I5 as restated in N14 section 3: the inherited value is "
                    "recorded and gated per direction; ORTH_TOL applies to the "
                    "increase over this value during training."},
        "menu_cycle": {
            "seed": C.MENU_PERMUTATION_SEED,
            "permutation": permutation.tolist(),
            "allocations": cycle.tolist(),
            "nominal_rates": [int(_nominal(a, anchor)) for a in cycle],
            "coverage_min": int(counts.min()),
            "coverage_max": int(counts.max()),
            "beta": C.AUX_BETA},
        "allow_tf32": C.ALLOW_TF32,
        "seconds": time.time() - started,
    }
    if meta["menu_cycle"]["coverage_min"] != 1 or \
            meta["menu_cycle"]["coverage_max"] != 1:
        raise SystemExit(
            f"INVALID_EXPERIMENT: [{anchor.name}] the menu cycle covers cells "
            f"between {counts.min()} and {counts.max()} times, not exactly "
            f"once; G3's mechanical half is false by construction")
    if any(r != anchor.rate for r in meta["menu_cycle"]["nominal_rates"]):
        raise SystemExit(
            f"INVALID_EXPERIMENT: [{anchor.name}] auxiliary nominal rates "
            f"{meta['menu_cycle']['nominal_rates']} != {anchor.rate}")

    (root / "init_menu.json").write_text(json.dumps(meta, indent=2))
    log(f"[{anchor.name}] saved {output}  ({meta['seconds'] / 60:.1f} min)")
    return meta


def _nominal(allocation, anchor):
    from .. import engine
    return engine.nominal_rate(np.asarray(allocation), anchor)


def save_reference(anchor, codec):
    """``U_0`` and every codebook, next to the checkpoint.

    Same convention as v9's ``frozen.save_reference`` -- no hashes, direct
    key-tensor comparison -- but written under v11's own menu directory so that
    v9's frozen object is not touched.
    """
    arrays = {"U0": codec.transform.get_rotation().detach().cpu().numpy()}
    for mode, quantizer in enumerate(codec.pq.quantizers):
        arrays[f"codebook_{mode}"] = quantizer.codebooks.detach().cpu().numpy()
    np.savez(C.menu_dir(anchor) / "codec_ref.npz", **arrays)


def load_checked(anchor, device):
    """Load the frozen v11 menu, asserting it still matches its reference."""
    from codec_v1 import load_codec_v1

    root = C.menu_dir(anchor)
    path = root / "codec.pt"
    meta = json.loads((root / "init_menu.json").read_text())
    codec = load_codec_v1(path, device=device).eval()

    if meta["anchor"] != anchor.name or meta["rate"] != anchor.rate:
        raise SystemExit(f"INVALID_EXPERIMENT: {path} carries metadata for "
                         f"{meta['anchor']}/R{meta['rate']}, expected "
                         f"{anchor.name}/R{anchor.rate}")
    if list(meta["mode_bits"]) != list(anchor.mode_bits):
        raise SystemExit(f"INVALID_EXPERIMENT: {path} menu {meta['mode_bits']} "
                         f"!= frozen menu {list(anchor.mode_bits)}")

    reference = np.load(root / "codec_ref.npz")
    rotation = codec.transform.get_rotation().detach().cpu().numpy()
    if not np.array_equal(rotation, reference["U0"]):
        raise SystemExit(
            f"INVALID_EXPERIMENT: U_0 in {path} differs from its reference "
            f"(max |d| = {np.abs(rotation - reference['U0']).max():.3e})")
    for mode, quantizer in enumerate(codec.pq.quantizers):
        book = quantizer.codebooks.detach().cpu().numpy()
        want = reference[f"codebook_{mode}"]
        if not np.array_equal(book, want):
            raise SystemExit(
                f"INVALID_EXPERIMENT: codebook {mode} in {path} differs from "
                f"its reference (max |d| = {np.abs(book - want).max():.3e})")
    return codec, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor", required=True, choices=list(C.ANCHOR_BY_NAME))
    ap.add_argument("--images", type=int, default=None,
                    help="debug only; the protocol uses all of train-fit")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    anchor = C.ANCHOR_BY_NAME[args.anchor]
    output = C.menu_dir(anchor) / "codec.pt"
    if output.exists() and not args.force:
        raise SystemExit(f"{output} exists; pass --force to overwrite")

    from .. import engine
    engine.configure_precision(C.ALLOW_TF32)
    meta = build(anchor, torch.device("cuda"), images=args.images)
    print(json.dumps({k: v for k, v in meta.items()
                      if k != "menu_cycle"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
