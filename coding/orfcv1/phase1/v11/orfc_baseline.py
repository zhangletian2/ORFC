"""N14: choose A0, the ORFC checkpoint v11 warm-starts from.

Plan v11 section 3.  Per anchor, every legal same-rate ORFC checkpoint is
evaluated on dev-500 under hard nearest neighbour with the full ViT tail at the
uniform allocation, and the argmin becomes A0.  Nothing is trained here.

Three things about this node are worth stating because they are choices, not
defaults:

*A0 is selected per rate, and that is the symmetric rule.*  R64's strongest
checkpoint is a 300-epoch run with no entropy term while R96's is a 100-epoch run
with ``lambda = 0.5``.  That is not an asymmetry in the protocol: the protocol is
"the strongest ORFC at this rate", applied identically at both.  The matched
configuration (same recipe at both rates) is evaluated too and reported as a
descriptive row, so the reader can see what the matched choice would have cost.

*ECVQ cannot enter here.*  Not by exclusion but by construction: the only
assignment rule in this code path is ``engine.build_bank``'s cdist argmin, which
has no lambda bias, no prior, and no rate term.  An entropy-constrained
assignment is not reachable from here.  It belongs to the real-entropy-rate
stage, where it can be compared against a real rANS rate rather than against a
fixed nominal one.

*The normalisation is checked, not assumed.*  Neither the ``.npz`` nor the
``.pt`` stores ``norm_mode`` or ``layer``; both are derived (see
``config.ORFC_PROVENANCE_NOTE``).  A derived fact that decides whether the whole
node is meaningful should not be left as a comment, so the relative quantisation
error in the rotated feature space is measured for every candidate and gated: if
the ORFC codebooks had been fitted to differently-normalised features they would
explain less of the signal than they miss, and the run stops.
"""

import argparse
import json
import time

import numpy as np
import torch

from codec_v1 import FeatureCodecV1
from cayley import DirectOrthogonalTransform
from multimode_pq import MultiModeSoftPQ

from . import config as C
from .. import engine
from .. import tail as tail_mod


# ----------------------------------------------------------- the candidates ---
def legal_candidates(anchor):
    """Re-derive the legal pool from disk and require the frozen list.

    The frozen literal in ``config.ORFC_CANDIDATES`` is the pre-registered pool.
    This function reconstructs it from the directory under the stated rule and
    fails if the two disagree, so a checkpoint that appears or vanishes later
    forces a human decision instead of silently changing what "all legal
    same-rate checkpoints" means.
    """
    k = C.UNIFORM_K[anchor.name]
    prefix = f"{C.BLOCK}_K{k}_emb{C.DIM}_"
    derived = []
    for path in sorted(C.ORFC_CHECKPOINT_DIR.glob(f"{prefix}*.npz")):
        stem = path.stem
        if not (C.ORFC_CHECKPOINT_DIR / f"{stem}.pt").exists():
            continue
        derived.append(stem)

    frozen = list(C.ORFC_CANDIDATES[anchor.name])
    if derived != sorted(frozen):
        raise SystemExit(
            f"INVALID_EXPERIMENT: [{anchor.name}] the legal pool on disk\n"
            f"  {derived}\n"
            f"differs from the pre-registered pool\n"
            f"  {sorted(frozen)}\n"
            f"The candidate list is frozen before evaluation (plan section 3); "
            f"changing it now would be choosing the pool after seeing it.")
    if not frozen:
        raise SystemExit(f"INVALID_EXPERIMENT: [{anchor.name}] no legal "
                         f"same-rate ORFC checkpoint exists")
    return frozen


def _rotation_from_state_dict(state, dim, device):
    """Recompute R through ORFC's own OrthogonalTransform.

    The ``.npz`` stores ``R`` directly; the ``.pt`` stores the Cayley parameters
    it was solved from.  Rebuilding one from the other is a cross-check of the
    two files against each other that uses no hash -- consistent with the
    standing convention here (checkpoint id + metadata + key-tensor comparison),
    and strictly more informative, since a mismatch reports which tensor moved
    and by how much.
    """
    from soft_pq import OrthogonalTransform

    sub = {key[len("transform."):]: value for key, value in state.items()
           if key.startswith("transform.")}
    if not sub:
        return None
    try:
        transform = OrthogonalTransform(dim)
    except TypeError:
        transform = OrthogonalTransform(D=dim)
    transform.load_state_dict(sub)
    transform = transform.to(device).eval()
    with torch.no_grad():
        return transform.get_rotation().detach()


def load_orfc_codec(anchor, stem, device, log=print):
    """Build a v11 three-mode codec sitting exactly on one ORFC checkpoint.

    The rotation is copied straight into ``DirectOrthogonalTransform.rotation``.
    It is emphatically *not* installed through ``init_from_opq``: that helper
    reorthogonalises via a float64 SVD, which would silently move every entry of
    R and make W-I1's ``array_equal(U_init, R_A0)`` false in N15.  v11 is fp32
    throughout and takes ORFC's rotation as it is, then measures its
    orthogonality rather than manufacturing it.

    Modes other than the uniform one are filled with exact zeros.  A0 is only
    ever evaluated at the uniform allocation, so they are never selected; making
    them zero means that if some later change ever did select one, the distortion
    would explode visibly instead of looking plausible.
    """
    npz_path = C.ORFC_CHECKPOINT_DIR / f"{stem}.npz"
    pt_path = C.ORFC_CHECKPOINT_DIR / f"{stem}.pt"
    archive = np.load(npz_path, allow_pickle=False)
    rotation = np.ascontiguousarray(archive["R"], dtype=np.float32)
    codebooks = np.ascontiguousarray(archive["codebooks"], dtype=np.float32)
    pmf = np.ascontiguousarray(archive["pmf"], dtype=np.float32)
    source = str(archive["source_pt"]) if "source_pt" in archive else ""

    dim = C.GROUPS * C.DIM
    k = C.UNIFORM_K[anchor.name]
    if rotation.shape != (dim, dim):
        raise SystemExit(f"INVALID_EXPERIMENT: {stem} R has shape "
                         f"{rotation.shape}, expected {(dim, dim)}")
    if codebooks.shape != (C.GROUPS, k, C.DIM):
        raise SystemExit(f"INVALID_EXPERIMENT: {stem} codebooks have shape "
                         f"{codebooks.shape}, expected "
                         f"{(C.GROUPS, k, C.DIM)}")

    # --- cross-check the .npz against the .pt, without hashing either --------
    blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
    meta = blob.get("meta", {}) if isinstance(blob, dict) else {}
    cross = {"pt_meta": {str(a): _jsonable(b) for a, b in meta.items()}}

    book_pt = state.get("pq.codebooks")
    if book_pt is None:
        raise SystemExit(f"INVALID_EXPERIMENT: {pt_path} has no pq.codebooks")
    book_pt = book_pt.detach().cpu().numpy().astype(np.float32)
    cross["codebooks_array_equal"] = bool(np.array_equal(book_pt, codebooks))
    cross["codebooks_max_abs_delta"] = float(np.abs(book_pt - codebooks).max())
    if not cross["codebooks_array_equal"]:
        raise SystemExit(
            f"INVALID_EXPERIMENT: {stem} codebooks in the .npz and the .pt "
            f"differ (max |d| = {cross['codebooks_max_abs_delta']:.3e}); the "
            f"two files do not describe the same checkpoint")

    recomputed = _rotation_from_state_dict(state, dim, torch.device("cpu"))
    if recomputed is None:
        cross["R_recomputed"] = False
    else:
        fresh = recomputed.numpy().astype(np.float32)
        difference = fresh - rotation
        relative = float(np.linalg.norm(difference)
                         / max(np.linalg.norm(rotation), 1e-12))
        cross["R_recomputed"] = True
        cross["R_max_abs_delta"] = float(np.abs(difference).max())
        cross["R_relative_frobenius"] = relative
        cross["R_relative_tolerance"] = C.ORFC_R_RELATIVE_TOL
        # Both R's are fp32 solves of the same ill-conditioned Cayley system, so
        # the gate is calibrated from that solve's floor and is scale-free; see
        # config.ORFC_R_RELATIVE_TOL for the measurements behind the number.
        if relative > C.ORFC_R_RELATIVE_TOL:
            raise SystemExit(
                f"INVALID_EXPERIMENT: {stem} R rebuilt from the .pt's Cayley "
                f"parameters differs from the .npz by a relative Frobenius "
                f"deviation of {relative:.3e} > {C.ORFC_R_RELATIVE_TOL:.1e}; "
                f"this is far above the fp32 solve floor and indicates the two "
                f"files do not describe the same checkpoint")

    log_prior = state.get("pq.log_prior")
    if log_prior is not None:
        # Recorded, never gated.  The two disagree by ~0.08 (R64) / ~0.03 (R96),
        # which is far too large to be arithmetic: the .npz's ``pmf`` is the
        # empirical codeword-usage histogram, not the softmax of the learned
        # prior.  Nothing in v11 reads either -- this stage has no rate term at
        # all -- so the number is provenance, not a check.
        recovered = torch.softmax(log_prior.detach().cpu().float(),
                                  dim=-1).numpy()
        cross["pmf_max_abs_delta"] = float(np.abs(recovered - pmf).max())
        cross["pmf_note"] = ("empirical usage histogram vs softmax(log_prior); "
                             "descriptive only, unused in v11")

    # --- build the three-mode codec at this rotation ------------------------
    pq = MultiModeSoftPQ(C.GROUPS, anchor.mode_sizes, C.DIM).to(device)
    for mode, quantizer in enumerate(pq.quantizers):
        if mode == anchor.uniform_mode:
            quantizer.codebooks.data.copy_(
                torch.from_numpy(codebooks).to(device))
        else:
            quantizer.codebooks.data.zero_()
        if hasattr(quantizer, "log_prior"):
            quantizer.log_prior.data.zero_()

    transform = DirectOrthogonalTransform(dim).to(device)
    transform.rotation.data.copy_(torch.from_numpy(rotation).to(device))
    codec = FeatureCodecV1(pq, transform).to(device).eval()

    installed = codec.transform.get_rotation().detach().cpu().numpy()
    if not np.array_equal(installed, rotation):
        raise SystemExit(
            f"INVALID_EXPERIMENT: installing R into DirectOrthogonalTransform "
            f"changed it (max |d| = {np.abs(installed - rotation).max():.3e}); "
            f"the warm start would not be the ORFC checkpoint")

    orth = orthogonality_error(codec)
    provenance = {"stem": stem, "npz": str(npz_path), "pt": str(pt_path),
                  "source_pt": source, "uniform_K": k,
                  "cross_check": cross,
                  "orthogonality_error": orth,
                  "orthogonality_per_direction": orth / float(np.sqrt(dim)),
                  "orthogonality_per_direction_tolerance":
                      C.ORFC_ORTH_PER_DIRECTION_TOL,
                  "training_drift_tolerance": C.ORTH_TOL,
                  "note": C.ORFC_PROVENANCE_NOTE}
    return codec, rotation, codebooks, provenance


def _jsonable(value):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def orthogonality_error(codec):
    """``||U^T U - I||_F`` in fp32 -- v11 uses no float64 anywhere.

    The fp32 accumulation floor on a 1024x1024 matrix was measured at 1.08e-05
    against a tolerance of 1e-4, and the reading is the true drift in quadrature
    with that floor, so the extra precision buys nothing.  See
    ``config.ORTH_MEASURE_FLOOR``.
    """
    rotation = codec.transform.get_rotation().detach()
    identity = torch.eye(rotation.shape[0], device=rotation.device,
                         dtype=rotation.dtype)
    return float((rotation.t() @ rotation - identity).norm())


# ------------------------------------------------------------- measurement ---
@torch.no_grad()
def relative_quantisation_error(codec, resident, anchor, images=100):
    """``||z - z_hat||^2 / ||z||^2`` in the rotated space at the uniform mode.

    The one measurable consequence of a normalisation mismatch: codebooks fitted
    to differently-scaled features stop explaining the signal.  Reported for
    every candidate and gated at 1.0, which is not a tuned threshold but the
    point at which the codebook misses more than it captures.
    """
    stop = min(int(images), resident.count)
    y = resident.y[:stop]
    banks, rotation = engine.build_bank(codec, y)
    z = (y.reshape(-1, y.shape[-1]) @ rotation).reshape(
        -1, C.GROUPS, C.DIM).permute(1, 0, 2)
    z_hat = banks[anchor.uniform_mode]
    residual = float((z - z_hat).square().sum())
    energy = float(z.square().sum())
    return residual / energy if energy > 0 else float("inf")


@torch.no_grad()
def evaluate_on_dev(codec, tail, dev, anchor):
    """Per-image hard-NN full-tail distortion at the uniform allocation."""
    allocation = engine.uniform_allocation(anchor)[None, :]
    if engine.nominal_rate(allocation[0], anchor) != anchor.rate:
        raise SystemExit(f"INVALID_EXPERIMENT: the uniform allocation has "
                         f"nominal rate "
                         f"{engine.nominal_rate(allocation[0], anchor)}, not "
                         f"{anchor.rate}")
    matrix = engine.evaluate_allocations(
        codec, tail, dev, allocation, image_batch=C.EVAL_IMAGE_BATCH,
        pair_budget=C.EVAL_PAIR_BUDGET, per_image=True)
    return np.asarray(matrix[0], dtype=np.float64)


@torch.no_grad()
def projection_cost(anchor, rotation, codebooks, tail, dev, device):
    """What the stored rotation's non-orthogonality is worth, in the metric.

    ``||R^T R - I||_F`` is 5.05e-03 at R64 and 1.40e-03 at R96 -- above
    ``ORTH_TOL``, and a number that invites the question "so should it be
    projected?".  Arguing that from the norm alone would be arguing from a form.
    This measures the consequence instead: the same codebooks are evaluated once
    at the stored ``R`` and once at its QR projection onto the orthogonal
    manifold, on the same dev images at the same allocation.

    The projected variant is reported and then discarded.  A0 is the stored
    checkpoint, because a baseline we had repaired is not the baseline anyone
    would compare against; the point of the measurement is to know the size of
    what we are declining to repair, not to choose between them.
    """
    dim = rotation.shape[0]
    q, r = torch.linalg.qr(torch.from_numpy(rotation).to(device))
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)     # fix the QR sign gauge

    pq = MultiModeSoftPQ(C.GROUPS, anchor.mode_sizes, C.DIM).to(device)
    for mode, quantizer in enumerate(pq.quantizers):
        if mode == anchor.uniform_mode:
            quantizer.codebooks.data.copy_(
                torch.from_numpy(codebooks).to(device))
        else:
            quantizer.codebooks.data.zero_()
        if hasattr(quantizer, "log_prior"):
            quantizer.log_prior.data.zero_()
    transform = DirectOrthogonalTransform(dim).to(device)
    transform.rotation.data.copy_(q)
    codec = FeatureCodecV1(pq, transform).to(device).eval()

    per_image = evaluate_on_dev(codec, tail, dev, anchor)
    identity = torch.eye(dim, device=q.device, dtype=q.dtype)
    report = {
        "projected_orthogonality_error": float((q.t() @ q - identity).norm()),
        "max_abs_rotation_shift": float(
            (q - torch.from_numpy(rotation).to(device)).abs().max()),
        "projected_dev_mean": float(per_image.mean()),
    }
    del codec
    torch.cuda.empty_cache()
    return report


# ---------------------------------------------------------------- the node ---
def run(anchor, device, log=print):
    started = time.time()
    engine.configure_precision(C.ALLOW_TF32)
    out = C.anchor_dir(anchor)
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates").mkdir(exist_ok=True)

    manifest = C.freeze_trainval_split()
    log(f"[{anchor.name}] split frozen: "
        + ", ".join(f"{k} {v['count']}"
                    for k, v in manifest["sets"].items()))

    candidates = legal_candidates(anchor)
    log(f"[{anchor.name}] {len(candidates)} legal same-rate ORFC checkpoints "
        f"(K={C.UNIFORM_K[anchor.name]}), frozen before evaluation")

    tail = tail_mod.build_tail(C.LAYER, device)
    feature_path, teacher_path, rows, basenames = C.load_split("dev")
    tail_mod.check_layer(C.LAYER, feature_path, teacher_path)
    dev = engine.ResidentSet(feature_path, teacher_path, rows, device)
    engine.assert_uniform_shape(dev.count)
    log(f"[{anchor.name}] dev resident: {dev.count} images")

    table = []
    for stem in candidates:
        step = time.time()
        codec, rotation, codebooks, provenance = load_orfc_codec(
            anchor, stem, device, log=log)
        # ORFC's rotation is orthogonal by construction and only approximately
        # so in fp32 arithmetic: ||A||_max reaches 121, so I - A is badly scaled
        # and the solve loses digits.  Requiring ORTH_TOL here would reject every
        # candidate and leave v11 with no warm start.  The gate is therefore on
        # the per-direction consequence -- see config.ORFC_ORTH_PER_DIRECTION_TOL
        # for the restatement of W-I5 and the measured numbers.
        if provenance["orthogonality_per_direction"] > \
                C.ORFC_ORTH_PER_DIRECTION_TOL:
            raise SystemExit(
                f"INVALID_EXPERIMENT: [{anchor.name}] {stem} has "
                f"||U^T U - I||_F / sqrt(D) = "
                f"{provenance['orthogonality_per_direction']:.3e} > "
                f"{C.ORFC_ORTH_PER_DIRECTION_TOL:.1e}; its singular values are "
                f"further than 1% from unity, so it is no longer a rotation in "
                f"any useful sense")
        qerror = relative_quantisation_error(codec, dev, anchor)
        if qerror > C.ORFC_MAX_RELATIVE_QERROR:
            raise SystemExit(
                f"INVALID_EXPERIMENT: [{anchor.name}] {stem} has relative "
                f"quantisation error {qerror:.4f} > "
                f"{C.ORFC_MAX_RELATIVE_QERROR}; the codebooks do not fit these "
                f"features, which is what a normalisation or layer mismatch "
                f"looks like.  norm_mode and layer are derived, not stored -- "
                f"see config.ORFC_PROVENANCE_NOTE.")
        per_image = evaluate_on_dev(codec, tail, dev, anchor)
        np.save(out / "candidates" / f"{stem}.npy", per_image)
        entry = {"stem": stem, "dev_mean": float(per_image.mean()),
                 "dev_std": float(per_image.std(ddof=1)),
                 "dev_min": float(per_image.min()),
                 "dev_max": float(per_image.max()),
                 "relative_quantisation_error": qerror,
                 "orthogonality_error": provenance["orthogonality_error"],
                 "orthogonality_per_direction":
                     provenance["orthogonality_per_direction"],
                 "cross_check": provenance["cross_check"],
                 "source_pt": provenance["source_pt"],
                 "seconds": time.time() - step}
        table.append(entry)
        log(f"[{anchor.name}] {stem}\n"
            f"    dev mean {entry['dev_mean']:.2f}   "
            f"rel q-error {qerror:.5f}   "
            f"orth {entry['orthogonality_error']:.2e}   "
            f"({entry['seconds']:.1f}s)")
        del codec
        torch.cuda.empty_cache()

    # argmin with the pre-registered tie-break: lexicographic on the stem.
    best = min(table, key=lambda e: (e["dev_mean"], e["stem"]))
    ties = [e["stem"] for e in table if e["dev_mean"] == best["dev_mean"]]

    codec, rotation, codebooks, provenance = load_orfc_codec(
        anchor, best["stem"], device, log=log)
    np.savez(out / "a0.npz", R=rotation, codebooks=codebooks)
    del codec
    torch.cuda.empty_cache()

    projection = projection_cost(anchor, rotation, codebooks, tail, dev, device)
    projection["stored_dev_mean"] = best["dev_mean"]
    projection["delta_dev_mean"] = (projection["projected_dev_mean"]
                                    - best["dev_mean"])
    projection["relative_delta"] = (projection["delta_dev_mean"]
                                    / best["dev_mean"])
    log(f"[{anchor.name}] orthogonality of the stored R: "
        f"{provenance['orthogonality_error']:.3e} "
        f"({provenance['orthogonality_per_direction']:.2e} per direction).  "
        f"QR-projecting it moves dev by "
        f"{projection['delta_dev_mean']:+.2f} "
        f"({100 * projection['relative_delta']:+.4f}%); A0 stays as stored.")

    matched = C.ORFC_MATCHED_CONFIG[anchor.name]
    payload = {
        "plan": "v11", "node": "N14", "anchor": anchor.name,
        "rate": anchor.rate, "uniform_K": C.UNIFORM_K[anchor.name],
        "metric": C.A0_METRIC, "tie_break": C.A0_TIE_BREAK,
        "candidate_dir": str(C.ORFC_CHECKPOINT_DIR),
        "candidates_frozen": list(candidates),
        "table": sorted(table, key=lambda e: e["dev_mean"]),
        "A0": {"stem": best["stem"], "dev_mean": best["dev_mean"],
               "reference": str(out / "a0.npz"),
               "provenance": provenance},
        "ties_at_argmin": ties,
        "orthogonality": {
            "stored": provenance["orthogonality_error"],
            "per_direction": provenance["orthogonality_per_direction"],
            "per_direction_tolerance": C.ORFC_ORTH_PER_DIRECTION_TOL,
            "training_drift_tolerance": C.ORTH_TOL,
            "qr_projection_diagnostic": projection,
            "decision": "A0 is the stored checkpoint, unprojected.  ORFC's "
                        "Cayley rotation is exactly orthogonal in exact "
                        "arithmetic and only approximately so in fp32 at "
                        "||A||_max ~ 121, which is a property of the object v11 "
                        "is asked to beat.  Projecting it would make the "
                        "baseline a codec ORFC never produced.  W-I5 therefore "
                        "records the initial value and gates it per direction; "
                        "ORTH_TOL keeps its full force on training drift, "
                        "measured as the increase over this initial value.",
        },
        "matched_configuration": {
            "stem": matched,
            "dev_mean": next(e["dev_mean"] for e in table
                             if e["stem"] == matched),
            "is_A0": matched == best["stem"],
            "note": "descriptive only; A0 is chosen per rate by the same rule "
                    "at both anchors, which is the symmetric protocol"},
        "ecvq": "not evaluated here by construction: this code path's only "
                "assignment rule is engine.build_bank's cdist argmin, which "
                "carries no lambda bias.  The strongest ECVQ result belongs to "
                "the real-entropy-rate stage.",
        "dev_images": int(dev.count),
        "dev_first_basename": basenames[0],
        "dev_last_basename": basenames[-1],
        "provenance_note": C.ORFC_PROVENANCE_NOTE,
        "seconds": time.time() - started,
    }
    C.baseline_path(anchor).write_text(json.dumps(payload, indent=2))

    spread = max(e["dev_mean"] for e in table) - best["dev_mean"]
    log(f"[{anchor.name}] A0 = {best['stem']}  dev {best['dev_mean']:.2f}  "
        f"(worst candidate is {spread:.2f} higher, "
        f"{100 * spread / best['dev_mean']:+.3f}%)")
    log(f"[{anchor.name}] N14 done in {payload['seconds'] / 60:.1f} min")
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor", required=True, choices=list(C.ANCHOR_BY_NAME))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    anchor = C.ANCHOR_BY_NAME[args.anchor]
    if C.baseline_path(anchor).exists() and not args.force:
        raise SystemExit(
            f"{C.baseline_path(anchor)} exists; A0 is frozen.  Re-selecting it "
            f"after seeing a downstream result would be choosing the warm start "
            f"from an outcome.")
    payload = run(anchor, torch.device("cuda"))
    print(json.dumps({k: v for k, v in payload.items() if k != "table"},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
