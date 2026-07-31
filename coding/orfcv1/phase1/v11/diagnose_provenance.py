"""Diagnostic for N16's null result: provenance, or ORFC's U?

N16 measured 0 improving candidates out of 992 at both anchors, with the best
one-bit swap 2.9% (R64) / 2.5% (R96) *worse* than uniform on cal.  That is a sign
flip against v9, which found -1.600% / -1.227% in the same neighbourhood with its
own frozen ``(U_0, Theta_0)``.  Two explanations are on the table and they lead
to opposite next steps:

  (P) **Provenance.**  N15's menu has an asymmetry forced by W-I2: mode 1 is
      A0's own task-trained codebook, modes 0 and 2 are plain k-means.  A swap
      moves two groups off the good codebook and onto two weaker ones, so it
      pays the provenance gap twice and the bit budget is never the binding
      term.  If this is the cause, N16's null says nothing about non-uniform
      allocation and everything about the initialisation, and the fix is the one
      already in the design -- the menu-coverage auxiliary of N18.

  (U) **The rotation.**  ORFC's U may genuinely have equalised the per-group
      distortion profile, leaving no group cheap enough to give a bit up.  If
      this is the cause, the non-uniform hypothesis is in trouble at this
      working point regardless of how the codebooks are fitted.

The two are separated by removing the asymmetry and nothing else.  This module
builds a codec on **the same A0 rotation** with **all three modes fitted the same
way** -- k-means++/Lloyd on A0-rotated train-fit -- and runs the identical
992-candidate cal sweep.  Everything else, including the searcher, the split, the
evaluation batch and the tie-break, is what N16 used.

What this is not:

* It is **not a selection.**  Nothing it writes can become A0, A1 or A2; the
  frozen menu codec is untouched and this codec is written to its own directory.
* It **does not read dev or holdout.**  cal-500 only -- the same set N16's search
  already saw, so no acceptance power is spent.
* Its outcome **cannot by itself re-open N16.**  N16's gate is pre-registered and
  has returned an empty eligible set.  Reopening it is a change to the
  pre-registration, which is the plan owner's call, not this module's.  What this
  module supplies is the evidence that decision would be made on.

Read it as a two-cell contrast on the fraction of improving candidates:

    uniform mode task-trained (N16)     0/992      0/992
    uniform mode k-means, all matched   ?          ?

A large positive count in row 2 is evidence for (P).  A zero in row 2 is evidence
for (U), and is the stronger result of the two -- it would say the one-bit
neighbourhood at ORFC's rotation is uphill in every direction even when nothing
is confounded.
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
from . import init_menu
from . import search as search_mod
from .. import engine
from .. import kmeans
from .. import tail as tail_mod


def build_matched(anchor, device, images=None, log=print):
    """A0's rotation, all three modes fitted identically by k-means."""
    reference = np.load(C.anchor_dir(anchor) / "a0.npz")
    rotation = np.ascontiguousarray(reference["R"], dtype=np.float32)
    a0_codebooks = np.ascontiguousarray(reference["codebooks"],
                                        dtype=np.float32)

    width = C.GROUPS * C.DIM
    transform = DirectOrthogonalTransform(width).to(device)
    transform.rotation.data.copy_(torch.from_numpy(rotation).to(device))
    rotation_t = transform.get_rotation().detach()

    y, names, _ = init_menu.load_train_fit_vectors(device, max_images=images,
                                                   log=log)
    pq = MultiModeSoftPQ(C.GROUPS, anchor.mode_sizes, C.DIM).to(device)
    per_mode = []
    for mode, (bits, size) in enumerate(zip(anchor.mode_bits,
                                            anchor.mode_sizes)):
        # The same seed offset per mode as init_menu, so modes 0 and 2 are the
        # *same* codebooks N16 used; only mode 1 changes.  That keeps the
        # contrast to one variable.
        generator = torch.Generator(device=device).manual_seed(
            C.CODEBOOK_SEED + mode)
        centroids = kmeans.kmeans_plusplus(y, rotation_t, C.GROUPS, C.DIM,
                                           size, generator, device)
        centroids = kmeans.lloyd(y, rotation_t, centroids,
                                 C.CODEBOOK_KMEANS_ITERS, C.GROUPS, C.DIM)
        pq.quantizers[mode].codebooks.data.copy_(centroids)
        if hasattr(pq.quantizers[mode], "log_prior"):
            pq.quantizers[mode].log_prior.data.zero_()
        mse = kmeans.quantisation_mse(y, rotation_t, centroids, C.GROUPS,
                                      C.DIM)
        entry = {"bits": bits, "K": size, "rotated_mse": mse,
                 "origin": "k-means++/Lloyd on A0-rotated train-fit"}
        if mode == anchor.uniform_mode:
            a0_mse = kmeans.quantisation_mse(
                y, rotation_t, torch.from_numpy(a0_codebooks).to(device),
                C.GROUPS, C.DIM)
            entry["a0_rotated_mse"] = a0_mse
            entry["a0_over_kmeans"] = a0_mse / mse
        per_mode.append(entry)
        log(f"[{anchor.name}] mode {bits}b (K={size:3d})  rotated mse "
            f"{mse:.4f}" + (f"   (A0's was {entry['a0_rotated_mse']:.4f}, "
                            f"{entry['a0_over_kmeans']:.3f}x)"
                            if mode == anchor.uniform_mode else ""))
    del y
    torch.cuda.empty_cache()
    return FeatureCodecV1(pq, transform).to(device), per_mode


def sweep(codec, tail, cal, anchor, log=print):
    """The identical 992-candidate one-bit sweep N16 ran, on cal."""
    uniform = engine.uniform_allocation(anchor)
    candidates, pairs = search_mod.legal_swaps(uniform, anchor)
    started = time.time()
    matrix = engine.evaluate_allocations(
        codec, tail, cal, np.vstack([uniform[None], candidates]),
        image_batch=C.EVAL_IMAGE_BATCH, pair_budget=C.EVAL_PAIR_BUDGET,
        per_image=True)
    means = np.asarray(matrix, dtype=np.float64).mean(1)
    best = int(np.argmin(means[1:]))
    return matrix, {
        "candidates": int(means.size - 1),
        "uniform_cal_mean": float(means[0]),
        "best_cal_mean": float(means[1:].min()),
        "best_relative_gain": float((means[0] - means[1:].min()) / means[0]),
        "best_down_group": int(pairs[best, 0]),
        "best_up_group": int(pairs[best, 1]),
        "improving_candidates": int(np.sum(means[1:] < means[0])),
        "seconds": time.time() - started,
    }


def run(anchor, device, images=None, log=print):
    started = time.time()
    engine.configure_precision(C.ALLOW_TF32)
    out = C.anchor_dir(anchor) / "diagnostic_provenance"
    out.mkdir(parents=True, exist_ok=True)

    codec, per_mode = build_matched(anchor, device, images=images, log=log)
    tail = tail_mod.build_tail(C.LAYER, device)
    feature_path, teacher_path, rows, _ = C.load_split("cal")
    cal = engine.ResidentSet(feature_path, teacher_path, rows, device,
                             max_images=images)
    engine.assert_uniform_shape(cal.count)

    matrix, matched = sweep(codec, tail, cal, anchor, log=log)
    np.save(out / "cal_distortion_matched.npy",
            np.asarray(matrix, dtype=np.float32))

    n16 = json.loads((C.bridge_dir(anchor) / "bridge.json").read_text())
    payload = {
        "plan": "v11", "node": "N16-diagnostic", "anchor": anchor.name,
        "rate": anchor.rate, "split": "cal", "images": int(cal.count),
        "question": "is N16's 0/992 caused by codebook provenance or by "
                    "ORFC's rotation?",
        "matched_menu": matched,
        "n16_menu": n16["round1"],
        "codebooks": per_mode,
        "contrast": {
            "improving_n16": int(n16["round1"]["improving_candidates"]),
            "improving_matched": int(matched["improving_candidates"]),
            "best_gain_n16": float(n16["round1"]["best_relative_gain"]),
            "best_gain_matched": float(matched["best_relative_gain"]),
        },
        "not_a_selection": (
            "This codec cannot become A0, A1 or A2.  N16's gate is "
            "pre-registered and has already returned an empty eligible set; "
            "reopening it is the plan owner's decision, and this file is the "
            "evidence that decision would rest on, not the decision."),
        "seconds": time.time() - started,
    }
    (out / "diagnostic.json").write_text(json.dumps(payload, indent=2))
    log(f"[{anchor.name}] matched-provenance menu: "
        f"{matched['improving_candidates']}/{matched['candidates']} candidates "
        f"improve (N16's task-trained menu: "
        f"{n16['round1']['improving_candidates']}/992); best "
        f"{100 * matched['best_relative_gain']:+.4f}% "
        f"(N16 {100 * n16['round1']['best_relative_gain']:+.4f}%)  "
        f"({payload['seconds'] / 60:.1f} min)")
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor", required=True, choices=list(C.ANCHOR_BY_NAME))
    ap.add_argument("--images", type=int, default=None, help="debug only")
    args = ap.parse_args(argv)
    payload = run(C.ANCHOR_BY_NAME[args.anchor], torch.device("cuda"),
                  images=args.images)
    print(json.dumps(payload["contrast"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
