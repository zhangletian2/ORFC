"""W-I and G gate verification for v11.

``--stage init`` (N15) checks W-I1..W-I6; ``--stage arms`` (N18) checks G1..G4.
Nothing here trains, searches or selects, and nothing here reads holdout.

One thing worth stating up front so the report is not read as stronger than it
is: **A1@0 and A2@0 are the same file**.  Both arms warm-start from
``menu/codec.pt``; they diverge only when A2's first outer event fires.  So the
content of W-I3 and W-I4 is not "the three objects agree" -- two of them are one
object -- it is "the v11 menu codec reproduces A0 exactly at the uniform
allocation, even though two of its three modes were just fitted from scratch".
That is a real check: if fitting the side modes had perturbed the rotation or the
uniform codebook, or if mode selection were not doing what it claims, the indices
would move.  The report says which comparison is which rather than counting the
trivial one as evidence.
"""

import argparse
import json
import time

import numpy as np
import torch

from . import config as C
from . import init_menu
from . import orfc_baseline
from . import qhard
from .. import engine
from .. import tail as tail_mod


@torch.no_grad()
def hard_indices(codec, resident, allocation, image_batch=C.EVAL_IMAGE_BATCH):
    """``[G, N_images * tokens]`` hard nearest-neighbour labels."""
    modes = torch.as_tensor(allocation, dtype=torch.long,
                            device=resident.device)
    out = []
    for start in range(0, resident.count, image_batch):
        y, _, _, _ = resident.slice(start, start + image_batch)
        _, labels = qhard.quantise_hard_ste(codec, y, modes)
        out.append(labels.cpu().numpy())
    return np.concatenate(out, axis=1)


@torch.no_grad()
def per_image_distortion(codec, tail, resident, allocation):
    matrix = engine.evaluate_allocations(
        codec, tail, resident, np.asarray(allocation)[None, :],
        image_batch=C.EVAL_IMAGE_BATCH, pair_budget=C.EVAL_PAIR_BUDGET,
        per_image=True)
    return np.asarray(matrix[0], dtype=np.float64)


def verify_init(anchor, device, log=print):
    started = time.time()
    engine.configure_precision(C.ALLOW_TF32)
    results = {}
    failures = []

    def record(name, ok, payload):
        results[name] = dict(payload, passed=bool(ok))
        log(f"[{anchor.name}] {name}  {'PASS' if ok else 'FAIL'}  "
            + "  ".join(f"{k}={v}" for k, v in payload.items()
                        if not isinstance(v, (list, dict))))
        if not ok:
            failures.append(name)

    baseline = json.loads(C.baseline_path(anchor).read_text())
    reference = np.load(C.anchor_dir(anchor) / "a0.npz")
    a0_rotation = np.ascontiguousarray(reference["R"], dtype=np.float32)
    a0_codebooks = np.ascontiguousarray(reference["codebooks"],
                                        dtype=np.float32)

    codec, meta = init_menu.load_checked(anchor, device)
    rotation = codec.transform.get_rotation().detach().cpu().numpy()
    book = codec.pq.quantizers[
        anchor.uniform_mode].codebooks.detach().cpu().numpy()

    # --- W-I1 / W-I2: the warm start is the checkpoint, element for element ---
    record("W-I1", np.array_equal(rotation, a0_rotation),
           {"max_abs_delta": float(np.abs(rotation - a0_rotation).max()),
            "A0_stem": baseline["A0"]["stem"]})
    record("W-I2", np.array_equal(book, a0_codebooks),
           {"max_abs_delta": float(np.abs(book - a0_codebooks).max()),
            "uniform_mode": anchor.uniform_mode})

    # --- the three objects -------------------------------------------------
    a0_codec, _, _, a0_prov = orfc_baseline.load_orfc_codec(
        anchor, baseline["A0"]["stem"], device)
    allocation = engine.uniform_allocation(anchor)

    tail = tail_mod.build_tail(C.LAYER, device)
    dev_feat, dev_teach, dev_rows, dev_names = C.load_split("dev")
    tail_mod.check_layer(C.LAYER, dev_feat, dev_teach)
    dev = engine.ResidentSet(dev_feat, dev_teach, dev_rows, device)

    fit_feat, fit_teach, fit_rows, fit_names = C.load_split("train_fit")
    tail_mod.check_layer(C.LAYER, fit_feat, fit_teach)
    probe = engine.ResidentSet(fit_feat, fit_teach, fit_rows[:C.WI_PROBE_IMAGES],
                               device)

    # --- W-I3: identical hard indices, element for element ------------------
    index_report = {}
    ok3 = True
    for label, resident, names in (("dev", dev, dev_names),
                                   ("train_fit_probe", probe, fit_names)):
        mine = hard_indices(codec, resident, allocation)
        theirs = hard_indices(a0_codec, resident, allocation)
        equal = np.array_equal(mine, theirs)
        differing = int((mine != theirs).sum())
        index_report[label] = {
            "images": int(resident.count), "cells": int(mine.size),
            "array_equal": bool(equal), "differing_cells": differing,
            "first_basename": names[0],
            "last_basename": names[min(resident.count, len(names)) - 1]}
        ok3 = ok3 and equal
    record("W-I3", ok3, {"comparison": "A0 vs A1@0 (== A2@0)",
                         **{f"{k}_differing": v["differing_cells"]
                            for k, v in index_report.items()}})
    results["W-I3"]["detail"] = index_report

    # --- W-I4: per-image tail distortion within the replay tolerance --------
    mine = per_image_distortion(codec, tail, dev, allocation)
    theirs = per_image_distortion(a0_codec, tail, dev, allocation)
    scale = float(np.abs(theirs).mean())
    max_rel = float(np.abs(mine - theirs).max() / scale)
    record("W-I4", max_rel <= C.REPLAY_REL_TOL,
           {"max_relative_gap": max_rel, "tolerance": C.REPLAY_REL_TOL,
            "A1_mean": float(mine.mean()), "A0_mean": float(theirs.mean()),
            "A0_dev_mean_from_N14": baseline["A0"]["dev_mean"]})

    # --- W-I5: recorded, gated per direction (see N14 section 3) ------------
    width = C.GROUPS * C.DIM
    tensor = codec.transform.get_rotation().detach()
    identity = torch.eye(width, device=tensor.device, dtype=tensor.dtype)
    orth = float((tensor.t() @ tensor - identity).norm())
    per_direction = orth / float(np.sqrt(width))
    record("W-I5", per_direction <= C.ORFC_ORTH_PER_DIRECTION_TOL,
           {"orthogonality_error": orth, "per_direction": per_direction,
            "per_direction_tolerance": C.ORFC_ORTH_PER_DIRECTION_TOL,
            "training_drift_tolerance": C.ORTH_TOL,
            "restated_in": "N14 section 3"})

    # --- W-I6: one layer, one tail, one normalisation -----------------------
    consistent = (meta["layer"] == C.LAYER == 20
                  and meta["norm_mode"] == C.NORM_MODE
                  and meta["block"] == C.BLOCK)
    record("W-I6", consistent,
           {"layer": meta["layer"], "block": meta["block"],
            "norm_mode": meta["norm_mode"],
            "tail": f"blocks[{C.LAYER + 1}:]",
            "checked_caches": "dev and train_fit via tail.check_layer"})

    # --- the auxiliary cannot steer U (measured, not argued) ----------------
    y, mu, std, teacher = probe.slice(0, C.EVAL_IMAGE_BATCH)
    rng = np.random.default_rng(C.MENU_PERMUTATION_SEED)
    cycle = C.menu_cycle_allocations(anchor, rng.permutation(C.GROUPS))
    aux_modes = torch.as_tensor(cycle[1], dtype=torch.long, device=device)
    grad_norm = qhard.assert_rotation_gradient_free(
        codec, tail, y, mu, std, teacher, aux_modes)
    record("AUX-SG", grad_norm == 0.0,
           {"rotation_grad_norm": grad_norm,
            "note": "dD_aux/dU must be exactly zero, so the auxiliary branch "
                    "cannot compete with the main branch for U"})

    payload = {"plan": "v11", "node": "N15", "stage": "init",
               "anchor": anchor.name, "rate": anchor.rate,
               "A0": baseline["A0"]["stem"],
               "A0_orthogonality": a0_prov["orthogonality_error"],
               "menu_checkpoint": meta["checkpoint_id"],
               "results": results, "failures": failures,
               "passed": not failures,
               "dev_images": int(dev.count),
               "probe_images": int(probe.count),
               "seconds": time.time() - started}
    out = C.menu_dir(anchor) / "verify_init.json"
    out.write_text(json.dumps(payload, indent=2))
    log(f"[{anchor.name}] W-I: {'ALL PASS' if not failures else failures}  "
        f"({payload['seconds'] / 60:.1f} min)  -> {out}")
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["init", "arms"])
    ap.add_argument("--anchor", default=None, choices=list(C.ANCHOR_BY_NAME))
    args = ap.parse_args(argv)

    anchors = ([C.ANCHOR_BY_NAME[args.anchor]] if args.anchor
               else list(C.ANCHORS))
    device = torch.device("cuda")
    failed = []
    for anchor in anchors:
        if args.stage == "init":
            payload = verify_init(anchor, device)
        else:
            raise SystemExit("--stage arms is implemented in N18")
        if not payload["passed"]:
            failed.append((anchor.name, payload["failures"]))
    if failed:
        raise SystemExit(
            f"INVALID_EXPERIMENT: initialisation identity failed: {failed}.  "
            f"Plan v11 section 4: a W-I failure is fixed in the implementation, "
            f"never absorbed by loosening the invariant.")
    print("all W-I gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
