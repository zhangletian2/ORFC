"""Independent re-computation of v10's invariants (plan v10 section 6).

Every check here is also enforced at the point where it matters -- ``W9`` inside
the first search round, ``W6``/``W7`` inside the training loop, ``W1``/``W2``
inside ``build_holdout`` -- and is recomputed here from what actually landed on
disk.  The two implementations exist to disagree: an invariant checked only by
the code that produced the number is a comment with a traceback.

    W1  holdout-500 is disjoint from all four v9 splits
    W2  holdout-500 is one image per class, so the image is the stratum unit
    W3  the blkNN tag of every cache agrees with LAYER
    W5  every arm started from v9's frozen (U_0, Theta_0)
    W6  every allocation an arm ever held has the anchor's exact nominal rate
    W7  every U on disk satisfies ||U^T U - I||_F < 1e-5 in float64
    W8  the arms are matched where matching is exactly checkable
    W9  Stage B reproduced v9's cal sweep (values and argmin)

W8 deserves its own paragraph, because it is stated more weakly here than in the
plan and the difference is deliberate.  The plan asked for A1 and A2 to be
"逐比特同态" over the first ``T_outer`` steps.  That is not attainable and v9
already measured why: two identical calls to this tail differ by up to one fp32
ULP because the reduction order is not reproducible, and over hundreds of steps
an initial ULP compounds.  Demanding bit-identity would produce an invariant
that has to be waived every time it is run, which is worse than a weaker
invariant that is actually enforced.  So W8 checks the things that *are* exact --
identical batch-index streams, identical hyper-parameters, identical starting
tensors, identical allocations over the first block -- plus a step-1 loss inside
the fp32 floor.  What the arms are matched on is therefore a fact about files,
not about floating-point luck.
"""

import argparse
import json

import numpy as np
import torch

from . import build_holdout
from . import config as C
from .. import engine
from .. import frozen
from ..tail import check_layer


def _ok(name, detail):
    print(f"  PASS  {name}: {detail}")


def stage_holdout(log=print):
    names = build_holdout.pool_basenames()
    if len(names) != C.N_HOLDOUT:
        raise SystemExit(f"INVALID_EXPERIMENT (W1): the holdout pool holds "
                         f"{len(names)} images, expected {C.N_HOLDOUT}")
    stored = (C.V10 / "holdout_names.txt").read_text().split()
    if stored != names:
        raise SystemExit("INVALID_EXPERIMENT (W1): the registered holdout name "
                         "list is not the pool's current contents")
    overlap = build_holdout.check_overlap(names)                  # W1
    _ok("W1", f"zero overlap with {', '.join(overlap)}")
    strat = build_holdout.check_stratification(names)             # W2
    _ok("W2", f"{strat['images']} images, {strat['distinct_classes']} classes, "
              f"max {strat['max_per_class']} per class -> the stratum unit is "
              f"the image")
    check_layer(C.LAYER, C.HOLDOUT_FEATURES, C.HOLDOUT_TEACHERS)  # W3
    _ok("W3", f"both holdout caches carry {C.BLOCK}")
    return {"W1": overlap, "W2": strat, "W3": "ok"}


def stage_frozen(device, log=print):
    """W5/W7 on v9's objects themselves, before anything reads them."""
    report = {}
    for anchor in C.ANCHORS:
        codec, meta = frozen.load_checked(anchor, device)          # elementwise
        orth = frozen.orthogonality_error(codec)
        if orth > C.ORTH_TOL:
            raise SystemExit(f"INVALID_EXPERIMENT (W7): [{anchor.name}] "
                             f"||U^T U - I||_F = {orth:.3e}")
        report[anchor.name] = {"checkpoint_id": meta["checkpoint_id"],
                               "orthogonality_error": orth}
        _ok("W5/W7", f"[{anchor.name}] {meta['checkpoint_id']} matches its "
                     f"reference elementwise, orth {orth:.2e}")
        del codec
        torch.cuda.empty_cache()
    return report


def stage_bridge(log=print):
    """W9 and the rate invariant, re-read from Stage B's own output."""
    report = {}
    for anchor in C.ANCHORS:
        path = C.run_dir(anchor, "bridge") / "bridge.json"
        if not path.exists():
            log(f"  SKIP  [{anchor.name}] Stage B has not run")
            continue
        payload = json.loads(path.read_text())
        replay = payload["w9_replay"]
        if replay["max_rel_deviation"] > C.REPLAY_REL_TOL:
            raise SystemExit(f"INVALID_EXPERIMENT (W9): [{anchor.name}] "
                             f"{replay['max_rel_deviation']:.3e}")
        if replay["v9_argmin"] != replay["v10_argmin"]:
            raise SystemExit(f"INVALID_EXPERIMENT (W9): [{anchor.name}] argmin "
                             f"{replay['v10_argmin']} != v9 "
                             f"{replay['v9_argmin']}")
        # Recompute the deviation from the stored matrix, not from the number
        # Stage B wrote about itself.
        stored = np.load(C.run_dir(anchor, "bridge")
                         / "cal_distortion_round1.npy").astype(np.float64)
        reference = np.load(anchor.root / "cal_distortion.npy").astype(np.float64)
        deviation = (np.abs(stored - reference).max()
                     / float(np.abs(reference).mean()))
        if deviation > C.REPLAY_REL_TOL:
            raise SystemExit(f"INVALID_EXPERIMENT (W9): [{anchor.name}] "
                             f"recomputed replay deviation {deviation:.3e}")
        allocation = np.asarray(payload["search"]["final_allocation"])
        rate = engine.nominal_rate(allocation, anchor)
        if rate != anchor.rate:                                    # W6
            raise SystemExit(f"INVALID_EXPERIMENT (W6): [{anchor.name}] the "
                             f"bridge allocation has nominal rate {rate}")
        for entry in payload["search"]["trace"]:
            base_rate = engine.nominal_rate(
                np.asarray(entry["base_allocation"]), anchor)
            if base_rate != anchor.rate:
                raise SystemExit(f"INVALID_EXPERIMENT (W6): [{anchor.name}] "
                                 f"step {entry['step']} base rate {base_rate}")
        report[anchor.name] = {
            "w9_deviation_recomputed": deviation,
            "multiples_of_floor": deviation / C.REPLAY_REL_TOL,
            "argmin": replay["v10_argmin"],
            "final_allocation": allocation.tolist(),
            "nominal_rate": rate,
            "accepted_swaps": payload["search"]["accepted_swaps"]}
        _ok("W9", f"[{anchor.name}] {deviation:.2e} relative "
                  f"({deviation / C.REPLAY_REL_TOL:.2f}x floor), argmin "
                  f"reproduced")
        _ok("W6", f"[{anchor.name}] every allocation in the trace is "
                  f"R{anchor.rate}")
    return report


def stage_arms(log=print):
    """W5/W6/W7/W8 on the three trained arms."""
    report = {}
    for anchor in C.ANCHORS:
        payloads, present = {}, []
        for arm in C.ARMS:
            path = C.run_dir(anchor, arm) / "train.json"
            if path.exists():
                payloads[arm] = json.loads(path.read_text())
                present.append(arm)
        if not present:
            log(f"  SKIP  [{anchor.name}] no arm has run")
            continue
        entry = {"arms_present": present}

        # W5: all arms report the same v9 checkpoint they were loaded from, and
        # that checkpoint still matches its own reference (stage_frozen).
        ids = {arm: payloads[arm]["checkpoint_id"] for arm in present}
        if len(set(ids.values())) != 1:
            raise SystemExit(f"INVALID_EXPERIMENT (W5): [{anchor.name}] the "
                             f"arms started from different checkpoints: {ids}")
        entry["checkpoint_id"] = next(iter(ids.values()))
        _ok("W5", f"[{anchor.name}] all arms started from "
                  f"{entry['checkpoint_id']}")

        # W6: every allocation any arm ever held is at the exact nominal rate.
        for arm in present:
            allocations = [payloads[arm]["final_allocation"]]
            allocations += [h["allocation"] for h in
                            payloads[arm].get("allocation_history", [])]
            for allocation in allocations:
                rate = engine.nominal_rate(np.asarray(allocation), anchor)
                if rate != anchor.rate:
                    raise SystemExit(f"INVALID_EXPERIMENT (W6): "
                                     f"[{anchor.name}/{arm}] rate {rate}")
        _ok("W6", f"[{anchor.name}] every allocation held by any arm is "
                  f"R{anchor.rate}")

        # W7: orthogonality of each arm's final U, recomputed in float64 from
        # the saved reference array rather than from the checkpoint's own claim.
        orth = {}
        for arm in present:
            matrix = np.load(C.run_dir(anchor, arm)
                             / "codec_ref.npz")["U"].astype(np.float64)
            error = float(np.linalg.norm(matrix.T @ matrix
                                         - np.eye(matrix.shape[0])))
            if error > C.ORTH_TOL:
                raise SystemExit(f"INVALID_EXPERIMENT (W7): "
                                 f"[{anchor.name}/{arm}] "
                                 f"||U^T U - I||_F = {error:.3e}")
            orth[arm] = error
        entry["orthogonality_error"] = orth
        _ok("W7", f"[{anchor.name}] max ||U^T U - I||_F = "
                  f"{max(orth.values()):.2e}")

        # W8: matched where matching is exact.
        if C.ARM_EQUAL_STEP in present and C.ARM_JOINT in present:
            a1, a2 = payloads[C.ARM_EQUAL_STEP], payloads[C.ARM_JOINT]
            keys = ("batch", "lr_U", "lr_theta", "T_outer", "N_inner",
                    "loss_scale")
            mismatch = {k: (a1["hyper_parameters"][k], a2["hyper_parameters"][k])
                        for k in keys
                        if a1["hyper_parameters"][k] != a2["hyper_parameters"][k]}
            if mismatch:
                raise SystemExit(f"INVALID_EXPERIMENT (W8): [{anchor.name}] A1 "
                                 f"and A2 disagree on {mismatch}")
            if a1["steps"] != a2["steps"]:
                raise SystemExit(f"INVALID_EXPERIMENT (W8): [{anchor.name}] A1 "
                                 f"ran {a1['steps']} steps, A2 {a2['steps']}")
            i1 = np.load(C.run_dir(anchor, C.ARM_EQUAL_STEP)
                         / "batch_indices.npy")
            i2 = np.load(C.run_dir(anchor, C.ARM_JOINT) / "batch_indices.npy")
            if not np.array_equal(i1, i2):
                first = int(np.argmax((i1 != i2).any(1)))
                raise SystemExit(f"INVALID_EXPERIMENT (W8): [{anchor.name}] the "
                                 f"batch streams first differ at step "
                                 f"{first + 1}")
            l1 = np.load(C.run_dir(anchor, C.ARM_EQUAL_STEP) / "losses.npy")
            l2 = np.load(C.run_dir(anchor, C.ARM_JOINT) / "losses.npy")
            step1 = abs(l1[0] - l2[0]) / max(abs(l1[0]), 1e-30)
            if step1 > C.REPLAY_REL_TOL:
                raise SystemExit(f"INVALID_EXPERIMENT (W8): [{anchor.name}] the "
                                 f"step-1 losses differ by {step1:.3e} "
                                 f"relative, above the fp32 floor -- the arms "
                                 f"did not start from the same state")
            t_outer = a2["hyper_parameters"]["T_outer"]
            first_event = min((e["at_step"] for e in a2["outer_events"]),
                              default=None)
            if first_event is not None and first_event < t_outer:
                raise SystemExit(f"INVALID_EXPERIMENT (W8): [{anchor.name}] A2's "
                                 f"first outer event fired at step "
                                 f"{first_event}, before step {t_outer}")
            drift = float(np.abs(l1[:t_outer] - l2[:t_outer]).max()
                          / np.abs(l1[:t_outer]).mean())
            entry["W8"] = {
                "batch_streams_identical": True,
                "step1_relative_difference": float(step1),
                "first_outer_event_step": first_event,
                "fp32_drift_over_first_block": drift,
                "note": "the drift is fp32 divergence from an identical start, "
                        "not a protocol difference; the allocation is the same "
                        "over this whole block",
            }
            _ok("W8", f"[{anchor.name}] identical batch streams and "
                      f"hyper-parameters; step-1 losses agree to "
                      f"{step1:.2e} relative; A2's first swap at step "
                      f"{first_event} (T_outer {t_outer})")

        # G1, reported per arm.
        entry["G1"] = {arm: {"relative_improvement":
                             payloads[arm]["train_relative_improvement"],
                             "pass": payloads[arm]["G1_pass"]}
                       for arm in present}
        for arm in present:
            if not payloads[arm]["G1_pass"]:
                log(f"  FAIL  G1 [{anchor.name}/{arm}]: train improvement "
                    f"{payloads[arm]['train_relative_improvement']:.3e}")
        report[anchor.name] = entry
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=("holdout", "frozen", "bridge", "arms", "all"))
    args = ap.parse_args(argv)
    device = torch.device("cuda")
    engine.configure_precision(C.ALLOW_TF32)

    report = {}
    if args.stage in ("holdout", "all"):
        print("=== holdout (W1, W2, W3) ===")
        report["holdout"] = stage_holdout()
    if args.stage in ("frozen", "all"):
        print("=== v9 frozen objects (W5, W7) ===")
        report["frozen"] = stage_frozen(device)
    if args.stage in ("bridge", "all"):
        print("=== Stage B (W6, W9) ===")
        report["bridge"] = stage_bridge()
    if args.stage in ("arms", "all"):
        print("=== Stage C arms (W5, W6, W7, W8, G1) ===")
        report["arms"] = stage_arms()

    (C.V10 / f"verify_{args.stage}.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(f"wrote {C.V10 / f'verify_{args.stage}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
