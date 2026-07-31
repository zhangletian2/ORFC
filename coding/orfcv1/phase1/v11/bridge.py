"""N16: can the one-bit search find a gain at the ORFC working point?

Plan v11 section 5.  The v11 menu codec -- A0's rotation, A0's uniform codebook,
freshly fitted side modes -- is handed to the same ``greedy_swap_search`` that
A2's outer loop will call, with only cal-500 visible.  The final allocation is
then judged once on dev-500, which the search never saw.  Nothing is trained.
holdout-500 stays sealed.

There is no v9 replay invariant here, and its absence is not an omission.  v10's
W9 required round 1 to reproduce v9's cal matrix because v10 searched over v9's
frozen ``(U_0, Theta_0)`` -- the same object, so the same numbers were owed.  v11
searches over ORFC's rotation and ORFC's codebook.  It is a different codec, so
v9's cal matrix is not a prediction about it, and demanding agreement would be
demanding that two different networks give the same answer.  What replaces W9 is
W-I1..W-I4, already verified in N15: the object being searched is A0 to the last
bit at the uniform allocation.

**A caveat this node cannot resolve, stated before its result is read.**  At this
point the three modes are not trained alike.  The uniform mode is ORFC's, trained
to convergence against ORFC's objective; the down- and up-modes are plain k-means
on the same rotated features.  N15 measured how far apart that leaves them: A0's
codebook sits 1.36x (R64) / 1.19x (R96) above a same-K k-means fit in rotated
MSE, and the K=2 down-mode reaches a *lower* rotated MSE than A0's K=4 uniform
mode despite carrying half the codewords.

So a swap evaluated here trades along two axes at once -- the bit budget, which
is the thing under study, and codebook provenance, which is not.  Whatever this
node reports is therefore a statement about *feasibility at the warm start*, not
about the value of non-uniform allocation.  The confound is removed by N18, where
the menu-coverage auxiliary trains all three modes against the same objective;
only there is A2 vs A1 a clean contrast.  The eligibility decision this node
makes is deliberately weak for that reason: it asks whether the search can move
at all, not whether the move is worth anything.
"""

import argparse
import json
import time

import numpy as np
import torch

from . import config as C
from . import init_menu
from . import search as search_mod
from . import stats
from .. import engine
from .. import tail as tail_mod


def _evaluator(codec, tail, resident, anchor, log):
    """``allocations [A, G] -> [A, N]``, with progress on a slow call."""
    def evaluate(allocations):
        started = time.time()

        def progress(done, total):
            rate = done / max(time.time() - started, 1e-9)
            log(f"[{anchor.name}] {len(allocations)} allocations, "
                f"{done}/{total} images  {rate:.1f} img/s  "
                f"eta {(total - done) / max(rate, 1e-9):.0f}s")

        return engine.evaluate_allocations(
            codec, tail, resident, allocations,
            image_batch=C.EVAL_IMAGE_BATCH, pair_budget=C.EVAL_PAIR_BUDGET,
            per_image=True, progress=progress)
    return evaluate


def run(anchor, device, images=None, log=print):
    started = time.time()
    engine.configure_precision(C.ALLOW_TF32)
    out = C.bridge_dir(anchor)
    out.mkdir(parents=True, exist_ok=True)

    codec, meta = init_menu.load_checked(anchor, device)      # elementwise
    width = C.GROUPS * C.DIM
    tensor = codec.transform.get_rotation().detach()
    identity = torch.eye(width, device=tensor.device, dtype=tensor.dtype)
    orth = float((tensor.t() @ tensor - identity).norm())
    log(f"[{anchor.name}] menu codec {meta['checkpoint_id']}, A0 "
        f"{meta['A0']['stem']}, orthogonality {orth:.2e}")

    tail = tail_mod.build_tail(C.LAYER, device)
    feature_path, teacher_path, rows, _ = C.load_split("cal")
    tail_mod.check_layer(C.LAYER, feature_path, teacher_path,
                         meta["feature_cache"])
    resident = engine.ResidentSet(feature_path, teacher_path, rows, device,
                                  max_images=images)
    engine.assert_uniform_shape(resident.count)
    evaluate = _evaluator(codec, tail, resident, anchor, log)

    first = {}

    def on_first_matrix(matrix):
        np.save(out / "cal_distortion_round1.npy", matrix.astype(np.float32))
        means = matrix.mean(1)
        best = int(np.argmin(means[1:]))
        _, pairs = engine.swap_candidates(anchor)
        first.update({
            "candidates": int(matrix.shape[0] - 1),
            "uniform_cal_mean": float(means[0]),
            "best_cal_mean": float(means[1:].min()),
            "best_relative_gain": float(
                (means[0] - means[1:].min()) / means[0]),
            "best_down_group": int(pairs[best, 0]),
            "best_up_group": int(pairs[best, 1]),
            "improving_candidates": int(np.sum(means[1:] < means[0]))})
        log(f"[{anchor.name}] round 1: {first['improving_candidates']}"
            f"/{first['candidates']} candidates improve on uniform; best "
            f"down g{first['best_down_group']} up g{first['best_up_group']} "
            f"gain {100 * first['best_relative_gain']:+.4f}%")

    result = search_mod.greedy_swap_search(
        evaluate, anchor, log=log, on_first_matrix=on_first_matrix)

    final = np.asarray(result["final_allocation"], dtype=np.int64)
    uniform = engine.uniform_allocation(anchor)
    if result["accepted_swaps"] == 0:
        log(f"[{anchor.name}] the search accepted no swap; dev is still "
            f"evaluated, at the uniform point against itself, so the record "
            f"shows a zero rather than a gap")

    cal_count = int(resident.count)
    del resident
    torch.cuda.empty_cache()

    feature_path, teacher_path, rows, _ = C.load_split("dev")
    tail_mod.check_layer(C.LAYER, feature_path, teacher_path)
    dev = engine.ResidentSet(feature_path, teacher_path, rows, device,
                             max_images=images)
    engine.assert_uniform_shape(dev.count)
    matrix = engine.evaluate_allocations(
        codec, tail, dev, np.vstack([uniform[None], final[None]]),
        image_batch=C.EVAL_IMAGE_BATCH, pair_budget=C.EVAL_PAIR_BUDGET,
        per_image=True)
    np.save(out / "dev_distortion.npy", matrix.astype(np.float32))

    uniform_d = matrix[0].astype(np.float64)
    final_d = matrix[1].astype(np.float64)
    differences = final_d - uniform_d
    np.save(out / "dev_differences.npy", differences)
    _, entry = stats.one_sided(differences)
    entry.update({
        "anchor": anchor.name, "rate": anchor.rate,
        "checkpoint_id": meta["checkpoint_id"],
        "dev_uniform_mean": float(uniform_d.mean()),
        "dev_final_mean": float(final_d.mean()),
        "relative_change": float(differences.mean() / uniform_d.mean()),
        "noise_floor_bound": float(uniform_d.mean() * C.REPLAY_REL_TOL),
        "effect_over_noise": float(abs(differences.mean())
                                   / (uniform_d.mean() * C.REPLAY_REL_TOL)),
    })

    payload = {
        "plan": "v11", "node": "N16", "stage": "bridge_on_orfc",
        "anchor": anchor.name, "rate": anchor.rate,
        "checkpoint_id": meta["checkpoint_id"],
        "A0": meta["A0"], "orthogonality_error": orth,
        "search_split": "cal", "acceptance_split": "dev",
        "cal_images": cal_count, "dev_images": int(dev.count),
        "round1": first,
        "search": result,
        "changed_groups": search_mod.changed_groups(uniform, final),
        "dev": entry,
        "codebook_provenance_caveat": (
            "The uniform mode is ORFC-trained and the side modes are plain "
            "k-means, so a swap here trades bit budget and codebook provenance "
            "together.  N15 measured the gap: A0's codebook is 1.36x (R64) / "
            "1.19x (R96) above a same-K k-means fit in rotated MSE.  This node "
            "therefore reports feasibility at the warm start, not the value of "
            "non-uniform allocation; N18 is where the modes are trained alike."),
        "seconds": time.time() - started,
    }
    (out / "bridge.json").write_text(json.dumps(payload, indent=2))
    log(f"[{anchor.name}] cal {result['accepted_swaps']} swaps accepted, "
        f"cal gain {100 * result.get('cal_total_relative_gain', 0.0):+.4f}%; "
        f"dev paired mean {entry['mean']:+.4f} "
        f"({100 * entry['relative_change']:+.4f}%) p {entry['p_value']:.5f}  "
        f"({payload['seconds'] / 60:.1f} min)")
    return payload


def decide(log=print):
    """Holm over the two anchors, then freeze the eligible set.

    Eligibility is deliberately a low bar: an anchor proceeds to N17/N18 if the
    search moved at all and dev did not get worse in a way the bootstrap can
    see.  A high bar here would be selecting anchors on a confounded measurement
    (see this module's docstring) and would also be spending dev's discriminating
    power on a question N18 answers properly.
    """
    entries, bootstraps, payloads = [], {}, {}
    for anchor in C.ANCHORS:
        path = C.bridge_dir(anchor) / "bridge.json"
        if not path.exists():
            raise SystemExit(f"INVALID_EXPERIMENT: [{anchor.name}] {path} is "
                             f"missing; both anchors must have run before the "
                             f"family can be corrected")
        payload = json.loads(path.read_text())
        payloads[anchor.name] = payload
        differences = np.load(C.bridge_dir(anchor) / "dev_differences.npy")
        means, entry = stats.one_sided(differences)
        entry.update({k: payload["dev"][k] for k in
                      ("anchor", "rate", "dev_uniform_mean", "dev_final_mean",
                       "relative_change", "effect_over_noise")})
        entry["final_allocation"] = payload["search"]["final_allocation"]
        entry["accepted_swaps"] = payload["search"]["accepted_swaps"]
        entry["cal_total_relative_gain"] = payload["search"].get(
            "cal_total_relative_gain")
        entries.append(entry)
        bootstraps[anchor.name] = means

    ordered = stats.finish(entries, bootstraps)
    eligible = []
    for entry in ordered:
        moved = entry["accepted_swaps"] > 0
        not_worse = entry["mean"] < 0 or entry["p_value"] > C.FWER
        entry["moved"] = bool(moved)
        entry["not_worse_on_dev"] = bool(not_worse)
        entry["eligible"] = bool(moved and not_worse)
        if entry["eligible"]:
            eligible.append(entry["anchor"])

    result = {
        "plan": "v11", "node": "N16", "stage": "bridge_decide", "split": "dev",
        "fwer": C.FWER, "family": [e["anchor"] for e in ordered],
        "anchors": ordered,
        "eligible_anchors": eligible,
        "verdict": "PASS" if eligible else "FAIL",
        "eligibility_rule": (
            "an anchor is eligible for N17/N18 if the cal search accepted at "
            "least one swap AND dev did not deteriorate detectably.  The Holm-"
            "corrected test is reported for the record but is not the gate: the "
            "measurement is confounded by codebook provenance (see bridge.py), "
            "so using it to select anchors would select on the confound."),
        "meaning": (
            "PASS: at the ORFC working point the greedy one-bit search, seeing "
            "only cal-500, moves off the uniform allocation and the move does "
            "not hurt on unseen dev-500.  This says the outer loop is operable "
            "at the warm start.  It does NOT say non-uniform allocation is "
            "worth anything -- N18 is that test."),
    }
    (C.V11 / "bridge_decision.json").write_text(json.dumps(result, indent=2))
    for entry in ordered:
        log(f"[{entry['anchor']}] swaps {entry['accepted_swaps']}  dev "
            f"{entry['mean']:+.4f} ({100 * entry['relative_change']:+.4f}%)  "
            f"p {entry['p_value']:.5f}  Holm alpha {entry['holm_alpha']:.4f}  "
            f"UCB {entry['ucb']:+.4f}  "
            f"{'ELIGIBLE' if entry['eligible'] else 'not eligible'}")
    log(f"N16 verdict: {result['verdict']}  eligible {eligible}")
    if not eligible:
        log("v11 ends here; holdout-500 stays sealed (plan section 9).")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor", choices=list(C.ANCHOR_BY_NAME))
    ap.add_argument("--decide", action="store_true",
                    help="Holm over both anchors from the saved dev differences")
    ap.add_argument("--images", type=int, default=None, help="debug only")
    args = ap.parse_args(argv)

    if args.decide:
        decide()
        return 0
    if not args.anchor:
        raise SystemExit("pass --anchor or --decide")
    payload = run(C.ANCHOR_BY_NAME[args.anchor], torch.device("cuda"),
                  images=args.images)
    print(json.dumps({k: v for k, v in payload.items() if k != "search"},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
