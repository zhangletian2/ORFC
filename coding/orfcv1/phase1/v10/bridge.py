"""Stage B: the algorithmic bridge (plan v10 section 4).

v9 established that the gain *exists*: with ``(U_0, Theta_0)`` frozen and the
nominal rate exactly equal, a one-bit non-uniform transfer lowers the hard tail
distortion on measure-3000 (R64 -1.600%, R96 -1.227%, Holm at FWER 0.05).  That
is a statement about the neighbourhood, not about any optimiser -- v9 found its
candidate by exhaustively evaluating all 992 points, which is not something the
outer loop of A2 can afford to do at every window.

Stage B asks the separate question the user split out: *can the optimiser find
it*.  The frozen codec is handed to ``search.greedy_swap_search``, which is the
same function A2's outer loop will call, with only cal-500 visible to it.  Its
final allocation is then judged once on dev-500, which the search never saw.

Two things are deliberately not here.  There is no training: ``(U_0, Theta_0)``
are the v9 objects, verified elementwise by ``frozen.load_checked``.  And there
is no holdout: dev-500 exists for exactly this, and holdout-500 stays sealed for
Stage D.  If Stage B fails, v10 stops -- a joint-learning arm whose outer loop
cannot beat the uniform point on a *frozen* codec would be reporting the inner
loop's work under the outer loop's name.

W9, the replay invariant, is what ties Stage B to v9 rather than merely running
beside it: the first search round evaluates exactly v9's 993 points, so its
matrix must reproduce ``cal_distortion.npy`` within the fp32 floor and its argmin
must be v9's frozen candidate, to the group index.  If the two disagree, some
part of the stack moved between v9 and v10 and every later comparison would be
meaningless, so the run aborts rather than continuing.
"""

import argparse
import json
import time

import numpy as np
import torch

from . import config as C
from . import search as search_mod
from . import stats
from .. import engine
from .. import frozen
from .. import splits
from .. import tail as tail_mod


def bridge_dir(anchor):
    return C.run_dir(anchor, "bridge")


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


def check_v9_replay(anchor, matrix, log=print):
    """W9: the first round must reproduce v9's cal sweep, values and argmin.

    ``matrix`` is ``[993, 500]`` with row 0 the uniform point, i.e. exactly the
    layout ``cal_sweep.py`` saved.  v9 stored it as float32, so the comparison
    carries that storage rounding (~6e-8 relative) inside the 9.54e-7 floor;
    that is a third of the budget at worst and is stated here rather than
    silently absorbed.
    """
    reference_path = anchor.root / "cal_distortion.npy"
    record_path = anchor.root / "cal_sweep.json"
    for path in (reference_path, record_path):
        if not path.exists():
            raise SystemExit(f"INVALID_EXPERIMENT (W9): [{anchor.name}] "
                             f"{path} is missing; there is nothing to replay "
                             f"against")
    reference = np.load(reference_path).astype(np.float64)
    record = json.loads(record_path.read_text())

    if reference.shape != matrix.shape:
        raise SystemExit(f"INVALID_EXPERIMENT (W9): [{anchor.name}] v9 stored a "
                         f"{reference.shape} cal matrix, v10 recomputed "
                         f"{matrix.shape}")
    level = float(np.abs(reference).mean())
    deviation = float(np.abs(reference - matrix).max()) / level

    means = matrix.mean(1)
    best = int(np.argmin(means[1:]))
    _, pairs = engine.swap_candidates(anchor)
    v9_down = int(record["candidate"]["down_group"])
    v9_up = int(record["candidate"]["up_group"])
    report = {
        "reference": str(reference_path),
        "shape": list(matrix.shape),
        "mean_level": level,
        "max_rel_deviation": deviation,
        "tolerance": C.REPLAY_REL_TOL,
        "multiples_of_floor": deviation / C.REPLAY_REL_TOL,
        "v9_argmin": {"index": int(record["candidate"]["index"]),
                      "down_group": v9_down, "up_group": v9_up},
        "v10_argmin": {"index": best,
                       "down_group": int(pairs[best, 0]),
                       "up_group": int(pairs[best, 1])},
        "v9_uniform_cal_mean": float(record["uniform"]["cal_mean"]),
        "v10_uniform_cal_mean": float(means[0]),
    }
    if deviation > C.REPLAY_REL_TOL:
        raise SystemExit(
            f"INVALID_EXPERIMENT (W9): [{anchor.name}] the cal matrix moved by "
            f"{deviation:.3e} relative against v9, above the "
            f"{C.REPLAY_REL_TOL:.3e} floor.  Something in the stack, the codec "
            f"or the split is not what v9 measured.  Detail: {report}")
    if (int(pairs[best, 0]), int(pairs[best, 1])) != (v9_down, v9_up):
        raise SystemExit(
            f"INVALID_EXPERIMENT (W9): [{anchor.name}] the cal argmin is now "
            f"down g{pairs[best, 0]} up g{pairs[best, 1]}, v9 froze "
            f"down g{v9_down} up g{v9_up}.  Detail: {report}")
    log(f"[{anchor.name}] W9 replay: {deviation:.2e} relative "
        f"({report['multiples_of_floor']:.2f}x floor), argmin down g{v9_down} "
        f"up g{v9_up} reproduced")
    return report


def run(anchor, device, images=None, dry_run_replay=False, log=print):
    started = time.time()
    engine.configure_precision(C.ALLOW_TF32)
    out = bridge_dir(anchor)
    out.mkdir(parents=True, exist_ok=True)

    codec, meta = frozen.load_checked(anchor, device)          # W5, elementwise
    orth = frozen.orthogonality_error(codec)
    if orth > C.ORTH_TOL:                                       # W7
        raise SystemExit(f"INVALID_EXPERIMENT (W7): [{anchor.name}] "
                         f"||U^T U - I||_F = {orth:.3e} > {C.ORTH_TOL:.1e}")
    log(f"[{anchor.name}] frozen codec {meta['checkpoint_id']}, "
        f"orthogonality {orth:.2e}")

    feature_path, teacher_path, rows, _ = splits.load_split("cal")
    tail_mod.check_layer(C.LAYER, feature_path, teacher_path,
                         meta["feature_cache"])
    tail = tail_mod.build_tail(C.LAYER, device)
    resident = engine.ResidentSet(feature_path, teacher_path, rows, device,
                                  max_images=images)
    engine.assert_uniform_shape(resident.count)
    evaluate = _evaluator(codec, tail, resident, anchor, log)

    if dry_run_replay:
        uniform = engine.uniform_allocation(anchor)
        candidates, _ = search_mod.legal_swaps(uniform, anchor)
        matrix = evaluate(np.vstack([uniform[None], candidates]))
        replay = check_v9_replay(anchor, np.asarray(matrix, dtype=np.float64),
                                 log=log)
        replay["seconds"] = time.time() - started
        (out / "w9_replay.json").write_text(json.dumps(replay, indent=2))
        return replay

    replay = {}

    def on_first_matrix(matrix):
        # W9 runs here, inside round 1, so a stack that has drifted since v9
        # costs one sweep rather than eight.
        replay.update(check_v9_replay(anchor, matrix, log=log))
        np.save(out / "cal_distortion_round1.npy", matrix.astype(np.float32))

    result = search_mod.greedy_swap_search(
        evaluate, anchor, log=log, on_first_matrix=on_first_matrix)

    final = np.asarray(result["final_allocation"], dtype=np.int64)
    uniform = engine.uniform_allocation(anchor)
    if result["accepted_swaps"] == 0:
        log(f"[{anchor.name}] the search accepted no swap; the bridge cannot "
            f"pass on this anchor and dev is still evaluated, at the uniform "
            f"point against itself, so the record shows a zero rather than a "
            f"gap")

    cal_count = int(resident.count)
    del resident
    torch.cuda.empty_cache()

    # dev-500: the acceptance set.  It was not visible to the search.
    feature_path, teacher_path, rows, _ = splits.load_split("dev")
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
        "plan": "v10", "stage": "B_bridge", "anchor": anchor.name,
        "rate": anchor.rate, "checkpoint_id": meta["checkpoint_id"],
        "orthogonality_error": orth,
        "search_split": "cal", "acceptance_split": "dev",
        "cal_images": cal_count,
        "dev_images": int(dev.count),
        "search": result,
        "changed_groups": search_mod.changed_groups(uniform, final),
        "w9_replay": replay,
        "dev": entry,
        "note": "Holm across the two anchors is applied by --decide, once both "
                "anchors have produced their dev differences.",
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
    """Holm over the two anchors, from the saved dev differences.

    Kept apart from :func:`run` so the two anchors can search in parallel on
    separate GPUs; the bootstrap is reproduced here from the stored per-image
    differences under the frozen seed, so this step is a pure function of files
    that already exist.
    """
    entries, bootstraps = [], {}
    for anchor in C.ANCHORS:
        path = bridge_dir(anchor) / "bridge.json"
        if not path.exists():
            raise SystemExit(f"INVALID_EXPERIMENT: [{anchor.name}] {path} is "
                             f"missing; both anchors must have run before the "
                             f"family can be corrected")
        payload = json.loads(path.read_text())
        differences = np.load(bridge_dir(anchor) / "dev_differences.npy")
        means, entry = stats.one_sided(differences)
        if abs(entry["mean"] - payload["dev"]["mean"]) > 1e-9:
            raise SystemExit(f"INVALID_EXPERIMENT: [{anchor.name}] the stored "
                             f"dev mean {payload['dev']['mean']} does not match "
                             f"the saved differences {entry['mean']}")
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
    verdict = "PASS" if any(e["passes"] for e in ordered) else "FAIL"
    result = {
        "plan": "v10", "stage": "B_decide", "split": "dev",
        "fwer": C.FWER, "family": [e["anchor"] for e in ordered],
        "anchors": ordered,
        "verdict": verdict,
        "meaning": ("PASS: the greedy one-bit search, seeing only cal-500, "
                    "found a same-rate allocation that is better than uniform "
                    "on unseen dev-500 with the frozen (U_0, Theta_0).  The "
                    "optimiser can find the gain v9 showed exists."),
    }
    (C.V10 / "bridge_decision.json").write_text(json.dumps(result, indent=2))
    for entry in ordered:
        log(f"[{entry['anchor']}] dev {entry['mean']:+.4f} "
            f"({100 * entry['relative_change']:+.4f}%)  p {entry['p_value']:.5f}"
            f"  Holm alpha {entry['holm_alpha']:.4f}  UCB {entry['ucb']:+.4f}  "
            f"{'PASS' if entry['passes'] else 'not rejected'}")
    log(f"Stage B verdict: {verdict}")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor", choices=list(C.ANCHOR_BY_NAME))
    ap.add_argument("--decide", action="store_true",
                    help="Holm over both anchors from the saved dev differences")
    ap.add_argument("--dry-run-replay", action="store_true",
                    help="W9 only: evaluate v9's 993 points and stop")
    ap.add_argument("--images", type=int, default=None, help="debug only")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.decide:
        print(json.dumps(decide(), indent=2))
        return 0
    if not args.anchor:
        raise SystemExit("--anchor or --decide is required")

    anchor = C.ANCHOR_BY_NAME[args.anchor]
    output = bridge_dir(anchor) / ("w9_replay.json" if args.dry_run_replay
                                   else "bridge.json")
    if output.exists() and not args.force:
        raise SystemExit(f"{output} exists; the bridge result is frozen. "
                         f"--force re-runs the search, which is a second "
                         f"selection, not a re-run.")
    result = run(anchor, torch.device("cuda"), images=args.images,
                 dry_run_replay=args.dry_run_replay)
    print(json.dumps({k: v for k, v in result.items() if k != "search"},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
