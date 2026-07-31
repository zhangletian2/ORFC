"""The 993-point exhaustive cal sweep and the frozen candidate (plan v9 section 5).

For one anchor: evaluate the uniform allocation and all 992 ordered one-bit
transfers on cal-500 under hard quantisation, then take

    m_hat_R  in  argmin over the 992 candidates of D_cal(m),

breaking ties by the frozen lexicographic order on ``(down_group, up_group)``.
The whole 992-point set is enumerated -- no proxy pre-screens it, no "promising"
group is chosen by hand -- so the candidate is an exact empirical argmin over
the entire one-transfer neighbourhood of the uniform point.

What this file may not do is compare anything on measure-3000.  The cal argmin
is a selection, and a selection evaluated on the set that produced it is the
winner's curse that voided the earlier self-fit numbers.  ``unseal.py`` does the
comparison, once, on data this script cannot open.
"""

import argparse
import json
import time

import numpy as np
import torch

from . import config as C
from . import engine
from . import frozen
from . import splits
from . import tail as tail_mod
from . import verify as verify_mod


def sweep(anchor, device, images=None, log=print):
    started = time.time()
    engine.configure_precision(C.ALLOW_TF32)

    report = json.loads((anchor.root / "verify.json").read_text())
    if report.get("verdict") != "PASS":
        raise SystemExit(f"INVALID_EXPERIMENT: [{anchor.name}] verify.json is "
                         f"not a PASS; run phase1.verify first")

    codec, meta = frozen.load_checked(anchor, device)
    feature_path, teacher_path, rows, names = splits.load_split("cal")
    tail_mod.check_layer(C.LAYER, feature_path, teacher_path,
                         meta["feature_cache"])
    tail = tail_mod.build_tail(C.LAYER, device)

    # Cross-process half of the replay invariant: the same probe, in a process
    # that has loaded the codec afresh, must land inside the fp32 noise floor.
    probe_resident = engine.ResidentSet(feature_path, teacher_path,
                                        rows[:C.REPLAY_IMAGES], device)
    probe = verify_mod.replay_probe(codec, tail, probe_resident, anchor)
    reference = np.load(anchor.root / "replay_probe.npy")
    level = float(np.abs(reference).mean())
    drift = float(np.abs(probe - reference).max()) / level
    if drift > C.REPLAY_REL_TOL:
        raise SystemExit(f"INVALID_EXPERIMENT: [{anchor.name}] hard replay "
                         f"differs from verify.py by a relative {drift:.3e} > "
                         f"{C.REPLAY_REL_TOL:.2e}")
    del probe_resident, probe
    torch.cuda.empty_cache()
    log(f"[{anchor.name}] cross-process replay within {drift:.2e} relative")

    resident = engine.ResidentSet(feature_path, teacher_path, rows, device,
                                  max_images=images)
    engine.assert_uniform_shape(resident.count)
    uniform = engine.uniform_allocation(anchor)
    candidates, pairs = engine.swap_candidates(anchor)
    verify_mod.check_rates(anchor, candidates, pairs)          # V1/V2, again
    allocations = np.vstack([uniform[None], candidates])       # row 0 = uniform

    def progress(done, total):
        if done % 100 == 0 or done == total:
            rate = done / max(time.time() - started, 1e-9)
            log(f"[{anchor.name}] cal {done}/{total} images  "
                f"{rate:.2f} img/s  eta {(total - done) / max(rate, 1e-9):.0f}s")

    matrix = engine.evaluate_allocations(
        codec, tail, resident, allocations, image_batch=C.EVAL_IMAGE_BATCH,
        pair_budget=C.EVAL_PAIR_BUDGET, per_image=True, progress=progress)
    np.save(anchor.root / "cal_distortion.npy", matrix.astype(np.float32))

    means = matrix.mean(1)
    uniform_mean = float(means[0])
    candidate_means = means[1:]
    # argmin over the candidates only; np.argmin returns the first minimiser and
    # the rows are already in the frozen tie-break order.
    best = int(np.argmin(candidate_means))
    ties = int(np.sum(candidate_means == candidate_means[best]))
    order = np.argsort(candidate_means, kind="stable")

    # An upper bound on how much of any gap could be fp32 replay noise rather
    # than allocation.  Reported as a ratio so a "gain" of the same size as the
    # noise cannot be read as a gain.
    noise = uniform_mean * C.REPLAY_REL_TOL
    gap_runner_up = float(candidate_means[order[1]] - candidate_means[best])

    result = {
        "plan": "v9", "stage": "cal_sweep", "anchor": anchor.name,
        "rate": anchor.rate, "checkpoint_id": meta["checkpoint_id"],
        "split": "cal", "images": int(resident.count),
        "evaluated": int(len(allocations)), "candidates": int(C.N_CANDIDATES),
        "replay_drift_vs_verify": drift,
        "tie_break": C.TIE_BREAK, "ties_at_minimum": ties,
        "uniform": {"allocation": uniform.tolist(), "cal_mean": uniform_mean},
        "candidate": {
            "index": best,
            "down_group": int(pairs[best, 0]), "up_group": int(pairs[best, 1]),
            "allocation": candidates[best].tolist(),
            "nominal_rate": engine.nominal_rate(candidates[best], anchor),
            "cal_mean": float(candidate_means[best]),
            "cal_gain_vs_uniform": float(uniform_mean - candidate_means[best]),
            "cal_relative_gain": float(
                (uniform_mean - candidate_means[best]) / uniform_mean)},
        "cal_top10": [
            {"down_group": int(pairs[i, 0]), "up_group": int(pairs[i, 1]),
             "cal_mean": float(candidate_means[i])} for i in order[:10]],
        "cal_worst3": [
            {"down_group": int(pairs[i, 0]), "up_group": int(pairs[i, 1]),
             "cal_mean": float(candidate_means[i])} for i in order[-3:]],
        "candidates_below_uniform": int(np.sum(candidate_means < uniform_mean)),
        "noise_floor": {
            "replay_rel_probe": drift, "replay_rel_tol": C.REPLAY_REL_TOL,
            "absolute_bound_at_uniform_level": float(noise),
            "gain_over_noise": float(
                (uniform_mean - candidate_means[best]) / noise),
            "gap_to_runner_up": gap_runner_up,
            "gap_to_runner_up_over_noise": float(gap_runner_up / noise)},
        "seconds": time.time() - started,
    }
    (anchor.root / "cal_sweep.json").write_text(json.dumps(result, indent=2))
    log(f"[{anchor.name}] frozen candidate: down g{result['candidate']['down_group']} "
        f"up g{result['candidate']['up_group']}  cal gain "
        f"{result['candidate']['cal_gain_vs_uniform']:.4f} "
        f"({100 * result['candidate']['cal_relative_gain']:.3f}%)  "
        f"{result['candidates_below_uniform']}/{C.N_CANDIDATES} candidates "
        f"below uniform  ({result['seconds'] / 60:.1f} min)")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True, choices=list(C.ANCHOR_BY_NAME))
    ap.add_argument("--images", type=int, default=None,
                    help="debug only; the protocol uses all of cal-500")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    anchor = C.ANCHOR_BY_NAME[args.anchor]
    output = anchor.root / "cal_sweep.json"
    if output.exists() and not args.force:
        raise SystemExit(f"{output} exists; the candidate is frozen. Passing "
                         f"--force re-selects it, which is a protocol change, "
                         f"not a re-run.")
    result = sweep(anchor, torch.device("cuda"), images=args.images)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("cal_top10", "cal_worst3")}, indent=2))


if __name__ == "__main__":
    main()
