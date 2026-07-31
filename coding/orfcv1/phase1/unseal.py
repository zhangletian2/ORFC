"""The single unsealing of measure-3000 and the confirmatory test (v9 section 6).

One process, one pass, both anchors.  For each anchor exactly two allocations
reach measure -- the uniform point and the candidate frozen by `cal_sweep.py` --
and the only statistic computed is the per-image paired difference

    d_i = D_candidate(x_i) - D_uniform(x_i),   i = 1..3000,

tested one-sided against zero by the pre-registered paired bootstrap.  The two
anchors are one family, Holm-controlled at FWER 0.05: an anchor passes iff its
Holm-corrected one-sided upper confidence bound is strictly below zero.

There is no argmin here, no scan over allocations, no second look.  The script
refuses to run unless both candidates were frozen to disk *before* the seal is
broken, and it writes the seal token itself so that the ordering is recorded by
the filesystem rather than asserted in prose.
"""

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import torch

from . import config as C
from . import engine
from . import frozen
from . import splits
from . import tail as tail_mod

RESULT_PATH = C.V9 / "unseal.json"


def paired_bootstrap(differences, resamples=C.BOOTSTRAP_RESAMPLES,
                     seed=C.BOOTSTRAP_SEED, chunk=2000):
    """Bootstrap distribution of the mean paired difference.

    Resampling is over images, keeping each image's pair intact -- the pairing
    is what removes the between-image variance that swamped v8's unpaired
    comparisons.
    """
    rng = np.random.default_rng(seed)
    count = len(differences)
    means = np.empty(resamples, dtype=np.float64)
    values = np.asarray(differences, dtype=np.float64)
    for start in range(0, resamples, chunk):
        stop = min(start + chunk, resamples)
        index = rng.integers(0, count, size=(stop - start, count))
        means[start:stop] = values[index].mean(1)
    return means


def one_sided(differences):
    """Bootstrap the mean paired difference once; return it with its p-value.

    The upper confidence bound is filled in later, from this same distribution,
    at whatever level Holm assigns -- so ``ucb < 0`` and ``p < holm_alpha``
    remain the same statement.
    """
    means = paired_bootstrap(differences)
    return means, {"mean": float(np.mean(differences)),
                   "p_value": float(np.mean(means >= 0.0)),
                   "bootstrap_resamples": int(C.BOOTSTRAP_RESAMPLES),
                   "bootstrap_seed": int(C.BOOTSTRAP_SEED)}


def holm(entries, fwer=C.FWER):
    """Holm step-down over the anchor family, on the bootstrap p-values."""
    order = sorted(entries, key=lambda e: e["p_value"])
    total = len(order)
    rejected_so_far = True
    for rank, entry in enumerate(order):
        entry["holm_alpha"] = fwer / (total - rank)
        entry["holm_rank"] = rank + 1
        entry["rejects_at_holm_alpha"] = bool(
            rejected_so_far and entry["p_value"] < entry["holm_alpha"])
        rejected_so_far = entry["rejects_at_holm_alpha"]
    return order


def frozen_candidates(log=print):
    """Read both frozen candidates and prove they predate the unsealing."""
    plans = []
    for anchor in C.ANCHORS:
        path = anchor.root / "cal_sweep.json"
        if not path.exists():
            raise SystemExit(f"INVALID_EXPERIMENT: [{anchor.name}] no frozen "
                             f"candidate at {path}; measure stays sealed")
        record = json.loads(path.read_text())
        if record["anchor"] != anchor.name:
            raise SystemExit(f"INVALID_EXPERIMENT: {path} holds "
                             f"{record['anchor']}")
        allocation = np.asarray(record["candidate"]["allocation"],
                                dtype=np.int64)
        uniform = engine.uniform_allocation(anchor)
        for name, alloc in (("uniform", uniform), ("candidate", allocation)):
            rate = engine.nominal_rate(alloc, anchor)
            if rate != anchor.rate:
                raise SystemExit(f"INVALID_EXPERIMENT: [{anchor.name}] the "
                                 f"{name} allocation has nominal rate {rate}, "
                                 f"not {anchor.rate}")
        plans.append({"anchor": anchor, "record": record,
                      "uniform": uniform, "candidate": allocation,
                      "frozen_at": path.stat().st_mtime})
        log(f"[{anchor.name}] candidate frozen at "
            f"{datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)}: "
            f"down g{record['candidate']['down_group']} "
            f"up g{record['candidate']['up_group']}")
    return plans


def write_seal(plans):
    """Break the seal, recording exactly what was frozen at that moment."""
    if splits.is_unsealed():
        return json.loads(splits.UNSEAL_TOKEN.read_text())
    token = {
        "plan": "v9", "unsealed_at": time.time(),
        "unsealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "family": [{"anchor": p["anchor"].name, "rate": p["anchor"].rate,
                    "candidate": p["candidate"].tolist(),
                    "uniform": p["uniform"].tolist(),
                    "cal_sweep_mtime": p["frozen_at"]} for p in plans],
        "fwer": C.FWER, "resamples": C.BOOTSTRAP_RESAMPLES,
        "seed": C.BOOTSTRAP_SEED}
    splits.UNSEAL_TOKEN.parent.mkdir(parents=True, exist_ok=True)
    splits.UNSEAL_TOKEN.write_text(json.dumps(token, indent=2))
    for plan in plans:
        if plan["frozen_at"] >= token["unsealed_at"]:
            raise SystemExit(f"INVALID_EXPERIMENT: "
                             f"[{plan['anchor'].name}] the candidate file is "
                             f"not older than the unseal token")
    return token


def unseal(device, log=print):
    started = time.time()
    engine.configure_precision(C.ALLOW_TF32)
    plans = frozen_candidates(log=log)
    token = write_seal(plans)
    log(f"measure-3000 unsealed at {token['unsealed_at_utc']}")

    feature_path, teacher_path, rows, names = splits.load_split(
        "measure", allow_measure=True)
    tail_mod.check_layer(C.LAYER, feature_path, teacher_path)
    tail = tail_mod.build_tail(C.LAYER, device)
    resident = engine.ResidentSet(feature_path, teacher_path, rows, device)
    if resident.count != C.N_MEASURE:
        raise SystemExit(f"INVALID_EXPERIMENT: measure has {resident.count} "
                         f"images, expected {C.N_MEASURE}")
    log(f"measure resident: {resident.count} images, {resident.tokens} tokens")

    entries, bootstraps = [], {}
    for plan in plans:
        anchor = plan["anchor"]
        codec, meta = frozen.load_checked(anchor, device)
        tail_mod.check_layer(C.LAYER, meta["feature_cache"])
        matrix = engine.evaluate_allocations(
            codec, tail, resident,
            np.vstack([plan["uniform"][None], plan["candidate"][None]]),
            image_batch=C.EVAL_IMAGE_BATCH, pair_budget=C.EVAL_PAIR_BUDGET,
            per_image=True)
        np.save(anchor.root / "measure_distortion.npy", matrix.astype(np.float32))

        uniform_d = matrix[0].astype(np.float64)
        candidate_d = matrix[1].astype(np.float64)
        differences = candidate_d - uniform_d
        bootstraps[anchor.name], entry = one_sided(differences)
        entry.update({
            "anchor": anchor.name, "rate": anchor.rate,
            "checkpoint_id": meta["checkpoint_id"],
            "down_group": plan["record"]["candidate"]["down_group"],
            "up_group": plan["record"]["candidate"]["up_group"],
            "measure_uniform_mean": float(uniform_d.mean()),
            "measure_candidate_mean": float(candidate_d.mean()),
            "relative_change": float(differences.mean() / uniform_d.mean()),
            "images_improved": int(np.sum(differences < 0)),
            "noise_floor_bound": float(uniform_d.mean() * C.REPLAY_REL_TOL),
            "effect_over_noise": float(
                abs(differences.mean())
                / (uniform_d.mean() * C.REPLAY_REL_TOL)),
            "cal_gain_vs_uniform": plan["record"]["candidate"]["cal_gain_vs_uniform"],
            "cal_mean_uniform": plan["record"]["uniform"]["cal_mean"],
            "cal_mean_candidate": plan["record"]["candidate"]["cal_mean"]})
        entries.append(entry)
        log(f"[{anchor.name}] measure paired mean {entry['mean']:+.4f} "
            f"({100 * entry['relative_change']:+.3f}%)  p {entry['p_value']:.5f}")
        del codec
        torch.cuda.empty_cache()

    ordered = holm(entries)
    for entry in ordered:
        means = bootstraps[entry["anchor"]]
        entry["ucb"] = float(np.quantile(means, 1.0 - entry["holm_alpha"]))
        entry["passes"] = bool(entry["rejects_at_holm_alpha"] and entry["ucb"] < 0)

    result = {
        "plan": "v9", "stage": "unseal", "split": "measure",
        "images": int(resident.count), "fwer": C.FWER,
        "family": [e["anchor"] for e in ordered],
        "unsealed_at_utc": token["unsealed_at_utc"],
        "anchors": ordered,
        "verdict": ("PASS" if any(e["passes"] for e in ordered) else "FAIL"),
        "seconds": time.time() - started}
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    for entry in ordered:
        log(f"[{entry['anchor']}] Holm rank {entry['holm_rank']} "
            f"alpha {entry['holm_alpha']:.4f}  UCB {entry['ucb']:+.4f}  "
            f"{'PASS' if entry['passes'] else 'not rejected'}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-unseal", action="store_true", required=True,
                    help="measure-3000 may be opened exactly once")
    args = ap.parse_args()
    if not args.confirm_unseal:
        raise SystemExit("refusing to unseal without --confirm-unseal")
    if RESULT_PATH.exists():
        raise SystemExit(f"INVALID_EXPERIMENT: {RESULT_PATH} exists; "
                         f"measure-3000 has already been opened and re-running "
                         f"it would be a second look")
    result = unseal(torch.device("cuda"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
