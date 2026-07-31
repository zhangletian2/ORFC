"""The implementation invariants of plan v9 section 3, as executable checks.

Run once per anchor, after `build.py` and before `cal_sweep.py`.  A failure here
is `INVALID_EXPERIMENT` -- the round produces no number at all -- rather than a
result to be interpreted, which is the whole point of separating this file from
the sweep.

    V1  every one of the 993 evaluated allocations has nominal rate exactly R
    V2  each candidate differs from uniform in exactly one down and one up group,
        and the 992 (down, up) pairs are distinct and lexicographically ordered
    V3  ||U_0^T U_0 - I||_F, in float64, under tolerance
    V4  codec provenance: metadata, U_0 and every codebook equal to the
        reference written at build time; codebook shapes (G, K_m, d)
    V5  hard-replay reproducibility at the frozen call shape -- repeated and
        re-issued evaluations of one allocation must agree to within the
        measured fp32 noise floor (config.REPLAY_REL_TOL, ~8 ULP).  The probe is
        saved so `cal_sweep.py` can repeat it in *its* process and close the
        cross-process half of the check
    V6  the engine carries no soft path at all
    V7  the four splits are disjoint and measure-3000 is still sealed
    V8  TF32 is off

The check that is deliberately absent is any comparison of distortions between
allocations.  Nothing in this file may look at which allocation is better; that
is cal's job and then measure's, once.
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


class Failure(Exception):
    pass


def check_rates(anchor, allocations, pairs):
    """V1 and V2, on the exact array the sweep will consume."""
    uniform = engine.uniform_allocation(anchor)
    rates = np.array([engine.nominal_rate(a, anchor)
                      for a in np.vstack([uniform[None], allocations])])
    if not np.all(rates == anchor.rate):
        bad = np.flatnonzero(rates != anchor.rate)
        raise Failure(f"{len(bad)} allocations have nominal rate != "
                      f"{anchor.rate} (first offender: index {bad[0]}, "
                      f"rate {rates[bad[0]]})")
    if len(allocations) != C.N_CANDIDATES:
        raise Failure(f"{len(allocations)} candidates, expected "
                      f"{C.N_CANDIDATES}")

    differs = allocations != uniform[None]
    if not np.all(differs.sum(1) == 2):
        raise Failure("some candidate differs from uniform in a number of "
                      "groups other than 2")
    down_ok = allocations[np.arange(len(pairs)), pairs[:, 0]] == anchor.down_mode
    up_ok = allocations[np.arange(len(pairs)), pairs[:, 1]] == anchor.up_mode
    if not (down_ok.all() and up_ok.all()):
        raise Failure("a candidate's (down, up) groups do not carry the "
                      "down/up modes")
    if len({tuple(p) for p in pairs}) != C.N_CANDIDATES:
        raise Failure("the (down, up) pairs are not distinct")
    order = pairs[:, 0] * C.GROUPS + pairs[:, 1]
    if not np.all(np.diff(order) > 0):
        raise Failure(f"the candidates are not in the frozen tie-break order "
                      f"({C.TIE_BREAK})")
    return {"rate_all": int(rates[0]), "candidates": int(len(allocations))}


def check_engine_is_hard_only():
    """V6.  The soft path was deleted in v9, not merely left uncalled."""
    leaked = [name for name in dir(engine) if "soft" in name.lower()]
    if leaked:
        raise Failure(f"engine exposes soft-path names: {leaked}")
    return {"soft_symbols": 0}


def check_splits():
    """V7.  Disjointness and the seal, read off the frozen manifest."""
    names = {}
    for name in ("train_core", "cal", "dev"):
        _, _, rows, basenames = splits.load_split(name)
        names[name] = set(basenames)
        if len(basenames) != len(names[name]):
            raise Failure(f"{name} contains duplicate basenames")
        if len(rows) != len(basenames):
            raise Failure(f"{name}: {len(rows)} rows vs {len(basenames)} names")
    for a in names:
        for b in names:
            if a < b and names[a] & names[b]:
                raise Failure(f"{a} and {b} share "
                              f"{len(names[a] & names[b])} basenames")
    try:
        splits.load_split("measure")
    except PermissionError:
        pass
    else:
        raise Failure("measure-3000 loaded without an unseal token")
    if splits.is_unsealed():
        raise Failure(f"the unseal token {splits.UNSEAL_TOKEN} already exists "
                      f"before the sweep has run")
    return {name: len(value) for name, value in names.items()}


def replay_probe(codec, tail, resident, anchor):
    """``REPLAY_REPEATS`` identical evaluations of the uniform allocation.

    Every row is the same allocation issued as its own tail call at the frozen
    shape, so the rows differ only by whatever the hardware fails to reproduce.
    """
    rows = np.repeat(engine.uniform_allocation(anchor)[None],
                     C.REPLAY_REPEATS, 0)
    return engine.evaluate_allocations(codec, tail, resident, rows,
                                       per_image=True)


def check_replay(codec, tail, resident, anchor):
    """V5, both halves that are checkable inside one process."""
    first = replay_probe(codec, tail, resident, anchor)
    second = replay_probe(codec, tail, resident, anchor)
    level = float(np.abs(first).mean())
    within = float(np.abs(first - first[0]).max()) / level
    across = float(np.abs(first - second).max()) / level
    mean_drift = float(abs(first.mean(1).astype(np.float64).max()
                           - first.mean(1).astype(np.float64).min())) / level
    if within > C.REPLAY_REL_TOL:
        raise Failure(f"repeated evaluations of one allocation differ by a "
                      f"relative {within:.3e} > {C.REPLAY_REL_TOL:.2e}")
    if across > C.REPLAY_REL_TOL:
        raise Failure(f"two identical passes differ by a relative "
                      f"{across:.3e} > {C.REPLAY_REL_TOL:.2e}")
    np.save(anchor.root / "replay_probe.npy", first)
    return {"repeats": int(first.shape[0]), "images": int(first.shape[1]),
            "distortion_level": level,
            "within_pass_rel": within, "across_pass_rel": across,
            "set_mean_rel_spread": mean_drift,
            "rel_tol": C.REPLAY_REL_TOL,
            "ulp_of_level": float(np.spacing(np.float32(level))),
            "uniform_mean": float(first[0].astype(np.float64).mean())}


def verify(anchor, device, log=print):
    started = time.time()
    engine.configure_precision(C.ALLOW_TF32)
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise Failure("TF32 is enabled")                              # V8

    report = {"plan": "v9", "stage": "verify", "anchor": anchor.name,
              "rate": anchor.rate, "allow_tf32": C.ALLOW_TF32}

    allocations, pairs = engine.swap_candidates(anchor)
    report["V1_V2_allocations"] = check_rates(anchor, allocations, pairs)
    report["V6_hard_only"] = check_engine_is_hard_only()
    report["V7_splits"] = check_splits()

    codec, meta = frozen.load_checked(anchor, device)                 # V4
    report["V4_provenance"] = {
        "checkpoint_id": meta["checkpoint_id"],
        "feature_cache": meta["feature_cache"],
        "codebook_shapes": [list(q.codebooks.shape) for q in codec.pq.quantizers],
        "reused_orfc_codebooks": meta["codebooks"]["reused_orfc_codebooks"],
        "task_tail_pretraining": meta["codebooks"]["task_tail_pretraining"]}

    orth = frozen.orthogonality_error(codec)                          # V3
    if orth > C.OPQ_ORTH_TOL:
        raise Failure(f"||U^T U - I||_F = {orth:.3e} exceeds "
                      f"{C.OPQ_ORTH_TOL:.1e}")
    report["V3_orthogonality"] = orth

    feature_path, teacher_path, rows, _ = splits.load_split("cal")
    tail_mod.check_layer(C.LAYER, feature_path, teacher_path,
                         meta["feature_cache"])
    tail = tail_mod.build_tail(C.LAYER, device)
    resident = engine.ResidentSet(feature_path, teacher_path,
                                  rows[:C.REPLAY_IMAGES], device)
    engine.assert_uniform_shape(resident.count)
    report["V5_replay"] = check_replay(codec, tail, resident, anchor)

    report["seconds"] = time.time() - started
    report["verdict"] = "PASS"
    (anchor.root / "verify.json").write_text(json.dumps(report, indent=2))
    log(f"[{anchor.name}] verify PASS  orth {orth:.2e}  replay within "
        f"{report['V5_replay']['within_pass_rel']:.1e} / across "
        f"{report['V5_replay']['across_pass_rel']:.1e} (relative, tol "
        f"{C.REPLAY_REL_TOL:.1e})")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True, choices=list(C.ANCHOR_BY_NAME))
    args = ap.parse_args()
    anchor = C.ANCHOR_BY_NAME[args.anchor]
    try:
        report = verify(anchor, torch.device("cuda"))
    except Failure as failure:
        raise SystemExit(f"INVALID_EXPERIMENT: [{anchor.name}] {failure}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
