"""Stage D: seal the arms, then open holdout-500 exactly once.

Two commands, in this order and no other:

``--seal``
    Copies, into ``v10/SEAL/``, the things the result must not be allowed to
    move afterwards: every arm's ``codec_ref.npz`` (the trained ``U`` and every
    codebook), its final allocation, its ``train.json``, the frozen profile, the
    holdout image list, and a byte copy of every ``phase1/v10/*.py``.  Then it
    writes ``SEAL.json``.  Copies rather than digests: the standing convention in
    this project is checkpoint id + metadata + a direct ``array_equal`` on the
    key tensors, which says *which* tensor moved and by how much, where a hash
    only says that something did.

``--confirm-unseal``
    Re-compares every sealed artifact against what is on disk now, writes the
    unseal token, and only then reads holdout-500.  Exactly two numbers per
    anchor reach the confirmatory statistic:

        d_n = D_n(nonuniform-joint) - D_n(uniform-equal-step),  n = 1..500

    tested one-sided by the pre-registered paired bootstrap, Holm-corrected
    across the {R64, R96} family at FWER 0.05.  The per-image resampling *is*
    the stratified resampling the protocol asks for, because W2 established
    holdout-500 is one image per class -- and W2 is re-checked here rather than
    recalled, since a changed pool would silently change what the interval means.

    A3 (uniform-equal-compute) is evaluated in the same pass and reported with
    an uncorrected interval, labelled secondary.  It answers a different
    question -- does the gain survive giving the uniform arm the compute A2 spent
    searching -- and folding it into the primary family would spend alpha on a
    robustness check the user asked for as a separate report.

There is no argmin here, no allocation scan, no second look.  holdout-500 was
never read by v8, v9, Stage B or Stage C; measure-3000 is not touched at all.
"""

import argparse
import json
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import torch

from codec_v1 import load_codec_v1

from . import build_holdout
from . import config as C
from . import stats
from .. import engine
from ..tail import build_tail, check_layer

SEAL_DIR = C.V10 / "SEAL"


def source_files():
    """Every v10 source file, by name.  ``__pycache__`` is not source."""
    root = C.REPO / "phase1" / "v10"
    return sorted(p for p in root.glob("*.py"))


def arm_artifacts(anchor, arm):
    run = C.run_dir(anchor, arm)
    return {"codec_ref.npz": run / "codec_ref.npz",
            "allocation.npy": run / "allocation.npy",
            "train.json": run / "train.json",
            "codec.pt": run / "codec.pt"}


def seal(log=print):
    if C.SEAL_PATH.exists():
        raise SystemExit(f"INVALID_EXPERIMENT: {C.SEAL_PATH} already exists; "
                         f"re-sealing after the arms have been inspected would "
                         f"be choosing what to seal")
    if C.UNSEAL_TOKEN.exists():
        raise SystemExit(f"INVALID_EXPERIMENT: holdout-500 has already been "
                         f"opened; there is nothing left to seal")

    record = {"plan": "v10", "stage": "D_seal",
              "sealed_at": time.time(),
              "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
              "anchors": {}, "code": [], "protocol": {
                  "primary": "d_n = D_n(nonuniform-joint) - "
                             "D_n(uniform-equal-step) on holdout-500",
                  "family": [a.name for a in C.ANCHORS],
                  "correction": "Holm", "fwer": C.FWER,
                  "bootstrap_resamples": C.BOOTSTRAP_RESAMPLES,
                  "bootstrap_seed": C.BOOTSTRAP_SEED,
                  "resampling_unit": "image (W2: one image per class)",
                  "secondary": "D(nonuniform-joint) - D(uniform-equal-compute), "
                               "uncorrected, reported separately"}}

    for anchor in C.ANCHORS:
        entry = {"rate": anchor.rate, "arms": {}}
        profile = C.profile_path(anchor)
        if not profile.exists():
            raise SystemExit(f"INVALID_EXPERIMENT: [{anchor.name}] no frozen "
                             f"profile at {profile}")
        target = SEAL_DIR / anchor.name
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(profile, target / "profile.json")
        for arm in C.ARMS:
            files = arm_artifacts(anchor, arm)
            missing = [k for k, p in files.items() if not p.exists()]
            if missing:
                raise SystemExit(f"INVALID_EXPERIMENT: [{anchor.name}/{arm}] "
                                 f"missing {missing}; every arm must have "
                                 f"finished before the seal is written")
            destination = target / arm
            destination.mkdir(parents=True, exist_ok=True)
            for name, path in files.items():
                if name != "codec.pt":       # copied by reference, not by bytes
                    shutil.copy2(path, destination / name)
            payload = json.loads(files["train.json"].read_text())
            entry["arms"][arm] = {
                "allocation": payload["final_allocation"],
                "nominal_rate": payload["nominal_rate"],
                "steps": payload["steps"],
                "checkpoint_id": payload["checkpoint_id"],
                "G1_pass": payload["G1_pass"],
                "codec_pt": str(files["codec.pt"]),
                "codec_pt_mtime": files["codec.pt"].stat().st_mtime}
            if payload["nominal_rate"] != anchor.rate:
                raise SystemExit(f"INVALID_EXPERIMENT (W6): "
                                 f"[{anchor.name}/{arm}] sealed allocation has "
                                 f"nominal rate {payload['nominal_rate']}")
        record["anchors"][anchor.name] = entry
        log(f"[{anchor.name}] sealed {len(entry['arms'])} arms")

    snapshot = C.CODE_SNAPSHOT
    snapshot.mkdir(parents=True, exist_ok=True)
    for path in source_files():
        shutil.copy2(path, snapshot / path.name)
        record["code"].append(path.name)
    shutil.copy2(C.V10 / "holdout_names.txt", SEAL_DIR / "holdout_names.txt")
    log(f"code snapshot: {len(record['code'])} files")

    C.SEAL_PATH.write_text(json.dumps(record, indent=2))
    log(f"sealed at {record['sealed_at_utc']} -> {C.SEAL_PATH}")
    return record


def check_seal(log=print):
    """W10 and the checkpoint invariant: nothing sealed has moved."""
    if not C.SEAL_PATH.exists():
        raise SystemExit(f"INVALID_EXPERIMENT: {C.SEAL_PATH} does not exist; "
                         f"holdout-500 may not be opened before the arms are "
                         f"sealed")
    record = json.loads(C.SEAL_PATH.read_text())

    for name in record["code"]:                                    # W10
        live = C.REPO / "phase1" / "v10" / name
        sealed = C.CODE_SNAPSHOT / name
        if not live.exists():
            raise SystemExit(f"INVALID_EXPERIMENT (W10): {name} was sealed but "
                             f"no longer exists")
        if live.read_bytes() != sealed.read_bytes():
            raise SystemExit(f"INVALID_EXPERIMENT (W10): {name} differs from "
                             f"the sealed snapshot; the code that produced the "
                             f"arms is not the code running now")
    log(f"  W10: {len(record['code'])} source files byte-identical to the seal")

    for anchor in C.ANCHORS:
        for arm in C.ARMS:
            sealed = np.load(SEAL_DIR / anchor.name / arm / "codec_ref.npz")
            live = np.load(C.run_dir(anchor, arm) / "codec_ref.npz")
            for key in sealed.files:
                if not np.array_equal(sealed[key], live[key]):
                    delta = float(np.abs(sealed[key].astype(np.float64)
                                         - live[key].astype(np.float64)).max())
                    raise SystemExit(
                        f"INVALID_EXPERIMENT: [{anchor.name}/{arm}] {key} "
                        f"differs from the sealed copy (max |d| = {delta:.3e})")
            allocation = np.load(C.run_dir(anchor, arm) / "allocation.npy")
            if allocation.tolist() != record["anchors"][anchor.name]["arms"][arm]["allocation"]:
                raise SystemExit(f"INVALID_EXPERIMENT: [{anchor.name}/{arm}] "
                                 f"the allocation on disk is not the sealed one")
        log(f"  [{anchor.name}] all three arms match their sealed tensors "
            f"elementwise")
    return record


def load_arm(anchor, arm, device):
    """The trained codec, checked elementwise against its sealed reference."""
    run = C.run_dir(anchor, arm)
    codec = load_codec_v1(run / "codec.pt", device=device).eval()
    reference = np.load(SEAL_DIR / anchor.name / arm / "codec_ref.npz")
    rotation = codec.transform.get_rotation().detach().cpu().numpy()
    if not np.array_equal(rotation, reference["U"]):
        raise SystemExit(f"INVALID_EXPERIMENT: [{anchor.name}/{arm}] the U in "
                         f"codec.pt is not the sealed U (max |d| = "
                         f"{np.abs(rotation - reference['U']).max():.3e})")
    for mode, quantizer in enumerate(codec.pq.quantizers):
        book = quantizer.codebooks.detach().cpu().numpy()
        want = reference[f"codebook_{mode}"]
        if not np.array_equal(book, want):
            raise SystemExit(f"INVALID_EXPERIMENT: [{anchor.name}/{arm}] "
                             f"codebook {mode} in codec.pt is not the sealed "
                             f"one (max |d| = {np.abs(book - want).max():.3e})")
    allocation = np.load(run / "allocation.npy")
    if engine.nominal_rate(allocation, anchor) != anchor.rate:      # W6
        raise SystemExit(f"INVALID_EXPERIMENT (W6): [{anchor.name}/{arm}] "
                         f"nominal rate "
                         f"{engine.nominal_rate(allocation, anchor)}")
    return codec, allocation


def unseal(device, log=print):
    started = time.time()
    engine.configure_precision(C.ALLOW_TF32)
    log("=== seal check ===")
    record = check_seal(log=log)

    log("=== W2, recomputed before the interval is claimed to be stratified ===")
    names = build_holdout.pool_basenames()
    if names != (SEAL_DIR / "holdout_names.txt").read_text().split():
        raise SystemExit("INVALID_EXPERIMENT: the holdout pool is not the "
                         "sealed image list")
    strat = build_holdout.check_stratification(names)
    log(f"  {strat['images']} images, {strat['distinct_classes']} classes, max "
        f"{strat['max_per_class']} per class -> per-image resampling is the "
        f"stratified resampling")

    token = {"plan": "v10", "stage": "D_unseal",
             "unsealed_at": time.time(),
             "unsealed_at_utc": datetime.now(timezone.utc).isoformat(),
             "sealed_at_utc": record["sealed_at_utc"],
             "protocol": record["protocol"],
             "stratification": strat}
    if token["unsealed_at"] <= record["sealed_at"]:
        raise SystemExit("INVALID_EXPERIMENT: the unseal token is not later "
                         "than the seal")
    C.UNSEAL_TOKEN.write_text(json.dumps(token, indent=2))
    log(f"holdout-500 unsealed at {token['unsealed_at_utc']}")

    check_layer(C.LAYER, C.HOLDOUT_FEATURES, C.HOLDOUT_TEACHERS)    # W3
    tail = build_tail(C.LAYER, device)
    resident = engine.ResidentSet(C.HOLDOUT_FEATURES, C.HOLDOUT_TEACHERS,
                                  np.arange(C.N_HOLDOUT), device)
    engine.assert_uniform_shape(resident.count)

    entries, bootstraps, secondary = [], {}, []
    for anchor in C.ANCHORS:
        per_arm = {}
        for arm in C.ARMS:
            codec, allocation = load_arm(anchor, arm, device)
            values = engine.evaluate_allocations(
                codec, tail, resident, allocation[None, :],
                image_batch=C.EVAL_IMAGE_BATCH,
                pair_budget=C.EVAL_PAIR_BUDGET, per_image=True)[0]
            per_arm[arm] = values.astype(np.float64)
            del codec
            torch.cuda.empty_cache()
            log(f"  [{anchor.name}/{arm}] holdout mean "
                f"{per_arm[arm].mean():.4f}")
        out = C.V10 / anchor.name
        np.save(out / "holdout_distortion.npy",
                np.stack([per_arm[arm] for arm in C.ARMS]).astype(np.float32))

        differences = per_arm[C.ARM_JOINT] - per_arm[C.ARM_EQUAL_STEP]
        np.save(out / "holdout_differences.npy", differences)
        means, entry = stats.one_sided(differences)
        baseline = float(per_arm[C.ARM_EQUAL_STEP].mean())
        entry.update({
            "anchor": anchor.name, "rate": anchor.rate,
            "A1_mean": baseline,
            "A2_mean": float(per_arm[C.ARM_JOINT].mean()),
            "A3_mean": float(per_arm[C.ARM_EQUAL_COMPUTE].mean()),
            "relative_change": float(differences.mean() / baseline),
            "noise_floor_bound": baseline * C.REPLAY_REL_TOL,
            "effect_over_noise": float(abs(differences.mean())
                                       / (baseline * C.REPLAY_REL_TOL)),
            "A2_allocation":
                record["anchors"][anchor.name]["arms"][C.ARM_JOINT]["allocation"],
        })
        entries.append(entry)
        bootstraps[anchor.name] = means

        compute = per_arm[C.ARM_JOINT] - per_arm[C.ARM_EQUAL_COMPUTE]
        compute_means, compute_entry = stats.one_sided(compute)
        compute_entry.update({
            "anchor": anchor.name,
            "comparison": "nonuniform-joint minus uniform-equal-compute",
            "relative_change": float(
                compute.mean() / per_arm[C.ARM_EQUAL_COMPUTE].mean()),
            "ci_uncorrected_upper": float(np.quantile(compute_means,
                                                      1.0 - C.FWER)),
            "note": "secondary, uncorrected; A3 spends A2's outer-search FLOPs "
                    "on extra uniform training"})
        secondary.append(compute_entry)

    ordered = stats.finish(entries, bootstraps)
    result = {
        "plan": "v10", "stage": "D_unseal", "split": "holdout-500",
        "images": int(resident.count), "fwer": C.FWER,
        "family": [e["anchor"] for e in ordered],
        "sealed_at_utc": record["sealed_at_utc"],
        "unsealed_at_utc": token["unsealed_at_utc"],
        "primary": ordered,
        "secondary_equal_compute": secondary,
        "verdict": "PASS" if any(e["passes"] for e in ordered) else "FAIL",
        "seconds": time.time() - started,
    }
    C.UNSEAL_PATH.write_text(json.dumps(result, indent=2))
    for entry in ordered:
        log(f"[{entry['anchor']}] D(A2)-D(A1) = {entry['mean']:+.4f} "
            f"({100 * entry['relative_change']:+.4f}%)  p {entry['p_value']:.5f}"
            f"  Holm alpha {entry['holm_alpha']:.4f}  UCB {entry['ucb']:+.4f}  "
            f"{'PASS' if entry['passes'] else 'not rejected'}")
    for entry in secondary:
        log(f"[{entry['anchor']}] secondary D(A2)-D(A3) = {entry['mean']:+.4f} "
            f"({100 * entry['relative_change']:+.4f}%)  "
            f"uncorrected upper {entry['ci_uncorrected_upper']:+.4f}")
    log(f"v10 verdict: {result['verdict']}")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seal", action="store_true")
    ap.add_argument("--confirm-unseal", action="store_true",
                    help="holdout-500 may be opened exactly once")
    args = ap.parse_args(argv)

    if args.seal:
        print(json.dumps(seal(), indent=2, default=str))
        return 0
    if not args.confirm_unseal:
        raise SystemExit("pass --seal or --confirm-unseal")
    if C.UNSEAL_PATH.exists() or C.UNSEAL_TOKEN.exists():
        raise SystemExit(f"INVALID_EXPERIMENT: holdout-500 has already been "
                         f"opened ({C.UNSEAL_PATH}); re-running would be a "
                         f"second look at the confirmatory set")
    result = unseal(torch.device("cuda"))
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("primary", "secondary_equal_compute")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
