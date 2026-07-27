#!/usr/bin/env python
"""Validation-only selector for ORFC-v1.1 (§15.5).

The selector never reads test metrics.  It first enforces the 5% validation
D0 gate relative to the frozen beta=0 baseline, then minimises full-group
M(alpha=1) pairwise dispersion.  Fixed-energy dispersion and D0 are the
registered tie-breakers.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _row(path):
    with open(path) as f:
        data = json.load(f)
    cfg = data.get("config", {})
    heldout = data.get("heldout_metrics", {}).get("val", {})
    response = data.get("response_config", {})
    alphas = response.get("alpha_list", cfg.get("alpha_list", [0.1, 0.5, 1.0]))
    alpha_main = max(float(a) for a in alphas)
    m = heldout.get(f"M_g_alpha{alpha_main}", {})
    s = heldout.get(f"S_g_alpha{alpha_main}", {})
    return {
        "path": str(path),
        "checkpoint": data.get("checkpoint"),
        "objective": response.get(
            "response_objective", cfg.get("response_objective")),
        "alpha_list": [float(a) for a in alphas],
        "alpha_weights": response.get("alpha_weights"),
        "requested_beta": float(response.get("beta", cfg.get("beta", 0.0))),
        "effective_beta": float(response.get(
            "effective_beta", response.get("beta", cfg.get("beta", 0.0)))),
        "beta_target_ratio": float(response.get(
            "beta_target_ratio", cfg.get("beta_target_ratio", 0.0))),
        "freeze_codebooks": bool(cfg.get("freeze_codebooks", False)),
        "evaluation_only": bool(data.get("evaluation_only", False)),
        "val_D0": heldout.get("D0_raw_mean"),
        "M_pairwise": m.get("pairwise_dispersion"),
        "S_pairwise": s.get("pairwise_dispersion"),
        "test_fields_present": bool(
            data.get("v1_1_acc") is not None
            or data.get("heldout_metrics", {}).get("test")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_d0_regret", type=float, default=0.05)
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.result_dir).glob("*.json")):
        if path.name in {"selection_v1_1.json", "split_manifest_s42.json"}:
            continue
        try:
            rows.append(_row(path))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

    # Test-containing files are excluded even if they happen to live beside
    # validation results.
    eligible_rows = [r for r in rows if not r["test_fields_present"]]
    baselines = [
        r for r in eligible_rows
        if r["objective"] == "legacy"
        and r["effective_beta"] == 0.0
        and not r["freeze_codebooks"]
        and r["val_D0"] is not None
    ]
    if len(baselines) != 1:
        raise SystemExit(
            f"expected exactly one validation beta=0 baseline, got "
            f"{len(baselines)}")
    baseline = baselines[0]
    d0_limit = baseline["val_D0"] * (1.0 + args.max_d0_regret)

    candidates = [
        r for r in eligible_rows
        if r["objective"] in {"legacy", "fixed_energy", "operational"}
        and r["effective_beta"] > 0
        and not r["freeze_codebooks"]
        and r["checkpoint"]
        and r["val_D0"] is not None
        and r["val_D0"] <= d0_limit
        and r["M_pairwise"] is not None
        and r["S_pairwise"] is not None
    ]
    if not candidates:
        raise SystemExit(
            f"no candidate passed validation D0 <= {d0_limit:.6g}")

    chosen = min(
        candidates,
        key=lambda r: (r["M_pairwise"], r["S_pairwise"], r["val_D0"]))
    payload = {
        "selection_uses_test": False,
        "rule": (
            "val_D0 <= beta0*(1+0.05); minimise M_alpha1 pairwise; "
            "tie-break by S_alpha1 pairwise then val_D0"),
        "baseline": baseline,
        "d0_limit": d0_limit,
        "n_candidates_before_gate": len([
            r for r in eligible_rows
            if r["effective_beta"] > 0 and not r["freeze_codebooks"]]),
        "n_candidates_after_gate": len(candidates),
        "chosen": chosen,
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
