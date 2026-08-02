"""Audit the paired v12 versus exact-budget-coverage short runs."""

import argparse
import json
from pathlib import Path

from ..v12 import config as C


def load(run_id, anchor):
    path = C.output_dir(C.ANCHOR_BY_NAME[anchor], run_id) / "train.json"
    return json.loads(path.read_text()), path


def run(baseline_run, coverage_run, output):
    result = {"baseline_run": baseline_run, "coverage_run": coverage_run,
              "anchors": {}, "passed": True}
    for name in ("R64", "R96"):
        base, base_path = load(baseline_run, name)
        cover, cover_path = load(coverage_run, name)
        bjoint, cjoint = base["joint_training"], cover["joint_training"]
        contract = cjoint["exact_budget_coverage"]
        initial_gap = abs(base["initial_distortion"] - cover["initial_distortion"])
        initial_tol = max(base["initial_distortion"], 1.0) * C.REPLAY_REL_TOL
        checks = {
            "same_initial_point": initial_gap <= initial_tol,
            "baseline_coverage_disabled": not bjoint[
                "exact_budget_coverage"]["enabled"],
            "coverage_enabled": contract["enabled"],
            "policy_excludes_coverage": not contract[
                "policy_gradient_uses_coverage"],
            "forced_cycle_complete": contract["forced_exposure_min"] > 0,
            "allocation_coverage_complete": contract[
                "allocation_exposure_min"] > 0,
            "last_gradient_window_covered": cjoint[
                "last_window_gradient_coverage_min"] > 0,
            "all_codebooks_updated": cjoint["codebook_relative_drift_min"] > 0,
            "exact_rate": cover["policy"]["map_rate"] == cover["rate"],
            "hard_parity": cover["hard_parity_final"]["rel_gap"] <= cover[
                "hard_parity_final"]["tolerance"],
            "orthogonality": cover["orthogonality_final"]
                - cover["orthogonality_initial"] <= C.ORTH_TOL,
            "hard_map_improves_from_initial": cover["validation"][-1][
                "map_distortion"] < cover["initial_distortion"],
        }
        passed = all(checks.values())
        result["anchors"][name] = {
            "baseline": str(base_path), "coverage": str(cover_path),
            "checks": checks, "passed": passed,
            "initial_distortion": cover["initial_distortion"],
            "baseline_final_map": base["validation"][-1]["map_distortion"],
            "coverage_final_map": cover["validation"][-1]["map_distortion"],
            "coverage_minus_baseline": cover["validation"][-1][
                "map_distortion"] - base["validation"][-1]["map_distortion"],
            "forced_exposure_min": contract["forced_exposure_min"],
            "last_gradient_coverage_min": cjoint[
                "last_window_gradient_coverage_min"],
        }
        result["passed"] &= passed
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--coverage-run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.baseline_run, args.coverage_run, args.output)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

