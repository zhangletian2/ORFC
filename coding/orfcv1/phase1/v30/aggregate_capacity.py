"""Aggregate eight independently executed V30 capacity-gate arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CONFIG_KEYS = (
    "plan", "block", "anchor", "source_codec", "mode_bits",
    "parameterization", "stage_sizes", "branch_sizes", "fixed_u",
    "epochs", "batch", "lr", "lr_schedule", "tau_schedule",
    "same_batch_order", "same_updates", "train_images",
    "updates_per_arm", "validation_select", "check_every",
    "convergence_contract", "centroid_update_contract", "threshold")


def load_anchor(paths):
    rows = [json.loads(Path(path).read_text()) for path in paths]
    if len(rows) != 4:
        raise ValueError("each anchor requires exactly four arm files")
    reference = {key: rows[0][key] for key in CONFIG_KEYS}
    for row in rows:
        if row.get("verdict") != "ARM_COMPLETE" or len(row["arms"]) != 1:
            raise ValueError("input is not one completed arm")
        if {key: row[key] for key in CONFIG_KEYS} != reference:
            raise ValueError("arm configurations differ")
    records = sorted((row["arms"][0] for row in rows),
                     key=lambda item: item["arm"])
    if [record["arm"] for record in records] != list(range(4)):
        raise ValueError("arm indices must be exactly 0,1,2,3")
    all_converged = all(
        record["convergence"][side]["converged"]
        for record in records for side in ("independent", "nested"))
    worst = max(record["paired_capacity_loss"]["one_sided_ucb95"]
                for record in records)
    threshold = float(reference["threshold"])
    verdict = ("INCONCLUSIVE" if not all_converged else
               "PASS" if worst <= threshold else "FAIL")
    result = dict(reference)
    result.update({
        "allocations": rows[0]["allocations"], "arms": records,
        "all_sides_converged": all_converged,
        "worst_one_sided_ucb95": worst, "verdict": verdict,
        "distributed_execution": True,
        "arm_files": [str(Path(path).resolve()) for path in paths],
        "peak_memory_bytes": max(row["peak_memory_bytes"] for row in rows),
        "seconds_before_optional_export": max(
            row["seconds_before_optional_export"] for row in rows)})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r64", nargs=4, required=True)
    parser.add_argument("--r96", nargs=4, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    anchors = {"R64": load_anchor(args.r64), "R96": load_anchor(args.r96)}
    for name, result in anchors.items():
        (out / f"{name}_capacity_gate.json").write_text(
            json.dumps(result, indent=2))
    verdicts = [result["verdict"] for result in anchors.values()]
    verdict = ("FAIL" if "FAIL" in verdicts else
               "INCONCLUSIVE" if "INCONCLUSIVE" in verdicts else "PASS")
    summary = {
        "plan": "v30_hierarchical_tree_capacity_gate_distributed",
        "anchors": {name: {
            "verdict": result["verdict"],
            "all_sides_converged": result["all_sides_converged"],
            "worst_one_sided_ucb95": result["worst_one_sided_ucb95"],
            "threshold": result["threshold"]}
            for name, result in anchors.items()},
        "verdict": verdict}
    (out / "capacity_gate_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
