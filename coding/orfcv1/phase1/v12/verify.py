"""Mechanical verification of a completed continuous joint-training run."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from . import config as C
from .allocation_policy import FixedBudgetAllocationPolicy


def verify(anchor, run_id, samples=4096):
    root = C.output_dir(anchor, C.ensure_run_id(run_id))
    required = [root / name for name in (
        "train.json", "codec.pt", "policy.pt", "allocation.npy",
        "policy_summary.json")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing v12 artifacts: {missing}")

    record = json.loads((root / "train.json").read_text())
    saved = torch.load(root / "policy.pt", map_location="cpu")
    policy = FixedBudgetAllocationPolicy(
        int(saved["groups"]), tuple(saved["bit_costs"]),
        int(saved["total_bits"]))
    policy.load_state_dict(saved["policy_state"])
    distribution = policy.build(float(saved["temperature"]))
    allocation = distribution.map_allocation()
    disk_allocation = torch.from_numpy(np.load(root / "allocation.npy")).long()
    if not torch.equal(allocation, disk_allocation):
        raise SystemExit("saved allocation is not the policy MAP")

    generator = torch.Generator().manual_seed(C.POLICY_SEED + 1)
    draws = distribution.sample(int(samples), generator=generator)
    rates = policy.actual_rate(draws)
    marginals = distribution.marginals().detach()
    expected_rate = float((
        marginals * torch.tensor(policy.bit_costs, dtype=marginals.dtype)).sum())
    payload = {
        "passed": bool(
            (rates == anchor.rate).all()
            and int(policy.actual_rate(allocation)) == anchor.rate
            and torch.allclose(marginals.sum(1), torch.ones(policy.groups),
                               atol=1e-5, rtol=1e-5)
            # The DP is trained in fp32; summing 96 marginals can accumulate
            # about 1e-4 roundoff although every hard draw is exactly on rate.
            and abs(expected_rate - anchor.rate) < 1e-3
            and record["hard_parity_initial"]["rel_gap"] <= C.REPLAY_REL_TOL
            and record["hard_parity_final"]["rel_gap"] <= C.REPLAY_REL_TOL
            and record["orthogonality_final"] - record["orthogonality_initial"]
                <= C.ORTH_TOL
            and record.get("joint_training", {}).get("protocol_valid", False)),
        "anchor": anchor.name,
        "run_id": run_id,
        "sample_count": int(samples),
        "sample_rate_min": int(rates.min()),
        "sample_rate_max": int(rates.max()),
        "map_rate": int(policy.actual_rate(allocation)),
        "expected_rate": expected_rate,
        "marginal_row_sum_max_error": float((marginals.sum(1) - 1).abs().max()),
        "map_matches_disk": True,
        "joint_training": record.get("joint_training"),
    }
    (root / "verify.json").write_text(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(json.dumps(payload, indent=2))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", required=True, choices=list(C.ANCHOR_BY_NAME))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--samples", type=int, default=4096)
    args = parser.parse_args(argv)
    payload = verify(C.ANCHOR_BY_NAME[args.anchor], args.run_id, args.samples)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
