"""Audit DP-ST gradients against exact-rate hard neighbor differences."""

import argparse
import json

import numpy as np
import torch

from codec_v1 import load_codec_v1

from . import config as C
from . import qhard
from .allocation_policy import FixedBudgetAllocationPolicy
from .train import fixed_rate_neighbors, validate_many
from .. import engine
from .. import tail as tail_mod


def _rank(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _corr(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if len(left) < 2 or left.std() == 0 or right.std() == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def run(anchor, run_id, device):
    root = C.output_dir(anchor, run_id)
    record = json.loads((root / "train.json").read_text())
    if record.get("policy_gradient") != "dp_st":
        raise SystemExit("audit requires a dp_st training run")
    codec = load_codec_v1(root / "codec.pt", device=device)
    policy = FixedBudgetAllocationPolicy(
        C.GROUPS, anchor.mode_bits, anchor.rate).to(device)
    saved = torch.load(root / "policy.pt", map_location=device)
    policy.load_state_dict(saved["policy_state"])
    tail = tail_mod.build_tail(C.LAYER, device)
    paths = C.load_split("train_val")
    resident = engine.ResidentSet(
        *paths[:3], device, max_images=int(record["val_images"]))
    temperature = float(record["temperature"])
    codeword_temperature = float(record["codeword_temperature"])
    distribution = policy.build(temperature)
    allocation = distribution.map_allocation()
    neighbors, moves = fixed_rate_neighbors(policy, allocation)
    base = allocation.detach().cpu().numpy()[None]
    values = validate_many(
        codec, tail, resident, np.concatenate((base, neighbors)))
    actual = values[1:] - values[0]

    gradient = torch.zeros_like(policy.logits)
    for start in range(0, resident.count, C.EVAL_IMAGE_BATCH):
        distribution = policy.build(temperature)
        marginals = distribution.marginals()
        loss, _ = qhard.distortions(
            codec, tail, *resident.slice(start, start + C.EVAL_IMAGE_BATCH),
            allocation[None], marginals=marginals,
            codeword_temperature=codeword_temperature)
        partial, = torch.autograd.grad(
            loss.sum() / resident.count, policy.logits)
        gradient.add_(partial.detach())
    base = base[0]
    predicted = []
    for candidate in neighbors:
        changed = np.flatnonzero(candidate != base)
        predicted.append(sum(
            float(gradient[group, candidate[group]] - gradient[group, base[group]])
            for group in changed))
    predicted = np.asarray(predicted)

    validation = [item for item in record["validation"]
                  if "map_distortion" in item and "entropy" in item]
    distortion = np.asarray([item["map_distortion"] for item in validation])
    entropy = np.asarray([item["entropy"] for item in validation])
    falling_entropy = np.diff(entropy) < 0
    synchronized = np.diff(distortion)[falling_entropy] <= 0
    best_actual = int(np.argmin(actual))
    best_predicted = int(np.argmin(predicted))
    result = {
        "anchor": anchor.name, "run_id": run_id,
        "images": resident.count, "neighbors": int(len(neighbors)),
        "gradient_actual_pearson": _corr(predicted, actual),
        "gradient_actual_spearman": _corr(_rank(predicted), _rank(actual)),
        "gradient_sign_agreement": float(
            np.mean(np.sign(predicted) == np.sign(actual))),
        "predicted_best_move": list(moves[best_predicted]),
        "actual_best_move": list(moves[best_actual]),
        "actual_best_delta": float(actual[best_actual]),
        "predicted_at_actual_best": float(predicted[best_actual]),
        "entropy_distortion_pearson": _corr(entropy, distortion),
        "entropy_fall_intervals": int(falling_entropy.sum()),
        "entropy_distortion_synchronous_fraction": float(
            synchronized.mean()) if len(synchronized) else float("nan"),
        "hard_distortion_initial": float(distortion[0]),
        "hard_distortion_final": float(distortion[-1]),
        "gradient_coverage_min": record["joint_training"][
            "gradient_coverage_min"],
        "last_window_gradient_coverage_min": record["joint_training"][
            "last_window_gradient_coverage_min"],
    }
    (root / "dp_st_audit.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", required=True, choices=list(C.ANCHOR_BY_NAME))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(
        C.ANCHOR_BY_NAME[args.anchor], args.run_id,
        torch.device(args.device)), indent=2))


if __name__ == "__main__":
    main()
