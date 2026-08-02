"""Post-training full two-group neighborhood and centroid-usage audit."""

import argparse
import json

import numpy as np
import torch

from codec_v1 import load_codec_v1
from opq import batch_normalize_gpu

from .config import SPECS, activate
from .. import engine
from .. import tail as tail_mod


def full_neighbors(allocation, bits):
    base = np.asarray(allocation, dtype=np.int64)
    candidates, moves, seen = [], [], set()
    for donor, source in enumerate(base):
        for lower in range(source):
            released = bits[source] - bits[lower]
            for receiver, target in enumerate(base):
                if donor == receiver:
                    continue
                for upper in range(target + 1, len(bits)):
                    if bits[upper] - bits[target] != released:
                        continue
                    candidate = base.copy()
                    candidate[donor], candidate[receiver] = lower, upper
                    key = tuple(candidate)
                    if key not in seen:
                        seen.add(key)
                        candidates.append(candidate)
                        moves.append([donor, int(source), int(lower),
                                      receiver, int(target), int(upper)])
    return np.stack(candidates), moves


@torch.no_grad()
def centroid_usage(codec, feature_path, count, device, norm_mode, image_batch):
    source = np.load(feature_path, mmap_mode="r")
    counts = [torch.zeros(q.codebooks.shape[:2], device=device)
              for q in codec.pq.quantizers]
    rotation = codec.transform.get_rotation()
    for first in range(0, count, image_batch):
        x = torch.from_numpy(np.array(
            source[first:first + image_batch], copy=True)).float().to(device)
        y, _, _ = batch_normalize_gpu(x, mode=norm_mode)
        z = y.reshape(-1, y.shape[-1]) @ rotation
        sub = z.reshape(-1, codec.pq.G, codec.pq.d).permute(1, 0, 2)
        for table, quantizer in zip(counts, codec.pq.quantizers):
            labels = torch.cdist(sub, quantizer.codebooks).argmin(-1)
            table.scatter_add_(
                1, labels, torch.ones_like(labels, dtype=table.dtype))
    result = []
    for table in counts:
        probability = table / table.sum(1, keepdim=True).clamp_min(1)
        entropy = -(probability * probability.clamp_min(1e-30).log()).sum(1)
        effective = entropy.exp()
        positive = table[table > 0]
        result.append({
            "K": int(table.shape[1]),
            "dead_count": int((table == 0).sum()),
            "dead_fraction": float((table == 0).float().mean()),
            "groups_with_dead": int((table == 0).any(1).sum()),
            "count_le_10_fraction": float((table <= 10).float().mean()),
            "min_positive_count": int(positive.min()) if positive.numel() else 0,
            "effective_k_ratio_mean": float(
                (effective / table.shape[1]).mean()),
            "effective_k_ratio_min": float(
                (effective / table.shape[1]).min())})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", required=True, choices=tuple(SPECS))
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    C = activate(args.block)
    anchor = C.ANCHOR_BY_NAME[args.anchor]
    root = C.output_dir(anchor, args.run_id)
    device = torch.device(args.device)
    codec = load_codec_v1(root / "codec.pt", device=device)
    allocation = np.load(root / "allocation.npy")
    candidates, moves = full_neighbors(allocation, anchor.mode_bits)
    val_paths = C.load_split("train_val")
    resident = engine.ResidentSet(*val_paths[:3], device)
    tail = tail_mod.build_tail(C.LAYER, device)
    matrix = engine.evaluate_allocations(
        codec, tail, resident, np.concatenate((allocation[None], candidates)),
        image_batch=C.EVAL_IMAGE_BATCH, pair_budget=C.EVAL_PAIR_BUDGET)
    means = matrix.mean(1)
    winner = int(np.argmin(means[1:]))
    neighborhood = {
        "definition": "all exact-rate two-group transfers of one or more bits",
        "candidate_count": len(candidates),
        "base_distortion": float(means[0]),
        "best_distortion": float(means[winner + 1]),
        "best_gain": float(means[0] - means[winner + 1]),
        "best_move": moves[winner],
        "locally_optimal": bool(means[winner + 1] >= means[0])}
    del tail, resident, matrix
    torch.cuda.empty_cache()
    usage = centroid_usage(
        codec, C.TRAIN_FEATURES, C.N_TRAIN, device, C.NORM_MODE,
        4 if max(anchor.mode_sizes) >= 1024 else 16)
    result = {"block": args.block, "anchor": anchor.name,
              "rate": int(sum(anchor.mode_bits[m] for m in allocation)),
              "allocation": allocation.tolist(),
              "full_two_group_neighborhood": neighborhood,
              "centroid_usage_train5k": usage}
    (root / "final_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
