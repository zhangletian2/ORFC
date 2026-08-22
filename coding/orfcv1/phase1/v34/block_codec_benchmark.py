"""Frozen ORFC/Block-VQ/Block-PQ rate, complexity and utilisation audit."""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from compressai import ans

ROOT = Path("/data4/workspace/zlt/featcodec")
ORFC = ROOT / "ORFC/coding/orfc"
ORFCV1 = ROOT / "ORFC/coding/orfcv1"
sys.path[:0] = [str(ORFC), str(ORFCV1)]

from opq import batch_inv_normalize_gpu, batch_normalize_gpu  # noqa: E402
from phase1.v21 import config as v21_config  # noqa: E402
from phase1.v34.block_quantizer import blockify, unblockify  # noqa: E402
from phase1.v34.block_vq_eval import load_codec  # noqa: E402
from phase1.v34.orfc_same_group_rans import cdf_from_counts  # noqa: E402

FULL_TOKENS, FEATURE_DIM = 257, 1024


def codec_spec(codec):
    if codec.arm == "orfc":
        grid_h = grid_w = 16
        block_shape = (1, 1)
    else:
        block_shape = codec.pq.block_shape
        grid_h = math.ceil(16 / block_shape[0])
        grid_w = math.ceil(16 / block_shape[1])
    return {
        "arm": codec.arm, "grid_h": grid_h, "grid_w": grid_w,
        "block_shape": list(block_shape), "positions": grid_h * grid_w,
        "groups": int(codec.pq.G), "K": int(codec.pq.K), "d": int(codec.pq.d),
        "symbols_per_image": grid_h * grid_w * int(codec.pq.G),
        "codebook_parameters": int(codec.pq.codebooks.numel()),
    }


def array_batches(source, count, batch):
    count = min(int(count), len(source))
    for first in range(0, count, int(batch)):
        yield np.array(source[first:first + int(batch)], copy=True)


def path_batches(paths, batch):
    for first in range(0, len(paths), int(batch)):
        rows = []
        for path in paths[first:first + int(batch)]:
            row = np.load(path)
            if row.ndim == 3 and row.shape[0] == 1:
                row = row[0]
            if row.ndim != 2:
                raise ValueError(f"unexpected feature shape {row.shape} in {path}")
            rows.append(row)
        yield np.stack(rows)


@torch.no_grad()
def hard_labels(codec, x_np, device, spec):
    x = torch.from_numpy(x_np).float().to(device)
    y, _, _ = batch_normalize_gpu(x, mode="per_image")
    codec(y)
    labels = codec.pq._last_labels.t().reshape(
        len(x), spec["positions"], spec["groups"])
    return labels.cpu().numpy().astype(np.int16, copy=False)


def context_ids(labels, spec, context_count):
    n, positions, groups = labels.shape
    height, width, k = spec["grid_h"], spec["grid_w"], spec["K"]
    grid = labels.reshape(n, height, width, groups).astype(np.int64)
    left = np.full_like(grid, k)
    up = np.full_like(grid, k)
    left[:, :, 1:] = grid[:, :, :-1]
    up[:, 1:] = grid[:, :-1]
    return ((left * (k + 1) + up) % context_count).reshape(n, positions, groups)


def empty_counts(spec, buckets):
    contexts = min((spec["K"] + 1) ** 2, int(buckets))
    return {
        "factor": np.zeros((spec["groups"], spec["K"]), dtype=np.uint64),
        "context": np.zeros((spec["groups"], contexts, spec["K"]), dtype=np.uint32),
        "contexts": contexts,
    }


def add_counts(counts, labels, spec):
    flat = labels.reshape(-1, spec["groups"]).astype(np.int64)
    ctx = context_ids(labels, spec, counts["contexts"]).reshape(-1, spec["groups"])
    for group in range(spec["groups"]):
        counts["factor"][group] += np.bincount(
            flat[:, group], minlength=spec["K"]).astype(np.uint64)
        np.add.at(counts["context"][group],
                  (ctx[:, group], flat[:, group]), 1)


def collect(codec, batches, device, spec):
    records = []
    for x in batches:
        labels = hard_labels(codec, x, device, spec)
        records.extend({"labels": row,
                        "grid_shape": (spec["grid_h"], spec["grid_w"]),
                        "full_tokens": FULL_TOKENS} for row in labels)
    return records


@torch.no_grad()
def variable_records(codec, paths, device, spec):
    records = []
    for path in paths:
        x = np.load(path)
        if x.ndim == 2:
            x = x[None]
        if x.ndim != 3 or x.shape[-1] != FEATURE_DIM:
            raise ValueError(f"unexpected feature shape {x.shape} in {path}")
        patch_side = int(round(math.sqrt(x.shape[1] - 1)))
        if patch_side ** 2 != x.shape[1] - 1:
            raise ValueError(f"non-square patch grid {x.shape} in {path}")
        block_h, block_w = spec["block_shape"]
        symbol_h = math.ceil(patch_side / block_h)
        symbol_w = math.ceil(patch_side / block_w)
        local = dict(spec, grid_h=symbol_h, grid_w=symbol_w,
                     positions=symbol_h * symbol_w)
        labels = hard_labels(codec, np.array(x, copy=True), device, local)
        records.extend({"labels": row, "grid_shape": (symbol_h, symbol_w),
                        "full_tokens": int(x.shape[1]), "source": path.name}
                       for row in labels)
    return records


def add_record_counts(counts, records, spec):
    for record in records:
        height, width = record["grid_shape"]
        local = dict(spec, grid_h=height, grid_w=width,
                     positions=height * width)
        add_counts(counts, record["labels"][None], local)


def utilisation(factor):
    total = factor.sum(1, keepdims=True)
    prob = factor / np.maximum(total, 1)
    entropy = -(prob * np.log(np.maximum(prob, 1e-30))).sum(1)
    perplexity = np.exp(entropy)
    used = (factor > 0).sum(1)
    max_mass = prob.max(1)
    return {
        "centroids": int(factor.size), "zero_use": int((factor == 0).sum()),
        "zero_use_fraction": float((factor == 0).mean()),
        "used_fraction_mean": float((used / factor.shape[1]).mean()),
        "perplexity_mean": float(perplexity.mean()),
        "perplexity_min": float(perplexity.min()),
        "perplexity_median": float(np.median(perplexity)),
        "perplexity_max": float(perplexity.max()),
        "max_symbol_mass_mean": float(max_mass.mean()),
        "max_symbol_mass_max": float(max_mass.max()),
    }


class CDFProvider:
    def __init__(self, counts, alpha):
        self.factor_counts = counts["factor"]
        self.context_counts = counts["context"]
        self.alpha = float(alpha)
        self.factor = [cdf_from_counts(x, self.alpha) for x in self.factor_counts]

    @functools.lru_cache(maxsize=32768)
    def spatial(self, group, context):
        row = self.context_counts[int(group), int(context)]
        if not row.any():
            row = self.factor_counts[int(group)]
        return cdf_from_counts(row, self.alpha)


def factor_stream(labels, provider, spec):
    flat = labels.reshape(-1, spec["groups"]).astype(np.int32)
    symbols = flat.reshape(-1).tolist()
    indexes = np.tile(np.arange(spec["groups"], dtype=np.int32), len(flat)).tolist()
    sizes = [spec["K"] + 2] * spec["groups"]
    offsets = [0] * spec["groups"]
    encoder = ans.RansEncoder()
    t0 = time.perf_counter()
    stream = encoder.encode_with_indexes(symbols, indexes, provider.factor, sizes, offsets)
    enc_s = time.perf_counter() - t0
    decoder = ans.RansDecoder()
    t0 = time.perf_counter()
    decoded = decoder.decode_with_indexes(stream, indexes, provider.factor, sizes, offsets)
    dec_s = time.perf_counter() - t0
    if decoded != symbols:
        raise RuntimeError("factor rANS roundtrip mismatch")
    return stream, enc_s, dec_s, 0.0


def context_stream(labels, provider, spec, contexts):
    flat = labels.reshape(-1, spec["groups"]).astype(np.int32)
    ctx = context_ids(labels[None], spec, contexts)[0].astype(np.int64)
    gids = np.arange(spec["groups"], dtype=np.int64)[None]
    ids = (gids * contexts + ctx).reshape(-1)
    t0 = time.perf_counter()
    unique, indexes = np.unique(ids, return_inverse=True)
    cdfs = [provider.spatial(int(x // contexts), int(x % contexts)) for x in unique]
    prep_s = time.perf_counter() - t0
    sizes = [spec["K"] + 2] * len(cdfs)
    offsets = [0] * len(cdfs)
    symbols = flat.reshape(-1).tolist()
    encoder = ans.RansEncoder()
    t0 = time.perf_counter()
    stream = encoder.encode_with_indexes(
        symbols, indexes.astype(np.int32).tolist(), cdfs, sizes, offsets)
    enc_s = time.perf_counter() - t0

    # Causal decoder: each left/up context is reconstructed only from symbols
    # already decoded at earlier raster positions.
    decoder = ans.RansDecoder()
    decoder.set_stream(stream)
    decoded = np.empty_like(flat)
    t0 = time.perf_counter()
    for pos in range(spec["positions"]):
        row, col = divmod(pos, spec["grid_w"])
        left = decoded[pos - 1] if col else np.full(spec["groups"], spec["K"])
        up = (decoded[pos - spec["grid_w"]] if row
              else np.full(spec["groups"], spec["K"]))
        current = ((left * (spec["K"] + 1) + up) % contexts).astype(np.int64)
        step_cdfs = [provider.spatial(g, int(current[g]))
                     for g in range(spec["groups"])]
        decoded[pos] = decoder.decode_stream(
            list(range(spec["groups"])), step_cdfs,
            [spec["K"] + 2] * spec["groups"], [0] * spec["groups"])
    dec_s = time.perf_counter() - t0
    if not np.array_equal(decoded, flat):
        raise RuntimeError("causal context rANS roundtrip mismatch")
    return stream, enc_s, dec_s, prep_s


def rate_dataset(records, provider, spec, counts):
    result = {}
    for name, coder in (("factor", factor_stream), ("context", context_stream)):
        bits = enc_s = dec_s = prep_s = 0.0
        total_patch_tokens = 0
        for record in records:
            image = record["labels"]
            height, width = record["grid_shape"]
            local = dict(spec, grid_h=height, grid_w=width,
                         positions=height * width)
            if name == "factor":
                stream, e, d, p = coder(image, provider, local)
            else:
                stream, e, d, p = coder(image, provider, local, counts["contexts"])
            bits += 8 * len(stream); enc_s += e; dec_s += d; prep_s += p
            total_patch_tokens += record["full_tokens"] - 1
        denom = total_patch_tokens * FEATURE_DIM
        patch_bpfp = bits / denom
        result[name] = {
            "bits": int(bits), "patch_bpfp": patch_bpfp,
            "encode_seconds": enc_s, "decode_seconds": dec_s,
            "context_prepare_seconds": prep_s,
            "streams": len(records), "patch_tokens": total_patch_tokens,
            "encode_ms_per_stream": 1000 * enc_s / len(records),
            "decode_ms_per_stream": 1000 * dec_s / len(records),
        }
    result["saving_bpfp"] = (result["factor"]["patch_bpfp"]
                              - result["context"]["patch_bpfp"])
    result["saving_fraction"] = (result["saving_bpfp"]
                                  / max(result["factor"]["patch_bpfp"], 1e-30))
    return result


@torch.no_grad()
def encode_gpu(codec, x):
    y, mu, std = batch_normalize_gpu(x, mode="per_image")
    rotation = codec.transform.get_rotation()
    patches = y[:, 1:] @ rotation
    if codec.arm == "orfc":
        vectors = patches.reshape(-1, codec.pq.G, codec.pq.d).permute(1, 0, 2)
    else:
        blocks = blockify(patches.reshape(-1, 16, 16, 32, 32),
                          block_shape=codec.pq.block_shape)
        vectors = codec.pq.split(blocks).permute(2, 0, 1, 3).reshape(
            codec.pq.G, -1, codec.pq.d)
    labels = torch.cdist(vectors, codec.pq.codebooks).argmin(-1)
    return labels.t().reshape(len(x), -1, codec.pq.G), y[:, :1], mu, std


@torch.no_grad()
def decode_gpu(codec, labels, cls, mu, std):
    b, positions, groups = labels.shape
    books = codec.pq.codebooks
    flat = labels.reshape(-1, groups).t().long()
    chosen = books[torch.arange(groups, device=books.device)[:, None], flat]
    chosen = chosen.permute(1, 0, 2).reshape(b, positions, groups, codec.pq.d)
    if codec.arm == "orfc":
        patch = chosen.reshape(b, 256, FEATURE_DIM)
    else:
        merged = codec.pq.merge(chosen)
        patch = unblockify(merged, 16, 16, codec.pq.block_shape)
    patch = patch @ codec.transform.get_rotation().t()
    return batch_inv_normalize_gpu(torch.cat((cls, patch), 1), mu, std)


def gpu_benchmark(codec, paths, device, batches, repeats):
    out = {}
    samples = np.stack([np.load(p) for p in paths[:max(batches)]])
    for batch in batches:
        x = torch.from_numpy(samples[:batch]).float().to(device)
        for _ in range(3):
            labels, cls, mu, std = encode_gpu(codec, x)
            decode_gpu(codec, labels, cls, mu, std)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        for _ in range(repeats):
            labels, cls, mu, std = encode_gpu(codec, x)
        torch.cuda.synchronize(device)
        enc = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(repeats):
            decode_gpu(codec, labels, cls, mu, std)
        torch.cuda.synchronize(device)
        dec = time.perf_counter() - t0
        out[str(batch)] = {
            "encode_ms_per_image": 1000 * enc / (repeats * batch),
            "decode_ms_per_image": 1000 * dec / (repeats * batch),
            "encode_images_per_second": repeats * batch / enc,
            "decode_images_per_second": repeats * batch / dec,
            "peak_gpu_mib": torch.cuda.max_memory_allocated(device) / 2 ** 20,
        }
    return out


def run(args):
    device = torch.device(args.device)
    codec, _ = load_codec(Path(args.checkpoint), device)
    spec = codec_spec(codec)
    cfg = v21_config.activate(args.block)
    train = np.load(cfg.TRAIN_FEATURES, mmap_mode="r")
    counts = empty_counts(spec, args.context_buckets)
    started = time.time()
    for x in array_batches(train, args.train_images, args.batch):
        add_counts(counts, hard_labels(codec, x, device, spec), spec)
    provider = CDFProvider(counts, args.alpha)

    cls_paths = sorted((ROOT / f"features/test/dinov2_vitl14/{args.block}").glob("*.npy"))
    voc_paths = sorted((ROOT / f"features/voc2012_100/dinov2_vitl14/{args.block}").glob("*.npy"))
    cls_records = collect(
        codec, path_batches(cls_paths[:args.cls_images], args.batch), device, spec)
    voc_records = variable_records(codec, voc_paths[:args.voc_images], device, spec)
    cls_counts = empty_counts(spec, args.context_buckets)
    voc_counts = empty_counts(spec, args.context_buckets)
    add_record_counts(cls_counts, cls_records, spec)
    add_record_counts(voc_counts, voc_records, spec)
    downstream = json.loads(Path(args.downstream).read_text())
    result = {
        "contract": "v34_frozen_block_codec_actual_rans_v1",
        "block": args.block, "checkpoint": str(args.checkpoint), "spec": spec,
        "prior_fit": {"split": "train", "images": min(args.train_images, len(train)),
                      "alpha": args.alpha, "context": "causal same-group left/up",
                      "context_count": counts["contexts"],
                      "context_hashed": counts["contexts"] < (spec["K"] + 1) ** 2},
        "rate_unit": "patch BPFP = rANS payload bits / (patch tokens * 1024)",
        "classification": {"images": len(cls_paths[:args.cls_images]),
                           "streams": len(cls_records),
                           "rate": rate_dataset(cls_records, provider, spec, counts),
                           "cls_acc": downstream["cls_acc"],
                           "utilisation": utilisation(cls_counts["factor"])},
        "segmentation": {"images": len(voc_paths[:args.voc_images]),
                         "streams": len(voc_records),
                         "rate": rate_dataset(voc_records, provider, spec, counts),
                         "seg_miou": downstream["seg_miou"],
                         "seg_acc": downstream["seg_acc"],
                         "utilisation": utilisation(voc_counts["factor"])},
        "train_utilisation": utilisation(counts["factor"]),
        "gpu_complexity": gpu_benchmark(
            codec, cls_paths, device,
            [int(x) for x in args.timing_batches.split(",")], args.timing_repeats),
        "model_storage": {
            "codebook_parameters": spec["codebook_parameters"],
            "codebook_bytes_fp32": spec["codebook_parameters"] * 4,
            "rotation_parameters": FEATURE_DIM ** 2,
        },
        "seconds": time.time() - started,
        "notes": [
            "rANS streams are independent per image and roundtrip checked",
            "context decode derives left/up contexts only from decoded symbols",
            "rate includes the complete rANS payload and excludes the bypassed CLS token",
            "entropy-model, codec parameters, and other side information are excluded",
        ],
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "block": args.block, "arm": spec["arm"],
        "cls_factor_bpfp": result["classification"]["rate"]["factor"]["patch_bpfp"],
        "cls_context_bpfp": result["classification"]["rate"]["context"]["patch_bpfp"],
        "voc_factor_bpfp": result["segmentation"]["rate"]["factor"]["patch_bpfp"],
        "voc_context_bpfp": result["segmentation"]["rate"]["context"]["patch_bpfp"],
    }, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--downstream", required=True)
    p.add_argument("--block", required=True, choices=("blk05", "blk10", "blk15", "blk20"))
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--train-images", type=int, default=5000)
    p.add_argument("--cls-images", type=int, default=500)
    p.add_argument("--voc-images", type=int, default=100)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--context-buckets", type=int, default=4096)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--timing-batches", default="1,32")
    p.add_argument("--timing-repeats", type=int, default=10)
    run(p.parse_args())


if __name__ == "__main__":
    main()
