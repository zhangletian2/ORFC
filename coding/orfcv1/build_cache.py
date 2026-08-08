#!/usr/bin/env python3
"""Build the stacked feature / teacher caches consumed by orfcv1.

The original builder was an ad-hoc step that was never committed. This
reconstructs it, and -- crucially -- can be validated bit-for-bit against the
frozen blk20 cache (subcommand `verify`) before it is trusted to build
anything new.

Layout produced (matches what run_allocation_short.sh / run_separability_h1.sh
already consume):
    artifacts/dinov2_vitl14/cache/{features,teacher}_{split}_{blk}_n{N}_ss{TAG}.npy
with shape (N, 257, 1024) float32.

teacher = tail(raw features), where tail = blocks[layer+1:] + backbone.norm.
Verified against allocation_train._distortions: the target is the tail applied
to the UN-normalised feature tensor.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ORFC = HERE.parent / "orfc"
sys.path[:0] = [str(HERE), str(ORFC)]

FEAT_ROOT = Path('/data4/workspace/zlt/featcodec/features')
CACHE = HERE / 'artifacts/dinov2_vitl14/cache'
# ORFC/features is a symlink to the above, so a single root covers every pool.
# 2026-07-30: train(5000) / val(3000, the measure set) / test(500) were all
# re-extracted in the same conda featcodec2 environment between 06:05 and 06:33,
# so provenance is uniform across fitting and measurement. Before that the
# stored features came from a Jan-Feb 2026 environment that the current one
# reproduces only to rel_fro 2e-2 (vs a 3e-7 run-to-run floor).
POOLS = ('train', 'val', 'test')


def basenames(pool, block):
    d = FEAT_ROOT / pool / 'dinov2_vitl14' / block
    return sorted(p.name for p in d.iterdir() if p.suffix == '.npy')


def build_tail(layer, device):
    from backbone.wrapper import Dinov2Wrapper
    from soft_pq import FrozenTail
    wrapper = Dinov2Wrapper(
        head_layers=1, model_name="dinov2_vitl14", device=device)
    blocks = list(wrapper.backbone.blocks)
    for block in blocks[:layer + 1]:
        block.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()
    return FrozenTail(blocks[layer + 1:], wrapper.backbone.norm, device=device)


def cmd_features(args):
    names = basenames(args.pool, args.block)
    if args.index:
        idx = np.load(args.index)
    else:
        # whole pool, in the same ascending-basename order the index convention
        # uses -- for the measure pool there is no sub-selection to make.
        idx = np.arange(len(names), dtype=np.int64)
    assert idx.ndim == 1 and idx.min() >= 0 and idx.max() < len(names), \
        (args.index, len(names))
    assert len(set(idx.tolist())) == len(idx), 'index has duplicates'
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    probe = np.load(FEAT_ROOT / args.pool / 'dinov2_vitl14' / args.block /
                    names[idx[0]], mmap_mode='r')
    arr = np.lib.format.open_memmap(
        out, mode='w+', dtype=np.float32, shape=(len(idx),) + probe.shape)
    src = FEAT_ROOT / args.pool / 'dinov2_vitl14' / args.block
    for r, i in enumerate(idx):
        a = np.load(src / names[i])
        assert a.shape == probe.shape, (names[i], a.shape)
        arr[r] = a.astype(np.float32, copy=False)
        if (r + 1) % 500 == 0:
            print(f'  {r + 1}/{len(idx)}', flush=True)
    arr.flush()
    print(f'wrote {out} {arr.shape}')


@torch.no_grad()
def _forward(tail, features, out, batch, device, ref=None):
    n = len(features)
    worst_abs = 0.0
    worst_rel = 0.0
    for s in range(0, n, batch):
        x = torch.from_numpy(
            np.asarray(features[s:s + batch])).float().to(device)
        y = tail.forward_nograd(x).float().cpu().numpy()
        if out is not None:
            out[s:s + batch] = y
        if ref is not None:
            r = np.asarray(ref[s:s + batch], dtype=np.float32)
            d = np.abs(y - r)
            worst_abs = max(worst_abs, float(d.max()))
            den = np.abs(r).max()
            worst_rel = max(worst_rel, float(d.max() / den) if den else 0.0)
        if (s + batch) % (batch * 20) == 0:
            print(f'  {min(s + batch, n)}/{n}', flush=True)
    return worst_abs, worst_rel


def cmd_teacher(args):
    device = torch.device('cuda')
    tail = build_tail(args.layer, device)
    features = np.load(args.features, mmap_mode='r')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(
        out, mode='w+', dtype=np.float32, shape=features.shape)
    _forward(tail, features, arr, args.batch, device)
    arr.flush()
    print(f'wrote {out} {arr.shape}')


def cmd_verify(args):
    """Rebuild a slice of the FROZEN blk20 cache and diff bit-for-bit."""
    device = torch.device('cuda')
    tail = build_tail(20, device)
    feats = np.load(args.features, mmap_mode='r')[:args.images]
    ref = np.load(args.teachers, mmap_mode='r')[:args.images]
    a, r = _forward(tail, feats, None, args.batch, device, ref=ref)
    exact = a == 0.0
    print(json.dumps(dict(images=int(args.images), max_abs_diff=a,
                          max_rel_diff=r, bit_exact=bool(exact)), indent=1))
    print('VERDICT:', 'BIT-EXACT' if exact
          else ('CLOSE (float non-determinism)' if r < 1e-5 else '*** MISMATCH ***'))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)

    f = sub.add_parser('features'); f.set_defaults(fn=cmd_features)
    f.add_argument('--pool', required=True, choices=POOLS)
    f.add_argument('--block', required=True)
    f.add_argument('--index', help='optional .npy of row indices into the '
                                   'sorted basename list; default = whole pool')
    f.add_argument('--out', required=True)

    t = sub.add_parser('teacher'); t.set_defaults(fn=cmd_teacher)
    t.add_argument('--layer', type=int, required=True)
    t.add_argument('--features', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--batch', type=int, default=16)

    v = sub.add_parser('verify'); v.set_defaults(fn=cmd_verify)
    v.add_argument('--features', required=True)
    v.add_argument('--teachers', required=True)
    v.add_argument('--images', type=int, default=32)
    v.add_argument('--batch', type=int, default=16)

    args = p.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
