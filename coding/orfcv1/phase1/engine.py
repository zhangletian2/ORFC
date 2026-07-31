"""Batched hard-quantisation allocation evaluation -- v9's only measurement.

One forward of the frozen ViT tail per (allocation, image) pair is the
irreducible cost, so this module keeps the GPU busy:

* the evaluation split is resident in VRAM (500 x 257 x 1024 fp32 ~ 0.53 GB per
  array), so no allocation ever waits on a host copy;
* all modes are quantised once per image batch into a "mode bank" and every
  allocation is then a gather out of that bank -- the PQ cost is paid once for
  993 allocations, not 993 times;
* each allocation gets its own tail call, all of them the same shape.  Folding
  allocations together was measured to be slower and 7x more VRAM-hungry (the
  numbers are in config.py), and equal shapes buy something the fold cannot:
  every candidate is computed by the identical kernel path, so the cal argmin
  cannot be decided by a batch offset.

The v8 version of this module also carried a soft-relaxation path used by the
inner training loop.  v9 trains nothing and every number it reports is a hard
nearest neighbour, so the soft path is gone rather than merely unused -- there
is then no way for a soft number to reach a result by accident (plan v9
section 3.4).
"""

import numpy as np
import torch

from opq import batch_inv_normalize_gpu, batch_normalize_gpu

from . import config as C


def configure_precision(allow_tf32=C.ALLOW_TF32):
    torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(allow_tf32)


class ResidentSet:
    """A whole evaluation split held in VRAM, pre-normalised."""

    def __init__(self, features_path, teachers_path, rows, device,
                 norm_mode=C.NORM_MODE, max_images=None):
        rows = np.asarray(rows)
        if max_images is not None:
            rows = rows[:max_images]
        x = np.asarray(np.load(features_path, mmap_mode="r")[rows])
        t = np.asarray(np.load(teachers_path, mmap_mode="r")[rows])
        self.device = device
        self.rows = rows
        x = torch.from_numpy(x).float().to(device)
        self.teacher = torch.from_numpy(t).float().to(device)
        self.y, self.mu, self.std = batch_normalize_gpu(x, mode=norm_mode)
        del x
        torch.cuda.empty_cache()
        self.count, self.tokens, self.dim = self.y.shape

    def slice(self, start, stop):
        return (self.y[start:stop], self.mu[start:stop],
                self.std[start:stop], self.teacher[start:stop])


@torch.no_grad()
def build_bank(codec, y):
    """Hard-quantise one image batch under every mode.

    Returns ``([M, G, B*T, d], rotation)`` in the rotated frame.
    """
    pq = codec.pq
    rotation = codec.transform.get_rotation()
    b, tokens, _ = y.shape
    z = y.reshape(b * tokens, -1) @ rotation
    sub = z.reshape(-1, pq.G, pq.d).permute(1, 0, 2)          # [G, N, d]
    banks = []
    for quantizer in pq.quantizers:
        book = quantizer.codebooks                            # [G, K, d]
        labels = torch.cdist(sub, book).square().argmin(dim=-1)
        banks.append(torch.gather(
            book, 1, labels.unsqueeze(-1).expand(-1, -1, pq.d)))
    return torch.stack(banks), rotation


def _tail_distortion(tail, bank, rotation, modes, mu, std, teacher, b, tokens):
    """Distortion of ``len(modes)`` allocations on one image batch.

    Returns ``[A, B]`` -- per allocation, per image, so every downstream paired
    statistic has the image axis it needs.
    """
    groups = torch.arange(bank.shape[1], device=bank.device)
    count = modes.shape[0]
    selected = bank[modes, groups]                            # [A, G, N, d]
    selected = selected.permute(0, 2, 1, 3).reshape(count * b, tokens, -1)
    xhat = batch_inv_normalize_gpu(
        selected @ rotation.t(),
        mu.unsqueeze(0).expand(count, *mu.shape).reshape(count * b, *mu.shape[1:]),
        std.unsqueeze(0).expand(count, *std.shape).reshape(count * b, *std.shape[1:]))
    output = tail(xhat)
    target = teacher.unsqueeze(0).expand(
        count, -1, *teacher.shape[1:]).reshape(count * b, *teacher.shape[1:])
    return (output - target).square().reshape(count, b, -1).sum(-1)


def assert_uniform_shape(count, image_batch=C.EVAL_IMAGE_BATCH):
    """Every tail call must have the same shape, including the last one.

    A short final image chunk would give the allocations evaluated in it a
    different kernel path from all the others, which is precisely the asymmetry
    the one-allocation-per-call layout exists to remove.
    """
    if count % image_batch:
        raise SystemExit(f"INVALID_EXPERIMENT: {count} images is not a "
                         f"multiple of the image batch {image_batch}; the "
                         f"final tail call would have a different shape")
    return count // image_batch


@torch.no_grad()
def evaluate_allocations(codec, tail, resident, allocations,
                         image_batch=C.EVAL_IMAGE_BATCH,
                         pair_budget=C.EVAL_PAIR_BUDGET, per_image=True,
                         progress=None):
    """Hard tail distortion ``[A, N]`` for many allocations.

    ``pair_budget`` caps ``len(alloc_chunk) * image_batch`` -- the number of
    images inside one tail call.  The frozen configuration sets it equal to
    ``image_batch``, i.e. one allocation per call; see config.py for why that is
    both the faster and the fairer choice.
    """
    allocations = np.asarray(allocations, dtype=np.int64)
    device = resident.device
    modes_all = torch.from_numpy(allocations).to(device)
    alloc_chunk = max(1, pair_budget // image_batch)
    out = torch.zeros(len(allocations), resident.count, device=device)
    done = 0
    for start in range(0, resident.count, image_batch):
        stop = min(start + image_batch, resident.count)
        y, mu, std, teacher = resident.slice(start, stop)
        bank, rotation = build_bank(codec, y)
        b, tokens = y.shape[0], y.shape[1]
        for first in range(0, len(allocations), alloc_chunk):
            last = min(first + alloc_chunk, len(allocations))
            out[first:last, start:stop] = _tail_distortion(
                tail, bank, rotation, modes_all[first:last],
                mu, std, teacher, b, tokens)
        del bank
        done += stop - start
        if progress is not None:
            progress(done, resident.count)
    matrix = out.cpu().numpy()
    return matrix if per_image else matrix.mean(1)


# ------------------------------------------------------- allocation algebra ---
def uniform_allocation(anchor, groups=C.GROUPS):
    return np.full(groups, anchor.uniform_mode, dtype=np.int64)


def swap_candidates(anchor, groups=C.GROUPS):
    """All 32x31 ordered one-bit transfers around the uniform point.

    Returns ``(allocations [992, G], pairs [992, 2])`` where ``pairs[k]`` is
    ``(down_group, up_group)``, enumerated in lexicographic order -- which is
    also the frozen tie-break order (``config.TIE_BREAK``).
    """
    base = uniform_allocation(anchor, groups)
    allocations, pairs = [], []
    for down in range(groups):
        for up in range(groups):
            if down == up:
                continue
            candidate = base.copy()
            candidate[down] = anchor.down_mode
            candidate[up] = anchor.up_mode
            allocations.append(candidate)
            pairs.append((down, up))
    return np.stack(allocations), np.asarray(pairs, dtype=np.int64)


def nominal_rate(allocation, anchor):
    """Exact integer nominal rate of one allocation, in bits per token."""
    bits = anchor.mode_bits
    return int(sum(bits[m] for m in np.asarray(allocation).ravel()))
