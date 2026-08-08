"""Fixed train_val list and resident helpers for V33 probes/search.

Aligns with v12 ``load_split("train_val")`` / ``N_VAL = 500``: the validation
list is the leading ``N_VAL`` rows of the frozen val feature/teacher caches.
"""

from __future__ import annotations

import numpy as np

from .. import engine
from ..v12 import config as C

N_VAL = C.N_VAL  # 500


def fixed_val_rows(n_val=N_VAL):
    """Canonical fixed validation index list ``np.arange(n_val)``."""
    n_val = int(n_val)
    if n_val < 1:
        raise ValueError("n_val must be positive")
    if n_val > C.N_VAL:
        raise ValueError(
            f"requested n_val={n_val} exceeds v12 N_VAL={C.N_VAL}")
    return np.arange(n_val, dtype=np.int64)


def val_paths():
    """``(features, teachers, rows, extras)`` for the frozen train_val split."""
    return C.load_split("train_val")


def load_val_resident(device, n_images=N_VAL, offset=0, image_batch=None):
    """Resident GPU cache over a contiguous slice of the fixed val list.

    ``offset`` / ``n_images`` select a subset of the fixed 500-row list
    (e.g. search body uses 128, top-k recheck uses 500).
    """
    feature, teacher, rows, _ = val_paths()
    rows = np.asarray(rows, dtype=np.int64)
    selected = rows[int(offset):int(offset) + int(n_images)]
    if len(selected) != int(n_images):
        raise ValueError(
            f"train_val slice [{offset}:{offset + n_images}] is short "
            f"(have {len(rows)} rows)")
    # ``image_batch`` is accepted for call-site symmetry only.  Every consumer
    # walks the resident with ``range(0, count, image_batch)`` and
    # ``ResidentSet.slice``, whose plain Python slicing clamps the final short
    # block, so a ragged tail is exact rather than an error.
    del image_batch
    return engine.ResidentSet(feature, teacher, selected, device)


def cache_length():
    """Total rows in the val feature cache (``load_split`` exposes N_VAL)."""
    feature, _, _, _ = val_paths()
    return int(np.load(feature, mmap_mode="r").shape[0])


def holdout_rows(start=N_VAL, stop=None):
    """Val-cache rows past the fixed list, untouched by search or validation.

    ``load_split("train_val")`` only ever hands out ``np.arange(N_VAL)``, so
    everything from ``N_VAL`` on is a genuine hold-out for anything selected
    on the 500-image list (allocations, switch points, early stopping).
    """
    total = cache_length()
    start = int(start)
    stop = total if stop is None else min(int(stop), total)
    if start < N_VAL:
        raise ValueError(f"holdout must start at or after N_VAL={N_VAL}")
    if stop <= start:
        raise ValueError(
            f"empty holdout range [{start}, {stop}) over {total} cached rows")
    return np.arange(start, stop, dtype=np.int64)


def disjoint_partitions(n_total, n_parts, seed=0):
    """Partition ``[0, n_total)`` into ``n_parts`` nearly-equal disjoint blocks.

    Returns a list of index arrays (row indices into the fixed val list).
    Remainders are distributed to the first blocks so sizes differ by at most 1.
    """
    n_total, n_parts = int(n_total), int(n_parts)
    if n_parts < 2 or n_total < n_parts:
        raise ValueError("need n_parts >= 2 and n_total >= n_parts")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n_total)
    sizes = [n_total // n_parts + (1 if i < n_total % n_parts else 0)
             for i in range(n_parts)]
    parts, cursor = [], 0
    for size in sizes:
        parts.append(np.sort(order[cursor:cursor + size]))
        cursor += size
    return parts


def resident_from_rows(device, row_indices):
    """Build a :class:`~phase1.engine.ResidentSet` for arbitrary val-list rows."""
    feature, teacher, _, _ = val_paths()
    rows = np.asarray(row_indices, dtype=np.int64)
    return engine.ResidentSet(feature, teacher, rows, device)
