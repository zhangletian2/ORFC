"""Four-way data split and the measure seal.

    train-core 4,000   builds the frozen object (OPQ warm-up + codebooks)
    cal-500      500   the 993-point exhaustive sweep; selects the candidate
    dev-500      500   unused by plan v9 (nothing is trained or early-stopped)
    measure-3000 3000  the single final unsealing

Written for plan v8 and reused by v9 without a byte changed, deliberately:
``SPLIT_SEED`` and the four sizes must not move or the sets would be redrawn,
and a redrawn cal would no longer be the set that never touched the build.

The train pool of 5,000 images was already split 4,500 / 500 by ``make_split.py``
(seed 20260730) into the rows of ``features_train_*_n4500`` and
``features_val_*_n500``.  Phase 1 subdivides the 4,500 into train-core and cal
with its own frozen seed, and takes the 3,000-row ``features_test_*_n3000``
cache (built from the ``val`` image pool) as measure.

``load_split("measure")`` raises unless the run directory carries an explicit
unseal token, which is what makes I19 mechanical rather than a matter of care.
"""

import json
import numpy as np
from pathlib import Path

from . import config as C

SPLIT_DIR = C.SPLIT / "phase1"
UNSEAL_TOKEN = C.PHASE1 / "MEASURE_UNSEALED.json"

_CACHES = {
    "train_core": ("features_train_{blk}_n4500_{tag}.npy",
                   "teacher_train_{blk}_n4500_{tag}.npy"),
    "cal":        ("features_train_{blk}_n4500_{tag}.npy",
                   "teacher_train_{blk}_n4500_{tag}.npy"),
    "dev":        ("features_val_{blk}_n500_{tag}.npy",
                   "teacher_val_{blk}_n500_{tag}.npy"),
    "measure":    ("features_test_{blk}_n3000_{tag}.npy",
                   "teacher_test_{blk}_n3000_{tag}.npy"),
}


def cache_paths(name, block=C.BLOCK, tag=C.CACHE_TAG):
    feat, teach = _CACHES[name]
    return (C.CACHE / feat.format(blk=block, tag=tag),
            C.CACHE / teach.format(blk=block, tag=tag))


def _pool_basenames(pool, block=C.BLOCK):
    directory = C.FEATURE_POOL / pool / "dinov2_vitl14" / block
    return sorted(p.stem for p in directory.iterdir() if p.suffix == ".npy")


def freeze_splits(force=False):
    """Write the four frozen index/basename lists.  Idempotent."""
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = SPLIT_DIR / "manifest.json"
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text())

    train_pool = _pool_basenames("train")
    measure_pool = _pool_basenames("val")
    assert len(train_pool) == 5000, len(train_pool)
    assert len(measure_pool) == C.N_MEASURE, len(measure_pool)

    train_rows = np.load(C.SPLIT / "train_idx_n4500.npy")
    dev_rows = np.load(C.SPLIT / "dev_idx_n500.npy")
    train_names = [train_pool[i] for i in train_rows]
    dev_names = [train_pool[i] for i in dev_rows]
    assert train_names == (C.SPLIT / "train_names_n4500.txt").read_text().split()
    assert dev_names == (C.SPLIT / "dev_names_n500.txt").read_text().split()

    rng = np.random.default_rng(C.SPLIT_SEED)
    perm = rng.permutation(len(train_names))
    cal_local = np.sort(perm[:C.N_CAL])
    core_local = np.sort(perm[C.N_CAL:])
    assert len(core_local) == C.N_TRAIN_CORE, len(core_local)

    tables = {
        "train_core": (core_local, [train_names[i] for i in core_local]),
        "cal": (cal_local, [train_names[i] for i in cal_local]),
        "dev": (np.arange(C.N_DEV), dev_names),
        "measure": (np.arange(C.N_MEASURE), measure_pool),
    }

    # I6: zero basename overlap across all four sets.
    for a in tables:
        for b in tables:
            if a < b:
                shared = set(tables[a][1]) & set(tables[b][1])
                assert not shared, f"{a}/{b} share {len(shared)} basenames"

    manifest = {"split_seed": C.SPLIT_SEED, "block": C.BLOCK,
                "cache_tag": C.CACHE_TAG, "sets": {}}
    for name, (rows, names) in tables.items():
        np.save(SPLIT_DIR / f"{name}_rows.npy", rows.astype(np.int64))
        (SPLIT_DIR / f"{name}_names.txt").write_text("\n".join(names) + "\n")
        feat, teach = cache_paths(name)
        manifest["sets"][name] = {
            "count": int(len(rows)), "features": str(feat),
            "teachers": str(teach), "first": names[0], "last": names[-1]}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def is_unsealed():
    return UNSEAL_TOKEN.exists()


def load_split(name, allow_measure=False):
    """Return ``(feature_path, teacher_path, rows, basenames)``.

    Requesting ``measure`` without both the on-disk unseal token and an
    explicit ``allow_measure=True`` is an error -- I19.
    """
    if name == "measure":
        if not allow_measure:
            raise PermissionError(
                "measure-3000 is sealed: the caller did not request it "
                "explicitly (plan section 4.0.1, invariant I19)")
        if not is_unsealed():
            raise PermissionError(
                f"measure-3000 is sealed: {UNSEAL_TOKEN} does not exist")
    rows = np.load(SPLIT_DIR / f"{name}_rows.npy")
    names = (SPLIT_DIR / f"{name}_names.txt").read_text().split()
    feat, teach = cache_paths(name)
    return feat, teach, rows, names


def assert_training_split(name):
    """Guard for every training / search entry point (I19)."""
    if name not in ("train_core", "cal", "dev"):
        raise PermissionError(
            f"training and search may only read train_core/cal/dev, got {name}")
    return name


if __name__ == "__main__":
    manifest = freeze_splits(force=False)
    print(json.dumps(manifest, indent=2))
