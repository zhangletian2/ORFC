"""N9: build and register holdout-500, v10's only confirmatory set.

holdout-500 is the 500-image ``test`` feature pool.  It is not one of the four
sets in ``phase1/splits.py`` and was never read by v8 or v9.  measure-3000 was
unsealed by v9 and is therefore spent as a blind set; it stays on disk as the
historical v9 baseline and v10 does not touch it.

The construction path is deliberately *the same code* that produced the other
four caches -- ``build_cache.cmd_features`` over the whole pool in ascending
basename order, then ``build_cache.cmd_teacher`` with ``--layer 20 --batch 16``,
exactly as ``stack_features.sh`` and ``teacher_forward.sh`` ran them.  A second
implementation would be a second thing to keep in agreement, so there is not
one: this module only chooses the arguments and then checks the result.

Four invariants are computed here and re-computed independently by
``verify.py --stage holdout``:

    W1  zero basename overlap with train-core / cal / dev / measure
    W2  exactly one image per class over 500 distinct classes (devkit truth)
    W3  the blkNN tag of both caches agrees with LAYER
    W4  the teacher cache replays from the tail within REPLAY_REL_TOL

W2 is the one that decides a statistical choice rather than catching a bug: if
the pool is one image per class then the stratum unit *is* the image, so the
per-image paired bootstrap already is the stratified bootstrap and no clustered
resampling is needed.  That is checked, not assumed.
"""

import argparse
import json
import sys
from argparse import Namespace

import numpy as np
import torch

from .. import config as C9
from .. import splits as S9
from ..tail import build_tail, check_layer
from . import config as C

import build_cache


# --------------------------------------------------------------- W1 and W2 ---
def pool_basenames(pool=C.HOLDOUT_POOL, block=C.BLOCK):
    directory = C.FEATURE_POOL / pool / "dinov2_vitl14" / block
    return sorted(p.stem for p in directory.iterdir() if p.suffix == ".npy")


def split_basenames(name):
    """Names only, straight off the frozen manifest.

    ``load_split("measure")`` refuses to hand back a sealed path, and rightly
    so, but W1 needs the measure *names* to prove disjointness.  A list of
    filenames carries no measurement, so it is read directly here; nothing in
    this module opens the measure arrays.
    """
    return (S9.SPLIT_DIR / f"{name}_names.txt").read_text().split()


def check_overlap(holdout):
    report = {}
    for name in ("train_core", "cal", "dev", "measure"):
        shared = sorted(set(holdout) & set(split_basenames(name)))
        report[name] = len(shared)
        if shared:
            raise SystemExit(
                f"INVALID_EXPERIMENT (W1): holdout-500 shares {len(shared)} "
                f"basenames with {name}, e.g. {shared[:3]}")
    return report


def image_classes(names, truth_path=C.DEVKIT_GROUND_TRUTH):
    """ILSVRC2012 validation class id per basename, from the devkit truth."""
    truth = truth_path.read_text().split()
    classes = []
    for name in names:
        index = int(name.rsplit("_", 1)[1])          # ILSVRC2012_val_00000001
        if not 1 <= index <= len(truth):
            raise SystemExit(f"INVALID_EXPERIMENT (W2): {name} is outside the "
                             f"{len(truth)}-image validation set")
        classes.append(int(truth[index - 1]))
    return np.asarray(classes, dtype=np.int64)


def check_stratification(names):
    classes = image_classes(names)
    _, counts = np.unique(classes, return_counts=True)
    report = {"images": int(len(classes)),
              "distinct_classes": int(len(counts)),
              "max_per_class": int(counts.max()),
              "singletons": int((counts == 1).sum())}
    if report["max_per_class"] != 1 or report["singletons"] != len(classes):
        raise SystemExit(
            "INVALID_EXPERIMENT (W2): holdout-500 is not one image per class "
            f"({report}).  The resampling unit for the paired bootstrap would "
            "then be the class, not the image, and unseal.py's per-image "
            "resampling would understate the variance.")
    return report


# ------------------------------------------------------------------ W3, W4 ---
# W4 is a provenance check: it must fail when the cache was produced by a
# different tail, different layer or different images, and pass otherwise.
#
# The first attempt compared the teacher *activations* element by element and
# failed at 9.6e-7 to 1.8e-6 relative against the 9.54e-7 floor -- including on
# whole batches, so it was not the ragged final chunk.  That comparison was the
# wrong quantity, not a defect and not a reason to widen the floor:
# REPLAY_REL_TOL was calibrated in v9 on the per-image *distortion*, a sum of
# 257 x 1024 squared terms in which the reduction noise largely cancels, while
# an elementwise max over 1.3e8 activations is an extreme-value statistic that
# grows with the number of elements compared.  Widening the constant so that the
# larger statistic fits would loosen every distortion gate in v9 and v10 at once
# to satisfy a check nothing downstream depends on.
#
# So W4 is stated in the quantity v10 actually consumes.  The teacher enters
# only through D_n = ||tail(x_hat_n) - teacher_n||^2, so the test is: recompute
# the teacher, and require the frozen v9 codec's per-image hard distortion to be
# the same under the cached and the recomputed teacher, to within the frozen
# REPLAY_REL_TOL.  The elementwise activation deviation is still reported, as a
# recorded number rather than a gate, and a deliberately mispaired teacher
# (rolled by one image) is measured alongside it so the record shows what the
# gate's discriminating power actually is.
REPLAY_SCRATCH = C.CACHE / (f"teacher_holdoutreplay_{C.BLOCK}_"
                            f"n{C.N_HOLDOUT}_{C.CACHE_TAG}.npy")


def _resident(teacher_path, device):
    from ..engine import ResidentSet
    return ResidentSet(C.HOLDOUT_FEATURES, teacher_path,
                       np.arange(C.N_HOLDOUT), device)


def check_teacher_replay(device, batch):
    from ..engine import configure_precision, evaluate_allocations, \
        uniform_allocation
    from ..frozen import load_checked

    check_layer(C.LAYER, C.HOLDOUT_FEATURES, C.HOLDOUT_TEACHERS)   # W3
    configure_precision()
    tail = build_tail(C.LAYER, device)

    # 1. Recompute the whole teacher through the same tail, into scratch.
    features = np.load(C.HOLDOUT_FEATURES, mmap_mode="r")
    cached = np.load(C.HOLDOUT_TEACHERS, mmap_mode="r")
    fresh = np.lib.format.open_memmap(
        REPLAY_SCRATCH, mode="w+", dtype=np.float32, shape=features.shape)
    worst_abs, worst_rel = build_cache._forward(
        tail, features, fresh, batch, device, ref=cached)
    fresh.flush()
    del fresh

    try:
        anchor = C.ANCHOR_BY_NAME["R64"]
        codec, _ = load_checked(anchor, device)
        allocation = uniform_allocation(anchor)[None, :]

        resident_cached = _resident(C.HOLDOUT_TEACHERS, device)
        d_cached = evaluate_allocations(codec, tail, resident_cached,
                                        allocation)[0]
        # The mispairing control reuses the resident that is already in VRAM:
        # rolling the teacher by one image is a wrong cache in the strongest
        # sense the gate is meant to catch (right tail, right layer, wrong
        # images), and costs one extra evaluation.
        resident_cached.teacher = torch.roll(resident_cached.teacher, 1, 0)
        d_rolled = evaluate_allocations(codec, tail, resident_cached,
                                        allocation)[0]
        del resident_cached
        torch.cuda.empty_cache()

        resident_fresh = _resident(REPLAY_SCRATCH, device)
        d_fresh = evaluate_allocations(codec, tail, resident_fresh,
                                       allocation)[0]
        del resident_fresh
        torch.cuda.empty_cache()
    finally:
        REPLAY_SCRATCH.unlink(missing_ok=True)

    scale = float(d_cached.mean())
    deviation = float(np.abs(d_cached - d_fresh).max()) / scale
    control = float(np.abs(d_cached - d_rolled).max()) / scale
    report = {
        "anchor_used": anchor.name,
        "allocation": "uniform",
        "mean_distortion": scale,
        "distortion_rel_deviation": deviation,
        "tolerance": C.REPLAY_REL_TOL,
        "multiples_of_floor": deviation / C.REPLAY_REL_TOL,
        "mispaired_control_rel": control,
        "control_over_deviation": control / deviation if deviation else None,
        "activation_max_abs_diff": worst_abs,
        "activation_max_rel_diff": worst_rel,
        "activation_note": "recorded, not a gate -- see the comment above",
    }
    if deviation > C.REPLAY_REL_TOL:                               # W4
        raise SystemExit(
            f"INVALID_EXPERIMENT (W4): the frozen R64 codec's per-image hard "
            f"distortion moves by {deviation:.3e} relative when the teacher is "
            f"recomputed, above the {C.REPLAY_REL_TOL:.3e} floor.  The cache "
            f"was not produced by this tail.  Detail: {report}")
    return report


# -------------------------------------------------------------------- main ---
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16,
                        help="teacher forward batch; 16 is what built the "
                             "other four caches (teacher_forward.sh)")
    parser.add_argument("--force", action="store_true",
                        help="rebuild the caches even if they already exist")
    args = parser.parse_args(argv)

    torch.backends.cuda.matmul.allow_tf32 = C.ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = C.ALLOW_TF32
    device = torch.device("cuda")

    names = pool_basenames()
    if len(names) != C.N_HOLDOUT:
        raise SystemExit(f"INVALID_EXPERIMENT: the {C.HOLDOUT_POOL} pool holds "
                         f"{len(names)} images, expected {C.N_HOLDOUT}")

    print("=== W1 basename disjointness ===", flush=True)
    overlap = check_overlap(names)
    print(json.dumps(overlap), flush=True)

    print("=== W2 class stratification ===", flush=True)
    stratification = check_stratification(names)
    print(json.dumps(stratification), flush=True)

    C.V10.mkdir(parents=True, exist_ok=True)

    if args.force or not C.HOLDOUT_FEATURES.exists():
        print("=== stacking features ===", flush=True)
        build_cache.cmd_features(Namespace(
            pool=C.HOLDOUT_POOL, block=C.BLOCK, index=None,
            out=str(C.HOLDOUT_FEATURES)))
    else:
        print(f"features cache exists: {C.HOLDOUT_FEATURES}", flush=True)

    if args.force or not C.HOLDOUT_TEACHERS.exists():
        print("=== teacher forward ===", flush=True)
        check_layer(C.LAYER, C.HOLDOUT_FEATURES, C.HOLDOUT_TEACHERS)
        build_cache.cmd_teacher(Namespace(
            layer=C.LAYER, features=str(C.HOLDOUT_FEATURES),
            out=str(C.HOLDOUT_TEACHERS), batch=args.batch))
    else:
        print(f"teacher cache exists: {C.HOLDOUT_TEACHERS}", flush=True)

    shape = np.load(C.HOLDOUT_FEATURES, mmap_mode="r").shape
    teacher_shape = np.load(C.HOLDOUT_TEACHERS, mmap_mode="r").shape
    if shape != teacher_shape or shape[0] != C.N_HOLDOUT:
        raise SystemExit(f"INVALID_EXPERIMENT: cache shapes {shape} / "
                         f"{teacher_shape} disagree or are not "
                         f"{C.N_HOLDOUT} rows")

    print("=== W3/W4 teacher replay ===", flush=True)
    replay = check_teacher_replay(device, args.batch)
    print(json.dumps(replay), flush=True)

    manifest = {
        "set": "holdout-500",
        "role": "the only confirmatory set of plan v10; single unsealing",
        "pool": C.HOLDOUT_POOL,
        "block": C.BLOCK,
        "layer": C.LAYER,
        "cache_tag": C.CACHE_TAG,
        "count": int(shape[0]),
        "shape": list(shape),
        "features": str(C.HOLDOUT_FEATURES),
        "teachers": str(C.HOLDOUT_TEACHERS),
        "first": names[0], "last": names[-1],
        "W1_overlap_with": overlap,
        "W2_stratification": stratification,
        "W4_replay": replay,
        "measure_3000": "unsealed by v9; historical baseline only, not read "
                        "by v10",
    }
    C.HOLDOUT_MANIFEST.write_text(json.dumps(manifest, indent=2))
    (C.V10 / "holdout_names.txt").write_text("\n".join(names) + "\n")
    print(f"wrote {C.HOLDOUT_MANIFEST}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
