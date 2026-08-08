"""V21 contracts: ORFC-rate objective across ViT split depths, R64/R96."""

from ..config import ANCHORS as BASE_ANCHORS
from ..v12 import config as C


# Every depth offers the same rate ladder; only the tail layer differs.
_LADDER = BASE_ANCHORS

SPECS = {
    "blk05": (5, _LADDER),
    "blk10": (10, _LADDER),
    "blk15": (15, _LADDER),
    "blk20": (20, _LADDER),
}

RATE_LAMBDA = 0.5
PRIOR_FLOOR = 0.0
FAIR_LOSS_ALPHA = 0.25


def activate(block):
    if block not in SPECS:
        raise ValueError(f"unknown block {block!r}")
    layer, anchors = SPECS[block]
    C.PLAN = "v21_entropy_constrained_joint"
    C.BLOCK, C.LAYER = block, layer
    C.GROUPS, C.DIM = 32, 32
    C.ANCHORS = anchors
    C.ANCHOR_BY_NAME = {anchor.name: anchor for anchor in anchors}
    C.V12 = C.PHASE1 / "v21" / block
    C.N_TRAIN, C.N_VAL = 5000, 500
    C.TRAIN_FEATURES = C.CACHE / f"features_train_{block}_n5000_v15.npy"
    C.TRAIN_TEACHERS = C.CACHE / f"teacher_train_{block}_n5000_v15.npy"
    if block == "blk20":
        C.TRAIN_FEATURES = C.CACHE / "features_train_blk20_n5000_v12.npy"
        C.TRAIN_TEACHERS = C.CACHE / "teacher_train_blk20_n5000_v12.npy"
    C.VAL_FEATURES = C.CACHE / f"features_test_{block}_n3000_ss20260730.npy"
    C.VAL_TEACHERS = C.CACHE / f"teacher_test_{block}_n3000_ss20260730.npy"
    C.EVAL_IMAGE_BATCH = 100
    C.EVAL_PAIR_BUDGET = 100
    return C
