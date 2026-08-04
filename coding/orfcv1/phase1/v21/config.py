"""V21 contracts: ORFC-rate objective at blk05/blk20, R64/R96."""

from ..config import Anchor
from ..v12 import config as C


SPECS = {
    "blk05": (5, (
        Anchor("R64", 64, 2, (1, 2, 3)),
        Anchor("R96", 96, 3, (2, 3, 4)))),
    "blk20": (20, (
        Anchor("R64", 64, 2, (1, 2, 3)),
        Anchor("R96", 96, 3, (2, 3, 4)))),
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
