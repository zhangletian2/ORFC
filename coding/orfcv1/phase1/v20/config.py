"""Runtime profiles for the blk20 nominal-rate sweep."""

from ..config import Anchor
from ..v12 import config as C


SPECS = {
    "R96": (32, 32, Anchor("R96", 96, 3, (2, 3, 4))),
    "R128": (32, 32, Anchor("R128", 128, 4, (3, 4, 5))),
    "R256": (32, 32, Anchor("R256", 256, 8, (6, 7, 8, 9, 10))),
    "R512": (64, 16, Anchor("R512", 512, 8, (6, 7, 8, 9, 10))),
}


def activate(profile):
    if profile not in SPECS:
        raise ValueError(f"unknown v20 profile {profile!r}")
    groups, dim, anchor = SPECS[profile]
    C.PLAN = "v20_blk20_alpha025_rate_sweep"
    C.BLOCK, C.LAYER = "blk20", 20
    C.GROUPS, C.DIM = groups, dim
    C.ANCHORS = (anchor,)
    C.ANCHOR_BY_NAME = {anchor.name: anchor}
    C.V12 = C.PHASE1 / "v20" / profile
    C.N_TRAIN, C.N_VAL = 5000, 500
    C.TRAIN_FEATURES = C.CACHE / "features_train_blk20_n5000_v12.npy"
    C.TRAIN_TEACHERS = C.CACHE / "teacher_train_blk20_n5000_v12.npy"
    C.VAL_FEATURES = C.CACHE / "features_test_blk20_n3000_ss20260730.npy"
    C.VAL_TEACHERS = C.CACHE / "teacher_test_blk20_n3000_ss20260730.npy"
    C.EVAL_IMAGE_BATCH = 100
    C.EVAL_PAIR_BUDGET = 100
    return C, anchor
