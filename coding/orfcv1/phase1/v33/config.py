"""Paths and defaults for V33 two-stage allocation training."""

from pathlib import Path

from .. import config as BASE
from ..v12 import config as V12

REPO = BASE.REPO
PHASE1 = BASE.PHASE1
V33 = PHASE1 / "v33"
CACHE = BASE.CACHE

BLOCK = BASE.BLOCK
LAYER = BASE.LAYER
GROUPS = BASE.GROUPS
DIM = BASE.DIM
ANCHORS = BASE.ANCHORS
ANCHOR_BY_NAME = BASE.ANCHOR_BY_NAME
NORM_MODE = BASE.NORM_MODE

ALLOW_TF32 = BASE.ALLOW_TF32
EVAL_IMAGE_BATCH = BASE.EVAL_IMAGE_BATCH
ORTH_TOL = V12.ORTH_TOL

# Reuse the v12 blk20 feature / teacher caches and OPQ init artefacts.
TRAIN_FEATURES = V12.TRAIN_FEATURES
TRAIN_TEACHERS = V12.TRAIN_TEACHERS
VAL_FEATURES = V12.VAL_FEATURES
VAL_TEACHERS = V12.VAL_TEACHERS
N_TRAIN = V12.N_TRAIN
N_VAL = V12.N_VAL

TRAIN_SEED = 20260807
SLATE_SEED = 20260808
DEFAULT_EPOCHS = 100
DEFAULT_LR = 3e-4
DEFAULT_BATCH = 32
DATALOADER_WORKERS = 4
LR_FLOOR_RATIO = V12.LR_FLOOR_RATIO
LOG_EVERY = 25
VAL_EVERY = 100
CKPT_EVERY = 500
GRAD_CLIP = 1.0
# Small L2 on Cayley skew params breaks the U0 / block-L gauge overlap in phase 1.
DEFAULT_L_DECAY = 1e-4
DEFAULT_TAU_START = 0.5
DEFAULT_TAU_END = 0.005
CHECKPOINT_FORMAT = "v33_codec_v1"


_SNAPSHOT = (
    "BLOCK", "LAYER", "GROUPS", "DIM", "ANCHORS", "ANCHOR_BY_NAME",
    "NORM_MODE", "EVAL_IMAGE_BATCH", "TRAIN_FEATURES", "TRAIN_TEACHERS",
    "VAL_FEATURES", "VAL_TEACHERS", "N_TRAIN", "N_VAL",
)


def activate(block):
    """Retarget V33 at another block, e.g. ``blk05``.

    The names above are copied out of the v12 config when this module is
    imported, so switching blocks through ``v21.config.activate`` alone would
    leave ``LAYER`` at 20 and pair another block's caches with the blk20 tail.
    Re-read them here after the switch.

    Callers that never invoke this keep the legacy blk20 wiring, including the
    ``phase1/v12/init_orfc_adam`` OPQ init the blk20 runs were validated on —
    ``activate`` moves the init root to ``phase1/v21/<block>/init_orfc_adam``.
    """
    from ..v21.config import activate as _activate

    _activate(block)
    globals().update({name: getattr(V12, name) for name in _SNAPSHOT})
    return globals()["BLOCK"]


def load_split(name):
    return V12.load_split(name)


def cosine_lr(step, total, base, floor_ratio=LR_FLOOR_RATIO):
    return V12.cosine_lr(step, total, base, floor_ratio=floor_ratio)


def v12_init_dir(anchor, parameterization="orfc_cayley"):
    return V12.init_dir(anchor, parameterization)


def output_dir(anchor, run_id):
    return V33 / str(run_id) / anchor.name


def ensure_run_id(run_id):
    value = str(run_id)
    if not value or Path(value).name != value:
        raise ValueError("run_id must be one non-empty path component")
    return value
