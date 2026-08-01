"""Frozen geometry and data contract for v12 joint allocation learning."""

from pathlib import Path

import numpy as np

from .. import config as BASE

REPO = BASE.REPO
PHASE1 = BASE.PHASE1
V12 = PHASE1 / "v12"
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
EVAL_PAIR_BUDGET = BASE.EVAL_PAIR_BUDGET
REPLAY_REL_TOL = BASE.REPLAY_REL_TOL
ORTH_TOL = 1e-4

TRAIN_FEATURES = CACHE / "features_train_blk20_n5000_v12.npy"
TRAIN_TEACHERS = CACHE / "teacher_train_blk20_n5000_v12.npy"
VAL_FEATURES = CACHE / "features_test_blk20_n3000_ss20260730.npy"
VAL_TEACHERS = CACHE / "teacher_test_blk20_n3000_ss20260730.npy"
N_TRAIN = 5000
N_VAL = 500

OPQ_ITERS = BASE.OPQ_ITERS
OPQ_KMEANS_ITERS = BASE.OPQ_KMEANS_ITERS
OPQ_SEED = BASE.OPQ_SEED
OPQ_ORTH_TOL = BASE.OPQ_ORTH_TOL
CODEBOOK_KMEANS_ITERS = BASE.CODEBOOK_KMEANS_ITERS
CODEBOOK_SEED = BASE.CODEBOOK_SEED

TRAIN_SEED = 20261201
POLICY_SEED = 20261202
POLICY_SAMPLES = 4
DEFAULT_POLICY_LR = 1e-2
DEFAULT_TEMPERATURE = 1.0
DEFAULT_ENTROPY_WEIGHT = 1e-2
DEFAULT_EPOCHS = 100
DEFAULT_LR_U = 1e-3
DEFAULT_LR_THETA = 3e-4
DEFAULT_BATCH = 32
LR_FLOOR_RATIO = 0.01
LOG_EVERY = 25
VAL_EVERY = 100
REVIVE_EVERY = 50
CODEBOOK_MOMENTUM = 0.0
CAYLEY_FIXED_POINT_ITERATIONS = 5
CAYLEY_REORTH_EVERY = 100
STAGE1_VAL_SAMPLES = 4
STAGE1_FINAL_SAMPLES = 16
STAGE1_STABILITY_POINTS = 5
STAGE1_VAL_SEED = 20261204


def load_split(name):
    if name == "train_fit":
        paths, count = (TRAIN_FEATURES, TRAIN_TEACHERS), N_TRAIN
    elif name == "train_val":
        paths, count = (VAL_FEATURES, VAL_TEACHERS), N_VAL
    else:
        raise PermissionError(f"unknown v12 split {name!r}")
    feature, teacher = paths
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return feature, teacher, np.arange(count, dtype=np.int64), []


def cosine_lr(step, total, base, floor_ratio=LR_FLOOR_RATIO):
    t = min(max(int(step), 0), int(total))
    floor_ratio = float(floor_ratio)
    if not 0 <= floor_ratio <= 1:
        raise ValueError("floor_ratio must lie in [0, 1]")
    cosine = 0.5 * (1.0 + np.cos(np.pi * t / float(total)))
    return float(base) * (floor_ratio + (1.0 - floor_ratio) * cosine)


def init_dir(anchor):
    return V12 / "init" / anchor.name


def output_dir(anchor, run_id):
    return V12 / str(run_id) / anchor.name


def ensure_run_id(run_id):
    value = str(run_id)
    if not value or Path(value).name != value:
        raise ValueError("run_id must be one non-empty path component")
    return value
