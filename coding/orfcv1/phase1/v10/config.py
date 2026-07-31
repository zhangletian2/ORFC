"""Frozen constants for plan v10.

Written before any v10 code produced a number (plan section 2).  Changing
anything here after Stage B has run is a new plan version, v11, not a patch --
this is the same discipline v9's config carries and for the same reason: v8
died at a gate it had chosen after seeing an outcome.

What v10 inherits verbatim from v9, by importing rather than restating:
geometry (G=32, d=32, blk20), the two anchors and their three-mode menus, the
normalisation mode, the cache tag, TF32-off, the evaluation batching, and the
fp32 replay tolerance.  If any of those moved, v10's Stage B replay invariant
(W9) would fail on the first swap and the run would abort -- which is the point.

What is genuinely new here:

* ``holdout-500`` -- the only confirmatory set of this round.  measure-3000 was
  unsealed by v9 and is *not* reused: it is a historical v9 baseline and v10
  does not touch it.
* the greedy one-bit swap search constants, shared by Stage B and by A2's outer
  loop (one implementation, no second copy);
* the three arm names and their run directories;
* the statistics seed (a fresh one, so v10's bootstrap is not v9's).

Deliberately *not* here: the training hyper-parameters.  ``lr_U``, ``lr_theta``,
``batch``, ``N_inner``, ``T_outer`` and ``delta_N`` are measured once by
``profile.py`` on train-core + cal and written to ``profile.json``; ``train.py``
refuses to start without that file.  Fixing them by profiling before the formal
runs, rather than by choosing them afterwards, is what makes "matched budget"
a fact rather than a claim.
"""

from pathlib import Path

from .. import config as C9

# ------------------------------------------------------- inherited verbatim ---
REPO = C9.REPO
ARTIFACTS = C9.ARTIFACTS
CACHE = C9.CACHE
SPLIT = C9.SPLIT
PHASE1 = C9.PHASE1
FEATURE_POOL = C9.FEATURE_POOL

BLOCK = C9.BLOCK
LAYER = C9.LAYER
CACHE_TAG = C9.CACHE_TAG
NORM_MODE = C9.NORM_MODE

GROUPS = C9.GROUPS
DIM = C9.DIM

ANCHORS = C9.ANCHORS
ANCHOR_BY_NAME = C9.ANCHOR_BY_NAME

EVAL_IMAGE_BATCH = C9.EVAL_IMAGE_BATCH
EVAL_PAIR_BUDGET = C9.EVAL_PAIR_BUDGET
ALLOW_TF32 = C9.ALLOW_TF32
REPLAY_REL_TOL = C9.REPLAY_REL_TOL

# v10's own tolerance on ||U^T U - I||_F, deliberately not v9's OPQ_ORTH_TOL.
# v9 never trained U, so its 1e-5 only ever had to cover a rotation that had been
# orthogonalised once in float64 and cast down (the frozen U sits at 8.99e-07).
# Training in fp32 cannot hold that: measured on this GPU for the 1024x1024
# rotation, with ||U^T U - I||_F evaluated in float64,
#
#     fp32 QR (CayleySGD's _project_qr)      1.825e-05   <- the projection floor
#     drift over 100 steps, no projection    1.150e-05   <- and it grows as sqrt(n)
#     float64 QR, cast back to fp32          1.149e-06
#
# so an fp32 pipeline sits around 2e-05 no matter how U is trained, and 1e-5 is
# unreachable by construction rather than by any property of the experiment.
# v10 runs everything in fp32, as ratified, and sets the tolerance to 1e-4 --
# roughly a 5x margin over the fp32 floor, and still four orders below the scale
# at which a non-orthogonal U would distort the rate accounting.  v9's constant
# is left alone so that relaxing this cannot reach backwards into v9's asserts.
ORTH_TOL = 1e-4

# --------------------------------------------------------------- v10 paths ---
V9 = C9.V9
V10 = PHASE1 / "v10"


def anchor_v9_root(anchor):
    """Where v9 left this anchor's frozen (U_0, Theta_0)."""
    return V9 / anchor.name


def run_dir(anchor, arm):
    return V10 / anchor.name / arm


# ------------------------------------------------------------- holdout-500 ---
# The 500-image ``test`` feature pool.  Never used by v8 or v9: it is not one of
# the four sets in phase1/splits.py, and its basenames are disjoint from both
# the train pool (5,000) and the val pool (3,000) that those four sets are cut
# from.  W1 recomputes that disjointness rather than trusting this comment.
HOLDOUT_POOL = "test"
N_HOLDOUT = 500
HOLDOUT_FEATURES = CACHE / f"features_holdout_{BLOCK}_n{N_HOLDOUT}_{CACHE_TAG}.npy"
HOLDOUT_TEACHERS = CACHE / f"teacher_holdout_{BLOCK}_n{N_HOLDOUT}_{CACHE_TAG}.npy"
HOLDOUT_MANIFEST = V10 / "holdout_manifest.json"

# Stratification is checked, not assumed (W2).  The pool is one image per class
# over 500 distinct classes, so the stratum unit *is* the image and the per-image
# paired bootstrap already is the stratified bootstrap.  If that ever stops
# being true, W2 fails and the resampling unit has to be reconsidered before any
# unsealing -- which is why it is an invariant and not a remark.
DEVKIT_GROUND_TRUTH = Path(
    "/data4/workspace/zlt/featcodec/data/imagenet/ILSVRC2012_devkit_t12/"
    "data/ILSVRC2012_validation_ground_truth.txt")

SEAL_PATH = V10 / "SEAL.json"
UNSEAL_PATH = V10 / "unseal.json"
UNSEAL_TOKEN = V10 / "HOLDOUT_UNSEALED.json"
CODE_SNAPSHOT = V10 / "code_snapshot"

# ------------------------------------------------------------ outer search ---
# Shared by Stage B and by A2's outer loop.  One implementation lives in
# search.py; these are its only tunables and both are fixed here.
S_OUTER_MAX = 8          # at most 8 accepted one-bit swaps
EPS_ACCEPT_MULT = 10.0   # accept only if the gain exceeds 10 x noise floor x D
TIE_BREAK = C9.TIE_BREAK  # lexicographic on (down_group, up_group)

# A2 spends its swap budget one swap at a time, ``T_OUTER`` inner steps apart,
# so the inner learner always gets a window to adapt to the allocation it was
# just given.  With one swap per event and ``S_OUTER_MAX`` events, the inner
# length is (S_OUTER_MAX + 1) * T_outer -- the extra block is there so the final
# allocation is trained under, rather than chosen and then never used.
OUTER_SWAPS_PER_EVENT = 1


def inner_steps(t_outer, s_outer_max=S_OUTER_MAX):
    return int(t_outer) * (int(s_outer_max) + 1)


# ------------------------------------------------------------- inner loop ---
# One seed for all three arms of an anchor: the shuffled batch stream is a
# function of (TRAIN_SEED, batch, train-core size) only, never of the arm, so
# A1, A2 and A3 see the identical image sequence.  verify.py compares the
# recorded streams with array_equal rather than trusting this comment.
TRAIN_SEED = 20261002
LOG_EVERY = 25
REVIVE_EVERY = 50        # dead-codeword revival, identical in all three arms

# Momentum-free optimisers on both sides, deliberately.  CayleySGD is already
# momentum-free and self-normalising; the codebooks use plain SGD so that a
# codeword with an exactly zero gradient receives an exactly zero update.  With
# Adam, a mode that stops being selected after a swap would keep drifting under
# stale momentum -- an effect A1 (which never swaps) could not have, so it would
# be an asymmetry between the arms rather than a shared cost.
CODEBOOK_MOMENTUM = 0.0
CAYLEY_FIXED_POINT_ITERATIONS = 5

# Reorthogonalisation is CayleySGD's own, in fp32, on its own schedule -- v10
# adds nothing here and cayley.py is not modified.  The period stays at 100; the
# tolerance that period has to satisfy is ORTH_TOL = 1e-4, set above with the
# measurements that fix it.  train.py samples ||U^T U - I||_F on the step before
# each projection fires, i.e. at the worst point of the cycle, and W7 is checked
# against that peak rather than against the post-projection value -- measuring
# only after the projection would report the projection's own residual and would
# be blind to drift, which is exactly how the first version of this hid a real
# problem behind a constant 1.8e-05.
CAYLEY_REORTH_EVERY = 100
ORTH_FP32_QR_FLOOR = 1.825e-05
ORTH_DRIFT_PER_100_STEPS = 1.150e-05

# --------------------------------------------------------------- the arms ---
ARM_EQUAL_STEP = "uniform-equal-step"      # A1, primary comparator
ARM_JOINT = "nonuniform-joint"             # A2, the arm under test
ARM_EQUAL_COMPUTE = "uniform-equal-compute"  # A3, secondary robustness
ARMS = (ARM_EQUAL_STEP, ARM_JOINT, ARM_EQUAL_COMPUTE)

# --------------------------------------------------------------- statistics ---
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 20261001       # v10's own seed; v9 used 20260731
FWER = 0.05                     # Holm over the {R64, R96} family

# ------------------------------------------------------------------- gates ---
# G1: the inner learner must actually move.  100 x the replay floor is three
# orders below any effect worth reporting and far above fp32 noise, so a failure
# here means the straight-through gradient is useless, not that the effect is
# small.
G1_TRAIN_IMPROVE_MULT = 100.0

def profile_path(anchor):
    """One profile per anchor: the loss scale and the step counts differ."""
    return V10 / anchor.name / "profile.json"


def load_profile(anchor):
    """The hyper-parameters fixed by C0.  Missing file is a hard stop."""
    import json
    path = profile_path(anchor)
    if not path.exists():
        raise SystemExit(
            f"INVALID_EXPERIMENT: {path} does not exist.  Stage C0 "
            "(profile.py) must fix lr_U / lr_theta / batch / T_outer / delta_N "
            "and the loss scale before any formal run, so that the budget is "
            "matched by construction rather than chosen after the fact.")
    return json.loads(path.read_text())
