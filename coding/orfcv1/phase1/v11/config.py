"""Frozen constants for plan v11 (``阶段一计划_v11_ORFC热启动联合学习_20260731``).

Written before any v11 code produced a number.  Editing anything here after N14
has run is a new plan version, not a patch.

v11's objective is one line::

    min_{U, Theta, m in A_R}  D(U, Theta, m)

with an inner hard-STE joint update of ``(U, Theta)`` and an outer full-hard
one-bit search over ``m`` on cal-500.  The outer search is part of the
optimisation, not a post-hoc analysis of a trained model.

What is inherited verbatim from v9, by importing rather than restating: geometry
(G=32, d=32, blk20), the two anchors and their three-mode bit menus, the
normalisation mode, the cache tag, TF32-off, the evaluation batching, and the
fp32 replay tolerance.  Those are the things that make v9's frozen cal matrix,
v10's Stage B replay and v11's bridge comparable at all.

What v11 changes relative to v10, and why:

* **A0 is an ORFC checkpoint, not v9's OPQ+k-means object.**  v10's endpoint was
  ``D(A2) - D(A1)``, an internal contrast; the user's ruling is that the main
  endpoint returns to "beat ORFC".  So the warm start *is* the strongest legal
  same-rate ORFC checkpoint, selected on dev-500 under hard nearest neighbour,
  and A1/A2 begin as elementwise copies of it.
* **Every mode codebook the outer search may invoke is trained.**  v10's defect 1
  was that a uniform allocation gives modes 0 and 2 an exactly zero gradient, so
  the outer search was always offered stale codebooks and accepted 0 of 16 swaps.
  v11 adds a deterministic three-step menu-coverage auxiliary term (section 6.2
  of the plan); ``AUX_BETA``, ``MENU_CYCLE`` and the permutation seed below are
  its only tunables and all three are fixed here.
* **The learning rate is chosen at full length under a cosine schedule.**  v10's
  defect 2 was a 200-step probe selecting a rate for a 3096-step run: at 200
  steps the largest rate always wins, so the argmin necessarily sat on the grid
  edge, and the edge was the direction that diverged.  Short probes now only
  screen for numerical validity and produce a shortlist; the formal choice comes
  from full-length annealed runs scored on train-val.

Deliberately *not* here: ``lr_U``, ``lr_theta``, ``batch``, ``T_outer``,
``N_inner``.  Those are measured by ``profile.py`` and written to
``profile.json``; ``train.py`` refuses to start without that file.

Everything in v11 is float32.  The only ``float64`` in the whole ORFC source tree
is the SVD inside ``cayley.DirectOrthogonalTransform.init_from_opq``, an
initialisation helper that v11 must not call anyway (it would reorthogonalise the
ORFC rotation and break W-I1).  ORFC's own ``OrthogonalTransform.get_rotation``
solves ``(I-A)^-1 (I+A)`` in the parameter dtype, i.e. fp32.  The orthogonality
diagnostic is fp32 too; see ``ORTH_MEASURE_FLOOR`` for the measurement that makes
that safe.
"""

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

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

EVAL_IMAGE_BATCH = C9.EVAL_IMAGE_BATCH
EVAL_PAIR_BUDGET = C9.EVAL_PAIR_BUDGET
ALLOW_TF32 = C9.ALLOW_TF32
REPLAY_REL_TOL = C9.REPLAY_REL_TOL
TIE_BREAK = C9.TIE_BREAK

CODEBOOK_KMEANS_ITERS = C9.CODEBOOK_KMEANS_ITERS
CODEBOOK_SEED = C9.CODEBOOK_SEED

V9 = C9.V9
V10 = PHASE1 / "v10"
V11 = PHASE1 / "v11"


# --------------------------------------------------------------- anchors ---
class Anchor(NamedTuple):
    """One low-rate anchor: a uniform point and its two neighbouring modes.

    Field-for-field identical to v9's ``Anchor`` except that ``root`` points at
    v11's directory.  ``assert_anchor_menus_match_v9`` below re-checks the bit
    menus against v9 elementwise, so this second declaration cannot silently
    drift away from the object v9's cal matrix and v10's Stage B were measured
    on -- which is the only reason the bridge replay in N16 means anything.
    """

    name: str
    rate: int
    uniform_bits: int
    mode_bits: tuple

    @property
    def mode_sizes(self):
        return tuple(2 ** b for b in self.mode_bits)

    @property
    def uniform_mode(self):
        return self.mode_bits.index(self.uniform_bits)

    @property
    def down_mode(self):
        return self.uniform_mode - 1

    @property
    def up_mode(self):
        return self.uniform_mode + 1

    @property
    def root(self):
        return V11 / self.name


ANCHORS = tuple(Anchor(name=a.name, rate=a.rate, uniform_bits=a.uniform_bits,
                       mode_bits=tuple(a.mode_bits)) for a in C9.ANCHORS)
ANCHOR_BY_NAME = {a.name: a for a in ANCHORS}


def assert_anchor_menus_match_v9():
    """v11's anchors must be v9's anchors in every field but ``root``."""
    for mine, theirs in zip(ANCHORS, C9.ANCHORS):
        if (mine.name, mine.rate, mine.uniform_bits, tuple(mine.mode_bits)) != \
           (theirs.name, theirs.rate, theirs.uniform_bits,
                tuple(theirs.mode_bits)):
            raise SystemExit(
                f"INVALID_EXPERIMENT: v11 anchor {mine} disagrees with v9's "
                f"{theirs}; the frozen cal matrix and Stage B replay would no "
                f"longer refer to the same menu")
    return True


assert_anchor_menus_match_v9()

# The uniform codebook size is what makes an ORFC checkpoint "same rate": R64's
# uniform mode is K=4, R96's is K=8.
UNIFORM_K = {a.name: a.mode_sizes[a.uniform_mode] for a in ANCHORS}


# ------------------------------------------------------- ORFC candidates ---
# The legal A0 pool, frozen before any of it was evaluated (plan section 3).
#
# Legality rule, stated as a rule rather than as a list, and re-derived from the
# directory by ``orfc_baseline.legal_candidates`` so that a checkpoint appearing
# or disappearing later is an INVALID_EXPERIMENT rather than a silent change of
# pool:
#
#   1. filename ``blk20_K{K}_emb32_*`` with K == the anchor's uniform codebook
#      size -- blk20 fixes the layer, emb32 fixes d = 32, K fixes the rate;
#   2. both the ``.npz`` and the ``.pt`` exist;
#   3. the ``.pt`` metadata carries G=32, d=32, D=1024, K as above, and
#      ``has_transform=True``.
#
# ECVQ does not enter this pool, and does not enter it *structurally* rather than
# by exclusion: v11's evaluator only ever does plain hard nearest neighbour
# (``engine.build_bank`` is a cdist argmin with no lambda bias), so an
# entropy-constrained assignment cannot be produced by this code path at all.
# The strongest ECVQ result is reported separately as a descriptive entry, and
# belongs to the real-entropy-rate stage, not to this fixed-nominal-rate one.
ORFC_CHECKPOINT_DIR = Path(
    "/data4/workspace/zlt/featcodec/ORFC/coding/orfc/checkpoints/dinov2_vitl14")

ORFC_CANDIDATES = {
    "R64": (
        "blk20_K4_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42",
        "blk20_K4_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s43",
        "blk20_K4_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s44",
        "blk20_K4_emb32_bt1024_ws_tau0.5_lr0.0003_ep300_n5000_s42",
        "blk20_K4_emb32_bt1024_ws_tau0.5_lr0.0005_ep300_n5000_s42",
        "blk20_K4_emb32_bt1024_ws_tau0.5_lr0.001_ep300_n5000_s42",
    ),
    "R96": (
        "blk20_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42",
        "blk20_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s43",
        "blk20_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s44",
    ),
}

# The descriptive "matched configuration" row of plan section 10: the same
# training recipe at both rates, so that "R64 used ep300/no-lambda while R96 used
# ep100/lambda0.5" can be shown not to be an asymmetry in the *rule* (the rule is
# "strongest per rate") by also reporting the matched pair.
ORFC_MATCHED_CONFIG = {
    "R64": "blk20_K4_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42",
    "R96": "blk20_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42",
}

# A0's selection metric and tie-break, both fixed before the first evaluation:
# the mean over dev-500 of the per-image hard-NN full-tail distortion at the
# uniform allocation; argmin; ties broken lexicographically on the stem.
A0_METRIC = "dev500_mean_per_image_hard_nn_full_tail_at_uniform"
A0_TIE_BREAK = "lexicographic_on_checkpoint_stem"

# The .npz stores R directly; the .pt stores the Cayley parameters it was solved
# from.  Recomputing R through ORFC's own OrthogonalTransform and comparing is a
# cross-check of the two files against each other without hashing either (the
# standing convention in this project: checkpoint id + metadata + key-tensor
# comparison).  The decisive identity check is on the codebooks, which involve no
# solve and are compared with ``array_equal``; R is supplementary and its gate is
# calibrated from the solve's own floor rather than guessed.
#
# Measured on this stack (R64 s42 / R96 s42):
#
#     ||A||_max                        1.210e+02   3.069e+01
#     max|R_pt(fp32) - R_npz|          9.459e-05   2.306e-05
#     max|R_pt(fp64) - R_npz|          6.108e-05   1.351e-05   <- stored vs true
#     max|R_pt(fp64) - R_pt(fp32)|     6.273e-05   1.445e-05   <- fresh vs true
#     ||dR||_F / ||R||_F               1.652e-04   4.570e-05
#     rms |R| entry                    3.125e-02   3.125e-02
#
# The stored R is exactly as close to the fp64 reference as an independently
# recomputed fp32 R is: both are fp32 solves of the same ill-conditioned system
# and they agree to the precision fp32 allows.  A genuine mismatch -- the wrong
# checkpoint -- would give a relative Frobenius deviation of order sqrt(2), four
# orders above the floor, so the gate is set scale-free at 1e-2: sixty times the
# measured floor and a hundred times below a real mismatch.
ORFC_R_RELATIVE_TOL = 1e-2

# ORFC's rotation is orthogonal by *construction* and only approximately so in
# *arithmetic*.  The Cayley form R = (I-A)^-1 (I+A) is exactly orthogonal for any
# skew A, but ORFC trains A freely and it grows large -- ||A||_max reaches 121 at
# R64 -- so I - A is ill-conditioned and the fp32 solve loses several digits.
# Measured on the stored checkpoints:
#
#     ||R^T R - I||_F      R64  5.052e-03      R96  1.401e-03
#
# i.e. 50x and 14x above ORTH_TOL.  This is a property of the object v11 is asked
# to beat, not of v11's arithmetic, and there are only two ways to respond.
# Projecting R onto the orthogonal manifold before evaluating would make A0 a
# codec ORFC never produced, and "we beat ORFC" would then be a claim about a
# baseline we had modified.  So A0 is evaluated **exactly as stored**, A1/A2 warm
# start from **exactly the same stored R** (which is what makes W-I1's
# array_equal hold at all), and the deviation is recorded rather than removed.
#
# The form of the invariant therefore moves to where it has consequences.
# ||R^T R - I||_F is an extensive quantity: spread over D directions it is a
# per-direction rms deviation of the singular values from 1, and *that* is what
# decides whether "rotation" still means rotation.  At R64 it is
# 5.052e-03 / sqrt(1024) = 1.58e-04, i.e. every direction is scaled to within
# 0.016%.  The gate below is 1e-2 -- every direction within 1% -- which is a
# consequence, not a form.
#
# W-I5 is restated accordingly (this is a wording correction of the same class as
# N10's W8 and N11's ORTH_TOL, recorded publicly rather than applied quietly):
#
#   (a) the inherited ORFC rotation's ||U_0^T U_0 - I||_F is *recorded*, and
#       gated only through ORTH_PER_DIRECTION_TOL;
#   (b) ORTH_TOL = 1e-4 keeps its full force on *training drift*, measured as the
#       increase over the initial value, so v11 still cannot degrade U.
#
# One consequence to state rather than discover later: CayleySGD's own QR
# projection fires every CAYLEY_REORTH_EVERY steps and will pull U from 5e-03 to
# the fp32 QR floor at step 100.  That is a real discontinuity in the trajectory,
# it is an improvement, and it happens identically in both arms -- so it cannot
# favour A2 over A1.  W-I3/W-I4 are step-0 invariants and are unaffected.
ORFC_ORTH_PER_DIRECTION_TOL = 1e-2

# A0 is only meaningful if the ORFC codebooks were fitted to features normalised
# the way v9's engine normalises them.  Neither the .npz nor the .pt records
# ``norm_mode``, so this is checked by measurement instead of trusted: the
# relative quantisation error in the rotated feature space,
# ``||z - z_hat||^2 / ||z||^2`` at the uniform allocation, must be below 1 --
# i.e. the codebook must explain more of the signal than it misses.  A
# normalisation mismatch puts this far above 1; a correct one puts it far below.
# The threshold is not tuned, it is the point where the codebook stops being a
# codebook.
ORFC_MAX_RELATIVE_QERROR = 1.0

# Provenance that is *derived*, not stored.  Neither checkpoint file carries a
# ``layer`` or a ``norm_mode`` field.  ``blk20`` comes from the filename, and
# ``norm_mode=per_image`` plus ``tail = blocks[layer+1:] = blocks[21:]`` come
# from ``run_soft_pq.py``'s frozen defaults (``layer_idx = int(args.layer[-2:])``
# and ``tail_blocks = list(wrapper.backbone.blocks[layer_idx + 1:])``).  N14's
# record must say this plainly rather than claim a stored field.
ORFC_PROVENANCE_NOTE = (
    "layer=20 is derived from the blk20 filename tag; norm_mode=per_image and "
    "tail=blocks[21:] are derived from run_soft_pq.py's frozen defaults.  "
    "Neither is stored in the .npz or the .pt.  The consistency of the "
    "normalisation is checked by measurement (ORFC_MAX_RELATIVE_QERROR), not "
    "by trusting this note.")


# ------------------------------------------------------------ data split ---
# Plan section 2.1.  Five sets, pairwise-disjoint basenames:
#
#   train-fit  3500   (U, Theta) gradients and the side-mode calibration
#   train-val   500   full-length hyper-parameter choice, G1/G2/G3
#   cal         500   A2's outer search
#   dev         500   A0 selection, N16 bridge, G4
#   holdout     500   one-shot confirmation after freezing (never read yet)
#
# train-fit and train-val are carved out of v9's train-core 4000 with v11's own
# seed.  v9's ``splits.py`` is not modified: the four v9 sets keep their exact
# row lists, and this carve is a further partition of one of them.
#
# Both sizes are multiples of 100 so ``engine.assert_uniform_shape`` holds and no
# evaluation call is ever a short final chunk of a different shape.
TRAINVAL_SEED = 20261101
N_TRAIN_FIT = 3500
N_TRAIN_VAL = 500

SPLIT_DIR = V11 / "split"
SPLIT_MANIFEST = SPLIT_DIR / "manifest.json"

HOLDOUT_POOL = "test"
N_HOLDOUT = 500
HOLDOUT_FEATURES = CACHE / f"features_holdout_{BLOCK}_n{N_HOLDOUT}_{CACHE_TAG}.npy"
HOLDOUT_TEACHERS = CACHE / f"teacher_holdout_{BLOCK}_n{N_HOLDOUT}_{CACHE_TAG}.npy"
HOLDOUT_MANIFEST = V10 / "holdout_manifest.json"

TRAINING_SETS = ("train_fit", "train_val", "cal", "dev")
CONFIRMATORY_SET = "holdout"


def _pool_basenames(pool):
    directory = FEATURE_POOL / pool / "dinov2_vitl14" / BLOCK
    return sorted(p.stem for p in directory.iterdir() if p.suffix == ".npy")


def holdout_basenames():
    """The 500 confirmatory basenames -- names only, never features.

    Reading the *names* is not reading the set: the overlap invariant cannot be
    checked without them, and a name carries no distortion.  The feature and
    teacher caches stay untouched until ``unseal.py``.
    """
    if HOLDOUT_MANIFEST.exists():
        manifest = json.loads(HOLDOUT_MANIFEST.read_text())
        for key in ("basenames", "names", "images"):
            if isinstance(manifest.get(key), list):
                return sorted(str(n) for n in manifest[key])
    return _pool_basenames(HOLDOUT_POOL)


def freeze_trainval_split(force=False):
    """Partition v9's train-core 4000 into train-fit 3500 / train-val 500.

    Idempotent.  Writes local indices *into v9's train_core row list* as well as
    the absolute cache rows, so that the relationship to v9's split is explicit
    in the manifest rather than implied by an ordering convention.
    """
    from .. import splits as splits9

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    if SPLIT_MANIFEST.exists() and not force:
        return json.loads(SPLIT_MANIFEST.read_text())

    _, _, core_rows, core_names = splits9.load_split("train_core")
    if len(core_rows) != N_TRAIN_FIT + N_TRAIN_VAL:
        raise SystemExit(
            f"INVALID_EXPERIMENT: train-core holds {len(core_rows)} images, but "
            f"v11 carves {N_TRAIN_FIT} + {N_TRAIN_VAL}")

    rng = np.random.default_rng(TRAINVAL_SEED)
    perm = rng.permutation(len(core_rows))
    val_local = np.sort(perm[:N_TRAIN_VAL])
    fit_local = np.sort(perm[N_TRAIN_VAL:])

    tables = {
        "train_fit": (fit_local, core_rows[fit_local],
                      [core_names[i] for i in fit_local]),
        "train_val": (val_local, core_rows[val_local],
                      [core_names[i] for i in val_local]),
    }

    # Five-way zero basename overlap, recomputed rather than asserted in prose.
    names = {name: table[2] for name, table in tables.items()}
    for other in ("cal", "dev"):
        names[other] = splits9.load_split(other)[3]
    names[CONFIRMATORY_SET] = holdout_basenames()
    for a in sorted(names):
        for b in sorted(names):
            if a < b:
                shared = set(names[a]) & set(names[b])
                if shared:
                    raise SystemExit(
                        f"INVALID_EXPERIMENT: {a} and {b} share "
                        f"{len(shared)} basenames, e.g. {sorted(shared)[:3]}")

    manifest = {"plan": "v11", "trainval_seed": TRAINVAL_SEED,
                "block": BLOCK, "cache_tag": CACHE_TAG,
                "derived_from": "v9 train_core (4000)", "sets": {}}
    for name, (local, rows, basenames) in tables.items():
        np.save(SPLIT_DIR / f"{name}_local.npy", local.astype(np.int64))
        np.save(SPLIT_DIR / f"{name}_rows.npy",
                np.asarray(rows, dtype=np.int64))
        (SPLIT_DIR / f"{name}_names.txt").write_text("\n".join(basenames) + "\n")
        manifest["sets"][name] = {"count": int(len(rows)),
                                  "first": basenames[0], "last": basenames[-1]}
    for name in ("cal", "dev", CONFIRMATORY_SET):
        manifest["sets"][name] = {"count": int(len(names[name])),
                                  "first": names[name][0],
                                  "last": names[name][-1]}
    manifest["pairwise_basename_overlap"] = 0
    SPLIT_MANIFEST.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_split(name):
    """``(feature_path, teacher_path, rows, basenames)`` for any v11 set.

    ``holdout`` is refused here on purpose.  Everything in v11 except
    ``unseal.py`` goes through this function, so "training and selection never
    read the confirmatory set" is enforced by the import graph rather than by
    remembering.
    """
    from .. import splits as splits9

    if name == CONFIRMATORY_SET:
        raise PermissionError(
            "holdout-500 is the confirmatory set; only unseal.py may read it "
            "(plan section 10).  Reaching it from training, profiling, search "
            "or checkpoint selection is INVALID_EXPERIMENT.")
    if name in ("cal", "dev"):
        return splits9.load_split(name)
    if name not in ("train_fit", "train_val"):
        raise PermissionError(f"unknown v11 split {name!r}")
    if not SPLIT_MANIFEST.exists():
        freeze_trainval_split()
    feature, teacher = splits9.cache_paths("train_core")
    rows = np.load(SPLIT_DIR / f"{name}_rows.npy")
    basenames = (SPLIT_DIR / f"{name}_names.txt").read_text().split()
    return feature, teacher, rows, basenames


# ------------------------------------------------- menu-coverage auxiliary ---
# Plan section 6.2.  The inner loss is
#
#     L = D(U, Theta, m_t) + beta * D(sg(U), Theta, a_s)
#
# where ``a_s`` cycles deterministically through three allocations that between
# them cover every (group, mode) exactly once per cycle:
#
#     s = 0   a^(1)    all 32 groups at the uniform mode
#     s = 1   a^(02)   permuted first half at mode 0, second half at mode 2
#     s = 2   ~a^(02)  the two halves swapped
#
# Each of the three sums to 32 mode indices, so each has exactly the anchor's
# nominal rate -- verified as an integer by ``engine.nominal_rate``, not argued.
#
# ``sg(U)`` is a stop-gradient: the auxiliary branch calibrates codebooks only,
# and contributes exactly zero gradient to U, so the main branch alone sets U's
# update direction.  The auxiliary term is *not* claimed to compute the inner
# optimum of a candidate allocation; its only job is to keep every codebook the
# outer search may invoke following the current U.
AUX_BETA = 1.0
MENU_CYCLE = 3
MENU_PERMUTATION_SEED = 20261101   # same stream in both arms; written to disk


def menu_cycle_allocations(anchor, permutation):
    """The three auxiliary allocations for one cycle, given a group permutation.

    Returns an ``int64`` array of shape ``(3, GROUPS)``.  Pure function of
    ``(anchor, permutation)``: both arms call it with the same permutation
    stream, and the realised stream is written to disk so "the arms saw the same
    auxiliary allocations" is a file comparison, not a claim about two RNGs.
    """
    permutation = np.asarray(permutation, dtype=np.int64)
    if permutation.shape != (GROUPS,) or \
            not np.array_equal(np.sort(permutation), np.arange(GROUPS)):
        raise SystemExit(f"INVALID_EXPERIMENT: {permutation!r} is not a "
                         f"permutation of 0..{GROUPS - 1}")
    half = GROUPS // 2
    out = np.full((MENU_CYCLE, GROUPS), anchor.uniform_mode, dtype=np.int64)
    out[1, permutation[:half]] = anchor.down_mode
    out[1, permutation[half:]] = anchor.up_mode
    out[2, permutation[:half]] = anchor.up_mode
    out[2, permutation[half:]] = anchor.down_mode
    return out


# W-I3 compares hard indices on dev-500 and on a fixed prefix of train-fit.  The
# prefix is one evaluation batch exactly, so the check runs through the same
# kernel path as every other tail call (engine.assert_uniform_shape's reason for
# existing), and it is a *prefix* of the frozen split rather than a sample, so
# there is no second seed to keep aligned between the arms.
WI_PROBE_IMAGES = EVAL_IMAGE_BATCH          # 100


# ------------------------------------------------------------- inner loop ---
TRAIN_SEED = 20261102
LOG_EVERY = 25
REVIVE_EVERY = 50          # revival covers m_t union a_s, same rule in both arms

# How often the train-val trajectory that G1 and G2 read is measured.  The plan
# says "the mean of the last 25 steps" and "a 25-step sliding mean" without
# saying how often train-val is evaluated, so the cadence is fixed here, before
# any arm runs, rather than being decided while looking at a curve.
#
# It cannot be every step.  train-val is 500 images; a training step at batch 64
# costs (1 + 2) x 64 = 192 forward-equivalents, so a per-step val evaluation
# would cost 500/192 = 2.6x the training itself and the experiment would be
# mostly measurement.  At VAL_EVERY = 25 the overhead is 10%, and a full run of
# N_inner ~ 5k steps yields ~200 points -- enough for a 25-point sliding window
# to mean something and for G2's "argmin in the last tenth" to have ~20 points
# of resolution in the region it tests.
#
# So "the last 25 steps" is read throughout as "the last 25 *measurements*",
# i.e. the last 25 x VAL_EVERY steps of training.  Both arms use the same
# cadence and the same train-val split, so the comparison is unaffected by the
# choice; what the choice sets is the resolution of the stability gate, and it
# is set here rather than after seeing whether a run passed.
VAL_EVERY = 25
VAL_TAIL_POINTS = 25       # "the last 25 steps" == the last 25 measurements
CODEBOOK_MOMENTUM = 0.0
CAYLEY_FIXED_POINT_ITERATIONS = 5
CAYLEY_REORTH_EVERY = 100

# The inner budget in passes over train-fit.  A constant, not a wall-clock
# target: two GPUs of different speed must run the same experiment.
INNER_EPOCHS = 100

# ||U^T U - I||_F, evaluated in fp32 like everything else in v11.  Measured on
# this stack for the 1024x1024 rotation:
#
#     trained U (v10 R64)           fp32 2.390e-05   fp64 2.130e-05
#     clean fp32-rounded orthogonal fp32 1.080e-05   fp64 1.149e-06
#
# and sqrt(2.130e-05^2 + 1.080e-05^2) = 2.388e-05, i.e. the fp32 reading is the
# true drift in quadrature with a 1.08e-05 measurement floor.  That floor is 9.3x
# below the tolerance, so measuring in fp32 costs nothing and keeps the whole
# pipeline in one dtype, as ratified.  1e-4 is still four orders of magnitude
# below the scale at which a non-orthogonal U would distort the rate accounting.
ORTH_TOL = 1e-4
ORTH_MEASURE_FLOOR = 1.08e-05


def cosine_lr(step, total, base):
    """``base * (1 + cos(pi * t / N)) / 2`` for ``t = 0 .. N`` (plan 7.2).

    Reaches exactly zero at ``t = N``.  v10's failure was not a bad learning
    rate but an absent schedule: the trajectory hit its minimum at step ~352 of
    3096 and then rose monotonically, so the endpoint was worse than the frozen
    start while a 20% improvement had existed and been passed through.  The
    schedule makes the endpoint the best point by construction, which is what
    lets G2 demand ``argmin >= 0.9 N`` instead of quietly rescuing a mid-run
    checkpoint.

    CayleySGD reads ``group["lr"]`` inside ``step()``, so annealing is applied by
    writing the param groups; its ``min(lr, 1/||skew||)`` trust region still
    applies on top and only ever makes the step smaller.
    """
    t = min(max(int(step), 0), int(total))
    return float(base) * 0.5 * (1.0 + np.cos(np.pi * t / float(total)))


# ------------------------------------------------------- rate selection ---
# Plan section 7.  Four steps, all fixed here before running:
#
#   1. the 5x5 grid at PROBE_STEPS is used ONLY to reject numerically invalid
#      cells and to produce a short-range ordering.  It does not select;
#   2. the shortlist is {short-range argmin} + {the same lr_U with lr_theta one
#      decade lower} + {the grid's geometric centre};
#   3. every shortlisted cell is run at the full N_inner under the cosine
#      schedule;
#   4. among the full-length cells that pass G2, the endpoint (train-val mean of
#      the last FULL_LENGTH_TAIL steps) selects.
#
# The centre cell is a *rule*, not a pick: index 2 of each grid, which is fixed
# by the grid's shape and cannot be chosen from an outcome.
LR_U_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
LR_THETA_GRID = (1e-3, 1e-2, 1e-1, 1e0, 1e1)
PROBE_STEPS = 200
PROBE_TAIL = 25
FULL_LENGTH_TAIL = 25
SHORTLIST_REFERENCE = (LR_U_GRID[len(LR_U_GRID) // 2],
                       LR_THETA_GRID[len(LR_THETA_GRID) // 2])
BATCH_CANDIDATES = (16, 32, 64, 128)
VRAM_CEILING_GIB = 12.0
BACKWARD_MULTIPLIER = 2.0


# ------------------------------------------------------------ outer search ---
S_OUTER_MAX = 8
OUTER_SWAPS_PER_EVENT = 1
EPS_ACCEPT_MULT = 10.0


def inner_steps(t_outer, s_outer_max=S_OUTER_MAX):
    return int(t_outer) * (int(s_outer_max) + 1)


# --------------------------------------------------------------- the arms ---
# v10's equal-compute A3 leaves the main flow by the user's ruling: A2's search
# FLOPs and cal usage are reported as a *method cost*, not compensated for with
# extra uniform training steps.
ARM_UNIFORM = "uniform-joint"          # A1
ARM_NONUNIFORM = "nonuniform-joint"    # A2
ARMS = (ARM_UNIFORM, ARM_NONUNIFORM)


# ------------------------------------------------------------------- gates ---
# G1  the endpoint really improved, relative to that arm's own step 0.
G1_IMPROVE_MULT = 100.0

# G2  the full run is stable: the smoothed train-val trajectory's argmin lies in
#     the last tenth, and the endpoint is within 1% of the running minimum.
#     Substituting a mid-run checkpoint for the endpoint is explicitly not
#     allowed -- that would be choosing the stopping point from the outcome.
G2_ARGMIN_FRACTION = 0.9
G2_END_TOLERANCE = 1.01
SMOOTH_WINDOW = 25

# G3  menu coverage and alignment with the current U, in two parts:
#     (1) mechanical -- every (group, mode) is invoked by the auxiliary branch
#         exactly once per three-step cycle;
#     (2) substantive -- before each outer event, with U_t held fixed, the fresh
#         codebooks are compared against the codebooks saved at the start of the
#         previous inner block on the fixed menu probe {a^(1), a^(02), ~a^(02)}
#         over train-val, and must not be worse.
#     Parameter drift, gradient norms and dead-codeword counts stay recorded but
#     are no longer evidence of alignment on their own -- that substitution is
#     what v10 got wrong.
G3_STALE_MULT = 10.0

# G4  measurable effect on dev, before anything is unsealed:
#     (1) A2 accepted at least one real swap and its allocation is not uniform;
#     (2) the paired one-sided upper confidence bound of D(A2) - D(A0) < 0;
#     (3) the same for D(A2) - D(A1).
#
# Zero-swap clause (plan section 8): if G1-G3 pass and A2 still accepts zero
# swaps, the only admissible statement is "the current trainer and one-bit
# search produced no non-uniform method arm at this anchor".  It may NOT be
# read as "U absorbed the non-uniform gain".
ZERO_SWAP_CLAUSE = (
    "Zero accepted swaps means only that this trainer and this one-bit search "
    "produced no non-uniform method arm at this anchor.  It is not evidence "
    "that U absorbed the non-uniform gain.")


# --------------------------------------------------------------- statistics ---
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 20261103        # v11's own seed; v9 used 20260731, v10 20261001
FWER = 0.05

# The confirmatory family is up to four hypotheses -- two comparisons at each of
# two anchors -- under one Holm correction (plan section 10).
COMPARISON_ORFC = "A2_minus_A0"     # H0: E[D(A2) - D(A0)] >= 0
COMPARISON_ALLOC = "A2_minus_A1"    # H0: E[D(A2) - D(A1)] >= 0
COMPARISONS = (COMPARISON_ORFC, COMPARISON_ALLOC)

SEAL_PATH = V11 / "SEAL.json"
UNSEAL_PATH = V11 / "unseal.json"
UNSEAL_TOKEN = V11 / "HOLDOUT_UNSEALED.json"
CODE_SNAPSHOT = V11 / "code_snapshot"


# --------------------------------------------------------------- paths ---
def anchor_dir(anchor):
    return V11 / anchor.name


def baseline_path(anchor):
    return anchor_dir(anchor) / "orfc_baseline.json"


def menu_dir(anchor):
    return anchor_dir(anchor) / "menu"


def bridge_dir(anchor):
    return anchor_dir(anchor) / "bridge"


def run_dir(anchor, arm):
    return anchor_dir(anchor) / arm


def profile_path(anchor):
    return anchor_dir(anchor) / "profile.json"


def load_profile(anchor):
    path = profile_path(anchor)
    if not path.exists():
        raise SystemExit(
            f"INVALID_EXPERIMENT: {path} does not exist.  N17 (profile.py) must "
            f"fix lr_U / lr_theta / batch / T_outer / N_inner at full length "
            f"before any formal arm runs, so that the hyper-parameters are a "
            f"construction rather than a choice made after seeing an outcome.")
    return json.loads(path.read_text())
