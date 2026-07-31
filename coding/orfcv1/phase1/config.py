"""Frozen constants for plan v9 (`阶段一计划_v9_低码率同码率非均匀检验`).

Written before running.  Editing anything here after the first anchor codec is
built means a new plan version, not a patch: v8 died at its own M3 gate and the
whole point of v9 is that no constant is chosen from an observed outcome.

What v9 dropped relative to v8, and why the corresponding constants are gone:

* no `U`/codebook training, so no learning rate, no temperature, no anneal, no
  soft relaxation.  Every number in this round comes from a hard nearest
  neighbour forward (plan v9 section 3.4);
* no dev-500 in any judgement (section 5), so no checkpoint selection and no
  hyper-parameter sweep.  dev stays defined only because the split manifest is
  reused bit-identically from v8;
* no menu audit M1--M5.  Per-group task-tail monotonicity is explicitly demoted
  to a recordable, not a criterion (section 3), which is exactly the gate that
  stopped v8 on an underpowered comparison;
* no proxies.  `Phi_J`, `Phi_sep`, `Omega` and Regret decide nothing here.

The two anchors are separate frozen objects.  Each gets its own OPQ warm-up at
its own uniform codebook size, so `U_0^(64)` and `U_0^(96)` are different
matrices; requirement 1 of section 3 is that a *single* anchor shares one `U_0`
across all of its candidates, which is what `build.py` asserts.
"""

from pathlib import Path
from typing import NamedTuple

# ----------------------------------------------------------------- paths ---
REPO = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO / "artifacts" / "dinov2_vitl14"
CACHE = ARTIFACTS / "cache"
SPLIT = ARTIFACTS / "split"
PHASE1 = REPO / "results" / "dinov2_vitl14" / "phase1"
V9 = PHASE1 / "v9"
FEATURE_POOL = Path("/data4/workspace/zlt/featcodec/features")

BLOCK = "blk20"
LAYER = 20
CACHE_TAG = "ss20260730"
NORM_MODE = "per_image"

# ------------------------------------------------------------- geometry ---
GROUPS = 32
DIM = 32              # d = D / G = 1024 / 32


class Anchor(NamedTuple):
    """One low-rate anchor: a uniform point and its two neighbouring modes.

    ``mode_bits`` is ordered low to high and holds exactly three entries -- the
    down-mode, the uniform mode and the up-mode.  A candidate moves one group
    down and one group up, so no other mode is reachable and none is built.
    """

    name: str
    rate: int                  # nominal bits per token, verified per candidate
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
        return V9 / self.name


# R64: uniform K4, one group to K2 and another to K8.
# R96: uniform K8, one group to K4 and another to K16.
ANCHORS = (
    Anchor(name="R64", rate=64, uniform_bits=2, mode_bits=(1, 2, 3)),
    Anchor(name="R96", rate=96, uniform_bits=3, mode_bits=(2, 3, 4)),
)
ANCHOR_BY_NAME = {a.name: a for a in ANCHORS}

# 32 x 31 ordered (down, up) group pairs, plus the uniform point itself.
N_CANDIDATES = GROUPS * (GROUPS - 1)
N_EVALUATED = N_CANDIDATES + 1

# --------------------------------------------------------- construction ---
# Identical procedure for both anchors and, inside an anchor, for all three
# modes.  The only per-anchor difference is the warm-up codebook size, which is
# tied to that anchor's uniform mode: the rotation is warmed up for the point
# the experiment perturbs around.
#
# There is deliberately NO task-tail pretraining of the codebooks.  v8 ran a
# balanced six-mode soft pretraining to equalise codebook maturity across the
# menu; here every mode is produced by the same k-means procedure with the same
# iteration count from the same rotated features, so the maturity confound is
# already absent, while soft pretraining would reintroduce a temperature, a
# soft-hard proxy gap, and a dev-500 stopping decision -- all three of which
# v9 section 5 and section 3 put outside this round.
OPQ_IMAGES = 4000            # all of train-core
OPQ_ITERS = 30               # alternations
OPQ_KMEANS_ITERS = 10        # Lloyd iterations per alternation
OPQ_SEED = 0
OPQ_ORTH_TOL = 1e-5          # ||U^T U - I||_F, evaluated in float64

CODEBOOK_KMEANS_ITERS = 100
CODEBOOK_SEED = 0

# ------------------------------------------------------------- data split ---
# Reused bit-identically from v8: the manifest under artifacts/.../split/phase1
# already exists and freeze_splits() is idempotent, so these values must not
# move or the four sets would be redrawn.
SPLIT_SEED = 20260731
N_TRAIN_CORE = 4000
N_CAL = 500
N_DEV = 500                  # defined but unused by v9 (plan section 5)
N_MEASURE = 3000

# --------------------------------------------------------------- statistics ---
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 20260731
FWER = 0.05                  # Holm across the two anchors, one family

# Tie-break inside the cal argmin, fixed before running: the smallest
# (down_group, up_group) pair in lexicographic order wins.  Ties are not
# expected at fp32 on 500 images, but "not expected" is not a rule.
TIE_BREAK = "lexicographic_on_(down_group, up_group)"

# --------------------------------------------------------------- numerics ---
# One allocation per tail call, and every tail call the same shape.  Measured on
# GPU 2 before any anchor was built (40 allocations x 100 cal images):
#
#   image_batch  pair_budget  alloc/call   img/s   peak VRAM
#            25         1024          40    1290    13.17 GiB
#            25           25           1    1614     0.74 GiB
#           100          100           1    1688     1.92 GiB
#           250          250           1    1690     1.92 GiB
#
# Folding many allocations into one call was supposed to fill the GPU; it does
# the opposite here, because the bank gather is already amortised over the image
# batch and the wide tail call only adds activation pressure.  One-per-call is
# faster, 7x cheaper in VRAM, and -- the reason it is frozen rather than merely
# preferred -- it gives all 993 allocations a bit-for-bit identical kernel path,
# so no candidate can win the cal argmin by sitting at a luckier batch offset.
#
# 100 divides both 500 (cal) and 3000 (measure) exactly, so no call is ever a
# short final chunk of a different shape.  `engine.assert_uniform_shape` enforces
# that rather than trusting it.
EVAL_IMAGE_BATCH = 100
EVAL_PAIR_BUDGET = 100                 # == image_batch  =>  one allocation/call
REPLAY_REPEATS = 8                     # identical calls compared to each other
REPLAY_IMAGES = EVAL_IMAGE_BATCH

# TF32 stays disabled.  Measured under v8 (noise_floor.json): the TF32-vs-fp32
# shift of a paired gain was 122.46 against an fp32 run-to-run sigma of 0.0038,
# i.e. 32,000x the noise floor, and it inverted 6 of 32 allocation ranks.  v9's
# entire cal step is an exact empirical argmin over 992 points, so a systematic
# shift of that size is disqualifying and the 1.54x throughput does not pay.
ALLOW_TF32 = False

# Hard replay: same codec, same images, same allocation, same call shape, in a
# fresh process, must return the same fp32 distortion.  "The same" cannot mean
# bit-for-bit here, and pretending otherwise would be a fake invariant: measured
# on this stack, two identical calls disagree by at most 0.125 on a per-image
# distortion of ~797,257, i.e. exactly one fp32 ULP (1.19e-7 relative).  The
# tail's reduction order is not reproducible to the last bit and no protocol
# choice can make it so.
#
# So the gate is 8 ULP of relative deviation.  It is three to four orders of
# magnitude below any effect this round can report, and any real defect -- wrong
# codec, wrong allocation, wrong split, a stale cache -- moves the distortion by
# a relative 1e-3 or more, so nothing that matters can hide under it.  The
# observed deviation is recorded in verify.json, and cal_sweep/unseal report
# their gaps as multiples of it rather than leaving the reader to guess.
REPLAY_REL_TOL = 8 * 2 ** -23        # 9.54e-7
