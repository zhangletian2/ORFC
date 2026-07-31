"""v11's statistics: v9's estimators, v11's constants.

The estimators themselves -- the paired bootstrap and the Holm step-down -- come
from ``phase1.unseal``, which is the code that produced v9's published numbers.
v11 owns no second copy of either, for the same reason it owns no second copy of
the searcher.

What v11 does *not* inherit is v10's constants.  ``phase1.v10.stats`` binds
``BOOTSTRAP_SEED = 20261001`` as a default argument at definition time, so
re-exporting its functions would have resampled v11 under v10's seed while
``config.py`` claimed 20261103.  The pre-registered seed would have been a
comment.  So this module re-exports nothing callable from v10: it wraps the base
estimators and passes v11's constants explicitly.

The wrapper is thin by design.  ``one_sided`` and ``finish`` are v10's bodies
verbatim -- same p-value, same "UCB and p are two readings of one distribution"
discipline -- with the constants rebound.  ``assert_estimators_are_shared``
fails the import if v10 ever stops routing through the same base functions,
which would mean the two rounds are no longer running the same estimator.

v11's family is larger than v10's -- up to four hypotheses, two per anchor
(``A2 - A0`` and ``A2 - A1``), against v10's two -- but that is a property of
what the caller puts in the family, not of the correction, so ``holm`` is used
unchanged.
"""

import numpy as np

from . import config as C
from ..unseal import holm as _holm
from ..unseal import paired_bootstrap as _paired_bootstrap


def paired_bootstrap(differences, resamples=C.BOOTSTRAP_RESAMPLES,
                     seed=C.BOOTSTRAP_SEED):
    """Resample images, keeping each image's pair intact.

    Each split is one image per class over 500 distinct classes, so the stratum
    unit *is* the image and this per-image resampling already is the stratified
    bootstrap the protocol asks for.
    """
    return _paired_bootstrap(differences, resamples=resamples, seed=seed)


def one_sided(differences):
    """Bootstrap the mean paired difference; return the draws and a summary.

    An all-zero difference vector -- what a zero-swap search produces, where the
    final allocation *is* the uniform one -- gives every draw exactly 0.0 and so
    ``p_value = 1.0``.  That is the right answer (nothing changed, nothing is
    detectable) rather than a division by a zero standard error, so no special
    case is needed here.
    """
    values = np.asarray(differences, dtype=np.float64)
    means = paired_bootstrap(values)
    return means, {
        "mean": float(values.mean()),
        "p_value": float(np.mean(means >= 0.0)),
        "images": int(values.size),
        "images_improved": int(np.sum(values < 0)),
        "bootstrap_resamples": int(C.BOOTSTRAP_RESAMPLES),
        "bootstrap_seed": int(C.BOOTSTRAP_SEED),
    }


def holm(entries, fwer=C.FWER):
    return _holm(entries, fwer)


def finish(entries, bootstraps, fwer=C.FWER):
    """Holm over the family, then the UCB out of each anchor's own draws."""
    ordered = holm(entries, fwer)
    for entry in ordered:
        means = bootstraps[entry["anchor"]]
        entry["ucb"] = float(np.quantile(means, 1.0 - entry["holm_alpha"]))
        entry["passes"] = bool(entry["rejects_at_holm_alpha"]
                               and entry["ucb"] < 0)
    return ordered


def assert_estimators_are_shared():
    """v10 and v11 must be calling the same estimator, differing only in seed."""
    from ..v10 import stats as _v10_stats
    from .. import unseal as _base
    drift = {}
    if _v10_stats._paired_bootstrap is not _base.paired_bootstrap:
        drift["paired_bootstrap"] = "v10 no longer wraps phase1.unseal's"
    if _v10_stats._holm is not _base.holm:
        drift["holm"] = "v10 no longer wraps phase1.unseal's"
    if drift:
        raise SystemExit(
            f"INVALID_EXPERIMENT: v11 and v10 were meant to share one bootstrap "
            f"implementation and differ only in the pre-registered seed, but "
            f"{drift}.  Two estimators that could disagree is exactly what the "
            f"shared base was for.")


assert_estimators_are_shared()
