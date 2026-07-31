"""v10's statistics: v9's estimators, v10's seed.

The paired bootstrap and the Holm step-down are imported from ``phase1.unseal``
rather than reimplemented -- they are the code that produced v9's published
numbers and there is no reason for v10 to own a second copy that could drift
from it.  What v10 changes is only the *constants*: a fresh resampling seed
(20261001, so this round's intervals are not v9's), and the same FWER over the
same two-anchor family.

``finish`` is the one piece that is not in v9's module boundary: v9 inlines the
"draw the UCB from the same bootstrap distribution at the Holm-assigned alpha"
step inside its unseal routine, and v10 needs it in two places (Stage B's dev
acceptance and Stage D's holdout unsealing).  Keeping ``ucb < 0`` and
``p < holm_alpha`` two readings of one distribution -- rather than two
independently computed quantities that could disagree -- is the point.
"""

import numpy as np

from . import config as C
from ..unseal import holm as _holm
from ..unseal import paired_bootstrap as _paired_bootstrap


def paired_bootstrap(differences, resamples=C.BOOTSTRAP_RESAMPLES,
                     seed=C.BOOTSTRAP_SEED):
    """Resample images, keeping each image's pair intact.

    holdout-500 is one image per class over 500 distinct classes (W2), so the
    stratum unit *is* the image and this per-image resampling already is the
    stratified bootstrap the protocol asks for.  W2 is re-checked before any
    unsealing precisely so that this sentence stays true.
    """
    return _paired_bootstrap(differences, resamples=resamples, seed=seed)


def one_sided(differences):
    """Bootstrap the mean paired difference; return the draws and a summary."""
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
