"""Sparse Kendall-τ and ranking utilities for V33 probes."""

from __future__ import annotations

import numpy as np


def sparse_kendall_tau(left, right, tie_threshold=0.0):
    """Kendall-τ over pairs that are decisive on *both* rankings.

    A pair ``(i, j)`` is a *tie* on a ranking when
    ``|score[i] - score[j]| <= tie_threshold`` (same-rank threshold).
    Ties on either side are dropped (sparse / top-k style Kendall), so the
    denominator is the number of jointly decisive pairs rather than
    ``n choose 2``.

    Returns ``(tau, n_concordant, n_discordant, n_decisive, n_dropped)``.
    ``tau`` is ``None`` when no decisive pairs remain.
    """
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("score vectors must share shape")
    threshold = float(tie_threshold)
    concordant = discordant = dropped = 0
    n = len(left)
    for i in range(n):
        for j in range(i + 1, n):
            a = left[i] - left[j]
            b = right[i] - right[j]
            if abs(a) <= threshold or abs(b) <= threshold:
                dropped += 1
                continue
            if (a > 0 and b > 0) or (a < 0 and b < 0):
                concordant += 1
            else:
                discordant += 1
    decisive = concordant + discordant
    tau = None if decisive == 0 else (concordant - discordant) / decisive
    return tau, concordant, discordant, decisive, dropped


def ranks_with_ties(scores, tie_threshold=0.0):
    """Average ranks with ``|Δ| <= tie_threshold`` collapsed to ties."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    threshold = float(tie_threshold)
    start = 0
    while start < len(order):
        stop = start + 1
        while (stop < len(order)
               and abs(scores[order[stop]] - scores[order[start]])
               <= threshold):
            stop += 1
        # 1-based mid-rank of the tied block.
        mid = 0.5 * (start + stop - 1) + 1.0
        ranks[order[start:stop]] = mid
        start = stop
    return ranks


def spearman_with_ties(left, right, tie_threshold=0.0):
    if len(left) < 2:
        return 1.0
    rl = ranks_with_ties(left, tie_threshold)
    rr = ranks_with_ties(right, tie_threshold)
    if rl.std() == 0 or rr.std() == 0:
        return None
    return float(np.corrcoef(rl, rr)[0, 1])
