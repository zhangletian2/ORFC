"""
Importance-weighted bit allocation via dynamic programming.

Given per-group importance scores (e.g. from cls_ablation ΔL_cls) and a
total bit budget, allocate integer bits to each PQ subspace so that
higher-importance groups receive more bits.

Two modes:

1. ``monotonic=False`` (default, legacy):
   No ordering constraint — any group can get any bit count independently.

2. ``monotonic=True`` (VAQ-style gaps constraint):
   Groups are sorted by importance descending.  A *gaps* constraint limits
   how fast bits can decrease between consecutive (importance-sorted) groups:

       bits[g-1] - bits[g]  <=  gap[g-1]

   where  gap = floor_pow2(importance[g-1] / importance[g]).
   After DP, the allocation is un-sorted back to original group ordering.

Hard constraints (both modes):

    min_bits <= b_g <= max_bits   for every g
    sum(b_g) == bit_budget

Objective maximised:

    sum_g  score(importance[g], b_g)

Score functions:
  * ``linear``:  importance * bits
  * ``rd``:      importance * (1 - 4^{-bits / d})   (concave, diminishing returns)
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


def _next_pow2_floor(x: float) -> int:
    """Largest power-of-2 <= |x|.  Used for gap bounds (same as VAQ)."""
    if x <= 0 or not np.isfinite(x):
        return 0
    return int(2 ** math.floor(math.log2(max(abs(float(x)), 1e-30))))


def _bit_score(imp: float, bits: int, d: int, objective: str) -> float:
    if objective == "linear":
        return imp * float(bits)
    if objective == "rd":
        return imp * (1.0 - 4.0 ** (-float(bits) / max(d, 1)))
    raise ValueError(f"unknown bit-allocation objective: {objective}")


def _dp_alloc(
    imp: np.ndarray,
    bit_budget: int,
    min_bits: int,
    max_bits: int,
    d: int,
    objective: str,
    gaps: Optional[List[int]] = None,
) -> List[int]:
    """Core DP allocator.  If *gaps* is provided, enforce VAQ-style
    monotonic-descent constraint between consecutive groups."""
    G = len(imp)
    if bit_budget < G * min_bits or bit_budget > G * max_bits:
        raise ValueError(
            f"infeasible bit budget {bit_budget}; "
            f"bounds allow [{G * min_bits}, {G * max_bits}]"
        )

    allowed = list(range(min_bits, max_bits + 1))

    DPEntry = Tuple[float, Optional[int], Optional[int]]
    dp: Dict[Tuple[int, int], DPEntry] = {}
    for b in allowed:
        dp[(b, b)] = (_bit_score(imp[0], b, d, objective), None, None)
    parents: List[Dict[Tuple[int, int], DPEntry]] = [dp]

    for g in range(1, G):
        ndp: Dict[Tuple[int, int], DPEntry] = {}
        for (prev_sum, prev_b), (score, _, _) in dp.items():
            for b in allowed:
                if gaps is not None and prev_b - b > gaps[g - 1]:
                    continue
                new_sum = prev_sum + b
                remaining = G - g - 1
                if new_sum + remaining * min_bits > bit_budget:
                    continue
                if new_sum + remaining * max_bits < bit_budget:
                    continue
                key = (new_sum, b)
                new_score = score + _bit_score(imp[g], b, d, objective)
                if key not in ndp or new_score > ndp[key][0]:
                    ndp[key] = (new_score, prev_sum, prev_b)
        if not ndp:
            raise RuntimeError(
                f"bit allocation became infeasible at group {g}"
            )
        dp = ndp
        parents.append(dp)

    candidates = [(key, val) for key, val in dp.items() if key[0] == bit_budget]
    if not candidates:
        raise RuntimeError(
            f"no exact allocation found for bit budget {bit_budget}"
        )
    (sum_bits, curr_b), _ = max(candidates, key=lambda item: item[1][0])

    bits = [0] * G
    bits[-1] = curr_b
    for g in range(G - 1, 0, -1):
        _, prev_sum, prev_b = parents[g][(sum_bits, bits[g])]
        bits[g - 1] = prev_b
        sum_bits = prev_sum
    return bits


def allocate_bits_importance(
    importance: np.ndarray,
    bit_budget: int,
    min_bits: int = 1,
    max_bits: int = 10,
    d: int = 32,
    objective: str = "rd",
    monotonic: bool = False,
) -> List[int]:
    """Allocate integer bits to PQ groups proportional to importance.

    Parameters
    ----------
    importance : array of shape (G,)
        Non-negative importance score per group (need not sum to 1).
    bit_budget : int
        Exact total bits to distribute, i.e. ``sum(result) == bit_budget``.
    min_bits, max_bits : int
        Per-group bounds.  ``K_g = 2 ** bits[g]``.
    d : int
        Sub-vector dimensionality (used only by the ``rd`` objective).
    objective : str
        ``"rd"`` (default, concave) or ``"linear"``.
    monotonic : bool
        If True, sort groups by importance descending and apply VAQ-style
        gaps constraint so that bits decrease smoothly with importance.

    Returns
    -------
    bits : list of int, length G
    """
    G = len(importance)
    imp = np.asarray(importance, dtype=np.float64).clip(min=0)
    if imp.sum() < 1e-30:
        imp = np.ones(G, dtype=np.float64)

    if monotonic:
        order = np.argsort(-imp)          # descending importance
        imp_sorted = imp[order]
        imp_normed = imp_sorted / imp_sorted.sum()

        gaps = []
        for g in range(G - 1):
            ratio = imp_sorted[g] / max(imp_sorted[g + 1], 1e-30)
            gaps.append(_next_pow2_floor(ratio))

        bits_sorted = _dp_alloc(
            imp_normed, bit_budget, min_bits, max_bits, d, objective,
            gaps=gaps,
        )

        bits = [0] * G
        for i, orig_idx in enumerate(order):
            bits[orig_idx] = bits_sorted[i]
        return bits

    imp = imp / imp.sum()
    return _dp_alloc(imp, bit_budget, min_bits, max_bits, d, objective)
