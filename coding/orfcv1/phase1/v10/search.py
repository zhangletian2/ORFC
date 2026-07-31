"""The greedy one-bit swap search -- one implementation, two callers.

Stage B runs this on the frozen ``(U_0, Theta_0)`` to answer the bridging
question ("can the optimiser find *any* same-rate allocation better than the
uniform point?"), and A2's outer loop runs the *same function* between inner
training windows.  Writing it twice would let the two differ, and then a Stage B
pass would not licence anything about A2; so it is written once.

The neighbourhood is every legal one-bit transfer from the current point:

    down group i   requires  m_i > 0        (there is a lower mode to fall to)
    up   group j   requires  m_j < M - 1    (there is a higher mode to rise to)
    i != j

Because an anchor's ``mode_bits`` are three consecutive integers, one mode step
is exactly one bit, so every candidate has *exactly* the anchor's nominal rate
-- not approximately, not on average.  :func:`legal_swaps` recomputes that for
every row it returns rather than arguing it here.

At the uniform point the neighbourhood is 32 x 31 = 992 and must coincide, row
for row and in order, with v9's ``engine.swap_candidates``.  That is not left as
a comment either: :func:`legal_swaps` asserts it whenever the base *is* uniform,
which is what makes Stage B's W9 replay a comparison of like with like.

The acceptance rule is deliberately conservative.  A swap is taken only if it
improves the cal mean by more than ``EPS_ACCEPT_MULT`` (10) times the fp32
replay floor times the current distortion level -- roughly 1e-5 relative.  A
greedy search over ~1000 correlated candidates will always turn up a *smallest*
one, and without a floor the search would happily spend all eight of its steps
walking around inside the noise and hand back a point that is different from
uniform without being better than it.
"""

import time

import numpy as np

from . import config as C
from ..engine import nominal_rate, swap_candidates, uniform_allocation


def legal_swaps(allocation, anchor, groups=C.GROUPS):
    """All one-bit transfers reachable from ``allocation``.

    Returns ``(candidates [A, G], pairs [A, 2])`` with ``pairs[k] =
    (down_group, up_group)``, enumerated in lexicographic order -- which is also
    the frozen tie-break order, so ``np.argmin`` over the candidate means already
    implements ``TIE_BREAK`` without a second sort.
    """
    base = np.asarray(allocation, dtype=np.int64)
    if base.shape != (groups,):
        raise SystemExit(f"INVALID_EXPERIMENT: allocation has shape "
                         f"{base.shape}, expected ({groups},)")
    top = len(anchor.mode_bits) - 1
    candidates, pairs = [], []
    for down in range(groups):
        if base[down] <= 0:
            continue
        for up in range(groups):
            if up == down or base[up] >= top:
                continue
            candidate = base.copy()
            candidate[down] -= 1
            candidate[up] += 1
            candidates.append(candidate)
            pairs.append((down, up))
    if not candidates:
        return (np.zeros((0, groups), dtype=np.int64),
                np.zeros((0, 2), dtype=np.int64))
    candidates = np.stack(candidates)
    pairs = np.asarray(pairs, dtype=np.int64)

    # Same rate, exactly, for every row -- checked, not asserted in prose.
    rates = np.array([nominal_rate(row, anchor) for row in candidates])
    if not np.all(rates == anchor.rate):
        bad = int(np.argmax(rates != anchor.rate))
        raise SystemExit(f"INVALID_EXPERIMENT: candidate {bad} has nominal rate "
                         f"{rates[bad]}, not {anchor.rate}")

    # At the uniform point this must *be* v9's enumeration, or Stage B's replay
    # of v9's cal sweep would be comparing two different candidate sets.
    if np.array_equal(base, uniform_allocation(anchor, groups)):
        v9_candidates, v9_pairs = swap_candidates(anchor, groups)
        if not (np.array_equal(candidates, v9_candidates)
                and np.array_equal(pairs, v9_pairs)):
            raise SystemExit(
                "INVALID_EXPERIMENT: the v10 neighbourhood of the uniform point "
                "differs from v9's engine.swap_candidates; the W9 replay would "
                "be meaningless")
    return candidates, pairs


def greedy_swap_search(evaluate, anchor, start=None, s_max=C.S_OUTER_MAX,
                       eps_mult=C.EPS_ACCEPT_MULT, rel_tol=C.REPLAY_REL_TOL,
                       log=print, on_first_matrix=None):
    """Greedy one-bit descent from ``start`` (default: the uniform point).

    ``evaluate(allocations [A, G]) -> [A, N]`` is the caller's hard per-image
    distortion; Stage B passes a cal-500 evaluator over the frozen codec, A2
    passes one over the codec as it currently stands.

    Every round re-evaluates the *current* point as row 0 of the same call as
    its candidates.  That costs one extra forward per round and buys the thing
    the accept rule depends on: in A2 the codec has moved since the last round,
    so a carried-over baseline would be a distortion of a different network, and
    the search would compare a new candidate against a stale reference.

    ``on_first_matrix(matrix)`` is called with round 1's ``[A+1, N]`` result the
    moment it exists, before anything is decided from it.  Stage B hangs its v9
    replay invariant there so a mismatch aborts before the remaining rounds are
    paid for, rather than after.

    Returns a dict with the final allocation and a per-round trace.  It reports
    what it did, and no gate lives here -- the callers decide.
    """
    started = time.time()
    current = (uniform_allocation(anchor) if start is None
               else np.asarray(start, dtype=np.int64).copy())
    if nominal_rate(current, anchor) != anchor.rate:
        raise SystemExit(f"INVALID_EXPERIMENT: the starting allocation has "
                         f"nominal rate {nominal_rate(current, anchor)}, not "
                         f"{anchor.rate}")

    trace = []
    accepted = 0
    for step in range(1, s_max + 1):
        candidates, pairs = legal_swaps(current, anchor)
        if len(candidates) == 0:
            log(f"[{anchor.name}] step {step}: no legal swap remains")
            break

        matrix = evaluate(np.vstack([current[None], candidates]))
        matrix = np.asarray(matrix, dtype=np.float64)
        if step == 1 and on_first_matrix is not None:
            on_first_matrix(matrix)

        means = matrix.mean(1)
        base_mean = float(means[0])
        candidate_means = means[1:]
        best = int(np.argmin(candidate_means))          # first minimiser == TIE_BREAK
        gain = base_mean - float(candidate_means[best])
        threshold = eps_mult * rel_tol * base_mean
        take = bool(gain > threshold)

        entry = {
            "step": step,
            "candidates": int(len(candidates)),
            "base_allocation": current.tolist(),
            "base_mean": base_mean,
            "best_down_group": int(pairs[best, 0]),
            "best_up_group": int(pairs[best, 1]),
            "best_mean": float(candidate_means[best]),
            "gain": gain,
            "relative_gain": gain / base_mean,
            "accept_threshold": threshold,
            "gain_over_threshold": gain / threshold if threshold else None,
            "gain_over_noise_floor": gain / (rel_tol * base_mean),
            "ties_at_minimum": int(np.sum(
                candidate_means == candidate_means[best])),
            "candidates_below_base": int(np.sum(candidate_means < base_mean)),
            "accepted": take,
        }
        trace.append(entry)
        log(f"[{anchor.name}] step {step}: down g{entry['best_down_group']} "
            f"up g{entry['best_up_group']}  gain {gain:.4f} "
            f"({100 * entry['relative_gain']:.4f}%) = "
            f"{entry['gain_over_noise_floor']:.1f}x floor  "
            f"{'ACCEPT' if take else 'stop (below threshold)'}")
        if not take:
            break

        current = candidates[best].copy()
        accepted += 1
        if nominal_rate(current, anchor) != anchor.rate:      # W6, every step
            raise SystemExit(f"INVALID_EXPERIMENT: after step {step} the "
                             f"nominal rate is "
                             f"{nominal_rate(current, anchor)}, not "
                             f"{anchor.rate}")

    result = {
        "anchor": anchor.name,
        "rate": anchor.rate,
        "start": (uniform_allocation(anchor).tolist() if start is None
                  else np.asarray(start).tolist()),
        "final_allocation": current.tolist(),
        "nominal_rate": nominal_rate(current, anchor),
        "accepted_swaps": accepted,
        "s_outer_max": int(s_max),
        "eps_accept_mult": float(eps_mult),
        "replay_rel_tol": float(rel_tol),
        "tie_break": C.TIE_BREAK,
        "trace": trace,
        "cal_mean_start": trace[0]["base_mean"] if trace else None,
        "cal_mean_final": (trace[-1]["best_mean"] if trace and trace[-1]["accepted"]
                           else (trace[-1]["base_mean"] if trace else None)),
        "seconds": time.time() - started,
    }
    if trace:
        result["cal_total_gain"] = (result["cal_mean_start"]
                                    - result["cal_mean_final"])
        result["cal_total_relative_gain"] = (result["cal_total_gain"]
                                             / result["cal_mean_start"])
    return result


def changed_groups(start, final):
    """Which groups moved, and by how many modes.  For the record, not a gate."""
    start = np.asarray(start, dtype=np.int64)
    final = np.asarray(final, dtype=np.int64)
    delta = final - start
    moved = np.nonzero(delta)[0]
    return {"groups": moved.tolist(),
            "delta": delta[moved].tolist(),
            "down": np.nonzero(delta < 0)[0].tolist(),
            "up": np.nonzero(delta > 0)[0].tolist()}
