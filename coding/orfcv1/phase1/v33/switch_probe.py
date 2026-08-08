"""Switch-point probe: subspace drift + fixed-pool argmin move + sparse Kendall-τ.

Record-only diagnostics (they never open the phase-2 gate by themselves).

Signals (logged every N steps / every checkpoint pair)
------------------------------------------------------
1. **Group subspace drift** (pure linear algebra, no forward)::

       o_g(t, t') = (1/d) || U_t[:, S_g]^T U_{t'}[:, S_g] ||_F^2

   Also records per-group energy ``σ_g`` (mean squared coordinates after ``U0``)
   and the induced cross-group ranking, when a feature cache is available.

2. **Fixed-candidate argmin movement** (default): score a fixed exact-budget
   pool once per checkpoint; ``search_hamming`` is the Hamming distance between
   successive pool argmins.  The expensive simplified local search is opt-in
   via ``--probe-search-move``.

3. **Sparse Kendall-τ** on the same fixed candidate pool between adjacent
   checkpoints, with the same-rank threshold from
   :mod:`phase1.v33.noise_floor`.

Switch suggestion
-----------------
Both (1) and (2) must look stable *and* the Kendall curve must have been
observed past its peak.  Stability is **not** evidence of correctness —
correctness is deferred to delivery-time top-k retrain (plan warning).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .. import engine, tail as tail_mod
from ..v12 import config as C
from ..v21.config import SPECS, activate
from . import checkpoint as ckpt
from . import noise_floor
from . import ranking
from . import search
from . import valset


# ---------------------------------------------------------- linear algebra ---

def group_slices(groups, dim):
    """Index slices ``S_g`` for contiguous groups of width ``dim``."""
    groups, dim = int(groups), int(dim)
    return [slice(g * dim, (g + 1) * dim) for g in range(groups)]


@torch.no_grad()
def group_subspace_overlap(U_t, U_tp, groups, dim):
    """``o_g = (1/d) ||U_t[:,S_g]^T U_tp[:,S_g]||_F^2`` for each group.

    For orthonormal ``U``, ``o_g ∈ [0, 1]`` with ``o_g = 1`` iff the two
    group subspaces coincide.
    """
    U_t = torch.as_tensor(U_t)
    U_tp = torch.as_tensor(U_tp)
    if U_t.shape != U_tp.shape:
        raise ValueError("U_t and U_tp must share shape")
    d = int(dim)
    overlaps = []
    for sl in group_slices(groups, dim):
        gram = U_t[:, sl].T @ U_tp[:, sl]
        overlaps.append(float(gram.square().sum() / d))
    return np.asarray(overlaps, dtype=np.float64)


@torch.no_grad()
def group_energies(U, y, groups, dim):
    """Per-group mean energy ``σ_g^2 = mean ||z_g||^2`` after ``z = y @ U``.

    ``y`` is ``[B, T, D]`` (or ``[N, D]``).  Returns energies ``[G]`` and the
    argsort ranking (descending energy).
    """
    U = torch.as_tensor(U)
    y = torch.as_tensor(y)
    if y.ndim == 3:
        flat = y.reshape(-1, y.shape[-1])
    elif y.ndim == 2:
        flat = y
    else:
        raise ValueError("y must be [B,T,D] or [N,D]")
    z = flat @ U.to(dtype=flat.dtype, device=flat.device)
    energies = []
    for sl in group_slices(groups, dim):
        block = z[:, sl]
        energies.append(float(block.square().mean() * block.shape[-1]))
    energies = np.asarray(energies, dtype=np.float64)
    order = np.argsort(-energies, kind="stable")
    return energies, order


def hamming(a, b):
    a = np.asarray(a, dtype=np.int64).reshape(-1)
    b = np.asarray(b, dtype=np.int64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("allocations must share shape")
    return int(np.sum(a != b))


# --------------------------------------------------------------- recording ---

def probe_drift(U_t, U_tp, groups, dim, y=None):
    overlaps = group_subspace_overlap(U_t, U_tp, groups, dim)
    record = {
        "overlaps": overlaps.tolist(),
        "overlap_min": float(overlaps.min()),
        "overlap_mean": float(overlaps.mean()),
        "drift_max": float(1.0 - overlaps.min()),
        "drift_mean": float(1.0 - overlaps.mean()),
    }
    if y is not None:
        e_t, rank_t = group_energies(U_t, y, groups, dim)
        e_tp, rank_tp = group_energies(U_tp, y, groups, dim)
        record["energy_t"] = e_t.tolist()
        record["energy_tp"] = e_tp.tolist()
        record["energy_rank_t"] = rank_t.tolist()
        record["energy_rank_tp"] = rank_tp.tolist()
        record["energy_rank_hamming"] = hamming(rank_t, rank_tp)
        # Spearman of energy ranks.
        record["energy_rank_spearman"] = ranking.spearman_with_ties(e_t, e_tp)
    return record


def probe_search_move(score_fn, groups, bits, rate, seed=0):
    """Simplified fixed-start search; returns allocation + evals used."""
    result = search.multi_start_search(
        score_fn, groups, bits, rate,
        simplified=True, seed=seed)
    best = result.get("best")
    return {
        "allocation": None if best is None else best["allocation"],
        "score": None if best is None else best["score"],
        "evals_used": result["evals_used"],
        "n_starts": result["n_starts"],
        "simplified": True,
    }


def probe_kendall(scores_t, scores_tp, tie_threshold):
    tau, c, d, decisive, dropped = ranking.sparse_kendall_tau(
        scores_t, scores_tp, tie_threshold=tie_threshold)
    return {
        "sparse_kendall_tau": tau,
        "concordant": int(c),
        "discordant": int(d),
        "decisive": int(decisive),
        "dropped": int(dropped),
        "tie_threshold": float(tie_threshold),
    }


def score_candidates(score_fn, allocations):
    allocations = np.asarray(allocations, dtype=np.int64)
    if allocations.ndim == 1:
        allocations = allocations[None]
    return np.asarray([float(score_fn(row)) for row in allocations],
                      dtype=np.float64)


# ----------------------------------------------------------- switch policy ---

def suggest_switch(curve, drift_stable_max=1e-3, move_stable_max=0,
                   tau_stable_min=0.9, window=3):
    """Suggest a switch index from a recorded curve.

    Requires:
    * a peak in adjacent sparse Kendall-τ already observed (non-monotonic
      warning: do not stop at first saturation);
    * the last ``window`` points after that peak have
      ``drift_max <= drift_stable_max`` AND ``search_hamming <= move_stable_max``.

    Returns a dict.  ``suggest=True`` never claims correctness — see module
    docstring.
    """
    if not curve:
        return {"suggest": False, "reason": "empty_curve",
                "note": _STABILITY_NOTE}

    taus = [p.get("kendall", {}).get("sparse_kendall_tau") for p in curve]
    numeric = [(i, t) for i, t in enumerate(taus) if t is not None]
    if len(numeric) < window + 1:
        return {"suggest": False, "reason": "curve_too_short",
                "note": _STABILITY_NOTE}

    peak_idx = max(numeric, key=lambda item: item[1])[0]
    # Must have at least one recorded point *after* the peak.
    if peak_idx >= len(curve) - 1:
        return {
            "suggest": False,
            "reason": "have_not_passed_peak",
            "peak_index": int(peak_idx),
            "peak_tau": float(taus[peak_idx]) if taus[peak_idx] is not None else None,
            "note": _STABILITY_NOTE,
        }

    tail = curve[max(peak_idx + 1, len(curve) - window):]
    if len(tail) < window:
        return {"suggest": False, "reason": "insufficient_post_peak_window",
                "peak_index": int(peak_idx), "note": _STABILITY_NOTE}

    drifts = [p.get("drift", {}).get("drift_max", np.inf) for p in tail]
    moves = [p.get("search_hamming", np.inf) for p in tail]
    taus_tail = [p.get("kendall", {}).get("sparse_kendall_tau") for p in tail]
    drift_ok = all(d <= drift_stable_max for d in drifts)
    move_ok = all(m <= move_stable_max for m in moves)
    tau_ok = all(t is not None and t >= tau_stable_min for t in taus_tail)

    suggest = bool(drift_ok and move_ok)
    reason = "both_stable_past_peak" if suggest else "not_yet_stable"
    if suggest and not tau_ok:
        # Kendall is record-only; do not block, but annotate.
        reason = "both_stable_past_peak_kendall_soft"

    return {
        "suggest": suggest,
        "reason": reason,
        "peak_index": int(peak_idx),
        "peak_tau": float(taus[peak_idx]) if taus[peak_idx] is not None else None,
        "window": int(window),
        "drift_ok": drift_ok,
        "move_ok": move_ok,
        "tau_ok": tau_ok,
        "tail_drift_max": drifts,
        "tail_search_hamming": moves,
        "tail_tau": taus_tail,
        "note": _STABILITY_NOTE,
    }


_STABILITY_NOTE = (
    "Stability is not correctness evidence. Ranking correlation is "
    "non-monotonic (rises then falls) — the curve must run past its peak. "
    "Correctness comes only from delivery-time top-k retrain."
)


def record_pair(U_t, U_tp, groups, dim, scores_t, scores_tp, allocations,
                bits, rate, tie_threshold, y=None, seed=0,
                score_t=None, score_tp=None, probe_search=False):
    """One adjacent-checkpoint probe record.

    ``scores_t`` / ``scores_tp`` are fixed-candidate scores (already computed
    once per checkpoint).  Optional expensive ``probe_search_move`` is gated
    by ``probe_search``.
    """
    drift = probe_drift(U_t, U_tp, groups, dim, y=y)
    scores_a = np.asarray(scores_t, dtype=np.float64)
    scores_b = np.asarray(scores_tp, dtype=np.float64)
    kendall = probe_kendall(scores_a, scores_b, tie_threshold)
    argmin_t = int(np.argmin(scores_a))
    argmin_tp = int(np.argmin(scores_b))
    alloc_t = np.asarray(allocations[argmin_t], dtype=np.int64)
    alloc_tp = np.asarray(allocations[argmin_tp], dtype=np.int64)
    move = hamming(alloc_t, alloc_tp)
    record = {
        "drift": drift,
        "search_hamming": move,  # fixed-pool argmin move (cheap proxy)
        "argmin_t": argmin_t,
        "argmin_tp": argmin_tp,
        "argmin_allocation_t": alloc_t.tolist(),
        "argmin_allocation_tp": alloc_tp.tolist(),
        "argmin_score_t": float(scores_a[argmin_t]),
        "argmin_score_tp": float(scores_b[argmin_tp]),
        "kendall": kendall,
        "scores_t": scores_a.tolist(),
        "scores_tp": scores_b.tolist(),
        "note": _STABILITY_NOTE,
    }
    if probe_search:
        if score_t is None or score_tp is None:
            raise ValueError("probe_search requires score_t and score_tp")
        search_t = probe_search_move(score_t, groups, bits, rate, seed=seed)
        search_tp = probe_search_move(score_tp, groups, bits, rate, seed=seed)
        search_move = None
        if (search_t["allocation"] is not None
                and search_tp["allocation"] is not None):
            search_move = hamming(search_t["allocation"], search_tp["allocation"])
        record["search_t"] = search_t
        record["search_tp"] = search_tp
        record["search_move_hamming"] = search_move
    return record


# --------------------------------------------------------------------- CLI ---

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", default="blk20", choices=tuple(SPECS))
    parser.add_argument("--anchor", default="R64")
    parser.add_argument("--checkpoints", nargs="+",
                        help=f"ordered {ckpt.FORMAT} paths (adjacent pairs probed)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--body-images", type=int, default=128)
    parser.add_argument("--image-batch", type=int, default=16)
    parser.add_argument("--n-candidates", type=int, default=64)
    parser.add_argument("--tie-threshold", type=float, default=None,
                        help="override; default from --noise-floor-json")
    parser.add_argument("--noise-floor-json",
                        help="calibration JSON providing tie_threshold")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--drift-stable-max", type=float, default=1e-3)
    parser.add_argument("--move-stable-max", type=int, default=0)
    parser.add_argument("--tau-stable-min", type=float, default=0.9)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--probe-search-move", action="store_true",
                        help="also run expensive simplified local search "
                             "(default: fixed-candidate argmin only)")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        report = _smoke_probe(args)
    else:
        report = _real_probe(args)

    report["plan"] = "v33_switch_probe"
    report["seconds"] = time.time() - started
    report["stability_note"] = _STABILITY_NOTE
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "n_pairs": len(report.get("curve", [])),
        "suggestion": report.get("suggestion"),
        "seconds": report["seconds"],
        "smoke": report.get("smoke", False),
    }, indent=2))


def _tie_threshold(args):
    if args.tie_threshold is not None:
        return float(args.tie_threshold)
    if args.noise_floor_json:
        payload = json.loads(Path(args.noise_floor_json).read_text())
        return float(payload["tie_threshold"])
    return 0.0


def _real_probe(args):
    if not args.checkpoints or len(args.checkpoints) < 2:
        raise SystemExit("need >=2 --checkpoints")
    activate(args.block)
    engine.configure_precision(C.ALLOW_TF32)
    device = torch.device(args.device)
    anchor = C.ANCHOR_BY_NAME[args.anchor]
    tie_thr = _tie_threshold(args)
    tail = tail_mod.build_tail(C.LAYER, device)
    resident = valset.load_val_resident(
        device, n_images=args.body_images, image_batch=args.image_batch)
    y_energy = resident.y[: min(32, resident.count)]

    # Geometry / candidate pool from the first checkpoint only.
    codec0, _ = ckpt.load_checkpoint(args.checkpoints[0], device=device)
    groups = codec0.pq.G
    bits = tuple(codec0.pq.mode_bits)
    dim = codec0.pq.d
    allocations = noise_floor.random_exact_budget_allocations(
        groups, bits, anchor.rate, args.n_candidates, seed=args.seed)
    del codec0
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Score each checkpoint once; adjacent pairs reuse cached scores.
    Us = []
    score_tables = []
    argmins = []
    scorers = []  # only retained when --probe-search-move
    for path in args.checkpoints:
        codec, payload = ckpt.load_checkpoint(path, device=device)
        Us.append(codec.transform.get_rotation().detach().cpu())
        score_fn = search.make_codec_scorer(
            codec, tail, resident, args.image_batch)
        scores = score_candidates(score_fn, allocations)
        score_tables.append(scores)
        argmins.append(int(np.argmin(scores)))
        if args.probe_search_move:
            scorers.append(score_fn)
        else:
            if hasattr(score_fn, "close"):
                score_fn.close()
            del codec
            if device.type == "cuda":
                torch.cuda.empty_cache()

    curve = []
    for index in range(len(args.checkpoints) - 1):
        score_t = scorers[index] if args.probe_search_move else None
        score_tp = scorers[index + 1] if args.probe_search_move else None
        record = record_pair(
            Us[index], Us[index + 1], groups, dim,
            score_tables[index], score_tables[index + 1],
            allocations, bits, anchor.rate,
            tie_thr, y=y_energy, seed=args.seed,
            score_t=score_t, score_tp=score_tp,
            probe_search=bool(args.probe_search_move))
        record["checkpoint_t"] = str(Path(args.checkpoints[index]).resolve())
        record["checkpoint_tp"] = str(Path(args.checkpoints[index + 1]).resolve())
        record["pair_index"] = index
        curve.append(record)

    if args.probe_search_move:
        for score_fn in scorers:
            if hasattr(score_fn, "close"):
                score_fn.close()

    suggestion = suggest_switch(
        curve,
        drift_stable_max=args.drift_stable_max,
        move_stable_max=args.move_stable_max,
        tau_stable_min=args.tau_stable_min,
        window=args.window)

    unique_argmins = sorted(set(argmins))
    degenerate = len(unique_argmins) == 1
    degeneration = {
        "argmin_unique_count": len(unique_argmins),
        "argmin_per_checkpoint": argmins,
        "degenerate": degenerate,
        "note": (
            "Fixed-candidate argmin never moved across checkpoints — signal "
            "may be too weak. Increase --n-candidates or seed the pool with "
            "uniform-neighbourhood allocations."
            if degenerate else
            "Fixed-candidate argmin moved at least once."
        ),
    }

    return {
        "curve": curve,
        "suggestion": suggestion,
        "tie_threshold": tie_thr,
        "n_candidates": int(len(allocations)),
        "allocations": allocations.tolist(),
        "body_images": args.body_images,
        "anchor": args.anchor,
        "probe_search_move": bool(args.probe_search_move),
        "argmin_degeneration": degeneration,
    }


def _smoke_probe(args):
    """Synthetic drift / kendall / search-move curve (no features)."""
    rng = np.random.default_rng(args.seed)
    groups, dim, bits, rate = 8, 4, (1, 2, 3), 16
    D = groups * dim
    # Simulate U drifting then settling.
    q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    Us = [q]
    for t in range(6):
        # Early: larger skew, late: tiny.
        scale = 0.08 * max(0.0, 1.0 - t / 3.5)
        skew = scale * rng.normal(size=(D, D))
        skew = 0.5 * (skew - skew.T)
        delta = np.eye(D) + skew
        q_next, _ = np.linalg.qr(Us[-1] @ delta)
        Us.append(q_next)

    score_tables = []
    base = rng.normal(size=(groups, len(bits)))
    for t in range(len(Us)):
        # Ranking signal strengthens then weakly degrades (non-monotonic).
        strength = 1.0 + 0.5 * np.sin(np.pi * t / (len(Us) - 1))
        noise = 0.15 * (1.0 - t / (len(Us))) * rng.normal(size=base.shape)
        score_tables.append(strength * base + noise)

    def make_score(table):
        def score(allocation):
            allocation = np.asarray(allocation, dtype=np.int64)
            return float(sum(table[g, int(m)] for g, m in enumerate(allocation)))
        return score

    allocations = []
    seen = set()
    # Enumerate a few exact-budget rows deterministically.
    from itertools import product
    for row in product(range(len(bits)), repeat=groups):
        if search.nominal_rate(row, bits) != rate:
            continue
        key = tuple(row)
        if key in seen:
            continue
        seen.add(key)
        allocations.append(np.asarray(row, dtype=np.int64))
        if len(allocations) >= max(6, args.n_candidates):
            break
    allocations = np.stack(allocations[:args.n_candidates])

    tie_thr = _tie_threshold(args)
    y = torch.from_numpy(rng.normal(size=(16, D)).astype(np.float32))
    curve = []
    score_tables_eval = []
    for t in range(len(Us)):
        score_fn = make_score(score_tables[t])
        score_tables_eval.append(score_candidates(score_fn, allocations))
    for index in range(len(Us) - 1):
        record = record_pair(
            torch.from_numpy(Us[index].astype(np.float32)),
            torch.from_numpy(Us[index + 1].astype(np.float32)),
            groups, dim,
            score_tables_eval[index], score_tables_eval[index + 1],
            allocations, bits, rate, tie_thr, y=y, seed=args.seed,
            score_t=make_score(score_tables[index]),
            score_tp=make_score(score_tables[index + 1]),
            probe_search=True)
        record["pair_index"] = index
        curve.append(record)

    suggestion = suggest_switch(
        curve,
        drift_stable_max=args.drift_stable_max,
        move_stable_max=args.move_stable_max,
        tau_stable_min=args.tau_stable_min,
        window=min(args.window, 2))
    return {
        "smoke": True,
        "curve": curve,
        "suggestion": suggestion,
        "tie_threshold": tie_thr,
        "geometry": {"groups": groups, "dim": dim, "mode_bits": list(bits),
                     "rate": rate},
    }


if __name__ == "__main__":
    main()
