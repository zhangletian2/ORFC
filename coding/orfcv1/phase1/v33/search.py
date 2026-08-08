"""Budget-preserving local search over two-group transfers.

Neighbourhood
-------------
One mode step down on group ``i`` and one mode step up on group ``j``
(``i != j``).  With consecutive ``mode_bits`` this is exactly a one-bit
transfer and preserves the nominal rate.  Size ≈ ``G*(G-1)`` ≈ 10³ at G=32.

Proposal / accept
-----------------
A single-group cost table ranks the neighbourhood; only the top-``propose_top``
(default 50) candidates receive a real Tail-MSE evaluation.  Acceptance is
decided solely by real ``D`` — a wrong cost table wastes budget, never
correctness.

Evaluation contract
-------------------
Callers pass ``score_fn(allocation) -> float``.  The real path builds this from
:func:`phase1.v33.distortion.evaluate` and **scores one allocation per call**
(no batch folding).  Evaluation budget counters increment by 1 per call.

Budget / starts
---------------
* Body: ``body_images`` (default 128), multi-start, ``eval_budget`` ≈ 5000.
* Starts: uniform, cost-table DP greedy, then random exact-budget draws.
* Top-``topk`` body survivors are re-scored on ``recheck_images`` (default 500).
* Winner gets an exhaustive neighbourhood 1-opt certificate on the recheck set.
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
from ..v12.allocation_policy import FixedBudgetAllocationPolicy
from ..v21.config import SPECS, activate
from ..v22.allocation_dp import topk_allocations
from . import checkpoint as ckpt
from . import distortion as D
from . import valset


# ------------------------------------------------------------------ algebra ---

def nominal_rate(allocation, bits):
    bits = tuple(map(int, bits))
    return int(sum(bits[int(mode)] for mode in np.asarray(allocation).ravel()))


def uniform_allocation(groups, bits, rate):
    """Exact-budget uniform mode if possible, else raise."""
    bits = tuple(map(int, bits))
    groups, rate = int(groups), int(rate)
    for mode, bit in enumerate(bits):
        if bit * groups == rate:
            return np.full(groups, mode, dtype=np.int64)
    raise ValueError(f"no uniform mode hits rate {rate} for bits {bits}")


def two_group_transfers(allocation, bits):
    """All exact-budget adjacent-mode two-group transfers from ``allocation``.

    Returns ``(candidates [N, G], moves [N, 4])`` where each move is
    ``(down_group, down_from, up_group, up_from)`` and the candidate applies
    ``down_group: mode-1``, ``up_group: mode+1``.
    """
    base = np.asarray(allocation, dtype=np.int64).reshape(-1)
    bits = tuple(map(int, bits))
    top = len(bits) - 1
    candidates, moves = [], []
    for down in range(len(base)):
        if base[down] <= 0:
            continue
        for up in range(len(base)):
            if up == down or base[up] >= top:
                continue
            # Adjacent mode steps preserve rate iff bits are consecutive
            # integers; still verify.
            cand = base.copy()
            cand[down] -= 1
            cand[up] += 1
            if nominal_rate(cand, bits) != nominal_rate(base, bits):
                continue
            candidates.append(cand)
            moves.append((down, int(base[down]), up, int(base[up])))
    if not candidates:
        return (np.zeros((0, len(base)), dtype=np.int64),
                np.zeros((0, 4), dtype=np.int64))
    return np.stack(candidates), np.asarray(moves, dtype=np.int64)


def local_cost_probes(base, n_modes):
    """Single-group mode flips (may break budget) + identity for cost table."""
    base = np.asarray(base, dtype=np.int64)
    rows, keys = [base.copy()], [(None, None)]
    for group in range(len(base)):
        for mode in range(int(n_modes)):
            if mode == base[group]:
                continue
            row = base.copy()
            row[group] = mode
            rows.append(row)
            keys.append((group, mode))
    return np.stack(rows), keys


def build_cost_table(score_fn, base, n_modes, budget_counter=None):
    """``costs[g, m] = score(flip g→m) - score(base)``; diagonal stays 0."""
    probes, keys = local_cost_probes(base, n_modes)
    values = []
    for row in probes:
        values.append(float(score_fn(row)))
        if budget_counter is not None:
            budget_counter[0] += 1
    values = np.asarray(values, dtype=np.float64)
    base_score = float(values[0])
    costs = np.zeros((len(base), int(n_modes)), dtype=np.float64)
    for index, (group, mode) in enumerate(keys[1:], start=1):
        costs[group, mode] = values[index] - base_score
    return costs, base_score


def predicted_delta(costs, base, candidate):
    base = np.asarray(base, dtype=np.int64)
    candidate = np.asarray(candidate, dtype=np.int64)
    return float(sum(
        costs[g, int(candidate[g])] - costs[g, int(base[g])]
        for g in range(len(base))))


def propose_top(neighbors, moves, costs, base, top=50):
    """Rank neighbours by cost-table predicted delta; return top improving ones."""
    if len(neighbors) == 0:
        return neighbors, moves, np.zeros(0)
    deltas = np.asarray([
        predicted_delta(costs, base, row) for row in neighbors], dtype=np.float64)
    order = np.argsort(deltas, kind="stable")[:int(top)]
    return neighbors[order], moves[order], deltas[order]


# -------------------------------------------------------------------- search ---

class Budget:
    def __init__(self, limit):
        self.limit = int(limit)
        self.used = 0

    def remaining(self):
        return max(0, self.limit - self.used)

    def charge(self, n=1):
        self.used += int(n)
        return self.used <= self.limit

    def allow(self, n=1):
        return self.used + int(n) <= self.limit


def _counting_score_fn(score_fn, budget):
    def wrapped(allocation):
        if not budget.allow(1):
            raise _BudgetExhausted()
        value = float(score_fn(allocation))
        budget.charge(1)
        return value

    return wrapped


class _BudgetExhausted(Exception):
    pass


def hill_climb(score_fn, start, bits, propose_top_k=50, max_steps=64,
               eval_budget=None, rebuild_cost_table=True):
    """Greedy descent with cost-table proposals and real-D acceptance.

    Returns a dict with the final allocation, trace, and evals used.
    """
    bits = tuple(map(int, bits))
    n_modes = len(bits)
    current = np.asarray(start, dtype=np.int64).copy()
    budget = Budget(10 ** 9 if eval_budget is None else eval_budget)
    scored = _counting_score_fn(score_fn, budget)
    try:
        current_score = scored(current)
    except _BudgetExhausted:
        return {
            "allocation": current.tolist(),
            "score": None,
            "steps_accepted": 0,
            "trace": [],
            "evals_used": budget.used,
            "stopped": "budget_before_start",
        }

    # Always build a cost table once at the start; optionally rebuild after
    # each accepted move (full search).  Simplified switch-probe keeps the
    # initial table to save evaluations.
    try:
        costs, _ = build_cost_table(scored, current, n_modes)
    except _BudgetExhausted:
        return {
            "allocation": current.tolist(),
            "score": current_score,
            "steps_accepted": 0,
            "trace": [],
            "evals_used": budget.used,
            "stopped": "budget_during_cost_table",
        }

    trace = []
    accepted = 0
    stopped = "max_steps"
    for step in range(1, int(max_steps) + 1):
        neighbors, moves = two_group_transfers(current, bits)
        if len(neighbors) == 0:
            stopped = "empty_neighborhood"
            break
        proposed, proposed_moves, pred = propose_top(
            neighbors, moves, costs, current, top=propose_top_k)
        best_score, best_idx = current_score, None
        evaluated = []
        try:
            for idx, row in enumerate(proposed):
                value = scored(row)
                evaluated.append(float(value))
                if value < best_score:
                    best_score, best_idx = value, idx
        except _BudgetExhausted:
            stopped = "budget"
            trace.append({
                "step": step,
                "proposed": len(evaluated),
                "accepted": False,
                "note": "budget_exhausted_mid_proposal",
            })
            break

        take = best_idx is not None and best_score < current_score
        entry = {
            "step": step,
            "proposed": int(len(proposed)),
            "evaluated": int(len(evaluated)),
            "base_score": float(current_score),
            "best_score": float(best_score),
            "best_predicted_delta":
                float(pred[best_idx]) if best_idx is not None else None,
            "accepted": bool(take),
        }
        if take:
            current = proposed[best_idx].copy()
            current_score = best_score
            accepted += 1
            entry["move"] = proposed_moves[best_idx].tolist()
            if rebuild_cost_table:
                try:
                    costs, _ = build_cost_table(scored, current, n_modes)
                except _BudgetExhausted:
                    stopped = "budget"
                    trace.append(entry)
                    break
        trace.append(entry)
        if not take:
            stopped = "local_optimum_proposal"
            break

    return {
        "allocation": current.tolist(),
        "score": float(current_score),
        "steps_accepted": int(accepted),
        "trace": trace,
        "evals_used": int(budget.used),
        "stopped": stopped,
    }


def cost_table_greedy_start(score_fn, groups, bits, rate, budget_counter=None):
    """DP-greedy exact-budget start from a uniform-centred cost table."""
    bits = tuple(map(int, bits))
    base = uniform_allocation(groups, bits, rate)
    costs, _ = build_cost_table(
        score_fn, base, len(bits), budget_counter=budget_counter)
    winners = topk_allocations(costs, bits, rate, topk=1)
    return np.asarray(winners[0][1], dtype=np.int64)


def random_starts(groups, bits, rate, count, seed):
    policy = FixedBudgetAllocationPolicy(groups, bits, rate)
    generator = torch.Generator().manual_seed(int(seed))
    draws = policy.build(1.0).sample(max(int(count) * 3, 1), generator=generator)
    rows, seen = [], set()
    for row in draws.cpu().numpy():
        key = tuple(map(int, row))
        if key in seen:
            continue
        seen.add(key)
        rows.append(np.asarray(row, dtype=np.int64))
        if len(rows) >= int(count):
            break
    return rows


def multi_start_search(score_fn, groups, bits, rate, eval_budget=5000,
                       propose_top_k=50, max_steps=64, n_random_starts=4,
                       seed=0, simplified=False):
    """Run several hill-climbs sharing one evaluation budget."""
    bits = tuple(map(int, bits))
    if simplified:
        propose_top_k = min(int(propose_top_k), 20)
        max_steps = min(int(max_steps), 12)
        n_random_starts = min(int(n_random_starts), 2)
        eval_budget = min(int(eval_budget), 400)

    class _Shared:
        used = 0
        limit = int(eval_budget)

        def __call__(self, allocation):
            if self.used >= self.limit:
                raise _BudgetExhausted()
            value = float(score_fn(allocation))
            self.used += 1
            return value

    shared = _Shared()

    starts = [("uniform", uniform_allocation(groups, bits, rate))]
    try:
        counter = [0]

        def _count_wrap(a):
            counter[0] += 1
            return shared(a)

        greedy = cost_table_greedy_start(
            _count_wrap, groups, bits, rate)
        starts.append(("cost_table_greedy", greedy))
    except _BudgetExhausted:
        pass

    for index, row in enumerate(random_starts(
            groups, bits, rate, n_random_starts, seed)):
        starts.append((f"random_{index}", row))

    unique, seen = [], set()
    for name, row in starts:
        key = tuple(map(int, row))
        if key not in seen:
            seen.add(key)
            unique.append((name, row))

    per_start = []
    best = None
    for name, row in unique:
        remaining = shared.limit - shared.used
        if remaining <= 0:
            break
        before = shared.used
        result = hill_climb(
            shared, row, bits,
            propose_top_k=propose_top_k,
            max_steps=max_steps,
            eval_budget=remaining,
            rebuild_cost_table=not simplified,
        )
        result["evals_used"] = int(shared.used - before)
        entry = {"start": name, "start_allocation": row.tolist(), **result}
        per_start.append(entry)
        if result["score"] is not None and (
                best is None or result["score"] < best["score"]):
            best = {
                "start": name,
                "allocation": result["allocation"],
                "score": result["score"],
            }

    return {
        "best": best,
        "per_start": per_start,
        "evals_used": int(shared.used),
        "eval_budget": int(eval_budget),
        "simplified": bool(simplified),
        "n_starts": len(per_start),
    }


def collect_topk_candidates(multi_result, topk=5):
    """Unique allocations from multi-start ranked by body score."""
    rows = []
    for entry in multi_result["per_start"]:
        if entry.get("score") is None:
            continue
        rows.append((float(entry["score"]),
                     np.asarray(entry["allocation"], dtype=np.int64),
                     entry["start"]))
    rows.sort(key=lambda item: item[0])
    unique, seen = [], set()
    for score, alloc, start in rows:
        key = tuple(map(int, alloc))
        if key in seen:
            continue
        seen.add(key)
        unique.append({"score_body": score, "allocation": alloc,
                       "start": start})
        if len(unique) >= int(topk):
            break
    return unique


def recheck_topk(score_fn, candidates):
    """Re-score candidates (typically on the 500-image set)."""
    scored = []
    for row in candidates:
        value = float(score_fn(row["allocation"]))
        scored.append({
            "allocation": np.asarray(row["allocation"], dtype=np.int64).tolist(),
            "score_body": float(row["score_body"]),
            "score_recheck": value,
            "start": row["start"],
        })
    scored.sort(key=lambda item: item["score_recheck"])
    return scored


def one_opt_certificate(score_fn, allocation, bits, propose_top_k=None):
    """Exhaustive neighbourhood check → 1-opt certificate.

    If ``propose_top_k`` is None, every neighbour is scored (true 1-opt).
    """
    bits = tuple(map(int, bits))
    base = np.asarray(allocation, dtype=np.int64)
    base_score = float(score_fn(base))
    neighbors, moves = two_group_transfers(base, bits)
    if propose_top_k is not None and len(neighbors) > 0:
        costs, _ = build_cost_table(score_fn, base, len(bits))
        neighbors, moves, _ = propose_top(
            neighbors, moves, costs, base, top=propose_top_k)
    best_score, best_move = base_score, None
    improvements = 0
    for row, move in zip(neighbors, moves):
        value = float(score_fn(row))
        if value < best_score:
            best_score, best_move = value, move.tolist()
            improvements += 1
    return {
        "allocation": base.tolist(),
        "base_score": base_score,
        "neighborhood_size": int(len(neighbors)),
        "best_neighbor_score": float(best_score),
        "best_neighbor_move": best_move,
        "gain": float(base_score - best_score),
        "is_1opt": bool(best_score >= base_score - 1e-15),
        "neighbors_strictly_better": int(improvements),
    }


def make_codec_scorer(codec, tail, resident, image_batch=16):
    """Return a scalar scorer that shares one frozen U0/L bank across calls."""
    from .quantise import frozen_rotations

    # Keep the context open for the scorer lifetime so every hill-climb
    # evaluation reuses the same materialised rotations.
    ctx = frozen_rotations(codec)
    ctx.__enter__()

    @torch.no_grad()
    def score(allocation):
        values = D.evaluate(
            codec, tail, resident, allocation, image_batch=image_batch)
        return float(np.asarray(values).reshape(-1).mean())

    score._v33_rotation_ctx = ctx  # keep GC from closing early
    score.close = lambda: ctx.__exit__(None, None, None)
    return score


def run_search(score_body, score_recheck, groups, bits, rate,
               eval_budget=5000, propose_top_k=50, max_steps=64,
               n_random_starts=4, topk=5, seed=0, simplified=False,
               certify=True):
    """Full pipeline: multi-start body → top-k recheck → optional 1-opt cert."""
    body = multi_start_search(
        score_body, groups, bits, rate,
        eval_budget=eval_budget, propose_top_k=propose_top_k,
        max_steps=max_steps, n_random_starts=n_random_starts,
        seed=seed, simplified=simplified)
    candidates = collect_topk_candidates(body, topk=topk)
    if not candidates:
        return {"body": body, "recheck": [], "winner": None, "certificate": None}
    recheck = recheck_topk(score_recheck, candidates)
    winner = recheck[0]
    certificate = None
    if certify and not simplified:
        certificate = one_opt_certificate(
            score_recheck, winner["allocation"], bits, propose_top_k=None)
    return {
        "body": body,
        "recheck": recheck,
        "winner": winner,
        "certificate": certificate,
    }


# ------------------------------------------------------------- synthetic / CLI ---

def synthetic_score_fn(bits, rate, seed=0, noise=0.01):
    """Separable planted costs + small noise — for unit smoke only."""
    bits = tuple(map(int, bits))
    rng = np.random.default_rng(seed)
    # Prefer lower modes on early groups, higher on late — creates structure.
    groups = 8  # overridden by closure after first call… set via attribute.

    def make(groups_):
        table = rng.normal(size=(groups_, len(bits)))
        table += np.arange(len(bits))[None, :] * 0.2

        def score(allocation):
            allocation = np.asarray(allocation, dtype=np.int64)
            value = float(sum(table[g, int(m)] for g, m in enumerate(allocation)))
            # Tiny hash noise so ties are rare but deterministic given allocation.
            noise_term = noise * ((hash(tuple(map(int, allocation))) % 1000) / 1000.0)
            return value + noise_term

        return score, table

    return make


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", default="blk20", choices=tuple(SPECS))
    parser.add_argument("--anchor", default="R64")
    parser.add_argument("--checkpoint", help=f"{ckpt.FORMAT} path")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--body-images", type=int, default=128)
    parser.add_argument("--recheck-images", type=int, default=valset.N_VAL)
    parser.add_argument("--image-batch", type=int, default=16)
    parser.add_argument("--eval-budget", type=int, default=5000)
    parser.add_argument("--propose-top", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--random-starts", type=int, default=4)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--simplified", action="store_true",
                        help="switch-probe budget (fewer evals / steps)")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        result = _smoke_search(args)
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required unless --smoke")
        activate(args.block)
        engine.configure_precision(C.ALLOW_TF32)
        device = torch.device(args.device)
        codec, payload = ckpt.load_checkpoint(args.checkpoint, device=device)
        anchor = C.ANCHOR_BY_NAME[args.anchor]
        bits = tuple(codec.pq.mode_bits)
        if bits != tuple(anchor.mode_bits):
            raise SystemExit(f"codec bits {bits} != anchor {anchor.mode_bits}")
        tail = tail_mod.build_tail(C.LAYER, device)
        body_resident = valset.load_val_resident(
            device, n_images=args.body_images, image_batch=args.image_batch)
        recheck_resident = valset.load_val_resident(
            device, n_images=args.recheck_images, image_batch=args.image_batch)
        score_body = make_codec_scorer(
            codec, tail, body_resident, args.image_batch)
        score_recheck = make_codec_scorer(
            codec, tail, recheck_resident, args.image_batch)
        result = run_search(
            score_body, score_recheck, codec.pq.G, bits, anchor.rate,
            eval_budget=args.eval_budget, propose_top_k=args.propose_top,
            max_steps=args.max_steps, n_random_starts=args.random_starts,
            topk=args.topk, seed=args.seed, simplified=args.simplified,
            certify=not args.simplified)
        result["checkpoint"] = str(Path(args.checkpoint).resolve())
        result["meta"] = payload.get("meta", {})
        result["body_images"] = args.body_images
        result["recheck_images"] = args.recheck_images

    result["plan"] = "v33_local_search"
    result["seconds"] = time.time() - started
    # JSON-safe: convert numpy in nested structures via default.
    out.write_text(json.dumps(result, indent=2, default=_json_default))
    winner = result.get("winner")
    if winner is not None and winner.get("allocation") is not None:
        # Phase-2 hand-off sibling (same contract as checkpoint.save_checkpoint).
        np.save(out.with_name(out.stem + "_allocation.npy"),
                np.asarray(winner["allocation"], dtype=np.int64))
        np.save(out.with_name("allocation.npy"),
                np.asarray(winner["allocation"], dtype=np.int64))
    summary = {
        "winner": winner,
        "certificate": result.get("certificate"),
        "body_evals": result.get("body", {}).get("evals_used"),
        "seconds": result["seconds"],
        "smoke": result.get("smoke", False),
        "allocation_npy": str(out.with_name("allocation.npy"))
        if winner is not None else None,
    }
    print(json.dumps(summary, indent=2))


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(type(obj))


def _smoke_search(args):
    groups, bits, rate = 8, (1, 2, 3), 16
    score, _ = synthetic_score_fn(bits, rate, seed=args.seed)(groups)

    def score_body(a):
        return score(a)

    result = run_search(
        score_body, score_body, groups, bits, rate,
        eval_budget=min(args.eval_budget, 800),
        propose_top_k=min(args.propose_top, 30),
        max_steps=min(args.max_steps, 20),
        n_random_starts=min(args.random_starts, 3),
        topk=min(args.topk, 3),
        seed=args.seed,
        simplified=args.simplified,
        certify=True)
    result["smoke"] = True
    result["geometry"] = {"groups": groups, "mode_bits": list(bits), "rate": rate}
    return result


if __name__ == "__main__":
    main()
