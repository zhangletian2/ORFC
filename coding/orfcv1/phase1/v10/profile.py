"""Stage C0: fix every training hyper-parameter before any arm runs.

The user's condition on the three-arm budget was explicit: *"额外步数应在正式训练
前根据 smoke profiling 固定，不能根据结果调整"*.  This module is that profiling
step, and it writes ``<anchor>/profile.json``; ``train.py`` refuses to start
without it.  Nothing downstream may edit that file -- a hyper-parameter chosen
after seeing an arm's outcome would make "matched budget" a claim rather than a
construction, which is exactly the failure mode v8 died of.

What is measured, per anchor:

``loss_scale``
    The frozen codec's mean per-image hard distortion on train-core at the
    uniform allocation.  The training loss is divided by it so the objective is
    O(1) and the learning rates mean the same thing at both anchors.  It is
    written to disk rather than recomputed per arm because CayleySGD's step size
    is ``min(lr, 1/||skew||)`` -- a loss scale differing by one ULP between two
    processes could put them on different sides of that ``min``.

``batch``
    The largest candidate that both fits under the VRAM ceiling with train-core
    resident and is fastest in images per second.  A2 additionally holds cal-500
    in VRAM, so the ceiling is set with that headroom already subtracted.

``lr_U``, ``lr_theta``
    A full grid, each probe starting from the frozen codec, judged by the mean
    train-core distortion over the last steps of the probe.  Two things make this
    safe: only train-core is read (dev and holdout are untouched), and the same
    pair is then used by all three arms.  The grid is evaluated at the *uniform*
    allocation, which is A1's setting throughout and A2's only at the start --
    so if that biases anything it biases it towards A1, and the primary endpoint
    D(A2) - D(A1) is being asked to come out negative.  The conservative
    direction is the one to be biased in.

``T_outer``, ``delta_N``
    The inner length is a fixed number of epochs over train-core (a constant,
    not a wall-clock budget, so a busier GPU cannot change the experiment), cut
    into ``S_OUTER_MAX + 1`` blocks.  ``delta_N`` is A3's extra step count,
    computed from the forward-equivalent cost of A2's outer sweeps: every
    allocation A2 will evaluate is counted exactly (992 candidates plus the base
    at the first event, 931 plus the base at each later one), a training step
    counts as ``3 x batch`` forward-equivalents (one forward, two for backward),
    and the ratio is ``delta_N``.  It is computed from the *budget*, not from
    what the outer loop turns out to accept, so it is knowable in advance.
"""

import argparse
import json
import time

import numpy as np
import torch

from . import config as C
from . import qhard
from . import search as search_mod
from . import train as train_mod
from .. import engine
from .. import frozen
from .. import splits
from .. import tail as tail_mod

# The inner budget, in passes over train-core.  A constant rather than a
# wall-clock target: two GPUs of different speed must run the same experiment.
INNER_EPOCHS = 100

BATCH_CANDIDATES = (16, 32, 64, 128)
# 24 GiB cards.  train-core resident is ~8.4 GiB (y + teacher) and A2 adds
# cal-500 at ~1.05 GiB, so the activation ceiling is set well below the card.
VRAM_CEILING_GIB = 12.0

# The grids are deliberately wider than they look like they need to be.  A first
# pass over lr_theta in {1e-3 .. 1e0} improved monotonically all the way to the
# largest value (+0.56%, +3.77%, +13.40%, +19.06% at lr_U = 1e-4), which puts the
# argmin on the boundary -- and a boundary optimum is not an optimum, it is a
# report that the grid stopped too early.  1e1 is there to be either chosen or
# rejected on its own evidence.  1e-5 does the same at the bottom of lr_U, where
# the first pass suggested the useful values are small.
LR_U_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
LR_THETA_GRID = (1e-3, 1e-2, 1e-1, 1e0, 1e1)
PROBE_STEPS = 200
PROBE_TAIL = 25          # steps averaged to score a probe

# Backward is charged at twice the forward, the usual accounting; the ratio is
# what matters and it is the same for both arms.
BACKWARD_MULTIPLIER = 2.0


def fresh_codec(anchor, device):
    codec, meta = frozen.load_checked(anchor, device)
    return train_mod.make_trainable(codec), meta


def measure_loss_scale(codec, tail, resident, anchor, log=print):
    started = time.time()
    allocation = engine.uniform_allocation(anchor)[None, :]
    matrix = engine.evaluate_allocations(
        codec, tail, resident, allocation, image_batch=C.EVAL_IMAGE_BATCH,
        pair_budget=C.EVAL_PAIR_BUDGET, per_image=True)
    scale = float(matrix[0].mean())
    log(f"[{anchor.name}] loss scale (frozen codec, uniform, train-core): "
        f"{scale:.4f}  ({time.time() - started:.1f}s)")
    return scale


def probe_batch(anchor, tail, resident, device, batch, scale, steps=20,
                log=print):
    """Throughput and peak VRAM of ``steps`` inner steps at this batch."""
    codec, _ = fresh_codec(anchor, device)
    opt_rotation, opt_books = train_mod.build_optimizers(
        codec, {"lr_U": 1e-4, "lr_theta": 1e-3})
    modes = torch.from_numpy(engine.uniform_allocation(anchor)).to(device)
    stream = train_mod.BatchStream(resident.count, batch, C.TRAIN_SEED)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    for _ in range(3):                                   # warm up the allocator
        train_mod.train_step(codec, tail, resident, stream.next().to(device),
                             modes, opt_rotation, opt_books, scale)
    torch.cuda.synchronize()
    started = time.time()
    for _ in range(steps):
        train_mod.train_step(codec, tail, resident, stream.next().to(device),
                             modes, opt_rotation, opt_books, scale)
    torch.cuda.synchronize()
    elapsed = time.time() - started
    peak = (torch.cuda.max_memory_allocated(device) - baseline) / 2 ** 30
    report = {"batch": int(batch), "seconds_per_step": elapsed / steps,
              "images_per_second": steps * batch / elapsed,
              "activation_gib": float(peak),
              "fits": bool(peak <= VRAM_CEILING_GIB)}
    log(f"[{anchor.name}] batch {batch:4d}: "
        f"{report['images_per_second']:7.1f} img/s  "
        f"{report['activation_gib']:5.2f} GiB  "
        f"{'ok' if report['fits'] else 'over ceiling'}")
    del codec, opt_rotation, opt_books
    torch.cuda.empty_cache()
    return report


def probe_lr(anchor, tail, resident, device, batch, lr_u, lr_theta, scale,
             steps=PROBE_STEPS):
    """Run a short training probe from the frozen codec; score it on train.

    The orthogonality figure recorded here is the *peak* over the probe, sampled
    on the step before each reorthogonalisation fires.  The first attempt at this
    module recorded only the final value, and because the probe length is a
    multiple of the reorthogonalisation period that value was always the
    projection's own residual -- all 16 cells reported an identical 1.8e-05 and
    said nothing whatever about the learning rate that produced them.  A number
    that cannot vary with the thing being profiled is not a measurement.
    """
    codec, _ = fresh_codec(anchor, device)
    opt_rotation, opt_books = train_mod.build_optimizers(
        codec, {"lr_U": lr_u, "lr_theta": lr_theta})
    modes = torch.from_numpy(engine.uniform_allocation(anchor)).to(device)
    stream = train_mod.BatchStream(resident.count, batch, C.TRAIN_SEED)
    history = np.zeros(steps, dtype=np.float64)
    diverged = False
    orth = frozen.orthogonality_error(codec)
    for step in range(1, steps + 1):
        if step % C.CAYLEY_REORTH_EVERY == 0:
            orth = max(orth, frozen.orthogonality_error(codec))
        _, mean, labels, y = train_mod.train_step(
            codec, tail, resident, stream.next().to(device), modes,
            opt_rotation, opt_books, scale)
        history[step - 1] = mean
        if not np.isfinite(mean):
            diverged = True
            break
        if step % C.REVIVE_EVERY == 0:
            qhard.revive_dead_codewords(codec, y, modes, labels)
    orth = max(orth, frozen.orthogonality_error(codec))
    del codec, opt_rotation, opt_books
    torch.cuda.empty_cache()

    first = float(history[:PROBE_TAIL].mean())
    last = float(history[-PROBE_TAIL:].mean()) if not diverged else float("inf")
    return {"lr_U": float(lr_u), "lr_theta": float(lr_theta),
            "first": first, "last": last,
            "relative_improvement": (first - last) / first if first else 0.0,
            "orthogonality_error": orth,
            "diverged": bool(diverged),
            "usable": bool(not diverged and np.isfinite(last)
                           and orth <= C.ORTH_TOL)}


def measure_eval_rate(codec, tail, cal, anchor, allocations=20, log=print):
    """Images per second of the outer sweep's hard evaluation."""
    candidates, _ = search_mod.legal_swaps(
        engine.uniform_allocation(anchor), anchor)
    subset = candidates[:allocations]
    torch.cuda.synchronize()
    started = time.time()
    engine.evaluate_allocations(codec, tail, cal, subset,
                                image_batch=C.EVAL_IMAGE_BATCH,
                                pair_budget=C.EVAL_PAIR_BUDGET, per_image=True)
    torch.cuda.synchronize()
    elapsed = time.time() - started
    rate = allocations * cal.count / elapsed
    log(f"[{anchor.name}] outer sweep rate: {rate:.1f} img/s")
    return rate


def outer_forward_equivalents(groups=C.GROUPS, s_outer_max=C.S_OUTER_MAX,
                              n_cal=500):
    """Exactly how many hard forwards A2's outer loop is budgeted.

    The neighbourhood shrinks as the search runs, because a three-mode menu has
    no headroom at its ends: after ``k`` swaps that each touched a fresh pair of
    groups, ``k`` groups sit at the bottom mode and cannot go down and ``k`` sit
    at the top and cannot go up, so the count is ``(G-k)^2 - (G-2k)`` -- the
    square minus the ``i == j`` diagonal that survives in both sets.  Each event
    also re-evaluates its own base point, hence the ``+1``.  Stage B measured
    993 / 932 / 873 rows on R64, which is this formula at k = 0, 1, 2.

    Distinct pairs every event is the *largest* the neighbourhood can be (a swap
    that moves a group back toward the middle re-opens both directions), so this
    is an upper bound on A2's outer cost.  Being an upper bound means ``delta_N``
    is, if anything, generous to A3 -- the uniform arm -- which is the
    conservative direction for an endpoint asked to come out negative.  It is a
    *budget*: it does not depend on how many swaps the search accepts, which is
    what makes it fixable before either arm runs.
    """
    per_event = [(groups - k) ** 2 - (groups - 2 * k) + 1
                 for k in range(s_outer_max)]
    allocations = sum(per_event)
    return {"allocations_evaluated": int(allocations),
            "per_event": [int(v) for v in per_event],
            "events": int(s_outer_max),
            "forward_equivalents": float(allocations * n_cal)}


def run(anchor, device, log=print):
    started = time.time()
    engine.configure_precision(C.ALLOW_TF32)
    out = C.V10 / anchor.name
    out.mkdir(parents=True, exist_ok=True)

    codec, meta = frozen.load_checked(anchor, device)
    tail = tail_mod.build_tail(C.LAYER, device)
    feature_path, teacher_path, rows, _ = splits.load_split("train_core")
    tail_mod.check_layer(C.LAYER, feature_path, teacher_path,
                         meta["feature_cache"])
    resident = engine.ResidentSet(feature_path, teacher_path, rows, device)
    engine.assert_uniform_shape(resident.count)
    log(f"[{anchor.name}] train-core resident: {resident.count} images")

    scale = measure_loss_scale(codec, tail, resident, anchor, log=log)

    cal_feature, cal_teacher, cal_rows, _ = splits.load_split("cal")
    cal = engine.ResidentSet(cal_feature, cal_teacher, cal_rows, device)
    eval_rate = measure_eval_rate(codec, tail, cal, anchor, log=log)
    del codec, cal
    torch.cuda.empty_cache()

    log(f"[{anchor.name}] === batch ===")
    batches = [probe_batch(anchor, tail, resident, device, b, scale, log=log)
               for b in BATCH_CANDIDATES]
    usable = [b for b in batches if b["fits"]]
    if not usable:
        raise SystemExit(f"INVALID_EXPERIMENT: no candidate batch fits under "
                         f"{VRAM_CEILING_GIB} GiB: {batches}")
    best_batch = max(usable, key=lambda b: b["images_per_second"])
    batch = int(best_batch["batch"])
    log(f"[{anchor.name}] batch fixed at {batch}")

    log(f"[{anchor.name}] === learning rates ({len(LR_U_GRID)}x"
        f"{len(LR_THETA_GRID)} grid, {PROBE_STEPS} steps each) ===")
    grid = []
    for lr_u in LR_U_GRID:
        for lr_theta in LR_THETA_GRID:
            entry = probe_lr(anchor, tail, resident, device, batch, lr_u,
                             lr_theta, scale)
            grid.append(entry)
            log(f"[{anchor.name}]   lr_U {lr_u:.0e} lr_theta {lr_theta:.0e}: "
                f"D {entry['first']:.1f} -> "
                f"{entry['last'] if np.isfinite(entry['last']) else float('nan'):.1f}"
                f"  ({100 * entry['relative_improvement']:+.3f}%)  "
                f"orth {entry['orthogonality_error']:.1e}"
                f"{'' if entry['usable'] else '  UNUSABLE'}")
    winners = [e for e in grid if e["usable"]]
    if not winners:
        raise SystemExit(f"INVALID_EXPERIMENT: every learning-rate probe "
                         f"diverged or broke orthogonality: {grid}")
    best_lr = min(winners, key=lambda e: e["last"])
    on_edge = [name for name, value, edges in
               (("lr_U", best_lr["lr_U"], (LR_U_GRID[0], LR_U_GRID[-1])),
                ("lr_theta", best_lr["lr_theta"],
                 (LR_THETA_GRID[0], LR_THETA_GRID[-1])))
               if value in edges]
    if on_edge:
        log(f"[{anchor.name}] NOTE: the chosen {', '.join(on_edge)} sits on the "
            f"edge of its grid.  The value is used as measured -- widening the "
            f"grid now, after seeing which cell won, would be choosing a "
            f"hyper-parameter from an outcome.  Recorded so the write-up can "
            f"say the search was bounded here.")

    steps_per_epoch = resident.count // batch
    t_outer = INNER_EPOCHS * steps_per_epoch // (C.S_OUTER_MAX + 1)
    n_inner = C.inner_steps(t_outer)
    budget = outer_forward_equivalents(n_cal=int(cal_rows.shape[0]))
    per_step = (1.0 + BACKWARD_MULTIPLIER) * batch
    delta_n = int(round(budget["forward_equivalents"] / per_step))

    profile = {
        "plan": "v10", "stage": "C0_profile", "anchor": anchor.name,
        "rate": anchor.rate, "checkpoint_id": meta["checkpoint_id"],
        "loss_scale": scale,
        "batch": batch,
        "lr_U": best_lr["lr_U"], "lr_theta": best_lr["lr_theta"],
        "lr_on_grid_edge": on_edge,
        "T_outer": int(t_outer),
        "N_inner": int(n_inner),
        "delta_N": delta_n,
        "inner_epochs": INNER_EPOCHS,
        "steps_per_epoch": int(steps_per_epoch),
        "images_seen_A1": int(n_inner * batch),
        "images_seen_A3": int((n_inner + delta_n) * batch),
        "outer_budget": budget,
        "forward_equivalents": {
            "per_training_step": per_step,
            "A1_inner": float(n_inner * per_step),
            "A2_inner": float(n_inner * per_step),
            "A2_outer": budget["forward_equivalents"],
            "A2_total": float(n_inner * per_step
                              + budget["forward_equivalents"]),
            "A3_total": float((n_inner + delta_n) * per_step),
            "A3_over_A2": float((n_inner + delta_n) * per_step
                                / (n_inner * per_step
                                   + budget["forward_equivalents"])),
            "backward_multiplier": BACKWARD_MULTIPLIER,
        },
        "measured": {
            "batch_probes": batches,
            "train_images_per_second": best_batch["images_per_second"],
            "eval_images_per_second": eval_rate,
            "lr_grid": grid,
            "estimated_minutes": {
                "A1": n_inner * batch / best_batch["images_per_second"] / 60,
                "A2_inner": n_inner * batch
                            / best_batch["images_per_second"] / 60,
                "A2_outer": budget["forward_equivalents"] / eval_rate / 60,
                "A3": (n_inner + delta_n) * batch
                      / best_batch["images_per_second"] / 60,
            },
        },
        "frozen_note": "train.py reads this file and will not start without "
                       "it; editing it after any arm has run is a protocol "
                       "change, not a patch",
        "seconds": time.time() - started,
    }
    C.profile_path(anchor).write_text(json.dumps(profile, indent=2))
    minutes = profile["measured"]["estimated_minutes"]
    log(f"[{anchor.name}] FROZEN: batch {batch}, lr_U {best_lr['lr_U']:.0e}, "
        f"lr_theta {best_lr['lr_theta']:.0e}, T_outer {t_outer}, "
        f"N_inner {n_inner}, delta_N {delta_n}")
    log(f"[{anchor.name}] estimated: A1 {minutes['A1']:.1f} min, "
        f"A2 {minutes['A2_inner'] + minutes['A2_outer']:.1f} min "
        f"({minutes['A2_inner']:.1f} inner + {minutes['A2_outer']:.1f} outer), "
        f"A3 {minutes['A3']:.1f} min")
    return profile


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor", required=True, choices=list(C.ANCHOR_BY_NAME))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    anchor = C.ANCHOR_BY_NAME[args.anchor]
    if C.profile_path(anchor).exists() and not args.force:
        raise SystemExit(f"{C.profile_path(anchor)} exists; the "
                         f"hyper-parameters are frozen.  Re-profiling after an "
                         f"arm has run would be choosing them from an outcome.")
    profile = run(anchor, torch.device("cuda"))
    print(json.dumps({k: v for k, v in profile.items() if k != "measured"},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
