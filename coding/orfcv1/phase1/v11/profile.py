"""N17: fix every training hyper-parameter before either arm runs.

Plan v11 section 7.  v10's failure here was not a bad learning rate but a bad
*procedure*: a 200-step probe chose ``lr_theta = 1e+1``, that value was used for
a 3096-step run, and the trajectory bottomed at step ~352 and rose monotonically
for the remaining 2744.  A short probe measures which cell descends fastest in
200 steps, which is a different question from which cell is lowest at the end of
a long annealed run, and v10 answered the second question with the first one's
answer.

So the probe no longer selects.  Four steps, all fixed in ``config.py`` before
anything ran:

  1. the 5x5 grid at ``PROBE_STEPS`` is used ONLY to reject numerically invalid
     cells (divergence, non-finite loss, orthogonality drift) and to produce a
     short-range ordering;
  2. the shortlist is three cells: the short-range argmin, the same ``lr_U`` with
     ``lr_theta`` one decade lower, and the grid's geometric centre.  The third
     is a *rule* -- index 2 of each grid, fixed by the grid's shape -- so it
     cannot be chosen from an outcome;
  3. all three are run at the full ``N_inner`` under the cosine schedule, through
     ``train.run`` itself.  Profiling a different code path from the one the arms
     take would be profiling something else;
  4. among the full-length cells that **pass G2**, the endpoint (train-val mean
     over the last ``FULL_LENGTH_TAIL`` measurements) selects.

If no full-length cell passes G2 at an anchor, that anchor does not enter the two
arms.  An empty set across both anchors ends v11.

The full-length runs are done in the uniform arm (``A1``'s setting): it is A1's
configuration throughout and A2's only at the start, so any bias this introduces
favours A1 -- and the endpoint being asked for is ``D(A2) - D(A1) < 0``.  Being
biased in the conservative direction is the point.

Cost is stated rather than hidden: three full-length runs per anchor before the
two real arms, i.e. profiling costs 1.5x what the arms themselves cost.  That is
the price of not choosing a long-run hyper-parameter from a short run.
"""

import argparse
import json
import time

import numpy as np
import torch

from . import config as C
from . import init_menu
from . import qhard
from . import search as search_mod
from . import train as train_mod
from .. import engine
from .. import frozen
from .. import tail as tail_mod


def measure_loss_scale(codec, tail, resident, anchor, log=print):
    """Mean per-image hard distortion at the uniform allocation on train-fit.

    The training loss is divided by it so the objective is O(1) and a learning
    rate means the same thing at both anchors.  Written to disk rather than
    recomputed per arm because CayleySGD's step is ``min(lr, 1/||skew||)`` -- a
    loss scale differing by one ULP between two processes could put them on
    different sides of that ``min``.
    """
    started = time.time()
    allocation = engine.uniform_allocation(anchor)[None, :]
    matrix = engine.evaluate_allocations(
        codec, tail, resident, allocation, image_batch=C.EVAL_IMAGE_BATCH,
        pair_budget=C.EVAL_PAIR_BUDGET, per_image=True)
    scale = float(matrix[0].mean())
    log(f"[{anchor.name}] loss scale (menu codec, uniform, train-fit): "
        f"{scale:.4f}  ({time.time() - started:.1f}s)")
    return scale


def probe_batch(anchor, tail, resident, device, batch, scale, steps=20,
                log=print):
    """Throughput and peak VRAM of ``steps`` inner steps at this batch.

    The step measured is v11's, i.e. main plus auxiliary, so the numbers already
    carry the ~2x the menu-coverage term costs.
    """
    codec, _ = init_menu.load_checked(anchor, device)
    train_mod.make_trainable(codec)
    opt_rotation, opt_books = train_mod.build_optimizers(
        codec, {"lr_U": 1e-4, "lr_theta": 1e-3})
    modes = torch.from_numpy(engine.uniform_allocation(anchor)).to(device)
    menu = train_mod.MenuStream(anchor)
    stream = train_mod.BatchStream(resident.count, batch, C.TRAIN_SEED)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    step = 0
    for _ in range(3):                                   # warm up the allocator
        step += 1
        _, aux_allocation = menu.allocation(step)
        train_mod.train_step(codec, tail, resident, stream.next().to(device),
                             modes, torch.from_numpy(aux_allocation).to(device),
                             opt_rotation, opt_books, scale)
    torch.cuda.synchronize()
    started = time.time()
    for _ in range(steps):
        step += 1
        _, aux_allocation = menu.allocation(step)
        train_mod.train_step(codec, tail, resident, stream.next().to(device),
                             modes, torch.from_numpy(aux_allocation).to(device),
                             opt_rotation, opt_books, scale)
    torch.cuda.synchronize()
    elapsed = time.time() - started
    peak = (torch.cuda.max_memory_allocated(device) - baseline) / 2 ** 30
    report = {"batch": int(batch), "seconds_per_step": elapsed / steps,
              "images_per_second": steps * batch / elapsed,
              "activation_gib": float(peak),
              "fits": bool(peak <= C.VRAM_CEILING_GIB)}
    log(f"[{anchor.name}] batch {batch:4d}: "
        f"{report['images_per_second']:7.1f} img/s  "
        f"{report['activation_gib']:5.2f} GiB  "
        f"{'ok' if report['fits'] else 'over ceiling'}")
    del codec, opt_rotation, opt_books
    torch.cuda.empty_cache()
    return report


def probe_lr(anchor, tail, resident, device, batch, lr_u, lr_theta, scale,
             steps=C.PROBE_STEPS):
    """A short probe under the *same* cosine schedule, scaled to its own length.

    Annealing over 200 steps is not the same trajectory as annealing over
    N_inner, and this probe does not pretend otherwise -- step 1 rejects invalid
    cells and orders the survivors, and that is all it is read for.  Running it
    unannealed would have made even the ordering incomparable with the
    full-length runs that follow.

    The orthogonality figure is the *peak* over the probe, sampled on the step
    before each reorthogonalisation fires.  Recording only the final value -- as
    a first version of v10's module did -- reports the projection's own residual
    and says nothing about the learning rate that produced it.
    """
    codec, _ = init_menu.load_checked(anchor, device)
    orth0 = frozen.orthogonality_error(codec)
    train_mod.make_trainable(codec)
    opt_rotation, opt_books = train_mod.build_optimizers(
        codec, {"lr_U": lr_u, "lr_theta": lr_theta})
    modes = torch.from_numpy(engine.uniform_allocation(anchor)).to(device)
    menu = train_mod.MenuStream(anchor)
    stream = train_mod.BatchStream(resident.count, batch, C.TRAIN_SEED)
    history = np.zeros(steps, dtype=np.float64)
    diverged = False
    orth = orth0
    hp = {"lr_U": lr_u, "lr_theta": lr_theta}
    for step in range(1, steps + 1):
        if step % C.CAYLEY_REORTH_EVERY == 0:
            orth = max(orth, frozen.orthogonality_error(codec))
        _, aux_allocation = menu.allocation(step)
        train_mod.set_learning_rates(opt_rotation, opt_books, step - 1, steps,
                                     hp)
        _, mean, _, main_labels, aux_labels, y = train_mod.train_step(
            codec, tail, resident, stream.next().to(device), modes,
            torch.from_numpy(aux_allocation).to(device), opt_rotation,
            opt_books, scale)
        history[step - 1] = mean
        if not np.isfinite(mean):
            diverged = True
            break
        if step % C.REVIVE_EVERY == 0:
            qhard.revive_dead_codewords(codec, y, modes, main_labels)
            qhard.revive_dead_codewords(
                codec, y, torch.from_numpy(aux_allocation).to(device),
                aux_labels)
    orth = max(orth, frozen.orthogonality_error(codec))
    del codec, opt_rotation, opt_books
    torch.cuda.empty_cache()

    first = float(history[:C.PROBE_TAIL].mean())
    last = float(history[-C.PROBE_TAIL:].mean()) if not diverged else float("inf")
    drift = orth - orth0
    return {"lr_U": float(lr_u), "lr_theta": float(lr_theta),
            "first": first, "last": last,
            "relative_improvement": (first - last) / first if first else 0.0,
            "orthogonality_initial": orth0, "orthogonality_peak": orth,
            "orthogonality_drift": drift,
            "diverged": bool(diverged),
            "valid": bool(not diverged and np.isfinite(last)
                          and drift <= C.ORTH_TOL)}


def shortlist(grid, log=print):
    """Step 2: three cells, by rule, from the short-range ordering."""
    valid = [e for e in grid if e["valid"]]
    if not valid:
        raise SystemExit(
            f"INVALID_EXPERIMENT: every learning-rate probe diverged or broke "
            f"orthogonality; there is no numerically valid cell to shortlist")
    best = min(valid, key=lambda e: e["last"])
    cells = [(best["lr_U"], best["lr_theta"], "short-range argmin")]

    lower = [t for t in C.LR_THETA_GRID if t < best["lr_theta"]]
    if lower:
        cells.append((best["lr_U"], max(lower),
                      "same lr_U, lr_theta one decade lower"))
    else:
        log(f"[shortlist] the short-range argmin already sits at the bottom of "
            f"the lr_theta grid, so the 'one decade lower' cell does not exist; "
            f"recorded rather than substituted, since substituting one would be "
            f"inventing a rule after seeing the outcome")

    reference = (float(C.SHORTLIST_REFERENCE[0]),
                 float(C.SHORTLIST_REFERENCE[1]))
    if not any(abs(u - reference[0]) < 1e-12 and abs(t - reference[1]) < 1e-12
               for u, t, _ in cells):
        cells.append((reference[0], reference[1], "pre-registered grid centre"))
    return cells


def measure_eval_rate(codec, tail, cal, anchor, allocations=20, log=print):
    """Images per second of the outer sweep's hard evaluation."""
    candidates, _ = search_mod.legal_swaps(
        engine.uniform_allocation(anchor), anchor)
    torch.cuda.synchronize()
    started = time.time()
    engine.evaluate_allocations(codec, tail, cal, candidates[:allocations],
                                image_batch=C.EVAL_IMAGE_BATCH,
                                pair_budget=C.EVAL_PAIR_BUDGET, per_image=True)
    torch.cuda.synchronize()
    elapsed = time.time() - started
    rate = allocations * cal.count / elapsed
    log(f"[{anchor.name}] outer sweep rate: {rate:.1f} img/s")
    return rate


def outer_forward_equivalents(groups=C.GROUPS, s_outer_max=C.S_OUTER_MAX,
                              n_cal=500):
    """A2's outer budget, counted exactly, as a reported method cost.

    The neighbourhood shrinks as the search runs: after ``k`` swaps that each
    touched a fresh pair of groups, ``k`` groups sit at the bottom mode and
    cannot go down and ``k`` sit at the top and cannot go up, so the count is
    ``(G-k)^2 - (G-2k)`` -- the square minus the ``i == j`` diagonal that
    survives in both sets.  Each event re-evaluates its own base point, hence the
    ``+1``.  N16 measured 993 rows at k = 0, which is this formula.

    Under the user's ruling this is *reported*, not compensated: v10's
    equal-compute third arm is out of the main flow, so no extra uniform steps
    are bought with this number.
    """
    per_event = [(groups - k) ** 2 - (groups - 2 * k) + 1
                 for k in range(s_outer_max)]
    allocations = sum(per_event)
    return {"allocations_evaluated": int(allocations),
            "per_event": [int(v) for v in per_event],
            "events": int(s_outer_max),
            "forward_equivalents": float(allocations * n_cal)}


def run(anchor, device, log=print, skip_full_length=False):
    started = time.time()
    engine.configure_precision(C.ALLOW_TF32)

    codec, meta = init_menu.load_checked(anchor, device)
    tail = tail_mod.build_tail(C.LAYER, device)
    feature_path, teacher_path, rows, _ = C.load_split("train_fit")
    tail_mod.check_layer(C.LAYER, feature_path, teacher_path,
                         meta["feature_cache"])
    resident = engine.ResidentSet(feature_path, teacher_path, rows, device)
    log(f"[{anchor.name}] train-fit resident: {resident.count} images")

    scale = measure_loss_scale(codec, tail, resident, anchor, log=log)
    cal_feature, cal_teacher, cal_rows, _ = C.load_split("cal")
    cal = engine.ResidentSet(cal_feature, cal_teacher, cal_rows, device)
    eval_rate = measure_eval_rate(codec, tail, cal, anchor, log=log)
    del codec, cal
    torch.cuda.empty_cache()

    log(f"[{anchor.name}] === batch ===")
    batches = [probe_batch(anchor, tail, resident, device, b, scale, log=log)
               for b in C.BATCH_CANDIDATES]
    usable = [b for b in batches if b["fits"]]
    if not usable:
        raise SystemExit(f"INVALID_EXPERIMENT: no candidate batch fits under "
                         f"{C.VRAM_CEILING_GIB} GiB: {batches}")
    best_batch = max(usable, key=lambda b: b["images_per_second"])
    batch = int(best_batch["batch"])
    log(f"[{anchor.name}] batch fixed at {batch}")

    steps_per_epoch = resident.count // batch
    t_outer = C.INNER_EPOCHS * steps_per_epoch // (C.S_OUTER_MAX + 1)
    n_inner = C.inner_steps(t_outer)

    log(f"[{anchor.name}] === step 1: {len(C.LR_U_GRID)}x"
        f"{len(C.LR_THETA_GRID)} grid at {C.PROBE_STEPS} steps "
        f"(validity + ordering only) ===")
    grid = []
    for lr_u in C.LR_U_GRID:
        for lr_theta in C.LR_THETA_GRID:
            entry = probe_lr(anchor, tail, resident, device, batch, lr_u,
                             lr_theta, scale)
            grid.append(entry)
            shown = entry["last"] if np.isfinite(entry["last"]) else float("nan")
            log(f"[{anchor.name}]   lr_U {lr_u:.0e} lr_theta {lr_theta:.0e}: "
                f"D {entry['first']:.1f} -> {shown:.1f}  "
                f"({100 * entry['relative_improvement']:+.3f}%)  "
                f"drift {entry['orthogonality_drift']:.1e}"
                f"{'' if entry['valid'] else '  INVALID'}")

    cells = shortlist(grid, log=log)
    log(f"[{anchor.name}] === step 2: shortlist ===")
    for lr_u, lr_theta, why in cells:
        log(f"[{anchor.name}]   lr_U {lr_u:.0e} lr_theta {lr_theta:.0e}  ({why})")

    full = []
    if not skip_full_length:
        log(f"[{anchor.name}] === step 3: {len(cells)} cells at full "
            f"N_inner = {n_inner} with cosine annealing ===")
        for position, (lr_u, lr_theta, why) in enumerate(cells):
            tag = f"lrU{lr_u:.0e}_lrth{lr_theta:.0e}"
            out = C.anchor_dir(anchor) / "profile_runs" / tag
            hp = {"anchor": anchor.name, "batch": batch, "lr_U": lr_u,
                  "lr_theta": lr_theta, "loss_scale": scale,
                  "T_outer": int(t_outer)}
            payload = train_mod.run(anchor, C.ARM_UNIFORM, device, log=log,
                                    hp_override=hp, out_dir=out, tag=why)
            full.append({"lr_U": lr_u, "lr_theta": lr_theta, "reason": why,
                         "tag": tag, "directory": str(out),
                         "val_endpoint": payload["G1"]["val_endpoint"],
                         "val_step0": payload["G1"]["val_step0"],
                         "relative_improvement":
                             payload["G1"]["relative_improvement"],
                         "G1": payload["G1"]["passed"],
                         "G2": payload["G2"]["passed"],
                         "G2_detail": payload["G2"],
                         "G3": payload["G3"]["passed"],
                         "seconds": payload["seconds"]})
            log(f"[{anchor.name}] full-length {tag}: val "
                f"{payload['G1']['val_step0']:.2f} -> "
                f"{payload['G1']['val_endpoint']:.2f}  "
                f"G1 {'PASS' if payload['G1']['passed'] else 'FAIL'}  "
                f"G2 {'PASS' if payload['G2']['passed'] else 'FAIL'}")

        stable = [f for f in full if f["G2"]]
        if not stable:
            summary = {"plan": "v11", "node": "N17", "anchor": anchor.name,
                       "eligible": False,
                       "reason": "no full-length shortlist cell passed G2; this "
                                 "anchor does not enter the two arms (plan 7.2)",
                       "grid": grid, "shortlist": full,
                       "seconds": time.time() - started}
            (C.anchor_dir(anchor) / "profile_failed.json").write_text(
                json.dumps(summary, indent=2))
            raise SystemExit(
                f"[{anchor.name}] no full-length cell passed G2; the anchor "
                f"leaves v11 rather than being run with an unstable schedule")
        chosen = min(stable, key=lambda f: f["val_endpoint"])
    else:
        chosen = {"lr_U": cells[0][0], "lr_theta": cells[0][1],
                  "reason": "skip_full_length: short-range argmin, NOT a valid "
                            "v11 selection", "val_endpoint": None}

    on_edge = [name for name, value, edges in
               (("lr_U", chosen["lr_U"], (C.LR_U_GRID[0], C.LR_U_GRID[-1])),
                ("lr_theta", chosen["lr_theta"],
                 (C.LR_THETA_GRID[0], C.LR_THETA_GRID[-1])))
               if value in edges]
    if on_edge:
        log(f"[{anchor.name}] NOTE: the chosen {', '.join(on_edge)} sits on the "
            f"edge of its grid.  Plan 7.2 step 5: the value is used as measured "
            f"and the grid is NOT widened inside v11 -- widening it now, after "
            f"seeing which cell won, would be choosing a hyper-parameter from "
            f"an outcome.  Recorded so the write-up can say the search was "
            f"bounded here.")

    budget = outer_forward_equivalents(n_cal=int(cal_rows.shape[0]))
    per_step = (1.0 + C.BACKWARD_MULTIPLIER) * batch * 2   # main + auxiliary
    profile = {
        "plan": "v11", "node": "N17", "stage": "profile", "anchor": anchor.name,
        "rate": anchor.rate, "checkpoint_id": meta["checkpoint_id"],
        "loss_scale": scale, "batch": batch,
        "lr_U": chosen["lr_U"], "lr_theta": chosen["lr_theta"],
        "selected_by": chosen["reason"],
        "lr_on_grid_edge": on_edge,
        "T_outer": int(t_outer), "N_inner": int(n_inner),
        "inner_epochs": C.INNER_EPOCHS,
        "steps_per_epoch": int(steps_per_epoch),
        "val_every": C.VAL_EVERY,
        "images_seen": int(n_inner * batch),
        "outer_budget": budget,
        "forward_equivalents": {
            "per_training_step": per_step,
            "note": "2x v10's, because every step runs the main and the "
                    "auxiliary branch; this is the stated cost of the "
                    "menu-coverage fix",
            "inner": float(n_inner * per_step),
            "A2_outer": budget["forward_equivalents"],
            "A2_total": float(n_inner * per_step
                              + budget["forward_equivalents"]),
            "outer_share_of_A2": float(
                budget["forward_equivalents"]
                / (n_inner * per_step + budget["forward_equivalents"])),
            "backward_multiplier": C.BACKWARD_MULTIPLIER,
            "reported_not_compensated": (
                "A2's search FLOPs and cal usage are a method cost, reported "
                "here and not offset with extra uniform training steps; v10's "
                "equal-compute arm is out of the main flow by the plan owner's "
                "ruling"),
        },
        "measured": {
            "batch_probes": batches,
            "train_images_per_second": best_batch["images_per_second"],
            "eval_images_per_second": eval_rate,
            "lr_grid": grid,
            "shortlist": full,
            "estimated_minutes": {
                "A1": n_inner * batch / best_batch["images_per_second"] / 60,
                "A2_inner": n_inner * batch
                            / best_batch["images_per_second"] / 60,
                "A2_outer": budget["forward_equivalents"] / eval_rate / 60,
            },
        },
        "frozen_note": "train.py reads this file and will not start without it; "
                       "editing it after either arm has run is a protocol "
                       "change, not a patch",
        "seconds": time.time() - started,
    }
    C.profile_path(anchor).write_text(json.dumps(profile, indent=2))
    minutes = profile["measured"]["estimated_minutes"]
    log(f"[{anchor.name}] FROZEN: batch {batch}, lr_U {chosen['lr_U']:.0e}, "
        f"lr_theta {chosen['lr_theta']:.0e}, T_outer {t_outer}, "
        f"N_inner {n_inner}  ({chosen['reason']})")
    log(f"[{anchor.name}] estimated: A1 {minutes['A1']:.1f} min, "
        f"A2 {minutes['A2_inner'] + minutes['A2_outer']:.1f} min "
        f"({minutes['A2_inner']:.1f} inner + {minutes['A2_outer']:.1f} outer)")
    return profile


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor", required=True, choices=list(C.ANCHOR_BY_NAME))
    ap.add_argument("--skip-full-length", action="store_true",
                    help="grid and shortlist only; the profile it writes is "
                         "NOT a valid v11 selection and train.py's arms must "
                         "not be run from it")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    anchor = C.ANCHOR_BY_NAME[args.anchor]
    if C.profile_path(anchor).exists() and not args.force:
        raise SystemExit(f"{C.profile_path(anchor)} exists; the "
                         f"hyper-parameters are frozen.  Re-profiling after an "
                         f"arm has run would be choosing them from an outcome.")
    profile = run(anchor, torch.device("cuda"),
                  skip_full_length=args.skip_full_length)
    print(json.dumps({k: v for k, v in profile.items() if k != "measured"},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
