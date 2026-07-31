"""N18: the two arms (plan v11 sections 6, 7 and 8).

Per anchor, two runs differing in exactly one thing:

    A1  uniform-joint      m == uniform for the whole run, no outer loop
    A2  nonuniform-joint   same init, same batch stream, same learning-rate
                           schedule, same auxiliary stream; one full hard
                           one-bit search on cal-500 every T_outer steps

Everything else is a function of the anchor alone, so "the only essential
difference is that A2 may change m_t" is a construction rather than a claim:

* the initial ``(U_0, Theta)`` are the N15 menu codec, checked elementwise;
* the batch stream is ``randperm`` under ``TRAIN_SEED``, which does not depend
  on the arm, and the realised index sequence is written to disk so the claim is
  a file comparison;
* the auxiliary allocation at step ``t`` is ``cycle[t % 3]``, with the cycle
  redrawn every three steps from a generator seeded by ``MENU_PERMUTATION_SEED``
  -- also arm-independent, also written to disk;
* the learning-rate schedule is a closed-form function of ``(step, N_inner)``.

The two v10 defects this module exists to fix:

**Zero gradient on the side modes.**  Under a uniform allocation only mode 1's
quantizer is selected, so in v10 modes 0 and 2 received an exactly zero gradient
-- codebook_0 and codebook_2 were byte-identical after 3096 steps while U drifted
8.6% / 11.1%.  The outer search was then choosing between one trained codebook
and two stale ones.  v11 adds the deterministic menu-coverage auxiliary

    L = D(U, Theta, m_t) + beta * D(sg(U), Theta, a_s),   beta = 1

with ``a_s`` cycling through three allocations that cover every ``(g, mode)``
exactly once per cycle.  ``sg`` on the rotation is what keeps this honest: the
auxiliary's gradient with respect to U is exactly zero (verified by N15's
AUX-SG at 0.0), so it calibrates codebooks without competing for U's update
direction.  The cost is honest and stated: two tail forwards and two backwards
per step, so the inner loop is about 2x v10's.

**No annealing.**  v10 selected ``lr_theta = 1e+1`` from a 200-step probe and ran
3096 steps with it; the trajectory bottomed at step ~352 and rose monotonically
after.  v11 anneals both learning rates as ``base * (1 + cos(pi t / N)) / 2``,
which makes the endpoint the best point by construction -- which is in turn what
lets G2 demand ``argmin >= 0.9 N`` rather than quietly rescuing a mid-run
checkpoint.

G1 and G2 read a train-val trajectory sampled every ``VAL_EVERY`` steps; G3's
mechanical half is the auxiliary coverage count, and its substantive half is the
stale-codebook probe fired before each outer event.  G4 lives in ``verify.py``,
because it is a statement about dev across both arms rather than about one run.
"""

import argparse
import json
import time

import numpy as np
import torch

from cayley import CayleySGD
from codec_v1 import save_codec_v1

from . import config as C
from . import init_menu
from . import qhard
from . import search as search_mod
from .. import engine
from .. import frozen
from .. import tail as tail_mod


class MenuStream:
    """The auxiliary allocation stream: a fresh permutation every three steps.

    Plan 6.2 says the permutation is *regenerated each cycle from the frozen
    seed*, not drawn once and reused -- one fixed permutation would pair the same
    16 groups with mode 0 for the whole run, so "every (g, mode) is covered" would
    be true while the codebooks still only ever saw half the groups in each side
    mode.  Redrawing decorrelates group from mode over the run at no cost to the
    per-cycle coverage guarantee, which holds for any permutation.

    A function of ``(seed, anchor)`` alone, so both arms see the same stream; the
    realised permutations are written to disk so that is a file comparison.
    """

    def __init__(self, anchor, seed=C.MENU_PERMUTATION_SEED):
        self.anchor = anchor
        self.rng = np.random.default_rng(int(seed))
        self.permutations = []
        self.cycle = None

    def allocation(self, step):
        """``a_s`` for one-based training ``step``."""
        slot = (step - 1) % C.MENU_CYCLE
        if slot == 0:
            permutation = self.rng.permutation(C.GROUPS)
            self.permutations.append(permutation)
            self.cycle = C.menu_cycle_allocations(self.anchor, permutation)
        return slot, self.cycle[slot]



# ------------------------------------------------------------- batch stream ---
class BatchStream:
    """Deterministic shuffled index stream over train-fit.

    A function of ``(seed, count, batch)`` only.  The realised indices are
    recorded so that "the two arms saw the same images in the same order" is a
    file comparison rather than a claim about two random number generators.
    """

    def __init__(self, count, batch, seed):
        self.count = int(count)
        self.batch = int(batch)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.per_epoch = self.count // self.batch     # drop the short remainder
        if self.per_epoch == 0:
            raise SystemExit(f"INVALID_EXPERIMENT: batch {batch} exceeds the "
                             f"{count}-image train-fit split")
        self.order = None
        self.cursor = self.per_epoch                  # forces a reshuffle

    def next(self):
        if self.cursor >= self.per_epoch:
            self.order = torch.randperm(self.count, generator=self.generator)
            self.cursor = 0
        start = self.cursor * self.batch
        self.cursor += 1
        return self.order[start:start + self.batch]


def gather(resident, index):
    """Index-select out of a ResidentSet without touching v9's engine."""
    return (resident.y[index], resident.mu[index],
            resident.std[index], resident.teacher[index])


# ------------------------------------------------------------------ the run ---
def make_trainable(codec):
    codec.transform.rotation.requires_grad_(True)
    for quantizer in codec.pq.quantizers:
        quantizer.codebooks.requires_grad_(True)
    return codec


def build_optimizers(codec, hp):
    rotation = CayleySGD([codec.transform.rotation], lr=float(hp["lr_U"]),
                         fixed_point_iterations=C.CAYLEY_FIXED_POINT_ITERATIONS,
                         reorthogonalize_every=C.CAYLEY_REORTH_EVERY)
    books = torch.optim.SGD([q.codebooks for q in codec.pq.quantizers],
                            lr=float(hp["lr_theta"]),
                            momentum=C.CODEBOOK_MOMENTUM)
    return rotation, books


def set_learning_rates(opt_rotation, opt_books, step, total, hp):
    """Write the annealed rates into the param groups before ``step()``.

    CayleySGD reads ``group["lr"]`` inside its own ``step``, so this is where
    the schedule takes effect; its ``min(lr, 1/||skew||)`` trust region applies
    on top and only ever makes the step smaller.
    """
    lr_u = C.cosine_lr(step, total, hp["lr_U"])
    lr_theta = C.cosine_lr(step, total, hp["lr_theta"])
    for group in opt_rotation.param_groups:
        group["lr"] = lr_u
    for group in opt_books.param_groups:
        group["lr"] = lr_theta
    return lr_u, lr_theta


def train_step(codec, tail, resident, index, modes, aux_modes, opt_rotation,
               opt_books, scale):
    """One inner step of the joint objective.

    Returns ``(loss, main_mean, aux_mean, main_labels, aux_labels, y)``.  The
    auxiliary branch sees a detached rotation, so ``dL_aux/dU == 0`` exactly --
    N15's AUX-SG measured that gradient norm at 0.0 rather than arguing it from
    the presence of a ``.detach()``.
    """
    y, mu, std, teacher = gather(resident, index)
    loss, main, aux, main_labels, aux_labels = qhard.joint_loss(
        codec, tail, y, mu, std, teacher, modes, aux_modes, beta=C.AUX_BETA)
    loss = loss / scale
    opt_rotation.zero_grad(set_to_none=True)
    opt_books.zero_grad(set_to_none=True)
    loss.backward()
    opt_rotation.step()
    opt_books.step()
    return (float(loss.detach()), float(main.mean().detach()),
            float(aux.mean().detach()), main_labels, aux_labels, y)


def validate(codec, tail, val, anchor, allocation):
    """Mean per-image hard tail distortion on train-val at ``allocation``."""
    matrix = engine.evaluate_allocations(
        codec, tail, val, np.asarray(allocation, dtype=np.int64)[None, :],
        image_batch=C.EVAL_IMAGE_BATCH, pair_budget=C.EVAL_PAIR_BUDGET,
        per_image=True)
    return float(matrix[0].mean())


def menu_probe(codec, tail, val, cycle):
    """Mean train-val distortion over the three menu allocations.

    This is G3's substantive half.  It is deliberately evaluated at *all three*
    cycle allocations rather than at ``m_t``: the question is whether the menu
    -- the thing the outer search will choose among -- has kept up with the
    current U, and ``m_t`` alone would only report on the mode the main branch
    has been training anyway.
    """
    matrix = engine.evaluate_allocations(
        codec, tail, val, np.asarray(cycle, dtype=np.int64),
        image_batch=C.EVAL_IMAGE_BATCH, pair_budget=C.EVAL_PAIR_BUDGET,
        per_image=True)
    return float(matrix.mean()), [float(row.mean()) for row in matrix]


def stale_codebooks(codec):
    """A detached copy of every codebook, for G3's stale/fresh comparison."""
    return [q.codebooks.detach().clone() for q in codec.pq.quantizers]


def g3_stale_check(codec, tail, val, probe_cycle, stale, log, anchor, step):
    """With U_t fixed, are the fresh codebooks at least as good as the stale?

    The comparison swaps the codebook tensors in place and restores them, so it
    is the *same* codec object and the same U for both readings -- which is the
    whole content of "fixed the current U_t".
    """
    fresh_mean, fresh_rows = menu_probe(codec, tail, val, cycle)
    current = [q.codebooks.data for q in codec.pq.quantizers]
    for quantizer, saved in zip(codec.pq.quantizers, stale):
        quantizer.codebooks.data = saved
    stale_mean, stale_rows = menu_probe(codec, tail, val, cycle)
    for quantizer, saved in zip(codec.pq.quantizers, current):
        quantizer.codebooks.data = saved

    budget = stale_mean + C.G3_STALE_MULT * C.REPLAY_REL_TOL * stale_mean
    passed = bool(fresh_mean <= budget)
    report = {"at_step": int(step), "fresh": fresh_mean, "stale": stale_mean,
              "fresh_per_allocation": fresh_rows,
              "stale_per_allocation": stale_rows,
              "budget": budget, "margin": budget - fresh_mean,
              "relative_change": (fresh_mean - stale_mean) / stale_mean,
              "passed": passed}
    log(f"[{anchor.name}] G3 stale probe at step {step}: fresh {fresh_mean:.2f} "
        f"vs stale {stale_mean:.2f} "
        f"({100 * report['relative_change']:+.4f}%)  "
        f"{'ok' if passed else 'FAIL'}")
    return report


def outer_event(codec, tail, cal, anchor, allocation, log):
    """One outer allocation update: at most one accepted one-bit swap on cal."""
    def evaluate(allocations):
        return engine.evaluate_allocations(
            codec, tail, cal, allocations, image_batch=C.EVAL_IMAGE_BATCH,
            pair_budget=C.EVAL_PAIR_BUDGET, per_image=True)

    return search_mod.greedy_swap_search(
        evaluate, anchor, start=allocation, s_max=C.OUTER_SWAPS_PER_EVENT,
        log=log)


def _smooth(values, window):
    """Trailing mean of ``window`` points, defined from the first point on."""
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    for i in range(values.size):
        out[i] = values[max(0, i - window + 1):i + 1].mean()
    return out


def evaluate_gates(val_curve, payload):
    """G1 and G2, both read off the train-val trajectory."""
    curve = np.asarray(val_curve, dtype=np.float64)
    start = float(curve[0])
    tail_points = min(C.VAL_TAIL_POINTS, curve.size)
    end = float(curve[-tail_points:].mean())
    improvement = (start - end) / start
    g1_threshold = C.G1_IMPROVE_MULT * C.REPLAY_REL_TOL

    smoothed = _smooth(curve, min(C.SMOOTH_WINDOW, curve.size))
    argmin = int(np.argmin(smoothed))
    fraction = argmin / max(curve.size - 1, 1)
    minimum = float(smoothed.min())
    endpoint = float(smoothed[-1])

    payload["G1"] = {
        "val_step0": start, "val_endpoint": end,
        "relative_improvement": improvement,
        "threshold": g1_threshold,
        "tail_points": int(tail_points),
        "passed": bool(improvement > g1_threshold),
    }
    payload["G2"] = {
        "argmin_index": argmin, "points": int(curve.size),
        "argmin_fraction": fraction,
        "argmin_fraction_required": C.G2_ARGMIN_FRACTION,
        "smoothed_min": minimum, "smoothed_endpoint": endpoint,
        "endpoint_over_min": endpoint / minimum if minimum else float("inf"),
        "endpoint_tolerance": C.G2_END_TOLERANCE,
        "passed": bool(fraction >= C.G2_ARGMIN_FRACTION
                       and endpoint <= C.G2_END_TOLERANCE * minimum),
    }
    return payload


def run(anchor, arm, device, log=print, images=None, steps_override=None,
        hp_override=None, out_dir=None, tag=None):
    """Train one arm.  ``hp_override``/``steps_override`` serve N17 only.

    N17 calls this with an explicit hyper-parameter dict and its own output
    directory so that the full-length shortlist runs are the *same* code path
    the arms will take -- a profiling run that differed from the real one would
    be profiling something else.  Neither override is reachable from the N18
    command line.
    """
    started = time.time()
    if arm not in C.ARMS:
        raise SystemExit(f"unknown arm {arm}; expected one of {C.ARMS}")
    hp = dict(hp_override) if hp_override is not None else C.load_profile(anchor)
    if hp_override is None and hp["anchor"] != anchor.name:
        raise SystemExit(f"INVALID_EXPERIMENT: {C.profile_path(anchor)} was "
                         f"profiled for {hp['anchor']}, not {anchor.name}")
    out = out_dir if out_dir is not None else C.run_dir(anchor, arm)
    out.mkdir(parents=True, exist_ok=True)

    engine.configure_precision(C.ALLOW_TF32)
    codec, meta = init_menu.load_checked(anchor, device)          # W-I1/W-I2
    orth0 = frozen.orthogonality_error(codec)
    make_trainable(codec)
    opt_rotation, opt_books = build_optimizers(codec, hp)

    tail = tail_mod.build_tail(C.LAYER, device)
    feature_path, teacher_path, rows, _ = C.load_split("train_fit")
    tail_mod.check_layer(C.LAYER, feature_path, teacher_path,
                         meta["feature_cache"])
    resident = engine.ResidentSet(feature_path, teacher_path, rows, device,
                                  max_images=images)
    val_feature, val_teacher, val_rows, _ = C.load_split("train_val")
    val = engine.ResidentSet(val_feature, val_teacher, val_rows, device,
                             max_images=images)
    log(f"[{anchor.name}/{arm}] train-fit {resident.count} images, "
        f"train-val {val.count} images")

    cal = None
    if arm == C.ARM_NONUNIFORM:
        cal_feature, cal_teacher, cal_rows, _ = C.load_split("cal")
        cal = engine.ResidentSet(cal_feature, cal_teacher, cal_rows, device,
                                 max_images=images)
        engine.assert_uniform_shape(cal.count)
        log(f"[{anchor.name}/{arm}] cal resident: {cal.count} images")

    batch = int(hp["batch"])
    t_outer = int(hp["T_outer"])
    total = int(steps_override) if steps_override else C.inner_steps(t_outer)
    scale = float(hp["loss_scale"])
    stream = BatchStream(resident.count, batch, C.TRAIN_SEED)
    menu = MenuStream(anchor)
    # G3's substantive probe uses the *frozen* N15 cycle, not the live stream:
    # a probe that moved with the thing it measures would compare two different
    # questions at the two ends of the comparison.
    probe_cycle = C.menu_cycle_allocations(
        anchor, np.asarray(meta["menu_cycle"]["permutation"], dtype=np.int64))

    allocation = engine.uniform_allocation(anchor)
    modes = torch.from_numpy(allocation).to(device)
    losses = np.zeros(total, dtype=np.float64)
    main_trace = np.zeros(total, dtype=np.float64)
    aux_trace = np.zeros(total, dtype=np.float64)
    lr_trace = np.zeros((total, 2), dtype=np.float64)
    indices = np.zeros((total, batch), dtype=np.int64)
    aux_index = np.zeros(total, dtype=np.int64)
    coverage = np.zeros((C.GROUPS, len(anchor.mode_sizes)), dtype=np.int64)
    val_steps, val_curve = [], []
    events, revived, allocation_log, stale_reports = [], 0, [], []
    orth_peak = orth0
    stale = stale_codebooks(codec)

    # Step 0 of the train-val trajectory, before any update: G1 measures against
    # this, so it must be the arm's own starting point, not A0's recorded number.
    val_steps.append(0)
    val_curve.append(validate(codec, tail, val, anchor, allocation))
    log(f"[{anchor.name}/{arm}] train-val at step 0: {val_curve[0]:.2f}")

    for step in range(1, total + 1):
        index = stream.next()
        indices[step - 1] = index.numpy()
        slot, aux_allocation = menu.allocation(step)
        aux_index[step - 1] = slot
        aux_modes = torch.from_numpy(aux_allocation).to(device)
        coverage[np.arange(C.GROUPS), aux_allocation] += 1

        if step % C.CAYLEY_REORTH_EVERY == 0:
            orth_peak = max(orth_peak, frozen.orthogonality_error(codec))
        lr_u, lr_theta = set_learning_rates(opt_rotation, opt_books, step - 1,
                                            total, hp)
        lr_trace[step - 1] = (lr_u, lr_theta)
        loss, main, aux, main_labels, aux_labels, y = train_step(
            codec, tail, resident, index.to(device), modes, aux_modes,
            opt_rotation, opt_books, scale)
        losses[step - 1] = loss
        main_trace[step - 1] = main
        aux_trace[step - 1] = aux

        if step % C.REVIVE_EVERY == 0:
            # Revival covers m_t union a_s: the auxiliary's dead codewords are
            # the side modes' dead codewords, which is exactly the population
            # v10 left to rot.
            revived += qhard.revive_dead_codewords(codec, y, modes, main_labels)
            revived += qhard.revive_dead_codewords(codec, y, aux_modes,
                                                   aux_labels)
        if step % C.VAL_EVERY == 0:
            val_steps.append(step)
            val_curve.append(validate(codec, tail, val, anchor, allocation))
        if step % C.LOG_EVERY == 0 or step == 1:
            log(f"[{anchor.name}/{arm}] step {step}/{total}  loss {loss:.6f}  "
                f"D {main:.2f}  aux {aux:.2f}  lr_U {lr_u:.2e} "
                f"lr_th {lr_theta:.2e}  val "
                f"{val_curve[-1]:.2f}  ({time.time() - started:.0f}s)")

        if step % t_outer == 0 and len(events) < C.S_OUTER_MAX:
            # G3's stale probe fires in BOTH arms, at the same steps, so the
            # menu-alignment evidence does not depend on which arm produced it.
            stale_reports.append(g3_stale_check(codec, tail, val, probe_cycle, stale,
                                                log, anchor, step))
            stale = stale_codebooks(codec)
            if arm == C.ARM_NONUNIFORM:
                result = outer_event(codec, tail, cal, anchor, allocation, log)
                result["at_step"] = step
                events.append(result)
                allocation = np.asarray(result["final_allocation"],
                                        dtype=np.int64)
                modes = torch.from_numpy(allocation).to(device)
                allocation_log.append({"step": step,
                                       "allocation": allocation.tolist(),
                                       "accepted": result["accepted_swaps"]})
                if engine.nominal_rate(allocation, anchor) != anchor.rate:
                    raise SystemExit(
                        f"INVALID_EXPERIMENT: [{anchor.name}/{arm}] the "
                        f"allocation after step {step} has nominal rate "
                        f"{engine.nominal_rate(allocation, anchor)}, not "
                        f"{anchor.rate}")
                log(f"[{anchor.name}/{arm}] outer event {len(events)} at step "
                    f"{step}: {result['accepted_swaps']} swap(s), allocation "
                    f"{allocation.tolist()}")

    if total % C.VAL_EVERY != 0:
        val_steps.append(total)
        val_curve.append(validate(codec, tail, val, anchor, allocation))

    orth1 = frozen.orthogonality_error(codec)
    orth_peak = max(orth_peak, orth1)
    drift = orth_peak - orth0                      # W-I5 as restated in N14 s3
    if drift > C.ORTH_TOL:
        raise SystemExit(f"INVALID_EXPERIMENT: [{anchor.name}/{arm}] training "
                         f"drove ||U^T U - I||_F from {orth0:.3e} to "
                         f"{orth_peak:.3e}, a drift of {drift:.3e} > "
                         f"{C.ORTH_TOL:.1e}")

    save_codec_v1(codec, out / "codec.pt")
    reference = {"U": codec.transform.get_rotation().detach().cpu().numpy()}
    for mode, quantizer in enumerate(codec.pq.quantizers):
        reference[f"codebook_{mode}"] = \
            quantizer.codebooks.detach().cpu().numpy()
    np.savez(out / "codec_ref.npz", **reference)
    np.save(out / "losses.npy", losses)
    np.save(out / "distortions.npy", main_trace)
    np.save(out / "aux_distortions.npy", aux_trace)
    np.save(out / "learning_rates.npy", lr_trace)
    np.save(out / "batch_indices.npy", indices)
    np.save(out / "aux_slots.npy", aux_index)
    np.save(out / "aux_permutations.npy",
            np.asarray(menu.permutations, dtype=np.int64))
    np.save(out / "val_curve.npy", np.asarray(val_curve, dtype=np.float64))
    np.save(out / "val_steps.npy", np.asarray(val_steps, dtype=np.int64))
    np.save(out / "allocation.npy", allocation)

    cycles = total // C.MENU_CYCLE
    coverage_ok = bool(cycles == 0
                       or (coverage.min() == cycles and coverage.max() == cycles)
                       or (total % C.MENU_CYCLE == 0
                           and coverage.min() == coverage.max()))
    payload = {
        "plan": "v11", "node": "N18", "stage": "train",
        "anchor": anchor.name, "arm": arm, "rate": anchor.rate,
        "checkpoint_id": meta["checkpoint_id"],
        "A0": meta["A0"], "tag": tag,
        "hyper_parameters": hp,
        "steps": int(total), "batch": batch,
        "images_seen": int(total * batch),
        "train_images": int(resident.count), "val_images": int(val.count),
        "loss_scale": scale,
        "schedule": "cosine, base * (1 + cos(pi t / N)) / 2",
        "val_every": C.VAL_EVERY,
        "train_distortion_first": float(main_trace[:C.LOG_EVERY].mean()),
        "train_distortion_last": float(main_trace[-C.LOG_EVERY:].mean()),
        "aux_distortion_first": float(aux_trace[:C.LOG_EVERY].mean()),
        "aux_distortion_last": float(aux_trace[-C.LOG_EVERY:].mean()),
        "orthogonality_error_initial": orth0,
        "orthogonality_error_final": orth1,
        "orthogonality_error_peak": orth_peak,
        "orthogonality_drift": drift,
        "orthogonality_drift_tolerance": C.ORTH_TOL,
        "codewords_revived": int(revived),
        "final_allocation": allocation.tolist(),
        "nominal_rate": int(engine.nominal_rate(allocation, anchor)),
        "outer_events": events,
        "allocation_history": allocation_log,
        "accepted_swaps": int(sum(e["accepted_swaps"] for e in events)),
        "outer_seconds": float(sum(e["seconds"] for e in events)),
        "G3": {
            "coverage_per_cycle_min": int(coverage.min() // max(cycles, 1)),
            "coverage_per_cycle_max": int(coverage.max() // max(cycles, 1)),
            "complete_cycles": int(cycles),
            "mechanical_passed": coverage_ok,
            "stale_probes": stale_reports,
            "substantive_passed": bool(all(r["passed"]
                                           for r in stale_reports)) if
                                  stale_reports else None,
        },
        "seconds": time.time() - started,
    }
    evaluate_gates(val_curve, payload)
    payload["G3"]["passed"] = bool(
        payload["G3"]["mechanical_passed"]
        and (payload["G3"]["substantive_passed"] is not False))
    (out / "train.json").write_text(json.dumps(payload, indent=2))

    log(f"[{anchor.name}/{arm}] done in {payload['seconds'] / 60:.1f} min  "
        f"val {payload['G1']['val_step0']:.2f} -> "
        f"{payload['G1']['val_endpoint']:.2f} "
        f"({100 * payload['G1']['relative_improvement']:+.4f}%)  "
        f"G1 {'PASS' if payload['G1']['passed'] else 'FAIL'}  "
        f"G2 {'PASS' if payload['G2']['passed'] else 'FAIL'}  "
        f"G3 {'PASS' if payload['G3']['passed'] else 'FAIL'}  "
        f"swaps {payload['accepted_swaps']}  "
        f"allocation {allocation.tolist()}")
    if arm == C.ARM_NONUNIFORM and payload["accepted_swaps"] == 0:
        log(f"[{anchor.name}/{arm}] {C.ZERO_SWAP_CLAUSE}")
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor", required=True, choices=list(C.ANCHOR_BY_NAME))
    ap.add_argument("--arm", required=True, choices=list(C.ARMS))
    ap.add_argument("--images", type=int, default=None, help="debug only")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    anchor = C.ANCHOR_BY_NAME[args.anchor]
    output = C.run_dir(anchor, args.arm) / "train.json"
    if output.exists() and not args.force:
        raise SystemExit(f"{output} exists; this arm has already run.  "
                         f"Re-running it and keeping the better result would "
                         f"be selecting an outcome.")
    payload = run(anchor, args.arm, torch.device("cuda"), images=args.images)
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("outer_events", "hyper_parameters")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
