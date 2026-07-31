"""Stage C: the three matched arms (plan v10 section 5).

Per anchor, three runs that differ in exactly one thing each:

    A1  uniform-equal-step     uniform allocation, N_inner inner steps
    A2  nonuniform-joint       same init, same stream, same lr, same N_inner,
                               plus S_OUTER_MAX one-bit swaps chosen on cal-500
    A3  uniform-equal-compute  uniform allocation, N_inner + delta_N steps,
                               delta_N fixed by profiling so the total FLOPs
                               match A2's inner + outer cost

The primary endpoint is D(A2) - D(A1) on holdout-500: A1 is what A2 would have
been without the allocation mechanism, so their difference isolates that
mechanism.  A3 answers the separate question of whether the gain survives giving
the uniform arm the compute A2 spent searching, and is reported as secondary.

Everything that could differ between arms by accident is made a function of the
anchor alone:

* the initial ``(U_0, Theta_0)`` are v9's frozen tensors, checked elementwise by
  ``frozen.load_checked`` (W5);
* the batch stream is ``randperm`` under ``TRAIN_SEED``, which does not depend on
  the arm -- the realised index sequence is written to disk and compared across
  arms by ``verify.py`` with ``array_equal`` (W8);
* the loss scale, learning rates, batch size and step counts all come out of
  ``profile.json``, which is written before any arm runs;
* dead-codeword revival runs on the same schedule under the same deterministic
  rule everywhere.

W8 is *not* stated as bit-identity of the loss trace, and that is a correction to
the plan's wording rather than a relaxation of it.  v9 already measured that two
identical calls to this tail disagree by up to one fp32 ULP because the reduction
order is not reproducible; two processes cannot be made bit-identical on this
stack, and over hundreds of steps an initial ULP compounds. What *is* exactly
checkable is checked exactly: identical initial tensors, identical batch index
streams, identical hyper-parameters, and a step-1 loss agreeing within the fp32
floor.  Divergence after that is arithmetic, not a protocol violation, and
calling it one would make the invariant a formality that always has to be waived.
"""

import argparse
import json
import time

import numpy as np
import torch

from cayley import CayleySGD
from codec_v1 import save_codec_v1

from . import config as C
from . import qhard
from . import search as search_mod
from .. import engine
from .. import frozen
from .. import splits
from .. import tail as tail_mod


# ------------------------------------------------------------- batch stream ---
class BatchStream:
    """Deterministic shuffled index stream over train-core.

    A function of ``(seed, count, batch)`` only.  The realised indices are
    recorded so that "the three arms saw the same images in the same order" is a
    file comparison rather than a claim about two random number generators.
    """

    def __init__(self, count, batch, seed):
        self.count = int(count)
        self.batch = int(batch)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.per_epoch = self.count // self.batch     # drop the short remainder
        if self.per_epoch == 0:
            raise SystemExit(f"INVALID_EXPERIMENT: batch {batch} exceeds the "
                             f"{count}-image train-core split")
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


def train_step(codec, tail, resident, index, modes, opt_rotation, opt_books,
               scale):
    """One inner step.  Returns ``(loss_value, mean_distortion, labels, y)``.

    Everything is fp32, including CayleySGD's own reorthogonalisation, which the
    optimiser fires on its internal schedule.  ``ORTH_TOL`` is set to 1e-4 to
    match what that arithmetic can hold; see ``config.py`` for the measurements.
    """
    y, mu, std, teacher = gather(resident, index)
    value, labels = qhard.distortion(codec, tail, y, mu, std, teacher, modes)
    mean = value.mean()
    loss = mean / scale
    opt_rotation.zero_grad(set_to_none=True)
    opt_books.zero_grad(set_to_none=True)
    loss.backward()
    opt_rotation.step()
    opt_books.step()
    return float(loss.detach()), float(mean.detach()), labels, y


def outer_event(codec, tail, cal, anchor, allocation, log):
    """One outer allocation update: at most one accepted one-bit swap on cal."""
    def evaluate(allocations):
        return engine.evaluate_allocations(
            codec, tail, cal, allocations, image_batch=C.EVAL_IMAGE_BATCH,
            pair_budget=C.EVAL_PAIR_BUDGET, per_image=True)

    return search_mod.greedy_swap_search(
        evaluate, anchor, start=allocation, s_max=C.OUTER_SWAPS_PER_EVENT,
        log=log)


def run(anchor, arm, device, log=print, images=None):
    started = time.time()
    if arm not in C.ARMS:
        raise SystemExit(f"unknown arm {arm}; expected one of {C.ARMS}")
    hp = C.load_profile(anchor)
    if hp["anchor"] != anchor.name:
        raise SystemExit(f"INVALID_EXPERIMENT: {C.profile_path(anchor)} was "
                         f"profiled for {hp['anchor']}, not {anchor.name}")
    out = C.run_dir(anchor, arm)
    out.mkdir(parents=True, exist_ok=True)

    engine.configure_precision(C.ALLOW_TF32)
    codec, meta = frozen.load_checked(anchor, device)            # W5
    orth0 = frozen.orthogonality_error(codec)
    if orth0 > C.ORTH_TOL:                                       # W7 at init
        raise SystemExit(f"INVALID_EXPERIMENT (W7): [{anchor.name}] initial "
                         f"||U^T U - I||_F = {orth0:.3e} > {C.ORTH_TOL:.1e}")
    make_trainable(codec)
    opt_rotation, opt_books = build_optimizers(codec, hp)

    tail = tail_mod.build_tail(C.LAYER, device)
    feature_path, teacher_path, rows, _ = splits.load_split("train_core")
    tail_mod.check_layer(C.LAYER, feature_path, teacher_path,
                         meta["feature_cache"])
    resident = engine.ResidentSet(feature_path, teacher_path, rows, device,
                                  max_images=images)
    log(f"[{anchor.name}/{arm}] train-core resident: {resident.count} images")

    cal = None
    if arm == C.ARM_JOINT:
        cal_feature, cal_teacher, cal_rows, _ = splits.load_split("cal")
        cal = engine.ResidentSet(cal_feature, cal_teacher, cal_rows, device,
                                 max_images=images)
        engine.assert_uniform_shape(cal.count)
        log(f"[{anchor.name}/{arm}] cal resident: {cal.count} images")

    batch = int(hp["batch"])
    t_outer = int(hp["T_outer"])
    total = C.inner_steps(t_outer)
    if arm == C.ARM_EQUAL_COMPUTE:
        total += int(hp["delta_N"])
    scale = float(hp["loss_scale"])
    stream = BatchStream(resident.count, batch, C.TRAIN_SEED)

    allocation = engine.uniform_allocation(anchor)
    modes = torch.from_numpy(allocation).to(device)
    losses = np.zeros(total, dtype=np.float64)
    distortions = np.zeros(total, dtype=np.float64)
    indices = np.zeros((total, batch), dtype=np.int64)
    events, revived, allocation_log = [], 0, []
    orth_peak = orth0

    for step in range(1, total + 1):
        index = stream.next()
        indices[step - 1] = index.numpy()
        # Measured on the step *before* a reorthogonalisation fires, i.e. at the
        # worst point of the cycle: this is the drift the projection has to
        # absorb, and W7's end-of-run check would not see it otherwise.
        if step % C.CAYLEY_REORTH_EVERY == 0:
            orth_peak = max(orth_peak, frozen.orthogonality_error(codec))
        loss, mean, labels, y = train_step(
            codec, tail, resident, index.to(device), modes, opt_rotation,
            opt_books, scale)
        losses[step - 1] = loss
        distortions[step - 1] = mean

        if step % C.REVIVE_EVERY == 0:
            revived += qhard.revive_dead_codewords(codec, y, modes, labels)
        if step % C.LOG_EVERY == 0 or step == 1:
            log(f"[{anchor.name}/{arm}] step {step}/{total}  loss {loss:.6f}  "
                f"D {mean:.2f}  ({time.time() - started:.0f}s)")

        # The outer event fires *after* the block, so A1 and A2 are running the
        # same allocation for the whole of the first T_outer steps.
        if (arm == C.ARM_JOINT and step % t_outer == 0
                and len(events) < C.S_OUTER_MAX):
            result = outer_event(codec, tail, cal, anchor, allocation, log)
            result["at_step"] = step
            events.append(result)
            allocation = np.asarray(result["final_allocation"], dtype=np.int64)
            modes = torch.from_numpy(allocation).to(device)
            allocation_log.append({"step": step,
                                   "allocation": allocation.tolist(),
                                   "accepted": result["accepted_swaps"]})
            if engine.nominal_rate(allocation, anchor) != anchor.rate:   # W6
                raise SystemExit(
                    f"INVALID_EXPERIMENT (W6): [{anchor.name}/{arm}] the "
                    f"allocation after step {step} has nominal rate "
                    f"{engine.nominal_rate(allocation, anchor)}, not "
                    f"{anchor.rate}")
            log(f"[{anchor.name}/{arm}] outer event {len(events)} at step "
                f"{step}: {result['accepted_swaps']} swap(s), allocation "
                f"{allocation.tolist()}")

    orth1 = frozen.orthogonality_error(codec)
    orth_peak = max(orth_peak, orth1)
    if orth_peak > C.ORTH_TOL:                                   # W7
        raise SystemExit(f"INVALID_EXPERIMENT (W7): [{anchor.name}/{arm}] "
                         f"||U^T U - I||_F reached {orth_peak:.3e} > "
                         f"{C.ORTH_TOL:.1e} (final {orth1:.3e})")

    save_codec_v1(codec, out / "codec.pt")
    reference = {"U": codec.transform.get_rotation().detach().cpu().numpy()}
    for mode, quantizer in enumerate(codec.pq.quantizers):
        reference[f"codebook_{mode}"] = \
            quantizer.codebooks.detach().cpu().numpy()
    np.savez(out / "codec_ref.npz", **reference)
    np.save(out / "losses.npy", losses)
    np.save(out / "distortions.npy", distortions)
    np.save(out / "batch_indices.npy", indices)
    np.save(out / "allocation.npy", allocation)

    window = min(C.LOG_EVERY, total)
    start_level = float(distortions[:window].mean())
    end_level = float(distortions[-window:].mean())
    improvement = (start_level - end_level) / start_level
    gate = C.G1_TRAIN_IMPROVE_MULT * C.REPLAY_REL_TOL
    payload = {
        "plan": "v10", "stage": "C_train", "anchor": anchor.name,
        "arm": arm, "rate": anchor.rate,
        "checkpoint_id": meta["checkpoint_id"],
        "hyper_parameters": hp,
        "steps": int(total), "batch": batch,
        "images_seen": int(total * batch),
        "train_images": int(resident.count),
        "loss_scale": scale,
        "train_distortion_first": start_level,
        "train_distortion_last": end_level,
        "train_relative_improvement": improvement,
        "G1_threshold": gate,
        "G1_pass": bool(improvement > gate),
        "orthogonality_error_initial": orth0,
        "orthogonality_error_final": orth1,
        "orthogonality_error_peak": orth_peak,
        "orthogonality_tolerance": C.ORTH_TOL,
        "codewords_revived": int(revived),
        "final_allocation": allocation.tolist(),
        "nominal_rate": engine.nominal_rate(allocation, anchor),
        "outer_events": events,
        "allocation_history": allocation_log,
        "outer_seconds": sum(e["seconds"] for e in events),
        "seconds": time.time() - started,
    }
    (out / "train.json").write_text(json.dumps(payload, indent=2))
    log(f"[{anchor.name}/{arm}] done in {payload['seconds'] / 60:.1f} min  "
        f"train D {start_level:.1f} -> {end_level:.1f} "
        f"({100 * improvement:+.4f}%)  G1 "
        f"{'PASS' if payload['G1_pass'] else 'FAIL'}  "
        f"allocation {allocation.tolist()}")
    if not payload["G1_pass"]:
        log(f"[{anchor.name}/{arm}] G1 FAILED: the hard straight-through "
            f"gradient did not move the training objective by more than "
            f"{gate:.2e} relative.  Every arm's endpoint would then be the "
            f"frozen codec under a different name.")
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
        raise SystemExit(f"{output} exists; re-running an arm after seeing its "
                         f"result is a selection. --force is a protocol change.")
    result = run(anchor, args.arm, torch.device("cuda"), images=args.images)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("outer_events", "allocation_history")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
