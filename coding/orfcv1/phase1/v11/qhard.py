"""The hard STE forward, plus v11's fix for the zero-gradient modes.

The numerics are v10's, imported rather than copied.  ``quantise_hard_ste``,
``distortion``, ``dead_codeword_counts`` and ``revive_dead_codewords`` are the
same objects v10 ran, so nothing about the forward value, the straight-through
path or the bit-exact agreement with ``engine.build_bank`` is re-litigated here.
Read ``phase1/v10/qhard.py`` for why the forward is written the way it is.

What v11 adds is the auxiliary term that fixes N12's first defect.

Under a uniform allocation only one of the three quantizers is ever selected, so
the other two receive an *exactly* zero gradient -- not a small one.  In v10 that
ran for 3096 steps: ``codebook_0`` and ``codebook_2`` came out byte-identical to
their initialisation while ``U`` had turned 8.6% / 11.1% and ``codebook_1`` had
moved 12-15%.  The outer search was then asked to evaluate swaps onto codebooks
fitted to a rotation that no longer existed, and it accepted zero swaps out of
sixteen events -- with the first-step gain having flipped sign from +1.65% on the
frozen codec to -0.70% after 344 steps of training.  The search was not failing
to find a gain; it was searching a hole the training had dug for it.

The fix is a second forward on a *hypothetical* allocation:

    L = D(U, Theta, m_t) + beta * D(sg(U), Theta, a_s)          beta = 1

with ``a_s`` cycling deterministically through three allocations that, over each
three-step period, touch every ``(group, mode)`` cell exactly once.  Two
properties are worth being explicit about, because both are load-bearing:

*``U`` is stop-gradiented in the auxiliary branch.*  Its gradient there is
identically zero, so the auxiliary cannot pull the rotation anywhere: the main
branch keeps sole authority over ``U``'s update direction, and the auxiliary only
keeps the unselected codebooks following it.  Without ``sg`` the two branches
would negotiate over ``U`` and the arm would no longer be minimising ``D`` at
``m_t``.

*The auxiliary is not a claim about the inner optimum.*  It is not "the exact
value the candidate allocation would reach if trained"; it is a mechanism that
stops the candidate codebooks going stale.  The outer search still measures every
candidate for real, on cal, at the current parameters.

Each auxiliary allocation's mode indices sum to 32, so its nominal rate is
conserved exactly, in integers -- see ``config.menu_cycle_allocations``.
"""

import torch

from ..v10.qhard import (            # noqa: F401  -- re-exported unchanged
    quantise_hard_ste,
    distortion,
    dead_codeword_counts,
    revive_dead_codewords,
    selfcheck,
)

from . import config as C


class _DetachedRotation:
    """``codec.transform`` with ``get_rotation`` stop-gradiented.

    A shim rather than a flag on the real transform: the auxiliary must reuse
    v10's forward *exactly*, and the cleanest way to guarantee that is to change
    nothing about the function and hand it a codec whose rotation happens to
    arrive detached.  Everything else -- the cdist, the argmin, the gather, the
    ``z_hat + (z - z.detach())`` parenthesisation -- is then literally the same
    code path as the main branch.
    """

    __slots__ = ("_transform",)

    def __init__(self, transform):
        self._transform = transform

    def get_rotation(self):
        return self._transform.get_rotation().detach()

    def __getattr__(self, name):
        return getattr(self._transform, name)


class _DetachedCodec:
    """``codec`` with ``sg(U)``; the codebooks keep their gradients."""

    __slots__ = ("pq", "transform", "_codec")

    def __init__(self, codec):
        self._codec = codec
        self.pq = codec.pq
        self.transform = _DetachedRotation(codec.transform)


def auxiliary_allocation(step, cycle):
    """Which of the three menu-coverage allocations this step uses.

    ``cycle`` is ``[3, G]`` from ``config.menu_cycle_allocations``.  The index is
    a pure function of the step, so the auxiliary stream is identical in both
    arms without any state to synchronise or any seed to match.
    """
    return cycle[int(step) % C.MENU_CYCLE]


def coverage_counts(cycle, groups=C.GROUPS, modes=3):
    """``[G, M]`` count of auxiliary visits per cell over one full cycle.

    G3's mechanical half: every entry must be exactly 1.  Returned rather than
    asserted so the caller can record the matrix and fail with the numbers.
    """
    counts = torch.zeros(groups, modes, dtype=torch.long)
    for allocation in cycle:
        for group, mode in enumerate(allocation):
            counts[int(group), int(mode)] += 1
    return counts


def joint_loss(codec, tail, y, mu, std, teacher, modes, aux_modes,
               beta=C.AUX_BETA):
    """``D(U, Theta, m) + beta * D(sg(U), Theta, a)``, per-image sums.

    Both branches are run as one ``2 x batch`` call would be were the tail
    stateless; they are kept as two calls because the tail's normalisation
    statistics ``(mu, std)`` are per-image and shared, so there is nothing to
    gain from concatenating and something to get wrong.

    Returns ``(loss, main_per_image, aux_per_image, main_labels, aux_labels)``.
    ``main_per_image`` is the quantity G1/G2 are computed on -- the auxiliary is
    never reported as the training distortion, because it is a different
    allocation and reporting it would mix an optimisation device into the metric.
    """
    main, main_labels = distortion(codec, tail, y, mu, std, teacher, modes)
    aux, aux_labels = distortion(_DetachedCodec(codec), tail, y, mu, std,
                                 teacher, aux_modes)
    loss = main.mean() + float(beta) * aux.mean()
    return loss, main, aux, main_labels, aux_labels


@torch.no_grad()
def assert_rotation_gradient_free(codec, tail, y, mu, std, teacher, aux_modes):
    """Checked, not asserted in prose: ``dD_aux/dU`` is exactly zero.

    Runs the auxiliary branch alone, backpropagates, and reports the rotation's
    gradient norm.  The claim that the auxiliary cannot steer ``U`` is the reason
    the two arms remain comparable, so it is measured once at N15 rather than
    argued from the presence of a ``.detach()``.
    """
    with torch.enable_grad():
        rotation = codec.transform.rotation
        previous = rotation.grad
        rotation.grad = None
        aux, _ = distortion(_DetachedCodec(codec), tail, y, mu, std, teacher,
                            aux_modes)
        aux.mean().backward()
        norm = 0.0 if rotation.grad is None else float(rotation.grad.norm())
        rotation.grad = previous
    return norm
