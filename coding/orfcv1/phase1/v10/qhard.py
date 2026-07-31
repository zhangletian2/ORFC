"""The hard straight-through quantised forward -- v10's only new numerics.

v8 died at its own M3 gate, and the chain that got it there was: menu gate ->
soft pretraining -> a temperature -> a soft/hard proxy gap.  v10 cuts that chain
at the root by making the training forward *be* the evaluation forward:

    z     = y @ U                                    y = per-image normalised
    c_g   = codebook_{m_g}[ argmin_k ||z_g - c_k||^2 ]      hard nearest neighbour
    z_hat = c + z - z.detach()                       forward value is exactly c
    x_hat = inv_normalize(z_hat @ U^T)
    loss  = mean_n || tail(x_hat_n) - teacher_n ||^2

The one line that does the work is ``z_hat = c + z - z.detach()``.  Its forward
value is ``c`` to the last bit, so nothing about the reported number changes,
while backward gives two gradients at once:

    d z_hat / d z = I      the straight-through path, so ``U`` gets a gradient;
    d z_hat / d c = 1      so the *selected* codewords get a task gradient.

There is no temperature, no anneal, no soft branch, and therefore no proxy: the
quantity being minimised is the quantity that will be reported.  It is also
self-diagnosing -- if the straight-through gradient were useless the hard train
loss simply would not fall, which is gate G1 and needs no separate probe.

``soft_pq.SoftPQ._quantise`` cannot be used for this.  Its straight-through path
is gated on ``self.training and self.temperature > 0`` and is a softmax-ST; at
``temperature == 0`` it degenerates to a plain ``gather`` and ``U`` receives no
gradient at all.  v10 does not modify that file -- it writes the forward here
and borrows the codebook and rotation tensors out of the existing codec.

Bit-for-bit agreement with the evaluator is not a hope, it is a construction:
:func:`quantise_hard_ste` performs the same ``cdist -> argmin -> gather``, over
all modes, in the same order as ``engine.build_bank``, and selects with the same
``bank[modes, groups]`` advanced index as ``engine._tail_distortion``.
:func:`selfcheck` asserts the agreement numerically.
"""

import torch

from opq import batch_inv_normalize_gpu

from . import config as C


def quantise_hard_ste(codec, y, modes):
    """Hard quantisation of ``y`` under ``modes``, differentiable by STE.

    Parameters
    ----------
    codec : FeatureCodecV1 with a MultiModeSoftPQ and a rotation transform.
    y     : [B, T, D] normalised features.
    modes : LongTensor [G], the allocation.

    Returns ``(y_hat [B, T, D], labels [G, B*T])``.
    """
    pq = codec.pq
    rotation = codec.transform.get_rotation()
    b, tokens, dim = y.shape
    z = y.reshape(b * tokens, dim) @ rotation
    sub = z.reshape(-1, pq.G, pq.d).permute(1, 0, 2)            # [G, N, d]

    # Same loop, same order, same ops as engine.build_bank.  The argmin is taken
    # under no_grad: the assignment is a discrete decision and is deliberately
    # not differentiated -- that is what makes this straight-through rather than
    # a relaxation.
    banks, all_labels = [], []
    for quantizer in pq.quantizers:
        book = quantizer.codebooks                              # [G, K, d]
        with torch.no_grad():
            labels = torch.cdist(sub.detach(),
                                 book.detach()).square().argmin(dim=-1)
        banks.append(torch.gather(
            book, 1, labels.unsqueeze(-1).expand(-1, -1, pq.d)))
        all_labels.append(labels)
    bank = torch.stack(banks)                                   # [M, G, N, d]
    labels = torch.stack(all_labels)                            # [M, G, N]

    groups = torch.arange(pq.G, device=y.device)
    chosen = bank[modes, groups]                                # [G, N, d]
    z_hat = chosen.permute(1, 0, 2).reshape(-1, dim)

    # The parenthesisation is not cosmetic.  ``(z_hat + z) - z.detach()`` would
    # round twice and leave the forward value a ULP or so away from ``chosen``;
    # ``z_hat + (z - z.detach())`` evaluates the bracket to exactly +0.0 first,
    # so the forward value is ``chosen`` bit for bit while the backward path is
    # unchanged.  "The training loss is the endpoint metric" is a claim about
    # bits, so it is written in the order that makes it literally true.
    z_hat = z_hat + (z - z.detach())                            # <- the STE
    return (z_hat @ rotation.t()).reshape(b, tokens, dim), labels[modes, groups]


def distortion(codec, tail, y, mu, std, teacher, modes):
    """Per-image hard tail distortion ``[B]``, with gradients.

    This is the training loss and the endpoint metric, in one expression.
    """
    y_hat, labels = quantise_hard_ste(codec, y, modes)
    x_hat = batch_inv_normalize_gpu(y_hat, mu, std)
    output = tail(x_hat)
    return (output - teacher).square().reshape(y.shape[0], -1).sum(-1), labels


def dead_codeword_counts(labels, codec, modes):
    """How many codewords of each group went unused in this window.

    Recorded, never gated.  ``labels`` is ``[G, N]`` as returned above.
    """
    counts = []
    for group in range(codec.pq.G):
        size = codec.pq.mode_sizes[int(modes[group])]
        used = torch.bincount(labels[group], minlength=size)
        counts.append(int((used == 0).sum()))
    return counts


@torch.no_grad()
def revive_dead_codewords(codec, y, modes, labels):
    """Reset unused codewords onto the currently worst-quantised vectors.

    Fully deterministic -- the targets are the top-``k`` squared residuals, so
    there is no generator and no seed to match across arms.  It runs on the same
    schedule with the same rule in all three arms, so it cannot favour one of
    them.  Returns the number of codewords moved.
    """
    pq = codec.pq
    rotation = codec.transform.get_rotation()
    b, tokens, dim = y.shape
    z = (y.reshape(b * tokens, dim) @ rotation).reshape(-1, pq.G, pq.d)
    z = z.permute(1, 0, 2)                                      # [G, N, d]
    moved = 0
    for group in range(pq.G):
        mode = int(modes[group])
        book = pq.quantizers[mode].codebooks[group]             # [K, d]
        used = torch.bincount(labels[group], minlength=book.shape[0])
        dead = torch.nonzero(used == 0, as_tuple=False).flatten()
        if dead.numel() == 0:
            continue
        residual = (z[group] - book[labels[group]]).square().sum(-1)
        worst = torch.topk(residual, k=min(dead.numel(), residual.numel())).indices
        book[dead[:worst.numel()]] = z[group][worst]
        moved += int(worst.numel())
    return moved


@torch.no_grad()
def selfcheck(codec, tail, resident, modes, image_batch=C.EVAL_IMAGE_BATCH):
    """Assert the STE forward equals v9's evaluator on the same inputs.

    The claim "the training loss is the endpoint metric" is only worth making if
    it is checked, so this runs the v9 evaluator and this module's forward over
    the same images and allocation and returns the relative gap, which the
    caller compares against ``REPLAY_REL_TOL``.
    """
    import numpy as np

    from ..engine import evaluate_allocations

    reference = evaluate_allocations(
        codec, tail, resident, modes.cpu().numpy()[None, :],
        image_batch=image_batch)[0]

    mine = []
    for start in range(0, resident.count, image_batch):
        y, mu, std, teacher = resident.slice(start, start + image_batch)
        value, _ = distortion(codec, tail, y, mu, std, teacher, modes)
        mine.append(value.cpu().numpy())
    mine = np.concatenate(mine)

    scale = float(np.abs(reference).mean())
    return {"max_abs_gap": float(np.abs(reference - mine).max()),
            "rel_gap": float(np.abs(reference - mine).max()) / scale,
            "mean_reference": scale,
            "tolerance": C.REPLAY_REL_TOL}
