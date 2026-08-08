"""Encode / decode / hard PQ paths for V33 (``U0`` then conditional ``L``).

Encoding: ``z = y @ U0``, split into groups, then ``z_g @ L[g, m_g]``.
Decoding: per-group ``· @ L[g, m_g].T``, then ``· @ U0.T``.

Hot-path note
-------------
``AnchoredCayleyTransform.get_rotation`` solves a 1024×1024 system and
``ConditionalBlockDiagL.select`` solves 32×32×G systems.  Both are independent
of the image batch, so every call site should either:

* pass precomputed ``rotation`` / ``blocks`` into :func:`linear_encode` /
  :func:`linear_decode` (single-call de-dupe), or
* enter :func:`frozen_rotations` once per scorer / evaluation pass so the
  full ``U0`` and ``L`` bank are materialised exactly once.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from phase1.v30.nested import sparse_select


_ATTR_U0 = "_v33_frozen_u0"
_ATTR_L = "_v33_frozen_L"


def _as_allocation(allocation, groups, device):
    allocation = torch.as_tensor(
        allocation, dtype=torch.long, device=device).reshape(-1)
    if allocation.numel() != groups:
        raise ValueError(
            f"allocation must contain one mode per group ({groups})")
    return allocation


def _resolve_u0(codec, rotation=None):
    if rotation is not None:
        return rotation
    cached = getattr(codec, _ATTR_U0, None)
    if cached is not None:
        return cached
    return codec.transform.get_rotation()


def _uses_L(codec):
    """``False`` when the codec was built with ``use_L=False`` (no bank)."""
    return getattr(codec, "L", None) is not None


def _resolve_blocks(codec, allocation, blocks=None):
    if blocks is not None:
        return blocks
    if not _uses_L(codec):
        return None
    bank = getattr(codec, _ATTR_L, None)
    if bank is not None:
        groups = torch.arange(codec.L.G, device=allocation.device)
        return bank[groups, allocation]
    return codec.L.select(allocation)


def linear_encode(codec, y, allocation, rotation=None, blocks=None):
    """Apply ``U0`` and selected ``L`` blocks without quantisation.

    Returns grouped coordinates ``[G, N, d]`` in the L-rotated frame.
    Optional ``rotation`` / ``blocks`` skip redundant Cayley solves.
    """
    allocation = _as_allocation(allocation, codec.pq.G, y.device)
    rotation = _resolve_u0(codec, rotation)
    batch, tokens, dim = y.shape
    z = y.reshape(batch * tokens, dim) @ rotation
    sub = z.reshape(-1, codec.pq.G, codec.pq.d).permute(1, 0, 2)
    if not _uses_L(codec):
        return sub, allocation
    if blocks is None and getattr(codec, _ATTR_L, None) is None:
        # Fall through to rotate_groups (computes select once internally).
        return codec.L.rotate_groups(sub, allocation), allocation
    blocks = _resolve_blocks(codec, allocation, blocks)
    return torch.bmm(sub, blocks), allocation


def linear_decode(codec, sub_L, allocation, batch, tokens,
                  rotation=None, blocks=None):
    """Inverse of :func:`linear_encode` for grouped ``[G, N, d]`` tensors."""
    allocation = _as_allocation(allocation, codec.pq.G, sub_L.device)
    if not _uses_L(codec):
        restored = sub_L
    elif blocks is None and getattr(codec, _ATTR_L, None) is None:
        restored = codec.L.rotate_groups(sub_L, allocation, transpose=True)
    else:
        blocks = _resolve_blocks(codec, allocation, blocks)
        restored = torch.bmm(sub_L, blocks.transpose(-1, -2))
    dim = codec.pq.G * codec.pq.d
    z = restored.permute(1, 0, 2).reshape(batch * tokens, dim)
    rotation = _resolve_u0(codec, rotation)
    decoded = z @ rotation.t()
    return decoded.reshape(batch, tokens, dim)


def quantise_sparse(codec, y, allocation, temperature=0.0,
                    rotation=None, blocks=None):
    """One hard allocation; unused (group, mode) banks are never materialised."""
    allocation = _as_allocation(allocation, codec.pq.G, y.device)
    batch, tokens, dim = y.shape
    # Materialise once per call when the caller did not pre-budget.
    if rotation is None:
        rotation = _resolve_u0(codec, None)
    if blocks is None:
        blocks = _resolve_blocks(codec, allocation, None)
    sub_L, allocation = linear_encode(
        codec, y, allocation, rotation=rotation, blocks=blocks)
    chosen_L, labels = sparse_select(
        codec.pq, sub_L, allocation, temperature=temperature)
    decoded = linear_decode(
        codec, chosen_L, allocation, batch, tokens,
        rotation=rotation, blocks=blocks)
    return decoded, labels


def quantise(codec, y, allocations, temperature=0.0):
    """Multi-allocation hard PQ; returns stacked ``[S*B, tokens, dim]``.

    Mirrors ``v12.qhard.quantise`` packing so a single tail call can score a
    slate.  Prefer :func:`quantise_sparse` when streaming one allocation.
    """
    allocations = torch.as_tensor(
        allocations, dtype=torch.long, device=y.device)
    if allocations.ndim == 1:
        allocations = allocations[None]
    # U0 is shared across the slate; L blocks still depend on allocation.
    rotation = _resolve_u0(codec, None)
    decoded_rows, label_rows = [], []
    for allocation in allocations:
        decoded, labels = quantise_sparse(
            codec, y, allocation, temperature=temperature,
            rotation=rotation)
        decoded_rows.append(decoded)
        label_rows.append(labels)
    return torch.cat(decoded_rows, dim=0), torch.stack(label_rows)


def reconstruct(codec, y, allocation, temperature=0.0):
    """Alias of :func:`quantise_sparse` matching ``v30.nested.reconstruct``."""
    return quantise_sparse(
        codec, y, allocation, temperature=temperature)


@contextmanager
def frozen_rotations(codec):
    """Budget ``U0`` and the full ``L`` bank once for a no-grad evaluation pass.

    Nested entry is a no-op so callers can wrap both a scorer constructor and
    an inner evaluate without double-materialising.
    """
    already = getattr(codec, _ATTR_U0, None) is not None
    if already:
        yield codec
        return
    with torch.no_grad():
        rotation = codec.transform.get_rotation().detach()
        bank = codec.L.get_rotations().detach() if _uses_L(codec) else None
    setattr(codec, _ATTR_U0, rotation)
    if bank is not None:
        setattr(codec, _ATTR_L, bank)
    try:
        yield codec
    finally:
        for attr in (_ATTR_U0, _ATTR_L):
            if hasattr(codec, attr):
                delattr(codec, attr)
