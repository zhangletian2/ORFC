"""V33 codec: full ``U0`` + conditional block-diagonal ``L`` + nested PQ."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from cayley import AnchoredCayleyTransform, DirectOrthogonalTransform
from phase1.v30.nested import NestedMultiModePQ

from .transform import ConditionalBlockDiagL


class V33Codec(nn.Module):
    """Grouped nested codec with an optional allocation-conditioned ``L``.

    Phase 1 trains ``U0``, every ``L[g, t]`` cell, and the nested tree together.
    Phase 2 calls :meth:`freeze_u0` so only ``L`` (used cells) and the tree move.

    ``use_L=False`` drops the block-diagonal bank entirely.  ``L`` carries no
    representational power here: each mode's ``composed_codebook`` is a free
    (indeed over-parameterised) set of centroids, so for an orthogonal
    ``L[g, m]`` the substitution ``c = L[g, m] c'`` gives
    ``||L z - c|| == ||z - c'||`` and ``(L, C) <-> (I, C L^T)`` is a bijection
    over attainable codebooks.  Keeping the bank only pays for ``G*T`` Cayley
    solves on every encode.  The flag stays so existing ``use_L`` checkpoints
    keep loading unchanged.
    """

    def __init__(self, transform, pq, conditional_L=None, use_L=True):
        super().__init__()
        if not hasattr(transform, "get_rotation"):
            raise TypeError("transform must expose get_rotation()")
        if not isinstance(pq, NestedMultiModePQ):
            raise TypeError("pq must be a NestedMultiModePQ")
        self.transform = transform
        self.pq = pq
        if not use_L:
            if conditional_L is not None:
                raise ValueError("use_L=False conflicts with an explicit L bank")
            self.L = None
            return
        if conditional_L is None:
            conditional_L = ConditionalBlockDiagL(
                pq.G, pq.num_modes, pq.d)
        if (conditional_L.G, conditional_L.T, conditional_L.d) != (
                pq.G, pq.num_modes, pq.d):
            raise ValueError(
                "ConditionalBlockDiagL geometry must match NestedMultiModePQ")
        self.L = conditional_L

    @property
    def uses_L(self):
        return getattr(self, "L", None) is not None

    def l_orth_error(self):
        """Max ``||L^T L - I||`` over cells; ``0.0`` when the bank is absent."""
        return float(self.L.orth_error()) if self.uses_L else 0.0

    @torch.no_grad()
    def fold_L_into_codebooks(self, allocation):
        """Absorb the used ``L[g, a_g]`` blocks into the tree and drop the bank.

        Exact for a fixed allocation.  ``composed_codebook`` is a plain sum of
        stage terms, so right-multiplying every stage of group ``g`` by
        ``L[g, a_g].T`` moves each composed centroid exactly where the decoder
        used to put it, leaving the reconstruction bit-identical.
        """
        if not self.uses_L:
            return self
        device = self.pq.stages[0].codebooks.device
        allocation = torch.as_tensor(
            allocation, dtype=torch.long, device=device).reshape(-1)
        inverse = self.L.select(allocation).transpose(-1, -2)   # [G, d, d]
        for stage in self.pq.stages:
            book = stage.codebooks
            flat = book.reshape(self.pq.G, -1, self.pq.d)
            stage.codebooks.copy_(
                torch.bmm(flat, inverse).reshape(book.shape))
        self.L = None
        return self

    @classmethod
    def build(cls, groups, mode_bits, dim, parameterization="direct",
              device=None, dtype=None, use_L=True):
        """Construct identity-initialised ``U0``, identity ``L``, empty nested PQ."""
        groups, dim = int(groups), int(dim)
        bits = tuple(map(int, mode_bits))
        width = groups * dim
        if parameterization == "direct":
            transform = DirectOrthogonalTransform(width)
        elif parameterization in ("orfc_cayley", "anchored_cayley"):
            transform = AnchoredCayleyTransform(width)
        else:
            raise ValueError(
                f"unknown transform parameterization {parameterization!r}")
        pq = NestedMultiModePQ(groups, bits, dim)
        for stage in pq.stages:
            nn.init.zeros_(stage.codebooks)
        codec = cls(transform, pq, use_L=use_L)
        if device is not None or dtype is not None:
            codec = codec.to(device=device, dtype=dtype)
        return codec

    def freeze_u0(self):
        """Phase-2 switch: lock the global rotation (grouping structure)."""
        for parameter in self.transform.parameters():
            parameter.requires_grad_(False)
        return self

    def unfreeze_u0(self):
        for parameter in self.transform.parameters():
            parameter.requires_grad_(True)
        return self

    @property
    def u0_frozen(self):
        params = list(self.transform.parameters())
        return bool(params) and not any(p.requires_grad for p in params)

    def trainable_parameters(self):
        """Yield currently trainable parameters (respects ``freeze_u0``)."""
        for parameter in self.parameters():
            if parameter.requires_grad:
                yield parameter

    def parameter_groups(self, l_decay=0.0):
        """Adam-style groups: ``u0`` / ``L`` (optional decay) / ``pq``."""
        groups = []
        u0 = [p for p in self.transform.parameters() if p.requires_grad]
        if u0:
            groups.append({"name": "u0", "params": u0, "weight_decay": 0.0})
        if self.uses_L:
            groups.append({
                "name": "L",
                "params": list(self.L.parameters()),
                "weight_decay": float(l_decay),
            })
        groups.append({
            "name": "pq",
            "params": list(self.pq.parameters()),
            "weight_decay": 0.0,
        })
        return groups

    def clone(self):
        return copy.deepcopy(self)
