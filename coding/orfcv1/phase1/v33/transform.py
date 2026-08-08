"""Conditional block-diagonal SO(d) banks via batched Cayley charts.

Each (group, mode) cell owns an independent skew-symmetric Cayley parameter
vector of length ``d*(d-1)//2`` (496 when ``d=32``).  The resulting rotations
are block-diagonal in the grouped coordinate frame after ``z = y @ U0``:
``(y U0 L)[:, S_g] = (y U0)[:, S_g] L[g, m_g]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def skew_param_count(dim):
    dim = int(dim)
    return dim * (dim - 1) // 2


class ConditionalBlockDiagL(nn.Module):
    """Per-(group, mode) ``SO(d)`` with skew parameters ``[G, T, n_skew]``."""

    def __init__(self, groups, modes, dim):
        super().__init__()
        self.G = int(groups)
        self.T = int(modes)
        self.d = int(dim)
        if self.G < 1 or self.T < 1 or self.d < 2:
            raise ValueError("groups, modes, and dim must be positive (dim>=2)")
        n_skew = skew_param_count(self.d)
        self.triu_params = nn.Parameter(
            torch.zeros(self.G, self.T, n_skew))
        indices = torch.triu_indices(self.d, self.d, offset=1)
        self.register_buffer("_triu_row", indices[0])
        self.register_buffer("_triu_col", indices[1])

    @property
    def n_skew(self):
        return int(self.triu_params.shape[-1])

    def _skew_bank(self):
        skew = self.triu_params.new_zeros(
            self.G, self.T, self.d, self.d)
        skew[..., self._triu_row, self._triu_col] = self.triu_params
        return skew - skew.transpose(-1, -2)

    @staticmethod
    def _cayley(skew):
        """Cayley map for a stack of skew matrices ``[..., d, d]``.

        Prefer one batched ``torch.linalg.solve``.  A previous MAGMA
        ``misaligned address`` crash was tied to non-contiguous views; we
        force ``.contiguous()`` and fall back to per-matrix solves if the
        batched path still fails.
        """
        shape = skew.shape
        flat = skew.reshape(-1, shape[-1], shape[-1]).contiguous()
        eye = torch.eye(
            flat.shape[-1], device=flat.device, dtype=flat.dtype)
        try:
            mapped = torch.linalg.solve(eye - flat, eye + flat)
        except RuntimeError:
            rows = [
                torch.linalg.solve(eye - flat[index], eye + flat[index])
                for index in range(flat.shape[0])]
            mapped = torch.stack(rows, dim=0)
        return mapped.reshape(shape)

    def get_rotations(self):
        """Cayley map ``[G, T, d, d]`` (identity at zero params)."""
        return self._cayley(self._skew_bank())

    def select(self, allocation):
        """Gather ``L[g, m_g]`` for one allocation ``[G]`` → ``[G, d, d]``."""
        allocation = torch.as_tensor(
            allocation, dtype=torch.long,
            device=self.triu_params.device).reshape(-1)
        if allocation.numel() != self.G:
            raise ValueError(
                f"allocation must have length {self.G}, got {allocation.numel()}")
        if bool((allocation < 0).any() or (allocation >= self.T).any()):
            raise ValueError("allocation mode index out of range")
        groups = torch.arange(self.G, device=allocation.device)
        # Only build the G active cells — avoids a full 96-way Cayley on the
        # training hot path.
        skew = self.triu_params.new_zeros(self.G, self.d, self.d)
        params = self.triu_params[groups, allocation]
        skew[:, self._triu_row, self._triu_col] = params
        skew = skew - skew.transpose(-1, -2)
        return self._cayley(skew)

    def rotate_groups(self, sub, allocation, transpose=False):
        """Apply selected blocks to grouped vectors ``[G, N, d]``."""
        if sub.ndim != 3 or sub.shape[0] != self.G or sub.shape[-1] != self.d:
            raise ValueError(
                f"sub must have shape [{self.G}, N, {self.d}], got {tuple(sub.shape)}")
        blocks = self.select(allocation)
        if transpose:
            blocks = blocks.transpose(-1, -2)
        return torch.bmm(sub, blocks)

    @torch.no_grad()
    def orth_error(self):
        """Max Frobenius ``||L^T L - I||`` over all (group, mode) cells."""
        rotations = self.get_rotations()
        eye = torch.eye(
            self.d, device=rotations.device, dtype=rotations.dtype)
        gram = rotations.transpose(-1, -2) @ rotations - eye
        return float(gram.reshape(self.G * self.T, -1).norm(dim=-1).max())

    @torch.no_grad()
    def reset_identity(self):
        self.triu_params.zero_()
