"""Two-sided orthogonal transform and hard multi-mode PQ helpers."""

from __future__ import annotations

import torch
import torch.nn as nn


class TwoSidedPQ(nn.Module):
    def __init__(self, token_rotation, channel_rotation, codebooks, bits,
                 groups=32, group_dim=32):
        super().__init__()
        self.register_buffer("V", torch.as_tensor(token_rotation).float())
        self.register_buffer("U", torch.as_tensor(channel_rotation).float())
        self.bits = tuple(map(int, bits))
        self.G, self.d = int(groups), int(group_dim)
        self.books = nn.ParameterList([
            nn.Parameter(torch.as_tensor(book).float(), requires_grad=False)
            for book in codebooks])
        if self.V.ndim != 2 or self.V.shape[0] != self.V.shape[1]:
            raise ValueError("V must be square")
        if self.U.shape != (self.G * self.d, self.G * self.d):
            raise ValueError("U geometry mismatch")

    def analysis(self, y):
        # Z = V^T F U; V columns are token modes.
        token = torch.einsum("ts,...td->...sd", self.V, y)
        return token @ self.U

    def synthesis(self, z):
        token = z @ self.U.t()
        return torch.einsum("ts,...sd->...td", self.V, token)

    @torch.no_grad()
    def mode_error_bank(self, y):
        """Quantisation errors [M,B,T,D] in the V/U coefficient frame."""
        z = self.analysis(y)
        b, t, d0 = z.shape
        sub = z.reshape(-1, self.G, self.d).permute(1, 0, 2)
        rows = []
        for book in self.books:
            labels = torch.cdist(sub, book).argmin(-1)
            chosen = torch.gather(
                book, 1, labels[..., None].expand(-1, -1, self.d))
            hat = chosen.permute(1, 0, 2).reshape(b, t, d0)
            rows.append(hat - z)
        return torch.stack(rows)

    @torch.no_grad()
    def error_for_allocation(self, bank, allocation):
        allocation = torch.as_tensor(
            allocation, dtype=torch.long, device=bank.device).reshape(self.G)
        # bank [M,B,T,D] -> [B,T,G,d], selecting one mode per group.
        grouped = bank.reshape(
            bank.shape[0], bank.shape[1], bank.shape[2], self.G, self.d)
        selected = torch.stack([
            grouped[int(allocation[g]), :, :, g] for g in range(self.G)], 2)
        return selected.reshape(bank.shape[1], bank.shape[2], self.G * self.d)

    def decode_error(self, coefficient_error):
        return self.synthesis(coefficient_error)

    @torch.no_grad()
    def reconstruct(self, y, allocation):
        error = self.error_for_allocation(self.mode_error_bank(y), allocation)
        return y + self.decode_error(error)

    @torch.no_grad()
    def orthogonality(self):
        def err(matrix):
            eye = torch.eye(matrix.shape[0], device=matrix.device,
                            dtype=torch.float64)
            value = matrix.double().t() @ matrix.double() - eye
            return float(value.norm())
        return {"V": err(self.V), "U": err(self.U)}
