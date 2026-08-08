"""Parent-conditioned hierarchical residual PQ used by the V30 gates.

Every child centroid is its parent centroid plus a parent-specific residual.
Thus a high-rate path updates its shared ancestors without imposing the global
Cartesian-sum restriction of the rejected additive parameterisation.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from codec_v1 import load_codec_v1
from opq import batch_inv_normalize_gpu


class StageBank(nn.Module):
    """One depth of the tree.

    The root carries no parent axis.  Deeper stages always keep theirs, even
    when they happen to have a single parent -- a zero-bit first mode gives
    stage 1 one parent, and collapsing on ``parents == 1`` would then make it
    indistinguishable from a root and break the composition.
    """

    def __init__(self, groups, parents, children, dim, root=False):
        super().__init__()
        shape = ((groups, children, dim) if root else
                 (groups, parents, children, dim))
        self.codebooks = nn.Parameter(torch.empty(*shape))


class NestedMultiModePQ(nn.Module):
    """Residual tree with exact ``2**bits`` centroids at every depth."""

    def __init__(self, groups, mode_bits, dim):
        super().__init__()
        bits = tuple(map(int, mode_bits))
        if not bits or any(b < 0 for b in bits) or any(
                right <= left for left, right in zip(bits, bits[1:])):
            raise ValueError("mode_bits must be non-negative and increasing")
        increments = (bits[0],) + tuple(
            right - left for left, right in zip(bits, bits[1:]))
        self.G, self.d, self.D = int(groups), int(dim), int(groups * dim)
        self.mode_bits = bits
        self.mode_sizes = tuple(2 ** bit for bit in bits)
        self.branch_sizes = tuple(2 ** bit for bit in increments)
        self.stage_sizes = self.mode_sizes
        parents = (1,) + self.mode_sizes[:-1]
        self.stages = nn.ModuleList([
            StageBank(self.G, parent, branch, self.d, root=(depth == 0))
            for depth, (parent, branch) in enumerate(
                zip(parents, self.branch_sizes))])

    @property
    def num_modes(self):
        return len(self.mode_bits)

    def composed_codebook(self, mode):
        mode = int(mode)
        if not 0 <= mode < self.num_modes:
            raise ValueError("invalid mode")
        book = self.stages[0].codebooks
        for stage in self.stages[1:mode + 1]:
            if stage.codebooks.shape[1] != book.shape[1]:
                raise RuntimeError("tree residuals do not match their parents")
            book = (book[:, :, None, :] + stage.codebooks).reshape(
                self.G, -1, self.d)
        if book.shape[1] != self.mode_sizes[mode]:
            raise RuntimeError("composed codebook has the wrong size")
        return book

    @torch.no_grad()
    def init_from_independent(self, independent, iterations=25):
        """Exactly factor each independent book through balanced tree edges."""
        del iterations  # Retained for checkpoint/API compatibility.
        books = [q.codebooks.detach() for q in independent.quantizers]
        if tuple(int(round(math.log2(b.shape[1]))) for b in books) != \
                self.mode_bits:
            raise ValueError("independent menu does not match nested bits")
        self.stages[0].codebooks.copy_(books[0])
        for mode in range(1, self.num_modes):
            prefix = self.composed_codebook(mode - 1)
            target = books[mode]
            branch = self.branch_sizes[mode]
            residuals = self.stages[mode].codebooks
            for group in range(self.G):
                remaining = torch.arange(
                    target.shape[1], device=target.device)
                for parent in range(prefix.shape[1]):
                    if parent + 1 == prefix.shape[1]:
                        selected = remaining
                    else:
                        distance = (target[group].index_select(0, remaining) -
                                    prefix[group, parent]).square().sum(-1)
                        local = distance.argsort()[:branch]
                        selected = remaining.index_select(0, local)
                        keep = torch.ones(
                            remaining.numel(), dtype=torch.bool,
                            device=remaining.device)
                        keep[local] = False
                        remaining = remaining[keep]
                    if selected.numel() != branch:
                        raise RuntimeError("unbalanced tree initialisation")
                    residuals[group, parent].copy_(
                        target[group].index_select(0, selected) -
                        prefix[group, parent])
        return self

    def bank(self, sub, temperature=0.0):
        """Return reconstructions ``[M,G,N,d]`` and labels for all modes."""
        reconstructions, labels = [], []
        for mode in range(self.num_modes):
            book = self.composed_codebook(mode)
            distance = torch.cdist(sub, book).square()
            index = distance.detach().argmin(-1)
            if float(temperature) > 0:
                soft = torch.softmax(-distance / float(temperature), -1)
                hard = F.one_hot(index, book.shape[1]).to(soft.dtype)
                weights = hard.detach() + soft - soft.detach()
                value = torch.einsum("gnk,gkd->gnd", weights, book)
            else:
                value = torch.gather(
                    book, 1, index[..., None].expand(-1, -1, self.d))
                value = value + sub - sub.detach()
            reconstructions.append(value)
            labels.append(index)
        return torch.stack(reconstructions), torch.stack(labels)


class NestedCodec(nn.Module):
    def __init__(self, transform, pq):
        super().__init__()
        self.transform, self.pq = transform, pq


def from_independent(codec, iterations=25):
    bits = tuple(int(round(math.log2(q.codebooks.shape[1])))
                 for q in codec.pq.quantizers)
    pq = NestedMultiModePQ(codec.pq.G, bits, codec.pq.d).to(
        next(codec.parameters()).device)
    pq.init_from_independent(codec.pq, iterations=iterations)
    return NestedCodec(copy.deepcopy(codec.transform), pq)


def reconstruct(codec, y, allocation, temperature=0.0):
    allocation = torch.as_tensor(
        allocation, dtype=torch.long, device=y.device).reshape(-1)
    if allocation.numel() != codec.pq.G:
        raise ValueError("allocation must contain one mode per group")
    rotation = codec.transform.get_rotation()
    batch, tokens, dim = y.shape
    z = y.reshape(batch * tokens, dim) @ rotation
    sub = z.reshape(-1, codec.pq.G, codec.pq.d).permute(1, 0, 2)
    chosen, labels = sparse_select(
        codec.pq, sub, allocation, temperature=temperature)
    z_hat = chosen.permute(1, 0, 2).reshape(batch * tokens, dim)
    decoded = z_hat @ rotation.t()
    return decoded.reshape(batch, tokens, dim), labels


def sparse_select(pq, sub, allocation, temperature=0.0):
    """Quantise only the group--mode cells used by one allocation."""
    allocation = torch.as_tensor(
        allocation, dtype=torch.long, device=sub.device).reshape(-1)
    chosen = torch.empty_like(sub)
    labels = torch.empty(
        pq.G, sub.shape[1], dtype=torch.long, device=sub.device)
    for mode in torch.unique(allocation).tolist():
        groups = torch.where(allocation == mode)[0]
        vectors = sub.index_select(0, groups)
        book = pq.composed_codebook(mode).index_select(0, groups)
        distance = torch.cdist(vectors, book).square()
        index = distance.detach().argmin(-1)
        if float(temperature) > 0:
            soft = torch.softmax(-distance / float(temperature), -1)
            hard = F.one_hot(index, book.shape[1]).to(soft.dtype)
            weights = hard.detach() + soft - soft.detach()
            value = torch.einsum("gnk,gkd->gnd", weights, book)
        else:
            value = torch.gather(
                book, 1, index[..., None].expand(-1, -1, pq.d))
            value = value + vectors - vectors.detach()
        chosen.index_copy_(0, groups, value)
        labels.index_copy_(0, groups, index)
    return chosen, labels


def distortion(codec, tail, y, mu, std, teacher, allocation,
               temperature=0.0):
    decoded, labels = reconstruct(codec, y, allocation, temperature)
    output = tail(batch_inv_normalize_gpu(decoded, mu, std))
    value = (output - teacher).square().reshape(y.shape[0], -1).sum(-1)
    return value, labels


@torch.no_grad()
def evaluate(codec, tail, resident, allocations, image_batch=16):
    allocations = torch.as_tensor(
        allocations, dtype=torch.long, device=resident.device)
    if allocations.ndim == 1:
        allocations = allocations[None]
    rows = []
    for allocation in allocations:
        pieces = []
        for first in range(0, resident.count, int(image_batch)):
            value, _ = distortion(
                codec, tail, *resident.slice(first, first + image_batch),
                allocation)
            pieces.append(value)
        rows.append(torch.cat(pieces))
    return torch.stack(rows).cpu().numpy()


def save_checkpoint(codec, source_codec, path, extra=None):
    payload = {
        "format": "v30_hierarchical_tree_v1",
        "source_codec": str(Path(source_codec).resolve()),
        "mode_bits": list(codec.pq.mode_bits),
        "state_dict": codec.state_dict(),
        "extra": extra or {}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, device="cuda"):
    payload = torch.load(Path(path), map_location=device)
    if payload.get("format") != "v30_hierarchical_tree_v1":
        raise ValueError("not a V30 hierarchical-tree checkpoint")
    source = load_codec_v1(Path(payload["source_codec"]), device=device)
    pq = NestedMultiModePQ(
        source.pq.G, payload["mode_bits"], source.pq.d).to(device)
    codec = NestedCodec(copy.deepcopy(source.transform), pq).to(device)
    codec.load_state_dict(payload["state_dict"])
    return codec, payload
