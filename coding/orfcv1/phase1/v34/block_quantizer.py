"""Spatial block codecs shared by V34 training and evaluation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

SIDE, DIM, GROUPS, GROUP_DIM = 16, 1024, 32, 32
PATCHES, BLOCK_TOKENS = SIDE * SIDE, 4


def blockify(z, pad_mode="zero", block_shape=(2, 2)):
    batch, height, width = z.shape[:3]
    block_h, block_w = block_shape
    pad_h, pad_w = (-height) % block_h, (-width) % block_w
    if pad_w and pad_mode == "replicate":
        z = torch.cat((z, z[:, :, -1:].expand(-1, -1, pad_w, -1, -1)), 2)
    if pad_h and pad_mode == "replicate":
        z = torch.cat((z, z[:, -1:].expand(-1, pad_h, -1, -1, -1)), 1)
    if pad_mode != "replicate" and (pad_h or pad_w):
        z = F.pad(z, (0, 0, 0, 0, 0, pad_w, 0, pad_h))
    padded_h, padded_w = z.shape[1:3]
    return z.reshape(batch, padded_h // block_h, block_h,
                     padded_w // block_w, block_w,
                     GROUPS, GROUP_DIM).permute(0, 1, 3, 5, 2, 4, 6).reshape(
                         batch, padded_h * padded_w // (block_h * block_w),
                         GROUPS, block_h * block_w * GROUP_DIM)


def unblockify(z, height=SIDE, width=SIDE, block_shape=(2, 2)):
    batch = z.shape[0]
    block_h, block_w = block_shape
    padded_h = height + (-height) % block_h
    padded_w = width + (-width) % block_w
    grid = z.reshape(batch, padded_h // block_h, padded_w // block_w,
                     GROUPS, block_h, block_w,
                     GROUP_DIM).permute(0, 1, 4, 2, 5, 3, 6).reshape(
                         batch, padded_h, padded_w, DIM)
    return grid[:, :height, :width].reshape(batch, height * width, DIM)


class BlockPQ(nn.Module):
    def __init__(self, parts, size, block_shape=(2, 2)):
        super().__init__()
        if GROUP_DIM % parts:
            raise ValueError(f"{parts=} must divide {GROUP_DIM}")
        self.block_shape = tuple(block_shape)
        self.block_tokens = self.block_shape[0] * self.block_shape[1]
        self.parts, self.channels = parts, GROUP_DIM // parts
        self.G, self.K = GROUPS * parts, size
        self.d = self.block_tokens * self.channels
        self.codebooks = nn.Parameter(torch.empty(self.G, self.K, self.d))
        self.temperature = 0.0
        self._last_labels = None
        self._last_rate = torch.tensor(0.0)

    def init_codebooks(self, books):
        with torch.no_grad():
            self.codebooks.copy_(books)

    def split(self, blocks):
        batch, count = blocks.shape[:2]
        values = blocks.reshape(batch, count, GROUPS, self.block_tokens,
                                self.parts, self.channels)
        return values.permute(0, 1, 2, 4, 3, 5).reshape(
            batch, count, self.G, self.d)

    def merge(self, values):
        batch, count = values.shape[:2]
        values = values.reshape(batch, count, GROUPS, self.parts,
                                self.block_tokens, self.channels)
        return values.permute(0, 1, 2, 4, 3, 5).reshape(
            batch, count, GROUPS, self.block_tokens * GROUP_DIM)

    def forward(self, blocks):
        batch, count = blocks.shape[:2]
        vectors = self.split(blocks)
        vectors = vectors.permute(2, 0, 1, 3).reshape(self.G, batch * count, self.d)
        distance = torch.cdist(vectors, self.codebooks).square()
        labels = distance.detach().argmin(-1)
        self._last_labels = labels
        if self.training and self.temperature > 0:
            soft = torch.softmax(-distance / self.temperature, -1)
            hard = F.one_hot(labels, self.K).to(soft.dtype)
            chosen = torch.einsum(
                "gnk,gkd->gnd", hard - soft.detach() + soft, self.codebooks)
        else:
            chosen = torch.gather(
                self.codebooks[:, None].expand(-1, batch * count, -1, -1), 2,
                labels[..., None, None].expand(-1, -1, 1, self.d)).squeeze(2)
        usage = torch.zeros(self.G, self.K, device=blocks.device)
        usage.scatter_add_(1, labels, torch.ones_like(labels, dtype=usage.dtype))
        chosen = chosen.reshape(self.G, batch, count, self.d).permute(1, 2, 0, 3)
        return self.merge(chosen), usage


class BlockVQ(BlockPQ):
    def __init__(self):
        super().__init__(parts=1, size=256)


class ProductBlockVQ(BlockPQ):
    def __init__(self):
        super().__init__(parts=2, size=16)


class TokenPairVQ(BlockPQ):
    def __init__(self):
        super().__init__(parts=1, size=16, block_shape=(1, 2))


class PatchOnlyCodec(nn.Module):
    def __init__(self, transform, pq, arm, pad_mode="zero"):
        super().__init__()
        self.transform, self.pq, self.arm = transform, pq, arm
        self.pad_mode = pad_mode

    @property
    def use_rate(self):
        return False

    def forward(self, y):
        batch, tokens, dim = y.shape
        patches = tokens - 1
        side = int(patches ** .5)
        if side * side != patches:
            raise ValueError(f"patch token count {patches} is not a square grid")
        rotation = self.transform.get_rotation()
        z = y[:, 1:].reshape(batch * patches, dim) @ rotation
        if self.arm == "orfc":
            zhat, usage = self.pq._quantise(z)
        else:
            grid = z.reshape(batch, side, side, GROUPS, GROUP_DIM)
            shape = self.pq.block_shape
            zhat, usage = self.pq(blockify(grid, self.pad_mode, shape))
            zhat = unblockify(zhat, side, side, shape).reshape(batch * patches, dim)
        patch = (zhat @ rotation.t()).reshape(batch, patches, dim)
        return torch.cat((y[:, :1], patch), 1), usage
