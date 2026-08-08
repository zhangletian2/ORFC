"""Multi-mode PQ built from the existing :class:`SoftPQ` implementation."""

import os
import sys
import torch
import torch.nn as nn

ORFC_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "orfc"))
if ORFC_DIR not in sys.path:
    sys.path.insert(0, ORFC_DIR)
from soft_pq import SoftPQ


class MultiModeSoftPQ(nn.Module):
    """A shared-group PQ menu with one existing SoftPQ per mode.

    Every mode uses the same rotated groups but has its own codebook size.
    ``modes[g]`` selects the mode used by group ``g``.  Omitting ``modes``
    selects the largest mode for every group.
    """

    def __init__(self, G, mode_sizes, d, lmbda=0.0, prior_floor=0.0):
        super().__init__()
        sizes = tuple(int(k) for k in mode_sizes)
        if not sizes or tuple(sorted(set(sizes))) != sizes:
            raise ValueError(
                "mode_sizes must be non-empty and strictly increasing")
        # K == 1 is the zero-bit mode: the group costs no rate and is served by
        # a single learned centroid.  Low-rate anchors need it to have any
        # allocation freedom at all.
        if any(k < 1 or (k & (k - 1)) for k in sizes):
            raise ValueError("every mode size must be a power of two >= 1")
        self.G, self.d, self.D = int(G), int(d), int(G * d)
        self.K = sizes[-1]
        self.mode_sizes = sizes
        self.lmbda = float(lmbda)
        self.use_rate = self.lmbda > 0
        self.prior_floor = float(prior_floor)
        self.quantizers = nn.ModuleList([
            SoftPQ(
                self.G, k, self.d,
                lmbda=self.lmbda, prior_floor=self.prior_floor)
            for k in sizes
        ])
        self._temperature = 0.0
        self._last_rate = None
        self._last_rate_per_group = None
        self._last_labels = None
        self._last_modes = None
        self._last_nominal_rate = None
        self._last_nominal_rate_per_group = None

    @property
    def num_modes(self):
        return len(self.mode_sizes)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = float(value)
        for quantizer in self.quantizers:
            quantizer.temperature = self._temperature

    def _validate_modes(self, modes, device):
        if modes is None:
            return torch.full(
                (self.G,), self.num_modes - 1,
                dtype=torch.long, device=device)
        modes = torch.as_tensor(modes, dtype=torch.long, device=device)
        if modes.shape != (self.G,):
            raise ValueError(
                f"modes must have shape ({self.G},), got {tuple(modes.shape)}")
        if bool(((modes < 0) | (modes >= self.num_modes)).any()):
            raise ValueError(
                f"mode indices must be in [0, {self.num_modes - 1}]")
        return modes

    def init_from_kmeans(
        self, Z_flat, device="cuda", max_iter=100, seed=42,
    ):
        """Reuse SoftPQ k-means initialisation for every menu mode."""
        for mode, quantizer in enumerate(self.quantizers):
            torch.manual_seed(int(seed) + mode)
            quantizer.init_from_kmeans(
                Z_flat, device=device, max_iter=max_iter)

    def init_codebooks(self, codebooks_by_mode):
        """Initialise from a sequence or ``{K: codebooks}`` mapping."""
        for mode, quantizer in enumerate(self.quantizers):
            source = (
                codebooks_by_mode[self.mode_sizes[mode]]
                if isinstance(codebooks_by_mode, dict)
                else codebooks_by_mode[mode])
            quantizer.init_codebooks(source)

    def nominal_rate_per_group(self, modes=None, device=None):
        if device is None:
            device = next(self.parameters()).device
        modes = self._validate_modes(modes, device)
        sizes = torch.as_tensor(
            self.mode_sizes, device=device, dtype=torch.float32)
        return torch.log2(sizes[modes])

    def _quantise(self, Z_flat, modes=None):
        N = Z_flat.shape[0]
        mode_ids = self._validate_modes(modes, Z_flat.device)
        mode_list = mode_ids.tolist()
        selected = {}
        usages = {}
        labels = {}
        rates = {}
        for mode in torch.unique(mode_ids).tolist():
            quantizer = self.quantizers[mode]
            quantizer.temperature = self.temperature
            Z_mode, usage = quantizer._quantise(Z_flat)
            selected[mode] = Z_mode.reshape(N, self.G, self.d)
            usages[mode] = usage
            labels[mode] = quantizer._last_labels
            if self.use_rate:
                rates[mode] = quantizer._last_rate_per_group

        groups = [
            selected[mode_list[g]][:, g, :]
            for g in range(self.G)
        ]
        Z_hat_flat = torch.stack(groups, dim=1).reshape(N, self.D)
        usage = torch.zeros(
            self.G, self.K, device=Z_flat.device)
        last_labels = torch.empty(
            self.G, N, dtype=torch.long, device=Z_flat.device)
        for g in range(self.G):
            mode = mode_list[g]
            k = self.mode_sizes[mode]
            usage[g, :k] = usages[mode][g]
            last_labels[g] = labels[mode][g]
        self._last_labels = last_labels.detach()
        self._last_modes = mode_ids.detach()

        nominal = self.nominal_rate_per_group(
            mode_ids, device=Z_flat.device)
        self._last_nominal_rate_per_group = nominal.detach()
        self._last_nominal_rate = nominal.sum().detach()
        if self.use_rate:
            per_group = torch.stack([
                rates[mode_list[g]][g] for g in range(self.G)
            ])
            self._last_rate_per_group = per_group
            self._last_rate = per_group.sum()
        else:
            self._last_rate_per_group = None
            self._last_rate = torch.tensor(
                0.0, device=Z_flat.device)
        return Z_hat_flat, usage

    def forward(self, Z_norm, modes=None):
        B, T, Dp = Z_norm.shape
        Z_hat, usage = self._quantise(
            Z_norm.reshape(B * T, Dp), modes=modes)
        return Z_hat.reshape(B, T, Dp), usage

    @torch.no_grad()
    def get_prior_pmf(self):
        return {
            k: quantizer.get_prior_pmf()
            for k, quantizer in zip(self.mode_sizes, self.quantizers)
        }
