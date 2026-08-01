"""Direct orthogonal transform and Cayley-SGD for square rotations.

The update follows Li et al., *Efficient Riemannian Optimization on the
Stiefel Manifold via the Cayley Transform* (ICLR 2020).  It uses the
fixed-point Cayley retraction adopted by SpinQuant, without differentiating
through a matrix inverse.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer


class DirectOrthogonalTransform(nn.Module):
    """Store the effective rotation directly on the orthogonal manifold."""

    def __init__(self, dimension):
        super().__init__()
        self.D = int(dimension)
        self.rotation = nn.Parameter(torch.eye(self.D))

    def get_rotation(self):
        return self.rotation

    def encode(self, x):
        return x @ self.rotation

    def decode(self, x):
        return x @ self.rotation.t()

    @torch.no_grad()
    def init_from_opq(self, rotation):
        value = torch.as_tensor(
            rotation, device=self.rotation.device,
            dtype=self.rotation.dtype)
        if value.shape != (self.D, self.D):
            raise ValueError(
                f"rotation must have shape {(self.D, self.D)}")
        left, _, right = torch.linalg.svd(value.double())
        self.rotation.copy_((left @ right).to(self.rotation.dtype))

    @torch.no_grad()
    def orth_error(self):
        eye = torch.eye(
            self.D, device=self.rotation.device,
            dtype=self.rotation.dtype)
        return (self.rotation.t() @ self.rotation - eye).norm().item()


class CayleySGD(Optimizer):
    """Cayley-retracted SGD for square orthogonal matrix parameters."""

    def __init__(
        self, params, lr, fixed_point_iterations=5,
        reorthogonalize_every=100, eps=1e-8,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if fixed_point_iterations < 1:
            raise ValueError("fixed_point_iterations must be positive")
        defaults = dict(
            lr=float(lr), iterations=int(fixed_point_iterations),
            reorthogonalize_every=int(reorthogonalize_every), eps=float(eps))
        super().__init__(params, defaults)

    @staticmethod
    def _project_qr(matrix):
        q, r = torch.linalg.qr(matrix)
        signs = torch.diagonal(r).sign()
        signs[signs == 0] = 1
        return q * signs

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.ndim != 2 or parameter.shape[0] != parameter.shape[1]:
                    raise ValueError(
                        "CayleySGD requires square matrix parameters")
                gradient, current = parameter.grad, parameter
                skew = gradient @ current.t() - current @ gradient.t()
                bound = skew.abs().sum(0).max()
                step_size = (bound + group["eps"]).reciprocal().clamp(
                    max=group["lr"])
                updated = current - step_size * (skew @ current)
                for _ in range(group["iterations"]):
                    updated = current - 0.5 * step_size * (
                        skew @ (current + updated))
                state = self.state[parameter]
                state["step"] = state.get("step", 0) + 1
                state["last_step_size"] = step_size.detach()
                interval = group["reorthogonalize_every"]
                if interval > 0 and state["step"] % interval == 0:
                    updated = self._project_qr(updated)
                parameter.copy_(updated)
        return loss
