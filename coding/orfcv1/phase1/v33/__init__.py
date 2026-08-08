"""V33: two-stage allocation solving with conditional block-diagonal ``L``.

* :class:`ConditionalBlockDiagL` — ``[G,T,496]`` Cayley bank (``d=32``)
* :class:`V33Codec` — ``U0`` + ``L`` + :class:`~phase1.v30.nested.NestedMultiModePQ`
* :mod:`phase1.v33.quantise` / :mod:`phase1.v33.distortion` — encode & Tail MSE
* :mod:`phase1.v33.checkpoint` — save / load (U0, L, pq, step, allocation)
* :mod:`phase1.v33.train` — phase-1 strict-fair + phase-2 frozen-U0

Note: the ``distortion`` / ``quantise`` *submodules* must remain importable as
``from phase1.v33 import distortion`` / ``quantise``; do not re-bind those names
to the callable helpers (use ``distortion_sparse`` / ``quantise_sparse``).
"""

from . import distortion  # noqa: F401  — keep submodule on package namespace
from . import quantise as quantise_mod  # noqa: F401
from .checkpoint import (
    allocation_rate,
    load_allocation,
    load_checkpoint,
    load_u0_rotation,
    meta_allocation,
    save_checkpoint,
    try_load_codec,
)
from .codec import V33Codec
from .distortion import (
    distortion_sparse,
    distortions,
    evaluate,
)
from .quantise import (
    frozen_rotations,
    linear_decode,
    linear_encode,
    quantise_sparse,
    reconstruct,
)
from .quantise import quantise as quantise_fn
from .transform import ConditionalBlockDiagL, skew_param_count

# Expose the submodule under its canonical name (not the function).
quantise = quantise_mod
# Callable still available without shadowing the module attribute permanently
# for ``from phase1.v33 import quantise`` users who expect the function —
# prefer ``quantise_sparse`` / ``quantise.quantise``.

__all__ = [
    "ConditionalBlockDiagL",
    "V33Codec",
    "skew_param_count",
    "linear_encode",
    "linear_decode",
    "quantise",
    "quantise_fn",
    "quantise_sparse",
    "reconstruct",
    "frozen_rotations",
    "distortion",
    "distortion_sparse",
    "distortions",
    "evaluate",
    "save_checkpoint",
    "load_checkpoint",
    "load_u0_rotation",
    "load_allocation",
    "allocation_rate",
    "meta_allocation",
    "try_load_codec",
]
