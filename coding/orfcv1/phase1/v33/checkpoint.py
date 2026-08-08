"""V33 checkpoint load/save protocol.

Train writes these; probe / search / noise-floor tools consume them.

Payload schema (``format == "v33_codec_v1"``)::

    {
      "format": "v33_codec_v1",
      "geometry": {
        "groups": 32,
        "mode_bits": [1, 2, 3],
        "dim": 32,
        "parameterization": "direct" | "orfc_cayley" | "anchored_cayley",
        "use_L": bool,   # absent in pre-ablation payloads => inferred from keys
      },
      "state_dict": <V33Codec.state_dict()>,
      "meta": {
        "step": int,
        "phase": 1 | 2,
        "allocation": list[int] | None,   # fixed assignment (phase 2 / search)
        "u0_frozen": bool,
        "l_decay": float,
        "anchor": str,
        "run_id": str,
        ...
      },
      "optimizer_state": optional Adam state (resume only),
    }

Hooks
-----
* :func:`save_checkpoint` / :func:`load_checkpoint` — full codec round-trip.
* :func:`load_u0_rotation` — read ``U0`` without reconstructing PQ (drift probe).
* :func:`try_load_codec` — accept path / payload / in-process ``V33Codec``.
* :func:`load_allocation` / :func:`allocation_rate` — phase-2 / search helpers.

If training extends ``meta`` or adds sibling keys, keep ``format`` and
``geometry`` stable; tools ignore unknown ``meta`` fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .codec import V33Codec

FORMAT = "v33_codec_v1"


def geometry_from_codec(codec):
    return {
        "groups": int(codec.pq.G),
        "mode_bits": list(map(int, codec.pq.mode_bits)),
        "dim": int(codec.pq.d),
        "parameterization": _parameterization_name(codec),
        "use_L": bool(codec.uses_L),
    }


def _use_L(geometry, state_dict):
    """Pre-``use_L`` payloads always carried a bank; fall back to the keys."""
    if "use_L" in geometry:
        return bool(geometry["use_L"])
    return any(key.startswith("L.") for key in state_dict)


def _build_from(geometry, state_dict, device):
    codec = V33Codec.build(
        geometry["groups"],
        tuple(geometry["mode_bits"]),
        geometry["dim"],
        parameterization=geometry.get("parameterization", "direct"),
        device=device,
        use_L=_use_L(geometry, state_dict),
    )
    codec.load_state_dict(state_dict)
    return codec


def _parameterization_name(codec):
    name = type(codec.transform).__name__
    if name == "DirectOrthogonalTransform":
        return "direct"
    if name == "AnchoredCayleyTransform":
        return "orfc_cayley"
    return name


def _allocation_list(allocation):
    if allocation is None:
        return None
    return [int(x) for x in torch.as_tensor(allocation).detach().cpu().reshape(-1)]


def _opq_init_summary(meta):
    """Keep checkpoint meta JSON-/pickle-safe and small."""
    if not isinstance(meta, dict):
        return {}
    keep = ("plan", "stage", "transform_parameterization", "anchor", "rate",
            "mode_bits", "images", "all_modes_kmeans")
    out = {key: meta[key] for key in keep if key in meta}
    if "resumed" in meta:
        out["resumed"] = bool(meta["resumed"])
    return out


def save_checkpoint(codec, path, meta=None, *, allocation=None,
                    optimizer=None):
    """Write a V33 codec payload.  ``meta`` is free-form bookkeeping.

    When ``allocation`` is set (or present in ``meta``), also writes sibling
    ``allocation.npy`` for the search → phase-2 hand-off.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    if "opq_init" in meta:
        meta["opq_init"] = _opq_init_summary(meta["opq_init"])
    if allocation is not None:
        meta["allocation"] = _allocation_list(allocation)
    elif "allocation" in meta:
        meta["allocation"] = _allocation_list(meta["allocation"])
    meta.setdefault("u0_frozen", bool(codec.u0_frozen))
    payload = {
        "format": FORMAT,
        "geometry": geometry_from_codec(codec),
        "state_dict": codec.state_dict(),
        "meta": meta,
        "optimizer_state": (None if optimizer is None
                            else optimizer.state_dict()),
    }
    torch.save(payload, path)
    alloc = meta.get("allocation")
    if alloc is not None:
        np.save(path.with_name("allocation.npy"),
                np.asarray(alloc, dtype=np.int64))
    path.with_suffix(".json").write_text(json.dumps({
        "format": FORMAT,
        "geometry": payload["geometry"],
        "meta": meta,
    }, indent=2))
    return path


def load_checkpoint(path, device="cpu", map_location=None, *, train=False):
    """Load ``(V33Codec, payload)``.  Raises ``ValueError`` on format mismatch.

    Restores ``freeze_u0()`` when ``meta.u0_frozen`` is true.  Defaults to
    ``eval()`` for probe tools; pass ``train=True`` for resume.
    """
    path = Path(path)
    location = device if map_location is None else map_location
    payload = torch.load(path, map_location=location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError(
            f"{path} is not a {FORMAT} checkpoint "
            f"(got format={payload.get('format')!r})")
    codec = _build_from(payload["geometry"], payload["state_dict"], device)
    meta = payload.get("meta") or {}
    if meta.get("u0_frozen"):
        codec.freeze_u0()
    if train:
        codec.train()
        if not meta.get("u0_frozen"):
            for parameter in codec.parameters():
                parameter.requires_grad_(True)
    else:
        codec.eval()
    return codec, payload


@torch.no_grad()
def load_u0_rotation(path, device="cpu"):
    """Return ``(U0 [D,D], geometry, meta)`` without building the PQ tree.

    Used by the switch-point subspace-drift probe (pure linear algebra).
    """
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError(f"{path} is not a {FORMAT} checkpoint")
    geometry = payload["geometry"]
    codec = V33Codec.build(
        geometry["groups"],
        tuple(geometry["mode_bits"]),
        geometry["dim"],
        parameterization=geometry.get("parameterization", "direct"),
        device=device,
        use_L=False,
    )
    state = payload["state_dict"]
    transform_state = {
        key[len("transform."):]: value
        for key, value in state.items()
        if key.startswith("transform.")
    }
    codec.transform.load_state_dict(transform_state, strict=True)
    return (codec.transform.get_rotation().detach(), geometry,
            payload.get("meta", {}))


def try_load_codec(source, device="cpu"):
    """``source`` may be a path, a payload dict, or a ``V33Codec``."""
    if isinstance(source, V33Codec):
        return source, {"format": FORMAT, "geometry": geometry_from_codec(source),
                        "meta": {}}
    if isinstance(source, dict):
        if source.get("format") != FORMAT:
            raise ValueError(f"dict payload format must be {FORMAT}")
        codec = _build_from(source["geometry"], source["state_dict"], device)
        codec.eval()
        return codec, source
    return load_checkpoint(source, device=device)


def load_allocation(source, groups=32, device=None):
    """Accept a ``.npy`` path, JSON list/dict, or tensor/array-like."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix == ".npy":
            values = np.load(path)
        elif path.suffix == ".json":
            payload = json.loads(path.read_text())
            if isinstance(payload, dict):
                values = payload.get("allocation", payload.get("map_allocation"))
                if values is None and "meta" in payload:
                    values = payload["meta"].get("allocation")
            else:
                values = payload
        else:
            raise ValueError(f"unsupported allocation file {path}")
    else:
        values = source
    allocation = torch.as_tensor(values, dtype=torch.long)
    if allocation.numel() != int(groups):
        raise ValueError(
            f"allocation length {allocation.numel()} != groups {groups}")
    if device is not None:
        allocation = allocation.to(device)
    return allocation.reshape(int(groups))


def allocation_rate(allocation, mode_bits):
    bits = tuple(map(int, mode_bits))
    modes = torch.as_tensor(allocation, dtype=torch.long).reshape(-1)
    return int(sum(bits[int(mode)] for mode in modes.tolist()))


def meta_allocation(payload):
    """Pull allocation list from a loaded checkpoint payload, or ``None``."""
    meta = payload.get("meta") or {}
    if meta.get("allocation") is not None:
        return meta["allocation"]
    return payload.get("allocation")
