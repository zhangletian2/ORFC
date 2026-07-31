"""Loading a frozen anchor object with the tensors actually checked.

Every consumer of a v9 anchor goes through :func:`load_checked`.  It re-reads
`U_0` and all codebooks from an independent reference file written at build
time and requires exact equality with the tensors inside the checkpoint, so a
silently re-built or half-overwritten `codec.pt` cannot reach a result.

This deliberately does not hash anything.  A digest tells you two files differ
but not which tensor moved or by how much; a direct ``allclose`` on the named
key tensors tells you both, and is what the standing convention in this project
asks for (checkpoint ID + metadata + key-tensor comparison).
"""

import json

import numpy as np
import torch

from codec_v1 import load_codec_v1


def reference_path(anchor):
    return anchor.root / "codec_ref.npz"


def save_reference(anchor, codec):
    """Write `U_0` and every codebook next to the checkpoint."""
    arrays = {"U0": codec.transform.get_rotation().detach().cpu().numpy()}
    for mode, quantizer in enumerate(codec.pq.quantizers):
        arrays[f"codebook_{mode}"] = quantizer.codebooks.detach().cpu().numpy()
    np.savez(reference_path(anchor), **arrays)


def load_checked(anchor, device):
    """Load the frozen codec, asserting it still matches its reference."""
    path = anchor.root / "codec.pt"
    meta = json.loads((anchor.root / "codec.json").read_text())
    codec = load_codec_v1(path, device=device).eval()

    if meta["anchor"] != anchor.name or meta["rate"] != anchor.rate:
        raise SystemExit(f"INVALID_EXPERIMENT: {path} carries metadata for "
                         f"{meta['anchor']}/R{meta['rate']}, expected "
                         f"{anchor.name}/R{anchor.rate}")
    if list(meta["mode_bits"]) != list(anchor.mode_bits):
        raise SystemExit(f"INVALID_EXPERIMENT: {path} menu {meta['mode_bits']} "
                         f"!= frozen menu {list(anchor.mode_bits)}")

    reference = np.load(reference_path(anchor))
    rotation = codec.transform.get_rotation().detach().cpu().numpy()
    if not np.array_equal(rotation, reference["U0"]):
        raise SystemExit(f"INVALID_EXPERIMENT: U_0 in {path} differs from its "
                         f"reference (max |d| = "
                         f"{np.abs(rotation - reference['U0']).max():.3e})")
    if len(codec.pq.quantizers) != len(anchor.mode_bits):
        raise SystemExit(f"INVALID_EXPERIMENT: {path} has "
                         f"{len(codec.pq.quantizers)} modes, expected "
                         f"{len(anchor.mode_bits)}")
    for mode, quantizer in enumerate(codec.pq.quantizers):
        book = quantizer.codebooks.detach().cpu().numpy()
        want = reference[f"codebook_{mode}"]
        if book.shape != (32, anchor.mode_sizes[mode], 32):
            raise SystemExit(f"INVALID_EXPERIMENT: codebook {mode} has shape "
                             f"{book.shape}, expected "
                             f"(32, {anchor.mode_sizes[mode]}, 32)")
        if not np.array_equal(book, want):
            raise SystemExit(f"INVALID_EXPERIMENT: codebook {mode} in {path} "
                             f"differs from its reference (max |d| = "
                             f"{np.abs(book - want).max():.3e})")

    # One rotation object, shared by every mode: requirement 1 of section 3.
    devices = {q.codebooks.device for q in codec.pq.quantizers}
    if devices != {codec.transform.rotation.device}:
        raise SystemExit("INVALID_EXPERIMENT: codebooks and U_0 are not on one "
                         f"device: {devices} vs {codec.transform.rotation.device}")
    return codec, meta


def orthogonality_error(codec):
    """||U^T U - I||_F in float64, on the fp32 matrix actually stored."""
    rotation = codec.transform.get_rotation().detach().double()
    identity = torch.eye(rotation.shape[0], device=rotation.device,
                         dtype=torch.float64)
    return float((rotation.t() @ rotation - identity).norm())
