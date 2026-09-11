#!/usr/bin/env python
"""Absorb the ORFC rotation R into the stage-1 conv weights (prefix keeps R).

Writes a NEW pair of checkpoints and never touches the originals:
    <orfc>.pt          -> <orfc>_absorbed.pt
    <orfc>_spatial.pt  -> <orfc>_absorbed_spatial.pt

Only meaningful when the spatial codec cls_mode is NOT "conv2".  There the
prefix tokens bypass analysis/synthesis, so R cannot be folded into a conv for
them and is kept as a dense matrix applied at the codec boundary; the patch
tokens do go through the convs, so R folds into those two weights and vanishes
from the runtime graph.  The PQ codebook stays shared by both groups.

Math.  analysis is Conv2d with weight [C, D, kh, kw] and synthesis is
ConvTranspose2d with weight [C, D, kh, kw]; both are indexed
[latent_channel, feature_channel, .., ..].  The codec computes Z = seq @ R.
    analysis  : want z_e = sum_c R[c,e] z_c   ->  W2[e,d] = sum_c R[c,e] W[c,d]
    synthesis : input is z_rot, z = z_rot R^T ->  W2[e,d] = sum_c R[c,e] W[c,d]
Both reduce to the same einsum, so one helper covers them.
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ORFC = HERE.parent / "orfc"
for _p in (str(ORFC), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from soft_pq import load_codec  # noqa: E402
from bilinear_residual import (  # noqa: E402
    BilinearSpatialCodec, load_residual_codec, load_spatial_weights,
)


class AbsorbedFeatureCodec(torch.nn.Module):
    """PQ on an absorbed model: patches arrive pre-rotated, prefix does not.

    The absorbed analysis already emits R-rotated patch latents, so only the
    prefix rows still need R (and R^T on the way out) to land in the same
    quantisation space.  Everything is sliced on the token axis so any batch
    size and any prefix length (dinov2 n_prefix=1, dinov3 n_prefix=5) works.
    """

    def __init__(self, codec, R_dense, n_prefix=1):
        super().__init__()
        self.codec = codec
        self.n_prefix = int(n_prefix)
        self.register_buffer("R_dense", R_dense)

    def forward(self, Y_norm):
        B, T, D = Y_norm.shape
        p = min(int(self.n_prefix), T)
        rotate = p > 0 and self.R_dense is not None
        if rotate:
            Z = torch.cat([Y_norm[:, :p] @ self.R_dense, Y_norm[:, p:]], dim=1)
        else:
            Z = Y_norm
        Z_hat, usage = self.codec.pq._quantise(Z.reshape(B * T, D))
        Z_hat = Z_hat.reshape(B, T, D)
        if rotate:
            Y_hat = torch.cat(
                [Z_hat[:, :p] @ self.R_dense.t(), Z_hat[:, p:]], dim=1)
        else:
            Y_hat = Z_hat
        return Y_hat, usage

    @property
    def use_rate(self):
        return self.codec.pq.use_rate

    @property
    def _last_rate(self):
        return self.codec.pq._last_rate

    @property
    def lmbda(self):
        return self.codec.pq.lmbda

    @torch.no_grad()
    def get_prior_pmf(self):
        return self.codec.pq.get_prior_pmf()


def fold_R(W, R):
    """W [C, D, kh, kw] with R [C, C] -> W2[e,d,i,j] = sum_c R[c,e] W[c,d,i,j]."""
    return torch.einsum("cdij,ce->edij", W.float(), R.float())


def build_spatial(meta, device, n_prefix):
    sp = BilinearSpatialCodec(
        int(meta["D"]), n_prefix=n_prefix, scale=2,
        down=meta.get("spatial_down", "conv2"),
        up=meta.get("spatial_up", "conv2"),
        cls_mode=meta.get("cls_mode", "learned")).to(device)
    load_spatial_weights(sp, meta)
    return sp.eval()


@torch.no_grad()
def verify(orfc_path, spat_path, abs_orfc_path, abs_spat_path,
           n_prefix, T, device):
    """Round-trip the original and absorbed cascades on the same random input."""
    codec = load_codec(str(orfc_path), device=device).eval()
    _, meta = load_residual_codec(str(spat_path), device=device)
    spatial = build_spatial(meta, device, n_prefix)

    abs_meta_o = torch.load(str(abs_orfc_path), map_location="cpu")
    abs_codec = load_codec(str(abs_orfc_path), device=device).eval()
    abs_codec = AbsorbedFeatureCodec(
        abs_codec, abs_meta_o["R_dense"].float().to(device),
        n_prefix=n_prefix).to(device).eval()
    _, abs_meta_s = load_residual_codec(str(abs_spat_path), device=device)
    abs_spatial = build_spatial(abs_meta_s, device, n_prefix)

    torch.manual_seed(0)
    Y = torch.randn(3, T, int(meta["D"]), device=device)

    seq, aux = spatial.encode(Y)
    seq_hat, _ = codec(seq)
    ref = spatial.decode(seq_hat, aux)
    lab = codec.pq._last_labels.clone()

    seq_a, aux_a = abs_spatial.encode(Y)
    seq_hat_a, _ = abs_codec(seq_a)
    got = abs_spatial.decode(seq_hat_a, aux_a)
    lab_a = abs_codec.codec.pq._last_labels.clone()

    # Folding R into float32 conv weights perturbs the pre-PQ latents by ~1e-5
    # relative, which is harmless except when a code sits exactly on a Voronoi
    # boundary: there the argmin flips and that one token moves by a full
    # codeword step.  So judge on relative RMS over the whole tensor and report
    # the flip rate separately rather than failing on a single max-abs outlier.
    flips = (lab != lab_a).float().mean().item()
    rel = (((ref - got) ** 2).mean().sqrt()
           / ((ref ** 2).mean().sqrt().clamp_min(1e-12))).item()
    print(f"  verify: coded_tokens={seq.shape[1]} relRMS={rel:.3e} "
          f"max|diff|={(ref - got).abs().max().item():.3e} "
          f"pq_label_flips={100 * flips:.4f}%")
    return rel, flips


def main():
    p = argparse.ArgumentParser("absorb ORFC R into stage-1 conv weights")
    p.add_argument("--orfc_ckpt", required=True,
                   help="stage-2 PQ codec .pt (the *_spatial.pt sibling is "
                        "used unless --spatial_ckpt is given)")
    p.add_argument("--spatial_ckpt", default="")
    p.add_argument("--n_prefix", type=int, default=5)
    p.add_argument("--T", type=int, default=201,
                   help="token count used only for the verification pass")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="max allowed relative round-trip mismatch")
    p.add_argument("--max_flip_rate", type=float, default=1e-3,
                   help="max fraction of PQ codes allowed to flip due to "
                        "float32 rounding at a Voronoi boundary")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    orfc_path = Path(args.orfc_ckpt)
    if not orfc_path.is_file():
        raise FileNotFoundError(orfc_path)
    spat_path = (Path(args.spatial_ckpt) if args.spatial_ckpt
                 else Path(str(orfc_path)[:-3] + "_spatial.pt"))
    if not spat_path.is_file():
        raise FileNotFoundError(spat_path)

    abs_orfc_path = Path(str(orfc_path)[:-3] + "_absorbed.pt")
    abs_spat_path = Path(str(orfc_path)[:-3] + "_absorbed_spatial.pt")
    if abs_orfc_path.is_file() and not args.force:
        raise SystemExit(f"exists (use --force): {abs_orfc_path}")

    print(f"[absorb] orfc    {orfc_path}")
    print(f"[absorb] spatial {spat_path}")

    codec = load_codec(str(orfc_path), device="cpu")
    if codec.transform is None or not hasattr(codec.transform, "get_rotation"):
        raise SystemExit("codec has no rotation to absorb")
    R = codec.transform.get_rotation().detach().float().cpu()

    orfc_meta = torch.load(str(orfc_path), map_location="cpu")
    spat_meta = torch.load(str(spat_path), map_location="cpu")
    cls_mode = spat_meta.get("cls_mode", "learned")
    if cls_mode == "conv2":
        raise SystemExit(
            "cls_mode=conv2 routes the prefix through the same conv, so R is "
            "absorbed for every token and no dense R should be kept; this "
            "script targets the prefix-bypass (cls_mode!=conv2) design.")
    print(f"[absorb] cls_mode={cls_mode} R={tuple(R.shape)} n_prefix={args.n_prefix}")

    sd = dict(spat_meta["spatial_state_dict"])
    for key in ("analysis.weight", "synthesis.weight"):
        if key not in sd:
            raise SystemExit(f"missing {key} in spatial_state_dict")
        sd[key] = fold_R(sd[key], R)
    spat_meta["spatial_state_dict"] = sd
    spat_meta["absorbed_R"] = True
    spat_meta["absorbed_from"] = str(spat_path)
    spat_meta["n_prefix"] = int(args.n_prefix)
    torch.save(spat_meta, str(abs_spat_path))

    orfc_meta["has_transform"] = False
    orfc_meta["transform_type"] = None
    orfc_meta["state_dict"] = {
        k: v for k, v in orfc_meta["state_dict"].items()
        if not k.startswith("transform.")
    }
    orfc_meta["R_dense"] = R
    orfc_meta["absorbed_R"] = True
    orfc_meta["absorbed_from"] = str(orfc_path)
    orfc_meta["n_prefix"] = int(args.n_prefix)
    torch.save(orfc_meta, str(abs_orfc_path))

    print(f"[absorb] wrote {abs_orfc_path.name}")
    print(f"[absorb] wrote {abs_spat_path.name}")

    rel, flips = verify(orfc_path, spat_path, abs_orfc_path, abs_spat_path,
                        args.n_prefix, args.T, device)
    # Two separate failure modes: a wrong fold shows up as a large relRMS with
    # zero label flips, whereas boundary flips are expected at a low rate and
    # inflate relRMS by a full codeword step each (very visible at small K).
    if flips > args.max_flip_rate:
        raise SystemExit(
            f"FAIL: pq label flip rate {100 * flips:.4f}% > "
            f"{100 * args.max_flip_rate:.4f}%")
    if flips == 0.0 and rel > args.tol:
        raise SystemExit(f"FAIL: relative mismatch {rel:.3e} > tol {args.tol:g}")
    print("[absorb] OK")


if __name__ == "__main__":
    main()
